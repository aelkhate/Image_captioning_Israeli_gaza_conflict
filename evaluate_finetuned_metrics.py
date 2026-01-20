#!/usr/bin/env python3
"""
evaluate_finetuned_metrics.py

Compute automatic captioning metrics for the fine-tuned model on VAL:
- BLEU (corpus BLEU via sacrebleu if available; otherwise NLTK fallback)
- ROUGE-L (rouge_score)
- BERTScore F1 (bert_score)

Inputs (defaults):
- splits/val.csv (GT column: caption_train)
- outputs/finetuned_val_predictions.csv (prediction column auto-detected)

Outputs:
- outputs/finetuned_val_aligned_for_eval.csv
- outputs/finetuned_val_metrics.json

Notes:
- Aligns rows by ID if present in both; else by "Img Name"/"img_name"/"image"/"filename";
  else falls back to row-order with a warning.
- Designed to be robust to minor column name differences.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import pandas as pd


# -----------------------------
# Hardcoded defaults (you can change these)
# -----------------------------
PROJECT_ROOT = os.path.expanduser("~/projects/Image_captioning_Israeli_gaza_conflict/")
VAL_CSV = os.path.join(PROJECT_ROOT, "splits/val.csv")
PRED_CSV = os.path.join(PROJECT_ROOT, "outputs/finetuned_val_predictions.csv")

OUT_ALIGNED = os.path.join(PROJECT_ROOT, "outputs/finetuned_val_aligned_for_eval.csv")
OUT_METRICS = os.path.join(PROJECT_ROOT, "outputs/finetuned_val_metrics.json")

GT_COL = "caption_train"  # agreed GT column
# prediction column will be auto-detected


# -----------------------------
# Helpers
# -----------------------------
def _normalize_colname(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


def _pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    norm_map = {_normalize_colname(c): c for c in df.columns}
    for cand in candidates:
        nc = _normalize_colname(cand)
        if nc in norm_map:
            return norm_map[nc]
    return None


def _clean_text(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip()
    # Normalize whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


@dataclass
class AlignmentInfo:
    method: str
    gt_key: Optional[str]
    pred_key: Optional[str]
    n_gt: int
    n_pred: int
    n_aligned: int
    dropped_gt: int
    dropped_pred: int


def align_gt_pred(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    gt_caption_col: str,
    pred_caption_col: str,
) -> Tuple[pd.DataFrame, AlignmentInfo]:
    """
    Align GT and predictions by a shared key (prefer ID, then image name), else fallback to row-order.
    Returns aligned df with columns: ID, img_name, caption_train, finetuned_pred
    """
    # Candidate key columns
    gt_id_col = _pick_first_existing(gt_df, ["ID", "id"])
    pred_id_col = _pick_first_existing(pred_df, ["ID", "id"])

    gt_img_col = _pick_first_existing(gt_df, ["Img Name", "img_name", "image", "image_name", "filename", "file", "img"])
    pred_img_col = _pick_first_existing(pred_df, ["Img Name", "img_name", "image", "image_name", "filename", "file", "img"])

    # Clean caption columns
    gt_df = gt_df.copy()
    pred_df = pred_df.copy()

    gt_df[gt_caption_col] = gt_df[gt_caption_col].map(_clean_text)
    pred_df[pred_caption_col] = pred_df[pred_caption_col].map(_clean_text)

    # Prefer ID alignment if both have it and it looks usable
    if gt_id_col and pred_id_col:
        method = "id"
        gt_df["_key"] = gt_df[gt_id_col].astype(str)
        pred_df["_key"] = pred_df[pred_id_col].astype(str)

    # Else try image filename alignment
    elif gt_img_col and pred_img_col:
        method = "img_name"
        gt_df["_key"] = gt_df[gt_img_col].astype(str)
        pred_df["_key"] = pred_df[pred_img_col].astype(str)

    # Else fallback: row-order
    else:
        method = "row_order"
        # Make keys by index
        gt_df["_key"] = gt_df.index.astype(str)
        pred_df["_key"] = pred_df.index.astype(str)

    # Deduplicate predictions on key (keep first)
    pred_df = pred_df.drop_duplicates(subset=["_key"], keep="first")
    gt_df = gt_df.drop_duplicates(subset=["_key"], keep="first")

    merged = gt_df.merge(
        pred_df[["_key", pred_caption_col] + ([pred_id_col] if pred_id_col else []) + ([pred_img_col] if pred_img_col else [])],
        on="_key",
        how="inner",
        suffixes=("_gt", "_pred"),
    )

    # Determine output ID/img_name columns
    out_id = None
    out_img = None
    if method == "id":
        out_id = gt_id_col  # prefer GT's ID
        out_img = gt_img_col
    elif method == "img_name":
        out_img = gt_img_col
        out_id = gt_id_col
    else:
        out_id = gt_id_col
        out_img = gt_img_col

    aligned = pd.DataFrame()
    aligned["ID"] = merged[out_id] if out_id and out_id in merged.columns else ""
    aligned["img_name"] = merged[out_img] if out_img and out_img in merged.columns else ""
    aligned[GT_COL] = merged[gt_caption_col]
    aligned["finetuned_pred"] = merged[pred_caption_col]

    info = AlignmentInfo(
        method=method,
        gt_key=(gt_id_col if method == "id" else gt_img_col if method == "img_name" else None),
        pred_key=(pred_id_col if method == "id" else pred_img_col if method == "img_name" else None),
        n_gt=len(gt_df),
        n_pred=len(pred_df),
        n_aligned=len(aligned),
        dropped_gt=len(gt_df) - len(merged),
        dropped_pred=len(pred_df) - len(merged),
    )
    return aligned, info


# -----------------------------
# Metrics
# -----------------------------
def compute_bleu(references: List[str], candidates: List[str]) -> float:
    """
    Returns BLEU score as a percentage-like number (0-100) if sacrebleu is available,
    else returns 0-100-ish via NLTK corpus_bleu * 100.
    """
    # Try sacrebleu first (better standardization)
    try:
        import sacrebleu  # type: ignore

        bleu = sacrebleu.corpus_bleu(candidates, [references])
        return float(bleu.score)
    except Exception:
        pass

    # Fallback: NLTK corpus_bleu
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction  # type: ignore

        smoothie = SmoothingFunction().method4
        refs_tok = [[r.split()] for r in references]  # list of list of refs per sample
        cands_tok = [c.split() for c in candidates]
        score = corpus_bleu(refs_tok, cands_tok, smoothing_function=smoothie)
        return float(score * 100.0)
    except Exception as e:
        print(f"[WARN] BLEU unavailable (install sacrebleu or nltk). Error: {e}")
        return float("nan")


def compute_rouge_l(references: List[str], candidates: List[str]) -> float:
    """
    Returns ROUGE-L F1 (0-1).
    """
    try:
        from rouge_score import rouge_scorer  # type: ignore

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = []
        for ref, cand in zip(references, candidates):
            s = scorer.score(ref, cand)["rougeL"].fmeasure
            scores.append(s)
        return float(sum(scores) / max(1, len(scores)))
    except Exception as e:
        print(f"[WARN] ROUGE-L unavailable (pip install rouge-score). Error: {e}")
        return float("nan")


def compute_bertscore_f1(references: List[str], candidates: List[str]) -> float:
    """
    Returns mean BERTScore F1 (0-1).
    """
    try:
        from bert_score import score as bert_score  # type: ignore

        # rescale_with_baseline=False keeps it raw; change if your baseline script used rescaling.
        P, R, F1 = bert_score(
            cands=candidates,
            refs=references,
            lang="en",
            verbose=False,
            rescale_with_baseline=False,
        )
        return float(F1.mean().item())
    except Exception as e:
        print(f"[WARN] BERTScore unavailable (pip install bert-score). Error: {e}")
        return float("nan")


def basic_stats(texts: List[str]) -> Dict[str, float]:
    lengths = [len(t.split()) for t in texts]
    if not lengths:
        return {"avg_words": 0.0, "min_words": 0.0, "max_words": 0.0}
    return {
        "avg_words": float(sum(lengths) / len(lengths)),
        "min_words": float(min(lengths)),
        "max_words": float(max(lengths)),
    }


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    print("=== Fine-tuned Evaluation (VAL) ===")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"VAL CSV: {VAL_CSV}")
    print(f"PRED CSV: {PRED_CSV}")

    if not os.path.isfile(VAL_CSV):
        raise FileNotFoundError(f"VAL file not found: {VAL_CSV}")
    if not os.path.isfile(PRED_CSV):
        raise FileNotFoundError(f"Predictions file not found: {PRED_CSV}")

    gt_df = pd.read_csv(VAL_CSV)
    pred_df = pd.read_csv(PRED_CSV)

    if GT_COL not in gt_df.columns:
        raise ValueError(f"GT column '{GT_COL}' not found in {VAL_CSV}. Columns: {list(gt_df.columns)}")

    # Auto-detect prediction column
    pred_col = "caption_pred_finetuned"


    # If not found, try: any column containing "pred"
    if pred_col is None:
        for c in pred_df.columns:
            if "pred" in _normalize_colname(c):
                pred_col = c
                break

    if pred_col is None:
        raise ValueError(
            "Could not auto-detect prediction column in predictions CSV. "
            f"Columns: {list(pred_df.columns)}\n"
            "Please rename your prediction column to one of: pred, caption_pred, predicted_caption, caption."
        )

    print(f"Using GT column: {GT_COL}")
    print(f"Using PRED column: {pred_col}")

    aligned_df, align_info = align_gt_pred(gt_df, pred_df, GT_COL, pred_col)

    if align_info.method == "row_order":
        print("[WARN] Could not find common ID/img_name columns. Falling back to row-order alignment.")
    else:
        print(f"Alignment method: {align_info.method} (gt_key={align_info.gt_key}, pred_key={align_info.pred_key})")

    print(f"GT rows: {align_info.n_gt} | Pred rows: {align_info.n_pred} | Aligned: {align_info.n_aligned}")

    if align_info.n_aligned == 0:
        raise RuntimeError("Alignment produced 0 pairs. Check that IDs or image names match between CSVs.")

    # Prepare lists
    refs = aligned_df[GT_COL].map(_clean_text).tolist()
    cands = aligned_df["finetuned_pred"].map(_clean_text).tolist()

    # Compute metrics
    bleu = compute_bleu(refs, cands)
    rouge_l = compute_rouge_l(refs, cands)
    bert_f1 = compute_bertscore_f1(refs, cands)

    # Save aligned pairs
    _ensure_dir(OUT_ALIGNED)
    aligned_df.to_csv(OUT_ALIGNED, index=False)
    print(f"Saved aligned eval file: {OUT_ALIGNED}")

    # Save metrics
    metrics = {
        "split": "val",
        "gt_column": GT_COL,
        "pred_column": pred_col,
        "alignment": {
            "method": align_info.method,
            "gt_key": align_info.gt_key,
            "pred_key": align_info.pred_key,
            "n_gt": align_info.n_gt,
            "n_pred": align_info.n_pred,
            "n_aligned": align_info.n_aligned,
            "dropped_gt": align_info.dropped_gt,
            "dropped_pred": align_info.dropped_pred,
        },
        "metrics": {
            "BLEU": bleu,
            "ROUGE_L": rouge_l,
            "BERTScore_F1": bert_f1,
        },
        "stats": {
            "gt": basic_stats(refs),
            "pred": basic_stats(cands),
        },
    }

    _ensure_dir(OUT_METRICS)
    with open(OUT_METRICS, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Saved metrics JSON: {OUT_METRICS}")
    print("---")
    print(f"BLEU:      {bleu:.4f}")
    print(f"ROUGE-L:   {rouge_l:.4f}")
    print(f"BERTScore: {bert_f1:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
