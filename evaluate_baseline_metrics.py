#!/usr/bin/env python3
"""
evaluate_baseline_metrics.py

Evaluates baseline caption predictions against ground truth captions using:
- BLEU (sacrebleu)
- ROUGE-L (rouge_score)
- BERTScore (bert-score)

This project chooses caption_train as the evaluation target (policy-aligned ground truth).
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd


# -----------------------------
# CONFIG
# -----------------------------
@dataclass
class Config:
    # Ground truth and predictions
    val_csv: str = "splits/val.csv"
    pred_csv: str = "outputs/baseline_val_predictions.csv"  # can be cleaned or raw
    pred_col: str = "caption_pred_baseline"

    # Ground truth column choice
    gt_col: str = "caption_train"

    # Outputs
    out_dir: str = "outputs"
    out_json: str = "outputs/baseline_val_metrics.json"
    out_csv_aligned: str = "outputs/baseline_val_aligned_for_eval.csv"

    # Text normalization (lightweight; do NOT over-clean)
    lowercase: bool = False
    normalize_whitespace: bool = True
    strip_chat_transcript: bool = True

    # Safety sanity checks
    drop_empty_pairs: bool = True


CFG = Config()


# -----------------------------
# UTILS
# -----------------------------
def _normalize_text(s: str, lowercase: bool, normalize_whitespace: bool) -> str:
    if s is None:
        return ""
    s = str(s)
    if normalize_whitespace:
        s = re.sub(r"\s+", " ", s).strip()
    if lowercase:
        s = s.lower()
    return s


def _strip_chat_if_needed(s: str) -> str:
    """
    Some Qwen runs may return a transcript containing 'system/user/assistant'.
    Keep only the final assistant segment if present.
    """
    if not isinstance(s, str):
        return ""
    marker = "assistant\n"
    if marker in s:
        return s.split(marker)[-1].strip()
    return s.strip()


def _require(pkg_name: str, import_name: str = None):
    """
    Helpful error message if user didn't install packages.
    """
    try:
        __import__(import_name or pkg_name)
    except Exception as e:
        raise RuntimeError(
            f"Missing dependency '{pkg_name}'. Install it with:\n"
            f"  pip install {pkg_name}\n\nOriginal error: {e}"
        )


def _load_and_align(cfg: Config) -> Tuple[List[str], List[str], pd.DataFrame]:
    if not os.path.exists(cfg.val_csv):
        raise FileNotFoundError(f"Validation CSV not found: {cfg.val_csv}")
    if not os.path.exists(cfg.pred_csv):
        raise FileNotFoundError(f"Predictions CSV not found: {cfg.pred_csv}")

    val = pd.read_csv(cfg.val_csv)
    pred = pd.read_csv(cfg.pred_csv)

    if "Img Name" not in val.columns:
        raise ValueError(f"'Img Name' column missing from {cfg.val_csv}")
    if "Img Name" not in pred.columns:
        raise ValueError(f"'Img Name' column missing from {cfg.pred_csv}")

    if cfg.gt_col not in val.columns:
        raise ValueError(f"Ground truth column '{cfg.gt_col}' missing from {cfg.val_csv}")
    if cfg.pred_col not in pred.columns:
        raise ValueError(f"Prediction column '{cfg.pred_col}' missing from {cfg.pred_csv}")

    # Align by Img Name
    merged = val[["Img Name", cfg.gt_col]].merge(
        pred[["Img Name", cfg.pred_col]],
        on="Img Name",
        how="inner",
        suffixes=("_gt", "_pred"),
    )

    # Extract and normalize
    gts: List[str] = []
    preds: List[str] = []

    for _, row in merged.iterrows():
        gt = row[cfg.gt_col]
        pr = row[cfg.pred_col]

        if cfg.strip_chat_transcript:
            pr = _strip_chat_if_needed(pr)

        gt = _normalize_text(gt, cfg.lowercase, cfg.normalize_whitespace)
        pr = _normalize_text(pr, cfg.lowercase, cfg.normalize_whitespace)

        if cfg.drop_empty_pairs and (gt == "" or pr == ""):
            continue

        gts.append(gt)
        preds.append(pr)

    if len(gts) == 0:
        raise RuntimeError("No aligned (gt, pred) pairs found after filtering.")

    # Save aligned CSV for debugging / reproducibility
    merged_out = pd.DataFrame({"gt": gts, "pred": preds})
    os.makedirs(cfg.out_dir, exist_ok=True)
    merged_out.to_csv(cfg.out_csv_aligned, index=False)

    return gts, preds, merged_out


# -----------------------------
# METRICS
# -----------------------------
def compute_bleu(gts: List[str], preds: List[str]) -> Dict:
    _require("sacrebleu")
    import sacrebleu

    # sacrebleu expects list of hypotheses and list of references lists
    bleu = sacrebleu.corpus_bleu(preds, [gts])
    return {
        "bleu": {
            "score": float(bleu.score),
            "bp": float(bleu.bp),
            "counts": list(map(int, bleu.counts)),
            "totals": list(map(int, bleu.totals)),
            "precisions": [float(x) for x in bleu.precisions],
            "sys_len": int(bleu.sys_len),
            "ref_len": int(bleu.ref_len),
        }
    }


def compute_rouge_l(gts: List[str], preds: List[str]) -> Dict:
    _require("rouge_score", "rouge_score")
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    f1s = []
    ps = []
    rs = []
    for gt, pr in zip(gts, preds):
        scores = scorer.score(gt, pr)["rougeL"]
        f1s.append(scores.fmeasure)
        ps.append(scores.precision)
        rs.append(scores.recall)

    def avg(xs): return float(sum(xs) / max(1, len(xs)))

    return {
        "rougeL": {
            "f1": avg(f1s),
            "precision": avg(ps),
            "recall": avg(rs),
        }
    }


def compute_bertscore(gts: List[str], preds: List[str]) -> Dict:
    _require("bert_score", "bert_score")
    from bert_score import score as bertscore

    # Default is roberta-large; good but heavier.
    # If runtime is slow, you can set model_type="distilbert-base-uncased"
    P, R, F1 = bertscore(
        preds,
        gts,
        lang="en",
        rescale_with_baseline=True,
        verbose=False,
    )

    return {
        "bertscore": {
            "precision": float(P.mean().item()),
            "recall": float(R.mean().item()),
            "f1": float(F1.mean().item()),
            "rescale_with_baseline": True,
        }
    }


# -----------------------------
# MAIN
# -----------------------------
def main():
    print("=== Baseline Automatic Metrics Evaluation ===")
    print(f"GT column: {CFG.gt_col}")
    print(f"VAL: {CFG.val_csv}")
    print(f"PRED: {CFG.pred_csv}")
    print(f"Aligned pairs CSV: {CFG.out_csv_aligned}")

    gts, preds, _ = _load_and_align(CFG)
    print(f"Aligned pairs: {len(gts)}")

    results = {
        "meta": {
            "gt_column": CFG.gt_col,
            "val_csv": CFG.val_csv,
            "pred_csv": CFG.pred_csv,
            "pred_column": CFG.pred_col,
            "num_pairs": len(gts),
            "lowercase": CFG.lowercase,
            "normalize_whitespace": CFG.normalize_whitespace,
            "strip_chat_transcript": CFG.strip_chat_transcript,
        }
    }

    # Metrics
    results.update(compute_bleu(gts, preds))
    results.update(compute_rouge_l(gts, preds))
    results.update(compute_bertscore(gts, preds))

    os.makedirs(CFG.out_dir, exist_ok=True)
    with open(CFG.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved metrics JSON: {CFG.out_json}")
    print("\nSummary:")
    print(f"  BLEU:     {results['bleu']['score']:.2f}")
    print(f"  ROUGE-L:  {results['rougeL']['f1']:.4f}")
    print(f"  BERTScore:{results['bertscore']['f1']:.4f}")


if __name__ == "__main__":
    main()
