#!/usr/bin/env python3
import os
import pandas as pd

PROJECT_ROOT = os.path.expanduser("~/projects/Image_captioning_Israeli_gaza_conflict/")

VAL_CSV  = os.path.join(PROJECT_ROOT, "splits/val.csv")
BASE_CSV = os.path.join(PROJECT_ROOT, "outputs/baseline_val_predictions.csv")
FT_CSV   = os.path.join(PROJECT_ROOT, "outputs/finetuned_val_predictions.csv")

OUT_CSV  = os.path.join(PROJECT_ROOT, "outputs/qualitative_15_samples.csv")

SEED = 42
N_SAMPLES = 15

KEY_COL = "Img Name"
GT_COL  = "caption_train"

BASE_PRED_COL = "caption_pred_baseline"   # confirmed by your header
FT_PRED_COL   = "caption_pred_finetuned"  # confirmed

def main():
    val  = pd.read_csv(VAL_CSV)
    base = pd.read_csv(BASE_CSV)
    ft   = pd.read_csv(FT_CSV)

    # Sanity checks
    for col in [KEY_COL, GT_COL]:
        if col not in val.columns:
            raise ValueError(f"Missing '{col}' in val.csv. Found: {list(val.columns)}")

    for col in [KEY_COL, BASE_PRED_COL]:
        if col not in base.columns:
            raise ValueError(f"Missing '{col}' in baseline CSV. Found: {list(base.columns)}")

    for col in [KEY_COL, FT_PRED_COL]:
        if col not in ft.columns:
            raise ValueError(f"Missing '{col}' in finetuned CSV. Found: {list(ft.columns)}")

    merged = (
        val[[KEY_COL, GT_COL]]
        .merge(base[[KEY_COL, BASE_PRED_COL]], on=KEY_COL, how="inner")
        .merge(ft[[KEY_COL, FT_PRED_COL]], on=KEY_COL, how="inner")
        .rename(columns={
            GT_COL: "caption_train",
            BASE_PRED_COL: "baseline_pred",
            FT_PRED_COL: "finetuned_pred",
        })
    )

    if len(merged) == 0:
        raise RuntimeError("Merge produced 0 rows. Check Img Name consistency across files.")

    # Deterministic sample
    out = merged.sample(n=min(N_SAMPLES, len(merged)), random_state=SEED)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    print(f"Saved: {OUT_CSV}")
    print(f"Rows: {len(out)} / {len(merged)} aligned")

if __name__ == "__main__":
    main()
