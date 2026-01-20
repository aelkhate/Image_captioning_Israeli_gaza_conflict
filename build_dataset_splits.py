#!/usr/bin/env python3
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------
# Config
# -----------------------------
INPUT_CSV = "dataset_train_ready.csv"   # your current main dataset
IMG_DIR = "images_resolved"             # resolved folder: <id>.jpg symlinks
IMG_COL = "Img Name"

OUT_PAIRED = "dataset_paired_only.csv"
OUT_DROPPED = "dataset_dropped_missing_images.csv"

SPLITS_DIR = "splits"
TRAIN_OUT = os.path.join(SPLITS_DIR, "train.csv")
VAL_OUT   = os.path.join(SPLITS_DIR, "val.csv")
TEST_OUT  = os.path.join(SPLITS_DIR, "test.csv")

SEED = 42

# ratios
TEST_SIZE = 0.10        # 10% test
VAL_SIZE_FROM_REST = 0.1111111111  # so that val is 10% of total (0.9 * 0.111.. = 0.10)

# Optional stratification column (only used if present)
STRATIFY_COL_CANDIDATES = ["is_arabic"]

# -----------------------------
# Helpers
# -----------------------------
def pick_stratify_col(df: pd.DataFrame):
    for c in STRATIFY_COL_CANDIDATES:
        if c in df.columns:
            # must have at least 2 classes and enough samples
            vc = df[c].value_counts(dropna=False)
            if len(vc) >= 2 and vc.min() >= 5:
                return c
    return None

def main():
    # Load
    df = pd.read_csv(INPUT_CSV)

    if IMG_COL not in df.columns:
        raise ValueError(f"Missing required column '{IMG_COL}' in {INPUT_CSV}. Found: {list(df.columns)}")

    # Normalize image names
    df[IMG_COL] = df[IMG_COL].astype(str).str.strip()

    # Verify image existence
    img_dir = Path(IMG_DIR)
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {IMG_DIR}")

    exists_mask = df[IMG_COL].apply(lambda n: (img_dir / n).exists())
    df_paired = df[exists_mask].copy()
    df_dropped = df[~exists_mask].copy()

    # Save audit outputs
    df_paired.to_csv(OUT_PAIRED, index=False)
    df_dropped.to_csv(OUT_DROPPED, index=False)

    print("=== Pairing Report ===")
    print("Input rows:", len(df))
    print("Paired rows:", len(df_paired))
    print("Dropped (missing images):", len(df_dropped))
    if len(df_dropped) > 0:
        print("First 20 dropped Img Names:", df_dropped[IMG_COL].head(20).tolist())
    print("Saved:", OUT_PAIRED)
    print("Saved:", OUT_DROPPED)

    # Choose stratify column if possible
    strat_col = pick_stratify_col(df_paired)
    strat_vals = df_paired[strat_col] if strat_col else None
    if strat_col:
        print(f"Stratifying splits by: {strat_col}")
    else:
        print("No stratification (column not found or not suitable).")

    # Split: test first
    train_val, test = train_test_split(
        df_paired,
        test_size=TEST_SIZE,
        random_state=SEED,
        shuffle=True,
        stratify=strat_vals
    )

    # Split train vs val from remaining
    strat_vals_2 = train_val[strat_col] if strat_col else None
    train, val = train_test_split(
        train_val,
        test_size=VAL_SIZE_FROM_REST,
        random_state=SEED,
        shuffle=True,
        stratify=strat_vals_2
    )

    # Save splits
    os.makedirs(SPLITS_DIR, exist_ok=True)
    train.to_csv(TRAIN_OUT, index=False)
    val.to_csv(VAL_OUT, index=False)
    test.to_csv(TEST_OUT, index=False)

    print("\n=== Split Sizes ===")
    print("Train:", len(train))
    print("Val  :", len(val))
    print("Test :", len(test))
    print("Saved:", TRAIN_OUT)
    print("Saved:", VAL_OUT)
    print("Saved:", TEST_OUT)

    # Final integrity checks
    def uniq_names(d): return set(d[IMG_COL].tolist())
    inter = uniq_names(train) & uniq_names(val) | uniq_names(train) & uniq_names(test) | uniq_names(val) & uniq_names(test)
    if inter:
        raise RuntimeError(f"Split leakage detected: {len(inter)} overlapping image names!")

    print("\n[OK] No overlap between splits (by Img Name).")
    print("[NEXT] You can now run baseline inference on val.csv, then fine-tuning.")

if __name__ == "__main__":
    main()
