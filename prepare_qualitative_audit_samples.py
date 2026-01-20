#!/usr/bin/env python3
"""
Stage -1f: Qualitative Caption Audit (15 samples)

What it does:
1) Loads dataset_normalized_en.csv
2) Samples 15 rows with a fixed seed
3) Prints a compact view to the terminal
4) Saves an annotation template CSV you can manually fill:
   - identity_leak
   - org_type (grounded / overlay / mixed / none)
   - overlay_text
   - groundedness
   - tone
   - notes
"""

import os
import sys
import pandas as pd

# ====== CONFIG ======
INPUT_CSV = "dataset_normalized_en.csv"
OUTPUT_DIR = "stage_-1f_outputs"
SEED = 42
N_SAMPLES = 15

# Columns we try to include if present (we will gracefully fallback if missing)
CANDIDATE_COLS = [
    "ID",
    "Img URL",
    "Img Name",
    "caption_original",
    "is_arabic",
    "caption_en",
]

# ====== HELPERS ======
def pick_existing_cols(df, cols):
    return [c for c in cols if c in df.columns]

def die(msg: str, code: int = 1):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)

# ====== MAIN ======
def main():
    if not os.path.exists(INPUT_CSV):
        die(f"Input file not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    if "caption_en" not in df.columns:
        die("Expected column 'caption_en' not found in dataset_normalized_en.csv")

    # Drop empty captions (should be none per your translation stage, but keep robust)
    df = df.copy()
    df["caption_en"] = df["caption_en"].astype(str)
    df = df[df["caption_en"].str.strip().astype(bool)]

    if len(df) < N_SAMPLES:
        die(f"Dataset has only {len(df)} usable rows, cannot sample {N_SAMPLES}.")

    sample_df = df.sample(N_SAMPLES, random_state=SEED).reset_index(drop=True)

    # Choose display columns
    cols = pick_existing_cols(sample_df, CANDIDATE_COLS)
    view_df = sample_df[cols].copy()

    # Print a readable terminal view
    print("\n=== Stage -1f Sample (15 rows) ===\n")
    for i, row in view_df.iterrows():
        print(f"[{i+1:02d}]")
        for c in cols:
            val = row[c]
            if isinstance(val, str) and len(val) > 260:
                val = val[:260] + " ...[truncated]"
            print(f"  {c}: {val}")
        print()

    # Create annotation template
    ann = view_df.copy()

    # Add audit label columns (fill manually)
    ann["identity_leak"] = ""   # yes / no
    ann["identity_type"] = ""   # person / org / both / unsure
    ann["org_type"] = ""        # grounded / overlay / mixed / none / unsure
    ann["overlay_text"] = ""    # yes / no
    ann["groundedness"] = ""    # grounded / partly / narrative / unsure
    ann["tone"] = ""            # neutral / empathetic / biased / celebratory / dehumanizing / unsure
    ann["policy_compliance"] = ""  # compliant / noncompliant / unsure
    ann["notes"] = ""           # free text

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"stage_-1f_sample_{N_SAMPLES}_seed_{SEED}.csv")
    ann.to_csv(out_path, index=False)

    print("=== Saved annotation template ===")
    print(out_path)
    print("\nOpen it, fill the new columns, then paste the filled rows (or upload the CSV) here.\n")

if __name__ == "__main__":
    main()
