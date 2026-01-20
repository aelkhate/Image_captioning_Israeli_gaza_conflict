
#!/usr/bin/env python3
"""
translate_arabic_captions.py
Translate Arabic captions to English (GPU-accelerated, batched, cached).

Input:
  - dataset.csv (expects a caption column, default: "Desc (S)")

Output:
  - dataset_normalized_en.csv
    Adds:
      * caption_original
      * is_arabic
      * caption_en
  - translation_cache_ar_en.json (SHA1 -> translation)

Run:
  python -u translate_arabic_captions.py
"""

import re
import json
import hashlib
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import torch
from transformers import pipeline

# -----------------
# Config
# -----------------
PATH_IN = "dataset.csv"
PATH_OUT = "dataset_normalized_en.csv"
CAP_COL = "Desc (S)"

CACHE_PATH = "translation_cache_ar_en.json"
DEVICE = 0          # 0 = first GPU, -1 = CPU
BATCH_SIZE = 32     # if OOM, reduce to 16 or 8
MAX_LEN = 256       # translation max length

# Arabic unicode block
AR_RE = re.compile(r"[\u0600-\u06FF]")

def is_arabic(text: str) -> bool:
    return bool(AR_RE.search(str(text)))

def sha1_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def load_cache(path: str) -> dict:
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}

def save_cache(path: str, cache: dict) -> None:
    Path(path).write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

def main():
    print("Starting translation script...")
    print("Input:", PATH_IN)

    df = pd.read_csv(PATH_IN)
    if CAP_COL not in df.columns:
        raise ValueError(f"Caption column '{CAP_COL}' not found. Available: {df.columns.tolist()}")

    df[CAP_COL] = df[CAP_COL].astype(str).fillna("").str.strip()
    df = df[df[CAP_COL].str.len() > 0].copy()

    df["caption_original"] = df[CAP_COL]
    df["is_arabic"] = df["caption_original"].apply(is_arabic)

    total = len(df)
    arabic_count = int(df["is_arabic"].sum())
    print(f"Total rows: {total}")
    print(f"Arabic rows: {arabic_count} ({arabic_count/total*100:.1f}%)")

    # speed knobs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    use_gpu = (DEVICE >= 0) and torch.cuda.is_available()
    dtype = torch.float16 if use_gpu else torch.float32
    print(f"Using GPU: {use_gpu} | device={DEVICE} | dtype={dtype}")

    translator = pipeline(
        task="translation",
        model="Helsinki-NLP/opus-mt-ar-en",
        device=DEVICE,
        torch_dtype=dtype,
    )

    cache = load_cache(CACHE_PATH)
    print(f"Loaded cache entries: {len(cache)}")

    # Gather arabic captions and check cache
    arabic_texts = df.loc[df["is_arabic"], "caption_original"].tolist()
    to_translate = []
    translations_by_key = {}

    for t in arabic_texts:
        k = sha1_key(t)
        if k in cache:
            translations_by_key[k] = cache[k]
        else:
            to_translate.append(t)

    print(f"Need to translate now: {len(to_translate)} (cached: {len(arabic_texts)-len(to_translate)})")

    # Translate (chunked loop for stability)
    if to_translate:
        outs = []
        for i in tqdm(range(0, len(to_translate), BATCH_SIZE), desc="Translating"):
            batch = to_translate[i:i+BATCH_SIZE]
            batch_out = translator(batch, batch_size=BATCH_SIZE, max_length=MAX_LEN)
            outs.extend(batch_out)

        for src, out in zip(to_translate, outs):
            k = sha1_key(src)
            tgt = out["translation_text"].strip()
            translations_by_key[k] = tgt
            cache[k] = tgt

        save_cache(CACHE_PATH, cache)
        print(f"Saved cache: {CACHE_PATH}")

    # Build caption_en
    df["caption_en"] = df["caption_original"]
    if arabic_count > 0:
        ar_keys = df.loc[df["is_arabic"], "caption_original"].apply(sha1_key)
        df.loc[df["is_arabic"], "caption_en"] = ar_keys.map(translations_by_key).fillna("")

    empty_trans = int(((df["is_arabic"]) & (df["caption_en"].str.len() == 0)).sum())
    print(f"Empty translations among Arabic rows: {empty_trans}")

    # Save output CSV
    out_path = Path(PATH_OUT).resolve()
    print("Writing CSV to:", out_path)
    df.to_csv(out_path, index=False)
    print("Saved normalized dataset:", out_path)

    if arabic_count > 0:
        print("\nSample Arabic -> English:")
        sample = df[df["is_arabic"]].sample(min(5, arabic_count), random_state=42)[["caption_original", "caption_en"]]
        for _, r in sample.iterrows():
            print("-" * 60)
            print("AR:", r["caption_original"])
            print("EN:", r["caption_en"])

    print("\nDone.")

if __name__ == "__main__":
    main()

