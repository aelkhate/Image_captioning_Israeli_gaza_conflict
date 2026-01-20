#!/usr/bin/env python3
"""
EDA.py — Stage -1 EDA for Gaza–Israel image-caption dataset (NO topic modeling)

Outputs:
1) caption_length_hist.png
2) eda_top_unigrams.png
3) eda_top_bigrams.png
4) eda_top_unigrams_deboilerplate.png
5) eda_top_bigrams_deboilerplate.png

Usage:
  python EDA.py
"""

print("EDA.py started")

import re
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# -------------------------
# Config
# -------------------------
PATH = "dataset.csv"          # <-- change to your file path if needed
CAP_COL = "Desc (S)"          # caption column name
TOPN = 20                     # number of unigrams/bigrams to plot


# -------------------------
# Helpers
# -------------------------
stop = set(ENGLISH_STOP_WORDS)

def clean_for_freq(t: str) -> str:
    """Lowercase + remove URLs + keep letters/spaces only. For frequency analysis."""
    t = str(t).lower()
    t = re.sub(r"http\S+", " ", t)        # remove URLs
    t = re.sub(r"[^a-z\s]", " ", t)       # keep only letters/spaces
    t = re.sub(r"\s+", " ", t).strip()    # normalize whitespace
    return t

# Words/phrases that are mostly annotation/template artifacts in your dataset.
# (We remove them only for the "deboilerplate" view, not the raw view.)
BOILER_PATTERNS = [
    r"\bbbc\b", r"\bcnn\b", r"\breuters\b", r"\bassociated\s+press\b", r"\bap\b",
    r"\bal\s*jazeera\b", r"\bjazeera\b", r"\bsky\s*news\b",
    r"\bnews\b", r"\blogo\b", r"\bwatermark\b", r"\bticker\b", r"\btext\b",
    r"\bleft\b", r"\bright\b", r"\btop\b", r"\bbottom\b", r"\bcorner\b",
    r"\bvisible\b", r"\bbackground\b", r"\bforeground\b",
    r"\bimage\b", r"\bshows\b", r"\bshowing\b", r"\bdisplays\b", r"\bdepicts\b",
    r"\bseen\b", r"\bpartially\b",
]

def remove_boilerplate(t: str) -> str:
    """Remove common overlay/layout boilerplate tokens/phrases."""
    for pat in BOILER_PATTERNS:
        t = re.sub(pat, " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def get_top_unigrams(series: pd.Series, topN: int):
    tokens = []
    for t in series:
        words = [w for w in str(t).split() if w not in stop and len(w) > 2]
        tokens.extend(words)
    return Counter(tokens).most_common(topN)

def get_top_bigrams(series: pd.Series, topN: int):
    bigrams = []
    for t in series:
        words = [w for w in str(t).split() if w not in stop and len(w) > 2]
        bigrams.extend(list(zip(words, words[1:])))
    return Counter(bigrams).most_common(topN)

def plot_barh(labels, counts, title, out_path):
    plt.figure(figsize=(10, 6))
    plt.barh(list(reversed(labels)), list(reversed(counts)))
    plt.xlabel("Frequency")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv(PATH)

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

if CAP_COL not in df.columns:
    raise ValueError(f"Caption column '{CAP_COL}' not found. Available columns: {df.columns.tolist()}")

df[CAP_COL] = df[CAP_COL].astype(str).fillna("").str.strip()
df = df[df[CAP_COL].str.len() > 0].copy()


# -------------------------
# Caption length stats + plot
# -------------------------
lens = df[CAP_COL].apply(lambda x: len(x.split()))
print("\nCaption length (words) summary:")
print(lens.describe(percentiles=[.1, .25, .5, .75, .9, .95]).to_string())

plt.figure()
plt.hist(lens, bins=30)
plt.xlabel("Caption length (words)")
plt.ylabel("Count")
plt.title("Caption length distribution")
plt.tight_layout()
plt.savefig("caption_length_hist.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved caption_length_hist.png")


# -------------------------
# Cleaning columns
# -------------------------
df["clean"] = df[CAP_COL].apply(clean_for_freq)
df["clean_debp"] = df["clean"].apply(remove_boilerplate)

print("Created df['clean'] and df['clean_debp']")


# -------------------------
# Raw n-grams (includes boilerplate)
# -------------------------
uni_raw = get_top_unigrams(df["clean"], TOPN)
bi_raw = get_top_bigrams(df["clean"], TOPN)

print("\nTop unigrams (raw, stopwords removed):")
for w, c in uni_raw:
    print(f"{w}\t{c}")

print("\nTop bigrams (raw, stopwords removed):")
for (w1, w2), c in bi_raw:
    print(f"{w1} {w2}\t{c}")

plot_barh(
    labels=[w for w, _ in uni_raw],
    counts=[c for _, c in uni_raw],
    title=f"Top {TOPN} Unigrams (raw, stopwords removed)",
    out_path="eda_top_unigrams.png",
)

plot_barh(
    labels=[" ".join(bg) for bg, _ in bi_raw],
    counts=[c for _, c in bi_raw],
    title=f"Top {TOPN} Bigrams (raw, stopwords removed)",
    out_path="eda_top_bigrams.png",
)


# -------------------------
# Deboilerplated n-grams (semantic view)
# -------------------------
uni_debp = get_top_unigrams(df["clean_debp"], TOPN)
bi_debp = get_top_bigrams(df["clean_debp"], TOPN)

print("\nTop unigrams (deboilerplate, stopwords removed):")
for w, c in uni_debp:
    print(f"{w}\t{c}")

print("\nTop bigrams (deboilerplate, stopwords removed):")
for (w1, w2), c in bi_debp:
    print(f"{w1} {w2}\t{c}")

plot_barh(
    labels=[w for w, _ in uni_debp],
    counts=[c for _, c in uni_debp],
    title=f"Top {TOPN} Unigrams (boilerplate removed)",
    out_path="eda_top_unigrams_deboilerplate.png",
)

plot_barh(
    labels=[" ".join(bg) for bg, _ in bi_debp],
    counts=[c for _, c in bi_debp],
    title=f"Top {TOPN} Bigrams (boilerplate removed)",
    out_path="eda_top_bigrams_deboilerplate.png",
)

print("\nEDA.py finished successfully.")


import spacy
from collections import Counter

print("\nLoading spaCy model...")
nlp = spacy.load("en_core_web_sm")

label_counts = Counter()
person_examples = Counter()
gpe_examples = Counter()
org_examples = Counter()

for text in df[CAP_COL].astype(str):
    doc = nlp(text)
    for ent in doc.ents:
        label_counts[ent.label_] += 1
        if ent.label_ == "PERSON":
            person_examples[ent.text] += 1
        elif ent.label_ == "GPE":
            gpe_examples[ent.text] += 1
        elif ent.label_ == "ORG":
            org_examples[ent.text] += 1

print("\nNER label counts (top):")
for lab, cnt in label_counts.most_common(15):
    print(lab, cnt)

print("\nTop PERSON entities:")
for ent, cnt in person_examples.most_common(15):
    print(ent, cnt)

print("\nTop GPE entities:")
for ent, cnt in gpe_examples.most_common(15):
    print(ent, cnt)

print("\nTop ORG entities:")
for ent, cnt in org_examples.most_common(15):
    print(ent, cnt)



def has_arabic(s: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", str(s)))

df["has_arabic"] = df[CAP_COL].apply(has_arabic)
print(df["has_arabic"].value_counts())
print("Arabic %:", df["has_arabic"].mean() * 100)
