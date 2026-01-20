# audit_baseline_quality.py
import pandas as pd
import spacy
from collections import Counter

IN_CSV = "outputs/baseline_val_predictions.cleaned.csv"

nlp = spacy.load("en_core_web_sm")

LOADED_WORDS = [
    "terrorist", "terrorists", "genocide", "genocidal", "barbaric", "evil",
    "heroic", "coward", "cowardly", "savage", "savages", "inhuman",
    "occupier", "occupiers", "martyr", "martyrs", "liberation",
    "massacre", "slaughter", "exterminate", "wipe out"
]

HARMFUL_TONE = [
    "vermin", "animals", "subhuman", "deserve", "good riddance",
    "glorious", "revenge", "wipe them", "kill them all"
]

SPECULATION = [
    "reportedly", "allegedly", "suspected", "likely", "apparently",
    "appears to", "seems to", "claimed", "it is said"
]

OVERLAY = [
    "bbc", "cnn", "al jazeera", "reuters", "ap", "news",
    "logo", "watermark", "caption", "lower third", "corner"
]

def contains_any(text, phrases):
    t = text.lower()
    return [p for p in phrases if p in t]

def main():
    df = pd.read_csv(IN_CSV)
    if "caption_pred_baseline" not in df.columns:
        raise ValueError("caption_pred_baseline column not found.")

    captions = df["caption_pred_baseline"].dropna().astype(str).tolist()
    total = len(captions)

    ner_label_counts = Counter()
    loaded_hits = 0
    harmful_hits = 0
    spec_hits = 0
    overlay_hits = 0
    lengths = []

    examples_loaded = []
    examples_harmful = []
    examples_spec = []
    examples_overlay = []

    for cap in captions:
        lengths.append(len(cap.split()))
        doc = nlp(cap)
        for ent in doc.ents:
            ner_label_counts[ent.label_] += 1

        lw = contains_any(cap, LOADED_WORDS)
        hw = contains_any(cap, HARMFUL_TONE)
        sw = contains_any(cap, SPECULATION)
        ow = contains_any(cap, OVERLAY)

        if lw:
            loaded_hits += 1
            if len(examples_loaded) < 5:
                examples_loaded.append((lw, cap))
        if hw:
            harmful_hits += 1
            if len(examples_harmful) < 5:
                examples_harmful.append((hw, cap))
        if sw:
            spec_hits += 1
            if len(examples_spec) < 5:
                examples_spec.append((sw, cap))
        if ow:
            overlay_hits += 1
            if len(examples_overlay) < 5:
                examples_overlay.append((ow, cap))

    def pct(x):
        return f"{(x / total * 100):.2f}%" if total else "n/a"

    print("\n=== Baseline Quality Audit (cleaned captions) ===")
    print("Total captions:", total)
    print("Avg length (words):", round(sum(lengths) / max(1, len(lengths)), 2))

    print("\nNER label counts (diagnostic):")
    for k, v in ner_label_counts.most_common():
        print(f"  {k:12s} {v}")

    print("\nTone / bias diagnostics:")
    print(f"  Captions with loaded conflict terms: {loaded_hits} ({pct(loaded_hits)})")
    print(f"  Captions with harmful/dehumanizing tone: {harmful_hits} ({pct(harmful_hits)})")
    print(f"  Captions with speculation markers: {spec_hits} ({pct(spec_hits)})")
    print(f"  Captions mentioning overlays/media terms: {overlay_hits} ({pct(overlay_hits)})")

    print("\nExample hits (up to 5 each):")

    print("\n[LOADED]")
    if not examples_loaded:
        print("  (none)")
    else:
        for phrases, cap in examples_loaded:
            print("  phrases:", phrases)
            print("  cap:", cap)
            print()

    print("\n[HARMFUL]")
    if not examples_harmful:
        print("  (none)")
    else:
        for phrases, cap in examples_harmful:
            print("  phrases:", phrases)
            print("  cap:", cap)
            print()

    print("\n[SPEC]")
    if not examples_spec:
        print("  (none)")
    else:
        for phrases, cap in examples_spec:
            print("  phrases:", phrases)
            print("  cap:", cap)
            print()

    print("\n[OVERLAY]")
    if not examples_overlay:
        print("  (none)")
    else:
        for phrases, cap in examples_overlay:
            print("  phrases:", phrases)
            print("  cap:", cap)
            print()

if __name__ == "__main__":
    main()
