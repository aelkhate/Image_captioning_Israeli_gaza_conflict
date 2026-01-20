import re
import spacy
import pandas as pd

INPUT = "dataset_normalized_en.csv"
OUTPUT = "dataset_train_ready.csv"

nlp = spacy.load("en_core_web_trf")

# --- HARD BLOCKLIST (always removed or masked) ---
PERSON_BLOCKLIST = [
    "netanyahu","benjamin netanyahu",
    "donald trump","trump",
    "emmanuel macron","macron",
    "starmer","david lammy",
    "pezeshkian","witkoff"
]

ORG_BLOCKLIST = [
    "bbc","bbc news","cnn","al jazeera","aljazeera",
    "reuters","associated press","ap"
]

OVERLAY_TERMS = [
    "logo","watermark","live updates","breaking",
    "bottom left","bottom right","top left","top right",
    "left corner","right corner","on screen","onscreen"
]

def blocklist_regex(lst):
    return re.compile(r"\b(" + "|".join(re.escape(x) for x in lst) + r")\b", re.IGNORECASE)

PERSON_RE = blocklist_regex(PERSON_BLOCKLIST)
ORG_RE = blocklist_regex(ORG_BLOCKLIST)
OVERLAY_RE = blocklist_regex(OVERLAY_TERMS)

HYSTERICAL_RE = re.compile(r"\bhysterical\b", re.IGNORECASE)

def clean(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()

def sanitize(text):

    # 1. Remove overlay artifacts
    text = OVERLAY_RE.sub("", text)

    # 2. Remove media org overlays
    text = ORG_RE.sub("", text)

    # 3. Mask known public figures
    text = PERSON_RE.sub("a public official", text)

    # 4. Mask remaining PERSON via NER
    doc = nlp(text)
    for ent in sorted([e for e in doc.ents if e.label_=="PERSON"],
                      key=lambda e: e.start_char, reverse=True):
        text = text[:ent.start_char] + "a person" + text[ent.end_char:]

    # 5. Light MT fix
    if re.search(r"\b(child|person|man|woman|boy|girl|someone)\b", text, re.I):
        text = HYSTERICAL_RE.sub("severely malnourished", text)

    return clean(text)

df = pd.read_csv(INPUT)
df["caption_train"] = df["caption_en"].astype(str).apply(sanitize)
df.to_csv(OUTPUT, index=False)

print("Saved:", OUTPUT)
