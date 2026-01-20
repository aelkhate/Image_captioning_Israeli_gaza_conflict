import spacy
import pandas as pd
from collections import Counter, defaultdict

nlp = spacy.load("en_core_web_trf")

df = pd.read_csv("dataset_normalized_en.csv")

counter = Counter()
examples = defaultdict(list)

for cap in df["caption_en"].astype(str):
    doc = nlp(cap)
    for ent in doc.ents:
        counter[ent.label_] += 1
        if len(examples[ent.label_]) < 20:
            examples[ent.label_].append(ent.text)

print(counter)
for k in examples:
    print(k, examples[k])
