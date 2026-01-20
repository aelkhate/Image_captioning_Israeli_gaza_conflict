import os
import pandas as pd

CSV = "dataset_train_ready.csv"
IMG_DIR = "images/downloaded_images/images_resolved/"
COL = "Img Name"

df = pd.read_csv(CSV)
names = df[COL].astype(str).str.strip().tolist()

missing = [n for n in names if not os.path.exists(os.path.join(IMG_DIR, n))]
extra = set(os.listdir(IMG_DIR)) - set(names)

print("Rows:", len(df))
print("Unique image names in CSV:", len(set(names)))
print("Missing images:", len(missing))
print("Extra images:", len(extra))

if missing:
    print("First missing:", missing[:20])
