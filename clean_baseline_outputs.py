# clean_baseline_outputs.py
import pandas as pd

IN_CSV = "outputs/baseline_val_predictions.csv"
OUT_CSV = "outputs/baseline_val_predictions.cleaned.csv"

def strip_chat(x: str) -> str:
    if not isinstance(x, str):
        return x
    marker = "assistant\n"
    if marker in x:
        return x.split(marker)[-1].strip()
    marker2 = "assistant:"
    if marker2 in x:
        return x.split(marker2)[-1].strip()
    return x.strip()

def main():
    df = pd.read_csv(IN_CSV)
    if "caption_pred_baseline" not in df.columns:
        raise ValueError("caption_pred_baseline column not found in input CSV.")
    df["caption_pred_baseline"] = df["caption_pred_baseline"].apply(strip_chat)
    df.to_csv(OUT_CSV, index=False)
    print("Saved:", OUT_CSV)

if __name__ == "__main__":
    main()
