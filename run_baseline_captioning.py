#!/usr/bin/env python3
import os
import json
from pathlib import Path

import pandas as pd
from PIL import Image

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

# -----------------------------
# Config
# -----------------------------
VAL_CSV = "splits/val.csv"
IMG_DIR = "images_resolved"

OUT_DIR = "outputs"
OUT_CSV = os.path.join(OUT_DIR, "baseline_val_predictions.csv")
OUT_JSONL = os.path.join(OUT_DIR, "baseline_val_predictions.jsonl")

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

MAX_NEW_TOKENS = 60
TEMPERATURE = 0.2
TOP_P = 0.9

USE_FP16 = True

SYSTEM_STYLE = (
    "You are a cautious image captioning system. "
    "Write ONE concise English caption grounded only in visible content. "
    "Be neutral and conflict-aware. "
    "Avoid naming specific individuals. "
    "Do not celebrate violence. "
    "Do not infer causes, blame, or unverified claims. "
    "If a media overlay/logo/text is visible, mention it briefly without focusing on it."
)

def build_messages():
    # IMPORTANT: Qwen2.5-VL expects the image to be represented as a structured block,
    # so the chat template inserts the special image tokens.
    return [
        {"role": "system", "content": SYSTEM_STYLE},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image in one sentence."},
            ],
        },
    ]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(VAL_CSV)
    if "Img Name" not in df.columns:
        raise ValueError(f"'Img Name' not found in {VAL_CSV}. Columns: {list(df.columns)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    dtype = torch.float16 if (device == "cuda" and USE_FP16) else torch.float32
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    results = []

    with open(OUT_JSONL, "w", encoding="utf-8") as jsonl_f:
        for i, row in df.iterrows():
            img_name = str(row["Img Name"]).strip()
            img_path = Path(IMG_DIR) / img_name
            if not img_path.exists():
                raise FileNotFoundError(f"Missing image: {img_path}")

            image = Image.open(img_path).convert("RGB")

            messages = build_messages()

            # This is the key: the chat template will insert the model’s special image tokens
            # because the message contains {"type":"image"}.
            prompt_text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Provide the image separately here.
            inputs = processor(
                text=[prompt_text],
                images=[image],
                return_tensors="pt",
            )

            # Move tensors to model device
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=(TEMPERATURE > 0),
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                )

            decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            # Qwen often returns the whole transcript: system/user/assistant
            # Keep only what comes after the last "assistant\n"
            marker = "assistant\n"
            if marker in decoded:
                pred = decoded.split(marker)[-1].strip()
            else:
                pred = decoded


            # Reference caption preference order
            caption_ref = None
            for c in ["caption_train", "caption_en", "Desc (S)", "caption_original"]:
                if c in df.columns:
                    val = row.get(c, None)
                    if pd.notna(val):
                        caption_ref = str(val)
                        break

            out = {
                "row_index": int(i),
                "ID": row.get("ID", None),
                "Img Name": img_name,
                "caption_ref": caption_ref,
                "caption_pred_baseline": pred,
            }

            results.append(out)
            jsonl_f.write(json.dumps(out, ensure_ascii=False) + "\n")

            if (i + 1) % 20 == 0:
                print(f"Processed {i+1}/{len(df)}")

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_CSV, index=False)

    print("Saved:", OUT_CSV)
    print("Saved:", OUT_JSONL)
    print("\nSample predictions:")
    print(out_df[["Img Name", "caption_pred_baseline"]].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
