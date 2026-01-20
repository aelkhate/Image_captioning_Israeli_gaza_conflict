#!/usr/bin/env python3
import os
import json
from typing import Dict, Any, List

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel


# =========================
# Config
# =========================
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# Your trained artifacts
ADAPTER_DIR = "outputs/qlora_qwen25vl_caption_train/adapter"
PROCESSOR_DIR = "outputs/qlora_qwen25vl_caption_train/processor"

VAL_CSV = "splits/val.csv"
IMAGES_DIR = "images_resolved"

OUT_DIR = "outputs"
OUT_CSV = os.path.join(OUT_DIR, "finetuned_val_predictions.csv")
OUT_JSONL = os.path.join(OUT_DIR, "finetuned_val_predictions.jsonl")

MAX_NEW_TOKENS = 48
TEMPERATURE = 0.2
TOP_P = 0.9

SYSTEM_PROMPT = (
    "You are an image captioning system for the Gaza–Israel conflict dataset. "
    "Write ONE concise English caption grounded only in visible content. "
    "Be accurate, descriptive, and unbiased. Use an empathetic tone. "
    "Avoid naming specific individuals unless the name is clearly shown in the image. "
    "Do not infer causes, blame, or unverified claims."
)
USER_PROMPT = "Describe this image in one sentence."


# =========================
# Helpers
# =========================
def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def build_multimodal_prompt(processor: AutoProcessor) -> str:
    """
    Build a chat-style prompt that *explicitly includes an image slot*.
    This is what prevents: tokens:0, features:384
    """
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt_text


def decode_new_tokens(processor: AutoProcessor, input_ids: torch.Tensor, generated_ids: torch.Tensor) -> str:
    """
    Keep only the newly generated tokens after the prompt.
    This avoids returning the whole system/user chat in your output.
    """
    prompt_len = input_ids.shape[-1]
    new_token_ids = generated_ids[0, prompt_len:]
    text = processor.decode(new_token_ids, skip_special_tokens=True).strip()
    return text


# =========================
# Main
# =========================
def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    # Load processor (prefer your saved processor to keep settings consistent with training)
    if os.path.isdir(PROCESSOR_DIR):
        processor = AutoProcessor.from_pretrained(PROCESSOR_DIR, trust_remote_code=True)
    else:
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Load base model
    base_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )

    # Attach LoRA adapter
    if not os.path.isdir(ADAPTER_DIR):
        raise FileNotFoundError(f"Adapter dir not found: {ADAPTER_DIR}")

    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    # Build prompt text once (same prompt for all images)
    prompt_text = build_multimodal_prompt(processor)

    # Load val split
    df = pd.read_csv(VAL_CSV)
    if "Img Name" not in df.columns:
        raise ValueError("VAL CSV must contain column: 'Img Name'")

    outputs: List[Dict[str, Any]] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Captioning (finetuned)"):
        img_name = str(row["Img Name"])
        img_path = os.path.join(IMAGES_DIR, img_name)

        if not os.path.exists(img_path):
            # Keep track but don't crash
            outputs.append(
                {
                    "Img Name": img_name,
                    "caption_pred_finetuned": "",
                    "error": f"Missing image: {img_path}",
                }
            )
            continue

        image = load_image(img_path)

        # IMPORTANT:
        # - DO NOT use truncation here. Truncation can delete the <image> token and cause mismatch.
        model_inputs = processor(
            images=image,
            text=prompt_text,
            return_tensors="pt",
            padding=True,
        )

        # Move tensors to the same device as the model
        for k, v in model_inputs.items():
            if isinstance(v, torch.Tensor):
                model_inputs[k] = v.to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )

        caption = decode_new_tokens(processor, model_inputs["input_ids"], generated_ids)

        outputs.append(
            {
                "Img Name": img_name,
                "caption_pred_finetuned": caption,
                "error": "",
            }
        )

    out_df = pd.DataFrame(outputs)
    out_df.to_csv(OUT_CSV, index=False)

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in outputs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved: {OUT_CSV}")
    print(f"Saved: {OUT_JSONL}")

    # quick peek
    print("\nSample predictions:")
    print(out_df[["Img Name", "caption_pred_finetuned"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
