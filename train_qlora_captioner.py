#!/usr/bin/env python3
"""
train_qlora_captioner.py

Fix applied:
- DO NOT use tokenizer truncation/max_length inside processor(...) because it can break
  Qwen2.5-VL's multimodal placeholder token alignment (<image>).
- Instead: truncate only the *caption string* safely, and call processor with truncation=False.

Inputs:
- splits/train.csv
- splits/val.csv (optional)
- images_resolved/<Img Name>

Expected CSV columns:
- Img Name
- caption_train

Outputs:
- outputs/qlora_qwen25vl_caption_train/
  - adapter/
  - processor/
  - checkpoints/
  - run_metadata.json
"""

import os
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import pandas as pd
from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


# -----------------------------
# Hardcoded paths / settings
# -----------------------------
TRAIN_CSV = "splits/train.csv"
VAL_CSV = "splits/val.csv"  # optional
IMAGES_DIR = "images_resolved"

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

RUN_DIR = "outputs/qlora_qwen25vl_caption_train"
CKPT_DIR = os.path.join(RUN_DIR, "checkpoints")
ADAPTER_DIR = os.path.join(RUN_DIR, "adapter")
PROCESSOR_DIR = os.path.join(RUN_DIR, "processor")
RUN_META_PATH = os.path.join(RUN_DIR, "run_metadata.json")

CAPTION_COL = "caption_train"
IMGNAME_COL = "Img Name"

SEED = 42

# Instead of token max_length (dangerous for <image>), we cap the *caption string*.
# Adjust if you want longer GT captions.
MAX_CAPTION_CHARS = 320

# Training parameters
NUM_EPOCHS = 1
BATCH_SIZE = 1
GRAD_ACCUM = 16
LR = 2e-4
WARMUP_RATIO = 0.05
SAVE_STEPS = 200
LOGGING_STEPS = 10

# DataLoader workers:
# Set to 0 to avoid multiprocessing + PIL edge cases and to get cleaner stack traces.
DATALOADER_WORKERS = 0

# QLoRA parameters
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# Prompting policy (aligned to official statement)
SYSTEM_PROMPT = (
    "You are a cautious image captioning system. Write ONE concise English caption "
    "grounded only in visible content. Be accurate, descriptive, and unbiased. "
    "Use an empathetic, non-celebratory tone. Do not infer causes, blame, or unverified claims. "
    "Avoid naming specific individuals unless the text explicitly shows a name; prefer role-based descriptions. "
    "If a media overlay/logo/text is visible, mention it briefly without focusing on it."
)
USER_PROMPT = "Describe this image in one sentence."


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    os.makedirs(RUN_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(ADAPTER_DIR, exist_ok=True)
    os.makedirs(PROCESSOR_DIR, exist_ok=True)


def safe_open_image(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def file_exists(p: str) -> bool:
    try:
        return os.path.isfile(p)
    except Exception:
        return False


def clamp_caption(caption: str, max_chars: int) -> str:
    caption = (caption or "").strip()
    if len(caption) <= max_chars:
        return caption
    # Cut at a boundary if possible
    cut = caption[:max_chars].rsplit(" ", 1)[0].strip()
    return cut if cut else caption[:max_chars].strip()


# -----------------------------
# Dataset
# -----------------------------
class CaptionTrainDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path: str, images_dir: str, caption_col: str, imgname_col: str):
        self.csv_path = csv_path
        self.images_dir = images_dir
        self.caption_col = caption_col
        self.imgname_col = imgname_col

        df = pd.read_csv(csv_path)

        total = len(df)
        kept = 0
        dropped_missing_img = 0
        dropped_empty_cap = 0

        items: List[Tuple[str, str, int]] = []

        for idx, row in df.iterrows():
            img_name = str(row.get(self.imgname_col, "")).strip()
            caption = str(row.get(self.caption_col, "")).strip()

            if not caption or caption.lower() == "nan":
                dropped_empty_cap += 1
                continue

            img_path = os.path.join(images_dir, img_name)
            if not file_exists(img_path):
                dropped_missing_img += 1
                continue

            row_id = int(row.get("ID", idx)) if str(row.get("ID", "")).strip() else idx
            items.append((img_path, caption, row_id))
            kept += 1

        self.items = items
        self.stats = {
            "csv_path": csv_path,
            "images_dir": images_dir,
            "total_rows": total,
            "kept_rows": kept,
            "dropped_missing_images": dropped_missing_img,
            "dropped_empty_captions": dropped_empty_cap,
        }

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        img_path, caption, row_id = self.items[i]
        return {"img_path": img_path, "caption": caption, "row_id": row_id}


# -----------------------------
# Collator (IMPORTANT FIX HERE)
# -----------------------------
@dataclass
class QwenVLCollator:
    processor: Any
    max_caption_chars: int

    def _build_messages_full(self, caption: str) -> List[Dict[str, Any]]:
        caption = clamp_caption(caption, self.max_caption_chars)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
            {"role": "assistant", "content": caption},
        ]

    def _build_messages_prompt_only(self) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ]

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images: List[Image.Image] = []
        full_texts: List[str] = []
        prompt_texts: List[str] = []

        for ex in batch:
            img = safe_open_image(ex["img_path"])
            images.append(img)

            msgs_full = self._build_messages_full(ex["caption"])
            msgs_prompt = self._build_messages_prompt_only()

            # Critical: ensure <image> placeholder stays intact
            full_text = self.processor.apply_chat_template(
                msgs_full, tokenize=False, add_generation_prompt=False
            )
            prompt_text = self.processor.apply_chat_template(
                msgs_prompt, tokenize=False, add_generation_prompt=True
            )

            full_texts.append(full_text)
            prompt_texts.append(prompt_text)

        # ✅ FIX: DO NOT TRUNCATE here (truncation can break <image> token alignment)
        model_inputs = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )

        input_ids = model_inputs["input_ids"]
        labels = input_ids.clone()

        prompt_inputs = self.processor(
            text=prompt_texts,
            images=images,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )

        prompt_ids = prompt_inputs["input_ids"]
        pad_id = getattr(self.processor.tokenizer, "pad_token_id", 0)

        # Mask all tokens up to the end of the prompt (system+user); compute loss on assistant only
        for i in range(labels.size(0)):
            prompt_len = int((prompt_ids[i] != pad_id).sum().item())
            labels[i, :prompt_len] = -100

        model_inputs["labels"] = labels
        return model_inputs


# -----------------------------
# Version-safe TrainingArguments builder
# -----------------------------
def build_training_args(ckpt_dir: str, val_ds_exists: bool) -> TrainingArguments:
    common = dict(
        output_dir=ckpt_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        per_device_eval_batch_size=1,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.0,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        fp16=True,
        bf16=False,
        report_to="none",
        dataloader_num_workers=DATALOADER_WORKERS,
        remove_unused_columns=False,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        max_grad_norm=1.0,
    )

    try:
        return TrainingArguments(
            **common,
            evaluation_strategy="steps" if val_ds_exists else "no",
            eval_steps=SAVE_STEPS if val_ds_exists else None,
            save_strategy="steps",
        )
    except TypeError:
        return TrainingArguments(**common)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ensure_dirs()
    set_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_ds = CaptionTrainDataset(
        csv_path=TRAIN_CSV,
        images_dir=IMAGES_DIR,
        caption_col=CAPTION_COL,
        imgname_col=IMGNAME_COL,
    )

    val_ds = None
    if file_exists(VAL_CSV):
        val_ds = CaptionTrainDataset(
            csv_path=VAL_CSV,
            images_dir=IMAGES_DIR,
            caption_col=CAPTION_COL,
            imgname_col=IMGNAME_COL,
        )

    run_meta = {
        "model_id": MODEL_ID,
        "train_stats": train_ds.stats,
        "val_stats": val_ds.stats if val_ds is not None else None,
        "caption_col": CAPTION_COL,
        "imgname_col": IMGNAME_COL,
        "images_dir": IMAGES_DIR,
        "seed": SEED,
        "max_caption_chars": MAX_CAPTION_CHARS,
        "training": {
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "lr": LR,
            "warmup_ratio": WARMUP_RATIO,
            "save_steps": SAVE_STEPS,
            "logging_steps": LOGGING_STEPS,
            "dataloader_workers": DATALOADER_WORKERS,
        },
        "qlora": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": "float16",
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
        },
    }
    with open(RUN_META_PATH, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    print("Train dataset:", train_ds.stats)
    if val_ds is not None:
        print("Val dataset:", val_ds.stats)

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    model = prepare_model_for_kbit_training(model)

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_config)

    # Print trainable params
    trainable = 0
    total = 0
    for _, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    pct = 100.0 * trainable / max(total, 1)
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.4f}%)")

    collator = QwenVLCollator(processor=processor, max_caption_chars=MAX_CAPTION_CHARS)
    args = build_training_args(CKPT_DIR, val_ds is not None)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds if val_ds is not None else None,
        data_collator=collator,
    )

    trainer.train()

    trainer.model.save_pretrained(ADAPTER_DIR)
    processor.save_pretrained(PROCESSOR_DIR)

    print(f"Saved adapter to: {ADAPTER_DIR}")
    print(f"Saved processor to: {PROCESSOR_DIR}")
    print(f"Saved run metadata to: {RUN_META_PATH}")


if __name__ == "__main__":
    main()
