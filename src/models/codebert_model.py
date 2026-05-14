"""Model 4 — CodeBERT fine-tuned on (message [SEP] diff).

Same training pipeline as ``distilbert_model`` (HF Trainer + balanced
subsampling); only the backbone changes.

Run:
    python -m src.models.codebert_model train [--max-train 6000]
    python -m src.models.codebert_model eval
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
from rich.console import Console
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.config import IDX_TO_CLASS, MODELS_DIR, NUM_CLASSES, RANDOM_SEED
from src.models.distilbert_model import (
    BATCH_SIZE,
    CommitDataset,
    EPOCHS,
    LR,
    MAX_SEQ_LEN,
    _balanced_subsample,
    _compute_metrics,
    _device,
)
from src.utils import evaluation_report, load_split, print_report, save_report

console = Console()
MODEL_NAME = "microsoft/codebert-base"
ARTIFACT_DIR = MODELS_DIR / "codebert"
REPORT_PATH = MODELS_DIR / "reports" / "codebert.json"

DEFAULT_MAX_TRAIN = 6000

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def train(max_train: int) -> int:
    console.log(f"Using device: {_device()}")

    train_df = load_split("train")
    val_df = load_split("val")
    train_df = _balanced_subsample(train_df, max_train)
    console.log(f"train (balanced subsample) = {len(train_df):,}, val = {len(val_df):,}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        id2label=IDX_TO_CLASS,
        label2id={v: k for k, v in IDX_TO_CLASS.items()},
    )

    train_ds = CommitDataset(train_df, tokenizer, MAX_SEQ_LEN)
    val_ds = CommitDataset(val_df, tokenizer, MAX_SEQ_LEN)

    args = TrainingArguments(
        output_dir=str(ARTIFACT_DIR / "_trainer"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to=[],
        seed=RANDOM_SEED,
        fp16=False,
        bf16=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=_compute_metrics,
    )
    trainer.train()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(ARTIFACT_DIR))
    tokenizer.save_pretrained(ARTIFACT_DIR)
    console.log(f"  ✓ Saved best model → {ARTIFACT_DIR}")

    return _eval_split("val", REPORT_PATH.with_name("codebert_val.json"))


def _eval_split(split_name: str, out_path) -> int:
    if not (ARTIFACT_DIR / "config.json").exists():
        console.print(f"[red]✗ Missing artifact in {ARTIFACT_DIR}. Train first.[/red]")
        return 1
    device = _device()
    tokenizer = AutoTokenizer.from_pretrained(ARTIFACT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(ARTIFACT_DIR).to(device).eval()

    df = load_split(split_name)
    ds = CommitDataset(df, tokenizer, MAX_SEQ_LEN)
    loader = DataLoader(ds, batch_size=BATCH_SIZE * 2)

    preds: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds.extend(logits.argmax(dim=-1).cpu().tolist())

    y_true = df["label_id"].values.astype(int)
    y_pred = np.array(preds, dtype=int)
    report = evaluation_report(y_true, y_pred, model_name=f"codebert [{split_name}]")
    print_report(report)
    save_report(report, out_path)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train", "eval", "all"])
    parser.add_argument("--max-train", type=int, default=DEFAULT_MAX_TRAIN)
    args = parser.parse_args()
    if args.action == "train":
        return train(args.max_train)
    if args.action == "eval":
        return _eval_split("test", REPORT_PATH)
    rc = train(args.max_train)
    return rc or _eval_split("test", REPORT_PATH)


if __name__ == "__main__":
    sys.exit(main())
