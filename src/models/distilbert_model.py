"""Model 3 — DistilBERT fine-tuned on (message [SEP] diff).

Uses the HuggingFace Trainer API (stable, well-tested) instead of a
hand-rolled loop. Class imbalance is handled by stratified
under-sampling of the majority class before training, which is more
reliable on MPS than passing class weights to CrossEntropyLoss.

Saved artifact:
    models_saved/distilbert/        (Trainer-saved checkpoint dir)

Run:
    python -m src.models.distilbert_model train [--max-train 8000]
    python -m src.models.distilbert_model eval
"""
from __future__ import annotations

import argparse
import sys
from typing import Dict

import numpy as np
import torch
from rich.console import Console
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.config import IDX_TO_CLASS, MODELS_DIR, NUM_CLASSES, RANDOM_SEED
from src.utils import evaluation_report, load_split, print_report, save_report

console = Console()
MODEL_NAME = "distilbert-base-uncased"
ARTIFACT_DIR = MODELS_DIR / "distilbert"
REPORT_PATH = MODELS_DIR / "reports" / "distilbert.json"

MAX_SEQ_LEN = 256
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
DEFAULT_MAX_TRAIN = 8000

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _balanced_subsample(df, max_n: int):
    """Undersample majority classes so each class has at most max_n / NUM_CLASSES rows."""
    per_class = max(64, max_n // NUM_CLASSES)
    chunks = []
    for label_id in range(NUM_CLASSES):
        sub = df[df["label_id"] == label_id]
        if len(sub) > per_class:
            sub = sub.sample(n=per_class, random_state=RANDOM_SEED)
        chunks.append(sub)
    out = (
        np.concatenate([c.index.values for c in chunks])
        if chunks
        else np.array([])
    )
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(out)
    return df.loc[out].reset_index(drop=True)


def _stratified_subsample(df, max_n: int):
    if max_n <= 0 or len(df) <= max_n:
        return df
    sub, _ = train_test_split(
        df, train_size=max_n, random_state=RANDOM_SEED, stratify=df["label_id"]
    )
    return sub.reset_index(drop=True)


class CommitDataset(Dataset):
    def __init__(self, df, tokenizer, max_length: int):
        self.messages = df["message_clean"].astype(str).tolist()
        self.diffs = df["diff_text"].astype(str).tolist()
        self.labels = df["label_id"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.messages[idx],
            self.diffs[idx],
            truncation="longest_first",
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    from sklearn.metrics import f1_score
    return {
        "accuracy": float((preds == labels).mean()),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


def train(max_train: int) -> int:
    console.log(f"Using device: {_device()}")

    train_df = load_split("train")
    val_df = load_split("val")
    train_df = _balanced_subsample(train_df, max_train)
    console.log(f"train (balanced subsample) = {len(train_df):,}, val = {len(val_df):,}")
    console.log("class counts after balancing:")
    for k, v in train_df["label"].value_counts().items():
        console.log(f"  {k}: {v}")

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

    return _eval_split("val", REPORT_PATH.with_name("distilbert_val.json"))


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
    report = evaluation_report(y_true, y_pred, model_name=f"distilbert [{split_name}]")
    print_report(report)
    save_report(report, out_path)
    return 0


def _run_epoch(*args, **kwargs):
    raise RuntimeError("Trainer API is used now; _run_epoch is no longer in use")


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
