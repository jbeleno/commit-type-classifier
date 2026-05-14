"""Unified inference layer.

Loads any of the trained models once and exposes ``predict()`` /
``predict_batch()``. Used by the Streamlit GUI and the CLI so neither
needs to know how each model is built or serialized.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.config import IDX_TO_CLASS, MODELS_DIR, TARGET_CLASSES
from src.models.baseline_tfidf import ARTIFACT as BASELINE_PATH, NUMERIC_COLS
from src.data.preprocess import diff_to_text_and_features, strip_prefix

AVAILABLE_MODELS = ["baseline_tfidf", "cnn_text", "distilbert", "codebert", "ensemble"]


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: Dict[str, float]
    model: str


def _row_from_inputs(message: str, diff: str) -> pd.DataFrame:
    msg_clean = strip_prefix(message)
    diff_text, files, added, removed = diff_to_text_and_features(diff)
    return pd.DataFrame(
        [
            {
                "message_clean": msg_clean,
                "diff_text": diff_text,
                "files_changed": files,
                "lines_added": added,
                "lines_removed": removed,
            }
        ]
    )


@lru_cache(maxsize=1)
def _baseline_bundle():
    return joblib.load(BASELINE_PATH)


def _proba_baseline(df: pd.DataFrame) -> np.ndarray:
    b = _baseline_bundle()
    msg_mat = b["vec_msg"].transform(df["message_clean"].astype(str).values)
    diff_mat = b["vec_diff"].transform(df["diff_text"].astype(str).values)
    num = b["scaler"].transform(df[NUMERIC_COLS].astype(float).values)
    X = sp.hstack([msg_mat, diff_mat, sp.csr_matrix(num)]).tocsr()
    return b["clf"].predict_proba(X)


@lru_cache(maxsize=1)
def _cnn_bundle():
    from src.models.cnn_text import _load_artifact

    return _load_artifact()


def _proba_cnn(df: pd.DataFrame) -> np.ndarray:
    from src.models.cnn_text import BATCH_SIZE, _to_dataset, _to_inputs

    model, scaler = _cnn_bundle()
    inputs = _to_inputs(df, scaler, fit=False)
    ds = _to_dataset(inputs, None, BATCH_SIZE, shuffle=False)
    return model.predict(ds, verbose=0)


@lru_cache(maxsize=2)
def _hf_bundle(artifact_dir: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.models.distilbert_model import _device

    tokenizer = AutoTokenizer.from_pretrained(artifact_dir)
    model = AutoModelForSequenceClassification.from_pretrained(artifact_dir)
    device = _device()
    model.to(device)
    model.eval()
    return tokenizer, model, device


def _proba_hf(df: pd.DataFrame, artifact_dir: Path) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader

    from src.models.distilbert_model import BATCH_SIZE, CommitDataset, MAX_SEQ_LEN

    tokenizer, model, device = _hf_bundle(str(artifact_dir))
    df_with_labels = df.copy()
    df_with_labels["label_id"] = 0
    ds = CommitDataset(df_with_labels, tokenizer, MAX_SEQ_LEN)
    loader = DataLoader(ds, batch_size=BATCH_SIZE)
    out = []
    with torch.no_grad():
        for batch in loader:
            batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def _proba_ensemble(df: pd.DataFrame) -> np.ndarray:
    cfg_path = MODELS_DIR / "ensemble" / "weights.json"
    cfg = json.loads(cfg_path.read_text())
    model_dispatch = {
        "baseline_tfidf": _proba_baseline,
        "cnn_text": _proba_cnn,
        "distilbert": lambda d: _proba_hf(d, MODELS_DIR / "distilbert"),
        "codebert": lambda d: _proba_hf(d, MODELS_DIR / "codebert"),
    }
    weights = np.array(cfg["weights"])
    weights = weights / weights.sum()
    probs = np.stack([model_dispatch[n](df) for n in cfg["models"]], axis=0)
    return (probs * weights[:, None, None]).sum(axis=0)


_DISPATCH = {
    "baseline_tfidf": _proba_baseline,
    "cnn_text": _proba_cnn,
    "distilbert": lambda d: _proba_hf(d, MODELS_DIR / "distilbert"),
    "codebert": lambda d: _proba_hf(d, MODELS_DIR / "codebert"),
    "ensemble": _proba_ensemble,
}


def model_is_available(model_name: str) -> bool:
    if model_name == "baseline_tfidf":
        return BASELINE_PATH.exists()
    if model_name == "cnn_text":
        return (MODELS_DIR / "cnn_text" / "model.weights.h5").exists()
    if model_name == "distilbert":
        return (MODELS_DIR / "distilbert" / "config.json").exists()
    if model_name == "codebert":
        return (MODELS_DIR / "codebert" / "config.json").exists()
    if model_name == "ensemble":
        return (MODELS_DIR / "ensemble" / "weights.json").exists()
    return False


def predict(message: str, diff: str, model_name: str) -> Prediction:
    if model_name not in _DISPATCH:
        raise ValueError(f"Unknown model: {model_name}")
    if not model_is_available(model_name):
        raise FileNotFoundError(
            f"Model '{model_name}' has no saved artifact. Train it first."
        )
    df = _row_from_inputs(message, diff)
    probs = _DISPATCH[model_name](df)[0]
    label_id = int(np.argmax(probs))
    return Prediction(
        label=IDX_TO_CLASS[label_id],
        confidence=float(probs[label_id]),
        probabilities={TARGET_CLASSES[i]: float(p) for i, p in enumerate(probs)},
        model=model_name,
    )


def predict_batch(records: List[Dict[str, str]], model_name: str) -> List[Prediction]:
    if model_name not in _DISPATCH:
        raise ValueError(f"Unknown model: {model_name}")
    if not model_is_available(model_name):
        raise FileNotFoundError(f"Model '{model_name}' has no saved artifact.")
    df = pd.concat(
        [_row_from_inputs(r.get("message", ""), r.get("diff", "")) for r in records],
        ignore_index=True,
    )
    probs = _DISPATCH[model_name](df)
    out: List[Prediction] = []
    for row in probs:
        label_id = int(np.argmax(row))
        out.append(
            Prediction(
                label=IDX_TO_CLASS[label_id],
                confidence=float(row[label_id]),
                probabilities={TARGET_CLASSES[i]: float(p) for i, p in enumerate(row)},
                model=model_name,
            )
        )
    return out
