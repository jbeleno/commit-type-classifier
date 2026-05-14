"""Model 2 — Dual-branch CNN-text (message + diff) with numeric features.

Architecture
    msg  -> TextVectorization -> Embedding -> Conv1D(filters=128, k=3) -> GMP
    diff -> TextVectorization -> Embedding -> Conv1D(filters=128, k=5) -> GMP
    num  -> StandardScaler -> Dense(16, relu)
    concat -> Dense(128, relu) -> Dropout(0.4) -> Dense(5, softmax)

Saved artifact:
    models_saved/cnn_text/         (Keras SavedModel + adjuncts)
    models_saved/cnn_text/scaler.joblib

Run:
    python -m src.models.cnn_text train
    python -m src.models.cnn_text eval
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from rich.console import Console
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

from src.config import MODELS_DIR, NUM_CLASSES, RANDOM_SEED
from src.utils import class_weights_for, evaluation_report, load_split, print_report, save_report

console = Console()
ARTIFACT_DIR = MODELS_DIR / "cnn_text"
REPORT_PATH = MODELS_DIR / "reports" / "cnn_text.json"

NUMERIC_COLS = ["files_changed", "lines_added", "lines_removed"]
MSG_VOCAB = 20_000
DIFF_VOCAB = 40_000
MSG_LEN = 48
DIFF_LEN = 384
EMBED_DIM = 64
BATCH_SIZE = 128
EPOCHS = 12

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def _make_vectorizers(msg_vocab: list[str] | None, diff_vocab: list[str] | None):
    msg_vec = layers.TextVectorization(
        max_tokens=MSG_VOCAB,
        output_sequence_length=MSG_LEN,
        standardize="lower_and_strip_punctuation",
    )
    diff_vec = layers.TextVectorization(
        max_tokens=DIFF_VOCAB,
        output_sequence_length=DIFF_LEN,
        standardize="lower_and_strip_punctuation",
    )
    if msg_vocab is not None:
        msg_vec.set_vocabulary(msg_vocab)
    if diff_vocab is not None:
        diff_vec.set_vocabulary(diff_vocab)
    return msg_vec, diff_vec


def build_model(msg_vec: layers.TextVectorization, diff_vec: layers.TextVectorization) -> keras.Model:
    msg_in = keras.Input(shape=(), dtype=tf.string, name="message")
    diff_in = keras.Input(shape=(), dtype=tf.string, name="diff")
    num_in = keras.Input(shape=(len(NUMERIC_COLS),), dtype=tf.float32, name="numeric")

    m = msg_vec(msg_in)
    m = layers.Embedding(MSG_VOCAB + 2, EMBED_DIM, mask_zero=True)(m)
    m = layers.Conv1D(128, 3, padding="same", activation="relu")(m)
    m = layers.GlobalMaxPooling1D()(m)

    d = diff_vec(diff_in)
    d = layers.Embedding(DIFF_VOCAB + 2, EMBED_DIM, mask_zero=True)(d)
    d = layers.Conv1D(128, 5, padding="same", activation="relu")(d)
    d = layers.GlobalMaxPooling1D()(d)

    n = layers.Dense(16, activation="relu")(num_in)

    x = layers.Concatenate()([m, d, n])
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model([msg_in, diff_in, num_in], out)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _to_inputs(df, scaler, fit: bool):
    msg = tf.constant(df["message_clean"].astype(str).tolist())
    diff = tf.constant(df["diff_text"].astype(str).tolist())
    num = df[NUMERIC_COLS].astype(float).values
    num = scaler.fit_transform(num) if fit else scaler.transform(num)
    return {"message": msg, "diff": diff, "numeric": tf.constant(num, dtype=tf.float32)}


def _to_dataset(inputs: dict, y: np.ndarray | None, batch_size: int, shuffle: bool):
    if y is None:
        ds = tf.data.Dataset.from_tensor_slices(inputs)
    else:
        ds = tf.data.Dataset.from_tensor_slices((inputs, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=8192, seed=RANDOM_SEED)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def train() -> int:
    train_df = load_split("train")
    val_df = load_split("val")
    console.log(f"train={len(train_df):,} val={len(val_df):,}")

    msg_vec, diff_vec = _make_vectorizers(None, None)
    console.log("Adapting TextVectorization layers ...")
    msg_vec.adapt(np.asarray(train_df["message_clean"].astype(str).tolist(), dtype=object))
    diff_vec.adapt(np.asarray(train_df["diff_text"].astype(str).tolist(), dtype=object))

    scaler = StandardScaler()
    X_train = _to_inputs(train_df, scaler, fit=True)
    X_val = _to_inputs(val_df, scaler, fit=False)
    y_train = train_df["label_id"].values.astype(np.int32)
    y_val = val_df["label_id"].values.astype(np.int32)

    model = build_model(msg_vec, diff_vec)
    model.summary(print_fn=lambda s: console.log(s))

    cw = class_weights_for(y_train)
    console.log(f"class_weights = {cw}")

    es = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    train_ds = _to_dataset(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_ds = _to_dataset(X_val, y_val, BATCH_SIZE, shuffle=False)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=cw,
        callbacks=[es],
        verbose=2,
    )

    val_pred = np.argmax(model.predict(val_ds, verbose=0), axis=1)
    report = evaluation_report(y_val, val_pred, model_name="cnn_text [val]")
    print_report(report)
    save_report(report, REPORT_PATH.with_name("cnn_text_val.json"))

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_weights(ARTIFACT_DIR / "model.weights.h5")
    joblib.dump(scaler, ARTIFACT_DIR / "scaler.joblib")
    (ARTIFACT_DIR / "msg_vocab.json").write_text(
        json.dumps(msg_vec.get_vocabulary(include_special_tokens=False))
    )
    (ARTIFACT_DIR / "diff_vocab.json").write_text(
        json.dumps(diff_vec.get_vocabulary(include_special_tokens=False))
    )
    console.print(f"[green]✓ Saved → {ARTIFACT_DIR}[/green]")
    return 0


def _load_artifact() -> tuple[keras.Model, StandardScaler]:
    msg_vocab = json.loads((ARTIFACT_DIR / "msg_vocab.json").read_text())
    diff_vocab = json.loads((ARTIFACT_DIR / "diff_vocab.json").read_text())
    msg_vec, diff_vec = _make_vectorizers(msg_vocab, diff_vocab)
    model = build_model(msg_vec, diff_vec)
    model.load_weights(ARTIFACT_DIR / "model.weights.h5")
    scaler = joblib.load(ARTIFACT_DIR / "scaler.joblib")
    return model, scaler


def evaluate() -> int:
    weights_path = ARTIFACT_DIR / "model.weights.h5"
    scaler_path = ARTIFACT_DIR / "scaler.joblib"
    if not weights_path.exists() or not scaler_path.exists():
        console.print(f"[red]✗ Missing artifact in {ARTIFACT_DIR}. Train first.[/red]")
        return 1
    model, scaler = _load_artifact()
    test_df = load_split("test")
    X = _to_inputs(test_df, scaler, fit=False)
    y_true = test_df["label_id"].values.astype(np.int32)
    test_ds = _to_dataset(X, None, BATCH_SIZE, shuffle=False)
    y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)
    report = evaluation_report(y_true, y_pred, model_name="cnn_text [test]")
    print_report(report)
    save_report(report, REPORT_PATH)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train", "eval", "all"])
    args = parser.parse_args()
    if args.action == "train":
        return train()
    if args.action == "eval":
        return evaluate()
    rc = train()
    return rc or evaluate()


if __name__ == "__main__":
    sys.exit(main())
