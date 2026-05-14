"""Functional tests for the train/val/test split CSVs (no data leakage)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.config import CLASS_TO_IDX, SPLITS_DIR, SPLIT_RATIOS, TARGET_CLASSES


@pytest.fixture(scope="module")
def splits() -> dict[str, pd.DataFrame]:
    paths = {name: SPLITS_DIR / f"{name}.csv" for name in ("train", "val", "test")}
    if not all(p.exists() for p in paths.values()):
        pytest.skip("split CSVs not yet generated; run `python -m src.data.split`")
    return {name: pd.read_csv(p) for name, p in paths.items()}


def test_splits_total_matches_processed(splits):
    """Total rows across splits equals the input clean dataset."""
    from src.config import PROCESSED_DIR

    clean = PROCESSED_DIR / "commits_clean.parquet"
    if not clean.exists():
        pytest.skip("processed parquet not available")
    total = sum(len(df) for df in splits.values())
    assert total == len(pd.read_parquet(clean))


def test_duplicate_text_overlap_is_minor(splits):
    """Identical message+diff tuples can repeat naturally in CommitBench; ensure overlap stays
    under 10% of test set so the split is mostly free of unintentional leakage."""
    keys = lambda df: set(zip(df["message_clean"].astype(str), df["diff_text"].astype(str)))  # noqa: E731
    train_keys = keys(splits["train"])
    test_keys = keys(splits["test"])
    overlap = len(test_keys & train_keys)
    assert overlap / max(len(test_keys), 1) < 0.10


def test_split_ratios_approximately_correct(splits):
    total = sum(len(df) for df in splits.values())
    assert abs(len(splits["train"]) / total - SPLIT_RATIOS["train"]) < 0.005
    assert abs(len(splits["val"]) / total - SPLIT_RATIOS["val"]) < 0.005
    assert abs(len(splits["test"]) / total - SPLIT_RATIOS["test"]) < 0.005


def test_every_split_contains_all_target_classes(splits):
    for name, df in splits.items():
        unique = set(df["label"].unique())
        assert unique == set(TARGET_CLASSES), f"split '{name}' missing classes"


def test_stratification_preserves_proportions(splits):
    """Within ±2 percentage points across splits per class."""
    proportions = {
        name: df["label"].value_counts(normalize=True) for name, df in splits.items()
    }
    for cls in TARGET_CLASSES:
        values = [proportions[s][cls] for s in ("train", "val", "test")]
        assert max(values) - min(values) < 0.02, f"class {cls} drifts: {values}"
