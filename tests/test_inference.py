"""Integration tests for the unified inference layer."""
from __future__ import annotations

import pytest

from src.config import TARGET_CLASSES
from src.inference import AVAILABLE_MODELS, model_is_available, predict


@pytest.fixture(scope="module")
def trained_models() -> list[str]:
    return [m for m in AVAILABLE_MODELS if model_is_available(m)]


def test_at_least_one_model_is_trained(trained_models):
    assert trained_models, (
        "No trained models found. Run "
        "`python -m src.models.<name> all` for at least one model before tests."
    )


def test_predict_returns_known_label(trained_models):
    if not trained_models:
        pytest.skip("no trained model")
    pred = predict("fix login bug", "diff --git a/auth.py b/auth.py\n+ fix", trained_models[0])
    assert pred.label in TARGET_CLASSES
    assert 0.0 <= pred.confidence <= 1.0
    assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-4


def test_predict_unknown_model_raises():
    with pytest.raises(ValueError):
        predict("hello", "", "not_a_real_model")


def test_predict_with_empty_diff_works(trained_models):
    if not trained_models:
        pytest.skip("no trained model")
    pred = predict("add new feature for user profiles", "", trained_models[0])
    assert pred.label in TARGET_CLASSES
