"""Hybrid commit-message generation: RAG + LLM + classifier verifier.

Pipeline:
    1. RAG: retrieve k=3 most similar commits from the train set (cosine
       over the baseline TF-IDF diff vectorizer).
    2. Generation: LLM with few-shot prompt using the retrieved examples.
    3. Verification: pass the generated subject through the baseline
       classifier to predict its type. If the classifier disagrees with
       the LLM-emitted type AND the classifier's confidence exceeds
       ``confidence_threshold``, we replace the type with the classifier's
       prediction (with a flag in the result).

This is the heterogeneous ensemble for the *generative* side of the
project, mirroring the original soft-voting ensemble on the
discriminative side.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import joblib
import numpy as np
import scipy.sparse as sp

from src.config import IDX_TO_CLASS, TARGET_CLASSES
from src.data.preprocess import diff_to_text_and_features
from src.llm.generator import GeneratedCommit, generate_commit_message
from src.llm.rag import RetrievedExample, retrieve, to_few_shot_format
from src.models.baseline_tfidf import ARTIFACT as BASELINE_PATH, NUMERIC_COLS

_CLASSIFIER_BUNDLE = None


def _classifier():
    global _CLASSIFIER_BUNDLE
    if _CLASSIFIER_BUNDLE is None:
        _CLASSIFIER_BUNDLE = joblib.load(BASELINE_PATH)
    return _CLASSIFIER_BUNDLE


@dataclass
class HybridResult:
    final_message: str
    final_type: str
    llm_type: Optional[str]
    verifier_type: str
    verifier_confidence: float
    type_changed: bool
    retrieved: List[RetrievedExample] = field(default_factory=list)
    raw_llm: str = ""
    latency_ms_total: float = 0.0
    latency_ms_llm: float = 0.0
    model: str = ""
    strategy: str = "hybrid_rag"

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["retrieved"] = [asdict(e) for e in self.retrieved]
        return d


def _classify_message(message: str) -> tuple[str, float]:
    b = _classifier()
    msg_mat = b["vec_msg"].transform([message or ""])
    diff_mat = b["vec_diff"].transform([""])
    num = b["scaler"].transform(np.zeros((1, len(NUMERIC_COLS))))
    X = sp.hstack([msg_mat, diff_mat, sp.csr_matrix(num)]).tocsr()
    proba = b["clf"].predict_proba(X)[0]
    idx = int(np.argmax(proba))
    return IDX_TO_CLASS[idx], float(proba[idx])


def hybrid_generate(
    diff: str,
    *,
    model: str,
    k_retrieval: int = 3,
    confidence_threshold: float = 0.60,
    temperature: float = 0.2,
    seed: int = 42,
) -> HybridResult:
    """Generate a commit message and verify the type with the baseline classifier."""
    import time

    t0 = time.perf_counter()
    examples = retrieve(diff, k=k_retrieval)
    fs = to_few_shot_format(examples)

    gc: GeneratedCommit = generate_commit_message(
        diff=diff,
        model=model,
        strategy="few_shot" if fs else "zero_shot",
        few_shot_examples=fs,
        temperature=temperature,
        seed=seed,
    )

    subject_for_classifier = gc.subject or gc.raw.splitlines()[0] if gc.raw else ""
    verifier_type, verifier_conf = _classify_message(subject_for_classifier)

    llm_type = gc.parsed_type
    type_changed = False
    final_type: str
    if llm_type is None:
        final_type = verifier_type
        type_changed = True
    elif llm_type != verifier_type and verifier_conf >= confidence_threshold:
        final_type = verifier_type
        type_changed = True
    else:
        final_type = llm_type

    if final_type not in TARGET_CLASSES:
        final_type = verifier_type

    scope_part = f"({gc.scope})" if gc.scope else ""
    final_message = f"{final_type}{scope_part}: {gc.subject}".strip()

    return HybridResult(
        final_message=final_message,
        final_type=final_type,
        llm_type=llm_type,
        verifier_type=verifier_type,
        verifier_confidence=verifier_conf,
        type_changed=type_changed,
        retrieved=examples,
        raw_llm=gc.raw,
        latency_ms_total=(time.perf_counter() - t0) * 1000,
        latency_ms_llm=gc.latency_ms,
        model=model,
    )


def hybrid_from_raw_diff(diff: str, *, model: str, **kwargs) -> HybridResult:
    """Convenience: accept a raw unified diff (any shape) and normalize it first."""
    text, *_ = diff_to_text_and_features(diff)
    return hybrid_generate(text, model=model, **kwargs)
