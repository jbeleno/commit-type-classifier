"""LLM-as-classifier mode.

Same input as the TF-IDF baseline (commit message + diff), same output
space (one of {feat, fix, docs, refactor, test}). Used for the
apples-to-apples comparison against the discriminative baseline.

Three strategies:

  * zero_shot — instructions only
  * few_shot  — k random in-context examples from the train split, one per class
  * rag       — top-k retrieved examples from the train split using
                the diff TF-IDF vectorizer (heterogeneous reuse)

Output is parsed with two cascading rules:
  1. JSON-mode → ``{"type": "..."}``
  2. Plain    → first occurrence of ``feat|fix|docs|refactor|test`` in the
               response (case-insensitive, word-boundary).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from src.config import TARGET_CLASSES
from src.llm.ollama_client import generate
from src.llm.prompts import sample_few_shot_examples
from src.llm.rag import retrieve

_TYPE_RE = re.compile(r"\b(feat|fix|docs|refactor|test)\b", re.IGNORECASE)

SYSTEM = (
    "You are a senior software engineer classifying Git commits into one of "
    f"the Conventional Commit types: {', '.join(TARGET_CLASSES)}. "
    "Given the commit message and diff, respond with EXACTLY one of the "
    "five labels — no explanation, no quotes, no punctuation."
)

SYSTEM_JSON = (
    "You classify Git commits into one of: "
    f"{', '.join(TARGET_CLASSES)}. "
    'Respond with a JSON object: {"type": "<one of the labels>"}.'
)


@dataclass
class ClassifyResult:
    model: str
    strategy: str
    predicted: Optional[str]
    raw: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int


def _trim_diff(text: str, limit: int = 2500) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


def _zero_shot(message: str, diff: str) -> str:
    return (
        f"Commit message:\n{message}\n\n"
        f"Diff:\n```diff\n{_trim_diff(diff)}\n```\n\n"
        f"Type:"
    )


def _few_shot(message: str, diff: str, examples: List[Dict[str, str]]) -> str:
    parts: List[str] = []
    for ex in examples:
        parts.append(
            f"Commit message: {ex.get('subject', '')}\n"
            f"Diff:\n```diff\n{_trim_diff(ex.get('diff', ''), 1200)}\n```\n"
            f"Type: {ex.get('type', '')}"
        )
    parts.append(
        f"Commit message: {message}\n"
        f"Diff:\n```diff\n{_trim_diff(diff)}\n```\n"
        f"Type:"
    )
    return "\n\n---\n\n".join(parts)


def _json_user(message: str, diff: str) -> str:
    return (
        f"Commit message:\n{message}\n\n"
        f"Diff:\n```diff\n{_trim_diff(diff)}\n```\n\n"
        'Respond with: {"type": "<one of feat, fix, docs, refactor, test>"}'
    )


def _parse(raw: str, strategy: str) -> Optional[str]:
    if not raw:
        return None
    if strategy == "json_mode":
        try:
            obj = json.loads(raw)
            t = str(obj.get("type", "")).lower().strip()
            return t if t in TARGET_CLASSES else None
        except json.JSONDecodeError:
            m = re.search(r"\{.*?\}", raw, re.DOTALL)
            if m:
                try:
                    obj = json.loads(m.group(0))
                    t = str(obj.get("type", "")).lower().strip()
                    if t in TARGET_CLASSES:
                        return t
                except json.JSONDecodeError:
                    pass
    m = _TYPE_RE.search(raw)
    return m.group(1).lower() if m else None


def classify(
    message: str,
    diff: str,
    *,
    model: str,
    strategy: str = "zero_shot",
    few_shot_examples: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.0,
    seed: int = 42,
    max_tokens: int = 16,
) -> ClassifyResult:
    """Classify a commit with an LLM, given message + diff."""
    if strategy == "zero_shot":
        user = _zero_shot(message, diff)
        system = SYSTEM
        fmt = None
    elif strategy in ("few_shot", "rag"):
        if not few_shot_examples:
            raise ValueError(f"strategy={strategy} requires few_shot_examples")
        user = _few_shot(message, diff, few_shot_examples)
        system = SYSTEM
        fmt = None
    elif strategy == "json_mode":
        user = _json_user(message, diff)
        system = SYSTEM_JSON
        fmt = "json"
        max_tokens = 32
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    gen = generate(
        model=model,
        prompt=user,
        system=system,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        fmt=fmt,
    )
    return ClassifyResult(
        model=model,
        strategy=strategy,
        predicted=_parse(gen.text, strategy),
        raw=gen.text,
        latency_ms=gen.latency_ms,
        prompt_tokens=gen.prompt_tokens,
        completion_tokens=gen.completion_tokens,
    )


def build_few_shot_pool(train_df: pd.DataFrame, n_per_class: int = 2) -> List[Dict[str, str]]:
    return sample_few_shot_examples(train_df, n_per_class=n_per_class)


def build_rag_examples(message: str, diff: str, k: int = 3) -> List[Dict[str, str]]:
    """For 'rag' strategy: retrieve top-k similar commits from the train set."""
    retrieved = retrieve(diff or message, k=k)
    return [{"type": e.type, "subject": e.subject, "diff": e.diff} for e in retrieved]
