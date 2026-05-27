"""High-level commit-message generator.

Wraps Ollama + a prompt strategy and returns a normalized
``GeneratedCommit`` (type, scope, subject, raw text, telemetry). This is
the module the GUI and the evaluation harness call.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from src.config import TARGET_CLASSES
from src.llm.ollama_client import Generation, generate
from src.llm.prompts import STRATEGIES, PromptStrategy, sample_few_shot_examples

CONV_LINE_RE = re.compile(
    r"^\s*(?P<type>feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)"
    r"(?:\((?P<scope>[^)]*)\))?(?P<bang>!)?\s*:\s*(?P<subject>.+?)\s*$",
    re.IGNORECASE,
)


@dataclass
class GeneratedCommit:
    model: str
    strategy: str
    raw: str
    parsed_type: Optional[str]
    scope: str
    subject: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    extras: Dict[str, str] = field(default_factory=dict)

    @property
    def one_liner(self) -> str:
        if not self.parsed_type:
            return self.subject or self.raw.splitlines()[0][:120] if self.raw else ""
        scope = f"({self.scope})" if self.scope else ""
        return f"{self.parsed_type}{scope}: {self.subject}".strip()

    @property
    def type_in_target(self) -> bool:
        return self.parsed_type in TARGET_CLASSES


def _parse_conventional_line(text: str) -> tuple[Optional[str], str, str]:
    """Find the first conventional-commit-shaped line in ``text``.

    Returns (type or None, scope, subject). Falls back to (None, "", first_line).
    """
    if not text:
        return None, "", ""
    for line in text.splitlines():
        m = CONV_LINE_RE.match(line.strip())
        if m:
            t = m.group("type").lower()
            return t, (m.group("scope") or "").strip(), m.group("subject").strip()
    first = text.strip().splitlines()[0] if text.strip() else ""
    return None, "", first[:120]


def _parse_json_output(text: str) -> tuple[Optional[str], str, str]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return _parse_conventional_line(text)
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return _parse_conventional_line(text)
    t = str(obj.get("type", "")).lower().strip() or None
    if t not in TARGET_CLASSES:
        t = None
    return t, str(obj.get("scope", "")).strip(), str(obj.get("subject", "")).strip()


def generate_commit_message(
    diff: str,
    *,
    model: str,
    strategy: str = "zero_shot",
    few_shot_examples: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.2,
    max_tokens: int = 192,
    seed: int = 42,
) -> GeneratedCommit:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Options: {list(STRATEGIES)}")
    spec: PromptStrategy = STRATEGIES[strategy]
    user_prompt = spec.build_user(diff, few_shot_examples or [])

    gen: Generation = generate(
        model=model,
        prompt=user_prompt,
        system=spec.system,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        fmt=spec.fmt,
    )

    if strategy == "json_mode":
        t, scope, subject = _parse_json_output(gen.text)
    else:
        t, scope, subject = _parse_conventional_line(gen.text)

    return GeneratedCommit(
        model=model,
        strategy=strategy,
        raw=gen.text,
        parsed_type=t,
        scope=scope,
        subject=subject,
        latency_ms=gen.latency_ms,
        prompt_tokens=gen.prompt_tokens,
        completion_tokens=gen.completion_tokens,
    )


def few_shot_pool(train_df: pd.DataFrame, n_per_class: int = 1) -> List[Dict[str, str]]:
    return sample_few_shot_examples(train_df, n_per_class=n_per_class)
