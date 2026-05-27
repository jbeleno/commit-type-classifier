"""Versioned prompt strategies for commit-message generation.

Each strategy is a (system, user_template) pair. We compare four:

    1. zero_shot      — just instructions, no examples
    2. few_shot       — 3 in-context examples sampled from train
    3. chain_of_thought — model reasons first, then emits final message
    4. json_mode      — strict JSON output, parsed downstream

All strategies emit text that *should* start with a Conventional Commit
prefix (``<type>(<scope>): <subject>``) and stay under ~120 chars in the
subject line. Strict adherence is enforced in post-processing, not here.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List

import pandas as pd

from src.config import TARGET_CLASSES

SYSTEM = (
    "You are a senior software engineer writing Conventional Commit messages. "
    "Given a git diff, output ONE single-line commit message in the format: "
    "<type>(<scope>): <subject>. "
    f"<type> must be one of: {', '.join(TARGET_CLASSES)}. "
    "Keep the subject under 72 characters, imperative mood, lower case, no trailing period. "
    "Do not output explanations, only the commit message line."
)

SYSTEM_COT = (
    "You are a senior software engineer writing Conventional Commit messages. "
    "Reason step by step about the diff, then on the LAST line emit the final commit "
    "message in the format: <type>(<scope>): <subject>. "
    f"<type> must be one of: {', '.join(TARGET_CLASSES)}. "
    "The final line MUST start with the type and a colon."
)

SYSTEM_JSON = (
    "You output ONLY a JSON object with keys "
    '{"type": str, "scope": str, "subject": str}. '
    f"type must be one of: {', '.join(TARGET_CLASSES)}. "
    "subject is imperative, lower case, < 72 chars, no trailing period. "
    "scope is a short module name or empty string."
)


@dataclass
class PromptStrategy:
    name: str
    system: str
    build_user: Callable[[str, List[Dict[str, str]]], str]
    fmt: str | None = None  # "json" forces strict JSON output


def _truncate(text: str, limit: int = 3500) -> str:
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


def _zero_shot_user(diff: str, _examples: List[Dict[str, str]]) -> str:
    return f"Diff:\n```diff\n{_truncate(diff)}\n```\n\nCommit message:"


def _few_shot_user(diff: str, examples: List[Dict[str, str]]) -> str:
    parts: List[str] = []
    for ex in examples:
        parts.append(
            f"Diff:\n```diff\n{_truncate(ex['diff'], 1500)}\n```\n"
            f"Commit message: {ex['type']}: {ex['subject']}"
        )
    parts.append(f"Diff:\n```diff\n{_truncate(diff)}\n```\nCommit message:")
    return "\n\n---\n\n".join(parts)


def _cot_user(diff: str, _examples: List[Dict[str, str]]) -> str:
    return (
        f"Diff:\n```diff\n{_truncate(diff)}\n```\n\n"
        "Think step by step:\n"
        "1) What files changed and what kind of change is it (new code, bug fix, docs, tests, structural)?\n"
        "2) Pick the best Conventional Commit type.\n"
        "3) Write the final one-line commit message on the LAST line, starting with the type and a colon."
    )


def _json_user(diff: str, _examples: List[Dict[str, str]]) -> str:
    return (
        f"Diff:\n```diff\n{_truncate(diff)}\n```\n\n"
        "Respond with a single JSON object: "
        '{"type": "...", "scope": "...", "subject": "..."}'
    )


STRATEGIES: Dict[str, PromptStrategy] = {
    "zero_shot": PromptStrategy("zero_shot", SYSTEM, _zero_shot_user),
    "few_shot": PromptStrategy("few_shot", SYSTEM, _few_shot_user),
    "chain_of_thought": PromptStrategy("chain_of_thought", SYSTEM_COT, _cot_user),
    "json_mode": PromptStrategy("json_mode", SYSTEM_JSON, _json_user, fmt="json"),
}


def sample_few_shot_examples(
    train_df: pd.DataFrame,
    n_per_class: int = 1,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """Sample one example per class from the train split for in-context use."""
    rng = random.Random(seed)
    examples: List[Dict[str, str]] = []
    for label in TARGET_CLASSES:
        sub = train_df[train_df["label"] == label]
        if sub.empty:
            continue
        rows = sub.sample(n=min(n_per_class, len(sub)), random_state=seed).to_dict("records")
        for row in rows:
            examples.append(
                {
                    "type": label,
                    "subject": str(row.get("message_clean", "")).splitlines()[0][:72],
                    "diff": str(row.get("diff_text", "")),
                }
            )
    rng.shuffle(examples)
    return examples
