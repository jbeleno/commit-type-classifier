"""Unit tests for the preprocessing helpers."""
from __future__ import annotations

import pytest

from src.data.preprocess import (
    CONV_PREFIX_RE,
    diff_to_text_and_features,
    extract_label,
    strip_prefix,
)


@pytest.mark.parametrize(
    "message,expected",
    [
        ("feat: add login", "feat"),
        ("fix(auth): bug", "fix"),
        ("FIX: race condition", "fix"),
        ("docs(README): update", "docs"),
        ("refactor!: rewrite api", "refactor"),
        ("perf: faster sort", "perf"),
        ("style: formatting", "style"),
        ("not a conventional commit", None),
        ("", None),
    ],
)
def test_extract_label(message, expected):
    assert extract_label(message) == expected


def test_extract_label_handles_none():
    assert extract_label(None) is None


def test_strip_prefix_removes_type():
    assert strip_prefix("feat: add login") == "add login"
    assert strip_prefix("fix(auth): handle null") == "handle null"
    assert strip_prefix("docs: README\n\nbody text") == "README\n\nbody text"


def test_strip_prefix_leaves_non_conventional_intact():
    assert strip_prefix("just a sentence") == "just a sentence"


def test_strip_prefix_truncates_long_messages():
    long = "feat: " + ("x" * 1000)
    out = strip_prefix(long)
    assert len(out) <= 500


def test_diff_to_text_and_features_counts_changes():
    diff = (
        "diff --git a/foo.py b/foo.py\n"
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1 +1,2 @@\n"
        "-old line\n"
        "+new line a\n"
        "+new line b\n"
    )
    text, files, added, removed = diff_to_text_and_features(diff)
    assert files == 1
    assert added == 2
    assert removed == 1
    assert "new line" in text


def test_diff_to_text_and_features_handles_list_of_dicts():
    diff = [{"old_path": "a.py", "new_path": "a.py", "diff": "@@ -1 +1 @@\n+new\n-old"}]
    text, files, added, removed = diff_to_text_and_features(diff)
    assert files >= 1
    assert added == 1
    assert removed == 1


def test_diff_to_text_and_features_empty():
    text, files, added, removed = diff_to_text_and_features(None)
    assert (text, files, added, removed) == ("", 0, 0, 0)


def test_conv_prefix_regex_does_not_match_random_colon():
    assert CONV_PREFIX_RE.match("note: just an FYI") is None
    assert CONV_PREFIX_RE.match("feat: add x") is not None
