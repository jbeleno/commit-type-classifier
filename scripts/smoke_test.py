"""End-to-end smoke test.

Verifies the inference layer can load each trained model and produce a
sensible prediction. Useful before showing the GUI/CLI to a user.

Run:
    python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.table import Table

from src.inference import AVAILABLE_MODELS, model_is_available, predict

console = Console()

SAMPLES = [
    {
        "label_hint": "fix",
        "message": "fix race condition in scheduler when two workers wake up",
        "diff": (
            "diff --git a/scheduler.py b/scheduler.py\n"
            "@@ -10,7 +10,8 @@ class Scheduler:\n"
            "-        if not self.locked:\n"
            "+        with self.lock:\n"
            "+            if self.is_done():\n"
            "                 return None\n"
        ),
    },
    {
        "label_hint": "feat",
        "message": "add OAuth2 login flow with PKCE",
        "diff": (
            "diff --git a/auth.py b/auth.py\n"
            "@@ -0,0 +1,15 @@\n"
            "+def login_oauth(client_id, redirect_uri):\n"
            "+    code_verifier = secrets.token_urlsafe(48)\n"
            "+    challenge = hashlib.sha256(code_verifier.encode()).digest()\n"
        ),
    },
    {
        "label_hint": "docs",
        "message": "update README with installation instructions",
        "diff": (
            "diff --git a/README.md b/README.md\n"
            "@@ -1,3 +1,10 @@\n"
            "+## Installation\n"
            "+\n"
            "+```bash\n"
            "+pip install commit-classifier\n"
            "+```\n"
        ),
    },
    {
        "label_hint": "test",
        "message": "add unit tests for diff parser",
        "diff": (
            "diff --git a/tests/test_diff.py b/tests/test_diff.py\n"
            "@@ -0,0 +1,8 @@\n"
            "+def test_parse_empty_diff():\n"
            "+    assert parse('') == []\n"
        ),
    },
    {
        "label_hint": "refactor",
        "message": "extract URL parsing into helper function",
        "diff": (
            "diff --git a/utils.py b/utils.py\n"
            "@@ -5,7 +5,13 @@ def fetch(url):\n"
            "-    parts = url.split('://')\n"
            "-    scheme = parts[0]\n"
            "-    rest = parts[1]\n"
            "+    scheme, rest = _split_url(url)\n"
        ),
    },
]


def main() -> int:
    trained = [m for m in AVAILABLE_MODELS if model_is_available(m)]
    if not trained:
        console.print("[red]No trained models found.[/red]")
        return 1
    console.print(f"[bold]Trained models:[/bold] {', '.join(trained)}\n")

    for sample in SAMPLES:
        table = Table(title=f"hint={sample['label_hint']!r}: {sample['message']}")
        table.add_column("model")
        table.add_column("predicted")
        table.add_column("confidence", justify="right")
        for model in trained:
            try:
                p = predict(sample["message"], sample["diff"], model)
                table.add_row(model, p.label, f"{p.confidence:.2f}")
            except Exception as e:  # noqa: BLE001
                table.add_row(model, f"[red]ERROR[/red]", str(e))
        console.print(table)
        console.print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
