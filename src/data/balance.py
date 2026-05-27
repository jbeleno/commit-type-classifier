"""Class balancing — explicit, dataset-level undersampling.

CommitBench is heavily imbalanced (≈ 62 % `fix`, 11 % `feat`, 11 %
`refactor`, 11 % `test`, 4 % `docs`). The project applies two
canonical balancing strategies, depending on the backend, both
explicitly named in the course rubric (item 4: *"Balanceo de clases
— SMOTE / class weights"*):

1. **Cost-sensitive learning** (data preserved). Used by
   ``baseline_tfidf`` and ``cnn_text``. The loss is multiplied by
   ``n_total / (n_classes · n_samples_c)`` per sample, which is
   mathematically equivalent to oversampling every minority class up
   to ``n_total / n_classes`` rows but without duplicating data.
   Implemented in ``src/utils.py::class_weights_for`` and passed to
   ``LogisticRegression(class_weight="balanced")`` /
   ``model.fit(..., class_weight=cw)``.

2. **Balanced subsampling** (this file). Physically reduces every
   class to the same target row count by random sampling. Used by
   the transformer fine-tuning (`distilbert_model.py`,
   `codebert_model.py`) because passing a class-weight tensor to
   PyTorch's ``nn.CrossEntropyLoss`` on Apple-Silicon MPS proved
   unreliable in this project (occasional hangs during the first
   backward pass) — undersampling sidesteps the issue entirely and
   shortens fine-tuning wall-time by 3–4×.

Reference:
    He, H. and Garcia, E. A. (2009). *Learning from imbalanced data*.
    IEEE Transactions on Knowledge and Data Engineering, 21(9),
    1263-1284. https://doi.org/10.1109/TKDE.2008.239

Run:
    python -m src.data.balance                        # writes data/splits/train_balanced.csv (smallest-class size)
    python -m src.data.balance --target 1600          # 1,600 rows per class (default for distilbert)
    python -m src.data.balance --target 1200          # 1,200 rows per class (default for codebert)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console

from src.config import RANDOM_SEED, SPLITS_DIR, TARGET_CLASSES

console = Console()


def balanced_subsample(
    df: pd.DataFrame,
    target_per_class: int | None = None,
    seed: int = RANDOM_SEED,
    label_col: str = "label",
) -> pd.DataFrame:
    """Return a copy of ``df`` where every class has the same row count.

    Parameters
    ----------
    df : pd.DataFrame
        Input frame with a categorical ``label_col`` column.
    target_per_class : int, optional
        Rows per class in the output. If ``None`` (default), uses the
        size of the smallest class — no duplication, no information
        loss beyond what undersampling the majority classes implies.
    seed : int
        Random seed for reproducibility (defaults to project-wide
        ``RANDOM_SEED = 42``).
    label_col : str
        Column to stratify on. Defaults to ``"label"``.

    Returns
    -------
    pd.DataFrame
        Balanced subsample, row-shuffled, indices reset. Classes that
        already have fewer than ``target_per_class`` rows are kept
        intact (no oversampling is performed; this is strictly
        undersampling).
    """
    sizes = df[label_col].value_counts()
    if target_per_class is None:
        target_per_class = int(sizes.min())

    chunks: list[pd.DataFrame] = []
    for cls in TARGET_CLASSES:
        sub = df[df[label_col] == cls]
        if len(sub) > target_per_class:
            sub = sub.sample(n=target_per_class, random_state=seed)
        chunks.append(sub)

    out = pd.concat(chunks, ignore_index=True)
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(SPLITS_DIR / "train.csv"),
                        help="Input CSV (default: data/splits/train.csv)")
    parser.add_argument("--output", default=str(SPLITS_DIR / "train_balanced.csv"),
                        help="Output CSV (default: data/splits/train_balanced.csv)")
    parser.add_argument("--target", type=int, default=None,
                        help="Rows per class (default: size of the smallest class)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        console.print(f"[red]✗ Missing {in_path}. Run src.data.split first.[/red]")
        return 1

    df = pd.read_csv(in_path)
    before = df["label"].value_counts().sort_index().to_dict()
    console.print(f"[bold]Before:[/bold] {before}  (total = {len(df):,})")

    out = balanced_subsample(df, target_per_class=args.target, seed=args.seed)
    after = out["label"].value_counts().sort_index().to_dict()
    console.print(f"[bold]After:[/bold]  {after}  (total = {len(out):,})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    console.print(f"[green]✓ Saved → {out_path}[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
