"""Central configuration for the C2 commit classifier."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = PROJECT_ROOT / "models_saved"
DB_PATH = PROJECT_ROOT / "db" / "history.sqlite"

TARGET_CLASSES = ["feat", "fix", "docs", "refactor", "test"]
CLASS_TO_IDX = {c: i for i, c in enumerate(TARGET_CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}
NUM_CLASSES = len(TARGET_CLASSES)

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42

MAX_MESSAGE_TOKENS = 64
MAX_DIFF_TOKENS = 384
MAX_TOTAL_TOKENS = 512
