"""Central paths and constants. Every path can be overridden with environment variables."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("CLAUSEIQ_ROOT", Path(__file__).resolve().parent.parent))
DATA_DIR = Path(os.environ.get("CLAUSEIQ_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = Path(os.environ.get("CLAUSEIQ_ARTIFACTS_DIR", ROOT / "artifacts"))
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

SEED = int(os.environ.get("CLAUSEIQ_SEED", "13"))
VAL_FRACTION = 0.15  # of the official CUAD training contracts

# Official sources. Zenodo is the canonical release; the GitHub repository ships the
# same SQuAD-format annotations (CUADv1.json + official train/test files).
ZENODO_URL = "https://zenodo.org/records/4595826/files/CUAD_v1.zip?download=1"
GITHUB_DATA_URL = "https://raw.githubusercontent.com/TheAtticusProject/cuad/main/data.zip"
GITHUB_CATEGORIES_URL = (
    "https://raw.githubusercontent.com/TheAtticusProject/cuad/main/category_descriptions.csv"
)
GITHUB_REPO = "https://github.com/TheAtticusProject/cuad.git"

MODEL_NAMES = ["tfidf_lr", "cnn", "bilstm", "bilstm_attn"]
MODEL_DISPLAY = {
    "tfidf_lr": "TF-IDF + Logistic Regression",
    "cnn": "Dilated CNN",
    "bilstm": "BiLSTM",
    "bilstm_attn": "BiLSTM + Attention",
}


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, CHECKPOINT_DIR, METRICS_DIR, REPORTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
