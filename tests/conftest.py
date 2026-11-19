import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clauseiq import config  # noqa: E402

SAMPLE = (
    "MASTER SUPPLY AGREEMENT\n\n"
    "This Master Supply Agreement (the \"Agreement\") is entered into on January 5, 2020 by and "
    "between Acme Corp., a Delaware corporation (\"Supplier\"), and Beta Inc. (\"Buyer\").\n\n"
    "1. Term. This Agreement shall commence on the Effective Date and continue for five (5) years. "
    "It shall automatically renew for successive one-year terms unless either party gives notice.\n\n"
    "2. Governing Law. This Agreement shall be governed by the laws of the State of New York.\n\n"
    "3. Assignment. Neither party may assign this Agreement without the prior written consent of the "
    "other party; provided, however, that Supplier may assign to an affiliate (e.g. a subsidiary).\n"
)


@pytest.fixture
def sample_text() -> str:
    return SAMPLE


def have_checkpoints() -> bool:
    return all((config.CHECKPOINT_DIR / f).exists() for f in ("tfidf_lr.pkl", "cnn.pt", "bilstm.pt", "bilstm_attn.pt"))


def have_processed() -> bool:
    return (config.PROCESSED_DIR / "contracts.jsonl").exists()
