"""Parse + validate + split + summarise CUAD and write ``data/processed``."""
from __future__ import annotations

import json
import logging

from clauseiq import config
from clauseiq.data import cuad

log = logging.getLogger(__name__)


def validate_raw() -> dict:
    raw = config.RAW_DIR
    contracts, keys = cuad.load_contracts(raw / "CUADv1.json")
    cats = cuad.align_categories(cuad.load_categories(raw), keys)
    split = cuad.make_split([c.id for c in contracts], raw)
    report = cuad.validate(contracts, cats, split)
    official = cuad.load_official_split(raw)
    report["official_split_available"] = official is not None
    if official:
        report["official_split_sizes"] = {k: len(v) for k, v in official.items()}
    return report


def prepare() -> dict:
    config.ensure_dirs()
    raw = config.RAW_DIR
    contracts, keys = cuad.load_contracts(raw / "CUADv1.json")
    cats = cuad.align_categories(cuad.load_categories(raw), keys)
    split = cuad.make_split([c.id for c in contracts], raw)
    report = cuad.validate(contracts, cats, split)
    stats = cuad.compute_stats(contracts, cats, split)
    cuad.save_processed(contracts, cats, split, stats, report)
    publish_dataset_metadata()
    log.info("Processed %d contracts, %d spans, split %s", len(contracts), stats["n_spans"],
             stats["split"])
    return {"validation": report, "stats_summary": {k: stats[k] for k in (
        "n_contracts", "n_categories", "n_spans", "split", "spans_by_split")}}


def publish_dataset_metadata() -> None:
    """Copy the small dataset metadata files into artifacts/dataset so the web app (and the
    shipped archive) work without the 40 MB raw dataset."""
    import shutil
    dst = config.ARTIFACTS_DIR / "dataset"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("categories.json", "dataset_stats.json", "splits.json", "validation.json",
                 "features_summary.json"):
        src = config.PROCESSED_DIR / name
        if src.exists():
            shutil.copy(src, dst / name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(prepare(), indent=2))
