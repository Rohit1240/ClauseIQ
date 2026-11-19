#!/usr/bin/env python3
"""Evaluate trained checkpoints on the untouched test split and save metrics JSON."""
import argparse

import _common  # noqa: F401

from clauseiq.config import MODEL_NAMES

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="all", choices=MODEL_NAMES + ["all"])
    ap.add_argument("--split", default="test", choices=["val", "test"])
    args = ap.parse_args()
    _common.setup_logging()
    from clauseiq.evaluation.evaluate import evaluate

    for name in (MODEL_NAMES if args.model == "all" else [args.model]):
        evaluate(name, args.split)
