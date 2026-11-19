#!/usr/bin/env python3
"""Parse CUAD, build the contract-level split, compute dataset statistics, then tokenize,
segment and build the training vocabulary. Writes data/processed/*."""
import argparse
import json

import _common  # noqa: F401

from clauseiq.data.prepare import prepare
from clauseiq.training.features import build_feature_cache

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--max-vocab", type=int, default=40000)
    args = ap.parse_args()
    _common.setup_logging()
    out = prepare()
    out["features"] = build_feature_cache(min_freq=args.min_freq, max_size=args.max_vocab)
    print(json.dumps(out, indent=2))
