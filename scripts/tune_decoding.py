#!/usr/bin/env python3
"""Re-tune decoding (smoothing radius, gap merge, per-category thresholds) of a trained
neural checkpoint on the validation split. Training already does this; use the script after
changing the decoding search space."""
import argparse
import json

import _common  # noqa: F401

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=["cnn", "bilstm", "bilstm_attn"])
    args = ap.parse_args()
    _common.setup_logging()
    from clauseiq.training.train_neural import retune_checkpoint

    print(json.dumps(retune_checkpoint(args.model), indent=2))
