#!/usr/bin/env python3
"""Train one model or all four.

Examples:
  python scripts/train.py --model tfidf_lr
  python scripts/train.py --model bilstm_attn --epochs 14
  python scripts/train.py --model all
"""
import argparse
import json

import _common  # noqa: F401

from clauseiq.config import MODEL_NAMES

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=MODEL_NAMES + ["all"])
    ap.add_argument("--epochs", type=int, default=None, help="max epochs (neural models)")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--neg-keep", type=float, default=0.25)
    ap.add_argument("--time-budget", type=float, default=60.0, help="minutes per neural model")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    ap.add_argument("--limit-docs", type=int, default=0, help="smoke test on N contracts (no checkpoint written)")
    args = ap.parse_args()
    _common.setup_logging()

    from clauseiq.training import train_neural, train_tfidf

    DEFAULT_EPOCHS = {"cnn": 20, "bilstm": 14, "bilstm_attn": 14}
    DEFAULT_LR = {"cnn": 1.5e-3, "bilstm": 2e-3, "bilstm_attn": 1.5e-3}
    names = MODEL_NAMES if args.model == "all" else [args.model]
    results = {}
    for name in names:
        if name == "tfidf_lr":
            results[name] = train_tfidf.train()
        else:
            cfg = train_neural.TrainConfig(
                arch=name, max_epochs=args.epochs or DEFAULT_EPOCHS[name], patience=args.patience,
                batch_size=args.batch_size, lr=args.lr or DEFAULT_LR[name], neg_keep=args.neg_keep,
                time_budget_min=args.time_budget, threads=args.threads, limit_docs=args.limit_docs)
            results[name] = train_neural.train(cfg)
        r = results[name]
        print(json.dumps({k: r[k] for k in ("model", "train_seconds", "best_epoch", "selected")
                          if k in r}, indent=2))
