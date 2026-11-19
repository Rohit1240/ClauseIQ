#!/usr/bin/env python3
"""Full reproduction: download → validate → preprocess → train all four models → evaluate on
the test split → comparison report. Each stage can be skipped with a flag."""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(*args: str) -> None:
    cmd = [sys.executable, str(HERE / args[0]), *args[1:]]
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-data", action="store_true", help="data already downloaded and preprocessed")
    ap.add_argument("--time-budget", type=float, default=60.0, help="minutes per neural model")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if not args.skip_data:
        run("download_data.py")
        run("validate_data.py")
        run("preprocess.py")
    for m in ("tfidf_lr", "cnn", "bilstm", "bilstm_attn"):
        run("train.py", "--model", m, "--time-budget", str(args.time_budget), "--threads", str(args.threads))
    run("evaluate.py", "--model", "all")
    run("generate_report.py")
