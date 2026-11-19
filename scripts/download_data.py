#!/usr/bin/env python3
"""Download CUAD v1 (Zenodo, falling back to the official GitHub repository)."""
import argparse
import json

import _common  # noqa: F401

from clauseiq.data.download import download_cuad

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-download even if files exist")
    args = ap.parse_args()
    _common.setup_logging()
    print(json.dumps(download_cuad(force=args.force), indent=2))
