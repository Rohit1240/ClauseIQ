#!/usr/bin/env python3
"""Validate the raw CUAD files: structure, 510 contracts, 41 categories, every answer offset,
and contract-level split disjointness. Exits non-zero on failure."""
import json
import sys

import _common  # noqa: F401

from clauseiq.data.prepare import validate_raw

if __name__ == "__main__":
    _common.setup_logging()
    try:
        report = validate_raw()
    except Exception as e:  # noqa: BLE001
        print(f"VALIDATION FAILED: {e}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(report, indent=2))
    print("VALIDATION OK" if report["ok"] else "VALIDATION FAILED")
    sys.exit(0 if report["ok"] else 1)
