#!/usr/bin/env python3
"""Print the four-model comparison table and write comparison.csv/json + figures."""
import _common  # noqa: F401

if __name__ == "__main__":
    _common.setup_logging()
    from clauseiq.reporting.report import generate

    print(generate()["table"])
