#!/usr/bin/env python3
"""Build the comparison table (CSV/JSON/Markdown), charts and evaluation_report.md from
saved metrics. Also available as scripts/compare.py."""
import _common  # noqa: F401

if __name__ == "__main__":
    _common.setup_logging()
    from clauseiq.reporting.report import generate

    print(generate()["table"])
