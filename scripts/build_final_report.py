#!/usr/bin/env python3
"""Assemble FINAL_REPORT.md: narrative from docs/final_report_narrative.md plus tables generated
from artifacts/ (dataset statistics, training details, test metrics) and the most recent
pytest run (artifacts/reports/test_results.txt). No figure is typed by hand."""
import re
from pathlib import Path

import _common  # noqa: F401

from clauseiq import config
from clauseiq.reporting import final_tables as ft

ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    narrative = (ROOT / "docs" / "final_report_narrative.md").read_text()
    tests = config.REPORTS_DIR / "test_results.txt"
    blocks = {
        "{{DATASET}}": ft.dataset_section(),
        "{{TRAINING}}": ft.training_section(),
        "{{COMPARISON}}": ft.comparison_section(),
        "{{PER_CATEGORY}}": ft.per_category_section(),
        "{{TESTS}}": "```\n" + (tests.read_text().strip() if tests.exists() else "pytest not run yet") + "\n```",
    }
    out = narrative
    for k, v in blocks.items():
        out = out.replace(k, v)
    left = re.findall(r"\{\{[A-Z_]+\}\}", out)
    assert not left, f"unfilled placeholders: {left}"
    (ROOT / "FINAL_REPORT.md").write_text(out)
    print(f"wrote FINAL_REPORT.md ({len(out):,} chars)")
