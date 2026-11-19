#!/usr/bin/env python3
"""Run trained models on a contract file (.txt or .pdf) from the command line.

Examples:
  python scripts/infer.py contract.pdf
  python scripts/infer.py contract.txt --model bilstm_attn --category "Governing Law" --json out.json
"""
import argparse
import json
import sys
from pathlib import Path

import _common  # noqa: F401

from clauseiq.config import MODEL_NAMES

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path)
    ap.add_argument("--model", default="all", choices=MODEL_NAMES + ["all"])
    ap.add_argument("--category", action="append", help="CUAD category key (repeatable); default: all 41")
    ap.add_argument("--json", type=Path, help="write results to this JSON file")
    args = ap.parse_args()
    _common.setup_logging()

    from backend.app.services.documents import DocumentError, extract_document, page_of
    from clauseiq.inference.predictor import load_predictor

    try:
        doc = extract_document(args.path.name, args.path.read_bytes(), 5_000_000)
    except DocumentError as e:
        sys.exit(f"error: {e}")
    names = MODEL_NAMES if args.model == "all" else [args.model]
    out = {"file": str(args.path), "pages": len(doc.pages), "chars": len(doc.text), "models": {}}
    for name in names:
        pred = load_predictor(name)
        p = pred.predict(doc.text, categories=args.category)
        rows = [{"category": k, "start": s, "end": e, "page": page_of(doc.pages, s), "score": sc,
                 "text": doc.text[s:e]} for k, spans in p.spans.items() for s, e, sc in spans]
        out["models"][name] = {"seconds": round(p.seconds, 3), "clauses": sorted(rows, key=lambda r: r["start"])}
        print(f"\n== {pred.display_name}: {len(rows)} clauses in {p.seconds:.2f}s")
        for r in sorted(rows, key=lambda r: r["start"]):
            snippet = " ".join(r["text"].split())
            print(f"  [{r['category']}] p.{r['page']} score={r['score']:.2f}  {snippet[:110]}{'…' if len(snippet) > 110 else ''}")
    if args.json:
        args.json.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")
