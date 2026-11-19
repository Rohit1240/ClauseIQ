"""Markdown tables for FINAL_REPORT.md, generated from saved JSON so no number is retyped."""
from __future__ import annotations

import json

from clauseiq import config
from clauseiq.reporting.report import load_results


def _f(v, d=3):
    return "—" if v is None else f"{v:.{d}f}"


def _dur(s):
    if s is None:
        return "—"
    m, sec = divmod(int(round(s)), 60)
    return f"{m} min {sec:02d} s" if m else f"{sec} s"


def dataset_section() -> str:
    st = json.loads((config.ARTIFACTS_DIR / "dataset" / "dataset_stats.json").read_text())
    fs = json.loads((config.ARTIFACTS_DIR / "dataset" / "features_summary.json").read_text())
    cats = json.loads((config.ARTIFACTS_DIR / "dataset" / "categories.json").read_text())
    ci = st["class_imbalance"]
    ma = st["missing_annotations"]
    lines = [
        f"| Item | Value |", "|---|---|",
        f"| Contracts | {st['n_contracts']} |",
        f"| Categories | {st['n_categories']} |",
        f"| Annotated clause spans | {st['n_spans']:,} |",
        f"| Split (contracts) | train {st['split']['train']} / val {st['split']['val']} / test {st['split']['test']} — {st['split_source']} |",
        f"| Spans per split | train {st['spans_by_split']['train']:,} / val {st['spans_by_split']['val']:,} / test {st['spans_by_split']['test']:,} |",
        f"| Contract length (chars) | median {st['contract_chars']['median']:,}, mean {st['contract_chars']['mean']:,}, min {st['contract_chars']['min']:,}, max {st['contract_chars']['max']:,} |",
        f"| Total characters | {st['total_chars']:,} |",
        f"| Characters inside a clause | {100 * st['annotated_char_fraction']:.1f}% |",
        f"| (contract, category) pairs without a clause | {ma['pairs_without_clause']:,} of {ma['contract_category_pairs']:,} ({ma['pct_pairs_without_clause']}%) |",
        f"| Class imbalance | max {ci['max_spans']:,} vs min {ci['min_spans']} spans (ratio {ci['imbalance_ratio']}×, Gini {ci['gini']}) |",
        f"| Tokens | train {fs['train']['tokens']:,} / val {fs['val']['tokens']:,} / test {fs['test']['tokens']:,} |",
        f"| Vocabulary (train only, min freq 2) | {fs['vocab_size']:,} (val OOV {100 * fs['val']['unk_rate']:.2f}%, test OOV {100 * fs['test']['unk_rate']:.2f}%) |",
        f"| Tokens inside a clause | train {100 * fs['train']['token_positive_rate']:.1f}% |",
        f"| Segments | train {fs['train']['segments']:,} ({fs['train']['segments_with_label']:,} labelled) / test {fs['test']['segments']:,} ({fs['test']['segments_with_label']:,} labelled) |",
        "", "Per-category annotation counts (all 510 contracts):", "",
        "| Category | Spans | Contracts with clause | Train / Val / Test spans | Median length (chars) |",
        "|---|---:|---:|---|---:|",
    ]
    for c in cats:
        s = st["categories"][c["key"]]
        b = s["spans_by_split"]
        lines.append(f"| {c['name']} | {s['n_spans']:,} | {s['n_contracts_with_clause']} ({s['pct_contracts_with_clause']}%) | "
                     f"{b['train']} / {b['val']} / {b['test']} | {int(s['span_chars']['median']) if s['span_chars'].get('median') is not None else '—'} |")
    return "\n".join(lines)


def training_section() -> str:
    rows = ["| Model | Parameters / features | Epochs run (best) | Decoding (val-tuned) | Val span F1 (tuned) | Training time |",
            "|---|---:|---|---|---:|---:|"]
    for n in config.MODEL_NAMES:
        p = config.METRICS_DIR / f"{n}_training.json"
        if not p.exists():
            rows.append(f"| {config.MODEL_DISPLAY[n]} | not trained | | | | |")
            continue
        t = json.loads(p.read_text())
        if n == "tfidf_lr":
            rows.append(f"| {config.MODEL_DISPLAY[n]} | {t['n_features']:,} features × 41 | C ∈ {t['config']['C_grid']} → C={t['selected']['C']} | "
                        f"segments, per-category thresholds | {_f(t['selected']['val_span_f1'])} | {_dur(t['train_seconds'])} |")
        else:
            d = t.get("decoding", {})
            rows.append(f"| {config.MODEL_DISPLAY[n]} | {t['n_parameters']:,} | {len(t['history'])} ({t['best_epoch']}) | "
                        f"smooth r={d.get('smooth')}, gap={d.get('max_gap')}; per category threshold, min length and mode ({d.get('segment_mode_categories')} of 41 in segment mode) | {_f(t['val_tuned']['span_f1'])} | {_dur(t['train_seconds'])} |")
    return "\n".join(rows)


def comparison_section() -> str:
    res = load_results("test")
    hdr = "| Metric | " + " | ".join(config.MODEL_DISPLAY[n] for n in res) + " |"
    sep = "|---|" + "|".join("---:" for _ in res) + "|"
    spec = [
        ("**Span extraction**", None),
        ("Span precision (IoU ≥ 0.5)", lambda r: r["span_extraction"]["span_precision"]),
        ("Span recall (IoU ≥ 0.5)", lambda r: r["span_extraction"]["span_recall"]),
        ("**Span F1 (IoU ≥ 0.5)**", lambda r: r["span_extraction"]["span_f1"]),
        ("Span macro-F1 (over categories)", lambda r: r["span_extraction"]["span_macro_f1"]),
        ("Exact Match precision", lambda r: r["span_extraction"]["exact_match_precision"]),
        ("Exact Match recall (= EM rate)", lambda r: r["span_extraction"]["exact_match_recall"]),
        ("Exact Match F1", lambda r: r["span_extraction"]["exact_match_f1"]),
        ("Character IoU (micro)", lambda r: r["span_extraction"]["char_iou"]),
        ("Character IoU (macro)", lambda r: r["span_extraction"]["char_iou_macro"]),
        ("Mean best IoU per gold span", lambda r: r["span_extraction"]["mean_best_iou"]),
        ("**Segment classification**", None),
        ("Accuracy (exact label set)", lambda r: r["segment_classification"]["accuracy"]),
        ("Accuracy on gold-labelled segments", lambda r: r["segment_classification"]["accuracy_on_labelled_segments"]),
        ("Precision (micro)", lambda r: r["segment_classification"]["micro_precision"]),
        ("Recall (micro)", lambda r: r["segment_classification"]["micro_recall"]),
        ("Macro F1", lambda r: r["segment_classification"]["macro_f1"]),
        ("Weighted F1", lambda r: r["segment_classification"]["weighted_f1"]),
        ("**Contract-level clause detection**", None),
        ("Accuracy", lambda r: r["contract_detection"]["accuracy"]),
        ("Precision", lambda r: r["contract_detection"]["precision"]),
        ("Recall", lambda r: r["contract_detection"]["recall"]),
        ("F1", lambda r: r["contract_detection"]["f1"]),
        ("Macro F1", lambda r: r["contract_detection"]["macro_f1"]),
        ("**Cost (CPU, 2 threads)**", None),
        ("Training time", "train"),
        ("Inference time per test contract", "inf"),
    ]
    lines = [hdr, sep]
    for label, fn in spec:
        if fn is None:
            lines.append(f"| {label} |" + "|".join(" " for _ in res) + "|")
            continue
        if fn == "train":
            vals = [_dur(r["timing"]["train_seconds"]) for r in res.values()]
        elif fn == "inf":
            vals = [f"{r['timing']['inference_seconds_per_contract']:.3f} s" for r in res.values()]
        else:
            nums = [fn(r) for r in res.values()]
            best = max(nums)
            vals = [f"**{v:.3f}**" if v == best else f"{v:.3f}" for v in nums]
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def per_category_section() -> str:
    res = load_results("test")
    cats = json.loads((config.ARTIFACTS_DIR / "dataset" / "categories.json").read_text())
    names = list(res)
    lines = ["| Category | Test gold spans | " + " | ".join(config.MODEL_DISPLAY[n] for n in names) + " |",
             "|---|---:|" + "|".join("---:" for _ in names) + "|"]
    for c in cats:
        vals = [res[n]["span_per_category"][c["key"]]["span_f1"] for n in names]
        g = res[names[0]]["span_per_category"][c["key"]]["n_gold"]
        best = max(vals)
        lines.append(f"| {c['name']} | {g} | " + " | ".join(f"**{v:.2f}**" if v == best and v > 0 else f"{v:.2f}" for v in vals) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(dataset_section(), "\n\n", training_section(), "\n\n", comparison_section(), "\n\n", per_category_section())
