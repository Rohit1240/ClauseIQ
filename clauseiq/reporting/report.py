"""Comparison table, charts and markdown evaluation report built from saved metrics JSON.

Nothing here computes model outputs: every figure is read from artifacts/metrics/*.json,
which are written by the training and evaluation pipelines.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

from clauseiq import config  # noqa: E402
from clauseiq.data import cuad  # noqa: E402

log = logging.getLogger(__name__)

# Fixed categorical slots (validated reference palette, light mode); colour follows the model.
MODEL_COLORS = {"tfidf_lr": "#2a78d6", "cnn": "#eb6834", "bilstm": "#1baf7a", "bilstm_attn": "#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("seqblue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

COLUMNS = [
    ("Span F1 (IoU≥0.5)", lambda r: r["span_extraction"]["span_f1"]),
    ("Span P", lambda r: r["span_extraction"]["span_precision"]),
    ("Span R", lambda r: r["span_extraction"]["span_recall"]),
    ("Span macro-F1", lambda r: r["span_extraction"]["span_macro_f1"]),
    ("Exact Match F1", lambda r: r["span_extraction"]["exact_match_f1"]),
    ("Exact Match rate", lambda r: r["span_extraction"]["exact_match_rate"]),
    ("Char IoU", lambda r: r["span_extraction"]["char_iou"]),
    ("Mean best IoU", lambda r: r["span_extraction"]["mean_best_iou"]),
    ("Seg. accuracy", lambda r: r["segment_classification"]["accuracy"]),
    ("Seg. micro-P", lambda r: r["segment_classification"]["micro_precision"]),
    ("Seg. micro-R", lambda r: r["segment_classification"]["micro_recall"]),
    ("Seg. macro-F1", lambda r: r["segment_classification"]["macro_f1"]),
    ("Seg. weighted-F1", lambda r: r["segment_classification"]["weighted_f1"]),
    ("Contract acc.", lambda r: r["contract_detection"]["accuracy"]),
    ("Contract F1", lambda r: r["contract_detection"]["f1"]),
    ("Train time (min)", lambda r: (r["timing"]["train_seconds"] or 0) / 60),
    ("Inference s/contract", lambda r: r["timing"]["inference_seconds_per_contract"]),
]


def load_results(split: str = "test") -> dict[str, dict]:
    out = {}
    for name in config.MODEL_NAMES:
        p = config.METRICS_DIR / f"{name}_{split}.json"
        if p.exists():
            out[name] = json.loads(p.read_text())
    return out


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _grouped_bars(results, metrics, title, path, ylim=(0, 1)):
    names = list(results)
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
    n = len(names)
    width = 0.8 / n
    x = np.arange(len(metrics))
    for i, name in enumerate(names):
        vals = [fn(results[name]) for _, fn in metrics]
        bars = ax.bar(x + (i - (n - 1) / 2) * width, vals, width * 0.92, color=MODEL_COLORS[name],
                      label=config.MODEL_DISPLAY[name], edgecolor="white", linewidth=1)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=6.5, color=INK2)
    ax.set_xticks(x, [m for m, _ in metrics], fontsize=9, color=INK)
    ax.set_ylim(*ylim)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    _style(ax)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.1))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_figures(results: dict[str, dict], out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    span_metrics = [(n, f) for n, f in COLUMNS if n in (
        "Span F1 (IoU≥0.5)", "Span macro-F1", "Exact Match F1", "Char IoU", "Mean best IoU")]
    _grouped_bars(results, span_metrics, "Span extraction on the CUAD test split (102 contracts)",
                  out_dir / "span_extraction_comparison.png")
    files.append("span_extraction_comparison.png")
    cls_metrics = [(n, f) for n, f in COLUMNS if n in (
        "Seg. micro-P", "Seg. micro-R", "Seg. macro-F1", "Seg. weighted-F1", "Contract acc.", "Contract F1")]
    _grouped_bars(results, cls_metrics, "Classification: segment labels and contract-level clause detection",
                  out_dir / "classification_comparison.png")
    files.append("classification_comparison.png")

    # per-category span F1 heatmap
    cats = cuad.load_processed_categories()
    names = list(results)
    M = np.array([[results[n]["span_per_category"][c.key]["span_f1"] for n in names] for c in cats])
    support = [results[names[0]]["span_per_category"][c.key]["n_gold"] for c in cats]
    fig, ax = plt.subplots(figsize=(7.5, 13), dpi=150)
    im = ax.imshow(M, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(names)), [config.MODEL_DISPLAY[n] for n in names], rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(cats)), [f"{c.name} (n={s})" for c, s in zip(cats, support)], fontsize=7.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if M[i, j] > 0.55 else INK)
    ax.set_title("Span F1 (IoU≥0.5) by category, test split", loc="left", fontsize=10, color=INK)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_dir / "per_category_span_f1.png", bbox_inches="tight")
    plt.close(fig)
    files.append("per_category_span_f1.png")

    # training curves (neural models)
    curves = {}
    for n in names:
        p = config.METRICS_DIR / f"{n}_training.json"
        if p.exists() and n != "tfidf_lr":
            curves[n] = json.loads(p.read_text())["history"]
    if curves:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=150)
        for n, h in curves.items():
            ep = [e["epoch"] for e in h]
            axes[0].plot(ep, [e["train_loss"] for e in h], color=MODEL_COLORS[n], lw=2, marker="o", ms=4,
                         label=config.MODEL_DISPLAY[n])
            axes[1].plot(ep, [e["val_span_f1"] for e in h], color=MODEL_COLORS[n], lw=2, marker="o", ms=4,
                         label=config.MODEL_DISPLAY[n])
        axes[0].set_title("Training loss (masked weighted BCE)", loc="left", fontsize=10, color=INK)
        axes[1].set_title("Validation span F1 (best global threshold)", loc="left", fontsize=10, color=INK)
        for ax in axes:
            _style(ax)
            ax.set_xlabel("epoch", fontsize=9, color=INK2)
        axes[1].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "training_curves.png", bbox_inches="tight")
        plt.close(fig)
        files.append("training_curves.png")

    # timing
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), dpi=150)
    labels = [config.MODEL_DISPLAY[n] for n in names]
    for ax, vals, title in (
        (axes[0], [(results[n]["timing"]["train_seconds"] or 0) / 60 for n in names], "Training time (minutes, CPU)"),
        (axes[1], [results[n]["timing"]["inference_seconds_per_contract"] for n in names], "Inference time (seconds / contract, CPU)"),
    ):
        bars = ax.barh(labels, vals, color=[MODEL_COLORS[n] for n in names], height=0.6)
        for b, v in zip(bars, vals):
            ax.text(v, b.get_y() + b.get_height() / 2, f" {v:.2f}", va="center", fontsize=8, color=INK2)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        _style(ax)
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, color=GRID)
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_dir / "timing_comparison.png", bbox_inches="tight")
    plt.close(fig)
    files.append("timing_comparison.png")

    # confusion matrices (segment primary label)
    short = [c.name[:22] for c in cats] + ["None"]
    for n in names:
        cm = np.array(results[n]["segment_classification"]["confusion_matrix"]["matrix"], dtype=float)
        fig, ax = plt.subplots(figsize=(11, 10), dpi=130)
        im = ax.imshow(cm + 1, cmap=SEQ, norm=LogNorm(vmin=1, vmax=cm.max() + 1))
        ax.set_xticks(range(len(short)), short, rotation=90, fontsize=6)
        ax.set_yticks(range(len(short)), short, fontsize=6)
        ax.set_xlabel("predicted primary label", fontsize=8, color=INK2)
        ax.set_ylabel("gold primary label", fontsize=8, color=INK2)
        ax.set_title(f"{config.MODEL_DISPLAY[n]} — segment confusion matrix (log colour scale)",
                     loc="left", fontsize=10, color=INK)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="segments + 1")
        fig.tight_layout()
        fig.savefig(out_dir / f"confusion_{n}.png", bbox_inches="tight")
        plt.close(fig)
        files.append(f"confusion_{n}.png")
    return files


def comparison_table(results: dict[str, dict]) -> tuple[str, list[dict]]:
    rows = []
    for n, r in results.items():
        row = {"model": config.MODEL_DISPLAY[n], "key": n}
        for col, fn in COLUMNS:
            row[col] = round(float(fn(r)), 4)
        rows.append(row)
    header = "| Model | " + " | ".join(c for c, _ in COLUMNS) + " |"
    sep = "|---|" + "|".join("---:" for _ in COLUMNS) + "|"
    lines = [header, sep]
    for row in rows:
        lines.append(f"| {row['model']} | " + " | ".join(
            f"{row[c]:.2f}" if "time" in c.lower() or "s/contract" in c else f"{row[c]:.3f}" for c, _ in COLUMNS) + " |")
    return "\n".join(lines), rows


def per_category_table(results: dict[str, dict]) -> str:
    cats = cuad.load_processed_categories()
    names = list(results)
    head = "| Category | Test gold spans | " + " | ".join(f"{config.MODEL_DISPLAY[n]} span F1" for n in names) + " |"
    lines = [head, "|---|---:|" + "|".join("---:" for _ in names) + "|"]
    for c in cats:
        pc = [results[n]["span_per_category"][c.key] for n in names]
        lines.append(f"| {c.name} | {pc[0]['n_gold']} | " + " | ".join(f"{p['span_f1']:.3f}" for p in pc) + " |")
    return "\n".join(lines)


def generate(split: str = "test") -> dict:
    config.ensure_dirs()
    results = load_results(split)
    if not results:
        raise RuntimeError("no evaluation results found; run scripts/evaluate.py first")
    figs = make_figures(results, config.FIGURES_DIR)
    table_md, rows = comparison_table(results)
    (config.METRICS_DIR / "comparison.json").write_text(json.dumps(rows, indent=2))
    with open(config.METRICS_DIR / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    stats = json.loads((config.PROCESSED_DIR / "dataset_stats.json").read_text())
    md = [
        "# ClauseIQ — evaluation report", "",
        f"Generated from `artifacts/metrics/*_{split}.json`. Split: **{split}** "
        f"({next(iter(results.values()))['n_contracts']} contracts, contract-level, {stats['split_source']}).", "",
        "## Model comparison", "", table_md, "",
        "*Span* metrics measure extraction of character spans; *Seg.* metrics are multi-label "
        "classification of sentence-like segments; *Contract* metrics are clause presence per "
        "(contract, category). See the Documentation page / README for definitions.", "",
        "![span](figures/span_extraction_comparison.png)", "",
        "![classification](figures/classification_comparison.png)", "",
        "![timing](figures/timing_comparison.png)", "",
        "![curves](figures/training_curves.png)", "",
        "## Per-category span F1 (IoU ≥ 0.5)", "", per_category_table(results), "",
        "![per-category](figures/per_category_span_f1.png)", "",
        "## Confusion matrices", "",
        *[f"![{n}](figures/confusion_{n}.png)" for n in results], "",
    ]
    (config.REPORTS_DIR / "evaluation_report.md").write_text("\n".join(md))
    log.info("report written with %d figures", len(figs))
    return {"figures": figs, "table": table_md}
