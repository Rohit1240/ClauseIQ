"""Per-category decision thresholds tuned on the validation split (never on test)."""
from __future__ import annotations

from typing import Callable

import numpy as np

from clauseiq.evaluation.metrics import greedy_match

GRID = np.round(np.concatenate([np.arange(0.10, 0.90, 0.05), [0.9, 0.93, 0.95, 0.97, 0.98, 0.99]]), 2)


def tune_thresholds(n_docs: int, cat_keys: list[str], gold: list[dict],
                    decode: Callable[[int, int, float], list[tuple[int, int]]],
                    grid: np.ndarray = GRID, min_support: int = 8) -> tuple[np.ndarray, dict]:
    """Choose, per category, the threshold maximising span F1 (IoU>=0.5) on validation.

    ``decode(doc_idx, cat_idx, threshold)`` must return the predicted spans of one category.
    Categories with fewer than ``min_support`` validation gold spans use the threshold that
    maximises micro span F1 over all categories (less noisy than tuning on a handful)."""
    C = len(cat_keys)
    tp = np.zeros((C, len(grid)))
    npred = np.zeros((C, len(grid)))
    ngold = np.zeros(C)
    for j, k in enumerate(cat_keys):
        for d in range(n_docs):
            g = gold[d].get(k, [])
            ngold[j] += len(g)
            for ti, t in enumerate(grid):
                p = decode(d, j, float(t))
                npred[j, ti] += len(p)
                if g and p:
                    tp[j, ti] += len(greedy_match(g, p, 0.5))
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(npred > 0, tp / npred, 0)
        rec = np.where(ngold[:, None] > 0, tp / ngold[:, None], 0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0)
        micro_tp, micro_pred = tp.sum(0), npred.sum(0)
        mp = np.where(micro_pred > 0, micro_tp / micro_pred, 0)
        mr = micro_tp / max(1, ngold.sum())
        micro_f1 = np.where(mp + mr > 0, 2 * mp * mr / (mp + mr), 0)
    global_t = float(grid[int(np.argmax(micro_f1))])
    th = np.full(C, global_t, dtype=np.float32)
    detail = {}
    for j, k in enumerate(cat_keys):
        if ngold[j] >= min_support:
            th[j] = float(grid[int(np.argmax(f1[j]))])
        detail[k] = {"threshold": float(th[j]), "val_gold": int(ngold[j]),
                     "val_span_f1_at_threshold": float(f1[j][int(np.argmin(np.abs(grid - th[j])))])}
    return th, {"global_threshold": global_t, "global_val_micro_f1": float(micro_f1.max()),
                "per_category": detail}


def tune_token_decoding(docs, probs: list[np.ndarray], gold: list[dict], cat_keys: list[str],
                        smooth_grid=(0, 2, 4, 8), gap_grid=(1, 3, 6, 12), log=None) -> dict:
    """Jointly choose smoothing radius and gap-merge distance (global) and per-category
    thresholds on the validation split, maximising micro span F1. Test data is never used."""
    from clauseiq.decoding import smooth_probs, tokens_to_spans
    from clauseiq.evaluation.metrics import greedy_match as _gm  # noqa: F401

    best = None
    for sm in smooth_grid:
        sprobs = [smooth_probs(P, sm) for P in probs]
        for gap in gap_grid:
            def decode(di, j, t, _sp=sprobs, _gap=gap):
                d = docs[di]
                spans = tokens_to_spans(_sp[di], d.starts, d.ends, np.full(len(cat_keys), t, dtype=np.float32),
                                        cat_keys, max_gap=_gap, only={j}, smoothed=_sp[di])
                return [(s, e) for s, e, _ in spans.get(cat_keys[j], [])]

            th, detail = tune_thresholds(len(docs), cat_keys, gold, decode)
            # micro F1 with the tuned per-category thresholds
            tp = npred = ngold = 0
            for di, d in enumerate(docs):
                spans = tokens_to_spans(sprobs[di], d.starts, d.ends, th, cat_keys, max_gap=gap, smoothed=sprobs[di])
                for k in cat_keys:
                    g = gold[di].get(k, [])
                    p = [(s, e) for s, e, _ in spans.get(k, [])]
                    ngold += len(g)
                    npred += len(p)
                    if g and p:
                        tp += len(greedy_match(g, p, 0.5))
            pr = tp / npred if npred else 0.0
            rc = tp / ngold if ngold else 0.0
            f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
            if log:
                log(f"decode tuning: smooth={sm} gap={gap} -> val span F1 {f1:.4f} (P {pr:.3f} R {rc:.3f})")
            if best is None or f1 > best["val_span_f1"]:
                best = {"smooth": sm, "max_gap": gap, "thresholds": th, "detail": detail,
                        "val_span_f1": f1, "val_span_precision": pr, "val_span_recall": rc}
    return best


def tune_hybrid_decoding(docs, probs: list[np.ndarray], segments: list[list[tuple[int, int]]],
                         gold: list[dict], cat_keys: list[str], smooth_grid=(0, 2, 4),
                         gap_grid=(3, 6, 12), min_tok_grid=(1, 4, 8, 16, 32), grid: np.ndarray = GRID,
                         min_support: int = 8, log=None) -> dict:
    """Validation-only search, per category, over decoding mode (token / segment), threshold
    and minimum span length, for each global (smoothing radius, gap) pair. Categories with
    fewer than ``min_support`` validation spans use the pooled best configuration."""
    from clauseiq.decoding import _runs, smooth_probs, token_to_segment_probs

    C, G = len(cat_keys), len(grid)
    ngold = np.array([sum(len(g.get(k, [])) for g in gold) for k in cat_keys], dtype=float)

    # segment-mode statistics do not depend on smoothing/gap: compute once
    seg_tp = np.zeros((C, G))
    seg_np = np.zeros((C, G))
    for di, d in enumerate(docs):
        segs = segments[di]
        SP = token_to_segment_probs(probs[di], d.starts, segs)
        for j, k in enumerate(cat_keys):
            g = gold[di].get(k, [])
            for ti, t in enumerate(grid):
                runs = _runs(SP[:, j] >= t, 0)
                pr = [(segs[a][0], segs[b - 1][1]) for a, b in runs]
                seg_np[j, ti] += len(pr)
                if g and pr:
                    seg_tp[j, ti] += len(greedy_match(g, pr, 0.5))

    def f1(tp, npred, ng):
        p = np.where(npred > 0, tp / np.maximum(npred, 1), 0.0)
        r = tp / ng if ng else 0.0
        return np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)

    best = None
    M = len(min_tok_grid)
    for sm in smooth_grid:
        sprobs = [smooth_probs(P, sm) for P in probs]
        for gap in gap_grid:
            tok_tp = np.zeros((C, G, M))
            tok_np = np.zeros((C, G, M))
            for di, d in enumerate(docs):
                S = sprobs[di]
                for j, k in enumerate(cat_keys):
                    g = gold[di].get(k, [])
                    col = S[:, j]
                    for ti, t in enumerate(grid):
                        runs = _runs(col >= t, gap)
                        if not runs:
                            continue
                        for mi, mt in enumerate(min_tok_grid):
                            pr = [(int(d.starts[a]), int(d.ends[b - 1])) for a, b in runs if b - a >= mt]
                            tok_np[j, ti, mi] += len(pr)
                            if g and pr:
                                tok_tp[j, ti, mi] += len(greedy_match(g, pr, 0.5))
            # candidate configs per category: (mode, threshold idx, min_tok idx)
            cfg_tp = np.concatenate([tok_tp.reshape(C, -1), seg_tp], axis=1)
            cfg_np = np.concatenate([tok_np.reshape(C, -1), seg_np], axis=1)
            pooled = f1(cfg_tp.sum(0), cfg_np.sum(0), ngold.sum())
            pooled_choice = int(np.argmax(pooled))
            choice = np.full(C, pooled_choice)
            for j in range(C):
                if ngold[j] >= min_support:
                    choice[j] = int(np.argmax(f1(cfg_tp[j], cfg_np[j], ngold[j])))
            tp = sum(cfg_tp[j, choice[j]] for j in range(C))
            npred = sum(cfg_np[j, choice[j]] for j in range(C))
            pr = tp / npred if npred else 0.0
            rc = tp / ngold.sum()
            micro = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
            if log:
                n_seg = int(sum(c >= G * M for c in choice))
                log(f"hybrid tuning: smooth={sm} gap={gap} -> val span F1 {micro:.4f} (P {pr:.3f} R {rc:.3f}); "
                    f"{n_seg} categories use segment mode")
            if best is None or micro > best["val_span_f1"]:
                modes, ths, mts, detail = [], [], [], {}
                for j, k in enumerate(cat_keys):
                    c = int(choice[j])
                    if c >= G * M:
                        modes.append("segment"); ths.append(float(grid[c - G * M])); mts.append(1)
                    else:
                        modes.append("token"); ths.append(float(grid[c // M])); mts.append(int(min_tok_grid[c % M]))
                    detail[k] = {"mode": modes[-1], "threshold": ths[-1], "min_tokens": mts[-1],
                                 "val_gold": int(ngold[j]), "per_category_tuned": bool(ngold[j] >= min_support),
                                 "val_span_f1": float(f1(cfg_tp[j, c], cfg_np[j, c], ngold[j]))}
                best = {"smooth": sm, "max_gap": gap, "mode": modes, "thresholds": np.array(ths, dtype=np.float32),
                        "min_tokens": mts, "detail": detail, "val_span_f1": micro,
                        "val_span_precision": pr, "val_span_recall": rc}
    return best
