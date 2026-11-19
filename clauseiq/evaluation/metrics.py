"""Evaluation metrics for clause extraction.

Three families, reported separately so classification is never confused with extraction:

1. Span extraction (primary): predicted character spans vs merged gold spans, per category.
   * Span P/R/F1 @ IoU>=0.5 — one-to-one greedy matching within (contract, category).
   * Exact Match P/R/F1 — boundaries identical after trimming whitespace/edge punctuation.
   * Mean best IoU — for every gold span, best IoU with a same-category prediction (0 if none).
   * Character-level IoU — |pred chars ∩ gold chars| / |pred chars ∪ gold chars|.
2. Segment classification: every sentence-like segment of a test contract gets a gold label
   set (categories whose gold spans cover it) and a predicted set (same rule applied to the
   predicted spans). Accuracy (exact set match), micro P/R, macro F1, weighted F1, and a
   primary-label confusion matrix.
3. Contract-level clause detection: for each (contract, category) pair, is the clause
   present? Accuracy, P/R/F1 and per-category confusion counts.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from clauseiq.text.segment import Segment, segment_labels

EDGE_PUNCT = " \t\r\n.,;:()[]\"'“”‘’"


def normalise_bounds(text: str, s: int, e: int) -> tuple[int, int]:
    while s < e and text[s] in EDGE_PUNCT:
        s += 1
    while e > s and text[e - 1] in EDGE_PUNCT:
        e -= 1
    return s, e


def iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def _union_length(spans: list[tuple[int, int]]) -> int:
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(spans):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def _intersection_length(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    def merge(xs):
        out = []
        for s, e in sorted(xs):
            if out and s <= out[-1][1]:
                out[-1][1] = max(out[-1][1], e)
            else:
                out.append([s, e])
        return out
    a, b = merge(a), merge(b)
    i = j = total = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def greedy_match(gold: list[tuple[int, int]], pred: list[tuple[int, int]], thr: float) -> list[tuple[int, int, float]]:
    pairs = []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            v = iou(g, p)
            if v >= thr and v > 0:
                pairs.append((v, gi, pi))
    pairs.sort(reverse=True)
    used_g, used_p, out = set(), set(), []
    for v, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        out.append((gi, pi, v))
    return out


def _prf(tp: float, n_pred: float, n_gold: float) -> tuple[float, float, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


class SpanAccumulator:
    def __init__(self, cat_keys: list[str]):
        self.cat_keys = cat_keys
        self.c = {k: defaultdict(float) for k in cat_keys}

    def add(self, text: str, gold: dict[str, list[tuple[int, int]]],
            pred: dict[str, list[tuple[int, int, float]]]) -> None:
        for k in self.cat_keys:
            g = gold.get(k, [])
            p = [(s, e) for s, e, *_ in pred.get(k, [])]
            acc = self.c[k]
            acc["n_gold"] += len(g)
            acc["n_pred"] += len(p)
            if not g and not p:
                continue
            acc["tp_iou50"] += len(greedy_match(g, p, 0.5))
            gn = [normalise_bounds(text, s, e) for s, e in g]
            pn = [normalise_bounds(text, s, e) for s, e in p]
            pset = set(pn)
            gused = set()
            for gi, x in enumerate(gn):
                if x in pset and x not in gused:
                    gused.add(x)
            acc["tp_exact"] += len(gused)
            for x in g:
                acc["sum_best_iou"] += max((iou(x, y) for y in p), default=0.0)
            inter = _intersection_length(g, p)
            acc["char_inter"] += inter
            acc["char_pred"] += _union_length(p)
            acc["char_gold"] += _union_length(g)
            acc["char_union"] += _union_length(g + p)

    def results(self) -> dict:
        per_cat = {}
        tot = defaultdict(float)
        for k in self.cat_keys:
            a = self.c[k]
            for kk, v in a.items():
                tot[kk] += v
            p50, r50, f50 = _prf(a["tp_iou50"], a["n_pred"], a["n_gold"])
            pem, rem, fem = _prf(a["tp_exact"], a["n_pred"], a["n_gold"])
            cp, cr, cf = _prf(a["char_inter"], a["char_pred"], a["char_gold"])
            per_cat[k] = {
                "n_gold": int(a["n_gold"]), "n_pred": int(a["n_pred"]),
                "span_precision": p50, "span_recall": r50, "span_f1": f50,
                "exact_match_precision": pem, "exact_match_recall": rem, "exact_match_f1": fem,
                "mean_best_iou": a["sum_best_iou"] / a["n_gold"] if a["n_gold"] else None,
                "char_iou": a["char_inter"] / a["char_union"] if a["char_union"] else None,
                "char_precision": cp, "char_recall": cr, "char_f1": cf,
            }
        p50, r50, f50 = _prf(tot["tp_iou50"], tot["n_pred"], tot["n_gold"])
        pem, rem, fem = _prf(tot["tp_exact"], tot["n_pred"], tot["n_gold"])
        cp, cr, cf = _prf(tot["char_inter"], tot["char_pred"], tot["char_gold"])
        supported = [k for k in self.cat_keys if self.c[k]["n_gold"] > 0]
        overall = {
            "n_gold": int(tot["n_gold"]), "n_pred": int(tot["n_pred"]),
            "span_precision": p50, "span_recall": r50, "span_f1": f50,
            "span_macro_f1": float(np.mean([per_cat[k]["span_f1"] for k in supported])) if supported else 0.0,
            "exact_match_precision": pem, "exact_match_recall": rem, "exact_match_f1": fem,
            "exact_match_rate": tot["tp_exact"] / tot["n_gold"] if tot["n_gold"] else 0.0,
            "mean_best_iou": tot["sum_best_iou"] / tot["n_gold"] if tot["n_gold"] else 0.0,
            "char_iou": tot["char_inter"] / tot["char_union"] if tot["char_union"] else 0.0,
            "char_iou_macro": float(np.mean([per_cat[k]["char_iou"] or 0.0 for k in supported])) if supported else 0.0,
            "char_precision": cp, "char_recall": cr, "char_f1": cf,
            "categories_with_test_support": len(supported),
        }
        return {"overall": overall, "per_category": per_cat}


class SegmentAccumulator:
    def __init__(self, cat_keys: list[str]):
        self.cat_keys = cat_keys
        self.idx = {k: i for i, k in enumerate(cat_keys)}
        self.y_true: list[np.ndarray] = []
        self.y_pred: list[np.ndarray] = []

    def add(self, segments: list[tuple[int, int]], gold_labels: list[list[str]],
            pred: dict[str, list[tuple[int, int, float]]]) -> None:
        segs = [Segment(s, e) for s, e in segments]
        pred_sets = segment_labels(segs, {k: [(s, e) for s, e, *_ in v] for k, v in pred.items()})
        C = len(self.cat_keys)
        yt = np.zeros((len(segs), C), dtype=np.uint8)
        yp = np.zeros((len(segs), C), dtype=np.uint8)
        for i, (g, p) in enumerate(zip(gold_labels, pred_sets)):
            for k in g:
                yt[i, self.idx[k]] = 1
            for k in p:
                yp[i, self.idx[k]] = 1
        self.y_true.append(yt)
        self.y_pred.append(yp)

    def results(self) -> dict:
        yt, yp = np.concatenate(self.y_true), np.concatenate(self.y_pred)
        support = yt.sum(0)
        labels = [i for i in range(len(self.cat_keys)) if support[i] > 0]
        out = {"n_segments": int(len(yt)), "n_segments_with_gold_label": int((yt.sum(1) > 0).sum())}
        out["accuracy"] = float(accuracy_score(yt, yp))
        pos = yt.sum(1) > 0
        out["accuracy_on_labelled_segments"] = float(accuracy_score(yt[pos], yp[pos])) if pos.any() else 0.0
        for avg in ("micro", "macro", "weighted"):
            p, r, f, _ = precision_recall_fscore_support(yt, yp, labels=labels, average=avg, zero_division=0)
            out[f"{avg}_precision"], out[f"{avg}_recall"], out[f"{avg}_f1"] = float(p), float(r), float(f)
        p, r, f, s = precision_recall_fscore_support(yt, yp, average=None, zero_division=0,
                                                     labels=list(range(len(self.cat_keys))))
        out["per_category"] = {k: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                                   "support": int(s[i])} for i, k in enumerate(self.cat_keys)}
        out["confusion_matrix"] = self._primary_confusion(yt, yp)
        return out

    def _primary_confusion(self, yt: np.ndarray, yp: np.ndarray) -> dict:
        """(C+1)x(C+1) matrix over a single primary label per segment; index C = 'None'.
        If the predicted set shares a category with the gold set, that category is both the
        gold and predicted primary label (a hit); otherwise the lowest-index category of
        each set is used."""
        C = len(self.cat_keys)
        m = np.zeros((C + 1, C + 1), dtype=np.int64)
        for t, p in zip(yt, yp):
            common = np.flatnonzero(t & p)
            if common.size:
                g = q = int(common[0])
            else:
                tn, pn = np.flatnonzero(t), np.flatnonzero(p)
                g = int(tn[0]) if tn.size else C
                q = int(pn[0]) if pn.size else C
            m[g, q] += 1
        return {"labels": self.cat_keys + ["None"], "matrix": m.tolist(),
                "note": "rows = gold primary label, columns = predicted primary label"}


class ContractAccumulator:
    def __init__(self, cat_keys: list[str]):
        self.cat_keys = cat_keys
        self.counts = {k: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for k in cat_keys}

    def add(self, gold: dict, pred: dict) -> None:
        for k in self.cat_keys:
            g, p = bool(gold.get(k)), bool(pred.get(k))
            self.counts[k]["tp" if g and p else "fp" if p else "fn" if g else "tn"] += 1

    def results(self) -> dict:
        per = {}
        T = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        f1s = []
        for k, c in self.counts.items():
            for kk in T:
                T[kk] += c[kk]
            p, r, f = _prf(c["tp"], c["tp"] + c["fp"], c["tp"] + c["fn"])
            n = sum(c.values())
            per[k] = {**c, "precision": p, "recall": r, "f1": f, "accuracy": (c["tp"] + c["tn"]) / n if n else 0}
            if c["tp"] + c["fn"] > 0:
                f1s.append(f)
        p, r, f = _prf(T["tp"], T["tp"] + T["fp"], T["tp"] + T["fn"])
        n = sum(T.values())
        return {"accuracy": (T["tp"] + T["tn"]) / n, "precision": p, "recall": r, "f1": f,
                "macro_f1": float(np.mean(f1s)) if f1s else 0.0, "confusion": T, "per_category": per}
