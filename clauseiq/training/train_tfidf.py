"""Training pipeline for Model 1 (TF-IDF + Logistic Regression)."""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import scipy.sparse as sp

from clauseiq import config
from clauseiq.data import cuad
from clauseiq.decoding import segments_to_spans
from clauseiq.evaluation.metrics import SpanAccumulator
from clauseiq.models.tfidf_lr import TfidfLRModel
from clauseiq.training.features import load_segments
from clauseiq.training.thresholds import tune_thresholds

log = logging.getLogger(__name__)


def _label_matrix(seg_docs: list[dict], cat_index: dict[str, int]) -> np.ndarray:
    rows = [l for d in seg_docs for l in d["labels"]]
    Y = np.zeros((len(rows), len(cat_index)), dtype=np.uint8)
    for i, labels in enumerate(rows):
        for k in labels:
            Y[i, cat_index[k]] = 1
    return Y


def _val_span_f1(model, val_docs, val_probs, contracts, keys, th) -> dict:
    acc = SpanAccumulator(keys)
    for d, P in zip(val_docs, val_probs):
        c = contracts[d["id"]]
        acc.add(c.text, cuad.gold_spans_merged(c), segments_to_spans(P, d["segments"], th, keys))
    return acc.results()["overall"]


def train(C_grid: tuple[float, ...] = (1.0, 4.0), max_features: int = 80000) -> dict:
    config.ensure_dirs()
    t0 = time.time()
    cats = cuad.load_processed_categories()
    keys = [c.key for c in cats]
    cat_index = {k: i for i, k in enumerate(keys)}
    contracts = cuad.load_processed_contracts()
    tr, va = load_segments("train"), load_segments("val")
    tr_texts = [[contracts[d["id"]].text[s:e] for s, e in d["segments"]] for d in tr]
    va_texts = [[contracts[d["id"]].text[s:e] for s, e in d["segments"]] for d in va]
    Y = _label_matrix(tr, cat_index)

    history, best = [], None
    for C in C_grid:
        t_fit = time.time()
        m = TfidfLRModel(C=C, max_features=max_features)
        m.fit_vectorizers(tr_texts)
        X = sp.vstack([m.featurize(t) for t in tr_texts]).tocsr()
        log.info("C=%s: train matrix %s, nnz=%d", C, X.shape, X.nnz)
        m.fit(X, Y, keys, log=log.debug)
        fit_s = time.time() - t_fit
        val_probs = [m.predict_proba_features(m.featurize(t)) for t in va_texts]
        gold = [cuad.gold_spans_merged(contracts[d["id"]]) for d in va]

        def decode(di, j, t, _p=val_probs):
            spans = segments_to_spans(_p[di], va[di]["segments"], np.full(len(keys), t), keys, only={j})
            return [(s, e) for s, e, _ in spans.get(keys[j], [])]

        th, th_detail = tune_thresholds(len(va), keys, gold, decode)
        val = _val_span_f1(m, va, val_probs, contracts, keys, th)
        entry = {"C": C, "fit_seconds": round(fit_s, 1), "val_span_f1": val["span_f1"],
                 "val_span_macro_f1": val["span_macro_f1"], "val_char_iou": val["char_iou"],
                 "global_threshold": th_detail["global_threshold"]}
        history.append(entry)
        log.info("C=%s -> %s", C, entry)
        if best is None or val["span_f1"] > best[0]:
            best = (val["span_f1"], m, th, th_detail, entry)

    _, model, th, th_detail, entry = best
    model.thresholds = th
    train_seconds = time.time() - t0
    cfg = {"model": "tfidf_lr", "C_grid": list(C_grid), "selected_C": entry["C"],
           "max_features": max_features, "ctx_features": 30000, "ngram_range": [1, 3],
           "ctx_ngram_range": [1, 2], "class_weight": "balanced", "solver": "liblinear",
           "unit": "sentence-like segment", "threshold_selection": "per-category on validation span F1"}
    model.save(config.CHECKPOINT_DIR / "tfidf_lr.pkl", extra={"config": cfg})
    out = {"model": "tfidf_lr", "config": cfg, "history": history, "selected": entry,
           "thresholds": th_detail, "train_seconds": round(train_seconds, 1),
           "n_train_segments": int(Y.shape[0]), "n_features": model.n_features}
    (config.METRICS_DIR / "tfidf_lr_training.json").write_text(json.dumps(out, indent=2))
    return out
