"""Evaluate a trained model on the untouched CUAD test split (102 contracts)."""
from __future__ import annotations

import json
import logging
import platform
import time

import torch

from clauseiq import config
from clauseiq.data import cuad
from clauseiq.evaluation.metrics import ContractAccumulator, SegmentAccumulator, SpanAccumulator
from clauseiq.inference.predictor import load_predictor
from clauseiq.training.features import load_segments

log = logging.getLogger(__name__)


def evaluate(name: str, split: str = "test", save: bool = True) -> dict:
    config.ensure_dirs()
    pred = load_predictor(name)
    cats = cuad.load_processed_categories()
    keys = [c.key for c in cats]
    assert keys == pred.cat_keys, "category order mismatch between data and checkpoint"
    contracts = cuad.load_processed_contracts()
    seg_docs = load_segments(split)
    span_acc, seg_acc, doc_acc = SpanAccumulator(keys), SegmentAccumulator(keys), ContractAccumulator(keys)
    total_s, total_units, total_chars = 0.0, 0, 0
    per_contract = []
    t0 = time.time()
    for d in seg_docs:
        c = contracts[d["id"]]
        gold = cuad.gold_spans_merged(c)
        p = pred.predict(c.text)
        total_s += p.seconds
        total_units += p.n_units
        total_chars += len(c.text)
        span_acc.add(c.text, gold, p.spans)
        seg_acc.add(d["segments"], d["labels"], p.spans)
        doc_acc.add(gold, p.spans)
        per_contract.append({"id": c.id, "seconds": round(p.seconds, 3), "chars": len(c.text),
                             "n_pred_spans": sum(len(v) for v in p.spans.values()),
                             "n_gold_spans": sum(len(v) for v in gold.values())})
    spans = span_acc.results()
    training = {}
    tpath = config.METRICS_DIR / f"{name}_training.json"
    if tpath.exists():
        training = json.loads(tpath.read_text())
    result = {
        "model": name,
        "display_name": config.MODEL_DISPLAY[name],
        "split": split,
        "n_contracts": len(seg_docs),
        "span_extraction": spans["overall"],
        "span_per_category": spans["per_category"],
        "segment_classification": seg_acc.results(),
        "contract_detection": doc_acc.results(),
        "timing": {
            "train_seconds": training.get("train_seconds"),
            "inference_seconds_total": round(total_s, 2),
            "inference_seconds_per_contract": round(total_s / max(1, len(seg_docs)), 4),
            "chars_per_second": round(total_chars / max(total_s, 1e-9)),
            "units_processed": total_units,
            "wall_seconds": round(time.time() - t0, 1),
            "hardware": f"{platform.processor() or platform.machine()} CPU, torch threads={torch.get_num_threads()}",
        },
        "per_contract": per_contract,
        "thresholds": dict(zip(keys, map(float, pred.thresholds))),
    }
    if save:
        (config.METRICS_DIR / f"{name}_{split}.json").write_text(json.dumps(result, indent=2))
    o = result["span_extraction"]
    log.info("%s on %s: span F1=%.3f EM=%.3f charIoU=%.3f | segment macroF1=%.3f | contract F1=%.3f",
             name, split, o["span_f1"], o["exact_match_f1"], o["char_iou"],
             result["segment_classification"]["macro_f1"], result["contract_detection"]["f1"])
    return result
