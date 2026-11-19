"""Span decoding and evaluation metrics."""
import numpy as np
import pytest

from clauseiq.decoding import _runs, segments_to_spans, tokens_to_spans
from clauseiq.evaluation.metrics import (ContractAccumulator, SegmentAccumulator, SpanAccumulator,
                                         greedy_match, iou, normalise_bounds)


def test_runs_merge_small_gaps():
    m = np.array([1, 1, 0, 1, 0, 0, 0, 0, 1], dtype=bool)
    assert _runs(m, 0) == [(0, 2), (3, 4), (8, 9)]
    assert _runs(m, 1) == [(0, 4), (8, 9)]
    assert _runs(np.zeros(4, bool), 3) == []


def test_tokens_to_spans_offsets_and_score():
    P = np.array([[0.9, 0.1], [0.8, 0.1], [0.1, 0.1], [0.7, 0.95]])
    starts, ends = np.array([0, 5, 10, 15]), np.array([4, 9, 14, 20])
    sp = tokens_to_spans(P, starts, ends, np.array([0.5, 0.5]), ["A", "B"], max_gap=1)
    assert sp["A"] == [(0, 20, pytest.approx(0.8, abs=1e-4))]
    assert sp["B"] == [(15, 20, 0.95)]
    only = tokens_to_spans(P, starts, ends, np.array([0.5, 0.5]), ["A", "B"], only={1})
    assert set(only) == {"B"}


def test_segments_to_spans_merges_consecutive():
    P = np.array([[0.9], [0.7], [0.1], [0.6]])
    segs = [(0, 10), (11, 20), (21, 30), (31, 40)]
    assert segments_to_spans(P, segs, np.array([0.5]), ["A"])["A"] == [(0, 20, 0.8), (31, 40, 0.6)]


def test_iou_and_matching():
    assert iou((0, 10), (0, 10)) == 1.0
    assert iou((0, 10), (10, 20)) == 0.0
    assert iou((0, 10), (5, 15)) == pytest.approx(5 / 15)
    m = greedy_match([(0, 10), (20, 30)], [(0, 9), (1, 10), (50, 60)], 0.5)
    assert len(m) == 1 and m[0][0] == 0  # one-to-one: second prediction cannot reuse gold 0


def test_normalise_bounds():
    t = '  "Acme Corp.", '
    s, e = normalise_bounds(t, 0, len(t))
    assert t[s:e] == "Acme Corp"


def test_span_accumulator_perfect_and_empty():
    text = "x" * 100
    acc = SpanAccumulator(["A", "B"])
    acc.add(text, {"A": [(0, 10)], "B": [(20, 40)]}, {"A": [(0, 10, 0.9)], "B": [(20, 40, 0.8)]})
    o = acc.results()["overall"]
    assert o["span_f1"] == 1 and o["exact_match_f1"] == 1 and o["char_iou"] == 1 and o["mean_best_iou"] == 1
    acc2 = SpanAccumulator(["A"])
    acc2.add(text, {"A": [(0, 10)]}, {"A": [(50, 60, 0.9)]})
    o2 = acc2.results()["overall"]
    assert o2["span_f1"] == 0 and o2["char_iou"] == 0 and o2["span_precision"] == 0


def test_span_accumulator_partial_overlap():
    text = "y" * 100
    acc = SpanAccumulator(["A"])
    acc.add(text, {"A": [(0, 10)]}, {"A": [(0, 16, 0.9)]})  # IoU 10/16 = 0.625 -> match, not exact
    o = acc.results()["overall"]
    assert o["span_f1"] == 1 and o["exact_match_f1"] == 0
    assert o["char_iou"] == pytest.approx(10 / 16)


def test_segment_accumulator():
    acc = SegmentAccumulator(["A", "B"])
    acc.add([(0, 10), (11, 20), (21, 30)], [["A"], [], ["B"]], {"A": [(0, 10, 0.9)], "B": [(11, 20, 0.7)]})
    r = acc.results()
    assert r["n_segments"] == 3
    assert r["accuracy"] == pytest.approx(1 / 3)
    assert r["per_category"]["A"]["f1"] == 1.0 and r["per_category"]["B"]["f1"] == 0.0
    cm = np.array(r["confusion_matrix"]["matrix"])
    assert cm[0, 0] == 1 and cm[2, 1] == 1 and cm[1, 2] == 1 and cm.sum() == 3


def test_contract_accumulator():
    acc = ContractAccumulator(["A", "B"])
    acc.add({"A": [(0, 1)]}, {"A": [(0, 1, 1.0)], "B": [(3, 4, 0.6)]})
    r = acc.results()
    assert r["confusion"] == {"tp": 1, "fp": 1, "fn": 0, "tn": 0}
    assert r["precision"] == 0.5 and r["recall"] == 1.0


def test_smoothing_and_min_tokens():
    from clauseiq.decoding import smooth_probs
    P = np.array([[0.9], [0.2], [0.9], [0.9], [0.1], [0.1], [0.1], [0.95]], dtype=np.float32)
    S = smooth_probs(P, 1)
    assert S.shape == P.shape and abs(S[1, 0] - (0.9 + 0.2 + 0.9) / 3) < 1e-6
    starts, ends = np.arange(8) * 10, np.arange(8) * 10 + 5
    sp = tokens_to_spans(P, starts, ends, np.array([0.5]), ["A"], max_gap=1, min_tokens=2)
    assert sp["A"] == [(0, 35, pytest.approx(0.9, abs=1e-4))]  # lone last token dropped


def test_decode_hybrid_modes():
    from clauseiq.decoding import decode_hybrid, token_to_segment_probs
    P = np.array([[0.9, 0.6], [0.9, 0.6], [0.1, 0.1], [0.1, 0.7]], dtype=np.float32)
    starts, ends = np.array([0, 6, 12, 18]), np.array([5, 11, 17, 23])
    segs = [(0, 11), (12, 23)]
    SP = token_to_segment_probs(P, starts, segs)
    assert np.allclose(SP, [[0.9, 0.6], [0.1, 0.4]])
    params = {"smooth": 0, "max_gap": 0, "mode": ["token", "segment"], "thresholds": [0.5, 0.5], "min_tokens": [1, 1]}
    out = decode_hybrid(P, starts, ends, segs, ["A", "B"], params)
    assert out["A"] == [(0, 11, pytest.approx(0.9))]
    assert out["B"] == [(0, 11, pytest.approx(0.6))]   # segment 2 mean 0.4 < 0.5
    assert set(decode_hybrid(P, starts, ends, segs, ["A", "B"], params, only={1})) == {"B"}
