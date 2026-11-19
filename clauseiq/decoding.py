"""Turn per-token or per-segment category probabilities into character-offset spans."""
from __future__ import annotations

import numpy as np

PredSpans = dict[str, list[tuple[int, int, float]]]  # category key -> [(start, end, score)]


def _runs(mask: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    """Contiguous True runs (inclusive start, exclusive end), merging runs separated by at
    most ``max_gap`` False positions."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > max_gap + 1)
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]])) + 1
    return list(zip(starts.tolist(), ends.tolist()))


def smooth_probs(probs: np.ndarray, radius: int) -> np.ndarray:
    """Centred moving average over 2*radius+1 tokens (per category). Reduces fragmentation of
    long clauses whose token probabilities fluctuate around the threshold."""
    if radius <= 0 or len(probs) == 0:
        return probs
    k = 2 * radius + 1
    pad = np.pad(probs, ((radius, radius), (0, 0)), mode="edge")
    c = np.cumsum(np.vstack([np.zeros((1, probs.shape[1]), dtype=pad.dtype), pad]), axis=0)
    return ((c[k:] - c[:-k]) / k).astype(probs.dtype)


def tokens_to_spans(probs: np.ndarray, starts: np.ndarray, ends: np.ndarray,
                    thresholds: np.ndarray, cat_keys: list[str], max_gap: int = 3,
                    min_tokens: int = 1, only: set[int] | None = None,
                    smooth: int = 0, smoothed: np.ndarray | None = None) -> PredSpans:
    """Token-level decoding. Probabilities are optionally smoothed (``smooth`` = radius), then
    thresholded per category; runs separated by at most ``max_gap`` tokens are merged.
    Score = mean (smoothed) probability of the above-threshold tokens in the span."""
    out: PredSpans = {}
    if smoothed is None:
        smoothed = smooth_probs(probs, smooth) if smooth else probs
    for j, key in enumerate(cat_keys):
        if only is not None and j not in only:
            continue
        p = smoothed[:, j]
        mask = p >= thresholds[j]
        spans = []
        for a, b in _runs(mask, max_gap):
            if b - a < min_tokens:
                continue
            seg = p[a:b]
            score = float(seg[seg >= thresholds[j]].mean())
            spans.append((int(starts[a]), int(ends[b - 1]), round(score, 4)))
        if spans:
            out[key] = spans
    return out


def segments_to_spans(probs: np.ndarray, segments: list[tuple[int, int]], thresholds: np.ndarray,
                      cat_keys: list[str], doc: str | None = None,
                      only: set[int] | None = None) -> PredSpans:
    """Segment-level decoding: consecutive predicted segments of a category merge into one span.
    Score = mean segment probability."""
    out: PredSpans = {}
    for j, key in enumerate(cat_keys):
        if only is not None and j not in only:
            continue
        mask = probs[:, j] >= thresholds[j]
        spans = []
        for a, b in _runs(mask, 0):
            spans.append((int(segments[a][0]), int(segments[b - 1][1]),
                          round(float(probs[a:b, j].mean()), 4)))
        if spans:
            out[key] = spans
    return out


def token_to_segment_probs(probs: np.ndarray, tok_starts: np.ndarray,
                           segments: list[tuple[int, int]]) -> np.ndarray:
    """Mean token probability inside each segment (segments cover every token)."""
    m, C = len(segments), probs.shape[1]
    out = np.zeros((m, C), dtype=np.float32)
    if m == 0 or len(probs) == 0:
        return out
    seg_starts = np.array([s for s, _ in segments])
    idx = np.clip(np.searchsorted(seg_starts, tok_starts, side="right") - 1, 0, m - 1)
    np.add.at(out, idx, probs)
    counts = np.bincount(idx, minlength=m).astype(np.float32)
    return out / np.maximum(counts, 1)[:, None]


def decode_hybrid(probs: np.ndarray, starts: np.ndarray, ends: np.ndarray,
                  segments: list[tuple[int, int]], cat_keys: list[str], params: dict,
                  only: set[int] | None = None) -> PredSpans:
    """Decode neural token probabilities with validation-tuned per-category settings.

    params = {"smooth": int, "max_gap": int, "mode": ["token"|"segment"] * C,
              "thresholds": [float] * C, "min_tokens": [int] * C}
    * token mode: smoothed token probabilities ≥ threshold, runs merged across ≤ max_gap
      tokens, runs shorter than min_tokens dropped.
    * segment mode: segment score = mean token probability inside a sentence-like segment;
      consecutive segments ≥ threshold are merged into one span.
    """
    th = np.asarray(params["thresholds"], dtype=np.float32)
    modes = params["mode"]
    min_tok = params["min_tokens"]
    tok_idx = {j for j in range(len(cat_keys)) if modes[j] == "token" and (only is None or j in only)}
    seg_idx = {j for j in range(len(cat_keys)) if modes[j] == "segment" and (only is None or j in only)}
    out: PredSpans = {}
    if tok_idx:
        sm = smooth_probs(probs, params["smooth"]) if params["smooth"] else probs
        for j in tok_idx:
            sp = tokens_to_spans(probs, starts, ends, th, cat_keys, max_gap=params["max_gap"],
                                 min_tokens=int(min_tok[j]), only={j}, smoothed=sm)
            out.update(sp)
    if seg_idx and segments:
        sp = segments_to_spans(token_to_segment_probs(probs, starts, segments), segments, th,
                               cat_keys, only=seg_idx)
        out.update(sp)
    return out
