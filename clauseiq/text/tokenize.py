"""Offset-preserving word tokenizer and token features.

Every token keeps its exact (start, end) character offsets in the original text, so model
predictions can be mapped back to the source document for highlighting.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*|\d+(?:[.,:/-]\d+)*|[^\sA-Za-z\d]")

# shape ids (0 = padding)
SHAPE_LOWER, SHAPE_TITLE, SHAPE_UPPER, SHAPE_MIXED, SHAPE_DIGIT, SHAPE_PUNCT = 1, 2, 3, 4, 5, 6
N_BASE_SHAPES = 6
N_SHAPES = 1 + N_BASE_SHAPES * 3       # x {no newline, one newline, blank line before}
N_REL_POS = 21                         # 20 relative-position buckets + padding
N_ABS_POS = 18                         # log2 buckets of absolute token index + padding


@dataclass
class Tokenized:
    tokens: list[str]          # normalised token strings (vocabulary lookup keys)
    starts: np.ndarray         # int64 char start offsets
    ends: np.ndarray           # int64 char end offsets (exclusive)
    shapes: np.ndarray         # int64 shape feature ids
    rel_pos: np.ndarray        # int64 relative position bucket ids
    abs_pos: np.ndarray        # int64 absolute position bucket ids

    def __len__(self) -> int:
        return len(self.tokens)


def normalise(tok: str) -> str:
    if tok[0].isdigit():
        # Collapse numbers to a length-preserving digit shape: "1999" -> "0000", "2.5" -> "0.0"
        return re.sub(r"\d", "0", tok)[:8]
    return tok.lower().replace("’", "'")


def _shape(tok: str) -> int:
    if tok[0].isdigit():
        return SHAPE_DIGIT
    if not tok[0].isalpha():
        return SHAPE_PUNCT
    if tok.islower():
        return SHAPE_LOWER
    if tok.isupper():
        return SHAPE_UPPER if len(tok) > 1 else SHAPE_TITLE
    if tok[0].isupper() and tok[1:].islower():
        return SHAPE_TITLE
    return SHAPE_MIXED


def tokenize(text: str) -> Tokenized:
    toks, starts, ends, shapes = [], [], [], []
    prev_end = 0
    for m in TOKEN_RE.finditer(text):
        s, e = m.span()
        raw = m.group()
        gap = text[prev_end:s]
        nl = gap.count("\n")
        nl_bucket = 0 if nl == 0 else (1 if nl == 1 else 2)
        toks.append(normalise(raw))
        starts.append(s)
        ends.append(e)
        shapes.append(_shape(raw) + N_BASE_SHAPES * nl_bucket)
        prev_end = e
    n = len(toks)
    idx = np.arange(n)
    rel = (1 + np.minimum(19, (20 * idx) // max(1, n))).astype(np.int64)
    absb = (1 + np.minimum(N_ABS_POS - 2, np.floor(np.log2(idx + 1)).astype(np.int64))).astype(np.int64)
    return Tokenized(
        tokens=toks,
        starts=np.asarray(starts, dtype=np.int64),
        ends=np.asarray(ends, dtype=np.int64),
        shapes=np.asarray(shapes, dtype=np.int64),
        rel_pos=rel,
        abs_pos=absb,
    )


def token_labels(tok: Tokenized, spans_by_cat: dict[str, list[tuple[int, int]]],
                 cat_index: dict[str, int]) -> np.ndarray:
    """Multi-label token targets as an (n_tokens, n_categories) uint8 matrix. A token is
    positive for category c when its characters overlap a gold span of c."""
    y = np.zeros((len(tok), len(cat_index)), dtype=np.uint8)
    if len(tok) == 0:
        return y
    for key, spans in spans_by_cat.items():
        j = cat_index[key]
        for s, e in spans:
            lo = np.searchsorted(tok.ends, s, side="right")
            hi = np.searchsorted(tok.starts, e, side="left")
            if hi > lo:
                y[lo:hi, j] = 1
    return y


def log_bucket(x: int) -> int:
    return int(math.log2(x + 1))
