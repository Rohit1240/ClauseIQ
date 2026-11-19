"""Split a contract into clause-sized segments with exact character offsets.

Segments are sentence-like units: paragraph/line blocks split at sentence boundaries
(abbreviation-aware), with very long sentences split at semicolons and then at whitespace.
They are the classification unit for the TF-IDF model and for segment-level metrics.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

ABBREVIATIONS = {
    "inc", "corp", "co", "ltd", "llc", "l.l.c", "l.p", "lp", "no", "nos", "sec", "secs", "mr", "mrs",
    "ms", "dr", "st", "u.s", "u.k", "e.g", "i.e", "etc", "vs", "v", "art", "para", "approx",
    "dept", "jr", "sr", "n.a", "s.a", "b.v", "n.v", "plc", "pty", "a.g", "g.m.b.h", "gmbh",
    "fig", "ex", "exh", "sch", "cl", "pp", "p", "ch", "par", "mo", "yr", "yrs",
}

BLOCK_RE = re.compile(r"\n\s*\n|\n(?=\s*(?:\(?[0-9]{1,2}(?:\.[0-9]+)*[.)]|\(?[a-zA-Z][.)]|[A-Z][A-Z \-]{3,}$))",
                      re.MULTILINE)
SENT_END_RE = re.compile(r"[.!?;:](?:[\"'”’)\]]*)\s+(?=[\"“(\[]?[A-Z0-9])")
MAX_CHARS = 900
MIN_CHARS = 3


@dataclass
class Segment:
    start: int
    end: int

    def text(self, doc: str) -> str:
        return doc[self.start:self.end]


def _strip(doc: str, s: int, e: int) -> tuple[int, int]:
    while s < e and doc[s].isspace():
        s += 1
    while e > s and doc[e - 1].isspace():
        e -= 1
    return s, e


def _is_abbrev(doc: str, punct_pos: int) -> bool:
    if doc[punct_pos] != ".":
        return False
    j = punct_pos
    while j > 0 and not doc[j - 1].isspace() and doc[j - 1] not in "(\"“":
        j -= 1
    word = doc[j:punct_pos].lower()
    if not word:
        return False
    if word in ABBREVIATIONS:
        return True
    if len(word) == 1 and word.isalpha():  # initials "J." or list items "a."
        return True
    if re.fullmatch(r"[0-9]+(\.[0-9]+)*", word):  # section numbers "1." / "2.3."
        return True
    return False


def _split_long(doc: str, s: int, e: int) -> list[tuple[int, int]]:
    if e - s <= MAX_CHARS:
        return [(s, e)]
    out, cur = [], s
    while e - cur > MAX_CHARS:
        window = doc[cur:cur + MAX_CHARS]
        cut = max(window.rfind("; "), window.rfind(": "))
        if cut < MAX_CHARS // 3:
            cut = window.rfind(", ")
        if cut < MAX_CHARS // 3:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = MAX_CHARS - 1
        out.append((cur, cur + cut + 1))
        cur = cur + cut + 1
    out.append((cur, e))
    return out


def segment(doc: str) -> list[Segment]:
    blocks, last = [], 0
    for m in BLOCK_RE.finditer(doc):
        blocks.append((last, m.start()))
        last = m.end()
    blocks.append((last, len(doc)))

    raw: list[tuple[int, int]] = []
    for bs, be in blocks:
        cur = bs
        for m in SENT_END_RE.finditer(doc, bs, be):
            p = m.start()
            if _is_abbrev(doc, p):
                continue
            raw.append((cur, m.end()))
            cur = m.end()
        raw.append((cur, be))

    segs: list[Segment] = []
    for s, e in raw:
        s, e = _strip(doc, s, e)
        if e - s < 1:
            continue
        for a, b in _split_long(doc, s, e):
            a, b = _strip(doc, a, b)
            if b - a >= 1:
                segs.append(Segment(a, b))
    # merge fragments that are too short to stand alone (e.g. "1." or "(a)") into the next one
    merged: list[Segment] = []
    carry: Segment | None = None
    for sg in segs:
        if carry is not None:
            sg = Segment(carry.start, sg.end)
            carry = None
        if sg.end - sg.start < MIN_CHARS or re.fullmatch(r"\(?[0-9a-zA-Z]{1,3}(\.[0-9]+)*[.)]?", doc[sg.start:sg.end]):
            carry = sg
            continue
        merged.append(sg)
    if carry is not None:
        if merged:
            merged[-1] = Segment(merged[-1].start, carry.end)
        else:
            merged.append(carry)
    return merged


def segment_labels(segs: list[Segment], spans_by_cat: dict[str, list[tuple[int, int]]],
                   doc: str | None = None) -> list[set[str]]:
    """Category set for each segment. A segment gets category c when it overlaps a span of c
    by at least half of the shorter of the two (so short entity spans such as party names
    label their containing segment, and long clauses label every segment they cover)."""
    out: list[set[str]] = [set() for _ in segs]
    starts = [s.start for s in segs]
    import bisect
    for key, spans in spans_by_cat.items():
        for s, e in spans:
            i = max(0, bisect.bisect_right(starts, s) - 1)
            while i < len(segs) and segs[i].start < e:
                sg = segs[i]
                ov = min(sg.end, e) - max(sg.start, s)
                if ov > 0 and ov >= 0.5 * min(sg.end - sg.start, e - s):
                    out[i].add(key)
                i += 1
    return out
