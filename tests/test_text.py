"""Tokenization, segmentation, vocabulary and label construction."""
import numpy as np

from clauseiq.text.segment import Segment, segment, segment_labels
from clauseiq.text.tokenize import N_ABS_POS, N_REL_POS, N_SHAPES, normalise, token_labels, tokenize
from clauseiq.text.vocab import Vocab


def test_token_offsets_roundtrip(sample_text):
    tk = tokenize(sample_text)
    assert len(tk) > 50
    for tok, s, e in zip(tk.tokens, tk.starts, tk.ends):
        assert normalise(sample_text[s:e]) == tok
    assert np.all(np.diff(tk.starts) > 0)
    assert np.all(tk.ends > tk.starts)


def test_token_features_in_range(sample_text):
    tk = tokenize(sample_text)
    assert tk.shapes.min() >= 1 and tk.shapes.max() < N_SHAPES
    assert tk.rel_pos.min() >= 1 and tk.rel_pos.max() < N_REL_POS
    assert tk.abs_pos.min() >= 1 and tk.abs_pos.max() < N_ABS_POS


def test_normalisation():
    assert normalise("1999") == "0000"
    assert normalise("Agreement") == "agreement"
    assert normalise("party’s") == "party's"


def test_empty_text():
    tk = tokenize("")
    assert len(tk) == 0 and tk.rel_pos.shape == (0,)


def test_segments_cover_text_and_respect_abbreviations(sample_text):
    segs = segment(sample_text)
    texts = [sample_text[s.start:s.end] for s in segs]
    assert texts[0] == "MASTER SUPPLY AGREEMENT"
    # "Acme Corp." and "Beta Inc." must not end a segment; "e.g." neither
    assert any("Acme Corp., a Delaware corporation" in t for t in texts)
    assert any("(e.g. a subsidiary)" in t for t in texts)
    # segments are ordered, non-overlapping and stripped
    for a, b in zip(segs, segs[1:]):
        assert a.end <= b.start
    for t in texts:
        assert t == t.strip() and t
    # every non-space character belongs to a segment
    covered = bytearray(len(sample_text))
    for s in segs:
        covered[s.start:s.end] = b"\x01" * (s.end - s.start)
    assert all(covered[i] or sample_text[i].isspace() for i in range(len(sample_text)))


def test_long_segment_is_split():
    text = "; ".join(["the Supplier shall deliver the products in accordance with the specifications"] * 40) + "."
    segs = segment(text)
    assert len(segs) > 1
    assert max(s.end - s.start for s in segs) <= 900


def test_segment_labels_rule():
    segs = [Segment(0, 100), Segment(101, 200)]
    labels = segment_labels(segs, {"Parties": [(10, 20)], "Governing Law": [(95, 200)]})
    assert labels[0] == {"Parties"}            # short span inside segment 0
    assert labels[1] == {"Governing Law"}      # 5 chars of overlap with seg 0 is not enough


def test_token_labels(sample_text):
    tk = tokenize(sample_text)
    s = sample_text.index("Acme Corp.")
    y = token_labels(tk, {"Parties": [(s, s + len("Acme Corp."))]}, {"Parties": 0, "Other": 1})
    pos = [tk.tokens[i] for i in np.flatnonzero(y[:, 0])]
    assert pos == ["acme", "corp", "."]
    assert y[:, 1].sum() == 0


def test_vocab_build_encode_save_load(tmp_path):
    v = Vocab.build([["a", "b", "a"], ["a", "c"]], min_freq=2)
    assert v.itos[:2] == ["<pad>", "<unk>"] and "a" in v.stoi and "b" not in v.stoi
    enc = v.encode(["a", "zzz"])
    assert enc.tolist() == [v.stoi["a"], v.unk_id]
    p = tmp_path / "vocab.json"
    v.save(p)
    assert Vocab.load(p).itos == v.itos
