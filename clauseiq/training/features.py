"""Tokenization / segmentation cache shared by all training pipelines.

Writes to ``data/processed``:
  vocab.json                   vocabulary built from the *training* split only
  tokens_{split}.npz           concatenated token features + labels with per-contract offsets
  segments_{split}.json        segment offsets + gold category sets (TF-IDF model, metrics)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from clauseiq import config
from clauseiq.data import cuad
from clauseiq.text.segment import segment, segment_labels
from clauseiq.text.tokenize import token_labels, tokenize
from clauseiq.text.vocab import Vocab

log = logging.getLogger(__name__)
SPLITS = ("train", "val", "test")


@dataclass
class TokenDoc:
    id: str
    ids: np.ndarray
    shapes: np.ndarray
    rel: np.ndarray
    abs: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    labels: np.ndarray  # (n, C) uint8


def build_feature_cache(min_freq: int = 2, max_size: int = 40000,
                        processed_dir: Path | None = None) -> dict:
    pdir = processed_dir or config.PROCESSED_DIR
    contracts = cuad.load_processed_contracts(pdir)
    cats = cuad.load_processed_categories(pdir)
    split = cuad.load_split(pdir)
    cat_index = {c.key: c.index for c in cats}

    tokenized = {cid: tokenize(c.text) for cid, c in contracts.items()}
    vocab = Vocab.build((tokenized[cid].tokens for cid in split["train"]), min_freq=min_freq,
                        max_size=max_size)
    vocab.save(pdir / "vocab.json")
    summary = {"vocab_size": len(vocab), "min_freq": min_freq}

    for name in SPLITS:
        ids = split[name]
        arrays = {k: [] for k in ("ids", "shapes", "rel", "abs", "starts", "ends", "labels")}
        offsets = [0]
        seg_out = []
        n_unk = n_tok = 0
        for cid in ids:
            c, tk = contracts[cid], tokenized[cid]
            gold = cuad.gold_spans_merged(c)
            enc = vocab.encode(tk.tokens)
            n_unk += int((enc == vocab.unk_id).sum())
            n_tok += len(enc)
            arrays["ids"].append(enc)
            arrays["shapes"].append(tk.shapes)
            arrays["rel"].append(tk.rel_pos)
            arrays["abs"].append(tk.abs_pos)
            arrays["starts"].append(tk.starts)
            arrays["ends"].append(tk.ends)
            arrays["labels"].append(token_labels(tk, gold, cat_index))
            offsets.append(offsets[-1] + len(enc))
            segs = segment(c.text)
            labs = segment_labels(segs, gold)
            seg_out.append({"id": cid, "segments": [[s.start, s.end] for s in segs],
                            "labels": [sorted(l) for l in labs]})
        np.savez_compressed(
            pdir / f"tokens_{name}.npz",
            doc_ids=np.array(ids), offsets=np.array(offsets, dtype=np.int64),
            **{k: np.concatenate(v) for k, v in arrays.items()},
        )
        (pdir / f"segments_{name}.json").write_text(json.dumps(seg_out))
        n_seg = sum(len(d["segments"]) for d in seg_out)
        n_pos_seg = sum(1 for d in seg_out for l in d["labels"] if l)
        lab = np.concatenate(arrays["labels"])
        summary[name] = {
            "contracts": len(ids), "tokens": n_tok, "unk_rate": round(n_unk / max(1, n_tok), 4),
            "segments": n_seg, "segments_with_label": n_pos_seg,
            "token_positive_rate": round(float((lab.sum(1) > 0).mean()), 4),
        }
        log.info("%s: %s", name, summary[name])
    (pdir / "features_summary.json").write_text(json.dumps(summary, indent=2))
    if processed_dir is None:
        from clauseiq.data.prepare import publish_dataset_metadata
        publish_dataset_metadata()
    return summary


def load_token_docs(name: str, processed_dir: Path | None = None) -> list[TokenDoc]:
    pdir = processed_dir or config.PROCESSED_DIR
    with np.load(pdir / f"tokens_{name}.npz") as z:
        arr = {k: z[k] for k in z.files}  # decompress each array exactly once
    off = arr["offsets"]
    out = []
    for i, cid in enumerate(arr["doc_ids"]):
        a, b = off[i], off[i + 1]
        out.append(TokenDoc(str(cid), arr["ids"][a:b], arr["shapes"][a:b], arr["rel"][a:b],
                            arr["abs"][a:b], arr["starts"][a:b], arr["ends"][a:b],
                            arr["labels"][a:b]))
    return out


def load_segments(name: str, processed_dir: Path | None = None) -> list[dict]:
    pdir = processed_dir or config.PROCESSED_DIR
    return json.loads((pdir / f"segments_{name}.json").read_text())
