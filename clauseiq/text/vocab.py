"""Vocabulary built only from training-split contracts (no pretrained embeddings)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

PAD, UNK = "<pad>", "<unk>"


class Vocab:
    def __init__(self, itos: list[str]):
        self.itos = itos
        self.stoi = {t: i for i, t in enumerate(itos)}
        self.pad_id = self.stoi[PAD]
        self.unk_id = self.stoi[UNK]

    @classmethod
    def build(cls, token_streams: Iterable[list[str]], min_freq: int = 2,
              max_size: int = 40000) -> "Vocab":
        counts = Counter()
        for toks in token_streams:
            counts.update(toks)
        items = [t for t, c in counts.most_common() if c >= min_freq][: max_size - 2]
        v = cls([PAD, UNK] + items)
        v.counts = counts  # type: ignore[attr-defined]
        return v

    def encode(self, tokens: list[str]) -> np.ndarray:
        unk = self.unk_id
        return np.fromiter((self.stoi.get(t, unk) for t in tokens), dtype=np.int64, count=len(tokens))

    def __len__(self) -> int:
        return len(self.itos)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"itos": self.itos}))

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        return cls(json.loads(Path(path).read_text())["itos"])
