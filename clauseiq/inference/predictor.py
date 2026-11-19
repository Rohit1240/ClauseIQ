"""Unified inference interface for the four trained models.

``load_predictor(name)`` returns an object with ``predict(text, categories=None)`` that yields
character-offset spans per category. The same code path is used for test-set evaluation and
by the web backend, so the application shows exactly what was evaluated.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from clauseiq import config
from clauseiq.decoding import PredSpans, decode_hybrid, segments_to_spans, tokens_to_spans
from clauseiq.inference.windows import predict_token_probs, window_starts
from clauseiq.models.neural import build_model
from clauseiq.models.tfidf_lr import TfidfLRModel
from clauseiq.text.segment import segment
from clauseiq.text.tokenize import tokenize
from clauseiq.text.vocab import Vocab

SCORE_DESCRIPTIONS = {
    "segment": "Mean logistic-regression probability of the merged segment(s); thresholds were "
               "tuned per category on the validation split. Not a calibrated likelihood.",
    "token": "Neural score: for token-mode categories, the mean smoothed sigmoid probability of the "
             "above-threshold tokens; for segment-mode categories, the mean token probability over the "
             "sentence segment(s). Compared against a validation-tuned threshold; not a calibrated likelihood.",
}


@dataclass
class Prediction:
    model: str
    spans: PredSpans
    seconds: float
    n_units: int  # segments or tokens processed
    extras: dict = field(default_factory=dict)


class BasePredictor:
    name: str
    kind: str
    cat_keys: list[str]
    thresholds: np.ndarray

    @property
    def display_name(self) -> str:
        return config.MODEL_DISPLAY[self.name]

    @property
    def score_description(self) -> str:
        return SCORE_DESCRIPTIONS[self.kind]

    def _only(self, categories) -> set[int] | None:
        if not categories:
            return None
        wanted = set(categories)
        return {i for i, k in enumerate(self.cat_keys) if k in wanted}

    def predict(self, text: str, categories=None) -> Prediction:  # pragma: no cover - interface
        raise NotImplementedError


class TfidfPredictor(BasePredictor):
    kind = "segment"

    def __init__(self, path: Path):
        self.name = "tfidf_lr"
        self.model = TfidfLRModel.load(path)
        self.cat_keys = self.model.cat_keys
        self.thresholds = np.asarray(self.model.thresholds, dtype=np.float32)
        self.meta = getattr(self.model, "extra", {})

    def predict(self, text: str, categories=None) -> Prediction:
        t0 = time.perf_counter()
        segs = [(s.start, s.end) for s in segment(text)]
        if not segs:
            return Prediction(self.name, {}, time.perf_counter() - t0, 0)
        P = self.model.predict_proba(text, segs)
        spans = segments_to_spans(P, segs, self.thresholds, self.cat_keys, only=self._only(categories))
        return Prediction(self.name, spans, time.perf_counter() - t0, len(segs))


class NeuralPredictor(BasePredictor):
    kind = "token"

    def __init__(self, path: Path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.name = ckpt["arch"]
        hp = dict(ckpt["hparams"])
        arch = hp.pop("arch")
        hp["dilations"] = tuple(hp["dilations"])
        self.model = build_model(arch, **hp)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.vocab = Vocab(ckpt["vocab_itos"])
        self.cat_keys = ckpt["cat_keys"]
        self.thresholds = np.asarray(ckpt["thresholds"], dtype=np.float32)
        self.max_gap = ckpt["max_gap"]
        self.smooth = int(ckpt.get("smooth", 0))
        self.decoding = ckpt.get("decoding") or {
            "smooth": self.smooth, "max_gap": self.max_gap, "mode": ["token"] * len(self.cat_keys),
            "thresholds": list(map(float, self.thresholds)), "min_tokens": [1] * len(self.cat_keys)}
        self.window = ckpt["window"]
        self.meta = {"best_epoch": ckpt.get("best_epoch"), "train_config": ckpt.get("train_config")}

    def token_probs(self, text: str):
        tk = tokenize(text)
        ids = self.vocab.encode(tk.tokens)
        P = predict_token_probs(self.model, ids, tk.shapes, tk.rel_pos, tk.abs_pos, length=self.window,
                                stride=self.window // 2)
        return tk, ids, P

    def predict(self, text: str, categories=None, explain: bool = False) -> Prediction:
        t0 = time.perf_counter()
        tk, ids, P = self.token_probs(text)
        only = self._only(categories)
        needs_segments = any(m == "segment" for j, m in enumerate(self.decoding["mode"]) if only is None or j in only)
        segs = [(s.start, s.end) for s in segment(text)] if needs_segments else []
        spans = decode_hybrid(P, tk.starts, tk.ends, segs, self.cat_keys, self.decoding, only=only)
        extras = {}
        if explain and self.name == "bilstm_attn":
            extras["attention"] = self.attention_focus(text, tk, ids, spans)
        return Prediction(self.name, spans, time.perf_counter() - t0, len(tk), extras)

    @torch.no_grad()
    def attention_focus(self, text, tk, ids, spans: PredSpans, top_k: int = 6) -> dict:
        """For each predicted span, the context tokens that received the most self-attention
        from the span's tokens (averaged over heads and span tokens), within the window that
        best centres the span (words of 3+ letters). Purely descriptive of the model's attention weights."""
        n, L = len(ids), min(self.window, len(ids))
        starts = window_starts(n, L, L // 2)
        out = {}
        for key, items in spans.items():
            res = []
            for s, e, _ in items:
                a = int(np.searchsorted(tk.ends, s, side="right"))
                b = int(np.searchsorted(tk.starts, e, side="left"))
                if b <= a:
                    res.append([])
                    continue
                centre = (a + b) // 2
                ws = min(starts, key=lambda st: abs(st + L // 2 - centre))
                sl = slice(ws, ws + L)
                t = lambda x: torch.from_numpy(np.asarray(x[sl], dtype=np.int64))[None]  # noqa: E731
                _, w = self.model(t(ids), t(tk.shapes), t(tk.rel_pos), t(tk.abs_pos), return_attention=True)
                w = w[0].numpy()  # (L, L)
                qa, qb = max(a, ws) - ws, min(b, ws + L) - ws
                if qb <= qa:
                    res.append([])
                    continue
                focus = w[qa:qb].mean(0)
                order = np.argsort(-focus)
                words, seen = [], set()
                for j in order:
                    tok = text[tk.starts[ws + j]:tk.ends[ws + j]]
                    if len(tok) < 3 or not any(ch.isalpha() for ch in tok) or tok.lower() in seen:
                        continue
                    seen.add(tok.lower())
                    words.append({"token": tok, "weight": round(float(focus[j]), 4),
                                  "start": int(tk.starts[ws + j]), "end": int(tk.ends[ws + j])})
                    if len(words) >= top_k:
                        break
                res.append(words)
            out[key] = res
        return out


def checkpoint_path(name: str, ckpt_dir: Path | None = None) -> Path:
    d = ckpt_dir or config.CHECKPOINT_DIR
    return d / ("tfidf_lr.pkl" if name == "tfidf_lr" else f"{name}.pt")


def load_predictor(name: str, ckpt_dir: Path | None = None) -> BasePredictor:
    path = checkpoint_path(name, ckpt_dir)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint for {name} not found at {path}")
    return TfidfPredictor(path) if name == "tfidf_lr" else NeuralPredictor(path)
