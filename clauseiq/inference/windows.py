"""Sliding-window inference for the neural taggers over arbitrarily long contracts.

The document is cut into overlapping windows (default 256 tokens, stride 128). Each token's
probability is a weighted average over every window containing it; weights favour the window
centre, where the model sees context on both sides.
"""
from __future__ import annotations

import numpy as np
import torch


def window_starts(n: int, length: int, stride: int) -> list[int]:
    if n <= length:
        return [0]
    starts = list(range(0, n - length + 1, stride))
    if starts[-1] + length < n:
        starts.append(n - length)
    return starts


def _centre_weights(length: int) -> np.ndarray:
    x = np.arange(length) + 0.5
    return (0.1 + np.minimum(x, length - x) / (length / 2)).astype(np.float32)


@torch.no_grad()
def predict_token_probs(model: torch.nn.Module, ids: np.ndarray, shapes: np.ndarray, rel: np.ndarray,
                        absp: np.ndarray, length: int = 256, stride: int = 128,
                        batch_size: int = 32) -> np.ndarray:
    model.eval()
    n = len(ids)
    n_labels = model.head.out_features
    if n == 0:
        return np.zeros((0, n_labels), dtype=np.float32)
    L = min(length, n)
    starts = window_starts(n, L, stride)
    acc = np.zeros((n, n_labels), dtype=np.float32)
    wsum = np.zeros((n, 1), dtype=np.float32)
    w = _centre_weights(L)[:, None]
    for b in range(0, len(starts), batch_size):
        chunk = starts[b:b + batch_size]
        sl = [slice(s, s + L) for s in chunk]
        t = lambda a: torch.from_numpy(np.stack([a[s] for s in sl]).astype(np.int64))  # noqa: E731
        probs = torch.sigmoid(model(t(ids), t(shapes), t(rel), t(absp))).numpy()
        for s, p in zip(chunk, probs):
            acc[s:s + L] += p * w
            wsum[s:s + L] += w
    return acc / np.maximum(wsum, 1e-6)
