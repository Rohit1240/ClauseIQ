"""Model 1 — TF-IDF + one-vs-rest Logistic Regression over contract segments.

Each sentence-like segment is represented by:
  * word 1–3-gram TF-IDF of the segment itself (sublinear tf),
  * word 1–2-gram TF-IDF of the neighbouring segments (context, scaled by 0.5),
  * one-hot position / length features (document position bucket, first-segments flag,
    heading flag, length bucket).
41 independent L2-regularised logistic regressions (class_weight="balanced") predict
P(category | segment). Extraction = consecutive positive segments merged into one span.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def _meta_features(texts: list[str], n_doc_segs: int) -> sp.csr_matrix:
    rows = []
    for i, t in enumerate(texts):
        f = np.zeros(10 + 4 + 2 + 5, dtype=np.float32)
        f[min(9, (10 * i) // max(1, n_doc_segs))] = 1           # relative position
        f[10 + min(3, i // 5)] = 1                               # first 5 / 10 / 15 / later
        f[14] = 1.0 if t.isupper() and len(t) < 120 else 0.0     # heading-like
        f[15] = 1.0 if i < 3 else 0.0                            # document head
        f[16 + min(4, int(np.log2(1 + len(t) / 25)))] = 1        # length bucket
        rows.append(f)
    return sp.csr_matrix(np.vstack(rows)) if rows else sp.csr_matrix((0, 21), dtype=np.float32)


class TfidfLRModel:
    name = "tfidf_lr"

    def __init__(self, C: float = 4.0, max_features: int = 80000, ctx_features: int = 30000,
                 min_df: int = 2):
        self.C = C
        self.vec = TfidfVectorizer(ngram_range=(1, 3), min_df=min_df, max_features=max_features,
                                   sublinear_tf=True, lowercase=True, dtype=np.float32,
                                   token_pattern=r"(?u)\b\w[\w'’-]*\b")
        self.ctx_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, max_features=ctx_features,
                                       sublinear_tf=True, lowercase=True, dtype=np.float32,
                                       token_pattern=r"(?u)\b\w[\w'’-]*\b")
        self.coef: np.ndarray | None = None       # (n_features, C) float32
        self.intercept: np.ndarray | None = None  # (C,)
        self.cat_keys: list[str] = []
        self.thresholds: np.ndarray | None = None

    # ------------------------------------------------------------------ features
    @staticmethod
    def _contexts(texts: list[str]) -> list[str]:
        return [(texts[i - 1] if i > 0 else "") + " " + (texts[i + 1] if i + 1 < len(texts) else "")
                for i in range(len(texts))]

    def _doc_texts(self, doc: str, segments: list[tuple[int, int]]) -> list[str]:
        return [doc[s:e] for s, e in segments]

    def fit_vectorizers(self, docs: list[list[str]]) -> None:
        flat = [t for d in docs for t in d]
        self.vec.fit(flat)
        self.ctx_vec.fit([c for d in docs for c in self._contexts(d)])

    def featurize(self, seg_texts: list[str]) -> sp.csr_matrix:
        if not seg_texts:
            return sp.csr_matrix((0, self.n_features), dtype=np.float32)
        X = self.vec.transform(seg_texts)
        Xc = self.ctx_vec.transform(self._contexts(seg_texts)) * 0.5
        M = _meta_features(seg_texts, len(seg_texts))
        return sp.hstack([X, Xc, M], format="csr", dtype=np.float32)

    @property
    def n_features(self) -> int:
        return len(self.vec.vocabulary_) + len(self.ctx_vec.vocabulary_) + 21

    # ------------------------------------------------------------------ training
    def fit(self, X: sp.csr_matrix, Y: np.ndarray, cat_keys: list[str], log=None) -> None:
        self.cat_keys = cat_keys
        n_feat, C = X.shape[1], Y.shape[1]
        coef = np.zeros((n_feat, C), dtype=np.float32)
        intercept = np.zeros(C, dtype=np.float32)
        for j in range(C):
            y = Y[:, j]
            if y.sum() == 0:
                intercept[j] = -20.0
                continue
            clf = LogisticRegression(C=self.C, class_weight="balanced", solver="liblinear",
                                     max_iter=200)
            clf.fit(X, y)
            coef[:, j] = clf.coef_[0]
            intercept[j] = clf.intercept_[0]
            if log:
                log(f"  [{j + 1:2d}/{C}] {cat_keys[j]} positives={int(y.sum())}")
        self.coef, self.intercept = coef, intercept

    def predict_proba_features(self, X: sp.csr_matrix) -> np.ndarray:
        z = X @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def predict_proba(self, doc: str, segments: list[tuple[int, int]]) -> np.ndarray:
        return self.predict_proba_features(self.featurize(self._doc_texts(doc, segments)))

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path, extra: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"C": self.C, "vec": self.vec, "ctx_vec": self.ctx_vec,
                         "coef": self.coef, "intercept": self.intercept,
                         "cat_keys": self.cat_keys, "thresholds": self.thresholds,
                         "extra": extra or {}}, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "TfidfLRModel":
        with open(path, "rb") as f:
            d = pickle.load(f)  # noqa: S301 - checkpoints are produced by this project
        m = cls(C=d["C"])
        m.vec, m.ctx_vec = d["vec"], d["ctx_vec"]
        m.coef = np.asarray(d["coef"], dtype=np.float32)
        m.intercept = d["intercept"]
        m.cat_keys, m.thresholds = d["cat_keys"], d["thresholds"]
        m.extra = d.get("extra", {})
        return m
