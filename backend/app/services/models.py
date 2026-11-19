"""Model registry: loads the four trained checkpoints once and runs analyses."""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

from clauseiq import config as ml_config
from clauseiq.inference.predictor import BasePredictor, checkpoint_path, load_predictor

from .documents import page_of

log = logging.getLogger(__name__)


class ModelUnavailable(RuntimeError):
    pass


class Catalog:
    """Read-only dataset + evaluation metadata from the artifacts directory."""

    def __init__(self, artifacts: Path):
        self.artifacts = artifacts
        ds = artifacts / "dataset"
        self.categories = json.loads((ds / "categories.json").read_text())
        self.stats = json.loads((ds / "dataset_stats.json").read_text())
        self.features = json.loads((ds / "features_summary.json").read_text()) if (ds / "features_summary.json").exists() else {}
        self.cat_by_key = {c["key"]: c for c in self.categories}
        self.metrics: dict[str, dict] = {}
        self.training: dict[str, dict] = {}
        self.reload_metrics()

    def reload_metrics(self) -> None:
        md = self.artifacts / "metrics"
        for name in ml_config.MODEL_NAMES:
            p = md / f"{name}_test.json"
            if p.exists():
                m = json.loads(p.read_text())
                m.pop("per_contract", None)
                self.metrics[name] = m
            t = md / f"{name}_training.json"
            if t.exists():
                self.training[name] = json.loads(t.read_text())


class Registry:
    def __init__(self, artifacts: Path, max_concurrent: int = 2):
        self.ckpt_dir = artifacts / "checkpoints"
        self._models: dict[str, BasePredictor] = {}
        self._errors: dict[str, str] = {}
        self._lock = threading.Lock()
        self._sem = threading.Semaphore(max_concurrent)

    def available(self) -> list[str]:
        return [n for n in ml_config.MODEL_NAMES if checkpoint_path(n, self.ckpt_dir).exists()]

    def get(self, name: str) -> BasePredictor:
        if name not in ml_config.MODEL_NAMES:
            raise ModelUnavailable(f"Unknown model '{name}'.")
        with self._lock:
            if name not in self._models:
                try:
                    t0 = time.time()
                    self._models[name] = load_predictor(name, self.ckpt_dir)
                    self._errors.pop(name, None)
                    log.info("loaded %s in %.1fs", name, time.time() - t0)
                except FileNotFoundError as e:
                    self._errors[name] = str(e)
                    raise ModelUnavailable(f"Model '{name}' has no trained checkpoint. Train it with "
                                           f"`python scripts/train.py --model {name}`.") from e
                except Exception as e:  # noqa: BLE001
                    self._errors[name] = repr(e)
                    log.exception("failed to load %s", name)
                    raise ModelUnavailable(f"Model '{name}' failed to load: {e}") from e
            return self._models[name]

    def preload(self) -> None:
        for n in self.available():
            try:
                self.get(n)
            except ModelUnavailable:
                pass

    def status(self) -> dict:
        return {n: ("loaded" if n in self._models else "error" if n in self._errors else
                    "available" if checkpoint_path(n, self.ckpt_dir).exists() else "missing")
                for n in ml_config.MODEL_NAMES}

    def analyse(self, text: str, pages: list[dict], models: list[str], categories: list[str] | None,
                catalog: Catalog) -> dict:
        with self._sem:
            runs, clauses = {}, []
            for name in models:
                pred = self.get(name)
                kwargs = {"explain": True} if name == "bilstm_attn" else {}
                p = pred.predict(text, categories=categories, **kwargs)
                attention = p.extras.get("attention", {})
                n = 0
                for key, spans in p.spans.items():
                    for i, (s, e, score) in enumerate(spans):
                        item = {
                            "id": f"{name}:{key}:{s}:{e}",
                            "model": name,
                            "category": key,
                            "category_name": catalog.cat_by_key[key]["name"],
                            "start": s, "end": e,
                            "text": text[s:e],
                            "score": score,
                            "threshold": float(pred.thresholds[pred.cat_keys.index(key)]),
                            "page_start": page_of(pages, s),
                            "page_end": page_of(pages, max(s, e - 1)),
                        }
                        if attention.get(key):
                            item["attention"] = attention[key][i]
                        clauses.append(item)
                        n += 1
                runs[name] = {"model": name, "display_name": pred.display_name, "kind": pred.kind,
                              "seconds": round(p.seconds, 3), "units": p.n_units, "n_clauses": n,
                              "score_description": pred.score_description}
            clauses.sort(key=lambda c: (c["start"], c["model"]))
            return {"runs": runs, "clauses": clauses}
