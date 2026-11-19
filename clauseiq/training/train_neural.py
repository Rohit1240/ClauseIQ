"""Training pipeline for the neural taggers (CNN, BiLSTM, BiLSTM+Attention).

* Windows of 256 tokens with a random phase every epoch (data augmentation).
* All windows containing at least one annotated token are used; windows with no annotation
  are subsampled (``neg_keep``) because ~89% of tokens are outside any clause.
* Loss: masked binary cross-entropy with per-category positive weights
  clip(sqrt(neg/pos), 1, 10) to counter class imbalance.
* AdamW + gradient clipping + ReduceLROnPlateau; early stopping on validation span F1
  (at the best of a few global thresholds).
* After training, decoding is tuned on the validation split: smoothing radius and gap-merge
  distance (global), and per category the mode (token spans or segment spans scored by mean
  token probability), threshold and minimum span length.
"""
from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from clauseiq import config
from clauseiq.data import cuad
from clauseiq.decoding import decode_hybrid, tokens_to_spans
from clauseiq.evaluation.metrics import SpanAccumulator
from clauseiq.inference.windows import predict_token_probs
from clauseiq.models.neural import build_model, count_parameters
from clauseiq.text.vocab import Vocab
from clauseiq.training.features import TokenDoc, load_segments, load_token_docs
from clauseiq.training.thresholds import tune_hybrid_decoding

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    arch: str
    max_epochs: int = 14
    patience: int = 3
    batch_size: int = 32
    window: int = 256
    neg_keep: float = 0.25
    lr: float = 2e-3
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    max_gap: int = 3
    seed: int = config.SEED
    time_budget_min: float = 60.0
    threads: int = 0  # 0 = torch default
    word_dim: int = 128
    hidden: int = 160
    cnn_dim: int = 192
    dropout: float = 0.3
    pos_weight_cap: float = 10.0
    limit_docs: int = 0  # >0: use only the first N train/val contracts (smoke tests)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _epoch_windows(docs: list[TokenDoc], L: int, neg_keep: float, rng: np.random.Generator):
    """Window start positions for one epoch: a random phase shifts the window grid each epoch."""
    out = []
    for di, d in enumerate(docs):
        n = len(d.ids)
        if n <= L:
            starts = [0]
        else:
            phase = int(rng.integers(0, L))
            starts = sorted({0, *(min(s, n - L) for s in range(phase, n, L))})
        for s in starts:
            if d.labels[s:s + L].any() or rng.random() < neg_keep:
                out.append((di, s))
    rng.shuffle(out)
    return out


def _batch(docs, items, L):
    B = len(items)
    arrs = {k: np.zeros((B, L), dtype=np.int64) for k in ("ids", "shapes", "rel", "abs")}
    C = docs[0].labels.shape[1]
    y = np.zeros((B, L, C), dtype=np.float32)
    for b, (di, s) in enumerate(items):
        d = docs[di]
        e = min(s + L, len(d.ids))
        m = e - s
        arrs["ids"][b, :m] = d.ids[s:e]
        arrs["shapes"][b, :m] = d.shapes[s:e]
        arrs["rel"][b, :m] = d.rel[s:e]
        arrs["abs"][b, :m] = d.abs[s:e]
        y[b, :m] = d.labels[s:e]
    t = {k: torch.from_numpy(v) for k, v in arrs.items()}
    return t, torch.from_numpy(y)


def predict_docs(model, docs: list[TokenDoc]) -> list[np.ndarray]:
    return [predict_token_probs(model, d.ids, d.shapes, d.rel, d.abs) for d in docs]


def evaluate_spans(docs: list[TokenDoc], probs: list[np.ndarray], contracts, keys, thresholds,
                   max_gap, smooth: int = 0) -> dict:
    acc = SpanAccumulator(keys)
    for d, P in zip(docs, probs):
        c = contracts[d.id]
        acc.add(c.text, cuad.gold_spans_merged(c),
                tokens_to_spans(P, d.starts, d.ends, thresholds, keys, max_gap=max_gap, smooth=smooth))
    return acc.results()["overall"]


EPOCH_THRESHOLDS = (0.5, 0.7, 0.85, 0.95)


def best_global_threshold(docs, probs, contracts, keys, max_gap) -> tuple[float, dict]:
    """Validation span F1 at the best of a few global thresholds. Positive-class weighting
    shifts probabilities upwards, so a fixed 0.5 would misjudge model quality."""
    best = None
    for t in EPOCH_THRESHOLDS:
        r = evaluate_spans(docs, probs, contracts, keys, np.full(len(keys), t, dtype=np.float32), max_gap)
        if best is None or r["span_f1"] > best[1]["span_f1"]:
            best = (t, r)
    return best


def train(cfg: TrainConfig) -> dict:
    config.ensure_dirs()
    set_seed(cfg.seed)
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()
    cats = cuad.load_processed_categories()
    keys = [c.key for c in cats]
    contracts = cuad.load_processed_contracts()
    vocab = Vocab.load(config.PROCESSED_DIR / "vocab.json")
    train_docs, val_docs = load_token_docs("train"), load_token_docs("val")
    if cfg.limit_docs:
        train_docs, val_docs = train_docs[:cfg.limit_docs], val_docs[:max(2, cfg.limit_docs // 4)]

    pos = np.sum([d.labels.sum(0, dtype=np.int64) for d in train_docs], axis=0).astype(np.float64)
    neg = sum(len(d.ids) for d in train_docs) - pos
    pos_weight = torch.tensor(np.clip(np.sqrt(neg / np.maximum(pos, 1)), 1, cfg.pos_weight_cap), dtype=torch.float32)

    model = build_model(cfg.arch, vocab_size=len(vocab), n_labels=len(keys), word_dim=cfg.word_dim,
                        hidden=cfg.hidden, cnn_dim=cfg.cnn_dim, dropout=cfg.dropout).to(device)
    n_params = count_parameters(model)
    log.info("%s: %d trainable parameters, device=%s, threads=%d", cfg.arch, n_params, device,
             torch.get_num_threads())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=1)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device), reduction="none")
    rng = np.random.default_rng(cfg.seed)

    history, best_f1, best_state, bad_epochs = [], -1.0, None, 0
    for epoch in range(1, cfg.max_epochs + 1):
        t_ep = time.time()
        model.train()
        items = _epoch_windows(train_docs, cfg.window, cfg.neg_keep, rng)
        tot_loss, n_batches = 0.0, 0
        for b in range(0, len(items), cfg.batch_size):
            x, y = _batch(train_docs, items[b:b + cfg.batch_size], cfg.window)
            x = {k: v.to(device) for k, v in x.items()}
            y = y.to(device)
            mask = (x["ids"] != 0).float().unsqueeze(-1)
            logits = model(x["ids"], x["shapes"], x["rel"], x["abs"])
            loss = (bce(logits, y) * mask).sum() / (mask.sum() * y.shape[-1])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tot_loss += loss.item()
            n_batches += 1
        train_s = time.time() - t_ep
        model.cpu()
        val_probs = predict_docs(model, val_docs)
        best_t, val = best_global_threshold(val_docs, val_probs, contracts, keys, cfg.max_gap)
        model.to(device)
        sched.step(val["span_f1"])
        entry = {"epoch": epoch, "train_loss": round(tot_loss / max(1, n_batches), 5),
                 "train_windows": len(items), "val_span_f1": round(val["span_f1"], 4),
                 "val_span_macro_f1": round(val["span_macro_f1"], 4),
                 "val_char_iou": round(val["char_iou"], 4), "val_threshold": best_t,
                 "lr": opt.param_groups[0]["lr"], "epoch_seconds": round(time.time() - t_ep, 1),
                 "train_seconds": round(train_s, 1)}
        history.append(entry)
        log.info("%s epoch %d: %s", cfg.arch, epoch, entry)
        if val["span_f1"] > best_f1 + 1e-4:
            best_f1, bad_epochs = val["span_f1"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                log.info("early stopping at epoch %d", epoch)
                break
        if (time.time() - t_start) / 60 > cfg.time_budget_min:
            log.warning("time budget of %.0f min reached after epoch %d", cfg.time_budget_min, epoch)
            break

    model.load_state_dict(best_state)
    model.cpu().eval()
    # ---- decoding (mode, smoothing, gap, min length, thresholds) tuned on validation
    dec, val_tuned = tune_decoding_for(model, val_docs, contracts, keys)
    th = dec["thresholds"]
    train_seconds = time.time() - t_start
    best_epoch = max(history, key=lambda h: h["val_span_f1"])["epoch"]
    ckpt = {
        "arch": cfg.arch, "hparams": model.hparams, "state_dict": model.state_dict(),
        "cat_keys": keys, "thresholds": th.tolist(), "max_gap": dec["max_gap"], "smooth": dec["smooth"],
        "decoding": decoding_params(dec),
        "window": cfg.window, "vocab_itos": vocab.itos, "train_config": asdict(cfg),
        "best_epoch": best_epoch,
    }
    if cfg.limit_docs:  # smoke runs never overwrite real checkpoints/metrics
        return {"model": cfg.arch, "history": history, "val_tuned": val_tuned, "smoke": True}
    torch.save(ckpt, config.CHECKPOINT_DIR / f"{cfg.arch}.pt")
    out = {"model": cfg.arch, "config": asdict(cfg), "n_parameters": n_params,
           "history": history, "best_epoch": best_epoch,
           "val_best_span_f1": best_f1, "val_tuned": val_tuned, "thresholds": dec["detail"],
           "decoding": decoding_summary(dec),
           "train_seconds": round(train_seconds, 1), "device": str(device),
           "torch_threads": torch.get_num_threads()}
    (config.METRICS_DIR / f"{cfg.arch}_training.json").write_text(json.dumps(out, indent=2))
    return out


def decoding_params(dec: dict) -> dict:
    return {"smooth": int(dec["smooth"]), "max_gap": int(dec["max_gap"]), "mode": list(dec["mode"]),
            "thresholds": [float(x) for x in dec["thresholds"]], "min_tokens": [int(x) for x in dec["min_tokens"]]}


def decoding_summary(dec: dict) -> dict:
    return {"smooth": dec["smooth"], "max_gap": dec["max_gap"], "val_span_f1": dec["val_span_f1"],
            "val_span_precision": dec["val_span_precision"], "val_span_recall": dec["val_span_recall"],
            "segment_mode_categories": sum(m == "segment" for m in dec["mode"])}


def tune_decoding_for(model, val_docs, contracts, keys) -> tuple[dict, dict]:
    """Validation-only decoding search (see thresholds.tune_hybrid_decoding) and the resulting
    validation span metrics."""
    val_probs = predict_docs(model, val_docs)
    gold = [cuad.gold_spans_merged(contracts[d.id]) for d in val_docs]
    seg_by_id = {d["id"]: [tuple(x) for x in d["segments"]] for d in load_segments("val")}
    segs = [seg_by_id[d.id] for d in val_docs]
    dec = tune_hybrid_decoding(val_docs, val_probs, segs, gold, keys, log=log.info)
    params = decoding_params(dec)
    acc = SpanAccumulator(keys)
    for d, P, sg in zip(val_docs, val_probs, segs):
        c = contracts[d.id]
        acc.add(c.text, cuad.gold_spans_merged(c), decode_hybrid(P, d.starts, d.ends, sg, keys, params))
    return dec, acc.results()["overall"]


def retune_checkpoint(arch: str) -> dict:
    """Re-run the validation decoding search for an existing checkpoint (same procedure as the
    end of ``train``) and update the checkpoint + training metrics in place."""
    t0 = time.time()
    path = config.CHECKPOINT_DIR / f"{arch}.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hp = dict(ckpt["hparams"])
    hp.pop("arch")
    hp["dilations"] = tuple(hp["dilations"])
    model = build_model(arch, **hp)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    keys = ckpt["cat_keys"]
    contracts = cuad.load_processed_contracts()
    dec, val_tuned = tune_decoding_for(model, load_token_docs("val"), contracts, keys)
    ckpt.update(thresholds=dec["thresholds"].tolist(), max_gap=dec["max_gap"], smooth=dec["smooth"],
                decoding=decoding_params(dec))
    torch.save(ckpt, path)
    mpath = config.METRICS_DIR / f"{arch}_training.json"
    m = json.loads(mpath.read_text())
    extra = round(time.time() - t0, 1)
    base = m.get("train_seconds", 0) - m.get("retune_seconds", 0)
    m.update(val_tuned=val_tuned, thresholds=dec["detail"], decoding=decoding_summary(dec),
             retune_seconds=extra, train_seconds=round(base + extra, 1))
    mpath.write_text(json.dumps(m, indent=2))
    return m["decoding"]
