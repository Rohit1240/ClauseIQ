"""Parse, validate, summarise and split the CUAD v1 annotations.

The source of truth is ``CUADv1.json`` (SQuAD 2.0 format, one paragraph per contract,
41 questions per contract). Category names and definitions come from the official
``category_descriptions.csv``; nothing is renamed or synthesised.
"""
from __future__ import annotations

import csv
import json
import logging
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from clauseiq import config

log = logging.getLogger(__name__)


@dataclass
class Category:
    key: str            # name as used in CUADv1.json question ids, e.g. "Ip Ownership Assignment"
    name: str           # official display name from category_descriptions.csv
    description: str
    answer_format: str
    group: str
    index: int


@dataclass
class Span:
    category: str  # Category.key
    start: int
    end: int       # exclusive
    text: str


@dataclass
class Contract:
    id: str
    text: str
    spans: list[Span] = field(default_factory=list)


def _strip_prefix(value: str, prefix: str) -> str:
    value = value.strip()
    return value[len(prefix):].strip() if value.startswith(prefix) else value


def load_categories(raw_dir: Path | None = None) -> list[Category]:
    raw_dir = raw_dir or config.RAW_DIR
    path = raw_dir / "category_descriptions.csv"
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    assert header[0].startswith("Category"), header
    cats = []
    for i, r in enumerate(body):
        name = _strip_prefix(r[0], "Category:")
        cats.append(Category(
            key="",  # filled in by align_categories
            name=name,
            description=" ".join(_strip_prefix(r[1], "Description:").split()),
            answer_format=_strip_prefix(r[2], "Answer Format:"),
            group=_strip_prefix(r[3], "Group:"),
            index=i,
        ))
    return cats


def align_categories(cats: list[Category], json_keys: set[str]) -> list[Category]:
    """Match CSV names to the question keys used in CUADv1.json (case-insensitive)."""
    by_lower = {k.lower(): k for k in json_keys}
    for c in cats:
        k = by_lower.get(c.name.lower())
        if k is None:
            raise ValueError(f"Category {c.name!r} not found in CUAD JSON keys")
        c.key = k
    if len(cats) != len(json_keys):
        raise ValueError(f"{len(cats)} CSV categories vs {len(json_keys)} JSON categories")
    return cats


def load_contracts(path: Path) -> tuple[list[Contract], set[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    contracts, keys = [], set()
    for doc in data["data"]:
        if len(doc["paragraphs"]) != 1:
            raise ValueError(f"{doc['title']}: expected exactly one paragraph")
        para = doc["paragraphs"][0]
        text = para["context"]
        c = Contract(id=doc["title"], text=text)
        seen = set()
        for qa in para["qas"]:
            key = qa["id"].split("__")[-1]
            keys.add(key)
            for a in qa["answers"]:
                s = int(a["answer_start"])
                e = s + len(a["text"])
                if (key, s, e) in seen:
                    continue
                seen.add((key, s, e))
                c.spans.append(Span(key, s, e, a["text"]))
        contracts.append(c)
    return contracts, keys


def load_official_split(raw_dir: Path) -> dict[str, list[str]] | None:
    tr, te = raw_dir / "train_separate_questions.json", raw_dir / "test.json"
    if not (tr.exists() and te.exists()):
        return None
    titles = {}
    for name, p in (("train", tr), ("test", te)):
        titles[name] = sorted({d["title"] for d in json.loads(p.read_text())["data"]})
    return titles


def make_split(contract_ids: list[str], raw_dir: Path, seed: int = config.SEED,
               val_fraction: float = config.VAL_FRACTION) -> dict:
    """Contract-level split. Test = official CUAD test partition (102 contracts) when
    available; validation = seeded random sample of the official training contracts."""
    official = load_official_split(raw_dir)
    ids = sorted(contract_ids)
    if official:
        test = sorted(set(official["test"]) & set(ids))
        pool = sorted(set(ids) - set(test))
        source = "official CUAD test partition + seeded validation split of official train"
    else:
        rng = random.Random(seed)
        shuffled = ids[:]
        rng.shuffle(shuffled)
        n_test = round(0.2 * len(ids))
        test, pool = sorted(shuffled[:n_test]), sorted(shuffled[n_test:])
        source = "seeded 80/20 contract-level split (official partition unavailable)"
    rng = random.Random(seed)
    pool_shuffled = pool[:]
    rng.shuffle(pool_shuffled)
    n_val = round(val_fraction * len(pool))
    val, train = sorted(pool_shuffled[:n_val]), sorted(pool_shuffled[n_val:])
    assert not (set(train) & set(val)) and not (set(train) & set(test)) and not (set(val) & set(test))
    return {"seed": seed, "source": source, "train": train, "val": val, "test": test}


def validate(contracts: list[Contract], cats: list[Category], split: dict | None = None) -> dict:
    """Integrity checks. Raises on hard failures, returns a report."""
    errors, warnings = [], []
    ids = [c.id for c in contracts]
    if len(set(ids)) != len(ids):
        errors.append("duplicate contract titles")
    bad_offsets = 0
    for c in contracts:
        if not c.text.strip():
            errors.append(f"empty text: {c.id}")
        for s in c.spans:
            if c.text[s.start:s.end] != s.text:
                bad_offsets += 1
    if bad_offsets:
        errors.append(f"{bad_offsets} answer offsets do not match the contract text")
    keys = {c.key for c in cats}
    unknown = {s.category for c in contracts for s in c.spans} - keys
    if unknown:
        errors.append(f"unknown categories: {sorted(unknown)}")
    if len(contracts) != 510:
        warnings.append(f"expected 510 contracts, found {len(contracts)}")
    if len(cats) != 41:
        warnings.append(f"expected 41 categories, found {len(cats)}")
    if split:
        for a in ("train", "val", "test"):
            for b in ("train", "val", "test"):
                if a < b and set(split[a]) & set(split[b]):
                    errors.append(f"contract leakage between {a} and {b}")
        covered = set(split["train"]) | set(split["val"]) | set(split["test"])
        if covered != set(ids):
            errors.append("split does not cover every contract exactly once")
    report = {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "n_contracts": len(contracts),
        "n_categories": len(cats),
        "n_spans": sum(len(c.spans) for c in contracts),
        "offsets_checked": sum(len(c.spans) for c in contracts),
        "offset_mismatches": bad_offsets,
    }
    if errors:
        raise ValueError("CUAD validation failed: " + "; ".join(errors))
    return report


def _describe(values: list[int]) -> dict:
    if not values:
        return {"count": 0}
    v = sorted(values)
    return {
        "count": len(v), "min": v[0], "max": v[-1], "mean": round(statistics.mean(v), 1),
        "median": statistics.median(v), "p90": v[int(0.9 * (len(v) - 1))],
    }


def compute_stats(contracts: list[Contract], cats: list[Category], split: dict) -> dict:
    by_id = {c.id: c for c in contracts}
    split_of = {cid: name for name in ("train", "val", "test") for cid in split[name]}
    per_cat = {}
    for cat in cats:
        spans = [s for c in contracts for s in c.spans if s.category == cat.key]
        with_clause = {c.id for c in contracts if any(s.category == cat.key for s in c.spans)}
        per_split = Counter(split_of[c.id] for c in contracts for s in c.spans if s.category == cat.key)
        per_split_contracts = Counter(split_of[cid] for cid in with_clause)
        per_cat[cat.key] = {
            "name": cat.name,
            "n_spans": len(spans),
            "n_contracts_with_clause": len(with_clause),
            "n_contracts_without_clause": len(contracts) - len(with_clause),
            "pct_contracts_with_clause": round(100 * len(with_clause) / len(contracts), 1),
            "span_chars": _describe([s.end - s.start for s in spans]),
            "spans_by_split": {k: per_split.get(k, 0) for k in ("train", "val", "test")},
            "contracts_by_split": {k: per_split_contracts.get(k, 0) for k in ("train", "val", "test")},
        }
    counts = [v["n_spans"] for v in per_cat.values()]
    lengths = [len(c.text) for c in contracts]
    total_chars = sum(lengths)
    covered = 0
    for c in contracts:
        mask = bytearray(len(c.text))
        for s in c.spans:
            mask[s.start:s.end] = b"\x01" * (s.end - s.start)
        covered += sum(mask)
    empty_pairs = sum(v["n_contracts_without_clause"] for v in per_cat.values())
    rarest = sorted(per_cat.items(), key=lambda kv: kv[1]["n_spans"])[:5]
    commonest = sorted(per_cat.items(), key=lambda kv: -kv[1]["n_spans"])[:5]
    return {
        "n_contracts": len(contracts),
        "n_categories": len(cats),
        "n_spans": sum(counts),
        "split": {k: len(split[k]) for k in ("train", "val", "test")},
        "split_source": split["source"],
        "spans_by_split": {k: sum(len(by_id[i].spans) for i in split[k]) for k in ("train", "val", "test")},
        "contract_chars": _describe(lengths),
        "total_chars": total_chars,
        "annotated_char_fraction": round(covered / total_chars, 4),
        "missing_annotations": {
            "contract_category_pairs": len(contracts) * len(cats),
            "pairs_without_clause": empty_pairs,
            "pct_pairs_without_clause": round(100 * empty_pairs / (len(contracts) * len(cats)), 1),
            "note": "In CUAD an empty answer means the clause type is absent from the contract "
                    "(SQuAD 2.0 'is_impossible'), not that annotation is missing.",
            "contracts_with_no_spans": sum(1 for c in contracts if not c.spans),
        },
        "class_imbalance": {
            "max_spans": max(counts), "min_spans": min(counts),
            "imbalance_ratio": round(max(counts) / max(1, min(counts)), 1),
            "gini": round(_gini(counts), 3),
            "most_common": [(per_cat[k]["name"], v["n_spans"]) for k, v in commonest],
            "rarest": [(per_cat[k]["name"], v["n_spans"]) for k, v in rarest],
        },
        "categories": per_cat,
    }


def _gini(xs: list[int]) -> float:
    xs = sorted(xs)
    n, s = len(xs), sum(xs)
    if n == 0 or s == 0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(xs))
    return (2 * cum) / (n * s) - (n + 1) / n


# ---------------------------------------------------------------- processed I/O

def save_processed(contracts: list[Contract], cats: list[Category], split: dict, stats: dict,
                   validation: dict, out_dir: Path | None = None) -> None:
    out_dir = out_dir or config.PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "contracts.jsonl", "w", encoding="utf-8") as f:
        for c in contracts:
            f.write(json.dumps({"id": c.id, "text": c.text,
                                "spans": [asdict(s) for s in c.spans]}) + "\n")
    (out_dir / "categories.json").write_text(json.dumps([asdict(c) for c in cats], indent=2))
    (out_dir / "splits.json").write_text(json.dumps(split, indent=2))
    (out_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    (out_dir / "validation.json").write_text(json.dumps(validation, indent=2))


def load_processed_categories(processed_dir: Path | None = None) -> list[Category]:
    p = (processed_dir or config.PROCESSED_DIR) / "categories.json"
    return [Category(**d) for d in json.loads(p.read_text())]


def load_processed_contracts(processed_dir: Path | None = None) -> dict[str, Contract]:
    p = (processed_dir or config.PROCESSED_DIR) / "contracts.jsonl"
    out = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out[d["id"]] = Contract(d["id"], d["text"], [Span(**s) for s in d["spans"]])
    return out


def load_split(processed_dir: Path | None = None) -> dict:
    return json.loads(((processed_dir or config.PROCESSED_DIR) / "splits.json").read_text())


def gold_spans_merged(contract: Contract) -> dict[str, list[tuple[int, int]]]:
    """Per-category gold spans with overlapping annotations merged (used for evaluation)."""
    by_cat: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for s in contract.spans:
        by_cat[s.category].append((s.start, s.end))
    merged = {}
    for k, spans in by_cat.items():
        spans.sort()
        out = [list(spans[0])]
        for s, e in spans[1:]:
            if s < out[-1][1]:  # strict overlap
                out[-1][1] = max(out[-1][1], e)
            else:
                out.append([s, e])
        merged[k] = [tuple(x) for x in out]
    return merged
