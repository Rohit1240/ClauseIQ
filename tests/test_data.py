"""CUAD download extraction, parsing, validation, statistics and splitting."""
import json
import zipfile

import pytest

from clauseiq.data import cuad
from clauseiq.data.download import _extract_members

from .conftest import have_processed

CSV = (
    "﻿Category (incl. context and answer),Description,Answer Format,Group\n"
    "Category: Parties,Description: The two or more parties who signed the contract,Answer Format: Entity,Group: -\n"
    "Category: Governing Law,Description: Which state's law governs?,Answer Format: State,Group: -\n"
)


def _squad(contracts):
    data = []
    for title, text, answers in contracts:
        qas = []
        for key in ("Parties", "Governing Law"):
            ans = [{"text": t, "answer_start": text.index(t)} for k, t in answers if k == key]
            qas.append({"id": f"{title}__{key}", "question": key, "answers": ans, "is_impossible": not ans})
        data.append({"title": title, "paragraphs": [{"context": text, "qas": qas}]})
    return {"version": "aok_v1.0", "data": data}


@pytest.fixture
def raw_dir(tmp_path):
    contracts = [(f"C{i}", f"Agreement between Alpha{i} LLC and Beta. Governed by laws of State{i}.",
                  [("Parties", f"Alpha{i} LLC"), ("Governing Law", f"Governed by laws of State{i}")]) for i in range(10)]
    (tmp_path / "CUADv1.json").write_text(json.dumps(_squad(contracts)))
    (tmp_path / "category_descriptions.csv").write_text(CSV, encoding="utf-8")
    return tmp_path


def test_parse_validate_stats_split(raw_dir):
    contracts, keys = cuad.load_contracts(raw_dir / "CUADv1.json")
    cats = cuad.align_categories(cuad.load_categories(raw_dir), keys)
    assert [c.name for c in cats] == ["Parties", "Governing Law"]
    assert cats[0].description.startswith("The two or more parties")
    split = cuad.make_split([c.id for c in contracts], raw_dir, seed=1)
    assert sum(len(split[k]) for k in ("train", "val", "test")) == 10
    rep = cuad.validate(contracts, cats, split)
    assert rep["ok"] and rep["n_spans"] == 20 and rep["offset_mismatches"] == 0
    stats = cuad.compute_stats(contracts, cats, split)
    assert stats["categories"]["Parties"]["n_spans"] == 10
    assert stats["class_imbalance"]["imbalance_ratio"] == 1.0
    # deterministic split
    assert cuad.make_split([c.id for c in contracts], raw_dir, seed=1) == split


def test_validation_detects_bad_offsets(raw_dir):
    contracts, keys = cuad.load_contracts(raw_dir / "CUADv1.json")
    cats = cuad.align_categories(cuad.load_categories(raw_dir), keys)
    contracts[0].spans[0].start += 1
    with pytest.raises(ValueError, match="offsets"):
        cuad.validate(contracts, cats)


def test_validation_detects_leakage(raw_dir):
    contracts, keys = cuad.load_contracts(raw_dir / "CUADv1.json")
    cats = cuad.align_categories(cuad.load_categories(raw_dir), keys)
    split = cuad.make_split([c.id for c in contracts], raw_dir)
    split["test"] = split["test"] + [split["train"][0]]
    with pytest.raises(ValueError, match="leakage"):
        cuad.validate(contracts, cats, split)


def test_gold_spans_merge_overlaps():
    c = cuad.Contract("x", "a" * 50, [cuad.Span("A", 0, 10, ""), cuad.Span("A", 5, 20, ""), cuad.Span("A", 30, 40, "")])
    assert cuad.gold_spans_merged(c) == {"A": [(0, 20), (30, 40)]}


def test_zip_member_extraction(tmp_path):
    z = tmp_path / "CUAD_v1.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("CUAD_v1/CUAD_v1.json", "{}")
        zf.writestr("CUAD_v1/other.txt", "x")
    out = tmp_path / "out"
    out.mkdir()
    assert _extract_members(z, out, ["CUADv1.json"]) == ["CUADv1.json"]
    assert (out / "CUADv1.json").exists() and not (out / "other.txt").exists()


@pytest.mark.skipif(not have_processed(), reason="real CUAD data not prepared")
def test_real_cuad_processed():
    contracts = cuad.load_processed_contracts()
    cats = cuad.load_processed_categories()
    split = cuad.load_split()
    assert len(contracts) == 510 and len(cats) == 41
    assert sum(len(c.spans) for c in contracts.values()) == 13823
    assert len(split["test"]) == 102
    assert not (set(split["train"]) | set(split["val"])) & set(split["test"])
    for c in list(contracts.values())[:50]:
        for s in c.spans:
            assert c.text[s.start:s.end] == s.text
