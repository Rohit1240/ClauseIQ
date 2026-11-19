"""API endpoints, upload validation and PDF/TXT parsing through FastAPI's TestClient."""
import io
import os

import pytest
from fastapi.testclient import TestClient
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from backend.app.services.documents import DocumentError, extract_document, page_of, reflow_lines

from .conftest import SAMPLE, have_checkpoints


def make_pdf(pages: list[str]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    for text in pages:
        y = 740
        for line in text.split("\n"):
            c.drawString(50, y, line)
            y -= 14
        c.showPage()
    c.save()
    return buf.getvalue()


def make_image_only_pdf() -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.rect(100, 100, 300, 300, fill=1)
    c.showPage()
    c.save()
    return buf.getvalue()


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    db = tmp_path_factory.mktemp("db") / "test.sqlite3"
    os.environ["CLAUSEIQ_DB_PATH"] = str(db)
    os.environ["CLAUSEIQ_PRELOAD_MODELS"] = "false"
    os.environ["CLAUSEIQ_MAX_UPLOAD_MB"] = "1"
    from backend.app.core.config import get_settings
    get_settings.cache_clear()
    import importlib
    import backend.app.main as main
    importlib.reload(main)
    with TestClient(main.app) as c:
        yield c
    get_settings.cache_clear()


# ------------------------------------------------------------------ parsing units
def test_txt_extraction_and_decoding():
    d = extract_document("contract.txt", SAMPLE.encode("cp1252"), 10_000)
    assert d.text.startswith("MASTER SUPPLY AGREEMENT") and d.pages == [{"page": 1, "start": 0, "end": len(d.text)}]
    d2 = extract_document("c.txt", "Café agreement ".encode("utf-8") * 10, 10_000)
    assert "Café" in d2.text


def test_pdf_extraction_pages():
    data = make_pdf(["Page one says the Supplier shall deliver goods to the Buyer promptly.",
                     "Page two: governed by the laws of the State of New York in all respects."])
    d = extract_document("x.pdf", data, 10_000)
    assert len(d.pages) == 2
    p2 = d.pages[1]
    assert "Page two" in d.text[p2["start"]:p2["end"]]
    assert page_of(d.pages, d.text.index("New York")) == 2
    assert page_of(d.pages, 0) == 1


@pytest.mark.parametrize("name,data,code", [
    ("a.exe", b"MZ....", 415),
    ("a.pdf", b"not a pdf at all", 415),
    ("a.pdf", b"%PDF-1.4\n garbage", 422),
    ("a.txt", b"", 422),
    ("a.txt", b"short", 422),
    ("a.txt", bytes(range(0, 32)) * 50, 415),
    ("a.txt", b"%PDF-1.4 disguised", 415),
])
def test_rejections(name, data, code):
    with pytest.raises(DocumentError) as e:
        extract_document(name, data, 10_000)
    assert e.value.status_code == code


def test_scanned_pdf_rejected_with_ocr_message():
    with pytest.raises(DocumentError, match="OCR"):
        extract_document("scan.pdf", make_image_only_pdf(), 10_000)


def test_char_limit():
    with pytest.raises(DocumentError) as e:
        extract_document("big.txt", ("word " * 5000).encode(), 1000)
    assert e.value.status_code == 413


def test_reflow_preserves_offsets():
    t = "This is a long line that was hard wrapped by the PDF renderer at the margin\nand continues here.\n\n1. Next item\n(a) sub item"
    r = reflow_lines(t)
    assert len(r) == len(t)
    assert "margin and continues" in r and "\n\n1. Next" in r


# ------------------------------------------------------------------ endpoints
def test_health_and_metadata(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and set(h["models"]) == {"tfidf_lr", "cnn", "bilstm", "bilstm_attn"}
    cats = client.get("/api/categories").json()
    assert len(cats) == 41 and all(c["description"] for c in cats)
    ds = client.get("/api/dataset").json()
    assert ds["stats"]["n_contracts"] == 510
    ms = client.get("/api/models").json()
    assert [m["name"] for m in ms] == ["tfidf_lr", "cnn", "bilstm", "bilstm_attn"]
    assert client.get("/api/models/unknown").status_code == 404
    d = client.get("/api/dashboard").json()
    assert d["n_categories"] == 41


def test_upload_validation(client):
    r = client.post("/api/documents", files={"file": ("x.docx", b"PK..", "application/octet-stream")})
    assert r.status_code == 415 and "pdf" in r.json()["detail"].lower()
    big = b"a" * (1024 * 1024 + 10)
    r = client.post("/api/documents", files={"file": ("big.txt", big, "text/plain")})
    assert r.status_code == 413
    r = client.post("/api/documents", files={"file": ("scan.pdf", make_image_only_pdf(), "application/pdf")})
    assert r.status_code == 422 and "OCR" in r.json()["detail"]
    r = client.post("/api/analyses", json={"document_id": "missing"})
    assert r.status_code == 404
    r = client.post("/api/analyses", json={})
    assert r.status_code == 422


def test_unknown_model_and_category(client):
    doc = client.post("/api/documents", files={"file": ("c.txt", SAMPLE.encode(), "text/plain")}).json()
    r = client.post("/api/analyses", json={"document_id": doc["id"], "models": ["gpt"]})
    assert r.status_code == 422
    r = client.post("/api/analyses", json={"document_id": doc["id"], "models": ["tfidf_lr"], "categories": ["Nope"]})
    assert r.status_code == 422
    r = client.post("/api/analyses", json={"document_id": doc["id"], "models": []})
    assert r.status_code == 422


@pytest.mark.skipif(not have_checkpoints(), reason="trained checkpoints not available")
def test_full_analysis_flow_all_models(client):
    pdf = make_pdf([l for l in SAMPLE.split("\n\n")])
    up = client.post("/api/documents", files={"file": ("sample.pdf", pdf, "application/pdf")})
    assert up.status_code == 201, up.text
    doc = up.json()
    assert doc["n_pages"] == 5
    r = client.post("/api/analyses", json={"document_id": doc["id"]})
    assert r.status_code == 201, r.text
    a = r.json()
    assert a["models"] == ["tfidf_lr", "cnn", "bilstm", "bilstm_attn"]
    assert set(a["runs"]) == set(a["models"])
    text = a["document"]["text"]
    for c in a["clauses"]:
        assert text[c["start"]:c["end"]] == c["text"]
        assert 1 <= c["page_start"] <= c["page_end"] <= 5
        assert c["score"] >= c["threshold"] - 1e-6
    assert "summary" in a["comparison"]
    # restricted categories
    r2 = client.post("/api/analyses", json={"document_id": doc["id"], "models": ["bilstm"], "categories": ["Governing Law"]})
    assert r2.status_code == 201
    assert {c["category"] for c in r2.json()["clauses"]} <= {"Governing Law"}
    # retrieval, reports, history, delete
    aid = a["id"]
    assert client.get(f"/api/analyses/{aid}").json()["id"] == aid
    for fmt, ctype in (("html", "text/html"), ("csv", "text/csv"), ("json", "application/json")):
        rep = client.get(f"/api/analyses/{aid}/report?format={fmt}")
        assert rep.status_code == 200 and rep.headers["content-type"].startswith(ctype)
        assert "attachment" in rep.headers["content-disposition"]
    assert client.get(f"/api/analyses/{aid}/report?format=exe").status_code == 422
    assert any(x["id"] == aid for x in client.get("/api/analyses").json())
    assert client.get("/api/dashboard").json()["totals"]["analyses"] >= 2
    assert client.delete(f"/api/analyses/{aid}").status_code == 204
    assert client.get(f"/api/analyses/{aid}").status_code == 404
