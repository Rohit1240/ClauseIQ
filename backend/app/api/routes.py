"""HTTP API. All responses are JSON unless a report file is requested."""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field

from clauseiq import __version__
from clauseiq import config as ml_config

from ..services import reports
from ..services.comparison import compare
from ..services.documents import DocumentError, extract_document, read_upload_limited
from ..services.models import ModelUnavailable

log = logging.getLogger("clauseiq.api")
router = APIRouter(prefix="/api")


def _state(request: Request):
    return request.app.state


# --------------------------------------------------------------------------- meta
@router.get("/health")
def health(request: Request):
    st = _state(request)
    status = st.registry.status()
    return {"status": "ok", "version": __version__, "models": status,
            "models_available": sum(v in ("loaded", "available") for v in status.values()),
            "metrics_available": sorted(st.catalog.metrics)}


@router.get("/dataset")
def dataset(request: Request):
    c = _state(request).catalog
    return {"stats": c.stats, "features": c.features,
            "attribution": "Contract Understanding Atticus Dataset (CUAD) v1 by The Atticus Project, "
                           "licensed CC BY 4.0. https://www.atticusprojectai.org/cuad"}


@router.get("/categories")
def categories(request: Request):
    c = _state(request).catalog
    out = []
    for cat in c.categories:
        s = c.stats["categories"][cat["key"]]
        perf = {}
        for m, met in c.metrics.items():
            sp = met["span_per_category"][cat["key"]]
            cd = met["contract_detection"]["per_category"][cat["key"]]
            perf[m] = {"span_f1": sp["span_f1"], "span_precision": sp["span_precision"],
                       "span_recall": sp["span_recall"], "exact_match_f1": sp["exact_match_f1"],
                       "char_iou": sp["char_iou"], "test_gold_spans": sp["n_gold"],
                       "contract_f1": cd["f1"],
                       "segment_f1": met["segment_classification"]["per_category"][cat["key"]]["f1"]}
        out.append({**cat, "stats": s, "performance": perf})
    return out


def _model_summary(name: str, catalog, registry) -> dict:
    met = catalog.metrics.get(name)
    tr = catalog.training.get(name, {})
    return {
        "name": name, "display_name": ml_config.MODEL_DISPLAY[name],
        "status": registry.status()[name],
        "kind": "segment" if name == "tfidf_lr" else "token",
        "evaluated": met is not None,
        "span_extraction": met["span_extraction"] if met else None,
        "segment_classification": {k: v for k, v in met["segment_classification"].items()
                                   if k not in ("per_category", "confusion_matrix")} if met else None,
        "contract_detection": {k: v for k, v in met["contract_detection"].items() if k != "per_category"} if met else None,
        "timing": met["timing"] if met else None,
        "training": {k: tr.get(k) for k in ("config", "history", "best_epoch", "n_parameters", "selected",
                                             "train_seconds", "n_features", "n_train_segments", "device", "decoding",
                                             "val_tuned")},
    }


@router.get("/models")
def models(request: Request):
    st = _state(request)
    return [_model_summary(n, st.catalog, st.registry) for n in ml_config.MODEL_NAMES]


@router.get("/models/{name}")
def model_detail(name: str, request: Request):
    st = _state(request)
    if name not in ml_config.MODEL_NAMES:
        raise HTTPException(404, f"Unknown model '{name}'.")
    out = _model_summary(name, st.catalog, st.registry)
    met = st.catalog.metrics.get(name)
    if met:
        out["span_per_category"] = met["span_per_category"]
        out["confusion_matrix"] = met["segment_classification"]["confusion_matrix"]
        out["contract_per_category"] = met["contract_detection"]["per_category"]
        out["thresholds"] = met["thresholds"]
    return out


@router.get("/dashboard")
def dashboard(request: Request):
    st = _state(request)
    return {"totals": st.store.totals(), "n_categories": len(st.catalog.categories),
            "models": [_model_summary(n, st.catalog, st.registry) for n in ml_config.MODEL_NAMES],
            "recent": st.store.recent(8), "dataset": {k: st.catalog.stats[k] for k in (
                "n_contracts", "n_spans", "split")}}


# --------------------------------------------------------------------------- documents
@router.post("/documents", status_code=201)
async def upload_document(request: Request, file: UploadFile = File(...)):
    st = _state(request)
    s = st.settings
    try:
        data = await read_upload_limited(file, s.max_upload_bytes)
        doc = extract_document(file.filename or "contract", data, s.max_document_chars)
    except DocumentError as e:
        log.info("rejected upload %r: %s", file.filename, e)
        raise HTTPException(e.status_code, str(e)) from e
    finally:
        await file.close()
    saved = st.store.add_document(doc)
    log.info("stored document %s (%s, %d chars, %d pages)", saved["id"], saved["filename"],
             saved["n_chars"], saved["n_pages"])
    return saved


@router.get("/documents/{doc_id}")
def get_document(doc_id: str, request: Request):
    d = _state(request).store.get_document(doc_id)
    if d is None:
        raise HTTPException(404, "Document not found.")
    return d


# --------------------------------------------------------------------------- analyses
class AnalysisRequest(BaseModel):
    document_id: str = Field(min_length=1, max_length=64)
    models: list[str] = Field(default_factory=lambda: list(ml_config.MODEL_NAMES))
    categories: list[str] | None = None  # None / empty = all 41


@router.post("/analyses", status_code=201)
def create_analysis(body: AnalysisRequest, request: Request):
    st = _state(request)
    doc = st.store.get_document(body.document_id)
    if doc is None:
        raise HTTPException(404, "Document not found. Upload it again.")
    models = list(dict.fromkeys(body.models))
    if not models:
        raise HTTPException(422, "Select at least one model.")
    bad = [m for m in models if m not in ml_config.MODEL_NAMES]
    if bad:
        raise HTTPException(422, f"Unknown model(s): {', '.join(bad)}.")
    cats = list(dict.fromkeys(body.categories or []))
    unknown = [c for c in cats if c not in st.catalog.cat_by_key]
    if unknown:
        raise HTTPException(422, f"Unknown category key(s): {', '.join(unknown[:5])}.")
    t0 = time.time()
    try:
        result = st.registry.analyse(doc["text"], doc["pages"], models, cats or None, st.catalog)
    except ModelUnavailable as e:
        raise HTTPException(503, str(e)) from e
    result["comparison"] = compare(result["clauses"], models)
    seconds = time.time() - t0
    aid = st.store.add_analysis(doc["id"], models, cats or [c["key"] for c in st.catalog.categories],
                                len(result["clauses"]), seconds, result)
    log.info("analysis %s: %d clauses from %s in %.2fs", aid, len(result["clauses"]), models, seconds)
    return _analysis_payload(aid, st)


def _analysis_payload(aid: str, st, include_text: bool = True) -> dict:
    a = st.store.get_analysis(aid)
    if a is None:
        raise HTTPException(404, "Analysis not found.")
    doc = st.store.get_document(a["document_id"])
    return {"id": a["id"], "created_at": a["created_at"], "seconds": a["seconds"], "models": a["models"],
            "categories": a["categories"], "n_clauses": a["n_clauses"], **a["result"],
            "document": {k: doc[k] for k in ("id", "filename", "file_type", "n_chars", "n_pages", "pages",
                                              "warnings", "created_at")} | ({"text": doc["text"]} if include_text else {})}


@router.get("/analyses")
def list_analyses(request: Request, limit: int = Query(20, ge=1, le=100)):
    return _state(request).store.recent(limit)


@router.get("/analyses/{aid}")
def get_analysis(aid: str, request: Request):
    return _analysis_payload(aid, _state(request))


@router.delete("/analyses/{aid}", status_code=204)
def delete_analysis(aid: str, request: Request):
    if not _state(request).store.delete_analysis(aid):
        raise HTTPException(404, "Analysis not found.")
    return Response(status_code=204)


@router.get("/analyses/{aid}/report")
def report(aid: str, request: Request, format: str = Query("html", pattern="^(html|csv|json)$")):
    st = _state(request)
    a = st.store.get_analysis(aid)
    if a is None:
        raise HTTPException(404, "Analysis not found.")
    doc = st.store.get_document(a["document_id"])
    stem = doc["filename"].rsplit(".", 1)[0][:60] or "contract"
    disp = lambda ext: {"Content-Disposition": f'attachment; filename="ClauseIQ_{stem}.{ext}"'}  # noqa: E731
    if format == "csv":
        return PlainTextResponse(reports.to_csv(a), media_type="text/csv", headers=disp("csv"))
    if format == "json":
        return Response(reports.to_json(a, doc), media_type="application/json", headers=disp("json"))
    return HTMLResponse(reports.to_html(a, doc), headers=disp("html"))
