"""Secure upload validation and text extraction for PDF / TXT contracts.

* Only .pdf and .txt are accepted; the content must match (PDF magic bytes, or decodable
  text without binary control characters).
* Size is capped (CLAUSEIQ_MAX_UPLOAD_MB) while streaming, before anything is parsed.
* Uploads are written to a private temporary file which is always deleted afterwards.
* PDF text is extracted page by page; the page character ranges in the concatenated text
  are kept so every extracted clause can be reported with its page number.
* Scanned (image-only) PDFs are rejected with a clear message: OCR is NOT implemented.
"""
from __future__ import annotations

import io
import logging
import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass, field

from pypdf import PdfReader
from pypdf.errors import PdfReadError

log = logging.getLogger(__name__)
ALLOWED_EXT = {".pdf", ".txt"}
PAGE_SEPARATOR = "\n\n"


class DocumentError(ValueError):
    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class ExtractedDocument:
    filename: str
    file_type: str
    text: str
    pages: list[dict] = field(default_factory=list)  # [{"page": 1, "start": 0, "end": 1234}]
    warnings: list[str] = field(default_factory=list)


def safe_filename(name: str | None) -> str:
    name = os.path.basename(name or "contract")
    name = unicodedata.normalize("NFKC", name)
    name = re.sub(r"[^\w.\- ()]+", "_", name).strip(" .") or "contract"
    return name[:150]


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t\f\v]+\n", "\n", text)
    return text


def _decode_txt(data: bytes) -> str:
    if data.startswith(b"%PDF-"):
        raise DocumentError("The file has a .txt extension but contains PDF data.", 415)
    sample = data[:4096]
    ctrl = sum(1 for b in sample if b < 9 or 13 < b < 32)
    if sample and ctrl / len(sample) > 0.02:
        raise DocumentError("The .txt file appears to be binary, not plain text.", 415)
    for enc in ("utf-8-sig", "utf-16") if data[:2] in (b"\xff\xfe", b"\xfe\xff") else ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise DocumentError("Could not decode the text file (tried UTF-8, Windows-1252, Latin-1).")


_LIST_START = re.compile(r"\s*(\(?[0-9]{1,3}(\.[0-9]+)*[.)]?\s|\(?[a-zA-Z]{1,3}[.)]\s|[A-Z][A-Z0-9 ,&\-]{3,}$|[•\-–]\s)")


def reflow_lines(text: str) -> str:
    """Undo PDF hard wraps without changing any character offset: a single newline inside a
    running paragraph becomes a space (same length). Paragraph breaks, short lines (headings,
    signature blocks) and lines followed by list markers keep their newline."""
    lines = text.split("\n")
    out = []
    for i, line in enumerate(lines):
        out.append(line)
        if i == len(lines) - 1:
            break
        nxt = lines[i + 1]
        joinable = (len(line.strip()) >= 55 and nxt.strip() and not _LIST_START.match(nxt))
        out.append(" " if joinable else "\n")
    return "".join(out)


def _extract_pdf(path: str) -> tuple[str, list[dict], list[str]]:
    try:
        reader = PdfReader(path)
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as e:  # noqa: BLE001
                raise DocumentError("The PDF is password-protected.") from e
        parts, pages, warnings, pos = [], [], [], 0
        empty_pages = 0
        for i, page in enumerate(reader.pages, start=1):
            try:
                t = reflow_lines(_clean(page.extract_text() or ""))
            except Exception:  # noqa: BLE001 - a broken page should not kill the document
                t = ""
                warnings.append(f"Text could not be extracted from page {i}.")
            if not t.strip():
                empty_pages += 1
            if parts:
                parts.append(PAGE_SEPARATOR)
                pos += len(PAGE_SEPARATOR)
            pages.append({"page": i, "start": pos, "end": pos + len(t)})
            parts.append(t)
            pos += len(t)
    except DocumentError:
        raise
    except (PdfReadError, ValueError, KeyError, OSError) as e:
        raise DocumentError(f"The PDF could not be read ({type(e).__name__}). It may be corrupted.") from e
    text = "".join(parts)
    if len(text.strip()) < 50:
        raise DocumentError(
            "No extractable text was found in this PDF. It is probably a scanned image. "
            "ClauseIQ does not perform OCR — please upload a text-based PDF or a .txt file.")
    if empty_pages:
        warnings.append(f"{empty_pages} page(s) contained no extractable text (scanned images are not OCR'd).")
    return text, pages, warnings


async def read_upload_limited(upload, max_bytes: int) -> bytes:
    buf = io.BytesIO()
    while True:
        chunk = await upload.read(1 << 20)
        if not chunk:
            break
        buf.write(chunk)
        if buf.tell() > max_bytes:
            raise DocumentError(f"File exceeds the {max_bytes / 1024 / 1024:.0f} MB upload limit.", 413)
    return buf.getvalue()


def extract_document(filename: str, data: bytes, max_chars: int) -> ExtractedDocument:
    fname = safe_filename(filename)
    ext = os.path.splitext(fname)[1].lower()
    if ext not in ALLOWED_EXT:
        raise DocumentError("Unsupported file type. Upload a .pdf or .txt contract.", 415)
    if not data:
        raise DocumentError("The uploaded file is empty.")
    warnings: list[str] = []
    if ext == ".pdf":
        if not data.lstrip()[:5] == b"%PDF-":
            raise DocumentError("The file does not look like a valid PDF (missing %PDF header).", 415)
        fd, tmp = tempfile.mkstemp(prefix="clauseiq_", suffix=".pdf")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            text, pages, warnings = _extract_pdf(tmp)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                log.warning("could not remove temporary file %s", tmp)
    else:
        text = _clean(_decode_txt(data))
        pages = [{"page": 1, "start": 0, "end": len(text)}]
        if len(text.strip()) < 50:
            raise DocumentError("The text file contains too little text to analyse (minimum 50 characters).")
    if len(text) > max_chars:
        raise DocumentError(f"The document has {len(text):,} characters; the limit is {max_chars:,}.", 413)
    return ExtractedDocument(fname, ext[1:], text, pages, warnings)


def page_of(pages: list[dict], offset: int) -> int | None:
    for p in pages:
        if p["start"] <= offset < p["end"] or (offset == p["end"] and p is pages[-1]):
            return p["page"]
    # offsets inside the separator between pages belong to the next page
    for p in pages:
        if offset < p["start"]:
            return p["page"]
    return pages[-1]["page"] if pages else None
