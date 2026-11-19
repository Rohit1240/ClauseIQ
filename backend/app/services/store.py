"""SQLite persistence for uploaded documents and analysis results (no ORM needed)."""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY,
  filename TEXT NOT NULL,
  file_type TEXT NOT NULL,
  n_chars INTEGER NOT NULL,
  n_pages INTEGER NOT NULL,
  text TEXT NOT NULL,
  pages_json TEXT NOT NULL,
  warnings_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS analyses (
  id TEXT PRIMARY KEY,
  document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  models_json TEXT NOT NULL,
  categories_json TEXT NOT NULL,
  n_clauses INTEGER NOT NULL,
  seconds REAL NOT NULL,
  result_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(created_at DESC);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # documents ------------------------------------------------------------
    def add_document(self, doc) -> dict:
        did = uuid.uuid4().hex[:16]
        row = dict(id=did, filename=doc.filename, file_type=doc.file_type, n_chars=len(doc.text),
                   n_pages=len(doc.pages), text=doc.text, pages_json=json.dumps(doc.pages),
                   warnings_json=json.dumps(doc.warnings), created_at=now())
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO documents VALUES (:id,:filename,:file_type,:n_chars,:n_pages,:text,"
                      ":pages_json,:warnings_json,:created_at)", row)
        return self.get_document(did)

    def get_document(self, did: str) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM documents WHERE id=?", (did,)).fetchone()
        if r is None:
            return None
        d = dict(r)
        d["pages"] = json.loads(d.pop("pages_json"))
        d["warnings"] = json.loads(d.pop("warnings_json"))
        return d

    # analyses -------------------------------------------------------------
    def add_analysis(self, document_id: str, models: list[str], categories: list[str],
                     n_clauses: int, seconds: float, result: dict) -> str:
        aid = uuid.uuid4().hex[:16]
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO analyses VALUES (?,?,?,?,?,?,?,?)",
                      (aid, document_id, json.dumps(models), json.dumps(categories), n_clauses,
                       seconds, json.dumps(result), now()))
        return aid

    def get_analysis(self, aid: str) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM analyses WHERE id=?", (aid,)).fetchone()
        if r is None:
            return None
        d = dict(r)
        d["models"] = json.loads(d.pop("models_json"))
        d["categories"] = json.loads(d.pop("categories_json"))
        d["result"] = json.loads(d.pop("result_json"))
        return d

    def delete_analysis(self, aid: str) -> bool:
        with self._lock, self._conn() as c:
            doc = c.execute("SELECT document_id FROM analyses WHERE id=?", (aid,)).fetchone()
            if doc is None:
                return False
            c.execute("DELETE FROM analyses WHERE id=?", (aid,))
            left = c.execute("SELECT COUNT(*) FROM analyses WHERE document_id=?", (doc[0],)).fetchone()[0]
            if left == 0:
                c.execute("DELETE FROM documents WHERE id=?", (doc[0],))
        return True

    def recent(self, limit: int = 10) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT a.id, a.document_id, a.models_json, a.categories_json, a.n_clauses, a.seconds, "
                "a.created_at, d.filename, d.n_pages, d.n_chars FROM analyses a JOIN documents d "
                "ON d.id = a.document_id ORDER BY a.created_at DESC, a.rowid DESC LIMIT ?", (limit,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["models"] = json.loads(d.pop("models_json"))
            d["n_categories"] = len(json.loads(d.pop("categories_json")))
            out.append(d)
        return out

    def totals(self) -> dict:
        with self._conn() as c:
            n_an, n_cl = c.execute("SELECT COUNT(*), COALESCE(SUM(n_clauses),0) FROM analyses").fetchone()
            n_docs = c.execute("SELECT COUNT(DISTINCT document_id) FROM analyses").fetchone()[0]
        return {"analyses": n_an, "contracts_analysed": n_docs, "clauses_extracted": n_cl}
