"""Download and extract the official CUAD v1 release.

Order of sources (first one that works wins):
  1. Zenodo record 4595826 (CUAD_v1.zip, canonical release)
  2. The official GitHub repository raw files (data.zip + category_descriptions.csv)
  3. ``git clone`` of the official GitHub repository

The GitHub ``data.zip`` contains ``CUADv1.json`` (all 510 contracts) plus the official
``train_separate_questions.json`` / ``test.json`` partition used in the CUAD paper.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from clauseiq import config

log = logging.getLogger(__name__)

REQUIRED = ["CUADv1.json", "category_descriptions.csv"]
OPTIONAL_SPLIT = ["train_separate_questions.json", "test.json"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url: str, dest: Path, timeout: int = 120) -> Path:
    log.info("Downloading %s", url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "ClauseIQ/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.replace(dest)
    log.info("Saved %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return dest


def _extract_members(zip_path: Path, out_dir: Path, wanted: list[str]) -> list[str]:
    """Extract files whose basename is in ``wanted`` (flattening any folders)."""
    found = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = Path(info.filename).name
            # CUAD_v1.zip names the full file CUAD_v1.json; normalise to CUADv1.json
            target = "CUADv1.json" if name == "CUAD_v1.json" else name
            if target in wanted and not info.is_dir():
                if ".." in Path(info.filename).parts:
                    raise ValueError(f"Unsafe path in archive: {info.filename}")
                with zf.open(info) as src, open(out_dir / target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                found.append(target)
    return found


def _try_zenodo(raw: Path) -> bool:
    try:
        z = _fetch(config.ZENODO_URL, raw / "CUAD_v1.zip", timeout=300)
        got = _extract_members(z, raw, REQUIRED + OPTIONAL_SPLIT)
        return "CUADv1.json" in got
    except Exception as e:  # noqa: BLE001 - we fall back to the next source
        log.warning("Zenodo download failed: %s", e)
        return False


def _try_github_raw(raw: Path) -> bool:
    try:
        z = _fetch(config.GITHUB_DATA_URL, raw / "data.zip")
        _extract_members(z, raw, REQUIRED + OPTIONAL_SPLIT)
        _fetch(config.GITHUB_CATEGORIES_URL, raw / "category_descriptions.csv")
        return all((raw / f).exists() for f in REQUIRED)
    except Exception as e:  # noqa: BLE001
        log.warning("GitHub raw download failed: %s", e)
        return False


def _try_git_clone(raw: Path) -> bool:
    if shutil.which("git") is None:
        return False
    try:
        with tempfile.TemporaryDirectory() as td:
            subprocess.run(
                ["git", "clone", "--depth", "1", config.GITHUB_REPO, td + "/cuad"],
                check=True, capture_output=True, timeout=600,
            )
            repo = Path(td) / "cuad"
            shutil.copy(repo / "data.zip", raw / "data.zip")
            shutil.copy(repo / "category_descriptions.csv", raw / "category_descriptions.csv")
        _extract_members(raw / "data.zip", raw, REQUIRED + OPTIONAL_SPLIT)
        return all((raw / f).exists() for f in REQUIRED)
    except Exception as e:  # noqa: BLE001
        log.warning("git clone failed: %s", e)
        return False


def download_cuad(force: bool = False) -> dict:
    """Download CUAD into ``data/raw``. Returns a manifest describing what was fetched."""
    config.ensure_dirs()
    raw = config.RAW_DIR
    if not force and all((raw / f).exists() for f in REQUIRED + OPTIONAL_SPLIT):
        log.info("CUAD already present in %s (use --force to re-download)", raw)
        source = json.loads((raw / "manifest.json").read_text()).get("source", "existing") if (
            raw / "manifest.json").exists() else "existing"
    else:
        source = None
        for name, fn in (("zenodo", _try_zenodo), ("github_raw", _try_github_raw),
                         ("git_clone", _try_git_clone)):
            if fn(raw):
                source = name
                break
        if source is None:
            raise RuntimeError(
                "Could not download CUAD from Zenodo or GitHub. Download CUAD_v1.zip manually "
                "from https://zenodo.org/records/4595826 (or data.zip from "
                "https://github.com/TheAtticusProject/cuad) and place CUADv1.json and "
                f"category_descriptions.csv in {raw}."
            )
        # Zenodo does not include the official train/test partition; fetch it from GitHub.
        if not all((raw / f).exists() for f in OPTIONAL_SPLIT):
            _try_github_raw(raw) or _try_git_clone(raw)
    manifest = {
        "source": source,
        "files": {
            p.name: {"bytes": p.stat().st_size, "sha256": _sha256(p)}
            for p in sorted(raw.iterdir())
            if p.is_file() and p.suffix in {".json", ".csv"} and p.name != "manifest.json"
        },
    }
    (raw / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
