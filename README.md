# ClauseIQ — Intelligent Contract Analysis

ClauseIQ finds and extracts legal clauses in commercial contracts. Upload a PDF or TXT
contract and four NLP models — **all trained from scratch on the Contract Understanding
Atticus Dataset (CUAD)** — return the exact clause text for any of the 41 CUAD categories,
with character offsets, page numbers, in-document highlighting, a side-by-side model
comparison and downloadable reports.

| Model | Type | Extraction unit |
|---|---|---|
| TF-IDF + Logistic Regression | traditional baseline | sentence-like segments |
| Dilated CNN | local n-gram patterns | tokens |
| BiLSTM | sequential context | tokens |
| BiLSTM + Attention | context + learned word relevance | tokens |

No LLMs, no pretrained transformers, no pretrained embeddings and no external NLP APIs are
used. Every weight and embedding is initialised randomly and learned from CUAD's training
contracts. Measured results are in [`FINAL_REPORT.md`](FINAL_REPORT.md) and
`artifacts/reports/evaluation_report.md`.

---

## Contents

1. [Environment setup](#1-environment-setup)
2. [Dependency installation](#2-dependency-installation)
3. [Dataset downloading](#3-dataset-downloading)
4. [Dataset validation](#4-dataset-validation)
5. [Data preprocessing](#5-data-preprocessing)
6. [Individual model training](#6-individual-model-training)
7. [Training all four models](#7-training-all-four-models)
8. [Evaluation](#8-evaluation)
9. [Generating comparison reports](#9-generating-comparison-reports)
10. [Running the backend](#10-running-the-backend)
11. [Running the frontend](#11-running-the-frontend)
12. [Using the application](#12-using-the-application)
13. [CPU and NVIDIA CUDA configuration](#13-cpu-and-nvidia-cuda-configuration)
14. [Deployment](#14-deployment)
15. [Troubleshooting](#15-troubleshooting)

Also: [project layout](#project-layout), [tests](#tests), [dataset attribution](#dataset-attribution).

The archive ships with trained checkpoints, metrics and reports in `artifacts/` and a
production frontend build in `frontend/dist/`, so after installing the Python dependencies you
can **skip straight to step 10**: `uvicorn backend.app.main:app --port 8000` serves the complete
app at http://localhost:8000. Steps 3–9 reproduce everything. Two CUAD test-split contracts in
`samples/` are ready to upload.

---

## 1. Environment setup

Requirements: Python 3.10–3.13, Node.js 18+ (20 or 22 recommended), ~2 GB free disk,
4 GB RAM (8 GB recommended for training). A GPU is optional.

```bash
cd ClauseIQ
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
cp .env.example .env                 # optional; defaults work for local use
```

## 2. Dependency installation

```bash
# CPU-only PyTorch (smallest download). For CUDA see section 13.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt      # runtime + test dependencies

cd frontend && npm ci && cd ..
```

`requirements.txt` holds runtime dependencies only (used by the Docker image);
`requirements-dev.txt` adds pytest, httpx and reportlab for the test-suite.

## 3. Dataset downloading

```bash
python scripts/download_data.py          # add --force to re-download
```

Sources are tried in order: Zenodo record 4595826 (`CUAD_v1.zip`), the official GitHub
repository's raw `data.zip`, then `git clone` of `TheAtticusProject/cuad`. Files land in
`data/raw/`: `CUADv1.json` (all 510 contracts), `category_descriptions.csv` (official names
and definitions) and the official `train_separate_questions.json` / `test.json` partition.
A `manifest.json` with SHA-256 checksums records what was downloaded.

## 4. Dataset validation

```bash
python scripts/validate_data.py
```

Checks: 510 contracts, 41 categories matched to the official CSV, every one of the 13,823
answer offsets reproduces its text exactly, no duplicate contracts, and no contract appears in
two splits. Exits non-zero on failure.

## 5. Data preprocessing

```bash
python scripts/preprocess.py
```

Writes `data/processed/`: parsed contracts, category table, the contract-level split
(`splits.json`), dataset statistics (`dataset_stats.json`: per-category counts, absent-clause
analysis, class imbalance), the training vocabulary (`vocab.json`, built from training
contracts only), token features with character offsets (`tokens_{split}.npz`) and segments
(`segments_{split}.json`). Small metadata files are also copied to `artifacts/dataset/` for
the web app. Takes ~15 seconds.

**Split.** Test = the official CUAD test partition (102 contracts). Validation = 15% of the
official training contracts chosen with seed 13 (61 contracts); train = the remaining 347.
Splitting is by contract, so no contract text leaks between splits.

## 6. Individual model training

```bash
python scripts/train.py --model tfidf_lr        # ~17 min on 2 CPU cores
python scripts/train.py --model cnn --epochs 30 --patience 4 --time-budget 60
python scripts/train.py --model bilstm --epochs 20 --time-budget 65
python scripts/train.py --model bilstm_attn --epochs 30 --patience 4 --time-budget 95
```

These are the exact settings used for the shipped checkpoints (2 CPU threads, no GPU). On a
GPU the same epochs take a few minutes. Run one training job at a time on small machines:
two PyTorch processes sharing two cores slowed each other down by an order of magnitude in our
runs (OpenMP thread oversubscription). If you must run jobs side by side, give each
`--threads 1` / `OMP_NUM_THREADS=1`.

After training, each neural model's decoding (smoothing, gap merge, and per category: token-
or segment-level spans, threshold, minimum length) is tuned on the validation split and saved
in its checkpoint. `python scripts/tune_decoding.py --model <name>` repeats only that step.

Useful options: `--epochs N`, `--time-budget MINUTES` (per neural model, default 60),
`--batch-size`, `--lr`, `--neg-keep`, `--threads N`, `--limit-docs N` (smoke test on N
contracts; writes nothing). Each run writes `artifacts/checkpoints/<model>.{pt,pkl}` and
`artifacts/metrics/<model>_training.json` (configuration, per-epoch history, chosen
thresholds, training time).

## 7. Training all four models

```bash
python scripts/train.py --model all                  # train the four models in sequence
python scripts/train_all.py                          # download → … → train → evaluate → report
python scripts/train_all.py --skip-data --time-budget 90
```

## 8. Evaluation

```bash
python scripts/evaluate.py --model all               # untouched test split (102 contracts)
python scripts/evaluate.py --model bilstm --split val
```

Writes `artifacts/metrics/<model>_test.json` with span-extraction metrics, segment
classification metrics, contract-level clause detection, per-category results, the confusion
matrix, timing and the thresholds used. Thresholds are tuned on the validation split during
training; the test split is used only here.

## 9. Generating comparison reports

```bash
python scripts/generate_report.py                    # or: python scripts/compare.py
```

Produces `artifacts/metrics/comparison.{csv,json}`, `artifacts/reports/evaluation_report.md`
and charts in `artifacts/reports/figures/` (metric comparisons, per-category heatmap,
training curves, timing, one confusion matrix per model).

Command-line inference on any contract:

```bash
python scripts/infer.py path/to/contract.pdf --model all --json results.json
```

## 10. Running the backend

```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Interactive API docs: <http://127.0.0.1:8000/api/docs>. Health check:
`curl http://127.0.0.1:8000/api/health`. Configuration comes from environment variables or
`.env` (see `.env.example`): upload limit (`CLAUSEIQ_MAX_UPLOAD_MB`, default 15),
character limit, allowed CORS origins, SQLite path, model preloading and concurrency.

## 11. Running the frontend

Development (hot reload, proxies `/api` to port 8000):

```bash
cd frontend
npm run dev                  # http://localhost:5173
```

Production build served by the backend on one port:

```bash
cd frontend && npm run build && cd ..
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000     # http://localhost:8000
```

When `frontend/dist` exists, FastAPI serves it at `/` and the API under `/api`.

## 12. Using the application

* **Dashboard** — contracts analysed, clauses extracted, categories, model availability,
  measured test-set performance, recent activity.
* **Analyse a contract** — drag-and-drop a PDF or TXT, preview the extracted text, choose one
  or more models and all or specific categories, then *Extract clauses*.
* **Results reader** — the contract with extracted passages highlighted (by agreement
  strength across models, or per model), a clause list grouped by category with page
  numbers, character offsets, score vs threshold and — for BiLSTM + Attention — the context
  words that received the most attention. Click a highlight or a clause to jump between
  them. Download an HTML, CSV or JSON report.
* **Model comparison** — test-set metrics for all four models, and for an analysed contract:
  per-category counts by model, pairwise overlap, passages found by all / some / one model,
  and a per-model output switcher.
* **Clause categories** — the 41 official definitions, annotation counts and per-model test
  performance per category.
* **Model performance** — every metric, per-category span F1, training curves, clause
  detection confusion table and the segment confusion matrix.
* **Documentation** — pipeline, architectures, dataset, training configuration, metric
  definitions and limitations.

Text-based PDFs only: scanned PDFs are rejected with an explanatory message because
**OCR is not implemented**.

## 13. CPU and NVIDIA CUDA configuration

Training and inference pick CUDA automatically when `torch.cuda.is_available()`.

```bash
# CUDA 12.x wheels (choose the index matching your driver: cu118, cu121, cu124, cu126 …)
pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python scripts/train.py --model bilstm_attn      # logs device=cuda
```

On CPU, `--threads N` (training) and `CLAUSEIQ_TORCH_THREADS` (backend) set PyTorch's thread
count; by default PyTorch uses all physical cores. Shipped checkpoints were trained on CPU and
load on either device (`map_location="cpu"` at inference). To restrict GPUs:
`CUDA_VISIBLE_DEVICES=0`.

## 14. Deployment

**Docker (single container, frontend + API on port 8000):**

```bash
docker compose up --build -d          # or: docker build -t clauseiq . && docker run -p 8000:8000 clauseiq
```

The image installs CPU PyTorch, builds the frontend, copies `artifacts/` (checkpoints +
metrics) and runs uvicorn as a non-root user with a health check. Analysis history is stored
in the `clauseiq-data` volume. For CUDA, build with
`--build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124` on an NVIDIA base image and
run with `--gpus all`.

**Without Docker:** build the frontend, then run
`uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --workers 1` behind a reverse proxy
(nginx, Caddy) that terminates TLS and sets `client_max_body_size` ≥ the upload limit.
Use one worker per CPU-heavy instance (models are loaded per process; ~150 MB RAM each set);
scale horizontally with a shared volume for `CLAUSEIQ_DB_PATH` or one database per instance.
Set `CLAUSEIQ_ALLOWED_ORIGINS` only if the frontend is served from a different origin.

Security measures: extension and content (magic-byte / binary) checks, streamed upload size
limit, character limit, temporary files deleted after parsing, sanitised file names, no
uploaded file is ever executed, `X-Content-Type-Options: nosniff`, generic 500 messages
with details only in server logs. There is no authentication: put the app behind your SSO /
reverse-proxy auth if it is exposed beyond a trusted network.

## 15. Troubleshooting

| Symptom | Fix |
|---|---|
| `download_data.py` fails on Zenodo | It falls back to GitHub automatically. If both are blocked, download `data.zip` from the CUAD GitHub repository and place `CUADv1.json`, `category_descriptions.csv`, `train_separate_questions.json`, `test.json` in `data/raw/`. |
| `Model 'x' has no trained checkpoint` (HTTP 503) | Train it (`scripts/train.py --model x`) or restore `artifacts/checkpoints/`. |
| Upload rejected: "No extractable text … OCR" | The PDF is scanned. Export a text PDF or convert to .txt with your own OCR tool. |
| `413` on upload | Raise `CLAUSEIQ_MAX_UPLOAD_MB` (and your reverse proxy's body limit). |
| Frontend shows "Cannot reach the ClauseIQ API" | Start the backend on port 8000, or set `VITE_PROXY_TARGET` in `frontend/.env`. |
| Training is killed (out of memory) | Lower `--batch-size`; close other processes; 8 GB RAM recommended. |
| Training is slow on CPU | Use `--threads` = physical cores, `--neg-keep 0.15`, or a CUDA GPU. |
| `torch` install fails on Python 3.13 | Use torch ≥ 2.5, or Python 3.11/3.12. |
| Port already in use | `uvicorn … --port 8010` and set `VITE_PROXY_TARGET=http://127.0.0.1:8010`. |

## Tests

```bash
python -m pytest tests -q
```

Covers dataset parsing/validation/splitting, archive extraction, tokenization and offsets,
segmentation, vocabulary, label construction, decoding, every metric, all four
architectures (forward/backward, a learnable toy task, save/load), sliding windows, the
trained checkpoints, PDF/TXT parsing and rejection paths, and every API endpoint including a
full upload → analyse → report → delete flow. Tests needing data or checkpoints skip when
those are absent.

## Project layout

```
clauseiq/               ML package
  data/                 download, CUAD parsing/validation/statistics/split, prepare
  text/                 tokenizer (offsets + features), segmenter, vocabulary
  models/               tfidf_lr.py, neural.py (CNN, BiLSTM, BiLSTM+Attention)
  training/             feature cache, TF-IDF and neural training loops, threshold tuning
  inference/            sliding windows, unified predictor (used by evaluation and API)
  evaluation/           metrics, test-set evaluation
  reporting/            comparison tables and charts
backend/app/            FastAPI app: api/routes.py, services/ (documents, models, store,
                        comparison, reports), core/config.py
frontend/               React + TypeScript (Vite)
scripts/                command-line entry points (download … report, infer)
tests/                  pytest suite
artifacts/              checkpoints/, metrics/, reports/, dataset/ (shipped)
data/                   raw/ and processed/ (generated, git-ignored)
```

## Dataset attribution

This project uses the **Contract Understanding Atticus Dataset (CUAD) v1** by The Atticus
Project, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
<https://www.atticusprojectai.org/cuad> · <https://zenodo.org/records/4595826> ·
<https://github.com/TheAtticusProject/cuad>

> Dan Hendrycks, Collin Burns, Anya Chen, Spencer Ball. *CUAD: An Expert-Annotated NLP
> Dataset for Legal Contract Review.* NeurIPS 2021 Datasets and Benchmarks.

Category names and definitions shown in the app are reproduced from CUAD. Model checkpoints in
`artifacts/checkpoints/` are derived from CUAD training data. ClauseIQ is a review aid and does
not provide legal advice.
