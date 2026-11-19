PY ?= python3

.PHONY: install data train evaluate report test backend frontend build all

install:
	$(PY) -m pip install -r requirements-dev.txt
	cd frontend && npm ci

data:
	$(PY) scripts/download_data.py
	$(PY) scripts/validate_data.py
	$(PY) scripts/preprocess.py

train:
	$(PY) scripts/train.py --model all

evaluate:
	$(PY) scripts/evaluate.py --model all

report:
	$(PY) scripts/generate_report.py

test:
	$(PY) -m pytest tests -q

backend:
	$(PY) -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

frontend:
	cd frontend && npm run dev

build:
	cd frontend && npm run build

all: data train evaluate report
