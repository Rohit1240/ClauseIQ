# Attribution

## Dataset

ClauseIQ is trained and evaluated on the **Contract Understanding Atticus Dataset (CUAD) v1**,
created by The Atticus Project and released under the
[Creative Commons Attribution 4.0 International licence (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

* Website: https://www.atticusprojectai.org/cuad
* Zenodo: https://zenodo.org/records/4595826
* GitHub: https://github.com/TheAtticusProject/cuad

Citation:

> Dan Hendrycks, Collin Burns, Anya Chen, Spencer Ball. *CUAD: An Expert-Annotated NLP Dataset
> for Legal Contract Review.* NeurIPS 2021 Datasets and Benchmarks Track. arXiv:2103.06268.

Derived material in this project: the 41 category names and definitions (shown in the app and
stored in `artifacts/dataset/categories.json`), dataset statistics, model checkpoints trained on
CUAD training contracts, and the two sample contracts in `samples/`. No changes were made to
CUAD's annotations; overlapping spans of the same category are merged only for evaluation.

## Software

Python: PyTorch, NumPy, SciPy, scikit-learn, matplotlib, FastAPI, Uvicorn, Pydantic, pypdf
(test-suite: pytest, httpx, reportlab). Frontend: React, React Router, Vite, TypeScript,
IBM Plex Sans and Source Serif 4 (SIL Open Font License, via Fontsource).

No pretrained models, embeddings or external AI/NLP APIs are used.
