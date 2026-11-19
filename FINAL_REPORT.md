# ClauseIQ — Final Report

ClauseIQ is a working contract-analysis application: a FastAPI backend and a React/TypeScript
frontend around four clause-extraction models trained from scratch on CUAD v1. This report
gives the dataset statistics, how the models were trained, their measured performance on the
untouched test split, the test-suite results and the known limitations. Every number in the
tables below is generated from `artifacts/` by `scripts/build_final_report.py`; none was
typed by hand.

## 1. Summary

* **Everything requested was built and run**: dataset download/validation/preprocessing,
  four independently trained models (TF-IDF + LR, dilated CNN, BiLSTM, BiLSTM + Attention),
  evaluation on the untouched 102-contract CUAD test split, comparison reports and charts, a
  FastAPI backend, a React/TypeScript frontend, Docker configuration, a 54-test pytest suite
  (all passing) and a browser-driven end-to-end check with a real PDF.
* **Best span extraction: BiLSTM** — span F1 0.437 (IoU ≥ 0.5), exact-match F1 0.207,
  character IoU 0.383 on the test split. BiLSTM + Attention is close (0.428), TF-IDF + LR
  reaches 0.353 and the dilated CNN 0.304.
* **The traditional baseline remains strong.** TF-IDF + LR has the best segment-classification
  macro-F1 (0.496) and contract-level clause-detection F1 (0.833), ties the BiLSTM on span
  macro-F1 (0.383) and has the best (or tied-best) span F1 in the most categories: 21 of the 39
  where any model scores above zero (BiLSTM 12, BiLSTM + Attention 9, ties counted for each), mainly long clauses.
  The neural taggers win overall span F1 because they can return short answers (parties,
  dates, document names) that a sentence-level model cannot isolate.
* **Absolute scores are modest.** These are small models trained from random initialisation on
  347 contracts with CPU-only budgets; they are far below published fine-tuned transformer
  results on CUAD, which use pretraining that this project's rules exclude.
* All training ran on 2 CPU threads; there was no GPU in the build environment.

## 2. Dataset

Source: the official CUAD v1 release. Zenodo (record 4595826) is the first source in
`scripts/download_data.py`; in the build environment Zenodo was blocked by the network policy,
so the downloader fell back automatically to the official GitHub repository
(`TheAtticusProject/cuad`, `data.zip` + `category_descriptions.csv`), which contains the same
`CUADv1.json` annotations and the official train/test partition. SHA-256 checksums of the files
used are recorded in `data/raw/manifest.json` when the downloader runs.

Validation (`scripts/validate_data.py`): 510 contracts, 41 categories matched one-to-one to the
official CSV names and definitions, all 13,823 answer offsets reproduce their text exactly, no
duplicate contracts, no contract shared between splits.

| Item | Value |
|---|---|
| Contracts | 510 |
| Categories | 41 |
| Annotated clause spans | 13,823 |
| Split (contracts) | train 347 / val 61 / test 102 — official CUAD test partition + seeded validation split of official train |
| Spans per split | train 9,520 / val 1,660 / test 2,643 |
| Contract length (chars) | median 33,143.0, mean 52,563.0, min 645, max 338,211 |
| Total characters | 26,807,133 |
| Characters inside a clause | 11.1% |
| (contract, category) pairs without a clause | 14,208 of 20,910 (67.9%) |
| Class imbalance | max 2,554 vs min 27 spans (ratio 94.6×, Gini 0.486) |
| Tokens | train 3,558,693 / val 534,186 / test 882,360 |
| Vocabulary (train only, min freq 2) | 16,224 (val OOV 2.45%, test OOV 2.24%) |
| Tokens inside a clause | train 10.9% |
| Segments | train 108,074 (8,268 labelled) / test 26,774 (2,255 labelled) |

Per-category annotation counts (all 510 contracts):

| Category | Spans | Contracts with clause | Train / Val / Test spans | Median length (chars) |
|---|---:|---:|---|---:|
| Document Name | 521 | 510 (100.0%) | 357 / 62 / 102 | 25 |
| Parties | 2,554 | 509 (99.8%) | 1678 / 333 / 543 | 17 |
| Agreement Date | 476 | 470 (92.2%) | 320 / 63 / 93 | 16 |
| Effective Date | 447 | 390 (76.5%) | 312 / 51 / 84 | 20 |
| Expiration Date | 467 | 413 (81.0%) | 324 / 60 / 83 | 195 |
| Renewal Term | 210 | 176 (34.5%) | 157 / 22 / 31 | 270 |
| Notice Period to Terminate Renewal | 122 | 111 (21.8%) | 90 / 14 / 18 | 298 |
| Governing Law | 464 | 437 (85.7%) | 319 / 55 / 90 | 178 |
| Most Favored Nation | 38 | 28 (5.5%) | 30 / 5 / 3 | 379 |
| Non-Compete | 259 | 119 (23.3%) | 163 / 37 / 59 | 309 |
| Exclusivity | 410 | 180 (35.3%) | 262 / 70 / 78 | 300 |
| No-Solicit of Customers | 58 | 34 (6.7%) | 40 / 3 / 15 | 258 |
| Competitive Restriction Exception | 125 | 76 (14.9%) | 83 / 15 / 27 | 366 |
| No-Solicit of Employees | 91 | 59 (11.6%) | 60 / 13 / 18 | 350 |
| Non-Disparagement | 65 | 38 (7.5%) | 50 / 3 / 12 | 226 |
| Termination for Convenience | 246 | 183 (35.9%) | 177 / 28 / 41 | 163 |
| Rofr/Rofo/Rofn | 367 | 85 (16.7%) | 263 / 36 / 68 | 328 |
| Change of Control | 253 | 121 (23.7%) | 178 / 13 / 62 | 304 |
| Anti-Assignment | 654 | 374 (73.3%) | 445 / 72 / 137 | 218 |
| Revenue/Profit Sharing | 418 | 166 (32.5%) | 282 / 49 / 87 | 267 |
| Price Restrictions | 27 | 15 (2.9%) | 18 / 9 / 0 | 282 |
| Minimum Commitment | 424 | 165 (32.4%) | 292 / 44 / 88 | 275 |
| Volume Restriction | 171 | 82 (16.1%) | 111 / 25 / 35 | 233 |
| IP Ownership Assignment | 318 | 124 (24.3%) | 231 / 26 / 61 | 337 |
| Joint IP Ownership | 115 | 46 (9.0%) | 95 / 6 / 14 | 299 |
| License Grant | 777 | 255 (50.0%) | 549 / 93 / 135 | 356 |
| Non-Transferable License | 298 | 138 (27.1%) | 216 / 39 / 43 | 351 |
| Affiliate License-Licensor | 69 | 23 (4.5%) | 42 / 7 / 20 | 491 |
| Affiliate License-Licensee | 115 | 59 (11.6%) | 72 / 16 / 27 | 484 |
| Unlimited/All-You-Can-Eat-License | 32 | 17 (3.3%) | 24 / 2 / 6 | 342 |
| Irrevocable or Perpetual License | 165 | 70 (13.7%) | 122 / 20 / 23 | 489 |
| Source Code Escrow | 66 | 13 (2.5%) | 61 / 0 / 5 | 252 |
| Post-Termination Services | 450 | 182 (35.7%) | 317 / 51 / 82 | 368 |
| Audit Rights | 643 | 214 (42.0%) | 465 / 73 / 105 | 253 |
| Uncapped Liability | 167 | 111 (21.8%) | 132 / 19 / 16 | 423 |
| Cap on Liability | 672 | 275 (53.9%) | 466 / 88 / 118 | 306 |
| Liquidated Damages | 121 | 61 (12.0%) | 82 / 16 / 23 | 289 |
| Warranty Duration | 176 | 75 (14.7%) | 121 / 36 / 19 | 261 |
| Insurance | 560 | 166 (32.5%) | 383 / 60 / 117 | 244 |
| Covenant Not to Sue | 173 | 100 (19.6%) | 105 / 24 / 44 | 335 |
| Third Party Beneficiary | 39 | 32 (6.3%) | 26 / 2 / 11 | 205 |

**Absent clauses vs missing annotations.** CUAD marks a category that does not occur in a
contract with an empty answer (SQuAD 2.0 *is_impossible*). These are true negatives, not missing
labels; no contract in CUAD has missing annotation records.

**Class imbalance.** Parties has ~95× more spans than Price Restrictions. All models use
per-category handling (class-balanced logistic regression; positive-class weights and
validation-tuned per-category thresholds for the neural models).

## 3. Method

**Split (contract-level, reproducible).** Test = the official CUAD test partition (102
contracts), never used for training, early stopping, threshold or decoding selection.
Validation = 61 contracts sampled with seed 13 from the official training partition; train =
the remaining 347.

**Preprocessing.** Offset-preserving regex tokenizer (lower-casing, digit normalisation, token
shape, newline/paragraph flags, relative and log-absolute position features); vocabulary of
16,224 words built from the training split only; abbreviation-aware segmentation into
sentence-like units (max 900 characters) for the TF-IDF model and segment metrics. Gold spans of
the same category that overlap are merged for evaluation.

**Models.**

| Model | Architecture |
|---|---|
| TF-IDF + LR | word 1–3-gram TF-IDF of each segment (80k features) + 1–2-gram TF-IDF of neighbouring segments (30k) + 21 position/length features → 41 L2 logistic regressions (class_weight=balanced). C ∈ {1, 4} chosen on validation. |
| Dilated CNN | embeddings (word 128 + shape/position 3×16) → linear 192 → 6 residual gated dilated Conv1d blocks (dilations 1,2,4,8,16,1; receptive field ≈ 65 tokens) → 41 sigmoids per token |
| BiLSTM | same embeddings → 2-layer BiLSTM (160 per direction) → 41 sigmoids per token |
| BiLSTM + Attention | BiLSTM as above → 4-head self-attention over the 256-token window + residual/LayerNorm → feed-forward block → 41 sigmoids per token. Attention weights are exposed in the app as each span's "attention focus" words. |

All weights and embeddings are randomly initialised; nothing pretrained is used.

**Neural training.** 256-token windows with a random offset each epoch; every window containing
a clause plus 25% of clause-free windows; masked BCE with positive weights
clip(√(neg/pos), 1, 10); AdamW (lr 1.5e-3 CNN/attention, 2e-3 BiLSTM, weight decay 0.01),
gradient clipping 1.0, dropout 0.3, 5% word dropout, ReduceLROnPlateau (×0.5, patience 1), early
stopping on validation span F1 (best of four global thresholds) and best-epoch restore.
Inference uses overlapping windows (stride 128) with centre-weighted averaging, so contracts of
any length are processed.

**Decoding (validation-only).** For the neural models a search on the 61 validation contracts
chooses a smoothing radius and gap-merge distance and, per category (≥ 8 validation spans,
otherwise a pooled setting), the threshold, the minimum span length and the mode: *token* spans
or *segment* spans (a sentence segment is selected when the mean of the model's token
probabilities inside it reaches the threshold). This was added after observing that raw token
decoding fragments long clauses: on validation, CNN false positives had a median length of 43
characters versus 231 for correct spans. The TF-IDF model uses per-category thresholds on
segments. All selected settings are stored in the checkpoints and in
`artifacts/metrics/*_training.json`.

## 4. Training details

| Model | Parameters / features | Epochs run (best) | Decoding (val-tuned) | Val span F1 (tuned) | Training time |
|---|---:|---|---|---:|---:|
| TF-IDF + Logistic Regression | 110,021 features × 41 | C ∈ [1.0, 4.0] → C=4.0 | segments, per-category thresholds | 0.404 | 17 min 14 s |
| Dilated CNN | 3,451,209 | 16 (12) | smooth r=0, gap=3; per category threshold, min length and mode (19 of 41 in segment mode) | 0.372 | 36 min 49 s |
| BiLSTM | 3,140,361 | 20 (20) | smooth r=0, gap=3; per category threshold, min length and mode (14 of 41 in segment mode) | 0.501 | 44 min 10 s |
| BiLSTM + Attention | 3,963,081 | 24 (23) | smooth r=0, gap=3; per category threshold, min length and mode (20 of 41 in segment mode) | 0.484 | 96 min 42 s |

Notes on the runs (all wall-clock on 2 CPU threads):

* **TF-IDF + LR** trained two full models (C = 1 and C = 4); C = 4 had the higher validation
  span F1 (0.404 vs 0.393).
* **Dilated CNN** stopped early at epoch 16 (patience 4) with epoch 12 restored. It was trained
  twice with the same seed: a first run (patience 3) stopped at epoch 15 with the same best
  epoch, and the shipped run reproduced its per-epoch validation scores exactly, which confirms
  that more patience would not have helped.
* **BiLSTM** reached the 20-epoch cap while validation F1 was still rising (best = last epoch);
  more epochs would probably improve it. Its decoding search was run as a separate step with
  `scripts/tune_decoding.py` (27 s, included in the training time) because the training run
  started before the search space was extended; the procedure is identical to the one built
  into training.
* **BiLSTM + Attention** ran until its 95-minute budget (24 epochs, best epoch 23). Each epoch
  costs about 2× a BiLSTM epoch, so it saw fewer updates per minute.
* Budgets were not identical across models (CPU time was the constraint). Running two training
  jobs at once on the 2-core machine slowed both by more than 10× (OpenMP oversubscription), so
  affected runs were discarded and repeated alone; all reported training and inference times
  come from uncontended runs.
* Validation span F1 in this table uses the tuned decoding; the per-epoch history uses the best
  of four global thresholds, so the two are not directly comparable.

Training histories: `artifacts/metrics/<model>_training.json`; curves:
`artifacts/reports/figures/training_curves.png`.

## 5. Test-set results (102 contracts, never seen during training or tuning)

Three families of metrics are kept separate. *Span extraction* is the actual task: a predicted
span counts when it matches a gold clause of the same category with character IoU ≥ 0.5
(one-to-one matching); *Exact Match* requires identical boundaries after trimming whitespace and
edge punctuation; *character IoU* compares the highlighted characters. *Segment classification*
only checks category labels on sentence-like segments (92% of segments contain no clause, which
inflates plain accuracy). *Contract-level detection* asks whether a contract contains a clause
type at all. Best value per row in bold.

| Metric | TF-IDF + Logistic Regression | Dilated CNN | BiLSTM | BiLSTM + Attention |
|---|---:|---:|---:|---:|
| **Span extraction** | | | | |
| Span precision (IoU ≥ 0.5) | 0.402 | 0.312 | **0.486** | 0.463 |
| Span recall (IoU ≥ 0.5) | 0.315 | 0.297 | 0.396 | **0.398** |
| **Span F1 (IoU ≥ 0.5)** | 0.353 | 0.304 | **0.437** | 0.428 |
| Span macro-F1 (over categories) | 0.383 | 0.243 | **0.383** | 0.380 |
| Exact Match precision | 0.196 | 0.100 | **0.231** | 0.222 |
| Exact Match recall (= EM rate) | 0.154 | 0.095 | 0.188 | **0.191** |
| Exact Match F1 | 0.172 | 0.097 | **0.207** | 0.205 |
| Character IoU (micro) | 0.365 | 0.236 | **0.383** | 0.367 |
| Character IoU (macro) | 0.357 | 0.213 | **0.359** | 0.351 |
| Mean best IoU per gold span | 0.337 | 0.298 | 0.403 | **0.405** |
| **Segment classification** | | | | |
| Accuracy (exact label set) | **0.924** | 0.891 | 0.922 | 0.921 |
| Accuracy on gold-labelled segments | 0.409 | 0.305 | 0.421 | **0.428** |
| Precision (micro) | **0.571** | 0.349 | 0.566 | 0.547 |
| Recall (micro) | **0.547** | 0.445 | 0.533 | 0.542 |
| Macro F1 | **0.496** | 0.320 | 0.489 | 0.483 |
| Weighted F1 | **0.566** | 0.424 | 0.545 | 0.538 |
| **Contract-level clause detection** | | | | |
| Accuracy | 0.902 | 0.846 | **0.903** | 0.897 |
| Precision | 0.843 | 0.755 | **0.867** | 0.846 |
| Recall | **0.823** | 0.715 | 0.795 | 0.797 |
| F1 | **0.833** | 0.734 | 0.829 | 0.821 |
| Macro F1 | 0.716 | 0.565 | **0.747** | 0.717 |
| **Cost (CPU, 2 threads)** | | | | |
| Training time | 17 min 14 s | 36 min 49 s | 44 min 10 s | 96 min 42 s |
| Inference time per test contract | 0.025 s | 0.246 s | 0.213 s | 0.432 s |

**Reading the results.**

* *Short answers decide the micro span F1.* Parties alone accounts for 486 of the 2,571 merged
  test gold spans. TF-IDF + LR scores 0.02 there (it must return whole sentences), while the
  BiLSTM scores 0.44; on Agreement Date the gap is 0.04 vs 0.77. Most of the BiLSTM's
  overall advantage comes from such categories.
* *Long clauses favour the sparse baseline.* TF-IDF + LR is best on, for example, Anti-Assignment
  (0.64), Cap on Liability (0.58), Covenant Not to Sue (0.63), IP Ownership Assignment (0.43) and
  Joint IP Ownership (0.55). With 347 training contracts, n-gram features with strong
  regularisation generalise better than ~3–4 M neural parameters on rare clause types.
* *Attention did not beat the plain BiLSTM* overall (0.428 vs 0.437 span F1; slightly higher
  recall and mean best IoU). It did help some categories (Irrevocable or Perpetual License 0.71
  vs 0.58, Third Party Beneficiary 0.43 vs 0.09, Expiration Date 0.71 vs 0.67) and is the most
  expensive model to train and run.
* *The CNN is the weakest* on every span metric: its ≈65-token receptive field and lack of
  sequential state fit long clauses poorly, and its validation F1 plateaued by epoch 12.
* *Exact match is hard for all models* (≤ 0.21 F1). CUAD boundaries are often a clause's first
  words through a mid-sentence cut-off, which neither tokens nor sentence segments reproduce
  exactly. IoU-based scores are the more informative extraction measure here.
* *Classification vs extraction.* Segment-level accuracy is 0.89–0.92 for every model only
  because ~92% of segments contain no clause; macro/weighted F1 and the span metrics are the
  meaningful numbers. Contract-level clause detection (does the contract contain a clause of
  this type?) is much easier than locating the clause: F1 0.73–0.83.
* *Speed.* TF-IDF + LR processes a test contract in ~25 ms; the neural models need 0.21–0.43 s
  on 2 CPU threads, including sliding-window inference over contracts of up to 338k characters.
* Price Restrictions has no gold span in the test split, so its per-category scores are 0 by
  construction (any prediction is a false positive); it is excluded from the macro averages.

Charts: `artifacts/reports/figures/span_extraction_comparison.png`,
`classification_comparison.png`, `timing_comparison.png`, `per_category_span_f1.png`,
`confusion_<model>.png`. Full per-category metrics and confusion matrices are in
`artifacts/metrics/<model>_test.json` and on the app's *Model performance* page.

### Per-category span F1 (IoU ≥ 0.5)

| Category | Test gold spans | TF-IDF + Logistic Regression | Dilated CNN | BiLSTM | BiLSTM + Attention |
|---|---:|---:|---:|---:|---:|
| Document Name | 102 | 0.12 | 0.14 | 0.35 | **0.38** |
| Parties | 486 | 0.02 | 0.34 | **0.44** | 0.41 |
| Agreement Date | 93 | 0.04 | 0.74 | **0.77** | 0.75 |
| Effective Date | 84 | 0.11 | 0.35 | **0.42** | 0.34 |
| Expiration Date | 83 | 0.62 | 0.50 | 0.67 | **0.71** |
| Renewal Term | 31 | **0.63** | 0.55 | 0.58 | 0.59 |
| Notice Period to Terminate Renewal | 18 | **0.67** | 0.45 | 0.54 | 0.65 |
| Governing Law | 90 | 0.82 | 0.67 | **0.87** | 0.81 |
| Most Favored Nation | 3 | 0.00 | 0.00 | **0.50** | 0.29 |
| Non-Compete | 57 | 0.39 | 0.26 | **0.41** | 0.31 |
| Exclusivity | 78 | **0.41** | 0.23 | 0.31 | 0.28 |
| No-Solicit of Customers | 15 | **0.42** | 0.00 | 0.16 | 0.15 |
| Competitive Restriction Exception | 27 | **0.12** | 0.03 | 0.04 | 0.08 |
| No-Solicit of Employees | 18 | 0.47 | 0.23 | **0.67** | 0.34 |
| Non-Disparagement | 12 | 0.15 | 0.00 | 0.21 | **0.24** |
| Termination for Convenience | 41 | **0.45** | 0.24 | 0.41 | **0.45** |
| Rofr/Rofo/Rofn | 68 | **0.36** | 0.21 | 0.25 | 0.18 |
| Change of Control | 62 | **0.38** | 0.22 | **0.38** | 0.33 |
| Anti-Assignment | 135 | **0.64** | 0.47 | 0.56 | 0.57 |
| Revenue/Profit Sharing | 86 | **0.36** | 0.25 | 0.28 | 0.31 |
| Price Restrictions | 0 | 0.00 | 0.00 | 0.00 | 0.00 |
| Minimum Commitment | 88 | **0.28** | 0.15 | 0.21 | 0.24 |
| Volume Restriction | 34 | 0.29 | 0.08 | **0.31** | 0.17 |
| IP Ownership Assignment | 61 | **0.43** | 0.23 | 0.22 | 0.22 |
| Joint IP Ownership | 14 | **0.55** | 0.00 | 0.39 | 0.27 |
| License Grant | 133 | **0.63** | 0.40 | 0.59 | 0.59 |
| Non-Transferable License | 43 | 0.33 | 0.18 | 0.38 | **0.39** |
| Affiliate License-Licensor | 20 | **0.37** | 0.00 | 0.06 | 0.28 |
| Affiliate License-Licensee | 26 | **0.42** | 0.10 | 0.21 | 0.24 |
| Unlimited/All-You-Can-Eat-License | 5 | 0.00 | 0.00 | 0.00 | 0.00 |
| Irrevocable or Perpetual License | 23 | 0.58 | 0.22 | 0.58 | **0.71** |
| Source Code Escrow | 5 | 0.29 | 0.00 | 0.18 | **0.33** |
| Post-Termination Services | 80 | 0.29 | 0.14 | **0.31** | **0.31** |
| Audit Rights | 104 | **0.52** | 0.39 | 0.48 | 0.45 |
| Uncapped Liability | 16 | 0.53 | 0.47 | **0.58** | 0.51 |
| Cap on Liability | 117 | **0.58** | 0.34 | 0.48 | 0.54 |
| Liquidated Damages | 23 | **0.51** | 0.22 | 0.29 | 0.36 |
| Warranty Duration | 19 | 0.26 | 0.24 | **0.41** | 0.30 |
| Insurance | 116 | **0.36** | 0.27 | 0.31 | 0.35 |
| Covenant Not to Sue | 44 | **0.63** | 0.42 | 0.41 | 0.34 |
| Third Party Beneficiary | 11 | 0.27 | 0.00 | 0.09 | **0.43** |

## 6. Testing and quality assurance

Automated suite (`python -m pytest tests -q`), run against the real data and the four trained
checkpoints:

```
tests/test_api.py::test_txt_extraction_and_decoding PASSED
tests/test_api.py::test_pdf_extraction_pages PASSED
tests/test_api.py::test_rejections[a.exe-MZ....-415] PASSED
tests/test_api.py::test_rejections[a.pdf-not a pdf at all-415] PASSED
tests/test_api.py::test_rejections[a.pdf-%PDF-1.4\n garbage-422] PASSED
tests/test_api.py::test_rejections[a.txt--422] PASSED
tests/test_api.py::test_rejections[a.txt-short-422] PASSED
tests/test_api.py::test_rejections[a.txt-\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f-415] PASSED
tests/test_api.py::test_rejections[a.txt-%PDF-1.4 disguised-415] PASSED
tests/test_api.py::test_scanned_pdf_rejected_with_ocr_message PASSED
tests/test_api.py::test_char_limit PASSED
tests/test_api.py::test_reflow_preserves_offsets PASSED
tests/test_api.py::test_health_and_metadata PASSED
tests/test_api.py::test_upload_validation PASSED
tests/test_api.py::test_unknown_model_and_category PASSED
tests/test_api.py::test_full_analysis_flow_all_models PASSED
tests/test_data.py::test_parse_validate_stats_split PASSED
tests/test_data.py::test_validation_detects_bad_offsets PASSED
tests/test_data.py::test_validation_detects_leakage PASSED
tests/test_data.py::test_gold_spans_merge_overlaps PASSED
tests/test_data.py::test_zip_member_extraction PASSED
tests/test_data.py::test_real_cuad_processed PASSED
tests/test_metrics_decoding.py::test_runs_merge_small_gaps PASSED
tests/test_metrics_decoding.py::test_tokens_to_spans_offsets_and_score PASSED
tests/test_metrics_decoding.py::test_segments_to_spans_merges_consecutive PASSED
tests/test_metrics_decoding.py::test_iou_and_matching PASSED
tests/test_metrics_decoding.py::test_normalise_bounds PASSED
tests/test_metrics_decoding.py::test_span_accumulator_perfect_and_empty PASSED
tests/test_metrics_decoding.py::test_span_accumulator_partial_overlap PASSED
tests/test_metrics_decoding.py::test_segment_accumulator PASSED
tests/test_metrics_decoding.py::test_contract_accumulator PASSED
tests/test_metrics_decoding.py::test_smoothing_and_min_tokens PASSED
tests/test_metrics_decoding.py::test_decode_hybrid_modes PASSED
tests/test_models.py::test_neural_forward_backward[cnn] PASSED
tests/test_models.py::test_neural_forward_backward[bilstm] PASSED
tests/test_models.py::test_neural_forward_backward[bilstm_attn] PASSED
tests/test_models.py::test_attention_weights_exposed PASSED
tests/test_models.py::test_neural_learns_toy_task PASSED
tests/test_models.py::test_sliding_windows_cover_long_documents PASSED
tests/test_models.py::test_tfidf_fit_predict_save_load PASSED
tests/test_models.py::test_trained_checkpoint_inference[tfidf_lr] PASSED
tests/test_models.py::test_trained_checkpoint_inference[cnn] PASSED
tests/test_models.py::test_trained_checkpoint_inference[bilstm] PASSED
tests/test_models.py::test_trained_checkpoint_inference[bilstm_attn] PASSED
tests/test_models.py::test_tokenizer_matches_checkpoint_vocab PASSED
tests/test_text.py::test_token_offsets_roundtrip PASSED
tests/test_text.py::test_token_features_in_range PASSED
tests/test_text.py::test_normalisation PASSED
tests/test_text.py::test_empty_text PASSED
tests/test_text.py::test_segments_cover_text_and_respect_abbreviations PASSED
tests/test_text.py::test_long_segment_is_split PASSED
tests/test_text.py::test_segment_labels_rule PASSED
tests/test_text.py::test_token_labels PASSED
tests/test_text.py::test_vocab_build_encode_save_load PASSED
============================== 54 passed in 6.15s ==============================
```

Integration checks run in addition to pytest (headless Chrome via Playwright against the
production build served by FastAPI on one port):

* Uploaded a 19-page PDF generated from CUAD test contract
  *MusclepharmCorp … Co-Branding Agreement* (87,438 characters) through the drag-and-drop
  input; the preview, page count and character count were shown.
* Ran all four models on all 41 categories: 227 clauses returned in 5.0 s
  (TF-IDF 61 ms, CNN 0.6 s, BiLSTM 0.7 s, BiLSTM + Attention 3.7 s including attention
  explanations); switching the reader to BiLSTM + Attention highlighted its 53 spans and showed
  attention-focus words; clicking a clause scrolled to and flashed its highlight.
* Every returned clause's text equals `document_text[start:end]` (asserted in the API test and
  checked in the smoke test); page numbers fall inside the document's page range.
* Downloaded the CSV report (`ClauseIQ_Musclepharm_Co-Branding_Agreement.csv`); HTML and JSON
  reports are covered by the API test.
* Visited Dashboard, Analyses, Model comparison (with the analysed contract), Clause categories
  (expanded a row), Model performance and Documentation; no JavaScript errors.
* Error paths: a `.docx` is rejected in the browser before upload; an image-only PDF is rejected
  by the server with the "no extractable text … does not perform OCR" message. The only console
  entry was the expected HTTP 422 for that rejection.
* `npm run build` (TypeScript strict check + Vite production build) succeeds:
  ~239 kB JS (75 kB gzipped).
* Not run in this environment: `docker build` (no Docker daemon available). The Dockerfile and
  compose file are provided but untested here.

## 7. Known limitations and incomplete work

* **Modest absolute accuracy.** Best span F1 is 0.437. The models are useful for pointing a
  reviewer at candidate clauses, not for unattended extraction. Rare categories (fewer than ~30
  training spans: e.g. Unlimited/All-You-Can-Eat License, Most Favored Nation, Source Code
  Escrow, No-Solicit of Customers) are unreliable.
* **Training was budget-limited on CPU.** The BiLSTM was still improving at its 20-epoch cap;
  larger budgets or a GPU would likely raise the neural scores. No systematic hyperparameter
  search was run for the neural models (one configuration each, chosen a priori); the TF-IDF
  model searched C ∈ {1, 4} only.
* **Decoding was tuned on 61 validation contracts.** Per-category thresholds, minimum lengths and
  token/segment modes can overfit for categories with few validation examples (categories with
  fewer than 8 validation spans share a pooled setting).
* **Segment mode inherits segmentation errors.** The sentence splitter is rule-based; unusual
  layouts (tables, numbered lists without punctuation, merged PDF columns) produce imperfect
  segments, which bounds TF-IDF and segment-mode spans.
* **No OCR.** Scanned PDFs are rejected with an explanation. PDF text extraction (pypdf) plus an
  offset-preserving line reflow is used; layout-heavy PDFs may extract text in a different order
  than CUAD's text files, which is a distribution shift for the models.
* **Scores are not calibrated probabilities.** The app labels them as model scores relative to
  validation-tuned thresholds.
* **Neural retraining is not bit-exact across machines/thread counts**; the shipped checkpoints
  are the reference.
* **Deployment:** the Docker image was not built here (no Docker daemon). The app has no
  authentication or multi-user isolation; it is intended for a trusted network or behind an
  authenticating reverse proxy. SQLite suits a single instance.
* **Data source:** Zenodo was unreachable from the build environment, so the identical data was
  taken from the official CUAD GitHub repository via the downloader's fallback.
* ClauseIQ is a review aid and not legal advice.

## 8. Reproduction

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements-dev.txt
python scripts/download_data.py && python scripts/validate_data.py && python scripts/preprocess.py
python scripts/train.py --model tfidf_lr
python scripts/train.py --model cnn --epochs 30 --patience 4 --time-budget 60
python scripts/train.py --model bilstm --epochs 20 --time-budget 65
python scripts/train.py --model bilstm_attn --epochs 30 --patience 4 --time-budget 95
python scripts/evaluate.py --model all
python scripts/generate_report.py && python scripts/build_final_report.py
```

Neural training is seeded but not bit-exact across machines or thread counts (PyTorch CPU
kernels are not deterministic across thread counts), so retrained numbers will differ
slightly from the shipped checkpoints.

Dataset: CUAD v1 by The Atticus Project, CC BY 4.0 — Hendrycks et al., *CUAD: An
Expert-Annotated NLP Dataset for Legal Contract Review*, NeurIPS 2021.
