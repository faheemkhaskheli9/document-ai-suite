# Document AI Suite

> Document AI & OCR portfolio project — independent open-source implementation.
> This is an original, from-scratch build. It is not affiliated with, and does not
> contain any code, prompts, data, or business logic from, any employer or client.

![status](https://img.shields.io/badge/status-planned-lightgrey)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

## Combines

This is a flagship suite that will combine the 2 Document AI/OCR repos into
one web app with both features selectable from a single dashboard UI — the
same combined-suite pattern already used in this portfolio for
`video-analytics-suite`, `medical-imaging-suite`, and `trading-ai-suite`:

- [`document-ai-yolo-ocr`](../document-ai-yolo-ocr/) — YOLO layout detection + OCR + field mapping to structured JSON (invoices/contracts/forms)
- [`intelligent-document-processing`](../intelligent-document-processing/) — document classification, unified parsing/OCR, validation rules, confidence scoring, human-review routing for low-confidence extractions

`intelligent-document-processing`'s own Phase 2 already names
`document-ai-yolo-ocr` as the pipeline it builds on, so this suite formalizes
that relationship as one app instead of a cross-repo dependency. The 2
originals will get an archived banner + `status-archived` badge and move to
`E:\Projects\portfolio-archived-repos\` once this suite reaches feature
parity with each of them — no code or git history is deleted, only relocated.

## 1. Problem

Extracting structured fields from a document needs layout detection + OCR
(`document-ai-yolo-ocr`) *and* the classification/validation/confidence/
human-review scaffolding around it (`intelligent-document-processing`) to be
production-usable — splitting them into 2 repos means the layout+OCR engine
has no review queue, and the review queue has no real extraction engine to
gate. One app, one pipeline.

## 2. Architecture

```text
Dashboard (pick a feature or run the full pipeline) ->
Document -> Classification (scanned vs digital, type) ->
YOLO Layout Detection -> OCR (per detected region) -> Field Mapping (structured JSON) ->
Validation + Confidence Scoring -> Auto-Accept or Human Review Queue
```

A shared `document_core` layer (upload/storage, OCR backend wrapper, the
structured-JSON schema) sits underneath 2 feature apps — one wrapping
`document-ai-yolo-ocr`'s layout+OCR+field-mapping pipeline, one wrapping
`intelligent-document-processing`'s classify+validate+review pipeline —
registered the same registry pattern used by this portfolio's other suites,
plus a combined "full pipeline" mode that chains both.

## 3. Technology Stack

- Python, Django 5.x (feature-picker web app + job/review-queue history)
- Ultralytics YOLO (layout detection), Tesseract or PaddleOCR (text extraction)
- OpenCV, FastAPI-style validators for field/confidence rules
- PostgreSQL in production, SQLite for local/dev (`DATABASE_URL` override)

## 4. Feature List

- **Layout + OCR extraction** (from `document-ai-yolo-ocr`): document layout
  detection, field-level detection, OCR text extraction, structured JSON
  output, invoice/contract parsing, table/field extraction
- **Classify + Review** (from `intelligent-document-processing`): document
  classification, PDF/scanned handling, validation rules, confidence scoring,
  human-review routing queue for low-confidence extractions
- **Full pipeline**: chain the two — classify, run layout+OCR, validate/score,
  route to review only when confidence is low
- Shared: one document upload path, per-user job/review history

## 5. Implementation Plan

1. Phase 1: `document_core` shared app (upload/storage, OCR backend wrapper,
   structured-JSON schema) + Django project skeleton with the dashboard shell
2. Phase 2: Port `document-ai-yolo-ocr`'s layout-detection + OCR + field-mapping
   flow into a feature app (reuse its existing dataset-prep CLI /
   `src/yolo_ocr/dataset.py` conversion logic where it fits `document_core`)
3. Phase 3: Port `intelligent-document-processing`'s classifier + validation +
   confidence scoring + review-queue flow into a second feature app
4. Phase 4: Wire the "full pipeline" mode that chains both feature apps
5. Phase 5: Archive the 2 original repos (banner + badge, move to
   `portfolio-archived-repos`) once parity is confirmed

## Task Tracking

Work will be broken into phase-tagged user stories tracked as GitHub Issues,
not in this file. Implement Phase 1 issues first (later phases depend on it).
When you start one, add label `status:in-progress`. When you finish, close it
referencing the commit (e.g. `git commit -m "... Closes #4"`) and push.

## 6. Repository Structure

```text
document-ai-suite/
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml
├── .env.example
├── docker/
├── docs/
│   ├── architecture.md
│   └── evaluation.md
├── src/
├── tests/
├── configs/
├── scripts/
├── notebooks/
├── examples/
├── assets/
└── .github/
    └── workflows/
```

## 7. Setup

```bash
git clone <this-repo-url>
cd document-ai-suite
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt   # or: pip install -e .
cp .env.example .env              # fill in API keys / config
```

## 8. Dataset

Target datasets are PubLayNet or DocLayNet (CDLA-Permissive-1.0, COCO-format
document layouts), same as `document-ai-yolo-ocr`; not vendored in this repo.
No proprietary, employer-owned, or client-identifiable data is used in this
project.

## 9. Training / Execution

```bash
# Once Phase 1 lands:
python manage.py migrate
python manage.py runserver   # open http://127.0.0.1:8000/ and pick a feature
```

## 10. Evaluation

Document evaluation metrics and how to reproduce them here (see
`docs/evaluation.md`).

## 11. Results

_To be filled in as the implementation progresses — screenshots, metrics
tables, and sample outputs go here._

## 12. API

_If this project exposes an API, document the main endpoints here (or link to
auto-generated OpenAPI docs, e.g. `/docs` for FastAPI)._

## 13. Docker

```bash
docker build -t document-ai-suite .
docker run -p 8000:8000 document-ai-suite
```

## 14. Tests

```bash
pytest tests/
```

## 15. Limitations

- This is a from-scratch, independent recreation built for portfolio purposes.
- Performance numbers, once added, are based on public datasets and are not
  representative of any production system's real-world results.
- Scaffold stage: no code has been ported from the 2 source repos yet — see
  §5 Implementation Plan.

## 16. Future Work

- Port each source repo's existing Phase-1 code (where present) rather than
  rewriting from scratch.
- Expand evaluation coverage and add CI-based regression checks.
- Track open items as GitHub Issues.

## 17. Disclosure

This repository is an **independent open-source recreation inspired by the
kind of production systems I have worked on professionally**. It contains no
employer or client source code, prompts, datasets, credentials, architecture
diagrams, or business logic. All code, data, and documentation here are
original or built on publicly available datasets and open-source tools.

---
_Last updated: 2026-09-12_
