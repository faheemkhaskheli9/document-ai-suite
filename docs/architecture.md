# Architecture Notes: Document AI Suite

## Pipeline

```text
Dashboard (pick a feature or run the full pipeline) ->
Document -> Classification (scanned vs digital, type) ->
  [layout_ocr engine: YOLO Layout Detection -> OCR (per detected region) -> Field Mapping]
  [llm_vision engine: multimodal LLM reads the page, returns structured fields directly]
  -> Validation + Confidence Scoring -> Auto-Accept or Human Review Queue
```

`layout_ocr` and `llm_vision` are alternative `document_core.extraction`
backends, chosen per run — see `document_core/extraction.py`'s docstring for
the two-shape contract (pipeline backend vs. direct backend). Both produce
the same `ExtractionResult`, so nothing downstream branches on which one ran.

## Components

- `document_core` — shared upload/storage, OCR backend wrapper (`ocr.py`,
  per-region text recognition), extraction backend wrapper (`extraction.py`,
  whole-document structured output — includes the `llm_vision` Claude-vision
  backend today), structured JSON schema
- `layout_ocr` feature app — ported from `document-ai-yolo-ocr`: YOLO layout
  detection (`detection.py`'s `LayoutDetector`), per-region OCR
  (`fields.py`'s `extract_fields_from_regions`, via `document_core.ocr`;
  a region that fails OCR is flagged with `ocr_error`, never dropped),
  invoice/contract field mapping to the shared `ExtractionResult` schema
  (`mapping.py`, regex heuristics for invoice number/date/total/vendor, no
  trained NER model); `services.py`'s `run_layout_ocr` chains detect ->
  per-region OCR and persists both artifacts per document via
  `document_core.storage`'s `save_artifact`/`load_artifact`; `backend.py`
  registers the whole pipeline as the `"layout_ocr"` `document_core.extraction`
  backend (`LayoutOcrConfig.ready()` triggers the registration)
- `classify_review` feature app — ported from `intelligent-document-processing`:
  document classifier, validation rules, confidence scoring, human-review
  queue; consumes whichever extraction backend's `ExtractionResult` it's
  given, unaware of which one produced it
- `full_pipeline` mode — chains the selected extraction backend's output into
  `classify_review`'s validation/confidence/review-routing stage

## Design Notes

- Registry pattern for feature apps *and* for backends one layer below them
  (OCR engines in `ocr.py`, extraction engines in `extraction.py`) — mirrors
  `medical-imaging-suite`'s `BaseImagingTask` / `@register_task`.
- `llm_vision` calls the Anthropic API (`anthropic` SDK) with
  `output_config`'s JSON-schema structured output so the response is
  guaranteed syntactically valid — the backend still validates the *content*
  (via `pydantic`) before trusting it, since a 200 response only guarantees
  shape, not correctness. Model confidence there is a self-report, not a
  calibrated probability like Tesseract's per-word confidence.
- Reuse `document-ai-yolo-ocr`'s existing `src/yolo_ocr/dataset.py` COCO ->
  YOLO conversion CLI for layout-model dataset prep rather than
  re-implementing it.
- Target datasets: PubLayNet / DocLayNet (CDLA-Permissive-1.0, COCO-format),
  not vendored — `examples/sample_layout_coco.json` +
  `examples/sample_images/` (from `document-ai-yolo-ocr`) stay as the tiny
  synthetic fixture for demoing the conversion CLI without the real dataset.
  `llm_vision` needs no training dataset at all.
