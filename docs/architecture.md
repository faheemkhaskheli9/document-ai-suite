# Architecture Notes: Document AI Suite

## Pipeline

```text
Dashboard (pick a feature or run the full pipeline) ->
Document -> Classification (scanned vs digital, type) ->
YOLO Layout Detection -> OCR (per detected region) -> Field Mapping (structured JSON) ->
Validation + Confidence Scoring -> Auto-Accept or Human Review Queue
```

## Components

- `document_core` — shared upload/storage, OCR backend wrapper, structured
  JSON schema
- `layout_ocr` feature app — ported from `document-ai-yolo-ocr`: YOLO layout
  detection, per-region OCR, field mapping to structured JSON
- `classify_review` feature app — ported from `intelligent-document-processing`:
  document classifier, validation rules, confidence scoring, human-review
  queue
- `full_pipeline` mode — chains `layout_ocr` output into `classify_review`'s
  validation/confidence/review-routing stage

## Design Notes

- Registry pattern for feature apps (mirrors `medical-imaging-suite`'s
  `BaseImagingTask` / `@register_task`).
- Reuse `document-ai-yolo-ocr`'s existing `src/yolo_ocr/dataset.py` COCO ->
  YOLO conversion CLI for layout-model dataset prep rather than
  re-implementing it.
- Target datasets: PubLayNet / DocLayNet (CDLA-Permissive-1.0, COCO-format),
  not vendored — `examples/sample_layout_coco.json` +
  `examples/sample_images/` (from `document-ai-yolo-ocr`) stay as the tiny
  synthetic fixture for demoing the conversion CLI without the real dataset.
