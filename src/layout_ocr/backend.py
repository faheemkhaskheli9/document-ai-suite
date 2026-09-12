"""
Registers `layout_ocr`'s detect -> per-region OCR -> field-mapping pipeline
as the `"layout_ocr"` `document_core.extraction` backend -- issue #6's
acceptance criterion that this feature app's output "validates against the
shared structured-JSON schema", and docs/architecture.md: "`layout_ocr` and
`llm_vision` are alternative `document_core.extraction` backends, chosen per
run ... [b]oth produce the same `ExtractionResult`".

Imported from `layout_ocr.apps.LayoutOcrConfig.ready()` rather than at
`layout_ocr` package import time, so the registration side effect only runs
once Django's app registry is actually loading this app (never as a
by-product of some unrelated module reaching into `layout_ocr`).
"""
from __future__ import annotations

from pathlib import Path

from document_core.extraction import (
    ExtractionBackend,
    ExtractionBackendError,
    ExtractionBackendUnavailable,
    register_extraction_backend,
)
from document_core.schema import ExtractionResult
from layout_ocr.detection import LayoutDetectionError, LayoutDetectionUnavailable, LayoutDetector
from layout_ocr.fields import extract_fields_from_regions
from layout_ocr.mapping import map_to_extraction_result


@register_extraction_backend
class LayoutOCRExtractionBackend(ExtractionBackend):
    """The layout-detection-driven alternative to `llm_vision`: YOLO layout
    detection + per-region OCR + invoice/contract field mapping, running
    entirely locally (given a trained checkpoint) with no per-document API
    call."""

    key = "layout_ocr"
    label = "Layout + OCR (YOLO + Tesseract)"

    def __init__(
        self, detector: LayoutDetector | None = None, ocr_backend_key: str = "tesseract"
    ):
        # `register_extraction_backend` instantiates this with no arguments
        # (mirrors `document_core.ocr`/`document_core.extraction`'s registry
        # pattern) -- both params exist so tests can inject a fake detector
        # without touching the registry.
        self._detector = detector
        self._ocr_backend_key = ocr_backend_key

    def extract(self, document_path: Path, document_id: str) -> ExtractionResult:
        document_path = Path(document_path)
        if not document_path.exists():
            raise ExtractionBackendError(f"No such document: {document_path}")

        detector = self._detector or LayoutDetector()
        try:
            regions = detector.detect(document_path)
        except LayoutDetectionUnavailable as exc:
            raise ExtractionBackendUnavailable(str(exc)) from exc
        except LayoutDetectionError as exc:
            raise ExtractionBackendError(str(exc)) from exc

        field_extractions = extract_fields_from_regions(
            document_path, regions, self._ocr_backend_key
        )
        return map_to_extraction_result(document_id, field_extractions)
