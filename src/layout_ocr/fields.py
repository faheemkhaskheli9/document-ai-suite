"""
Per-region OCR for `layout_ocr` -- issue #5 ("Port field-level detection and
OCR text extraction"). Runs `document_core.ocr.extract_text` against each
`DetectedRegion` a `layout_ocr.detection.LayoutDetector` found, so a caller
gets one OCR result per field instead of a single OCR dump of the whole
page. Field mapping to the shared `document_core.schema.ExtractionResult`
shape is a later phase's concern (README.md Section 5) -- this module only
produces the per-region text/confidence pairing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from document_core.ocr import OCRBackendError, extract_text
from layout_ocr.detection import DetectedRegion


@dataclass(frozen=True)
class FieldExtraction:
    """One detected region's OCR result.

    `text`/`confidence` are `None` when OCR failed on this specific region
    (`ocr_error` holds why) -- distinct from OCR succeeding with empty text,
    so a region that couldn't be read is never confused with a field that
    genuinely has no text.
    """

    region: DetectedRegion
    text: str | None
    confidence: float | None
    ocr_error: str | None = None

    @property
    def failed(self) -> bool:
        return self.ocr_error is not None

    def to_dict(self) -> dict:
        return {
            "region": self.region.to_dict(),
            "text": self.text,
            "confidence": self.confidence,
            "ocr_error": self.ocr_error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FieldExtraction":
        return cls(
            region=DetectedRegion.from_dict(data["region"]),
            text=data["text"],
            confidence=data["confidence"],
            ocr_error=data.get("ocr_error"),
        )


def extract_fields_from_regions(
    image_path: str | Path,
    regions: list[DetectedRegion],
    ocr_backend_key: str = "tesseract",
) -> list[FieldExtraction]:
    """Run OCR against each region's bbox on `image_path`.

    A region whose OCR call raises `OCRBackendError` (including
    `OCRBackendUnavailable`) is returned with `ocr_error` set rather than
    being dropped from the result list -- the caller decides how to flag it
    (e.g. route to the human-review queue in a later phase) instead of the
    field silently vanishing.
    """
    image_path = Path(image_path)
    results: list[FieldExtraction] = []
    for region in regions:
        try:
            ocr_result = extract_text(ocr_backend_key, image_path, region.bbox)
        except OCRBackendError as exc:
            results.append(
                FieldExtraction(region=region, text=None, confidence=None, ocr_error=str(exc))
            )
            continue
        results.append(
            FieldExtraction(
                region=region, text=ocr_result.text, confidence=ocr_result.confidence
            )
        )
    return results
