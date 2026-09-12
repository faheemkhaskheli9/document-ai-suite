"""
Field mapping for `layout_ocr` -- issue #6 ("Add invoice/contract parsing
output format"). Maps `layout_ocr.fields`'s per-region OCR output into the
shared `document_core.schema.ExtractionResult` shape, and lifts common
invoice/contract key-value fields (invoice number, date, total, vendor) out
of plain OCR'd text via lightweight regex heuristics -- no trained NER model,
matching this backend's "no training data beyond the layout detector"
design goal (docs/architecture.md).
"""
from __future__ import annotations

import re

from document_core.schema import ExtractedField, ExtractionResult
from layout_ocr.fields import FieldExtraction

_KEY_VALUE_CONFIDENCE = 0.6

# Recognized invoice/contract key-value patterns: "<label>[:\s]*<value>"
# somewhere in one region's OCR'd text, case-insensitive. Each value capture
# stops at the next recognized label (or end of string) rather than
# consuming greedily to the end of the text, since a "text" region can carry
# more than one line of OCR'd content.
_NEXT_LABEL_LOOKAHEAD = r"(?=\s+(?:invoice|date|total|vendor)\b|$)"
_KEY_VALUE_PATTERNS: dict[str, re.Pattern[str]] = {
    "invoice_number": re.compile(
        r"invoice\s*(?:number|no\.?|#)\s*[:\-]?\s*(?P<value>\S+)", re.I
    ),
    "date": re.compile(
        r"(?:invoice\s*)?date\s*[:\-]?\s*(?P<value>\d{1,4}[/-]\d{1,2}[/-]\d{1,4})", re.I
    ),
    "total": re.compile(
        r"total(?:\s*(?:amount|due))?\s*[:\-]?\s*\$?\s*(?P<value>[\d,]+\.\d{2})", re.I
    ),
    "vendor": re.compile(
        rf"vendor\s*[:\-]?\s*(?P<value>.+?){_NEXT_LABEL_LOOKAHEAD}", re.I
    ),
}


def _extract_key_value_fields(text: str) -> list[ExtractedField]:
    """Pull recognized invoice/contract key-value pairs out of one region's
    OCR'd text. More than one pattern may match the same text -- each match
    becomes its own field."""
    fields = []
    for name, pattern in _KEY_VALUE_PATTERNS.items():
        match = pattern.search(text)
        if match:
            value = match.group("value").strip()
            if value:
                fields.append(
                    ExtractedField(name=name, value=value, confidence=_KEY_VALUE_CONFIDENCE)
                )
    return fields


def map_to_extraction_result(
    document_id: str, field_extractions: list[FieldExtraction]
) -> ExtractionResult:
    """Build the shared `ExtractionResult` from `layout_ocr`'s per-region OCR
    output.

    Every successfully-OCR'd region becomes a plain `ExtractedField` named
    after its detected layout class (e.g. "text", "table"), carrying that
    region's `bbox` so a caller can still locate it on the page; text that
    also matches a recognized invoice/contract key-value pattern
    additionally contributes a named field (e.g. "total", "vendor"). A
    region that failed OCR contributes no field -- it lowers
    `overall_confidence` by its absence rather than being counted as
    present-but-empty -- and pushes `status` to `needs_review` so a failed
    region is never silently treated as a clean extraction.
    """
    fields: list[ExtractedField] = []
    any_failed = False
    for extraction in field_extractions:
        if extraction.failed:
            any_failed = True
            continue
        if not extraction.text:
            continue
        fields.append(
            ExtractedField(
                name=extraction.region.class_name,
                value=extraction.text,
                confidence=extraction.confidence or 0.0,
                bbox=extraction.region.bbox,
            )
        )
        fields.extend(_extract_key_value_fields(extraction.text))

    overall_confidence = (
        sum(f.confidence for f in fields) / len(fields) if fields else None
    )
    return ExtractionResult(
        document_id=document_id,
        fields=fields,
        overall_confidence=overall_confidence,
        status="needs_review" if any_failed else "processing",
    )
