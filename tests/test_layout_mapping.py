"""Tests for `layout_ocr.mapping`'s field mapping -- issue #6."""
from __future__ import annotations

from document_core.schema import BoundingBox, ExtractionResult
from layout_ocr.detection import DetectedRegion
from layout_ocr.fields import FieldExtraction
from layout_ocr.mapping import map_to_extraction_result


def _region(class_name: str) -> DetectedRegion:
    return DetectedRegion(
        class_name=class_name, confidence=0.9, bbox=BoundingBox(x=0, y=0, width=10, height=10)
    )


def test_maps_to_extraction_result_shape():
    extractions = [
        FieldExtraction(region=_region("text"), text="hello world", confidence=0.8)
    ]

    result = map_to_extraction_result("doc-1", extractions)

    assert isinstance(result, ExtractionResult)
    assert result.document_id == "doc-1"
    assert any(f.name == "text" and f.value == "hello world" for f in result.fields)


def test_recognizes_invoice_key_value_fields():
    extractions = [
        FieldExtraction(
            region=_region("text"), text="Invoice Number: INV-1001", confidence=0.9
        ),
        FieldExtraction(region=_region("text"), text="Invoice Date: 2024-05-01", confidence=0.9),
        FieldExtraction(region=_region("text"), text="Vendor: Acme Corp", confidence=0.9),
        FieldExtraction(region=_region("text"), text="Total Due: $1,234.56", confidence=0.9),
    ]

    result = map_to_extraction_result("doc-1", extractions)

    by_name = {f.name: f.value for f in result.fields if f.name != "text"}
    assert by_name["invoice_number"] == "INV-1001"
    assert by_name["date"] == "2024-05-01"
    assert by_name["vendor"] == "Acme Corp"
    assert by_name["total"] == "1,234.56"


def test_vendor_pattern_does_not_swallow_the_rest_of_the_line():
    """A single region's OCR text carrying more than one line must not let
    the greedy-looking vendor pattern consume text belonging to a different
    field."""
    extractions = [
        FieldExtraction(
            region=_region("text"),
            text="Vendor: Acme Corp Total Due: $1,234.56",
            confidence=0.9,
        )
    ]

    result = map_to_extraction_result("doc-1", extractions)

    by_name = {f.name: f.value for f in result.fields}
    assert by_name["vendor"] == "Acme Corp"
    assert by_name["total"] == "1,234.56"


def test_a_region_with_no_recognizable_pattern_only_yields_the_raw_field():
    extractions = [FieldExtraction(region=_region("text"), text="just some text", confidence=0.5)]

    result = map_to_extraction_result("doc-1", extractions)

    assert len(result.fields) == 1
    assert result.fields[0].name == "text"


def test_failed_region_is_excluded_and_flags_needs_review():
    extractions = [
        FieldExtraction(region=_region("text"), text="Vendor: Acme Corp", confidence=0.9),
        FieldExtraction(region=_region("table"), text=None, confidence=None, ocr_error="boom"),
    ]

    result = map_to_extraction_result("doc-1", extractions)

    assert all(f.name != "table" for f in result.fields)
    assert result.status == "needs_review"


def test_no_extractions_yields_no_fields_and_none_confidence():
    result = map_to_extraction_result("doc-1", [])
    assert result.fields == []
    assert result.overall_confidence is None
    assert result.status == "processing"
