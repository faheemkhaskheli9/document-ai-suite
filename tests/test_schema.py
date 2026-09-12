import pytest
from pydantic import ValidationError

from document_core.schema import BoundingBox, ExtractedField, ExtractionResult


def test_extraction_result_defaults():
    result = ExtractionResult(document_id="doc-1")
    assert result.status == "uploaded"
    assert result.fields == []
    assert result.document_type is None


def test_extraction_result_with_fields_round_trips():
    result = ExtractionResult(
        document_id="doc-1",
        document_type="invoice",
        status="needs_review",
        fields=[
            ExtractedField(
                name="total",
                value="123.45",
                confidence=0.42,
                bbox=BoundingBox(x=10, y=20, width=100, height=15),
            )
        ],
        overall_confidence=0.42,
    )
    payload = result.model_dump_json()
    reloaded = ExtractionResult.model_validate_json(payload)
    assert reloaded.fields[0].name == "total"
    assert reloaded.fields[0].bbox.width == 100


def test_confidence_out_of_range_is_rejected():
    with pytest.raises(ValidationError):
        ExtractedField(name="total", value="1", confidence=1.5)


def test_bbox_requires_positive_dimensions():
    with pytest.raises(ValidationError):
        BoundingBox(x=0, y=0, width=0, height=10)
