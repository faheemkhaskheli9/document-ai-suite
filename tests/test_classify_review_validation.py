"""Tests for `classify_review.validation` -- issue #8."""
from __future__ import annotations

from classify_review.classifier import ClassificationResult
from classify_review.validation import (
    AUTO_ACCEPT_CONFIDENCE_THRESHOLD,
    ValidationResult,
    validate,
)
from document_core.schema import ExtractedField, ExtractionResult


def _extraction(**overrides) -> ExtractionResult:
    defaults = dict(
        document_id="doc-1",
        document_type="invoice",
        fields=[ExtractedField(name="total", value="$100", confidence=0.9)],
        overall_confidence=0.9,
    )
    defaults.update(overrides)
    return ExtractionResult(**defaults)


def test_missing_extraction_result_fails_validation():
    result = validate(classification=None, extraction=None)

    assert result.passed is False
    assert "no_extraction_result" in result.failures
    assert result.status == "failed"


def test_no_fields_extracted_fails_validation():
    extraction = _extraction(fields=[])

    result = validate(classification=None, extraction=extraction)

    assert result.passed is False
    assert "no_fields_extracted" in result.failures
    assert result.status == "failed"


def test_empty_field_value_fails_validation():
    extraction = _extraction(
        fields=[ExtractedField(name="vendor", value="   ", confidence=0.8)]
    )

    result = validate(classification=None, extraction=extraction)

    assert result.passed is False
    assert "empty_field_value" in result.failures
    assert result.status == "failed"


def test_document_type_mismatch_fails_validation():
    classification = ClassificationResult(is_scanned=False, document_type="contract", confidence=0.8)
    extraction = _extraction(document_type="invoice")

    result = validate(classification, extraction)

    assert result.passed is False
    assert "document_type_mismatch" in result.failures
    assert result.status == "failed"


def test_matching_document_type_does_not_fail():
    classification = ClassificationResult(is_scanned=False, document_type="Invoice", confidence=0.8)
    extraction = _extraction(document_type="invoice")

    result = validate(classification, extraction)

    assert result.passed is True
    assert result.failures == []


def test_high_confidence_valid_document_is_auto_accepted():
    extraction = _extraction(overall_confidence=0.95)

    result = validate(classification=None, extraction=extraction)

    assert isinstance(result, ValidationResult)
    assert result.passed is True
    assert result.status == "auto_accepted"
    assert result.overall_confidence >= AUTO_ACCEPT_CONFIDENCE_THRESHOLD


def test_low_confidence_valid_document_needs_review():
    extraction = _extraction(overall_confidence=0.3)

    result = validate(classification=None, extraction=extraction)

    assert result.passed is True
    assert result.status == "needs_review"


def test_failed_validation_is_distinct_from_low_confidence():
    """A structurally-broken result (issue #8 acceptance criterion 3) must
    never be reported the same way as a merely-low-confidence one, even if
    the failing document also happens to score low."""
    low_confidence_but_valid = validate(classification=None, extraction=_extraction(overall_confidence=0.3))
    invalid = validate(classification=None, extraction=_extraction(fields=[]))

    assert low_confidence_but_valid.status == "needs_review"
    assert invalid.status == "failed"
    assert low_confidence_but_valid.status != invalid.status


def test_overall_confidence_averages_classification_and_extraction():
    classification = ClassificationResult(is_scanned=False, document_type="invoice", confidence=0.6)
    extraction = _extraction(overall_confidence=1.0)

    result = validate(classification, extraction)

    assert result.overall_confidence == 0.8  # mean(0.6, 1.0)


def test_extraction_without_overall_confidence_falls_back_to_field_mean():
    extraction = _extraction(
        overall_confidence=None,
        fields=[
            ExtractedField(name="a", value="1", confidence=0.4),
            ExtractedField(name="b", value="2", confidence=0.6),
        ],
    )

    result = validate(classification=None, extraction=extraction)

    assert result.overall_confidence == 0.5


def test_result_round_trips_through_to_dict_and_from_dict():
    result = validate(classification=None, extraction=_extraction())

    restored = ValidationResult.from_dict(result.to_dict())

    assert restored == result
