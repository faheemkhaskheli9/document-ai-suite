"""Tests for `full_pipeline.pipeline` -- issue #10."""
from __future__ import annotations

import pytest

from classify_review.classifier import ClassificationResult, DocumentClassificationError
from document_core.extraction import ExtractionBackendError
from document_core.schema import ExtractedField, ExtractionResult
from full_pipeline.pipeline import (
    STAGE_CLASSIFICATION,
    STAGE_EXTRACTION,
    PipelineResult,
    run_full_pipeline,
)


def _classification(confidence=0.9):
    return ClassificationResult(is_scanned=False, document_type="invoice", confidence=confidence)


def _extraction(doc_id="doc-1", confidence=0.9):
    return ExtractionResult(
        document_id=doc_id,
        document_type="invoice",
        fields=[ExtractedField(name="total", value="$100", confidence=confidence)],
        overall_confidence=confidence,
    )


def test_pipeline_chains_all_three_stages_on_success(tmp_path, monkeypatch):
    path = tmp_path / "invoice.pdf"
    path.write_bytes(b"%PDF-fake")

    monkeypatch.setattr(
        "full_pipeline.pipeline.DocumentClassifier.classify",
        lambda self, p: _classification(),
    )
    monkeypatch.setattr(
        "full_pipeline.pipeline.extract_fields",
        lambda backend_key, document_path, document_id: _extraction(document_id),
    )

    result = run_full_pipeline("doc-1", path, "layout_ocr")

    assert isinstance(result, PipelineResult)
    assert result.classification.document_type == "invoice"
    assert result.extraction.document_id == "doc-1"
    assert result.validation.status == "auto_accepted"
    assert result.stage_errors == {}


def test_classification_failure_is_recorded_not_swallowed(tmp_path, monkeypatch):
    path = tmp_path / "invoice.pdf"
    path.write_bytes(b"%PDF-fake")

    def _raise(self, p):
        raise DocumentClassificationError("could not read PDF: corrupt")

    monkeypatch.setattr("full_pipeline.pipeline.DocumentClassifier.classify", _raise)
    monkeypatch.setattr(
        "full_pipeline.pipeline.extract_fields",
        lambda backend_key, document_path, document_id: _extraction(document_id),
    )

    result = run_full_pipeline("doc-1", path, "layout_ocr")

    assert result.classification is None
    assert STAGE_CLASSIFICATION in result.stage_errors
    assert "corrupt" in result.stage_errors[STAGE_CLASSIFICATION]
    # Extraction still ran despite classification failing -- one stage's
    # failure must not silently block an independent stage.
    assert result.extraction is not None


def test_extraction_failure_is_recorded_not_swallowed(tmp_path, monkeypatch):
    path = tmp_path / "invoice.pdf"
    path.write_bytes(b"%PDF-fake")

    monkeypatch.setattr(
        "full_pipeline.pipeline.DocumentClassifier.classify",
        lambda self, p: _classification(),
    )

    def _raise(backend_key, document_path, document_id):
        raise ExtractionBackendError("engine crashed")

    monkeypatch.setattr("full_pipeline.pipeline.extract_fields", _raise)

    result = run_full_pipeline("doc-1", path, "layout_ocr")

    assert result.extraction is None
    assert STAGE_EXTRACTION in result.stage_errors
    assert "engine crashed" in result.stage_errors[STAGE_EXTRACTION]
    # A missing extraction result is itself a validation failure -- the
    # pipeline never fabricates a passing result from a failed stage.
    assert result.validation.status == "failed"
    assert "no_extraction_result" in result.validation.failures


def test_both_stages_failing_still_returns_a_well_defined_result(tmp_path, monkeypatch):
    path = tmp_path / "invoice.pdf"
    path.write_bytes(b"%PDF-fake")

    def _raise_classify(self, p):
        raise DocumentClassificationError("bad file")

    def _raise_extract(backend_key, document_path, document_id):
        raise ExtractionBackendError("bad engine")

    monkeypatch.setattr("full_pipeline.pipeline.DocumentClassifier.classify", _raise_classify)
    monkeypatch.setattr("full_pipeline.pipeline.extract_fields", _raise_extract)

    result = run_full_pipeline("doc-1", path, "layout_ocr")

    assert result.classification is None
    assert result.extraction is None
    assert set(result.stage_errors) == {STAGE_CLASSIFICATION, STAGE_EXTRACTION}
    assert result.validation.status == "failed"


def test_low_confidence_result_needs_review(tmp_path, monkeypatch):
    path = tmp_path / "invoice.pdf"
    path.write_bytes(b"%PDF-fake")

    monkeypatch.setattr(
        "full_pipeline.pipeline.DocumentClassifier.classify",
        lambda self, p: _classification(confidence=0.3),
    )
    monkeypatch.setattr(
        "full_pipeline.pipeline.extract_fields",
        lambda backend_key, document_path, document_id: _extraction(document_id, confidence=0.3),
    )

    result = run_full_pipeline("doc-1", path, "layout_ocr")

    assert result.validation.status == "needs_review"


# -- contributing_stages() -- issue #11 --------------------------------------


def test_contributing_stages_is_empty_when_both_stages_are_confident(tmp_path, monkeypatch):
    path = tmp_path / "invoice.pdf"
    path.write_bytes(b"%PDF-fake")
    monkeypatch.setattr(
        "full_pipeline.pipeline.DocumentClassifier.classify",
        lambda self, p: _classification(confidence=0.9),
    )
    monkeypatch.setattr(
        "full_pipeline.pipeline.extract_fields",
        lambda backend_key, document_path, document_id: _extraction(document_id, confidence=0.9),
    )

    result = run_full_pipeline("doc-1", path, "layout_ocr")

    assert result.contributing_stages() == []


def test_contributing_stages_names_only_the_low_confidence_classification(tmp_path, monkeypatch):
    path = tmp_path / "invoice.pdf"
    path.write_bytes(b"%PDF-fake")
    monkeypatch.setattr(
        "full_pipeline.pipeline.DocumentClassifier.classify",
        lambda self, p: _classification(confidence=0.2),
    )
    monkeypatch.setattr(
        "full_pipeline.pipeline.extract_fields",
        lambda backend_key, document_path, document_id: _extraction(document_id, confidence=0.9),
    )

    result = run_full_pipeline("doc-1", path, "layout_ocr")

    assert result.contributing_stages() == [STAGE_CLASSIFICATION]


def test_contributing_stages_names_only_the_low_confidence_extraction(tmp_path, monkeypatch):
    path = tmp_path / "invoice.pdf"
    path.write_bytes(b"%PDF-fake")
    monkeypatch.setattr(
        "full_pipeline.pipeline.DocumentClassifier.classify",
        lambda self, p: _classification(confidence=0.9),
    )
    monkeypatch.setattr(
        "full_pipeline.pipeline.extract_fields",
        lambda backend_key, document_path, document_id: _extraction(document_id, confidence=0.2),
    )

    result = run_full_pipeline("doc-1", path, "layout_ocr")

    assert result.contributing_stages() == [STAGE_EXTRACTION]


def test_contributing_stages_includes_a_stage_that_failed_outright(tmp_path, monkeypatch):
    path = tmp_path / "invoice.pdf"
    path.write_bytes(b"%PDF-fake")

    def _raise(self, p):
        raise DocumentClassificationError("bad file")

    monkeypatch.setattr("full_pipeline.pipeline.DocumentClassifier.classify", _raise)
    monkeypatch.setattr(
        "full_pipeline.pipeline.extract_fields",
        lambda backend_key, document_path, document_id: _extraction(document_id, confidence=0.9),
    )

    result = run_full_pipeline("doc-1", path, "layout_ocr")

    assert result.contributing_stages() == [STAGE_CLASSIFICATION]
