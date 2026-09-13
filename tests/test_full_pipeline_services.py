"""Tests for `full_pipeline.services` -- issue #10."""
from __future__ import annotations

import pytest
from django.test import override_settings

from classify_review.classifier import ClassificationResult
from document_core.schema import ExtractedField, ExtractionResult
from documents.services import get_document_store
from full_pipeline.services import get_persisted_pipeline_result, run_pipeline_and_persist

pytestmark = pytest.mark.django_db


@pytest.fixture
def store_root(tmp_path):
    with override_settings(DOCUMENT_STORE_ROOT=str(tmp_path / "documents")):
        yield tmp_path


def _stub_pipeline(monkeypatch, status="auto_accepted"):
    confidence = 0.95 if status == "auto_accepted" else 0.3

    def fake_extract_fields(backend_key, document_path, document_id):
        return ExtractionResult(
            document_id=document_id,
            document_type="invoice",
            fields=[ExtractedField(name="total", value="$5", confidence=0.9)],
            overall_confidence=confidence,
        )

    monkeypatch.setattr("full_pipeline.pipeline.extract_fields", fake_extract_fields)
    monkeypatch.setattr(
        "full_pipeline.pipeline.DocumentClassifier.classify",
        lambda self, p: ClassificationResult(
            is_scanned=False, document_type="invoice", confidence=confidence
        ),
    )


def test_run_pipeline_and_persist_writes_the_artifact(store_root, monkeypatch):
    _stub_pipeline(monkeypatch)
    store = get_document_store()
    record = store.store(b"%PDF-fake", "invoice.pdf", owner="alice")

    result = run_pipeline_and_persist(record.id, "layout_ocr")

    assert result.validation.status == "auto_accepted"
    persisted = get_persisted_pipeline_result(record.id)
    assert persisted["extraction_backend_key"] == "layout_ocr"
    assert persisted["extraction"]["fields"][0]["name"] == "total"
    assert persisted["validation"]["status"] == "auto_accepted"
    assert persisted["stage_errors"] == {}


def test_get_persisted_pipeline_result_is_none_before_a_run(store_root):
    store = get_document_store()
    record = store.store(b"%PDF-fake", "invoice.pdf", owner="alice")

    assert get_persisted_pipeline_result(record.id) is None


def test_run_pipeline_unknown_document_raises_keyerror(store_root):
    with pytest.raises(KeyError, match="No such document"):
        run_pipeline_and_persist("does-not-exist", "layout_ocr")
