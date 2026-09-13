"""Tests for `classify_review.services`'s validation persistence -- issue #8
acceptance criterion: "Overall confidence score is computed and persisted per
document"."""
from __future__ import annotations

import pytest
from django.test import override_settings

from classify_review.classifier import ClassificationResult
from classify_review.services import get_persisted_validation, validate_and_persist
from document_core.schema import ExtractedField, ExtractionResult
from documents.services import get_document_store


@pytest.fixture
def store_root(tmp_path):
    with override_settings(DOCUMENT_STORE_ROOT=str(tmp_path / "documents")):
        yield tmp_path


@pytest.fixture
def uploaded_doc(store_root):
    store = get_document_store()
    return store.store(b"%PDF-1.4 test", "invoice.pdf", owner="alice")


def test_validate_and_persist_writes_and_returns_result(uploaded_doc):
    classification = ClassificationResult(is_scanned=False, document_type="invoice", confidence=0.8)
    extraction = ExtractionResult(
        document_id=uploaded_doc.id,
        document_type="invoice",
        fields=[ExtractedField(name="total", value="$100", confidence=0.9)],
        overall_confidence=0.9,
    )

    result = validate_and_persist(uploaded_doc.id, classification, extraction)

    assert result.passed is True
    assert result.status == "auto_accepted"
    assert get_persisted_validation(uploaded_doc.id) == result


def test_get_persisted_validation_before_run_returns_none(uploaded_doc):
    assert get_persisted_validation(uploaded_doc.id) is None


def test_validate_and_persist_unknown_document_raises_keyerror(store_root):
    with pytest.raises(KeyError, match="No such document"):
        validate_and_persist("does-not-exist", classification=None, extraction=None)


def test_validate_and_persist_stores_failed_status_distinctly(uploaded_doc):
    extraction = ExtractionResult(document_id=uploaded_doc.id, fields=[])

    result = validate_and_persist(uploaded_doc.id, classification=None, extraction=extraction)

    assert result.status == "failed"
    assert get_persisted_validation(uploaded_doc.id).status == "failed"
