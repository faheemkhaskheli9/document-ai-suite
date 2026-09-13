"""Persists `classify_review`'s validation output against a document (issue
#8 acceptance criterion: "Overall confidence score is computed and persisted
per document"), via the shared `document_core.storage.DocumentStore.save_artifact`.

Reuses `documents.services.get_document_store()` for the settings-backed
store lookup rather than re-reading `DOCUMENT_STORE_ROOT` here -- same reuse
`layout_ocr.services` already follows.
"""
from __future__ import annotations

from classify_review.classifier import ClassificationResult
from classify_review.validation import ValidationResult, validate
from document_core.schema import ExtractionResult
from documents.services import get_document_store

VALIDATION_ARTIFACT_NAME = "validation_result"


def validate_and_persist(
    doc_id: str,
    classification: ClassificationResult | None,
    extraction: ExtractionResult | None,
) -> ValidationResult:
    """Run `validate()` against `classification`/`extraction` and persist the
    result against `doc_id`.

    Raises `KeyError` if `doc_id` isn't a document that was actually
    uploaded, so a validation result can never be persisted against a
    document that doesn't exist.
    """
    store = get_document_store()
    if store.get(doc_id) is None:
        raise KeyError(f"No such document: {doc_id}")

    result = validate(classification, extraction)
    store.save_artifact(doc_id, VALIDATION_ARTIFACT_NAME, result.to_dict())
    return result


def get_persisted_validation(doc_id: str) -> ValidationResult | None:
    """Read back previously persisted validation output for `doc_id`, or
    `None` if `validate_and_persist` hasn't been run for it yet."""
    store = get_document_store()
    data = store.load_artifact(doc_id, VALIDATION_ARTIFACT_NAME)
    if data is None:
        return None
    return ValidationResult.from_dict(data)
