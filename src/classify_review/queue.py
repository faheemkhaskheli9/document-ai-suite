"""
Human-review routing queue -- issue #9 (Phase 3, `classify_review` feature
app).

Surfaces only documents whose persisted `classify_review.validation` result
is `status="needs_review"` -- i.e. below `AUTO_ACCEPT_CONFIDENCE_THRESHOLD`
but not structurally failed (see `validation.py`'s status distinction) --
and lets a reviewer accept or reject one.

This module only reads/writes state `validate_and_persist` (issue #8) has
already written; it never re-runs classification or extraction itself.
"""
from __future__ import annotations

from dataclasses import dataclass

from classify_review.services import (
    get_persisted_validation,
    overwrite_validation_status,
)
from classify_review.validation import ValidationResult
from document_core.storage import DocumentRecord, DocumentStore
from documents.services import get_document_store


@dataclass
class ReviewQueueItem:
    """One document awaiting human review, paired with why it's here."""

    document: DocumentRecord
    validation: ValidationResult


class NotAwaitingReviewError(ValueError):
    """Raised by `accept`/`reject` when the document's current validation
    status isn't `needs_review` -- reviewing an already-decided document
    (or one that never needed review) is a caller bug, not something to
    silently allow."""


def list_review_queue(store: DocumentStore | None = None) -> list[ReviewQueueItem]:
    """Every uploaded document currently awaiting human review, across all
    owners (issue #9 acceptance criteria 1/2: only below-threshold documents
    are surfaced; at/above-threshold documents are not).

    Ordered oldest-uploaded first, so a reviewer works the queue in the order
    documents arrived.
    """
    store = store or get_document_store()
    items = []
    for record in store.list_all():
        validation = get_persisted_validation(record.id)
        if validation is not None and validation.status == "needs_review":
            items.append(ReviewQueueItem(document=record, validation=validation))
    return sorted(items, key=lambda item: item.document.uploaded_at)


def accept(doc_id: str) -> ValidationResult:
    """Reviewer accepts a queued document: overrides its status to
    `auto_accepted` without recomputing confidence -- a human decision
    stands regardless of the original score."""
    return _decide(doc_id, "auto_accepted")


def reject(doc_id: str) -> ValidationResult:
    """Reviewer rejects a queued document: overrides its status to
    `failed`, same distinct-from-low-confidence status validation.py uses
    for a structurally-broken result, since a rejected document also
    shouldn't be treated as usable output."""
    return _decide(doc_id, "failed")


def _decide(doc_id: str, status: str) -> ValidationResult:
    current = get_persisted_validation(doc_id)
    if current is None:
        raise KeyError(f"No such document in the review queue: {doc_id}")
    if current.status != "needs_review":
        raise NotAwaitingReviewError(
            f"document {doc_id} is not awaiting review (status={current.status})"
        )
    return overwrite_validation_status(doc_id, status)
