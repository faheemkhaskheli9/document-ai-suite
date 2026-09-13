"""Tests for `classify_review.queue` -- issue #9."""
from __future__ import annotations

import pytest
from django.test import override_settings

from classify_review.queue import (
    NotAwaitingReviewError,
    ReviewQueueItem,
    accept,
    list_review_queue,
    reject,
)
from classify_review.services import validate_and_persist
from document_core.schema import ExtractedField, ExtractionResult
from documents.services import get_document_store


@pytest.fixture
def store_root(tmp_path):
    with override_settings(DOCUMENT_STORE_ROOT=str(tmp_path / "documents")):
        yield tmp_path


def _extraction(doc_id: str, confidence: float, fields=None) -> ExtractionResult:
    return ExtractionResult(
        document_id=doc_id,
        document_type="invoice",
        fields=fields if fields is not None else [ExtractedField(name="total", value="$1", confidence=confidence)],
        overall_confidence=confidence,
    )


def test_low_confidence_document_is_routed_to_the_queue(store_root):
    store = get_document_store()
    record = store.store(b"content", "a.pdf", owner="alice")
    validate_and_persist(record.id, classification=None, extraction=_extraction(record.id, 0.3))

    queue = list_review_queue(store)

    assert len(queue) == 1
    item = queue[0]
    assert isinstance(item, ReviewQueueItem)
    assert item.document.id == record.id
    assert item.validation.status == "needs_review"


def test_high_confidence_document_is_not_routed_to_the_queue(store_root):
    store = get_document_store()
    record = store.store(b"content", "a.pdf", owner="alice")
    validate_and_persist(record.id, classification=None, extraction=_extraction(record.id, 0.95))

    assert list_review_queue(store) == []


def test_failed_document_is_not_routed_to_the_review_queue(store_root):
    """A structurally-failed document (validation.py's `status="failed"`) is
    a different queue entirely -- issue #9's queue is specifically for
    below-threshold-but-valid documents, not broken ones."""
    store = get_document_store()
    record = store.store(b"content", "a.pdf", owner="alice")
    validate_and_persist(record.id, classification=None, extraction=_extraction(record.id, 0.3, fields=[]))

    assert list_review_queue(store) == []


def test_unvalidated_document_is_not_routed_to_the_queue(store_root):
    store = get_document_store()
    store.store(b"content", "a.pdf", owner="alice")

    assert list_review_queue(store) == []


def test_queue_spans_multiple_owners(store_root):
    store = get_document_store()
    alice_doc = store.store(b"one", "a.pdf", owner="alice")
    bob_doc = store.store(b"two", "b.pdf", owner="bob")
    validate_and_persist(alice_doc.id, None, _extraction(alice_doc.id, 0.2))
    validate_and_persist(bob_doc.id, None, _extraction(bob_doc.id, 0.4))

    queue_ids = {item.document.id for item in list_review_queue(store)}

    assert queue_ids == {alice_doc.id, bob_doc.id}


def test_accept_marks_document_auto_accepted(store_root):
    store = get_document_store()
    record = store.store(b"content", "a.pdf", owner="alice")
    validate_and_persist(record.id, None, _extraction(record.id, 0.3))

    result = accept(record.id)

    assert result.status == "auto_accepted"
    assert list_review_queue(store) == []


def test_reject_marks_document_failed(store_root):
    store = get_document_store()
    record = store.store(b"content", "a.pdf", owner="alice")
    validate_and_persist(record.id, None, _extraction(record.id, 0.3))

    result = reject(record.id)

    assert result.status == "failed"
    assert list_review_queue(store) == []


def test_accept_on_document_not_awaiting_review_raises(store_root):
    store = get_document_store()
    record = store.store(b"content", "a.pdf", owner="alice")
    validate_and_persist(record.id, None, _extraction(record.id, 0.95))  # auto_accepted already

    with pytest.raises(NotAwaitingReviewError, match="not awaiting review"):
        accept(record.id)


def test_accept_unknown_document_raises_keyerror(store_root):
    with pytest.raises(KeyError, match="No such document in the review queue"):
        accept("does-not-exist")
