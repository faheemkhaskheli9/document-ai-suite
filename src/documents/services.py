"""Thin adapter between this Django app and the framework-agnostic
`document_core.storage.DocumentStore` (README.md Section 2: `document_core`
shared upload/storage layer). No Django import leaks into `document_core`
itself -- it stays reusable by feature apps built outside Django too.

`get_document_store()` re-reads `settings.DOCUMENT_STORE_ROOT` on every call
instead of caching a single instance at import time: `DocumentStore.__init__`
is just a cheap `mkdir`, and caching it would freeze whatever
`DOCUMENT_STORE_ROOT` was on the *first* call -- breaking test isolation via
`override_settings` (robustness rule 11/12: nothing import-time, nothing
memoized on hidden settings state).

The Django `User.pk` (not `username`, which a user can change) is used as
`document_core`'s `owner` string -- a stable id so a renamed account can't
suddenly stop matching its own past uploads or, worse, collide with a new
account that reused the old username.
"""
from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.contrib.auth.models import AbstractBaseUser

from document_core.extraction import extract_fields
from document_core.schema import ExtractionResult
from document_core.storage import DocumentStore

# Artifact name `run_extraction` persists its result under (issue #14
# acceptance criterion: "the chosen engine is recorded against the
# document/job so results are traceable to how they were produced").
EXTRACTION_ARTIFACT_NAME = "extraction_result"


def get_document_store() -> DocumentStore:
    return DocumentStore(settings.DOCUMENT_STORE_ROOT)


def owner_id(user: AbstractBaseUser) -> str:
    return str(user.pk)


def run_extraction(doc_id: str, backend_key: str) -> ExtractionResult:
    """Run `backend_key` against `doc_id`'s stored file, through the one
    `document_core.extraction.extract_fields` interface -- no feature-app
    code here imports a concrete engine (`layout_ocr`/`llm_vision`)
    directly, so adding a third engine never touches this function.

    Persists the result *and* which engine produced it, so a later view of
    this document can show how its fields were produced. Raises `KeyError`
    if `doc_id` isn't a document that was actually uploaded.
    """
    store = get_document_store()
    record = store.get(doc_id)
    if record is None:
        raise KeyError(f"No such document: {doc_id}")

    result = extract_fields(backend_key, Path(record.stored_path), doc_id)
    store.save_artifact(
        doc_id,
        EXTRACTION_ARTIFACT_NAME,
        {"backend_key": backend_key, "result": result.model_dump()},
    )
    return result


def get_extraction_result(doc_id: str) -> dict | None:
    """Read back the persisted `{"backend_key", "result"}` for `doc_id`, or
    `None` if no extraction has been run for it yet."""
    store = get_document_store()
    return store.load_artifact(doc_id, EXTRACTION_ARTIFACT_NAME)
