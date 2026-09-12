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

from django.conf import settings
from django.contrib.auth.models import AbstractBaseUser

from document_core.storage import DocumentStore


def get_document_store() -> DocumentStore:
    return DocumentStore(settings.DOCUMENT_STORE_ROOT)


def owner_id(user: AbstractBaseUser) -> str:
    return str(user.pk)
