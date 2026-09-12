"""Tests for issue #14 -- letting the user pick which extraction engine runs
on a document, through `document_core.extraction.extract_fields` only, with
the choice recorded against the document."""
from __future__ import annotations

import io

import pytest
from django.contrib.auth import get_user_model
from django.test import override_settings
from django.urls import reverse

from document_core.extraction import ExtractionBackendError
from document_core.schema import ExtractedField, ExtractionResult
from documents import services as documents_services

pytestmark = pytest.mark.django_db

User = get_user_model()


@pytest.fixture
def store_root(tmp_path):
    with override_settings(DOCUMENT_STORE_ROOT=str(tmp_path / "documents")):
        yield tmp_path


@pytest.fixture
def user(db):
    return User.objects.create_user(username="alice", password="s3cret-pass!")


@pytest.fixture
def uploaded_doc(client, user, store_root):
    client.force_login(user)
    upload = io.BytesIO(b"%PDF-1.4 test")
    upload.name = "invoice.pdf"
    response = client.post(reverse("documents:upload"), {"file": upload})
    return response.url.rstrip("/").rsplit("/", 1)[-1]


def _stub_extract_fields(monkeypatch, result_factory=None):
    def fake(backend_key, document_path, document_id):
        if result_factory:
            return result_factory(backend_key, document_id)
        return ExtractionResult(document_id=document_id, fields=[])

    monkeypatch.setattr(documents_services, "extract_fields", fake)


def test_detail_page_offers_both_engine_choices(client, user, uploaded_doc):
    response = client.get(reverse("documents:detail", args=[uploaded_doc]))
    assert b'value="layout_ocr"' in response.content
    assert b'value="llm_vision"' in response.content


def test_running_extraction_calls_extract_fields_with_the_chosen_backend(
    client, user, uploaded_doc, monkeypatch
):
    calls = []

    def fake_extract_fields(backend_key, document_path, document_id):
        calls.append(backend_key)
        return ExtractionResult(
            document_id=document_id,
            fields=[ExtractedField(name="total", value="12.00", confidence=0.9)],
        )

    monkeypatch.setattr(documents_services, "extract_fields", fake_extract_fields)

    response = client.post(
        reverse("documents:detail", args=[uploaded_doc]), {"backend_key": "llm_vision"}
    )

    assert response.status_code == 302
    assert calls == ["llm_vision"]

    result = documents_services.get_extraction_result(uploaded_doc)
    assert result["backend_key"] == "llm_vision"
    assert result["result"]["fields"][0]["name"] == "total"


def test_the_chosen_engine_is_recorded_and_shown_on_the_page(
    client, user, uploaded_doc, monkeypatch
):
    _stub_extract_fields(monkeypatch)
    client.post(reverse("documents:detail", args=[uploaded_doc]), {"backend_key": "layout_ocr"})

    response = client.get(reverse("documents:detail", args=[uploaded_doc]))
    assert b"layout_ocr" in response.content


def test_llm_vision_result_shows_the_self_reported_confidence_caveat(
    client, user, uploaded_doc, monkeypatch
):
    _stub_extract_fields(monkeypatch)
    client.post(reverse("documents:detail", args=[uploaded_doc]), {"backend_key": "llm_vision"})

    response = client.get(reverse("documents:detail", args=[uploaded_doc]))
    assert b"self-report" in response.content


def test_layout_ocr_result_does_not_show_the_llm_caveat(client, user, uploaded_doc, monkeypatch):
    _stub_extract_fields(monkeypatch)
    client.post(reverse("documents:detail", args=[uploaded_doc]), {"backend_key": "layout_ocr"})

    response = client.get(reverse("documents:detail", args=[uploaded_doc]))
    assert b"self-report" not in response.content


def test_extraction_backend_failure_is_shown_and_not_persisted(
    client, user, uploaded_doc, monkeypatch
):
    def failing_extract_fields(backend_key, document_path, document_id):
        raise ExtractionBackendError("no checkpoint configured")

    monkeypatch.setattr(documents_services, "extract_fields", failing_extract_fields)

    response = client.post(
        reverse("documents:detail", args=[uploaded_doc]),
        {"backend_key": "layout_ocr"},
        follow=True,
    )

    assert b"Extraction failed" in response.content
    assert documents_services.get_extraction_result(uploaded_doc) is None


def test_run_extraction_unknown_document_raises_keyerror(store_root):
    with pytest.raises(KeyError, match="No such document"):
        documents_services.run_extraction("does-not-exist", "layout_ocr")
