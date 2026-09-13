"""Tests for the `full_pipeline` web views -- issue #10."""
from __future__ import annotations

import io

import pytest
from django.contrib.auth import get_user_model
from django.test import override_settings
from django.urls import reverse

from document_core.schema import ExtractedField, ExtractionResult

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
def other_user(db):
    return User.objects.create_user(username="bob", password="s3cret-pass!")


@pytest.fixture
def uploaded_doc(client, user, store_root):
    client.force_login(user)
    upload = io.BytesIO(b"%PDF-1.4 test")
    upload.name = "invoice.pdf"
    response = client.post(reverse("documents:upload"), {"file": upload})
    return response.url.rstrip("/").rsplit("/", 1)[-1]


def _stub_extract_fields(monkeypatch, confidence=0.95):
    def fake(backend_key, document_path, document_id):
        return ExtractionResult(
            document_id=document_id,
            document_type="invoice",
            fields=[ExtractedField(name="total", value="$5", confidence=confidence)],
            overall_confidence=confidence,
        )

    monkeypatch.setattr("full_pipeline.pipeline.extract_fields", fake)


def test_list_requires_login(client, store_root):
    response = client.get(reverse("full_pipeline:list"))
    assert response.status_code == 302
    assert reverse("accounts:login") in response.url


def test_list_only_shows_the_current_user_documents(client, user, other_user, uploaded_doc):
    client.logout()
    client.force_login(other_user)
    response = client.get(reverse("full_pipeline:list"))

    assert b"invoice.pdf" not in response.content


def test_run_page_shows_no_result_before_running(client, user, uploaded_doc):
    response = client.get(reverse("full_pipeline:run", args=[uploaded_doc]))
    assert response.status_code == 200
    assert b"hasn't run" in response.content


def test_posting_runs_the_pipeline_and_shows_all_three_stages(
    client, user, uploaded_doc, monkeypatch
):
    _stub_extract_fields(monkeypatch)
    response = client.post(
        reverse("full_pipeline:run", args=[uploaded_doc]), {"backend_key": "layout_ocr"}, follow=True
    )

    assert response.status_code == 200
    assert b"Classification" in response.content
    assert b"Extraction" in response.content
    assert b"Validation" in response.content
    assert b"auto_accepted" in response.content


def test_a_user_cannot_run_the_pipeline_on_another_users_document(
    client, user, other_user, uploaded_doc
):
    client.logout()
    client.force_login(other_user)
    response = client.get(reverse("full_pipeline:run", args=[uploaded_doc]))
    assert response.status_code == 404
