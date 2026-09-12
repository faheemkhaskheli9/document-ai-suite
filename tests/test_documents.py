import io

import pytest
from django.contrib.auth import get_user_model
from django.test import override_settings
from django.urls import reverse

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


def _upload_file(client, content=b"%PDF-1.4 test", name="invoice.pdf"):
    upload = io.BytesIO(content)
    upload.name = name
    return client.post(reverse("documents:upload"), {"file": upload})


def test_upload_requires_login(client, store_root):
    response = client.get(reverse("documents:upload"))
    assert response.status_code == 302
    assert reverse("accounts:login") in response.url


def test_successful_upload_redirects_to_detail(client, user, store_root):
    client.force_login(user)
    response = _upload_file(client)

    assert response.status_code == 302
    detail = client.get(response.url)
    assert detail.status_code == 200
    assert b"invoice.pdf" in detail.content


def test_unsupported_extension_is_rejected_with_a_visible_error(client, user, store_root):
    client.force_login(user)
    response = _upload_file(client, content=b"not a document", name="notes.exe")

    # Rejected uploads must never redirect as if they succeeded -- the error
    # is shown back on the same form (robustness rule: loud failure).
    assert response.status_code == 200
    assert b"Unsupported file type" in response.content


def test_list_only_shows_the_current_user_documents(client, user, other_user, store_root):
    client.force_login(other_user)
    _upload_file(client, name="bobs-file.pdf")
    client.logout()

    client.force_login(user)
    response = client.get(reverse("documents:list"))

    assert b"bobs-file.pdf" not in response.content
    assert b"No documents uploaded yet." in response.content


def test_a_user_cannot_view_another_users_document_by_id(client, user, other_user, store_root):
    client.force_login(other_user)
    upload_response = _upload_file(client, name="bobs-file.pdf")
    doc_id = upload_response.url.rstrip("/").rsplit("/", 1)[-1]
    client.logout()

    client.force_login(user)
    response = client.get(reverse("documents:detail", args=[doc_id]))

    assert response.status_code == 404
