import pytest
from django.contrib.auth import get_user_model
from django.urls import reverse

pytestmark = pytest.mark.django_db

User = get_user_model()


@pytest.fixture
def user(db):
    return User.objects.create_user(username="alice", password="s3cret-pass!")


def test_home_redirects_unauthenticated_users_to_login(client):
    response = client.get(reverse("dashboard:home"))

    assert response.status_code == 302
    assert reverse("accounts:login") in response.url


def test_home_lists_every_registered_feature(client, user):
    client.login(username="alice", password="s3cret-pass!")
    response = client.get(reverse("dashboard:home"))

    assert response.status_code == 200
    for label in ("Documents", "Layout + OCR extraction", "Classify + Review", "Full pipeline"):
        assert label.encode() in response.content


def test_home_links_available_feature_to_its_own_flow(client, user):
    client.login(username="alice", password="s3cret-pass!")
    response = client.get(reverse("dashboard:home"))

    content = response.content.decode()
    assert reverse("documents:list") in content


def test_home_documents_and_full_pipeline_features_are_open(client, user):
    client.login(username="alice", password="s3cret-pass!")
    response = client.get(reverse("dashboard:home"))

    content = response.content.decode()
    # Documents and (issue #10) Full pipeline have their own flow now;
    # Layout + OCR and Classify + Review stay "Coming soon" -- neither has
    # its own UI, they're only reachable through Documents/Full pipeline.
    assert content.count(">Open<") == 2
    assert content.count("Coming soon") == 2
    assert reverse("full_pipeline:list") in content
