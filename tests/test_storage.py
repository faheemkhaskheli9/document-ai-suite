import pytest

from document_core.storage import DocumentStore, DocumentValidationError


@pytest.fixture
def store(tmp_path):
    return DocumentStore(tmp_path / "docs")


def test_store_pdf_creates_record(store):
    record = store.store(b"%PDF-1.4 fake pdf bytes", "invoice.pdf", owner="alice")

    assert record.owner == "alice"
    assert record.filename == "invoice.pdf"
    assert record.status == "uploaded"
    assert record.content_hash

    fetched = store.get(record.id)
    assert fetched == record


def test_store_image_is_accepted(store):
    record = store.store(b"\x89PNG fake bytes", "scan.png", owner="bob")
    assert record.status == "uploaded"


def test_unsupported_extension_is_rejected(store):
    with pytest.raises(DocumentValidationError, match="Unsupported file type"):
        store.store(b"hello", "notes.txt", owner="alice")


def test_empty_file_is_rejected(store):
    with pytest.raises(DocumentValidationError, match="empty"):
        store.store(b"", "invoice.pdf", owner="alice")


def test_same_filename_different_owners_do_not_collide(store):
    r1 = store.store(b"content-1", "invoice.pdf", owner="alice")
    r2 = store.store(b"content-2", "invoice.pdf", owner="bob")

    assert r1.id != r2.id
    assert store.get(r1.id).owner == "alice"
    assert store.get(r2.id).owner == "bob"


def test_identical_content_from_two_uploads_does_not_collide(store):
    r1 = store.store(b"same bytes", "a.pdf", owner="alice")
    r2 = store.store(b"same bytes", "b.pdf", owner="alice")

    assert r1.id != r2.id
    assert r1.content_hash == r2.content_hash


def test_list_for_owner_only_returns_that_owners_documents(store):
    store.store(b"one", "a.pdf", owner="alice")
    store.store(b"two", "b.pdf", owner="bob")
    store.store(b"three", "c.pdf", owner="alice")

    alice_docs = store.list_for_owner("alice")
    assert {d.filename for d in alice_docs} == {"a.pdf", "c.pdf"}


def test_set_status_updates_and_persists(store):
    record = store.store(b"content", "a.pdf", owner="alice")
    updated = store.set_status(record.id, "processing")

    assert updated.status == "processing"
    assert store.get(record.id).status == "processing"


def test_get_unknown_id_returns_none(store):
    assert store.get("does-not-exist") is None


def test_set_status_unknown_id_raises_keyerror(store):
    with pytest.raises(KeyError):
        store.set_status("does-not-exist", "done")


def test_save_and_load_artifact_round_trips(store):
    record = store.store(b"content", "a.pdf", owner="alice")
    store.save_artifact(record.id, "layout_regions", {"regions": [{"class_name": "table"}]})

    assert store.load_artifact(record.id, "layout_regions") == {
        "regions": [{"class_name": "table"}]
    }


def test_load_artifact_not_yet_produced_returns_none(store):
    record = store.store(b"content", "a.pdf", owner="alice")
    assert store.load_artifact(record.id, "layout_regions") is None


def test_save_artifact_unknown_document_raises_keyerror(store):
    with pytest.raises(KeyError, match="No such document"):
        store.save_artifact("does-not-exist", "layout_regions", {"regions": []})


def test_get_after_interrupted_write_sees_nothing_or_complete_record(store, monkeypatch):
    """Rule 1/2: a write interrupted mid-way must never leave a record that
    looks stored but is truncated/half-written."""
    from document_core import storage as storage_module

    original_write = storage_module.DocumentStore._atomic_write

    calls = {"n": 0}

    def flaky_write(path, content):
        calls["n"] += 1
        if calls["n"] == 2:  # fail on the record.json write, after the file write succeeded
            raise OSError("simulated disk failure")
        return original_write(path, content)

    monkeypatch.setattr(storage_module.DocumentStore, "_atomic_write", staticmethod(flaky_write))

    with pytest.raises(OSError):
        store.store(b"content", "a.pdf", owner="alice")

    # No record.json was ever written for this doc, so nothing claims to be
    # a stored document; a real caller sees "not found", not a broken one.
    docs_root = list(store.root.iterdir())
    for doc_dir in docs_root:
        assert not (doc_dir / "record.json").exists()
