"""Tests for `layout_ocr.services`'s detection persistence -- issue #4
acceptance criterion: "Detection output is persisted per document"."""
from __future__ import annotations

import pytest
from django.test import override_settings

from document_core.schema import BoundingBox
from documents.services import get_document_store
from layout_ocr.detection import DetectedRegion, LayoutDetector
from layout_ocr.fields import FieldExtraction
from layout_ocr.services import (
    detect_and_persist,
    get_persisted_fields,
    get_persisted_regions,
    run_layout_ocr,
)


class _FakeDetector(LayoutDetector):
    def __init__(self, regions):
        self._regions = regions

    def detect(self, image_path):
        return self._regions


@pytest.fixture
def store_root(tmp_path):
    with override_settings(DOCUMENT_STORE_ROOT=str(tmp_path / "documents")):
        yield tmp_path


@pytest.fixture
def uploaded_doc(store_root, tmp_path):
    store = get_document_store()
    record = store.store(b"%PDF-1.4 test", "invoice.pdf", owner="alice")
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"not-a-real-image")
    return record, image_path


def test_detect_and_persist_writes_and_returns_regions(uploaded_doc):
    record, image_path = uploaded_doc
    regions = [
        DetectedRegion(class_name="table", confidence=0.8, bbox=BoundingBox(x=1, y=2, width=3, height=4))
    ]
    detector = _FakeDetector(regions)

    result = detect_and_persist(record.id, image_path, detector=detector)

    assert result == regions
    assert get_persisted_regions(record.id) == regions


def test_get_persisted_regions_before_detection_returns_none(uploaded_doc):
    record, _ = uploaded_doc
    assert get_persisted_regions(record.id) is None


def test_detect_and_persist_unknown_document_raises_keyerror(store_root, tmp_path):
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"not-a-real-image")

    with pytest.raises(KeyError, match="No such document"):
        detect_and_persist("does-not-exist", image_path, detector=_FakeDetector([]))


def test_run_layout_ocr_persists_regions_and_fields(uploaded_doc, monkeypatch):
    record, image_path = uploaded_doc
    region = DetectedRegion(
        class_name="text", confidence=0.9, bbox=BoundingBox(x=0, y=0, width=10, height=10)
    )
    detector = _FakeDetector([region])
    fake_extraction = FieldExtraction(region=region, text="hello", confidence=0.7)

    monkeypatch.setattr(
        "layout_ocr.services.extract_fields_from_regions",
        lambda image_path, regions, ocr_backend_key="tesseract": [fake_extraction],
    )

    fields = run_layout_ocr(record.id, image_path, detector=detector)

    assert fields == [fake_extraction]
    assert get_persisted_regions(record.id) == [region]
    assert get_persisted_fields(record.id) == [fake_extraction]


def test_get_persisted_fields_before_run_returns_none(uploaded_doc):
    record, _ = uploaded_doc
    assert get_persisted_fields(record.id) is None
