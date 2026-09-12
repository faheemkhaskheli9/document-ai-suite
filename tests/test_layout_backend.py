"""Tests for `layout_ocr.backend`'s registration as the `"layout_ocr"`
`document_core.extraction` backend -- issue #6."""
from __future__ import annotations

import pytest

from document_core.extraction import (
    ExtractionBackendError,
    ExtractionBackendUnavailable,
    extract_fields,
    is_registered,
)
from document_core.schema import BoundingBox, ExtractionResult
from layout_ocr.backend import LayoutOCRExtractionBackend
from layout_ocr.detection import DetectedRegion, LayoutDetector
from layout_ocr.fields import FieldExtraction


class _FakeDetector(LayoutDetector):
    def __init__(self, regions=None, error=None):
        self._regions = regions or []
        self._error = error

    def detect(self, image_path):
        if self._error is not None:
            raise self._error
        return self._regions


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "page.png"
    path.write_bytes(b"not-a-real-image")
    return path


def test_layout_ocr_backend_is_registered_by_default():
    assert is_registered("layout_ocr")


def test_extract_via_registry_returns_extraction_result(image_path, monkeypatch):
    region = DetectedRegion(
        class_name="text", confidence=0.9, bbox=BoundingBox(x=0, y=0, width=10, height=10)
    )
    backend = LayoutOCRExtractionBackend(detector=_FakeDetector([region]))
    monkeypatch.setattr(
        "layout_ocr.backend.extract_fields_from_regions",
        lambda path, regions, ocr_backend_key: [
            FieldExtraction(region=regions[0], text="Vendor: Acme Corp", confidence=0.8)
        ],
    )

    result = backend.extract(image_path, "doc-1")

    assert isinstance(result, ExtractionResult)
    assert result.document_id == "doc-1"
    assert any(f.name == "vendor" for f in result.fields)


def test_missing_document_raises_extraction_backend_error(tmp_path):
    backend = LayoutOCRExtractionBackend(detector=_FakeDetector([]))
    with pytest.raises(ExtractionBackendError, match="No such document"):
        backend.extract(tmp_path / "nope.png", "doc-1")


def test_unavailable_detector_raises_extraction_backend_unavailable(image_path):
    from layout_ocr.detection import LayoutDetectionUnavailable

    backend = LayoutOCRExtractionBackend(
        detector=_FakeDetector(error=LayoutDetectionUnavailable("no checkpoint"))
    )
    with pytest.raises(ExtractionBackendUnavailable, match="no checkpoint"):
        backend.extract(image_path, "doc-1")


def test_detection_error_raises_extraction_backend_error(image_path):
    from layout_ocr.detection import LayoutDetectionError

    backend = LayoutOCRExtractionBackend(
        detector=_FakeDetector(error=LayoutDetectionError("engine crash"))
    )
    with pytest.raises(ExtractionBackendError, match="engine crash"):
        backend.extract(image_path, "doc-1")


def test_extract_fields_entry_point_reaches_layout_ocr_backend(image_path):
    """`extract_fields("layout_ocr", ...)` is the one interface other feature
    apps should use -- proves the registered singleton is reachable through
    it, not just directly instantiated. No checkpoint is configured in this
    environment, so it must fail loudly the same way `backend.extract()`
    does directly, rather than the registry lookup masking the error."""
    with pytest.raises(ExtractionBackendUnavailable, match="no layout-detection model"):
        extract_fields("layout_ocr", image_path, "doc-1")
