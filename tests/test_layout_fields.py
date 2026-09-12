"""Tests for `layout_ocr.fields`'s per-region OCR -- issue #5."""
from __future__ import annotations

from pathlib import Path

import pytest

from document_core import ocr as ocr_module
from document_core.ocr import OCRBackend, OCRBackendError, OCRTextResult, register_ocr_backend
from document_core.schema import BoundingBox
from layout_ocr.detection import DetectedRegion
from layout_ocr.fields import FieldExtraction, extract_fields_from_regions


class _FakeOCRBackend(OCRBackend):
    """Returns text keyed off the bbox so each region's result is
    distinguishable, and raises for one designated "poison" bbox to exercise
    the failure path."""

    key = "fake-field-ocr"
    label = "Fake (tests only)"
    poison_x = 999.0

    def recognize(self, image_path: Path, bbox: BoundingBox | None = None) -> OCRTextResult:
        if bbox is not None and bbox.x == self.poison_x:
            raise OCRBackendError("simulated OCR engine failure")
        suffix = f"-{int(bbox.x)}" if bbox else ""
        return OCRTextResult(text=f"text{suffix}", confidence=0.8)


@pytest.fixture(scope="module", autouse=True)
def _register_fake_backend():
    register_ocr_backend(_FakeOCRBackend)
    yield
    ocr_module._REGISTRY.pop(_FakeOCRBackend.key, None)


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "page.png"
    path.write_bytes(b"not-a-real-image")
    return path


def _region(x: float) -> DetectedRegion:
    return DetectedRegion(
        class_name="text", confidence=0.9, bbox=BoundingBox(x=x, y=0, width=10, height=10)
    )


def test_extract_fields_runs_ocr_per_region_and_associates_text(image_path):
    regions = [_region(0), _region(10)]

    results = extract_fields_from_regions(image_path, regions, ocr_backend_key="fake-field-ocr")

    assert [r.text for r in results] == ["text-0", "text-10"]
    assert all(r.region in regions for r in results)
    assert all(not r.failed for r in results)


def test_a_region_that_fails_ocr_is_flagged_not_dropped(image_path):
    regions = [_region(0), _region(_FakeOCRBackend.poison_x), _region(20)]

    results = extract_fields_from_regions(image_path, regions, ocr_backend_key="fake-field-ocr")

    # all three regions are still present -- the failing one isn't dropped
    assert len(results) == 3
    assert results[1].failed
    assert results[1].text is None
    assert "simulated OCR engine failure" in results[1].ocr_error
    assert not results[0].failed and not results[2].failed


def test_no_regions_returns_empty_list(image_path):
    assert extract_fields_from_regions(image_path, [], ocr_backend_key="fake-field-ocr") == []


def test_field_extraction_round_trips_through_dict():
    region = _region(5)
    extraction = FieldExtraction(region=region, text="hello", confidence=0.6)
    assert FieldExtraction.from_dict(extraction.to_dict()) == extraction

    failed = FieldExtraction(region=region, text=None, confidence=None, ocr_error="boom")
    assert FieldExtraction.from_dict(failed.to_dict()) == failed
    assert failed.failed
