"""Tests for the pluggable OCR backend abstraction -- issue #2."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from document_core import ocr
from document_core.ocr import (
    OCRBackend,
    OCRBackendError,
    OCRBackendUnavailable,
    OCRTextResult,
    TesseractOCRBackend,
    extract_text,
    get_ocr_backend,
    is_registered,
    register_ocr_backend,
)
from document_core.schema import BoundingBox


class _FakeOCRBackend(OCRBackend):
    """A second backend, registered only for this test module, used to prove
    the interface is swappable without touching any caller."""

    key = "fake-test-backend"
    label = "Fake (tests only)"

    def recognize(self, image_path: Path, bbox: BoundingBox | None = None) -> OCRTextResult:
        suffix = f" (cropped to {bbox.width:.0f}x{bbox.height:.0f})" if bbox else ""
        return OCRTextResult(text=f"fake text from {image_path.name}{suffix}", confidence=0.5)


@pytest.fixture(scope="module", autouse=True)
def _register_fake_backend():
    register_ocr_backend(_FakeOCRBackend)
    yield
    ocr._REGISTRY.pop(_FakeOCRBackend.key, None)


def test_tesseract_backend_is_registered_by_default():
    assert is_registered("tesseract")
    assert isinstance(get_ocr_backend("tesseract"), TesseractOCRBackend)


def test_registering_a_duplicate_key_raises():
    with pytest.raises(ValueError, match="duplicate OCR backend key"):
        register_ocr_backend(_FakeOCRBackend)


def test_unknown_key_raises_keyerror():
    with pytest.raises(KeyError, match="no OCR backend registered"):
        get_ocr_backend("does-not-exist")


def test_extract_text_is_the_one_interface_feature_apps_use(tmp_path):
    """Calling through `extract_text(key, ...)` behaves identically to
    `get_ocr_backend(key).recognize(...)` -- the acceptance criterion that
    feature apps go through one interface, not a per-backend import."""
    image_path = tmp_path / "doc.png"
    image_path.write_bytes(b"not-a-real-image")

    result = extract_text("fake-test-backend", image_path)

    assert result == OCRTextResult(text="fake text from doc.png", confidence=0.5)


def test_swapping_the_backend_key_needs_no_caller_change(tmp_path):
    """Same call site, different `backend_key` -- this is what 'swapping the
    backend does not require changes in calling feature apps' means: the
    caller code below is identical regardless of which key is passed in."""
    image_path = tmp_path / "doc.png"
    image_path.write_bytes(b"not-a-real-image")

    def call_site(backend_key: str) -> str:
        return extract_text(backend_key, image_path).text

    assert call_site("fake-test-backend") == "fake text from doc.png"


def test_fake_backend_respects_bbox(tmp_path):
    image_path = tmp_path / "doc.png"
    image_path.write_bytes(b"not-a-real-image")
    bbox = BoundingBox(x=0, y=0, width=100, height=40)

    result = extract_text("fake-test-backend", image_path, bbox=bbox)

    assert "cropped to 100x40" in result.text


def test_tesseract_backend_raises_on_missing_file(tmp_path):
    missing = tmp_path / "nope.png"
    with pytest.raises(OCRBackendError, match="No such image"):
        get_ocr_backend("tesseract").recognize(missing)


@pytest.mark.skipif(
    shutil.which("tesseract") is not None,
    reason="only exercises the missing-binary path; a real engine is tested separately",
)
def test_tesseract_backend_raises_unavailable_without_the_binary(tmp_path):
    """CPU-only boundary mock: this environment has no `tesseract` binary
    installed, so recognize() must fail loudly (OCRBackendUnavailable)
    instead of returning fabricated text -- never silently succeed."""
    from PIL import Image

    image_path = tmp_path / "doc.png"
    Image.new("RGB", (20, 20), color="white").save(image_path)

    with pytest.raises(OCRBackendUnavailable, match="tesseract"):
        get_ocr_backend("tesseract").recognize(image_path)


@pytest.mark.skipif(
    shutil.which("tesseract") is None,
    reason="requires the real tesseract binary on PATH",
)
def test_tesseract_backend_recognizes_real_text(tmp_path):
    from PIL import Image, ImageDraw

    image_path = tmp_path / "doc.png"
    image = Image.new("RGB", (200, 60), color="white")
    ImageDraw.Draw(image).text((10, 10), "HELLO", fill="black")
    image.save(image_path)

    result = get_ocr_backend("tesseract").recognize(image_path)

    assert "HELLO" in result.text.upper()
