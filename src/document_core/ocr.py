"""
Pluggable OCR backend abstraction shared by every feature app -- README.md
Section 2 ("document_core layer ... OCR backend wrapper") and issue #2.

Feature apps call `extract_text(backend_key, ...)` (or `get_ocr_backend(key)`
directly) and never import a concrete OCR library themselves -- swapping the
backend behind a given key, or adding a new one, never touches `layout_ocr`
or `classify_review`. Mirrors the string-keyed registry pattern already used
for feature-task registration in this portfolio (see
`medical-imaging-suite/imaging_core/registry.py`'s `BaseImagingTask` /
`@register_task`), applied here one layer lower, to OCR engines instead of
whole feature tasks.

Concrete backends must do any heavy/native import (here: `pytesseract`,
`PIL.Image`) lazily inside `recognize()`, not at module import time, so
importing this module never requires the engine to be installed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from document_core.schema import BoundingBox


@dataclass
class OCRTextResult:
    """What every backend's `recognize()` returns -- the one shape feature
    apps consume regardless of which engine produced it."""

    text: str
    confidence: float  # 0.0-1.0, mean word confidence where the engine reports one


class OCRBackendError(RuntimeError):
    """Recognition failed -- bad input, or the underlying engine errored."""


class OCRBackendUnavailable(OCRBackendError):
    """The engine behind this backend isn't installed/reachable in this
    environment (e.g. the `tesseract` binary isn't on PATH). Raised instead
    of silently returning empty text, so a missing engine fails loudly."""


class OCRBackend(ABC):
    """Contract every concrete OCR engine wrapper must implement."""

    key: str
    label: str

    @abstractmethod
    def recognize(self, image_path: Path, bbox: BoundingBox | None = None) -> OCRTextResult:
        """Run OCR on `image_path` (or, if `bbox` is given, the region of it
        `bbox` describes) and return the recognized text + confidence.

        Must raise `OCRBackendError` (or `OCRBackendUnavailable` if the
        underlying engine isn't available) rather than returning a fabricated
        result on failure.
        """


_REGISTRY: dict[str, OCRBackend] = {}


def register_ocr_backend(backend_cls: type[OCRBackend]) -> type[OCRBackend]:
    """Class decorator: instantiate `backend_cls` and register it by `.key`."""
    instance = backend_cls()
    if not instance.key:
        raise ValueError(f"{backend_cls.__name__}.key must be a non-empty string")
    if instance.key in _REGISTRY:
        raise ValueError(f"duplicate OCR backend key: {instance.key!r}")
    _REGISTRY[instance.key] = instance
    return backend_cls


def get_ocr_backend(key: str) -> OCRBackend:
    try:
        return _REGISTRY[key]
    except KeyError:
        raise KeyError(f"no OCR backend registered for key {key!r}") from None


def all_ocr_backends() -> list[OCRBackend]:
    """All registered backends, sorted by label for stable UI ordering."""
    return sorted(_REGISTRY.values(), key=lambda b: b.label)


def is_registered(key: str) -> bool:
    return key in _REGISTRY


def extract_text(
    backend_key: str, image_path: Path, bbox: BoundingBox | None = None
) -> OCRTextResult:
    """Convenience entry point every feature app should call instead of
    reaching for `get_ocr_backend` + `.recognize` directly -- the one
    interface issue #2 requires. Swapping which engine `backend_key` maps to
    (or adding a new key) never requires a change here or in any caller."""
    return get_ocr_backend(backend_key).recognize(image_path, bbox)


@register_ocr_backend
class TesseractOCRBackend(OCRBackend):
    """Wraps the `pytesseract` binding for the Tesseract OCR engine -- free,
    CPU-only, no paid API. Requires the `tesseract` binary on PATH; if it
    isn't installed, `recognize()` raises `OCRBackendUnavailable` rather than
    returning empty/fabricated text (boundary mocked in tests -- see
    `tests/test_ocr.py` -- since the CI environment may not have the
    `tesseract` binary installed)."""

    key = "tesseract"
    label = "Tesseract (local, CPU)"

    def recognize(self, image_path: Path, bbox: BoundingBox | None = None) -> OCRTextResult:
        image_path = Path(image_path)
        if not image_path.exists():
            raise OCRBackendError(f"No such image: {image_path}")

        try:
            import pytesseract
            from PIL import Image
        except ImportError as exc:
            raise OCRBackendUnavailable(
                "pytesseract/Pillow not installed; add them to requirements.txt"
            ) from exc

        try:
            with Image.open(image_path) as image:
                if bbox is not None:
                    image = image.crop(
                        (bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height)
                    )
                data = pytesseract.image_to_data(
                    image, output_type=pytesseract.Output.DICT
                )
        except pytesseract.TesseractNotFoundError as exc:
            raise OCRBackendUnavailable(
                "the 'tesseract' binary is not installed or not on PATH"
            ) from exc
        except Exception as exc:  # engine-internal failure, not a missing binary
            raise OCRBackendError(f"tesseract failed on {image_path}: {exc}") from exc

        words = [w for w in data["text"] if w.strip()]
        confidences = [float(c) for c in data["conf"] if str(c) not in ("-1", "")]
        text = " ".join(words)
        confidence = (sum(confidences) / len(confidences) / 100.0) if confidences else 0.0
        return OCRTextResult(text=text, confidence=confidence)
