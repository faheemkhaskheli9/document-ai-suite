"""
Layout detection for the `layout_ocr` feature app -- issue #4 ("Port layout
detection model"). Identifies field-level regions (text, title, list, table,
figure -- PubLayNet's category set, see `layout_ocr.dataset.DEFAULT_CLASSES`)
on an uploaded document page, so `document_core.ocr` can later be run
per-region instead of over the whole page (issue #5).

`document-ai-yolo-ocr` never shipped a standalone inference wrapper -- its
`train.py` only fine-tunes a checkpoint. This module is the actual detection
call the layout+OCR pipeline needs, built the same way `train.run_training`
was: a thin wrapper around Ultralytics' real API with a `model=` injection
point for tests, so a unit test never has to download a real checkpoint or
run real inference.

The Ultralytics import is lazy, inside `_load_model()`, so importing this
module never requires `ultralytics`/`torch` to be installed. No trained
layout-detection checkpoint ships with this repo (see
`docs/architecture.md`'s dataset notes); a missing/unset `LAYOUT_MODEL_WEIGHTS`
raises `LayoutDetectionUnavailable` rather than fabricating empty detections --
mirrors `document_core.ocr`'s "missing engine fails loudly" contract.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from document_core.schema import BoundingBox
from layout_ocr.dataset import DEFAULT_CLASSES

DEFAULT_CONFIDENCE_THRESHOLD = 0.25


@dataclass(frozen=True)
class DetectedRegion:
    """One field-level region found on a page."""

    class_name: str
    confidence: float
    bbox: BoundingBox

    def to_dict(self) -> dict:
        return {
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": self.bbox.model_dump(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DetectedRegion":
        return cls(
            class_name=data["class_name"],
            confidence=data["confidence"],
            bbox=BoundingBox(**data["bbox"]),
        )


class LayoutDetectionError(RuntimeError):
    """Detection failed -- bad input, or the underlying engine errored."""


class LayoutDetectionUnavailable(LayoutDetectionError):
    """No trained layout-detection model is configured/available in this
    environment (missing `ultralytics` package, or no `LAYOUT_MODEL_WEIGHTS`
    checkpoint). Raised instead of silently returning zero regions, which
    would look identical to "this page genuinely has no fields"."""


class LayoutDetector:
    """Wraps an Ultralytics YOLO layout-detection checkpoint.

    `model` is an injection point for tests: pass a fake object that is
    itself callable as `model(image_path, conf=...)` and returns a list of
    fake Ultralytics `Results`-shaped objects (`.boxes` iterable of objects
    exposing `.cls`, `.conf`, `.xyxy`), so a unit test never has to load a
    real checkpoint or run real inference. When `model` is None (the
    default, real-use path), the checkpoint at `weights_path` (or the
    `LAYOUT_MODEL_WEIGHTS` env var) is loaded via the real `ultralytics.YOLO`
    the first time `detect()` is called.
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        classes: list[str] | None = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        model: Any = None,
    ):
        self._weights_path = weights_path or os.environ.get("LAYOUT_MODEL_WEIGHTS")
        self.classes = classes or DEFAULT_CLASSES
        self.confidence_threshold = confidence_threshold
        self._model = model

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        if not self._weights_path:
            raise LayoutDetectionUnavailable(
                "no layout-detection model configured -- set LAYOUT_MODEL_WEIGHTS "
                "to a trained YOLO checkpoint (see layout_ocr.train.run_training), "
                "or pass weights_path= explicitly"
            )
        if not Path(self._weights_path).exists():
            raise LayoutDetectionUnavailable(
                f"layout-detection weights not found: {self._weights_path}"
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise LayoutDetectionUnavailable(
                "the 'ultralytics' package is not installed; add it to requirements.txt"
            ) from exc
        self._model = YOLO(str(self._weights_path))
        return self._model

    def detect(self, image_path: str | Path) -> list[DetectedRegion]:
        """Run layout detection on `image_path` and return every region
        found at or above `confidence_threshold`.

        Raises `LayoutDetectionError` (or `LayoutDetectionUnavailable` if no
        engine/checkpoint is available) rather than returning a fabricated
        result on failure.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise LayoutDetectionError(f"No such image: {image_path}")

        model = self._load_model()
        try:
            results = model(str(image_path), conf=self.confidence_threshold)
        except Exception as exc:  # engine-internal failure, not a config problem
            raise LayoutDetectionError(
                f"layout detection failed on {image_path}: {exc}"
            ) from exc

        if not results:
            return []

        boxes = getattr(results[0], "boxes", None) or []

        regions: list[DetectedRegion] = []
        for box in boxes:
            try:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            except (TypeError, ValueError, IndexError) as exc:
                raise LayoutDetectionError(
                    f"malformed detection box returned by engine: {exc}"
                ) from exc

            class_name = (
                self.classes[cls_id] if 0 <= cls_id < len(self.classes) else str(cls_id)
            )
            regions.append(
                DetectedRegion(
                    class_name=class_name,
                    confidence=confidence,
                    bbox=BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1),
                )
            )
        return regions
