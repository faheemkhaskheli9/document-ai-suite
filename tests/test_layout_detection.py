"""Tests for `layout_ocr.detection`'s YOLO layout-detection wrapper --
issue #4. CPU-only, no real checkpoint: the Ultralytics boundary is mocked
via `LayoutDetector(model=...)`, mirroring `layout_ocr.train.run_training`'s
`trainer=` injection point."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from layout_ocr.detection import (
    DetectedRegion,
    LayoutDetectionError,
    LayoutDetectionUnavailable,
    LayoutDetector,
)


@dataclass
class _FakeBox:
    cls: list
    conf: list
    xyxy: list


class _FakeBoxes(list):
    """Mimics Ultralytics' `Boxes` container -- just an iterable of boxes."""


@dataclass
class _FakeResult:
    boxes: _FakeBoxes


class _FakeYoloModel:
    """Stands in for an `ultralytics.YOLO` instance: callable, returns a
    list of fake `Results`."""

    def __init__(self, results):
        self._results = results
        self.calls = []

    def __call__(self, image_path, conf=None):
        self.calls.append((image_path, conf))
        return self._results


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "page.png"
    path.write_bytes(b"not-a-real-image")
    return path


def test_detect_returns_regions_from_fake_model(image_path):
    fake_model = _FakeYoloModel(
        [
            _FakeResult(
                boxes=_FakeBoxes(
                    [
                        _FakeBox(cls=[1], conf=[0.91], xyxy=[[10.0, 20.0, 110.0, 60.0]]),
                        _FakeBox(cls=[3], conf=[0.55], xyxy=[[0.0, 0.0, 50.0, 50.0]]),
                    ]
                )
            )
        ]
    )
    detector = LayoutDetector(model=fake_model)

    regions = detector.detect(image_path)

    assert len(regions) == 2
    assert regions[0].class_name == "title"
    assert regions[0].confidence == pytest.approx(0.91)
    assert regions[0].bbox.x == pytest.approx(10.0)
    assert regions[0].bbox.y == pytest.approx(20.0)
    assert regions[0].bbox.width == pytest.approx(100.0)
    assert regions[0].bbox.height == pytest.approx(40.0)
    assert regions[1].class_name == "table"  # DEFAULT_CLASSES[3]


def test_detect_passes_confidence_threshold_to_engine(image_path):
    fake_model = _FakeYoloModel([_FakeResult(boxes=_FakeBoxes([]))])
    detector = LayoutDetector(model=fake_model, confidence_threshold=0.4)

    detector.detect(image_path)

    assert fake_model.calls == [(str(image_path), 0.4)]


def test_detect_with_no_boxes_returns_empty_list(image_path):
    fake_model = _FakeYoloModel([_FakeResult(boxes=_FakeBoxes([]))])
    detector = LayoutDetector(model=fake_model)

    assert detector.detect(image_path) == []


def test_unknown_class_id_falls_back_to_its_numeric_string(image_path):
    fake_model = _FakeYoloModel(
        [
            _FakeResult(
                boxes=_FakeBoxes(
                    [_FakeBox(cls=[99], conf=[0.5], xyxy=[[0.0, 0.0, 10.0, 10.0]])]
                )
            )
        ]
    )
    detector = LayoutDetector(model=fake_model)

    regions = detector.detect(image_path)

    assert regions[0].class_name == "99"


def test_missing_file_raises_layout_detection_error(tmp_path):
    detector = LayoutDetector(model=_FakeYoloModel([]))
    with pytest.raises(LayoutDetectionError, match="No such image"):
        detector.detect(tmp_path / "nope.png")


def test_no_weights_and_no_model_raises_unavailable(image_path):
    detector = LayoutDetector(weights_path=None)
    with pytest.raises(LayoutDetectionUnavailable, match="no layout-detection model"):
        detector.detect(image_path)


def test_nonexistent_weights_path_raises_unavailable(tmp_path, image_path):
    detector = LayoutDetector(weights_path=tmp_path / "does-not-exist.pt")
    with pytest.raises(LayoutDetectionUnavailable, match="weights not found"):
        detector.detect(image_path)


def test_engine_internal_failure_raises_layout_detection_error(image_path):
    class _BrokenModel:
        def __call__(self, image_path, conf=None):
            raise RuntimeError("simulated engine crash")

    detector = LayoutDetector(model=_BrokenModel())

    with pytest.raises(LayoutDetectionError, match="layout detection failed"):
        detector.detect(image_path)


def test_malformed_box_raises_layout_detection_error(image_path):
    fake_model = _FakeYoloModel(
        [_FakeResult(boxes=_FakeBoxes([_FakeBox(cls=[], conf=[0.5], xyxy=[[0, 0, 1, 1]])]))]
    )
    detector = LayoutDetector(model=fake_model)

    with pytest.raises(LayoutDetectionError, match="malformed detection box"):
        detector.detect(image_path)


def test_region_round_trips_through_dict():
    from document_core.schema import BoundingBox

    region = DetectedRegion(
        class_name="table", confidence=0.7, bbox=BoundingBox(x=1, y=2, width=3, height=4)
    )
    assert DetectedRegion.from_dict(region.to_dict()) == region
