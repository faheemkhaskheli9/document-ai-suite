"""Tests for `layout_ocr.dataset`'s COCO->YOLO conversion -- ported from
`document-ai-yolo-ocr`'s `tests/test_dataset.py` (issue #4) unchanged except
for the import path."""
import logging

import pytest
import yaml
from PIL import Image

from layout_ocr.dataset import (
    coco_bbox_to_yolo,
    convert_coco_to_yolo_labels,
    split_image_ids,
    write_yolo_dataset,
)

SAMPLE_COCO = {
    "categories": [
        {"id": 1, "name": "text"},
        {"id": 2, "name": "title"},
        {"id": 3, "name": "table"},
    ],
    "images": [
        {"id": 1, "file_name": "a.png", "width": 100, "height": 200},
        {"id": 2, "file_name": "b.png", "width": 100, "height": 200},
    ],
    "annotations": [
        {"id": 1, "image_id": 1, "category_id": 2, "bbox": [10, 10, 20, 20]},
        {"id": 2, "image_id": 1, "category_id": 1, "bbox": [0, 100, 100, 50]},
    ],
}


def test_coco_bbox_to_yolo_center_and_normalization():
    xc, yc, w, h = coco_bbox_to_yolo([10, 10, 20, 40], img_width=100, img_height=200)
    assert xc == pytest.approx(0.20)
    assert yc == pytest.approx(0.15)
    assert w == pytest.approx(0.20)
    assert h == pytest.approx(0.20)


@pytest.mark.parametrize(
    "bbox,width,height",
    [
        ([0, 0, 0, 10], 100, 100),  # zero width
        ([0, 0, 10, 0], 100, 100),  # zero height
        ([0, 0, 10, 10], 0, 100),  # invalid image size
        ([0, 0, 10, 10], 100, -1),  # negative image size
    ],
)
def test_coco_bbox_to_yolo_rejects_degenerate_input(bbox, width, height):
    with pytest.raises(ValueError):
        coco_bbox_to_yolo(bbox, width, height)


def test_convert_coco_to_yolo_labels_maps_classes_and_keeps_empty_images():
    classes = ["text", "title", "table"]
    per_image, resolved_classes = convert_coco_to_yolo_labels(SAMPLE_COCO, classes)

    assert resolved_classes == classes
    # image 2 has no annotations but must still appear (with an empty box list)
    assert set(per_image.keys()) == {1, 2}
    file_name, boxes = per_image[1]
    assert file_name == "a.png"
    assert len(boxes) == 2
    class_ids = sorted(b.class_id for b in boxes)
    assert class_ids == [0, 1]  # text=0, title=1

    _, empty_boxes = per_image[2]
    assert empty_boxes == []


def test_convert_coco_to_yolo_labels_skips_bad_records(caplog):
    coco = {
        "categories": [{"id": 1, "name": "text"}],
        "images": [{"id": 1, "file_name": "a.png", "width": 100, "height": 100}],
        "annotations": [
            # references an image that doesn't exist
            {"id": 1, "image_id": 999, "category_id": 1, "bbox": [0, 0, 10, 10]},
            # unmapped category
            {"id": 2, "image_id": 1, "category_id": 42, "bbox": [0, 0, 10, 10]},
            # degenerate bbox
            {"id": 3, "image_id": 1, "category_id": 1, "bbox": [0, 0, 0, 0]},
            # valid
            {"id": 4, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},
        ],
    }
    with caplog.at_level(logging.WARNING):
        per_image, _ = convert_coco_to_yolo_labels(coco, ["text"])

    _, boxes = per_image[1]
    assert len(boxes) == 1


def test_split_image_ids_is_deterministic_and_covers_all_ids():
    ids = list(range(10))
    train_a, val_a = split_image_ids(ids, val_split=0.3, seed=13)
    train_b, val_b = split_image_ids(ids, val_split=0.3, seed=13)
    assert (train_a, val_a) == (train_b, val_b)
    assert sorted(train_a + val_a) == ids
    assert len(val_a) == 3


def test_split_image_ids_rejects_invalid_fraction():
    with pytest.raises(ValueError):
        split_image_ids([1, 2, 3], val_split=1.5)


def _write_sample_dataset(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    for name in ("a.png", "b.png"):
        Image.new("RGB", (100, 200), "white").save(images_dir / name)

    coco_path = tmp_path / "coco.json"
    import json

    coco_path.write_text(json.dumps(SAMPLE_COCO), encoding="utf-8")
    return coco_path, images_dir


def test_write_yolo_dataset_end_to_end(tmp_path):
    coco_path, images_dir = _write_sample_dataset(tmp_path)
    output_dir = tmp_path / "yolo_out"

    counts = write_yolo_dataset(
        coco_path=coco_path,
        images_dir=images_dir,
        output_dir=output_dir,
        classes=["text", "title", "table"],
        val_split=0.5,
        seed=13,
    )

    assert counts["train"] + counts["val"] == 2
    assert (output_dir / "dataset.yaml").exists()

    dataset_yaml = yaml.safe_load((output_dir / "dataset.yaml").read_text())
    assert dataset_yaml["names"] == {0: "text", 1: "title", 2: "table"}

    all_label_files = list((output_dir).rglob("*.txt"))
    assert len(all_label_files) == 2
    all_images = list((output_dir).rglob("*.png"))
    assert len(all_images) == 2

    # a.png has 2 boxes -> its label file should have exactly 2 lines
    a_label = next(f for f in all_label_files if f.stem == "a")
    lines = a_label.read_text().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        parts = line.split()
        assert len(parts) == 5
        class_id = int(parts[0])
        assert class_id in (0, 1)
        for coord in parts[1:]:
            assert 0.0 <= float(coord) <= 1.0


def test_write_yolo_dataset_missing_image_is_skipped_not_fatal(tmp_path, caplog):
    coco_path, images_dir = _write_sample_dataset(tmp_path)
    (images_dir / "a.png").unlink()  # simulate a missing source image
    output_dir = tmp_path / "yolo_out"

    with caplog.at_level(logging.WARNING):
        counts = write_yolo_dataset(
            coco_path=coco_path,
            images_dir=images_dir,
            output_dir=output_dir,
            classes=["text", "title", "table"],
            val_split=0.0,
            seed=13,
        )

    # b.png is written; a.png is skipped and logged rather than crashing the run
    assert counts["train"] == 1
    assert (output_dir / "dataset.yaml").exists()
