"""COCO-to-YOLO conversion for document layout datasets.

Ported from `document-ai-yolo-ocr`'s `src/yolo_ocr/dataset.py` (issue #4;
docs/architecture.md: "Reuse document-ai-yolo-ocr's existing
src/yolo_ocr/dataset.py COCO -> YOLO conversion CLI for layout-model dataset
prep rather than re-implementing it"). Logic is unchanged from the original
-- only the module's new home (`layout_ocr`, the ported feature app) and this
docstring differ.

This module does not download any dataset itself -- PubLayNet/DocLayNet are
tens of gigabytes and licensed for manual download (see
`docs/architecture.md` for links and license notes). It only reformats
COCO-style annotations (the format both datasets ship in) that are already
on disk into YOLO's per-image ``.txt`` label format, and splits images into
train/val directories.

YOLO label line format: ``<class_id> <x_center> <y_center> <width> <height>``
with all four numeric values normalized to [0, 1] relative to image size.
"""
from __future__ import annotations

import json
import logging
import os
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

logger = logging.getLogger(__name__)

# PubLayNet's category set -- also a reasonable default for DocLayNet-style
# sources. Callers may pass a different ordered class list to `classes=`.
DEFAULT_CLASSES = ["text", "title", "list", "table", "figure"]


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_line(self) -> str:
        return (
            f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} "
            f"{self.width:.6f} {self.height:.6f}"
        )


def coco_bbox_to_yolo(
    bbox: Iterable[float], img_width: float, img_height: float
) -> tuple[float, float, float, float]:
    """Convert a COCO ``[x, y, w, h]`` absolute-pixel bbox to normalized YOLO coords.

    COCO's (x, y) is the top-left corner; YOLO wants the box center.
    """
    if img_width <= 0 or img_height <= 0:
        raise ValueError(f"Invalid image size {img_width}x{img_height}")
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        raise ValueError(f"Degenerate bbox with non-positive size: {list(bbox)}")
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    return x_center, y_center, w / img_width, h / img_height


def load_category_mapping(
    coco: dict, classes: list[str] | None = None
) -> tuple[dict[int, int], list[str]]:
    """Map COCO ``category_id`` -> YOLO class index.

    If ``classes`` (an ordered list of category names) is given, categories
    are mapped to that order and any category not present in it is dropped
    (and logged) rather than silently included with a wrong index. Otherwise
    categories are mapped in ascending ``category_id`` order.
    """
    categories = {c["id"]: c["name"] for c in coco.get("categories", [])}
    if classes is None:
        classes = [name for _, name in sorted(categories.items())]
    name_to_index = {name: i for i, name in enumerate(classes)}
    mapping: dict[int, int] = {}
    for cat_id, name in categories.items():
        if name in name_to_index:
            mapping[cat_id] = name_to_index[name]
        else:
            logger.warning(
                "Dropping unmapped category %r (id=%s) -- not in class list %s",
                name,
                cat_id,
                classes,
            )
    return mapping, classes


def convert_coco_to_yolo_labels(
    coco: dict, classes: list[str] | None = None
) -> tuple[dict[int, tuple[str, list[YoloBox]]], list[str]]:
    """Convert a loaded COCO annotation dict into per-image YOLO boxes.

    Returns ``{image_id: (file_name, [YoloBox, ...])}`` (every image in the
    COCO file appears, even with an empty box list) plus the resolved class
    list. Malformed or unmapped annotations are skipped and logged rather
    than aborting the whole conversion -- one bad record shouldn't poison
    every other image in the dataset.
    """
    mapping, classes = load_category_mapping(coco, classes)
    images = {img["id"]: img for img in coco.get("images", [])}
    result: dict[int, tuple[str, list[YoloBox]]] = {
        image_id: (img["file_name"], []) for image_id, img in images.items()
    }

    skipped = 0
    for ann in coco.get("annotations", []):
        image_id = ann.get("image_id")
        img = images.get(image_id)
        if img is None:
            logger.warning(
                "Annotation %s references unknown image_id=%s -- skipping",
                ann.get("id"),
                image_id,
            )
            skipped += 1
            continue
        class_id = mapping.get(ann.get("category_id"))
        if class_id is None:
            skipped += 1
            continue
        try:
            xc, yc, w, h = coco_bbox_to_yolo(ann["bbox"], img["width"], img["height"])
        except (KeyError, ValueError) as exc:
            logger.warning("Skipping malformed annotation %s: %s", ann.get("id"), exc)
            skipped += 1
            continue
        result[image_id][1].append(YoloBox(class_id, xc, yc, w, h))

    if skipped:
        logger.info("Skipped %d annotation(s) during COCO->YOLO conversion", skipped)
    return result, classes


def split_image_ids(
    image_ids: Iterable[int], val_split: float = 0.2, seed: int = 13
) -> tuple[list[int], list[int]]:
    """Deterministically split image ids into (train_ids, val_ids)."""
    if not 0 <= val_split < 1:
        raise ValueError(f"val_split must be in [0, 1), got {val_split}")
    ids = sorted(image_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = round(len(ids) * val_split)
    val_ids = set(ids[:n_val])
    train_ids = [i for i in ids if i not in val_ids]
    val_ids_ordered = [i for i in ids if i in val_ids]
    return train_ids, val_ids_ordered


def write_yolo_dataset(
    coco_path: str | Path,
    images_dir: str | Path,
    output_dir: str | Path,
    classes: list[str] | None = None,
    val_split: float = 0.2,
    seed: int = 13,
    copy_images: bool = True,
) -> dict[str, int]:
    """Convert a COCO layout dataset to a YOLO-ready dataset tree.

    Writes, under ``output_dir``::

        train/images/*, train/labels/*.txt
        val/images/*,   val/labels/*.txt
        dataset.yaml   (Ultralytics-style dataset config)

    ``dataset.yaml`` is written last, atomically, so its presence is a
    reliable "this dataset is complete" marker even if the process is
    interrupted midway through copying images/labels.

    Returns a ``{"train": n, "val": n}`` count of images written per split.
    """
    coco_path = Path(coco_path)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)

    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    per_image, classes = convert_coco_to_yolo_labels(coco, classes)

    train_ids, val_ids = split_image_ids(per_image.keys(), val_split, seed)

    counts = {"train": 0, "val": 0}
    for split, ids in (("train", train_ids), ("val", val_ids)):
        img_out = output_dir / split / "images"
        lbl_out = output_dir / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for image_id in ids:
            file_name, boxes = per_image[image_id]
            if copy_images:
                src_image = images_dir / file_name
                if not src_image.exists():
                    logger.warning("Source image missing, skipping: %s", src_image)
                    continue
                shutil.copy2(src_image, img_out / Path(file_name).name)
            label_path = lbl_out / (Path(file_name).stem + ".txt")
            label_lines = "\n".join(box.to_line() for box in boxes)
            label_path.write_text(
                label_lines + ("\n" if boxes else ""), encoding="utf-8"
            )
            counts[split] += 1

    dataset_yaml = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": {i: name for i, name in enumerate(classes)},
    }
    _write_yaml_atomic(output_dir / "dataset.yaml", dataset_yaml)

    return counts


def _write_yaml_atomic(path: Path, data: dict) -> None:
    """Write ``data`` as YAML to ``path``, atomically (temp file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.remove(tmp_name)
        except OSError:
            pass
        raise
