"""Persists `layout_ocr`'s detected regions against a document (issue #4
acceptance criterion: "Detection output is persisted per document"), via the
shared `document_core.storage.DocumentStore.save_artifact`.

Reuses `documents.services.get_document_store()` for the settings-backed
store lookup rather than re-reading `DOCUMENT_STORE_ROOT` here -- one place
owns "how do we get a DocumentStore instance" (robustness rule 15: avoid a
second copy of logic that must stay in sync with the original).
"""
from __future__ import annotations

from pathlib import Path

from documents.services import get_document_store
from layout_ocr.detection import DetectedRegion, LayoutDetector
from layout_ocr.fields import FieldExtraction, extract_fields_from_regions

REGIONS_ARTIFACT_NAME = "layout_regions"
FIELDS_ARTIFACT_NAME = "layout_fields"


def detect_and_persist(
    doc_id: str, image_path: str | Path, detector: LayoutDetector | None = None
) -> list[DetectedRegion]:
    """Run layout detection on `image_path` and persist the result against
    `doc_id`.

    Raises `KeyError` if `doc_id` isn't a document that was actually
    uploaded, so detections can never be persisted against a document that
    doesn't exist.
    """
    store = get_document_store()
    if store.get(doc_id) is None:
        raise KeyError(f"No such document: {doc_id}")

    detector = detector or LayoutDetector()
    regions = detector.detect(image_path)
    store.save_artifact(
        doc_id, REGIONS_ARTIFACT_NAME, {"regions": [r.to_dict() for r in regions]}
    )
    return regions


def get_persisted_regions(doc_id: str) -> list[DetectedRegion] | None:
    """Read back previously persisted detection output for `doc_id`, or
    `None` if detection hasn't been run for it yet."""
    store = get_document_store()
    data = store.load_artifact(doc_id, REGIONS_ARTIFACT_NAME)
    if data is None:
        return None
    return [DetectedRegion.from_dict(r) for r in data["regions"]]


def run_layout_ocr(
    doc_id: str,
    image_path: str | Path,
    detector: LayoutDetector | None = None,
    ocr_backend_key: str = "tesseract",
) -> list[FieldExtraction]:
    """Full layout+OCR run for one document: detect regions (issue #4), OCR
    each one (issue #5), and persist both artifacts.

    Raises `KeyError` if `doc_id` isn't a document that was actually
    uploaded. A region that fails OCR is still persisted (with `ocr_error`
    set) rather than dropped -- see `layout_ocr.fields.extract_fields_from_regions`.
    """
    regions = detect_and_persist(doc_id, image_path, detector)
    fields = extract_fields_from_regions(image_path, regions, ocr_backend_key)

    store = get_document_store()
    store.save_artifact(
        doc_id, FIELDS_ARTIFACT_NAME, {"fields": [f.to_dict() for f in fields]}
    )
    return fields


def get_persisted_fields(doc_id: str) -> list[FieldExtraction] | None:
    """Read back previously persisted per-field OCR output for `doc_id`, or
    `None` if `run_layout_ocr` hasn't been run for it yet."""
    store = get_document_store()
    data = store.load_artifact(doc_id, FIELDS_ARTIFACT_NAME)
    if data is None:
        return None
    return [FieldExtraction.from_dict(f) for f in data["fields"]]
