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

ARTIFACT_NAME = "layout_regions"


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
        doc_id, ARTIFACT_NAME, {"regions": [r.to_dict() for r in regions]}
    )
    return regions


def get_persisted_regions(doc_id: str) -> list[DetectedRegion] | None:
    """Read back previously persisted detection output for `doc_id`, or
    `None` if detection hasn't been run for it yet."""
    store = get_document_store()
    data = store.load_artifact(doc_id, ARTIFACT_NAME)
    if data is None:
        return None
    return [DetectedRegion.from_dict(r) for r in data["regions"]]
