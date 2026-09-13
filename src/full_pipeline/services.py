"""Persists a full-pipeline run's result against a document (issue #10
acceptance criterion: "Pipeline result includes output from all three
stages"), via the shared `document_core.storage.DocumentStore.save_artifact`.

Reuses `documents.services.get_document_store()` for the settings-backed
store lookup, same as every other feature app's services module.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from documents.services import get_document_store
from full_pipeline.pipeline import PipelineResult, run_full_pipeline

PIPELINE_ARTIFACT_NAME = "full_pipeline_result"


def run_pipeline_and_persist(doc_id: str, extraction_backend_key: str) -> PipelineResult:
    """Run the full pipeline against `doc_id`'s stored file and persist its
    result.

    Raises `KeyError` if `doc_id` isn't a document that was actually
    uploaded.
    """
    store = get_document_store()
    record = store.get(doc_id)
    if record is None:
        raise KeyError(f"No such document: {doc_id}")

    result = run_full_pipeline(doc_id, Path(record.stored_path), extraction_backend_key)
    store.save_artifact(doc_id, PIPELINE_ARTIFACT_NAME, _to_artifact_dict(result, extraction_backend_key))
    return result


def get_persisted_pipeline_result(doc_id: str) -> dict | None:
    """Read back the persisted full-pipeline artifact for `doc_id`, or
    `None` if `run_pipeline_and_persist` hasn't been run for it yet."""
    store = get_document_store()
    return store.load_artifact(doc_id, PIPELINE_ARTIFACT_NAME)


def _to_artifact_dict(result: PipelineResult, extraction_backend_key: str) -> dict:
    return {
        "extraction_backend_key": extraction_backend_key,
        "classification": (asdict(result.classification) if result.classification else None),
        "extraction": (result.extraction.model_dump() if result.extraction else None),
        "validation": result.validation.to_dict(),
        "stage_errors": dict(result.stage_errors),
    }
