"""
Pluggable document-extraction backend abstraction shared by every feature app
-- README.md Section 2/4 ("...either by custom-trained model, generic OCR
models, or a multimodal LLM model").

This sits one layer above `document_core.ocr`: an OCR backend recognizes text
in one region of a page; an *extraction* backend produces the whole
`document_core.schema.ExtractionResult` for one document. Two shapes of
backend fit this contract:

- pipeline backends -- layout detection (e.g. YOLO) + `document_core.ocr` per
  detected region + field mapping. This is `layout_ocr`'s engine (Phase 2);
  it doesn't need its own class here, it composes the OCR registry directly.
- direct backends -- a single multimodal-LLM call that reads the whole page
  image/PDF and returns structured fields directly, skipping layout
  detection entirely. `LLMVisionExtractionBackend` below is the first one.

Feature apps call `extract_fields(backend_key, ...)` (or
`get_extraction_backend(key)` directly) and never import a concrete engine
(Claude, YOLO, ...) themselves -- swapping which engine a key maps to, or
adding a new one, never touches a feature app. Mirrors the string-keyed
registry pattern in `document_core.ocr` (itself mirroring
`medical-imaging-suite/imaging_core/registry.py`'s `BaseImagingTask` /
`@register_task`).

Concrete backends must do any heavy/native import (here: `anthropic`) lazily
inside `extract()`, not at module import time, so importing this module never
requires the SDK or an API key to be present.
"""
from __future__ import annotations

import base64
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import ValidationError

from document_core.schema import ExtractedField, ExtractionResult

# Media types Claude's vision/document input accepts directly. TIFF isn't one
# of them -- a `.tiff` upload must be converted to PNG/JPEG upstream before it
# reaches this backend; recognize() raises rather than silently mis-reading it
# as some other type (robustness rule: respect the format's actual domain).
_IMAGE_MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
_PDF_MEDIA_TYPE = {".pdf": "application/pdf"}

DEFAULT_LLM_EXTRACTION_MODEL = "claude-opus-5"

# Structured-output schema for the LLM call. `document_type` is required
# (empty string means "the model didn't recognize a type") rather than
# nullable, since strict JSON-schema output doesn't reliably support
# nullable properties across models.
_LLM_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "document_type": {"type": "string"},
        "fields": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "value": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["name", "value", "confidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["document_type", "fields"],
    "additionalProperties": False,
}

_EXTRACTION_PROMPT = (
    "Extract every field you can identify from this document (e.g. invoice "
    "number, dates, line items, totals, names, addresses -- whatever the "
    "document actually contains) as structured fields. For each field give "
    "your own confidence from 0.0 to 1.0 that the value is correct and "
    "correctly attributed -- this is your self-assessment, not a calibrated "
    "probability. If you cannot classify the document type, return an empty "
    "string for document_type rather than guessing."
)


class ExtractionBackendError(RuntimeError):
    """Extraction failed -- bad input, or the underlying engine/model errored."""


class ExtractionBackendUnavailable(ExtractionBackendError):
    """The engine behind this backend isn't installed/configured in this
    environment (e.g. the `anthropic` package isn't installed, or no API
    credential is configured). Raised instead of silently returning an empty
    result, so a missing/misconfigured engine fails loudly."""


class ExtractionBackend(ABC):
    """Contract every concrete document-extraction engine wrapper must
    implement."""

    key: str
    label: str

    @abstractmethod
    def extract(self, document_path: Path, document_id: str) -> ExtractionResult:
        """Run extraction on `document_path` and return the full structured
        result for `document_id`.

        Must raise `ExtractionBackendError` (or `ExtractionBackendUnavailable`
        if the underlying engine isn't available) rather than returning a
        fabricated result on failure.
        """


_REGISTRY: dict[str, ExtractionBackend] = {}


def register_extraction_backend(backend_cls: type[ExtractionBackend]) -> type[ExtractionBackend]:
    """Class decorator: instantiate `backend_cls` and register it by `.key`."""
    instance = backend_cls()
    if not instance.key:
        raise ValueError(f"{backend_cls.__name__}.key must be a non-empty string")
    if instance.key in _REGISTRY:
        raise ValueError(f"duplicate extraction backend key: {instance.key!r}")
    _REGISTRY[instance.key] = instance
    return backend_cls


def get_extraction_backend(key: str) -> ExtractionBackend:
    try:
        return _REGISTRY[key]
    except KeyError:
        raise KeyError(f"no extraction backend registered for key {key!r}") from None


def all_extraction_backends() -> list[ExtractionBackend]:
    """All registered backends, sorted by label for stable UI ordering."""
    return sorted(_REGISTRY.values(), key=lambda b: b.label)


def is_registered(key: str) -> bool:
    return key in _REGISTRY


def extract_fields(backend_key: str, document_path: Path, document_id: str) -> ExtractionResult:
    """Convenience entry point every feature app should call instead of
    reaching for `get_extraction_backend` + `.extract` directly. Swapping
    which engine `backend_key` maps to (or adding a new key) never requires a
    change here or in any caller."""
    return get_extraction_backend(backend_key).extract(document_path, document_id)


@register_extraction_backend
class LLMVisionExtractionBackend(ExtractionBackend):
    """Sends the whole document (image or PDF) to a multimodal LLM (Claude,
    by default) in one call and asks it to return structured fields
    directly -- no layout-detection or per-region OCR step. Trades the
    precision/cost of the YOLO+OCR pipeline for zero training data and one
    call per document.

    The model, not a calibrated OCR engine, produces each field's
    `confidence` -- treat it as a self-report, not a measured probability;
    `layout_ocr`'s Tesseract/PaddleOCR-backed confidences are not directly
    comparable to these.
    """

    key = "llm_vision"
    label = "Multimodal LLM (Claude vision, no layout model)"

    def extract(self, document_path: Path, document_id: str) -> ExtractionResult:
        document_path = Path(document_path)
        if not document_path.exists():
            raise ExtractionBackendError(f"No such document: {document_path}")

        content_block = self._build_content_block(document_path)

        try:
            import anthropic
        except ImportError as exc:
            raise ExtractionBackendUnavailable(
                "the 'anthropic' package is not installed; add it to requirements.txt"
            ) from exc

        model = os.environ.get("LLM_EXTRACTION_MODEL", DEFAULT_LLM_EXTRACTION_MODEL)

        try:
            client = anthropic.Anthropic()
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": [content_block, {"type": "text", "text": _EXTRACTION_PROMPT}],
                    }
                ],
                output_config={"format": {"type": "json_schema", "schema": _LLM_OUTPUT_SCHEMA}},
            )
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            raise ExtractionBackendUnavailable(
                "Anthropic API credentials are missing or invalid "
                "(set ANTHROPIC_API_KEY)"
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise ExtractionBackendError(f"could not reach the Anthropic API: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise ExtractionBackendError(
                f"Anthropic API request failed ({exc.status_code}): {exc.message}"
            ) from exc

        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None)
            raise ExtractionBackendError(
                f"the model declined to process this document (category: {category})"
            )

        # `output_config.format` guarantees syntactically valid JSON matching
        # the schema on success, but a 200 response still isn't proof the
        # *content* is usable -- parse and validate before trusting it rather
        # than assuming the shape is right.
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise ExtractionBackendError("model response contained no text content")

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ExtractionBackendError(f"model returned malformed JSON: {exc}") from exc

        try:
            fields = [
                ExtractedField(
                    name=f["name"],
                    value=f["value"],
                    confidence=max(0.0, min(1.0, float(f["confidence"]))),
                )
                for f in payload.get("fields", [])
            ]
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            raise ExtractionBackendError(f"model returned an invalid field: {exc}") from exc

        document_type = payload.get("document_type") or None
        overall_confidence = (
            sum(f.confidence for f in fields) / len(fields) if fields else None
        )

        try:
            return ExtractionResult(
                document_id=document_id,
                document_type=document_type,
                fields=fields,
                overall_confidence=overall_confidence,
            )
        except ValidationError as exc:
            raise ExtractionBackendError(f"could not build extraction result: {exc}") from exc

    @staticmethod
    def _build_content_block(document_path: Path) -> dict:
        ext = document_path.suffix.lower()
        data = base64.standard_b64encode(document_path.read_bytes()).decode("ascii")

        if ext in _IMAGE_MEDIA_TYPES:
            return {
                "type": "image",
                "source": {"type": "base64", "media_type": _IMAGE_MEDIA_TYPES[ext], "data": data},
            }
        if ext in _PDF_MEDIA_TYPE:
            return {
                "type": "document",
                "source": {"type": "base64", "media_type": _PDF_MEDIA_TYPE[ext], "data": data},
            }
        raise ExtractionBackendError(
            f"unsupported file type '{ext or '(none)'}' for LLM vision extraction "
            f"-- supported: {', '.join(sorted({**_IMAGE_MEDIA_TYPES, **_PDF_MEDIA_TYPE}))}"
        )
