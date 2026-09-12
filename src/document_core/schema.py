"""
Structured-JSON output schema shared by every feature app (`layout_ocr`,
`classify_review`, and the chained `full_pipeline` mode) — README.md
Section 2/4. Pure Pydantic, no framework import, so any feature app can
produce/consume it regardless of its own stack.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DocumentStatus = Literal["uploaded", "processing", "auto_accepted", "needs_review", "failed"]


class BoundingBox(BaseModel):
    """Pixel coordinates of a detected region, top-left origin."""

    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)


class ExtractedField(BaseModel):
    """One field pulled out of a document (e.g. an invoice line, a form
    value) — the unit both `layout_ocr` and `classify_review` produce and
    consume."""

    name: str = Field(min_length=1)
    value: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox | None = None


class ExtractionResult(BaseModel):
    """The one structured-JSON shape every feature app writes into.

    `document_type` and `fields` are filled in progressively by later
    phases (layout+OCR fills `fields`, classify+review fills
    `document_type` + validation-driven `status`); Phase 1 only needs the
    shape to exist and validate.
    """

    document_id: str = Field(min_length=1)
    document_type: str | None = None
    status: DocumentStatus = "uploaded"
    fields: list[ExtractedField] = Field(default_factory=list)
    overall_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
