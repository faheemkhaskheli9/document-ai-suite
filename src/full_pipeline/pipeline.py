"""
Full-pipeline chaining logic -- issue #10 (Phase 4, `full_pipeline` feature
app).

Chains `classify_review`'s classifier -> a `document_core.extraction`
backend -> `classify_review`'s validation/confidence scoring into one run
against a single uploaded document -- docs/architecture.md's "full_pipeline
mode" component: "chains the selected extraction backend's output into
classify_review's validation/confidence/review-routing stage".

Neither the classification stage nor the extraction stage failing aborts the
whole run: each stage's own failure is captured on
`PipelineResult.stage_errors` instead, so a caller can always tell a stage
that ran-and-returned-nothing apart from a stage that never ran, and a
broken extraction doesn't also block classification's independent
contribution to the confidence score. The validation stage always runs --
`classify_review.validation.validate` already treats a missing extraction
result as a validation failure (`status="failed"`), so a `PipelineResult`
with a failed stage still comes back well-defined rather than raising past
the caller.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from classify_review.classifier import (
    ClassificationResult,
    DocumentClassificationError,
    DocumentClassifier,
)
from classify_review.validation import AUTO_ACCEPT_CONFIDENCE_THRESHOLD, ValidationResult, validate
from document_core.extraction import ExtractionBackendError, extract_fields
from document_core.schema import ExtractionResult

# Stage-name keys used in `PipelineResult.stage_errors`, spelled out once so
# callers/tests never have to repeat the raw strings.
STAGE_CLASSIFICATION = "classification"
STAGE_EXTRACTION = "extraction"


@dataclass
class PipelineResult:
    """Everything one full-pipeline run against a document produced.

    `classification`/`extraction` are `None` exactly when that stage raised
    -- `stage_errors` (keyed by the `STAGE_*` constants above) carries why,
    distinct from a stage that ran and legitimately returned an empty/low
    result. `validation` is never `None`: `classify_review.validation.validate`
    always returns a `ValidationResult`, even from two `None` inputs.
    """

    document_id: str
    classification: ClassificationResult | None
    extraction: ExtractionResult | None
    validation: ValidationResult
    stage_errors: dict[str, str] = field(default_factory=dict)

    def contributing_stages(self) -> list[str]:
        """Which stage(s) pulled the overall confidence down -- issue #11
        acceptance criterion: "Reviewer sees which pipeline stage(s)
        contributed to the low confidence."

        A stage "contributes" if it failed outright (a missing signal is the
        strongest possible drag on `classify_review.validation._score`'s
        mean) or it ran but scored below the same auto-accept threshold
        `validate()` itself uses. Meaningful regardless of the overall
        `validation.status` -- callers routing to a review UI should check
        `validation.status == "needs_review"` first (a value here doesn't by
        itself mean the document needs review; a `status="failed"` document
        skips this signal entirely in favor of `validation.failures`).
        """
        stages: list[str] = []

        if STAGE_CLASSIFICATION in self.stage_errors or self.classification is None:
            stages.append(STAGE_CLASSIFICATION)
        elif self.classification.confidence < AUTO_ACCEPT_CONFIDENCE_THRESHOLD:
            stages.append(STAGE_CLASSIFICATION)

        if STAGE_EXTRACTION in self.stage_errors or self.extraction is None:
            stages.append(STAGE_EXTRACTION)
        else:
            confidence = self.extraction.overall_confidence
            if confidence is None and self.extraction.fields:
                confidence = sum(f.confidence for f in self.extraction.fields) / len(
                    self.extraction.fields
                )
            if confidence is None or confidence < AUTO_ACCEPT_CONFIDENCE_THRESHOLD:
                stages.append(STAGE_EXTRACTION)

        return stages


def run_full_pipeline(
    document_id: str, document_path: str | Path, extraction_backend_key: str
) -> PipelineResult:
    """Run classify -> extract -> validate/score against one document, in
    that order, and return everything each stage produced.

    Does not persist anything -- see `full_pipeline.services` for the
    persistence + review-queue-routing layer that wraps this against a
    real uploaded document.
    """
    document_path = Path(document_path)
    stage_errors: dict[str, str] = {}

    classification: ClassificationResult | None = None
    try:
        classification = DocumentClassifier().classify(document_path)
    except DocumentClassificationError as exc:
        stage_errors[STAGE_CLASSIFICATION] = str(exc)

    extraction: ExtractionResult | None = None
    try:
        extraction = extract_fields(extraction_backend_key, document_path, document_id)
    except ExtractionBackendError as exc:
        stage_errors[STAGE_EXTRACTION] = str(exc)

    validation = validate(classification, extraction)

    return PipelineResult(
        document_id=document_id,
        classification=classification,
        extraction=extraction,
        validation=validation,
        stage_errors=stage_errors,
    )
