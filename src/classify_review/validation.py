"""
Validation rules and confidence scoring -- issue #8 (Phase 3, `classify_review`
feature app).

Runs after classification (`classify_review.classifier`) and an extraction
backend (`document_core.extraction`) have both produced their output for a
document, and decides what happens to it next:

- a document that *fails a validation rule* (missing/empty required data, or
  classification and extraction disagreeing about the document type) is
  flagged `status="failed"` -- something is structurally wrong with the
  result, independent of how confident either stage was.
- a document that *passes* validation but scores below the acceptance
  threshold is `status="needs_review"` -- nothing is wrong with it, a human
  should just double check it.
- a document that passes validation and scores at/above the threshold is
  `status="auto_accepted"`.

This distinction (issue #8 acceptance criterion 3) is why `ValidationResult`
carries both `passed` and `status` rather than collapsing them into one
field -- collapsing them would make "failed validation" and "merely low
confidence" indistinguishable to a caller building a review queue (issue #9).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from classify_review.classifier import ClassificationResult
from document_core.schema import DocumentStatus, ExtractionResult

# At/above this overall confidence, a document that passed validation is
# auto-accepted rather than routed to human review. Deliberately a plain
# module constant (not user-configurable yet) -- same "simple heuristic, not
# a trained/calibrated model" trade-off as classifier.py's keyword matching.
AUTO_ACCEPT_CONFIDENCE_THRESHOLD = 0.75


@dataclass
class ValidationResult:
    """What `validate()` returns."""

    passed: bool
    overall_confidence: float
    status: DocumentStatus
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "overall_confidence": self.overall_confidence,
            "status": self.status,
            "failures": list(self.failures),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ValidationResult":
        return cls(
            passed=data["passed"],
            overall_confidence=data["overall_confidence"],
            status=data["status"],
            failures=list(data.get("failures", [])),
        )


def _score(classification: ClassificationResult | None, extraction: ExtractionResult | None) -> float:
    """Combine whatever confidence signals are available into one overall
    score. Neither signal is a calibrated probability (classification's is a
    keyword-match strength, extraction's may be Tesseract's mean word
    confidence *or* an LLM's self-report -- see docs/architecture.md) so this
    is a plain mean of whichever are present, not a weighted model."""
    scores = []
    if classification is not None:
        scores.append(classification.confidence)
    if extraction is not None:
        if extraction.overall_confidence is not None:
            scores.append(extraction.overall_confidence)
        elif extraction.fields:
            scores.append(sum(f.confidence for f in extraction.fields) / len(extraction.fields))
    return sum(scores) / len(scores) if scores else 0.0


def validate(
    classification: ClassificationResult | None,
    extraction: ExtractionResult | None,
) -> ValidationResult:
    """Apply validation rules to `classification`/`extraction` output and
    compute an overall confidence score.

    Either argument may be `None` (e.g. classification ran but extraction
    hasn't yet) -- a missing extraction result is itself a validation
    failure, since there is nothing to hand to a review queue.
    """
    failures: list[str] = []

    if extraction is None:
        failures.append("no_extraction_result")
    else:
        if not extraction.fields:
            failures.append("no_fields_extracted")
        if any(not f.value.strip() for f in extraction.fields):
            failures.append("empty_field_value")

        if (
            classification is not None
            and classification.document_type is not None
            and extraction.document_type is not None
            and classification.document_type.lower() != extraction.document_type.lower()
        ):
            failures.append("document_type_mismatch")

    passed = not failures
    overall_confidence = _score(classification, extraction)

    if not passed:
        status: DocumentStatus = "failed"
    elif overall_confidence >= AUTO_ACCEPT_CONFIDENCE_THRESHOLD:
        status = "auto_accepted"
    else:
        status = "needs_review"

    return ValidationResult(
        passed=passed,
        overall_confidence=overall_confidence,
        status=status,
        failures=failures,
    )
