"""Feature-app registry backing the dashboard's "pick a feature" shell
(README.md Section 2/5). Mirrors the string-keyed registry pattern already
used in this portfolio for OCR backends (`document_core.ocr`) and for whole
imaging tasks in `medical-imaging-suite`'s `BaseImagingTask` / `@register_task`
-- applied here one layer up, to whole feature apps.

Phase 2/3/4 land `layout_ocr`, `classify_review`, and `full_pipeline` by
registering a `FeatureCard` each with a real `url_name`; until then they're
registered with `url_name=None` so the dashboard can show them as
"coming soon" without a dangling/guessed URL. Registering the entry now
(instead of hardcoding the card list in a template) means a later phase
plugs in by adding one `register_feature(...)` call, not by editing the
dashboard template.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureCard:
    key: str
    label: str
    description: str
    url_name: str | None = None  # None => not implemented yet ("coming soon")

    @property
    def is_available(self) -> bool:
        return self.url_name is not None


_REGISTRY: dict[str, FeatureCard] = {}


def register_feature(card: FeatureCard) -> FeatureCard:
    if card.key in _REGISTRY:
        raise ValueError(f"duplicate feature key: {card.key!r}")
    _REGISTRY[card.key] = card
    return card


def all_features() -> list[FeatureCard]:
    """Registered features in registration order (Phase 1's `documents`
    app first, then the ported feature apps in their planned build order --
    README.md Section 5)."""
    return list(_REGISTRY.values())


register_feature(
    FeatureCard(
        key="documents",
        label="Documents",
        description=(
            "Upload a document and track its status -- the shared upload "
            "path and per-user history every feature below builds on."
        ),
        url_name="documents:list",
    )
)
register_feature(
    FeatureCard(
        key="layout_ocr",
        label="Layout + OCR extraction",
        description=(
            "YOLO layout detection, per-region OCR, and field mapping to "
            "structured JSON (invoices, contracts, forms)."
        ),
    )
)
register_feature(
    FeatureCard(
        key="classify_review",
        label="Classify + Review",
        description=(
            "Document classification, validation rules, confidence "
            "scoring, and a human-review queue for low-confidence "
            "extractions."
        ),
    )
)
register_feature(
    FeatureCard(
        key="full_pipeline",
        label="Full pipeline",
        description=(
            "Chains Layout + OCR into Classify + Review: classify, "
            "extract, validate/score, and route to review only when "
            "confidence is low."
        ),
    )
)
