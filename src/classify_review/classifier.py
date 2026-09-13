"""
Document classifier -- issue #7 (Phase 3, `classify_review` feature app).

Classifies an uploaded document along two axes *before* either extraction
engine (`layout_ocr`/`llm_vision`) runs -- see docs/architecture.md's
pipeline: `Classification (scanned vs digital, type) -> [extraction engine]`.

- `is_scanned`: whether the document has no machine-readable text layer (a
  photographed/scanned page, or a standalone image file) vs. a digital PDF
  with an extractable text layer.
- `document_type`: a coarse label (e.g. "invoice", "contract") from keyword
  heuristics over whatever text is cheaply available at this stage -- the
  PDF's own text layer for digital PDFs, or one lightweight OCR pass for
  scanned/image input. This is intentionally cheaper than the full
  layout_ocr/llm_vision engines that run *after* classification (it exists to
  route the document, not to extract its fields), same "regex heuristics, no
  trained model" trade-off `layout_ocr/mapping.py` already documents.

README.md Section 2 frames this feature app as "ported from
`intelligent-document-processing`", but that repo's `src/` was scaffold-only
(`.gitkeep` only, no committed code, verified against
`portfolio-archived-repos/intelligent-document-processing`) -- only its
README Section 5 described the intended classifier. There is nothing to
port; this module implements that described Phase-1 classifier from scratch
against this suite's own `document_core.ocr` abstraction and a PDF-text-layer
check, matching the architecture this suite's own docs already commit to.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from document_core.ocr import OCRBackendError, OCRBackendUnavailable, extract_text

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tiff")
_PDF_EXTENSION = ".pdf"

# Below this many extracted characters, a PDF's "text layer" is treated as
# absent -- a handful of stray characters from a scanned page's metadata or a
# watermark isn't a real text layer. Matches the common scanned-PDF
# detection heuristic (a real digital page's text layer is always far longer).
MIN_DIGITAL_TEXT_CHARS = 20

# Coarse type keywords, checked in order -- first type with any hit wins.
# Deliberately simple keyword matching (mirrors layout_ocr/mapping.py's regex
# field-mapping approach), not a trained classifier.
_TYPE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("invoice", ("invoice", "bill to", "amount due")),
    ("contract", ("agreement", "contract", "terms and conditions")),
    ("receipt", ("receipt", "subtotal", "thank you for your purchase")),
    ("form", ("please fill", "signature required", "date of birth")),
)


class DocumentClassificationError(RuntimeError):
    """Classification failed outright (bad/missing input, unreadable file) --
    distinct from a document that classified successfully but with
    `document_type=None` (couldn't confidently pick a type)."""


@dataclass
class ClassificationResult:
    """What `DocumentClassifier.classify()` returns."""

    is_scanned: bool
    document_type: str | None
    confidence: float  # 0.0-1.0; a heuristic-match strength, not a calibrated probability


def _classify_text(text: str) -> tuple[str | None, float]:
    """Keyword match over `text`; returns `(type, confidence)`. `confidence`
    reflects how many distinct keyword hits the winning type got, same
    self-report caveat as `layout_ocr`'s regex mapping and `llm_vision`'s
    model confidence (docs/architecture.md) -- not a calibrated probability."""
    lowered = text.lower()
    for doc_type, keywords in _TYPE_KEYWORDS:
        hits = sum(1 for kw in keywords if kw in lowered)
        if hits:
            return doc_type, min(1.0, 0.5 + 0.25 * hits)
    return None, 0.0


class DocumentClassifier:
    """Classifies scanned-vs-digital and coarse document type ahead of
    extraction.

    `ocr_backend_key` picks which `document_core.ocr` backend runs the
    lightweight OCR pass used for scanned/image input (default: Tesseract,
    the only CPU-only backend registered today) -- injectable so tests don't
    need the real `tesseract` binary (see `test_classify_review_classifier.py`).
    """

    def __init__(self, ocr_backend_key: str = "tesseract"):
        self.ocr_backend_key = ocr_backend_key

    def classify(self, document_path: Path) -> ClassificationResult:
        document_path = Path(document_path)
        if not document_path.exists():
            raise DocumentClassificationError(f"No such document: {document_path}")

        ext = document_path.suffix.lower()
        if ext == _PDF_EXTENSION:
            return self._classify_pdf(document_path)
        if ext in _IMAGE_EXTENSIONS:
            return self._classify_image(document_path)
        raise DocumentClassificationError(
            f"unsupported file type '{ext or '(none)'}' for classification"
        )

    def _classify_pdf(self, document_path: Path) -> ClassificationResult:
        text = self._read_pdf_text(document_path)
        if len(text.strip()) >= MIN_DIGITAL_TEXT_CHARS:
            doc_type, confidence = _classify_text(text)
            return ClassificationResult(
                is_scanned=False, document_type=doc_type, confidence=confidence
            )

        # Image-based ("scanned") PDF: no usable text layer at this stage. A
        # full OCR pass over rasterized pages belongs to the extraction
        # backends that run *after* classification (docs/architecture.md's
        # pipeline) -- doing that here would duplicate that work, so this
        # stage reports is_scanned=True with no type guess rather than
        # fabricating one.
        return ClassificationResult(is_scanned=True, document_type=None, confidence=0.0)

    def _classify_image(self, document_path: Path) -> ClassificationResult:
        # Any standalone image file is scanned input by definition -- there
        # is no such thing as a digital text layer on a raster image.
        try:
            ocr_result = extract_text(self.ocr_backend_key, document_path)
        except OCRBackendUnavailable:
            # document_core.ocr treats a missing engine as "fail loudly,
            # don't fabricate" for the extraction stage -- but classification
            # is a best-effort pre-step, not extraction itself, so a missing
            # OCR engine degrades to "scanned, unknown type" rather than
            # blocking the whole upload.
            return ClassificationResult(is_scanned=True, document_type=None, confidence=0.0)
        except OCRBackendError as exc:
            raise DocumentClassificationError(str(exc)) from exc

        doc_type, confidence = _classify_text(ocr_result.text)
        return ClassificationResult(is_scanned=True, document_type=doc_type, confidence=confidence)

    @staticmethod
    def _read_pdf_text(document_path: Path) -> str:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise DocumentClassificationError(
                "pypdf is not installed; add it to requirements.txt"
            ) from exc
        try:
            reader = PdfReader(str(document_path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:  # pypdf raises several exception types for malformed PDFs
            raise DocumentClassificationError(f"could not read PDF: {exc}") from exc
