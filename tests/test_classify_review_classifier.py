"""Tests for `classify_review.classifier` -- issue #7."""
from __future__ import annotations

import pytest

from classify_review.classifier import (
    ClassificationResult,
    DocumentClassificationError,
    DocumentClassifier,
)
from document_core.ocr import OCRBackendError, OCRBackendUnavailable, OCRTextResult


def _blank_pdf(path):
    """A real, minimal PDF with no text layer at all -- the "scanned PDF"
    case, without pulling in a PDF-authoring dependency just for tests."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with open(path, "wb") as f:
        writer.write(f)


def test_missing_document_raises(tmp_path):
    classifier = DocumentClassifier()
    with pytest.raises(DocumentClassificationError, match="No such document"):
        classifier.classify(tmp_path / "nope.pdf")


def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "doc.txt"
    path.write_text("hello")
    classifier = DocumentClassifier()
    with pytest.raises(DocumentClassificationError, match="unsupported file type"):
        classifier.classify(path)


def test_digital_pdf_with_text_layer_is_not_scanned(tmp_path, monkeypatch):
    path = tmp_path / "invoice.pdf"
    path.write_bytes(b"%PDF-fake")  # content irrelevant, _read_pdf_text is mocked below
    monkeypatch.setattr(
        DocumentClassifier,
        "_read_pdf_text",
        staticmethod(lambda p: "INVOICE\nBill To: Acme Corp\nAmount Due: $500"),
    )

    result = DocumentClassifier().classify(path)

    assert isinstance(result, ClassificationResult)
    assert result.is_scanned is False
    assert result.document_type == "invoice"
    assert result.confidence > 0.5


def test_digital_pdf_type_keywords_pick_contract(tmp_path, monkeypatch):
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-fake")
    monkeypatch.setattr(
        DocumentClassifier,
        "_read_pdf_text",
        staticmethod(
            lambda p: "This Agreement sets out the Terms and Conditions between the parties."
        ),
    )

    result = DocumentClassifier().classify(path)

    assert result.is_scanned is False
    assert result.document_type == "contract"


def test_digital_pdf_with_no_keyword_match_has_no_type(tmp_path, monkeypatch):
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-fake")
    monkeypatch.setattr(
        DocumentClassifier,
        "_read_pdf_text",
        staticmethod(lambda p: "A" * 50),  # long enough to be "digital", no keywords
    )

    result = DocumentClassifier().classify(path)

    assert result.is_scanned is False
    assert result.document_type is None
    assert result.confidence == 0.0


def test_scanned_pdf_with_no_text_layer_is_scanned(tmp_path):
    path = tmp_path / "scanned.pdf"
    _blank_pdf(path)

    result = DocumentClassifier().classify(path)

    assert result.is_scanned is True
    assert result.document_type is None
    assert result.confidence == 0.0


def test_image_input_runs_ocr_and_classifies_type(tmp_path, monkeypatch):
    path = tmp_path / "scan.png"
    path.write_bytes(b"not-a-real-image")
    monkeypatch.setattr(
        "classify_review.classifier.extract_text",
        lambda backend_key, image_path: OCRTextResult(
            text="RECEIPT\nSubtotal: $12.00\nThank you for your purchase", confidence=0.9
        ),
    )

    result = DocumentClassifier().classify(path)

    assert result.is_scanned is True
    assert result.document_type == "receipt"
    assert result.confidence > 0.5


def test_image_input_with_ocr_engine_unavailable_degrades_gracefully(tmp_path, monkeypatch):
    path = tmp_path / "scan.png"
    path.write_bytes(b"not-a-real-image")

    def _raise(*args, **kwargs):
        raise OCRBackendUnavailable("tesseract binary not on PATH")

    monkeypatch.setattr("classify_review.classifier.extract_text", _raise)

    result = DocumentClassifier().classify(path)

    assert result.is_scanned is True
    assert result.document_type is None
    assert result.confidence == 0.0


def test_image_input_with_ocr_error_raises_classification_error(tmp_path, monkeypatch):
    path = tmp_path / "scan.png"
    path.write_bytes(b"not-a-real-image")

    def _raise(*args, **kwargs):
        raise OCRBackendError("engine crashed")

    monkeypatch.setattr("classify_review.classifier.extract_text", _raise)

    with pytest.raises(DocumentClassificationError, match="engine crashed"):
        DocumentClassifier().classify(path)


def test_custom_ocr_backend_key_is_passed_through(tmp_path, monkeypatch):
    path = tmp_path / "scan.jpg"
    path.write_bytes(b"not-a-real-image")
    seen = {}

    def _fake_extract_text(backend_key, image_path):
        seen["backend_key"] = backend_key
        return OCRTextResult(text="", confidence=0.0)

    monkeypatch.setattr("classify_review.classifier.extract_text", _fake_extract_text)

    DocumentClassifier(ocr_backend_key="fake-backend").classify(path)

    assert seen["backend_key"] == "fake-backend"
