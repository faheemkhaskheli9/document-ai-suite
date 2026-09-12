"""Tests for the pluggable document-extraction backend abstraction, the
multimodal-LLM sibling of `document_core.ocr` (README.md Section 2: extract
via a custom-trained model, generic OCR, or a multimodal LLM)."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from document_core import extraction
from document_core.extraction import (
    DEFAULT_LLM_EXTRACTION_MODEL,
    ExtractionBackend,
    ExtractionBackendError,
    ExtractionBackendUnavailable,
    LLMVisionExtractionBackend,
    extract_fields,
    get_extraction_backend,
    is_registered,
    register_extraction_backend,
)
from document_core.schema import ExtractionResult


class _FakeExtractionBackend(ExtractionBackend):
    """A second backend, registered only for this test module, used to prove
    the interface is swappable without touching any caller."""

    key = "fake-test-backend"
    label = "Fake (tests only)"

    def extract(self, document_path: Path, document_id: str) -> ExtractionResult:
        return ExtractionResult(document_id=document_id, fields=[])


@pytest.fixture(scope="module", autouse=True)
def _register_fake_backend():
    register_extraction_backend(_FakeExtractionBackend)
    yield
    extraction._REGISTRY.pop(_FakeExtractionBackend.key, None)


def test_llm_vision_backend_is_registered_by_default():
    assert is_registered("llm_vision")
    assert isinstance(get_extraction_backend("llm_vision"), LLMVisionExtractionBackend)


def test_registering_a_duplicate_key_raises():
    with pytest.raises(ValueError, match="duplicate extraction backend key"):
        register_extraction_backend(_FakeExtractionBackend)


def test_unknown_key_raises_keyerror():
    with pytest.raises(KeyError, match="no extraction backend registered"):
        get_extraction_backend("does-not-exist")


def test_extract_fields_is_the_one_interface_feature_apps_use(tmp_path):
    doc_path = tmp_path / "doc.png"
    doc_path.write_bytes(b"not-a-real-image")

    result = extract_fields("fake-test-backend", doc_path, "doc-1")

    assert result == ExtractionResult(document_id="doc-1", fields=[])


def test_llm_backend_raises_on_missing_file(tmp_path):
    missing = tmp_path / "nope.png"
    with pytest.raises(ExtractionBackendError, match="No such document"):
        get_extraction_backend("llm_vision").extract(missing, "doc-1")


def test_llm_backend_rejects_unsupported_extension(tmp_path):
    doc_path = tmp_path / "doc.tiff"
    doc_path.write_bytes(b"fake-tiff-bytes")
    with pytest.raises(ExtractionBackendError, match="unsupported file type"):
        get_extraction_backend("llm_vision").extract(doc_path, "doc-1")


def test_llm_backend_raises_unavailable_without_the_sdk(tmp_path, monkeypatch):
    """CPU/no-dependency boundary: if `anthropic` isn't installed, extract()
    must fail loudly instead of silently returning an empty/fabricated
    result."""
    doc_path = tmp_path / "doc.png"
    doc_path.write_bytes(b"fake-png-bytes")
    monkeypatch.setitem(sys.modules, "anthropic", None)  # forces ImportError

    with pytest.raises(ExtractionBackendUnavailable, match="anthropic"):
        get_extraction_backend("llm_vision").extract(doc_path, "doc-1")


class _FakeAnthropicModule:
    """Minimal stand-in for the `anthropic` package so tests never make a
    real network call. `make_response` controls what `messages.create`
    returns; exceptions mirror the real SDK's exception hierarchy closely
    enough for the backend's except clauses to match them by type.
    """

    class AuthenticationError(Exception):
        pass

    class PermissionDeniedError(Exception):
        pass

    class APIConnectionError(Exception):
        pass

    class APIStatusError(Exception):
        def __init__(self, message, status_code):
            super().__init__(message)
            self.message = message
            self.status_code = status_code

    def __init__(self, response=None, raise_exc=None):
        self._response = response
        self._raise_exc = raise_exc
        self.last_request = None

    def Anthropic(self):  # noqa: N802 -- matches anthropic.Anthropic's name
        return self

    @property
    def messages(self):
        return self

    def create(self, **kwargs):
        self.last_request = kwargs
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._response


def _install_fake_anthropic(monkeypatch, response=None, raise_exc=None):
    fake = _FakeAnthropicModule(response=response, raise_exc=raise_exc)
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    return fake


def _fake_response(payload: dict, stop_reason: str = "end_turn", stop_details=None):
    text_block = types.SimpleNamespace(type="text", text=json.dumps(payload))
    return types.SimpleNamespace(
        content=[text_block], stop_reason=stop_reason, stop_details=stop_details
    )


def test_llm_backend_parses_a_valid_structured_response(tmp_path, monkeypatch):
    doc_path = tmp_path / "invoice.png"
    doc_path.write_bytes(b"fake-png-bytes")
    payload = {
        "document_type": "invoice",
        "fields": [
            {"name": "total", "value": "$42.00", "confidence": 0.9},
            {"name": "invoice_number", "value": "INV-1", "confidence": 0.8},
        ],
    }
    fake = _install_fake_anthropic(monkeypatch, response=_fake_response(payload))

    result = get_extraction_backend("llm_vision").extract(doc_path, "doc-1")

    assert result.document_id == "doc-1"
    assert result.document_type == "invoice"
    assert [f.name for f in result.fields] == ["total", "invoice_number"]
    assert result.overall_confidence == pytest.approx(0.85)
    assert fake.last_request["model"] == DEFAULT_LLM_EXTRACTION_MODEL


def test_llm_backend_maps_empty_document_type_to_none(tmp_path, monkeypatch):
    doc_path = tmp_path / "doc.png"
    doc_path.write_bytes(b"fake-png-bytes")
    payload = {"document_type": "", "fields": []}
    _install_fake_anthropic(monkeypatch, response=_fake_response(payload))

    result = get_extraction_backend("llm_vision").extract(doc_path, "doc-1")

    assert result.document_type is None
    assert result.overall_confidence is None


def test_llm_backend_clamps_out_of_range_confidence(tmp_path, monkeypatch):
    doc_path = tmp_path / "doc.png"
    doc_path.write_bytes(b"fake-png-bytes")
    payload = {
        "document_type": "form",
        "fields": [{"name": "x", "value": "y", "confidence": 1.5}],
    }
    _install_fake_anthropic(monkeypatch, response=_fake_response(payload))

    result = get_extraction_backend("llm_vision").extract(doc_path, "doc-1")

    assert result.fields[0].confidence == 1.0


def test_llm_backend_raises_on_malformed_json(tmp_path, monkeypatch):
    doc_path = tmp_path / "doc.png"
    doc_path.write_bytes(b"fake-png-bytes")
    text_block = types.SimpleNamespace(type="text", text="not json at all")
    response = types.SimpleNamespace(content=[text_block], stop_reason="end_turn", stop_details=None)
    _install_fake_anthropic(monkeypatch, response=response)

    with pytest.raises(ExtractionBackendError, match="malformed JSON"):
        get_extraction_backend("llm_vision").extract(doc_path, "doc-1")


def test_llm_backend_raises_on_refusal(tmp_path, monkeypatch):
    doc_path = tmp_path / "doc.png"
    doc_path.write_bytes(b"fake-png-bytes")
    stop_details = types.SimpleNamespace(category="cyber")
    response = _fake_response({"document_type": "", "fields": []}, stop_reason="refusal", stop_details=stop_details)
    _install_fake_anthropic(monkeypatch, response=response)

    with pytest.raises(ExtractionBackendError, match="declined"):
        get_extraction_backend("llm_vision").extract(doc_path, "doc-1")


def test_llm_backend_wraps_authentication_error_as_unavailable(tmp_path, monkeypatch):
    doc_path = tmp_path / "doc.png"
    doc_path.write_bytes(b"fake-png-bytes")
    fake = _FakeAnthropicModule()
    fake._raise_exc = fake.AuthenticationError("bad key")
    monkeypatch.setitem(sys.modules, "anthropic", fake)

    with pytest.raises(ExtractionBackendUnavailable, match="credentials"):
        get_extraction_backend("llm_vision").extract(doc_path, "doc-1")


def test_llm_backend_wraps_api_status_error(tmp_path, monkeypatch):
    doc_path = tmp_path / "doc.png"
    doc_path.write_bytes(b"fake-png-bytes")
    fake = _FakeAnthropicModule()
    fake._raise_exc = fake.APIStatusError("server exploded", status_code=500)
    monkeypatch.setitem(sys.modules, "anthropic", fake)

    with pytest.raises(ExtractionBackendError, match="500"):
        get_extraction_backend("llm_vision").extract(doc_path, "doc-1")


def test_llm_backend_accepts_pdf_input(tmp_path, monkeypatch):
    doc_path = tmp_path / "doc.pdf"
    doc_path.write_bytes(b"%PDF-1.4 fake")
    fake = _install_fake_anthropic(
        monkeypatch, response=_fake_response({"document_type": "contract", "fields": []})
    )

    get_extraction_backend("llm_vision").extract(doc_path, "doc-1")

    content = fake.last_request["messages"][0]["content"]
    assert content[0]["type"] == "document"
    assert content[0]["source"]["media_type"] == "application/pdf"


def test_llm_backend_respects_model_override_env_var(tmp_path, monkeypatch):
    doc_path = tmp_path / "doc.png"
    doc_path.write_bytes(b"fake-png-bytes")
    fake = _install_fake_anthropic(
        monkeypatch, response=_fake_response({"document_type": "", "fields": []})
    )
    monkeypatch.setenv("LLM_EXTRACTION_MODEL", "claude-sonnet-5")

    get_extraction_backend("llm_vision").extract(doc_path, "doc-1")

    assert fake.last_request["model"] == "claude-sonnet-5"
