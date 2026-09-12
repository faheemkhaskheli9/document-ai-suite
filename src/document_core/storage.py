"""
Document upload/storage shared by every feature app — README.md Section 2.

Framework-agnostic (no Django import) so it's reusable as-is once the Phase 1
Django skeleton (issue #3) lands and wraps it in a view.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

SUPPORTED_EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg", ".tiff")

DocumentStoreStatus = Literal["uploaded", "processing", "done", "failed"]


class DocumentValidationError(ValueError):
    """Raised when an uploaded document is rejected before being stored.

    Carries a short, user-facing `reason` separate from `str(exc)`.
    """

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def _extension(filename: str) -> str:
    idx = filename.rfind(".")
    return filename[idx:].lower() if idx != -1 else ""


@dataclass
class DocumentRecord:
    """The document_core-owned record every feature app reads (issue #1
    acceptance criteria: owner, filename, status)."""

    id: str
    owner: str
    filename: str
    status: DocumentStoreStatus
    content_hash: str
    stored_path: str
    uploaded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return asdict(self)


class DocumentStore:
    """Filesystem-backed store for uploaded documents plus their metadata.

    Each document gets its own directory named by a random id (never by
    filename or content-hash alone — two different owners uploading a file
    named `invoice.pdf`, or two uploads with identical content, must not
    collide and overwrite each other's record). The file write and the
    metadata write are each atomic (temp file in the same directory,
    `os.replace` onto the final path) so a crash mid-upload never leaves a
    record that looks stored but isn't.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _doc_dir(self, doc_id: str) -> Path:
        return self.root / doc_id

    def store(self, content: bytes, filename: str, owner: str) -> DocumentRecord:
        ext = _extension(filename)
        if ext not in SUPPORTED_EXTENSIONS:
            raise DocumentValidationError(
                f"Unsupported file type '{ext or '(none)'}'. "
                f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}."
            )
        if not content:
            raise DocumentValidationError("The file is empty.")

        doc_id = uuid.uuid4().hex
        content_hash = hashlib.sha256(content).hexdigest()
        doc_dir = self._doc_dir(doc_id)
        doc_dir.mkdir(parents=True, exist_ok=False)

        final_path = doc_dir / f"original{ext}"
        self._atomic_write(final_path, content)

        record = DocumentRecord(
            id=doc_id,
            owner=owner,
            filename=filename,
            status="uploaded",
            content_hash=content_hash,
            stored_path=str(final_path),
        )
        self._atomic_write(
            doc_dir / "record.json",
            json.dumps(record.to_dict(), indent=2).encode("utf-8"),
        )
        return record

    def get(self, doc_id: str) -> DocumentRecord | None:
        record_path = self._doc_dir(doc_id) / "record.json"
        if not record_path.exists():
            return None
        data = json.loads(record_path.read_text(encoding="utf-8"))
        return DocumentRecord(**data)

    def list_for_owner(self, owner: str) -> list[DocumentRecord]:
        records = []
        for record_path in sorted(self.root.glob("*/record.json")):
            data = json.loads(record_path.read_text(encoding="utf-8"))
            if data.get("owner") == owner:
                records.append(DocumentRecord(**data))
        return records

    def set_status(self, doc_id: str, status: DocumentStoreStatus) -> DocumentRecord:
        record = self.get(doc_id)
        if record is None:
            raise KeyError(f"No such document: {doc_id}")
        record.status = status
        self._atomic_write(
            self._doc_dir(doc_id) / "record.json",
            json.dumps(record.to_dict(), indent=2).encode("utf-8"),
        )
        return record

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        """Write to a temp file in the same directory, then rename onto
        `path` only once the write is complete — never leaves a truncated
        file at `path` if the process is interrupted mid-write."""
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(content)
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
