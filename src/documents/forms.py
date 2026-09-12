from django import forms

from document_core.extraction import all_extraction_backends
from document_core.storage import SUPPORTED_EXTENSIONS


class DocumentUploadForm(forms.Form):
    file = forms.FileField(
        label="Document",
        help_text=f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}",
    )


class ExtractionRunForm(forms.Form):
    """Lets the user pick which `document_core.extraction` backend to run --
    issue #14. Choices are read from the live registry (`all_extraction_backends`)
    rather than hardcoded, so a third engine registering itself later needs
    no change here."""

    backend_key = forms.ChoiceField(label="Extraction engine")

    def __init__(self, *args, initial_backend_key: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["backend_key"].choices = [
            (backend.key, backend.label) for backend in all_extraction_backends()
        ]
        if initial_backend_key:
            self.fields["backend_key"].initial = initial_backend_key
