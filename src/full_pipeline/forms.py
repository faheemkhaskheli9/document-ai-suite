from django import forms

from document_core.extraction import all_extraction_backends


class FullPipelineRunForm(forms.Form):
    """Lets the user pick which `document_core.extraction` backend the
    pipeline's extraction stage uses -- same choices/registry-driven pattern
    as `documents.forms.ExtractionRunForm`."""

    backend_key = forms.ChoiceField(label="Extraction engine")

    def __init__(self, *args, initial_backend_key: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["backend_key"].choices = [
            (backend.key, backend.label) for backend in all_extraction_backends()
        ]
        if initial_backend_key:
            self.fields["backend_key"].initial = initial_backend_key
