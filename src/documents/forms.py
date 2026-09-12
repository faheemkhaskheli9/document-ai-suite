from django import forms

from document_core.storage import SUPPORTED_EXTENSIONS


class DocumentUploadForm(forms.Form):
    file = forms.FileField(
        label="Document",
        help_text=f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}",
    )
