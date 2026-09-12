from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import Http404
from django.shortcuts import redirect, render

from document_core.storage import DocumentValidationError

from .forms import DocumentUploadForm
from .services import get_document_store, owner_id


@login_required
def upload(request):
    if request.method == "POST":
        form = DocumentUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded = form.cleaned_data["file"]
            store = get_document_store()
            try:
                record = store.store(
                    content=uploaded.read(),
                    filename=uploaded.name,
                    owner=owner_id(request.user),
                )
            except DocumentValidationError as exc:
                # Loud, specific failure surfaced back to the form -- never
                # a silent drop of a rejected upload (robustness rules).
                form.add_error("file", exc.reason)
            else:
                messages.success(request, f'"{record.filename}" uploaded.')
                return redirect("documents:detail", doc_id=record.id)
    else:
        form = DocumentUploadForm()

    return render(request, "documents/upload.html", {"form": form})


@login_required
def document_list(request):
    store = get_document_store()
    records = store.list_for_owner(owner_id(request.user))
    return render(request, "documents/list.html", {"records": records})


@login_required
def detail(request, doc_id: str):
    store = get_document_store()
    record = store.get(doc_id)
    # Wrong id AND someone else's document both 404 identically -- never
    # reveal that a given id belongs to another user.
    if record is None or record.owner != owner_id(request.user):
        raise Http404("No such document.")
    return render(request, "documents/detail.html", {"record": record})
