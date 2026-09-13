from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import Http404
from django.shortcuts import redirect, render

from documents.services import get_document_store, owner_id

from .forms import FullPipelineRunForm
from .services import get_persisted_pipeline_result, run_pipeline_and_persist


@login_required
def document_list(request):
    store = get_document_store()
    records = store.list_for_owner(owner_id(request.user))
    return render(request, "full_pipeline/list.html", {"records": records})


@login_required
def run(request, doc_id: str):
    store = get_document_store()
    record = store.get(doc_id)
    # Wrong id AND someone else's document both 404 identically -- never
    # reveal that a given id belongs to another user (documents.views.detail
    # follows the same rule).
    if record is None or record.owner != owner_id(request.user):
        raise Http404("No such document.")

    if request.method == "POST":
        form = FullPipelineRunForm(
            request.POST, initial_backend_key=settings.DEFAULT_EXTRACTION_BACKEND
        )
        if form.is_valid():
            backend_key = form.cleaned_data["backend_key"]
            run_pipeline_and_persist(doc_id, backend_key)
            messages.success(request, f"Full pipeline complete ({backend_key}).")
            return redirect("full_pipeline:run", doc_id=doc_id)
    else:
        form = FullPipelineRunForm(initial_backend_key=settings.DEFAULT_EXTRACTION_BACKEND)

    result = get_persisted_pipeline_result(doc_id)
    return render(
        request,
        "full_pipeline/run.html",
        {"record": record, "form": form, "result": result},
    )
