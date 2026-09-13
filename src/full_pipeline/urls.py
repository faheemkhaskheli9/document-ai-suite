from django.urls import path

from . import views

app_name = "full_pipeline"

urlpatterns = [
    path("", views.document_list, name="list"),
    path("<str:doc_id>/", views.run, name="run"),
]
