"""Root URLConf. Each feature area is namespaced and included wholesale so a
later phase (layout_ocr, classify_review, full_pipeline) only has to add one
`path("...", include(("app.urls", "app"), namespace="app"))` line here."""
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("accounts/", include("accounts.urls", namespace="accounts")),
    path("documents/", include("documents.urls", namespace="documents")),
    path("", include("dashboard.urls", namespace="dashboard")),
]
