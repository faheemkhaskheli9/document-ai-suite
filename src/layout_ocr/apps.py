from django.apps import AppConfig


class LayoutOcrConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "layout_ocr"

    def ready(self):
        from layout_ocr import backend  # noqa: F401 -- registers the "layout_ocr" extraction backend
