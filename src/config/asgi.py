"""ASGI entrypoint (kept alongside wsgi.py for parity; not required until an
async-only feature, e.g. streaming OCR progress, lands in a later phase)."""
import os

from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.prod")

application = get_asgi_application()
