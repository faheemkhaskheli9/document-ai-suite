#!/usr/bin/env python
"""Django's command-line utility -- run from the repo root (README.md
Section 9: `python manage.py migrate` / `runserver`).

The project lives under `src/` (same src-layout `tests/conftest.py` already
uses), so `src/` is put on `sys.path` here before anything under it --
`config`, `dashboard`, `documents`, `accounts`, `document_core` -- is
imported.
"""
import os
import sys
from pathlib import Path


def main():
    src_dir = Path(__file__).resolve().parent / "src"
    sys.path.insert(0, str(src_dir))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.dev")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Is it installed and available on your "
            "PYTHONPATH environment variable? Did you forget to activate a "
            "virtual environment? Run `pip install -r requirements.txt`."
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
