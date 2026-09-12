"""Settings for `pytest` / `pytest-django` (`DJANGO_SETTINGS_MODULE` in
pyproject.toml's `[tool.pytest.ini_options]`). Fast password hasher and an
in-memory SQLite DB so the test suite never touches dev.py's `db.sqlite3` or
prod's Postgres; `DOCUMENT_STORE_ROOT` is overridden per-test via
`override_settings` + `tmp_path` rather than here, so tests never share a
document store directory."""
from .base import *  # noqa: F401,F403

DEBUG = False
ALLOWED_HOSTS = ["testserver"]

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }
}

PASSWORD_HASHERS = ["django.contrib.auth.hashers.MD5PasswordHasher"]

# base.py's whitenoise manifest storage requires `collectstatic` to have run
# first (it 404s any static file not in its hashed manifest) -- irrelevant
# for the test suite, which never serves static files, so fall back to the
# plain filesystem storage that needs no manifest.
STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
}
