"""
Base Django settings shared by every environment (dev/test/prod) --
README.md Section 3 ("Django 5.x") / Section 9. Environment-specific files
(`dev.py`, `test.py`, `prod.py`) import `*` from here and override only what
differs, so a setting added once here (e.g. a new INSTALLED_APPS entry)
never has to be duplicated across environments.

All secrets/environment-dependent values are read via `django-environ` with
explicit defaults for local dev only -- `prod.py` hard-fails if a
production-required value (SECRET_KEY, ALLOWED_HOSTS) is still on its dev
default, rather than silently booting an insecure server (robustness rule 7:
be lenient with defaults, strict about explicit/production input).
"""
from __future__ import annotations

from pathlib import Path

import environ

# src/config/settings/base.py -> src/config/settings -> src/config -> src -> repo root
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = BASE_DIR / "src"

env = environ.Env()
_env_file = BASE_DIR / ".env"
if _env_file.exists():
    environ.Env.read_env(str(_env_file))

DEV_INSECURE_SECRET_KEY = "dev-insecure-secret-key-change-me"
SECRET_KEY = env("DJANGO_SECRET_KEY", default=DEV_INSECURE_SECRET_KEY)
DEBUG = env.bool("DJANGO_DEBUG", default=False)
ALLOWED_HOSTS = env.list("DJANGO_ALLOWED_HOSTS", default=[])

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # First-party apps
    "accounts",
    "dashboard",
    "documents",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

# Database: SQLite for local/dev, override with DATABASE_URL in prod
# (README.md Section 3: "PostgreSQL in production, SQLite for local/dev").
DATABASES = {
    "default": env.db(
        "DATABASE_URL", default=f"sqlite:///{BASE_DIR / 'db.sqlite3'}"
    )
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LOGIN_URL = "accounts:login"
LOGIN_REDIRECT_URL = "dashboard:home"
LOGOUT_REDIRECT_URL = "dashboard:home"

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_DIRS = [BASE_DIR / "static"]
STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {"BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage"},
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Root directory `document_core.storage.DocumentStore` writes uploaded
# documents + their metadata under (see `documents/services.py`). Kept
# separate from STATIC_ROOT/MEDIA_ROOT naming since it's not Django-managed
# media, just the framework-agnostic document_core store's data directory.
DOCUMENT_STORE_ROOT = env(
    "DOCUMENT_STORE_ROOT", default=str(BASE_DIR / "data" / "documents")
)

# Which OCR backend `document_core.ocr.extract_text` is called with by
# default (feature apps landing in later phases will read this too).
DEFAULT_OCR_BACKEND = env("DEFAULT_OCR_BACKEND", default="tesseract")

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {"class": "logging.StreamHandler"},
    },
    "root": {
        "handlers": ["console"],
        "level": env("DJANGO_LOG_LEVEL", default="INFO"),
    },
}
