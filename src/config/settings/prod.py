"""Production settings. Fails loudly at import time (before the process ever
binds a port) if a production-required value was left on its insecure dev
default -- robustness rule 7: defaults are fine for dev, but a value that
*must* be explicit in prod must hard-error, never silently fall back."""
from django.core.exceptions import ImproperlyConfigured

from .base import *  # noqa: F401,F403
from .base import ALLOWED_HOSTS, DEV_INSECURE_SECRET_KEY, SECRET_KEY

DEBUG = False

if SECRET_KEY == DEV_INSECURE_SECRET_KEY:
    raise ImproperlyConfigured(
        "DJANGO_SECRET_KEY must be set to a real secret in production "
        "(refusing to boot with the dev default)."
    )
if not ALLOWED_HOSTS:
    raise ImproperlyConfigured(
        "DJANGO_ALLOWED_HOSTS must be set in production (comma-separated)."
    )

SECURE_SSL_REDIRECT = env.bool("DJANGO_SECURE_SSL_REDIRECT", default=True)
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_HSTS_SECONDS = env.int("DJANGO_HSTS_SECONDS", default=60 * 60 * 24 * 7)
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
