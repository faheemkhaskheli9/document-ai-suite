"""Local development settings -- `manage.py` defaults here (see repo-root
`manage.py`). Never used in production: DEBUG on, permissive ALLOWED_HOSTS."""
from .base import *  # noqa: F401,F403

DEBUG = True
ALLOWED_HOSTS = ["*"]
