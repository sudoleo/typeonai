"""Canonical public origin shared by HTTP routers and background services."""

from __future__ import annotations

import os
from urllib.parse import urlsplit


DEFAULT_PUBLIC_SITE_URL = "https://www.consens.io"


def normalize_public_site_url(value: str | None) -> str:
    candidate = str(value or DEFAULT_PUBLIC_SITE_URL).strip().rstrip("/")
    parsed = urlsplit(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError("PUBLIC_SITE_URL must be an absolute http(s) origin")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise RuntimeError("PUBLIC_SITE_URL must not contain credentials, query or fragment")
    if parsed.path not in {"", "/"}:
        raise RuntimeError("PUBLIC_SITE_URL must not contain a path")
    return f"{parsed.scheme}://{parsed.netloc}"


SITE_URL = normalize_public_site_url(os.environ.get("PUBLIC_SITE_URL"))

