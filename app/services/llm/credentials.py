"""Single OpenRouter credential source for every LLM operation."""

from __future__ import annotations

import os

OPENROUTER_KEY_LABEL = "OpenRouter"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"


def resolve_developer_api_keys(_providers: list[str] | None = None) -> dict[str, str | None]:
    value = str(os.environ.get(OPENROUTER_API_KEY_ENV) or "").strip()
    return {OPENROUTER_KEY_LABEL: value or None}


def openrouter_api_key(api_keys: dict | None) -> str | None:
    return str((api_keys or {}).get(OPENROUTER_KEY_LABEL) or "").strip() or None


def missing_credentials(
    api_keys: dict[str, str | None],
    _required: list[str] | None = None,
) -> list[str]:
    return [] if openrouter_api_key(api_keys) else [OPENROUTER_KEY_LABEL]
