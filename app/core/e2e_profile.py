"""Fail-closed runtime contract for the browser E2E profile.

The E2E server is allowed to use exactly one local Firestore emulator project.
Keeping this guard independent from ``firebase_admin`` lets unit tests prove the
startup decision without opening a Firebase connection.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from urllib.parse import urlsplit


E2E_PROJECT_ID = "demo-consensio-e2e"
_PROJECT_ENV_KEYS = (
    "GOOGLE_CLOUD_PROJECT",
    "GCLOUD_PROJECT",
    "FIREBASE_PROJECT_ID",
)
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def e2e_test_mode_enabled(env: Mapping[str, str] | None = None) -> bool:
    values = os.environ if env is None else env
    return values.get("E2E_TEST_MODE") == "1"


def assert_safe_e2e_environment(env: Mapping[str, str] | None = None) -> None:
    """Reject every E2E target except the dedicated local demo project.

    A ``demo-`` project ID has no corresponding live Firebase project. Together
    with the mandatory loopback emulator endpoint this prevents a missing or
    misspelled emulator setting from falling through to production services.
    """

    values = os.environ if env is None else env
    if not e2e_test_mode_enabled(values):
        return

    emulator_host = str(values.get("FIRESTORE_EMULATOR_HOST") or "").strip()
    try:
        parsed = urlsplit(f"//{emulator_host}")
        host = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise RuntimeError(
            "Refusing to start E2E: FIRESTORE_EMULATOR_HOST is invalid."
        ) from exc
    if host not in _LOOPBACK_HOSTS or port is None:
        raise RuntimeError(
            "Refusing to start E2E: FIRESTORE_EMULATOR_HOST must point to a "
            "loopback host with an explicit port."
        )

    configured_projects = {
        key: str(values.get(key) or "").strip()
        for key in _PROJECT_ENV_KEYS
        if str(values.get(key) or "").strip()
    }
    if not configured_projects:
        raise RuntimeError(
            "Refusing to start E2E: no Firebase/Google project ID is configured."
        )
    unexpected = {
        key: project_id
        for key, project_id in configured_projects.items()
        if project_id != E2E_PROJECT_ID
    }
    if unexpected:
        details = ", ".join(f"{key}={value}" for key, value in unexpected.items())
        raise RuntimeError(
            "Refusing to start E2E: project ID is not allowlisted "
            f"({details}); expected {E2E_PROJECT_ID}."
        )


def firebase_project_id(env: Mapping[str, str] | None = None) -> str:
    values = os.environ if env is None else env
    for key in _PROJECT_ENV_KEYS:
        project_id = str(values.get(key) or "").strip()
        if project_id:
            return project_id
    return ""
