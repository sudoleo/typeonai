"""Non-enumerating e-mail registration provisioning.

The anonymous caller never chooses a credential that can immediately be used
to distinguish a newly-created Firebase user from an existing one. New users
receive an unguessable temporary password and, just like existing users, a
Firebase-hosted password-setup/reset e-mail. Mailbox access is therefore the
first observable proof of ownership.
"""

from __future__ import annotations

import logging
import os
import secrets

import firebase_admin
from firebase_admin import auth
import requests

from app.core.observability import safe_exception


PASSWORD_SETUP_ENDPOINT = (
    "https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode"
)
PASSWORD_SETUP_CONNECT_TIMEOUT_SECONDS = 5.0
PASSWORD_SETUP_READ_TIMEOUT_SECONDS = 15.0


class RegistrationUnavailable(RuntimeError):
    """Raised when the neutral registration flow cannot be started safely."""


def is_password_setup_configured() -> bool:
    return bool(os.environ.get("FIREBASE_API_KEY", "").strip())


def find_or_provision_user(email: str):
    """Return ``(user, created)`` without accepting an attacker-known secret."""
    try:
        return auth.get_user_by_email(email), False
    except firebase_admin.auth.UserNotFoundError:
        pass

    temporary_password = secrets.token_urlsafe(48)
    try:
        return auth.create_user(email=email, password=temporary_password), True
    except firebase_admin.auth.EmailAlreadyExistsError:
        # A concurrent request won the create race. Treat it exactly like every
        # other existing address and continue with the same mailbox-only flow.
        return auth.get_user_by_email(email), False


def send_password_setup_email(email: str) -> None:
    """Ask Firebase to deliver its hosted password setup/reset e-mail."""
    api_key = os.environ.get("FIREBASE_API_KEY", "").strip()
    if not api_key:
        raise RegistrationUnavailable("Firebase password setup is not configured")

    try:
        response = requests.post(
            PASSWORD_SETUP_ENDPOINT,
            params={"key": api_key},
            json={"requestType": "PASSWORD_RESET", "email": email},
            timeout=(
                PASSWORD_SETUP_CONNECT_TIMEOUT_SECONDS,
                PASSWORD_SETUP_READ_TIMEOUT_SECONDS,
            ),
        )
        response.raise_for_status()
    except Exception as exc:
        # Never include the address or the upstream response body in logs.
        raise RegistrationUnavailable(
            f"Firebase password setup failed ({safe_exception(exc)})"
        ) from exc


def deliver_password_setup_email(email: str) -> bool:
    """Background-task wrapper that keeps logs content-free."""
    try:
        send_password_setup_email(email)
        return True
    except Exception as exc:
        logging.error("Registration setup delivery failed: %s", safe_exception(exc))
        return False
