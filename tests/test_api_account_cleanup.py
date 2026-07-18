from types import SimpleNamespace

import pytest

from app.services import api_account_cleanup
from app.services.api_account_cleanup import (
    ApiAccountInactive,
    ApiAccountStatusUnavailable,
    FirestoreApiAccountCleanup,
)


def test_blocked_account_fails_before_firebase(monkeypatch):
    repository = FirestoreApiAccountCleanup(None)
    monkeypatch.setattr(repository, "is_blocked", lambda uid: True)
    monkeypatch.setattr(
        api_account_cleanup.auth,
        "get_user",
        lambda uid: (_ for _ in ()).throw(AssertionError("must not be called")),
    )

    with pytest.raises(ApiAccountInactive):
        repository.ensure_active("user-1")


@pytest.mark.parametrize(
    "user",
    [
        SimpleNamespace(disabled=True, email_verified=True),
        SimpleNamespace(disabled=False, email_verified=False),
    ],
)
def test_disabled_or_unverified_firebase_account_is_inactive(monkeypatch, user):
    repository = FirestoreApiAccountCleanup(None)
    monkeypatch.setattr(repository, "is_blocked", lambda uid: False)
    monkeypatch.setattr(api_account_cleanup.auth, "get_user", lambda uid: user)

    with pytest.raises(ApiAccountInactive):
        repository.ensure_active("user-1")


def test_firebase_outage_fails_closed_as_unavailable(monkeypatch):
    repository = FirestoreApiAccountCleanup(None)
    monkeypatch.setattr(repository, "is_blocked", lambda uid: False)
    monkeypatch.setattr(
        api_account_cleanup.auth,
        "get_user",
        lambda uid: (_ for _ in ()).throw(RuntimeError("outage")),
    )

    with pytest.raises(ApiAccountStatusUnavailable):
        repository.ensure_active("user-1")
