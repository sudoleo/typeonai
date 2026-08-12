import logging
from types import SimpleNamespace
from unittest.mock import patch

import firebase_admin
import pytest
import requests

from app.services import registration


def test_new_user_gets_an_unguessable_server_side_password():
    created = SimpleNamespace(uid="new-owner")
    with (
        patch.object(
            registration.auth,
            "get_user_by_email",
            side_effect=firebase_admin.auth.UserNotFoundError("missing"),
        ),
        patch.object(registration.auth, "create_user", return_value=created) as create,
    ):
        user, was_created = registration.find_or_provision_user("new@example.test")

    assert user is created
    assert was_created is True
    password = create.call_args.kwargs["password"]
    assert len(password) >= 48
    assert password != "attacker-known"


def test_existing_user_uses_the_same_mailbox_setup_path():
    existing = SimpleNamespace(uid="existing-owner")
    with (
        patch.object(registration.auth, "get_user_by_email", return_value=existing),
        patch.object(registration.auth, "create_user") as create,
    ):
        user, was_created = registration.find_or_provision_user("existing@example.test")

    assert user is existing
    assert was_created is False
    create.assert_not_called()


def test_password_setup_request_is_bounded_and_does_not_log_email(
    monkeypatch, caplog
):
    monkeypatch.setenv("FIREBASE_API_KEY", "test-key")
    secret_email = "private@example.test"
    with patch.object(
        registration.requests,
        "post",
        side_effect=requests.Timeout(f'upstream included "{secret_email}"'),
    ) as post:
        with caplog.at_level(logging.ERROR):
            assert registration.deliver_password_setup_email(secret_email) is False

    assert post.call_args.kwargs["timeout"] == (5.0, 15.0)
    assert secret_email not in caplog.text
    assert "upstream included" not in caplog.text


def test_password_setup_requires_operator_configuration(monkeypatch):
    monkeypatch.delenv("FIREBASE_API_KEY", raising=False)
    with pytest.raises(registration.RegistrationUnavailable):
        registration.send_password_setup_email("private@example.test")
