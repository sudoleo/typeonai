import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routers import auth as auth_router
from app.core.rate_limit import limiter


class AuthSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = FastAPI()
        app.include_router(auth_router.router)
        cls.client = TestClient(app)

    def setUp(self):
        limiter.reset()

    def test_confirm_registration_sets_httponly_session_cookie(self):
        with (
            patch.object(auth_router, "verify_user_token", return_value="owner-1"),
            patch.object(
                auth_router.auth, "get_user",
                return_value=SimpleNamespace(email_verified=True),
            ),
        ):
            response = self.client.post(
                "/confirm-registration", json={"id_token": "firebase-id-token"}
            )
        self.assertEqual(response.status_code, 200)
        cookie = response.headers.get("set-cookie", "").lower()
        self.assertIn("session=firebase-id-token", cookie)
        self.assertIn("httponly", cookie)
        self.assertIn("samesite=lax", cookie)
        self.assertEqual(response.headers.get("cache-control"), "no-store")

    def test_confirm_registration_notifies_for_recent_google_account(self):
        user = SimpleNamespace(
            email_verified=True,
            user_metadata=SimpleNamespace(creation_timestamp=time.time() * 1000),
            provider_data=[SimpleNamespace(provider_id="google.com")],
        )
        with (
            patch.object(auth_router, "verify_user_token", return_value="google-owner"),
            patch.object(auth_router.auth, "get_user", return_value=user),
            patch.object(
                auth_router,
                "send_new_user_registration_notification",
            ) as notify,
        ):
            response = self.client.post(
                "/confirm-registration", json={"id_token": "firebase-id-token"}
            )

        self.assertEqual(response.status_code, 200)
        notify.assert_called_once_with("google", "google-owner")

    def test_new_email_registration_schedules_telegram_notification(self):
        user = SimpleNamespace(uid="new-owner", email="new@example.test")
        with (
            patch.object(auth_router, "is_password_setup_configured", return_value=True),
            patch.object(
                auth_router, "find_or_provision_user", return_value=(user, True)
            ) as provision,
            patch.object(auth_router, "deliver_password_setup_email") as deliver,
            patch.object(
                auth_router,
                "send_new_user_registration_notification",
            ) as notify,
        ):
            response = self.client.post(
                "/register",
                json={"email": "new@example.test"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "check_inbox"})
        provision.assert_called_once_with("new@example.test")
        deliver.assert_called_once_with("new@example.test")
        notify.assert_called_once_with("email/password", "new-owner")

    def test_existing_email_registration_does_not_notify(self):
        with (
            patch.object(auth_router, "is_password_setup_configured", return_value=True),
            patch.object(
                auth_router,
                "find_or_provision_user",
                return_value=(SimpleNamespace(uid="existing-owner"), False),
            ),
            patch.object(auth_router, "deliver_password_setup_email") as deliver,
            patch.object(
                auth_router,
                "send_new_user_registration_notification",
            ) as notify,
        ):
            response = self.client.post(
                "/register",
                json={"email": "existing@example.test"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "check_inbox"})
        deliver.assert_called_once_with("existing@example.test")
        notify.assert_not_called()

    def test_new_and_existing_registration_responses_are_identical(self):
        new_user = SimpleNamespace(uid="new-owner", email="same@example.test")
        with (
            patch.object(auth_router, "is_password_setup_configured", return_value=True),
            patch.object(
                auth_router,
                "find_or_provision_user",
                return_value=(new_user, True),
            ),
            patch.object(auth_router, "deliver_password_setup_email"),
            patch.object(auth_router, "send_new_user_registration_notification"),
        ):
            created = self.client.post(
                "/register",
                json={"email": "same@example.test"},
            )
        with (
            patch.object(auth_router, "is_password_setup_configured", return_value=True),
            patch.object(
                auth_router,
                "find_or_provision_user",
                return_value=(SimpleNamespace(uid="existing-owner"), False),
            ),
            patch.object(auth_router, "deliver_password_setup_email"),
            patch.object(auth_router, "send_new_user_registration_notification"),
        ):
            existing = self.client.post(
                "/register",
                json={"email": "same@example.test"},
            )

        self.assertEqual(created.status_code, existing.status_code)
        self.assertEqual(created.content, existing.content)

    def test_registration_ignores_cached_client_password(self):
        user = SimpleNamespace(uid="new-owner")
        with (
            patch.object(auth_router, "is_password_setup_configured", return_value=True),
            patch.object(
                auth_router, "find_or_provision_user", return_value=(user, True)
            ) as provision,
            patch.object(auth_router, "deliver_password_setup_email"),
            patch.object(auth_router, "send_new_user_registration_notification"),
        ):
            response = self.client.post(
                "/register",
                json={"email": "new@example.test", "password": "attacker-known"},
            )

        self.assertEqual(response.status_code, 200)
        provision.assert_called_once_with("new@example.test")

    def test_registration_rejects_control_characters_before_firebase(self):
        with patch.object(auth_router, "find_or_provision_user") as provision:
            response = self.client.post(
                "/register",
                json={"email": "log-line-one\nlog-line-two@example.test"},
            )

        self.assertEqual(response.status_code, 422)
        provision.assert_not_called()

    def test_logout_clears_session_cookie(self):
        response = self.client.delete("/auth/session")
        self.assertEqual(response.status_code, 200)
        cookie = response.headers.get("set-cookie", "").lower()
        self.assertIn("session=", cookie)
        self.assertIn("max-age=0", cookie)
        self.assertEqual(response.headers.get("cache-control"), "no-store")


if __name__ == "__main__":
    unittest.main()
