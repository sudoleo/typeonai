import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routers import auth as auth_router


class AuthSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = FastAPI()
        app.include_router(auth_router.router)
        cls.client = TestClient(app)

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
            patch.object(
                auth_router.auth,
                "get_user_by_email",
                side_effect=auth_router.firebase_admin.auth.UserNotFoundError("missing"),
            ),
            patch.object(auth_router.auth, "create_user", return_value=user),
            patch.object(
                auth_router.auth,
                "create_custom_token",
                return_value=b"custom-token",
            ),
            patch.object(
                auth_router,
                "send_new_user_registration_notification",
            ) as notify,
        ):
            response = self.client.post(
                "/register",
                json={"email": "new@example.test", "password": "secret123"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["uid"], "new-owner")
        notify.assert_called_once_with("email/password", "new-owner")

    def test_existing_email_registration_does_not_notify(self):
        with (
            patch.object(
                auth_router.auth,
                "get_user_by_email",
                return_value=SimpleNamespace(uid="existing-owner"),
            ),
            patch.object(
                auth_router,
                "send_new_user_registration_notification",
            ) as notify,
        ):
            response = self.client.post(
                "/register",
                json={"email": "existing@example.test", "password": "secret123"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "check_inbox"})
        notify.assert_not_called()

    def test_logout_clears_session_cookie(self):
        response = self.client.delete("/auth/session")
        self.assertEqual(response.status_code, 200)
        cookie = response.headers.get("set-cookie", "").lower()
        self.assertIn("session=", cookie)
        self.assertIn("max-age=0", cookie)
        self.assertEqual(response.headers.get("cache-control"), "no-store")


if __name__ == "__main__":
    unittest.main()
