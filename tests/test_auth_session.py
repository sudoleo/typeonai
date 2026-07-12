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

    def test_logout_clears_session_cookie(self):
        response = self.client.delete("/auth/session")
        self.assertEqual(response.status_code, 200)
        cookie = response.headers.get("set-cookie", "").lower()
        self.assertIn("session=", cookie)
        self.assertIn("max-age=0", cookie)


if __name__ == "__main__":
    unittest.main()
