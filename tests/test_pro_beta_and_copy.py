from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routers import users as users_router
from app.core.rate_limit import limiter


ROOT = Path(__file__).resolve().parents[1]


class WaitlistSnapshot:
    def __init__(self, data):
        self.data = data

    @property
    def exists(self):
        return self.data is not None


class WaitlistDocument:
    def __init__(self):
        self.data = None
        self.writes = 0

    def get(self):
        return WaitlistSnapshot(self.data)

    def set(self, data):
        self.data = dict(data)
        self.writes += 1


class WaitlistCollection:
    def __init__(self, document):
        self.waitlist_document = document

    def document(self, uid):
        assert uid == "uid-1"
        return self.waitlist_document

    def where(self, *_args):
        return self

    def limit(self, _value):
        return self

    def stream(self):
        return []


class WaitlistDb:
    def __init__(self):
        self.waitlist_document = WaitlistDocument()

    def collection(self, name):
        assert name == "pro_waitlist"
        return WaitlistCollection(self.waitlist_document)


def _client():
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(users_router.router)
    return TestClient(app)


def test_pro_beta_request_is_idempotent_and_active_pro_is_rejected():
    database = WaitlistDb()
    with patch.object(users_router, "verify_user_token", return_value="uid-1"), \
            patch.object(users_router, "is_user_pro", return_value=False), \
            patch.object(users_router, "db_firestore", database), \
            patch.object(users_router.auth, "get_user", return_value=SimpleNamespace(email="user@example.test")):
        first = _client().post("/track-interest", json={"id_token": "token", "source": "pro_beta_modal"})
        second = _client().post("/track-interest", json={"id_token": "token", "source": "pro_beta_modal"})
    assert first.status_code == 200
    assert first.json()["status"] == "success"
    assert second.json() == {
        "status": "pending",
        "already_requested": True,
        "message": "Your Pro beta request is already pending.",
    }
    assert database.waitlist_document.writes == 1
    assert database.waitlist_document.data["status"] == "pending"

    with patch.object(users_router, "verify_user_token", return_value="uid-1"), \
            patch.object(users_router, "is_user_pro", return_value=True):
        active = _client().post("/track-interest", json={"id_token": "token"})
    assert active.status_code == 409


def test_pro_beta_modal_has_inline_states_without_pricing_or_purchase_copy():
    html = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    js = (ROOT / "static" / "js" / "app-init.js").read_text(encoding="utf-8")
    modal = html[html.index('id="proFeatureModal"'):html.index('id="popupContainer"')]
    handler_start = js.index("if (upgradeBtn)")
    handler = js[handler_start:js.index("// --- DEEP THINK", handler_start)]
    assert "Join Pro beta" in modal
    assert "Request Pro access" in modal
    assert 'id="proRequestStatus"' in modal
    assert "pricing-grid" not in modal
    assert "€" not in modal
    assert "alert(" not in handler
    assert "is-pending" in handler and "is-success" in handler and "is-error" in handler
    assert "if (window.isUserPro) return false" in js


def test_sidebar_uses_compact_pro_beta_entry_but_keeps_full_accessible_label():
    html = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    layout = (ROOT / "static" / "css" / "layout.css").read_text(encoding="utf-8")
    assert 'aria-label="Request Pro access"' in html
    assert '>Pro beta</a>' in html
    assert "#upgradeLink" in layout
    assert "white-space: nowrap" in layout


def test_user_visible_plan_copy_has_no_stale_literal_plan_values():
    targets = [
        ROOT / "templates" / "index.html",
        ROOT / "static" / "js" / "watch.js",
        ROOT / "static" / "js" / "app-init.js",
        ROOT / "static" / "firebase.js",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in targets)
    for stale in (
        "25 / day", "100 / day", "€10", "Pricing preview",
        "Pro includes 5 active Watches", "up to 25 standard AI-powered queries",
    ):
        assert stale not in text
    assert "window.APP_LIMITS" in text
    assert 'id="watchUsageDisplay"' in text
