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


def test_locked_feature_modal_explains_the_cost_and_sells_nothing():
    html = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    js = (ROOT / "static" / "js" / "app-init.js").read_text(encoding="utf-8")
    modal = html[html.index('id="proFeatureModal"'):html.index('id="popupContainer"')]
    assert "while I'm testing it, it's free" in modal
    assert "nothing to buy today" in modal
    assert "switched off by default" in modal
    # Kein Zukunftsversprechen in beide Richtungen: Pro-Features gibt es, eine
    # spaetere Mitgliedschaft ist moeglich und wird als moeglich benannt.
    assert "membership" in modal
    assert "per account instead of for everyone" in modal
    assert "contact@consens.io" in modal
    for sales_copy in ("Request Pro access", "Join Pro beta", "smokeTestUpgradeBtn", "pricing-grid", "€"):
        assert sales_copy not in modal
    # Der Zugangs-Request ist aus dem Frontend entfernt; der Endpunkt bleibt ungenutzt bestehen.
    assert 'fetch("/track-interest"' not in js
    assert "if (window.isUserPro) return false" in js


def test_sidebar_link_explains_limits_instead_of_offering_an_upgrade():
    html = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    layout = (ROOT / "static" / "css" / "layout.css").read_text(encoding="utf-8")
    assert 'aria-label="Why there are limits"' in html
    assert '>Why limits?</a>' in html
    assert "#upgradeLink" in layout
    assert "white-space: nowrap" in layout


def test_public_pages_state_the_free_while_testing_position():
    pages = {
        "landing": ROOT / "templates" / "landing.html",
        "about": ROOT / "templates" / "about.html",
        "app": ROOT / "templates" / "index.html",
    }
    for name, path in pages.items():
        text = path.read_text(encoding="utf-8").lower()
        assert "free while i'm testing it" in text, name
        assert "costs me real money" in text or "costs me money" in text or "lands on me" in text, name
        # Kein pauschales "es wird nie etwas kosten": die Moeglichkeit einer
        # spaeteren Mitgliedschaft steht ausdruecklich auf jeder dieser Seiten,
        # zusammen mit dem Kontaktweg, ueber den sich Interesse zeigen kann.
        assert "membership" in text, name
        assert "contact@consens.io" in text, name


def test_no_page_claims_that_nothing_will_ever_be_for_sale():
    for path in (ROOT / "templates").glob("*.html"):
        text = path.read_text(encoding="utf-8")
        for absolute_claim in (
            "There is nothing to buy here",
            "Nothing on it is for sale",
            "no subscription, no waiting list",
            "cannot be bought",
        ):
            assert absolute_claim not in text, f"{path.name}: {absolute_claim}"


def test_no_purchase_call_to_action_survives_anywhere_in_the_ui():
    targets = list((ROOT / "templates").glob("*.html")) + list((ROOT / "static" / "js").glob("*.js"))
    targets.append(ROOT / "static" / "firebase.js")
    for path in targets:
        if path.name == "admin.html":
            continue
        text = path.read_text(encoding="utf-8")
        for sales_copy in ("Request Pro access", "Join Pro beta", "upgrade your plan", "Pro plan", "Free plan"):
            assert sales_copy not in text, f"{path.name}: {sales_copy}"


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
