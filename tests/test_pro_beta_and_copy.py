from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routers import users as users_router
from app.core.rate_limit import limiter
from app.services import persistence_guard


ROOT = Path(__file__).resolve().parents[1]


class WaitlistSnapshot:
    def __init__(self, data):
        self.data = data

    @property
    def exists(self):
        return self.data is not None

    def to_dict(self):
        return None if self.data is None else dict(self.data)


class WaitlistDocument:
    def __init__(self):
        self.data = None
        self.writes = 0

    def get(self, transaction=None):
        return WaitlistSnapshot(self.data)

    def set(self, data, merge=False):
        if merge and self.data is not None:
            data = {**self.data, **dict(data)}
        self.data = dict(data)
        self.writes += 1


class WaitlistCollection:
    def __init__(self, document, *, allow_query=False):
        self.waitlist_document = document
        self.allow_query = allow_query

    def document(self, uid):
        assert uid == "uid-1"
        return self.waitlist_document

    def where(self, *_args):
        assert self.allow_query
        return self

    def limit(self, _value):
        return self

    def stream(self):
        return []


class WaitlistDb:
    def __init__(self):
        self.waitlist_document = WaitlistDocument()
        self.deletion_document = WaitlistDocument()

    def collection(self, name):
        if name == "pro_waitlist":
            return WaitlistCollection(self.waitlist_document, allow_query=True)
        if name == persistence_guard.ACCOUNT_DELETION_JOBS_COLLECTION:
            return WaitlistCollection(self.deletion_document)
        raise AssertionError(f"unexpected collection: {name}")

    def run_transaction(self, operation):
        transaction = WaitlistTransaction()
        result = operation(transaction)
        transaction.commit()
        return result


class WaitlistTransaction:
    def __init__(self):
        self.writes = []

    def set(self, ref, data, merge=False):
        self.writes.append((ref, dict(data), merge))

    def commit(self):
        for ref, data, merge in self.writes:
            ref.set(data, merge=merge)


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


def test_pro_beta_request_is_fenced_during_account_deletion():
    database = WaitlistDb()
    database.deletion_document.data = {"status": "pending"}
    with patch.object(users_router, "verify_user_token", return_value="uid-1"), \
            patch.object(users_router, "is_user_pro", return_value=False), \
            patch.object(users_router, "db_firestore", database), \
            patch.object(users_router.auth, "get_user", return_value=SimpleNamespace(email="user@example.test")):
        response = _client().post(
            "/track-interest",
            json={"id_token": "token", "source": "pro_beta_modal"},
        )

    assert response.status_code == 503
    assert database.waitlist_document.data is None
    assert database.waitlist_document.writes == 0


def test_locked_feature_modal_explains_the_cost_and_sells_nothing():
    html = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    js = (ROOT / "static" / "js" / "app-init.js").read_text(encoding="utf-8")
    modal = html[html.index('id="proFeatureModal"'):html.index('id="popupContainer"')]
    assert "free while I’m testing it" in modal
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
    assert ">Why limits</span>" in html
    assert "#upgradeLink" in layout
    assert "white-space: nowrap" in layout
    # Der Kontoname stand daneben und hat die Zeile ueberfuellt; er ist raus.
    assert 'id="accountLabel"' not in html


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


def test_cost_flow_matches_the_pipeline_it_claims_to_describe():
    """Der Flow im "Why limits"-Popup nennt konkrete Zahlen. Sie muessen dem
    Code entsprechen, sonst erklaert das Popup ein Produkt, das es nicht gibt.

    Geprueft wird die Kette selbst: so viele Antwortmodelle wie ein Lauf
    zulaesst, EIN Synthese-Call, und ZWEI Judges (Differences + Coverage)."""
    import app.core.config as cfg

    html = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    flow = html[html.index('class="cost-flow"'):html.index('class="pro-beta-actions"')]

    assert f"{cfg.MAX_RUN_FAMILIES} answers" in flow
    assert "1 synthesis" in flow
    assert "2 judges" in flow
    # Ein Punkt = ein bezahlter Call. Die Summe im Titel muss aufgehen.
    total = cfg.MAX_RUN_FAMILIES + 1 + 2
    assert flow.count("<i></i>") == total
    words = ["zero", "one", "two", "three", "four", "five", "six",
             "seven", "eight", "nine", "ten"]
    # Der Prosa-Satz ueber dem Flow nennt dieselbe Summe wie die Punkte.
    assert f"{words[total]} calls to {words[cfg.MAX_RUN_FAMILIES]} providers" in html
    assert f"{words[total].capitalize()} calls before you see a word" in html

    # Der zweite Judge ist der Coverage-Judge; ohne ihn waere "2 judges" falsch.
    engine = (ROOT / "app" / "services" / "llm" / "consensus_engine.py").read_text(encoding="utf-8")
    assert "_coverage_attempts" in engine
    assert (ROOT / "app" / "services" / "llm" / "coverage_judge.py").is_file()


def test_footer_shows_the_running_commit_and_links_to_the_repository():
    from app.core.version import REPO_URL, get_commit_short

    html = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    # Die handgepflegte Versionsnummer ist raus -- sie wurde nie hochgezaehlt.
    assert "v1.11.1" not in html
    assert '{{ app_commit }}' in html
    assert '{{ repo_url }}' in html
    assert 'class="sidebar-footer-repo-icon"' in html

    pages = (ROOT / "app" / "api" / "routers" / "pages.py").read_text(encoding="utf-8")
    assert '"app_commit": get_commit_short()' in pages
    assert '"repo_url": REPO_URL' in pages

    assert REPO_URL.startswith("https://github.com/")
    # Im Checkout ist der Commit lesbar; ohne .git bleibt der Wert leer und das
    # Template blendet die Zeile aus, statt etwas Falsches zu behaupten.
    commit = get_commit_short()
    assert commit == "" or (len(commit) == 7 and all(c in "0123456789abcdef" for c in commit))
