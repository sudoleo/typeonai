"""Kontostufen aus dem Admin-Dashboard setzen.

Vorher wurde ``users/{uid}.tier`` von Hand in der Firebase-Konsole gesetzt.
Diese Tests halten die drei Dinge fest, die daran wichtig sind: nur Admins
duerfen es, die Aenderung wird protokolliert, und der TTL-Cache wird verworfen
(sonst greift die neue Stufe erst nach einer Minute).
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routers import admin as admin_router
from app.core.rate_limit import limiter
from app.services import account_tier

AUTH_HEADER = {"Authorization": "Bearer admin-token"}


# --- Firestore-Attrappe ----------------------------------------------------

class FakeSnapshot:
    def __init__(self, doc_id, data):
        self.id = doc_id
        self._data = data

    @property
    def exists(self):
        return self._data is not None

    def to_dict(self):
        return None if self._data is None else dict(self._data)


class FakeDocument:
    def __init__(self, store, doc_id):
        self.store = store
        self.doc_id = doc_id

    def get(self, transaction=None):
        return FakeSnapshot(self.doc_id, self.store.get(self.doc_id))

    def set(self, data, merge=False):
        existing = self.store.get(self.doc_id) or {}
        self.store[self.doc_id] = {**existing, **dict(data)} if merge else dict(data)

    def delete(self):
        self.store.pop(self.doc_id, None)


class FakeQuery:
    def __init__(self, rows):
        self.rows = rows

    def limit(self, _value):
        return self

    def order_by(self, _field, direction=None):
        return self

    def stream(self):
        return [FakeSnapshot(doc_id, data) for doc_id, data in self.rows]


class FakeCollection:
    def __init__(self, store):
        self.store = store

    def document(self, doc_id):
        return FakeDocument(self.store, doc_id)

    def where(self, *args, filter=None):
        # Nur die eine Form, die account_tier benutzt: Gleichheit auf "tier".
        if filter is not None:
            field, _op, value = filter.field_path, filter.op_string, filter.value
        else:
            field, _op, value = args
        return FakeQuery([
            (doc_id, data) for doc_id, data in self.store.items()
            if data.get(field) == value
        ])

    def order_by(self, _field, direction=None):
        return FakeQuery(list(self.store.items()))

    def limit(self, _value):
        return FakeQuery(list(self.store.items()))


class FakeDb:
    def __init__(self):
        self.stores = {
            "users": {},
            "account_tier_audit": {},
            "account_deletion_jobs": {},
        }

    def collection(self, name):
        return FakeCollection(self.stores.setdefault(name, {}))


@pytest.fixture
def db():
    fake = FakeDb()
    with patch.object(account_tier, "db_firestore", fake), \
         patch.object(account_tier.persistence_guard, "ensure_account_write_allowed",
                      lambda **kwargs: None):
        yield fake


@pytest.fixture
def client():
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(admin_router.router)
    limiter.enabled = False
    try:
        yield TestClient(app)
    finally:
        limiter.enabled = True


def as_admin():
    return (
        patch.object(admin_router, "extract_id_token", return_value="admin-token"),
        patch.object(admin_router, "verify_user_token", return_value="admin-uid"),
        patch.object(admin_router, "is_user_admin", return_value=True),
    )


def auth_user(uid="uid-1", email="tester@example.test"):
    return patch.object(
        account_tier.auth, "get_user", return_value=SimpleNamespace(uid=uid, email=email)
    )


# --- Service ---------------------------------------------------------------

def test_set_tier_writes_the_field_audits_it_and_drops_the_cache(db):
    with auth_user(), patch.object(account_tier, "invalidate_tier_cache") as invalidate:
        account = account_tier.set_tier(
            "uid-1", "plus", admin_uid="admin-uid", note="Asked by email")

    assert db.stores["users"]["uid-1"]["tier"] == "plus"
    assert db.stores["users"]["uid-1"]["tier_updated_by"] == "admin-uid"
    assert isinstance(db.stores["users"]["uid-1"]["tier_updated_at"], datetime)
    invalidate.assert_called_once_with("uid-1")

    entries = list(db.stores["account_tier_audit"].values())
    assert len(entries) == 1
    assert entries[0]["from_tier"] == "free"
    assert entries[0]["to_tier"] == "plus"
    assert entries[0]["changed_by"] == "admin-uid"

    # Die Antwort zeigt genau die Faehigkeiten, die der Lauf spaeter anwendet.
    assert account["tier"] == "plus"
    assert account["previous_tier"] == "free"
    assert account["attachments"] is True
    assert account["resolve"] is True
    assert account["is_pro"] is False
    assert account["deep_think"] is False


def test_set_tier_keeps_unrelated_profile_fields(db):
    db.stores["users"]["uid-1"] = {"role": "admin", "email_opt_in": True}
    with auth_user(), patch.object(account_tier, "invalidate_tier_cache"):
        account_tier.set_tier("uid-1", "pro", admin_uid="admin-uid")
    assert db.stores["users"]["uid-1"]["role"] == "admin"
    assert db.stores["users"]["uid-1"]["email_opt_in"] is True


def test_set_tier_rejects_an_unknown_tier(db):
    with auth_user(), pytest.raises(account_tier.AccountTierError) as excinfo:
        account_tier.set_tier("uid-1", "premium-plus", admin_uid="admin-uid")
    assert excinfo.value.code == "invalid_tier"
    assert db.stores["users"] == {}


def test_an_account_being_deleted_gets_no_tier():
    fake = FakeDb()

    def blocked(**kwargs):
        raise account_tier.persistence_guard.AccountDeletionInProgress("gone")

    with patch.object(account_tier, "db_firestore", fake), \
         patch.object(account_tier.persistence_guard, "ensure_account_write_allowed", blocked), \
         auth_user(), \
         pytest.raises(account_tier.persistence_guard.AccountDeletionInProgress):
        account_tier.set_tier("uid-1", "plus", admin_uid="admin-uid")
    assert fake.stores["users"] == {}


def test_lookup_accepts_an_email(db):
    with patch.object(account_tier.auth, "get_user_by_email",
                      return_value=SimpleNamespace(uid="uid-1")), \
         auth_user():
        account = account_tier.get_account("tester@example.test")
    assert account["uid"] == "uid-1"
    assert account["tier"] == "free"


def test_listing_covers_the_legacy_premium_tag(db):
    db.stores["users"] = {
        "uid-plus": {"tier": "plus"},
        "uid-pro": {"tier": "pro"},
        "uid-legacy": {"tier": "premium"},
        "uid-free": {"tier": "free"},
        "uid-none": {},
    }
    accounts = account_tier.list_elevated_accounts()
    by_uid = {item["uid"]: item["tier"] for item in accounts}
    assert by_uid == {"uid-plus": "plus", "uid-pro": "pro", "uid-legacy": "pro"}


def test_recent_changes_are_readable(db):
    db.stores["account_tier_audit"]["a"] = {
        "uid": "uid-1", "from_tier": "free", "to_tier": "plus",
        "changed_by": "admin-uid", "note": "test",
        "changed_at": datetime(2026, 8, 31, tzinfo=timezone.utc),
    }
    changes = account_tier.list_recent_changes()
    assert changes[0]["to_tier"] == "plus"
    assert changes[0]["changed_at"].startswith("2026-08-31")


# --- Endpoints -------------------------------------------------------------

def test_endpoints_require_admin(client, db):
    with patch.object(admin_router, "extract_id_token", return_value="token"), \
         patch.object(admin_router, "verify_user_token", return_value="uid-1"), \
         patch.object(admin_router, "is_user_admin", return_value=False):
        assert client.get("/api/admin/account-tier?identifier=uid-1",
                          headers=AUTH_HEADER).status_code == 403
        assert client.put("/api/admin/account-tier", headers=AUTH_HEADER,
                          json={"identifier": "uid-1", "tier": "pro"}).status_code == 403
        assert client.get("/api/admin/account-tiers",
                          headers=AUTH_HEADER).status_code == 403


def test_put_sets_the_tier(client, db):
    p1, p2, p3 = as_admin()
    with p1, p2, p3, auth_user(), patch.object(account_tier, "invalidate_tier_cache"):
        response = client.put(
            "/api/admin/account-tier", headers=AUTH_HEADER,
            json={"identifier": "uid-1", "tier": "plus", "note": "Tester"})
    assert response.status_code == 200
    assert response.json()["account"]["tier"] == "plus"
    assert db.stores["users"]["uid-1"]["tier"] == "plus"


def test_put_rejects_a_tier_outside_the_three(client, db):
    p1, p2, p3 = as_admin()
    with p1, p2, p3, auth_user():
        response = client.put(
            "/api/admin/account-tier", headers=AUTH_HEADER,
            json={"identifier": "uid-1", "tier": "enterprise"})
    assert response.status_code == 422
    assert db.stores["users"] == {}


def test_lookup_of_an_unknown_account_is_a_404(client, db):
    p1, p2, p3 = as_admin()
    with p1, p2, p3, \
         patch.object(account_tier.auth, "get_user",
                      side_effect=account_tier.auth.UserNotFoundError("nope")):
        response = client.get("/api/admin/account-tier?identifier=missing",
                              headers=AUTH_HEADER)
    assert response.status_code == 404
    assert response.json()["detail"]["error_code"] == "not_found"
