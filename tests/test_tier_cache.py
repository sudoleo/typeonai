"""Tests fuer den TTL-Cache der Firestore-Tier-Lookups (app/core/security.py).

is_user_pro/is_user_admin teilen sich einen gecachten Fetch des
users/{uid}-Dokuments: ein Firestore-Read statt zwei pro Aufrufstelle, 60s TTL,
Fehler werden nicht gecacht.
"""

from unittest.mock import MagicMock, patch

import pytest
from cachetools import TTLCache

import app.core.security as security


def make_firestore_mock(data, exists=True):
    doc = MagicMock()
    doc.exists = exists
    doc.to_dict.return_value = data
    db = MagicMock()
    db.collection.return_value.document.return_value.get.return_value = doc
    return db


def get_call_count(db):
    return db.collection.return_value.document.return_value.get.call_count


@pytest.fixture
def fresh_cache():
    """Jeder Test bekommt einen eigenen Cache mit steuerbarer Uhr."""
    clock = [0.0]
    cache = TTLCache(maxsize=64, ttl=security.TIER_CACHE_TTL_SECONDS, timer=lambda: clock[0])
    with patch.object(security, "_tier_cache", cache):
        yield clock


def test_tier_checks_share_one_firestore_read(fresh_cache):
    db = make_firestore_mock({"tier": "pro", "role": "admin"})
    with patch.object(security, "db_firestore", db):
        assert security.is_user_pro("uid-a") is True
        assert security.is_user_admin("uid-a") is True
    assert get_call_count(db) == 1


def test_flags_derived_like_before(fresh_cache):
    cases = [
        ({"tier": "premium"}, {"pro": True, "admin": False}),
        ({"tier": "pro"}, {"pro": True, "admin": False}),
        ({"tier": "early"}, {"pro": False, "admin": False}),
        ({"early": "true"}, {"pro": False, "admin": False}),
        ({"role": "admin"}, {"pro": False, "admin": True}),
        ({}, {"pro": False, "admin": False}),
    ]
    for i, (data, expected) in enumerate(cases):
        uid = f"uid-flags-{i}"
        db = make_firestore_mock(data)
        with patch.object(security, "db_firestore", db):
            assert security.is_user_pro(uid) is expected["pro"], data
            assert security.is_user_admin(uid) is expected["admin"], data


def test_missing_document_is_not_pro(fresh_cache):
    db = make_firestore_mock(None, exists=False)
    with patch.object(security, "db_firestore", db):
        assert security.is_user_pro("uid-missing") is False
    assert get_call_count(db) == 1


def test_cache_expires_after_ttl(fresh_cache):
    clock = fresh_cache
    db = make_firestore_mock({"tier": "free"})
    with patch.object(security, "db_firestore", db):
        assert security.is_user_pro("uid-ttl") is False
        assert get_call_count(db) == 1

        # Innerhalb der TTL: kein weiterer Read.
        clock[0] += security.TIER_CACHE_TTL_SECONDS - 1
        assert security.is_user_pro("uid-ttl") is False
        assert get_call_count(db) == 1

        # Nach Ablauf der TTL greift ein frischer Read (z.B. neu vergebener Pro-Tag).
        clock[0] += 2
        db.collection.return_value.document.return_value.get.return_value.to_dict.return_value = {"tier": "pro"}
        assert security.is_user_pro("uid-ttl") is True
        assert get_call_count(db) == 2


def test_invalidate_tier_cache_forces_fresh_read(fresh_cache):
    db = make_firestore_mock({"tier": "pro"})
    with patch.object(security, "db_firestore", db):
        assert security.is_user_pro("uid-inval") is True
        security.invalidate_tier_cache("uid-inval")
        db.collection.return_value.document.return_value.get.return_value.to_dict.return_value = {}
        assert security.is_user_pro("uid-inval") is False
    assert get_call_count(db) == 2


def test_firestore_errors_are_not_cached(fresh_cache):
    db = MagicMock()
    db.collection.return_value.document.return_value.get.side_effect = RuntimeError("boom")
    with patch.object(security, "db_firestore", db):
        # Fail-closed wie vorher: Fehler -> keine Rechte.
        assert security.is_user_pro("uid-err") is False
        assert security.is_user_admin("uid-err") is False
    # Beide Aufrufe haben Firestore erneut versucht (kein Caching des Fehlers).
    assert get_call_count(db) == 2


def test_mock_auth_hook_bypasses_firestore(fresh_cache, monkeypatch):
    monkeypatch.setenv("MOCK_AUTH", "1")
    db = MagicMock()
    with patch.object(security, "db_firestore", db):
        assert security.is_user_pro(security.E2E_MOCK_UID) is False
        assert security.is_user_admin(security.E2E_MOCK_UID) is False
    db.collection.assert_not_called()
