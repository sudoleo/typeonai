"""Sensitive actions use live revocation; ordinary auth uses tombstone cache."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.api.routers import admin as admin_router
from app.core import security


def test_live_revocation_flag_is_forwarded_for_sensitive_verification():
    decoded = {"uid": "owner", "email_verified": True}
    with (
        patch.object(security.auth, "verify_id_token", return_value=decoded) as verify,
        patch.object(security, "_is_account_tombstoned", return_value=False),
    ):
        assert security.verify_user_token("token", check_revoked=True) == "owner"

    assert verify.call_args.kwargs["check_revoked"] is True


def test_account_deletion_tombstone_rejects_otherwise_valid_token():
    decoded = {"uid": "deleted-owner", "email_verified": True}
    with (
        patch.object(security.auth, "verify_id_token", return_value=decoded),
        patch.object(security, "_is_account_tombstoned", return_value=True),
    ):
        with pytest.raises(Exception, match="Invalid token"):
            security.verify_user_token("still-cryptographically-valid")


def test_pending_deletion_never_expires_before_cleanup_completes():
    expired = datetime.now(timezone.utc) - timedelta(days=1)
    snapshot = SimpleNamespace(
        exists=True,
        to_dict=lambda: {
            "status": "pending",
            "tombstone_expires_at": expired,
        },
    )
    document = SimpleNamespace(get=lambda: snapshot)
    collection = SimpleNamespace(document=lambda _uid: document)
    database = SimpleNamespace(collection=lambda _name: collection)
    security.invalidate_auth_tombstone_cache("pending-owner")
    with patch.object(security, "db_firestore", database):
        assert security._is_account_tombstoned("pending-owner") is True
    security.invalidate_auth_tombstone_cache("pending-owner")


def test_admin_boundary_checks_revocation_and_maps_tier_outage_to_503():
    request = SimpleNamespace(
        headers={"Authorization": "Bearer token"},
        cookies={},
    )
    with (
        patch.object(admin_router, "verify_user_token", return_value="admin") as verify,
        patch.object(
            admin_router,
            "is_user_admin",
            side_effect=security.TierStatusUnavailable("outage"),
        ),
    ):
        with pytest.raises(admin_router.HTTPException) as exc_info:
            admin_router._require_admin(request, {})

    assert verify.call_args.kwargs["check_revoked"] is True
    assert exc_info.value.status_code == 503
