"""Hashed, user-bound API keys for the public Consensus API."""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone

from google.cloud.firestore_v1.base_query import FieldFilter


API_KEYS_COLLECTION = "api_consensus_keys"
API_KEY_PREFIX = "cns_live_"
API_KEY_SECRET_BYTES = 32


class ApiKeyError(Exception):
    pass


class InvalidApiKey(ApiKeyError):
    pass


class ApiKeyNotFound(ApiKeyError):
    pass


@dataclass(frozen=True)
class AuthenticatedApiKey:
    key_id: str
    uid: str
    label: str


def hash_api_key(api_key: str) -> str:
    value = str(api_key or "").strip()
    if not value or not value.startswith(API_KEY_PREFIX):
        raise InvalidApiKey("Invalid API key")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class FirestoreApiKeyRepository:
    def __init__(self, db):
        self._db = db

    def issue(self, uid: str, *, label: str = "", created_by: str = "") -> dict:
        uid = str(uid or "").strip()
        if not uid:
            raise ValueError("uid must not be empty")
        clean_label = str(label or "").strip()[:80]
        plaintext = API_KEY_PREFIX + secrets.token_urlsafe(API_KEY_SECRET_BYTES)
        key_id = hash_api_key(plaintext)
        now = datetime.now(timezone.utc)
        self._ref(key_id).set(
            {
                "schema_version": 1,
                "uid": uid,
                "label": clean_label,
                "prefix": plaintext[:18],
                "status": "active",
                "created_at": now,
                "created_by": str(created_by or "").strip(),
                "updated_at": now,
            }
        )
        return {
            "key_id": key_id,
            "api_key": plaintext,
            "uid": uid,
            "label": clean_label,
            "prefix": plaintext[:18],
            "status": "active",
            "created_at": now,
        }

    def authenticate(self, api_key: str) -> AuthenticatedApiKey:
        key_id = hash_api_key(api_key)
        snap = self._ref(key_id).get()
        if not snap.exists:
            raise InvalidApiKey("Invalid API key")
        data = snap.to_dict() or {}
        uid = str(data.get("uid") or "").strip()
        if data.get("status") != "active" or not uid:
            raise InvalidApiKey("Invalid API key")
        try:
            self._ref(key_id).update({"last_used_at": datetime.now(timezone.utc)})
        except Exception:
            # Authentication must not fail just because best-effort audit
            # metadata could not be refreshed.
            pass
        return AuthenticatedApiKey(
            key_id=key_id,
            uid=uid,
            label=str(data.get("label") or ""),
        )

    def revoke(self, key_id: str) -> dict:
        key_id = _validate_key_id(key_id)
        ref = self._ref(key_id)
        snap = ref.get()
        if not snap.exists:
            raise ApiKeyNotFound("API key not found")
        now = datetime.now(timezone.utc)
        ref.update({"status": "revoked", "revoked_at": now, "updated_at": now})
        data = snap.to_dict() or {}
        data.update({"key_id": key_id, "status": "revoked", "revoked_at": now})
        return data

    def list(self, *, uid: str | None = None) -> list[dict]:
        collection = self._db.collection(API_KEYS_COLLECTION)
        query = collection.where(filter=FieldFilter("uid", "==", uid)) if uid else collection
        rows = []
        for snap in query.stream():
            data = snap.to_dict() or {}
            data["key_id"] = snap.id
            data.pop("created_by", None)
            rows.append(data)
        rows.sort(
            key=lambda item: item.get("created_at")
            or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        return rows

    def _ref(self, key_id: str):
        return self._db.collection(API_KEYS_COLLECTION).document(key_id)


def _validate_key_id(key_id: str) -> str:
    value = str(key_id or "").strip().lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ApiKeyNotFound("API key not found")
    return value
