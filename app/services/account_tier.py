"""Kontostufen administrieren: nachschlagen, setzen, protokollieren.

Bis hierher wurde ``users/{uid}.tier`` von Hand in der Firebase-Konsole
gesetzt. Mit drei Stufen ist das zu fehleranfaellig -- dieses Modul ist die
einzige Schreibstelle und haelt drei Dinge zusammen:

* das Feld ``tier`` selbst (plus ``tier_updated_at``/``tier_updated_by``),
* ein unveraenderliches Audit-Dokument je Aenderung,
* die Invalidierung des TTL-Caches aus app/core/security.py, damit die neue
  Stufe sofort statt erst nach 60 Sekunden greift.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from firebase_admin import auth
from google.cloud.firestore_v1.base_query import FieldFilter

from app.core.entitlements import TIERS, entitlements_for, normalize_tier
from app.core.observability import safe_exception
from app.core.security import db_firestore, invalidate_tier_cache
from app.services import persistence_guard

USERS_COLLECTION = "users"
AUDIT_COLLECTION = "account_tier_audit"
MAX_NOTE_CHARS = 300
#: Obergrenze fuer die Uebersichtsliste. Bezahlte/erhoehte Konten sind eine
#: kleine, kuratierte Menge; wer mehr braucht, sucht gezielt per Identifier.
LIST_LIMIT = 200


class AccountTierError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _where_equal(collection, field: str, value):
    """Use the current Firestore filter API with mock-compatible fallback."""
    try:
        return collection.where(filter=FieldFilter(field, "==", value))
    except TypeError:
        return collection.where(field, "==", value)


def validate_tier(value) -> str:
    text = str(value or "").strip().lower()
    if text not in TIERS:
        raise AccountTierError(
            "invalid_tier", "Tier must be one of: " + ", ".join(TIERS)
        )
    return text


def resolve_uid(identifier: str) -> str:
    """UID aus einer UID ODER einer E-Mail. Der Admin tippt, was er hat."""
    text = str(identifier or "").strip()
    if not text:
        raise AccountTierError("invalid_request", "Enter a UID or an email address.")
    try:
        if "@" in text:
            return auth.get_user_by_email(text).uid
        return auth.get_user(text).uid
    except auth.UserNotFoundError:
        raise AccountTierError("not_found", "No account matches this UID or email.") from None
    except ValueError:
        raise AccountTierError("invalid_request", "That is not a valid UID or email address.") from None


def _profile(uid: str, db) -> dict:
    snapshot = db.collection(USERS_COLLECTION).document(uid).get()
    return (snapshot.to_dict() or {}) if snapshot.exists else {}


def _serialize(uid: str, data: dict, *, email: str = "") -> dict:
    entitlements = entitlements_for(data.get("tier"))
    updated_at = data.get("tier_updated_at")
    return {
        "uid": uid,
        "email": email,
        "tier": entitlements.tier,
        # Dieselben Flags, die auch der Nutzer bekommt -- so sieht der Admin im
        # Dashboard genau das, was der Account tatsaechlich darf.
        "is_pro": entitlements.is_pro,
        "deep_think": entitlements.deep_think,
        "premium_models": entitlements.premium_models,
        "attachments": entitlements.attachments,
        "resolve": entitlements.resolve,
        "role": str(data.get("role") or ""),
        "tier_updated_at": updated_at.isoformat() if isinstance(updated_at, datetime) else "",
        "tier_updated_by": str(data.get("tier_updated_by") or ""),
        "tier_note": str(data.get("tier_note") or ""),
    }


def get_account(identifier: str, db=None) -> dict:
    db = db if db is not None else db_firestore
    uid = resolve_uid(identifier)
    email = ""
    try:
        email = auth.get_user(uid).email or ""
    except Exception as exc:  # Auth-Ausfall darf die Anzeige nicht brechen.
        logging.warning("account tier email lookup failed category=%s", safe_exception(exc))
    return _serialize(uid, _profile(uid, db), email=email)


def set_tier(identifier: str, tier, *, admin_uid: str, note: str = "", db=None) -> dict:
    db = db if db is not None else db_firestore
    target_tier = validate_tier(tier)
    uid = resolve_uid(identifier)
    note = str(note or "").strip()[:MAX_NOTE_CHARS]
    now = _utcnow()

    # Ein in Loeschung befindliches Konto bekommt keine neue Stufe: der Sweep
    # wuerde das Feld sonst als Waise zuruecklassen. Bewusst ohne Transaktion --
    # das hier ist eine manuelle Adminaktion, keine nebenlaeufige Nutzermutation;
    # das enge Rennen zwischen "Admin tippt" und "Nutzer loescht" ist die
    # Komplexitaet einer Transaktion nicht wert.
    persistence_guard.ensure_account_write_allowed(uid=uid, db=db, now=now)

    previous = _profile(uid, db)
    previous_tier = normalize_tier(previous.get("tier"))

    db.collection(USERS_COLLECTION).document(uid).set(
        {
            "tier": target_tier,
            "tier_updated_at": now,
            "tier_updated_by": str(admin_uid or ""),
            "tier_note": note,
        },
        merge=True,
    )
    # Erst nach dem Schreiben: sonst koennte ein paralleler Request den alten
    # Wert sofort wieder in den Cache legen.
    invalidate_tier_cache(uid)

    try:
        db.collection(AUDIT_COLLECTION).document(uuid.uuid4().hex).set(
            {
                "schema_version": 1,
                "uid": uid,
                "from_tier": previous_tier,
                "to_tier": target_tier,
                "changed_by": str(admin_uid or ""),
                "note": note,
                "changed_at": now,
            }
        )
    except Exception as exc:
        # Die Stufe steht bereits; ein fehlendes Audit-Dokument darf den
        # Adminvorgang nicht scheitern lassen, muss aber sichtbar sein.
        logging.error("account tier audit write failed category=%s", safe_exception(exc))

    email = ""
    try:
        email = auth.get_user(uid).email or ""
    except Exception:
        pass
    account = _serialize(uid, {**previous, **{
        "tier": target_tier,
        "tier_updated_at": now,
        "tier_updated_by": str(admin_uid or ""),
        "tier_note": note,
    }}, email=email)
    account["previous_tier"] = previous_tier
    return account


def list_elevated_accounts(db=None) -> list[dict]:
    """Alle Konten oberhalb von Free. "premium" ist der historische Pro-Tag."""
    db = db if db is not None else db_firestore
    seen: dict[str, dict] = {}
    for tag in ("plus", "pro", "premium"):
        query = _where_equal(db.collection(USERS_COLLECTION), "tier", tag).limit(LIST_LIMIT)
        for doc in query.stream():
            if doc.id in seen:
                continue
            seen[doc.id] = _serialize(doc.id, doc.to_dict() or {})
    accounts = list(seen.values())
    accounts.sort(key=lambda item: (item["tier"], item["uid"]))
    return accounts[:LIST_LIMIT]


def list_recent_changes(limit: int = 25, db=None) -> list[dict]:
    db = db if db is not None else db_firestore
    query = (
        db.collection(AUDIT_COLLECTION)
        .order_by("changed_at", direction="DESCENDING")
        .limit(max(1, min(int(limit or 25), 100)))
    )
    entries = []
    for doc in query.stream():
        data = doc.to_dict() or {}
        changed_at = data.get("changed_at")
        entries.append({
            "uid": str(data.get("uid") or ""),
            "from_tier": normalize_tier(data.get("from_tier")),
            "to_tier": normalize_tier(data.get("to_tier")),
            "changed_by": str(data.get("changed_by") or ""),
            "note": str(data.get("note") or ""),
            "changed_at": changed_at.isoformat() if isinstance(changed_at, datetime) else "",
        })
    return entries
