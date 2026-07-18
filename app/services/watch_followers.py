"""Visitor follow subscriptions for public Consensus Watch pages.

Besucher können eine öffentliche Watch-Seite per E-Mail "followen" und werden
bei materiellen Konsens-Änderungen benachrichtigt. Double-Opt-in: der POST
erzeugt nur einen signierten Confirm-Token (keine Persistenz); erst der Klick
auf den Bestätigungslink legt den Follower an. Bewusst ohne IP/UA-Speicherung –
gespeichert werden nur share_id, E-Mail und Zeitstempel.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime

from firebase_admin import firestore

from app.core.security import db_firestore
from app.services import share_snapshots
from app.services.watch_service import (
    WatchError,
    get_public_watch_meta,
    parse_token_payload,
    sign_token_payload,
)

FOLLOWERS_COLLECTION = "watch_followers"
MAX_FOLLOWERS_PER_SHARE = 200
CONFIRM_TOKEN_MAX_AGE_DAYS = 3

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]{2,}$")


def normalize_email(value) -> str:
    email = str(value or "").strip().lower()
    if not email or len(email) > 254 or not _EMAIL_RE.fullmatch(email):
        raise WatchError("invalid_email", "Enter a valid e-mail address.")
    return email


def follower_id(share_id: str, email: str) -> str:
    """Deterministic doc id – ein Follower pro (Seite, Adresse)."""
    return hashlib.sha256(f"{share_id}:{email}".encode("utf-8")).hexdigest()[:32]


def _followable_share(share_id: str, db) -> dict:
    share = share_snapshots.get_share(share_id, db=db)
    if (not share or share.get("status") != "active"
            or str(share.get("visibility") or "public") != "public"):
        raise WatchError("not_found", "This page cannot be followed.")
    if get_public_watch_meta(share_id, db=db) is None:
        raise WatchError("not_watched", "This page has no Consensus Watch to follow.")
    return share


def make_confirm_token(share_id: str, email: str, *, now=None) -> str:
    return sign_token_payload(
        {"sid": share_id, "em": email}, now=now, max_age_days=CONFIRM_TOKEN_MAX_AGE_DAYS,
    )


def make_follow_unsubscribe_token(share_id: str, email: str, *, now=None) -> str:
    return sign_token_payload({"sid": share_id, "em": email, "un": 1}, now=now)


def request_follow(share_id: str, email, db=None) -> dict:
    """Validate the request and hand back a confirm token (nothing persisted).

    Liefert ``token`` nur, wenn die Adresse noch nicht bestätigt folgt –
    der Router antwortet in beiden Fällen mit derselben generischen Meldung.
    """
    db = db if db is not None else db_firestore
    email = normalize_email(email)
    share = _followable_share(share_id, db)
    doc = db.collection(FOLLOWERS_COLLECTION).document(follower_id(share_id, email)).get()
    if doc.exists:
        return {"email": email, "question": str(share.get("question") or ""), "token": ""}
    return {
        "email": email,
        "question": str(share.get("question") or ""),
        "token": make_confirm_token(share_id, email),
    }


def confirm_follow(token: str, db=None) -> dict:
    db = db if db is not None else db_firestore
    payload = parse_token_payload(token)
    if payload.get("un"):
        raise WatchError("invalid_token", "This link is invalid.")
    share_id = str(payload.get("sid") or "")
    email = normalize_email(payload.get("em"))
    share = _followable_share(share_id, db)

    ref = db.collection(FOLLOWERS_COLLECTION).document(follower_id(share_id, email))
    if not ref.get().exists:
        count = sum(
            1 for _ in db.collection(FOLLOWERS_COLLECTION)
            .where("share_id", "==", share_id).stream()
        )
        if count >= MAX_FOLLOWERS_PER_SHARE:
            raise WatchError("limit_reached", "This page has reached its follower limit.")
        ref.set({
            "share_id": share_id,
            "email": email,
            "created_at": firestore.SERVER_TIMESTAMP,
        })
    return {
        "share_id": share_id,
        "email": email,
        "question": str(share.get("question") or ""),
        "share_path": share_snapshots.share_path(share.get("slug") or "", share_id),
    }


def unsubscribe_follow(token: str, db=None) -> dict:
    db = db if db is not None else db_firestore
    payload = parse_token_payload(token)
    if not payload.get("un"):
        raise WatchError("invalid_token", "This link is invalid.")
    share_id = str(payload.get("sid") or "")
    email = normalize_email(payload.get("em"))
    db.collection(FOLLOWERS_COLLECTION).document(follower_id(share_id, email)).delete()
    return {"share_id": share_id, "email": email}


def list_followers(share_id: str, db=None, max_items=MAX_FOLLOWERS_PER_SHARE) -> list[dict]:
    db = db if db is not None else db_firestore
    followers = []
    for doc in db.collection(FOLLOWERS_COLLECTION).where("share_id", "==", share_id).stream():
        data = doc.to_dict() or {}
        email = str(data.get("email") or "")
        if not email:
            continue
        created = data.get("created_at")
        followers.append({
            "id": doc.id,
            "email": email,
            "created_at": created.isoformat() if isinstance(created, datetime) else "",
        })
        if len(followers) >= max_items:
            break
    return followers


def count_followers(share_id: str, db=None) -> int:
    return len(list_followers(share_id, db=db))


def delete_followers_for_share(share_id: str, db=None) -> int:
    db = db if db is not None else db_firestore
    deleted = 0
    for doc in db.collection(FOLLOWERS_COLLECTION).where("share_id", "==", share_id).stream():
        doc.reference.delete()
        deleted += 1
    return deleted
