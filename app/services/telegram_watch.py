"""Per-user Telegram connection and Consensus Watch delivery helpers."""

from __future__ import annotations

import hashlib
import hmac
import html
import logging
import os
import re
import secrets
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

from google.cloud.firestore_v1.base_query import FieldFilter

from app.core.site import SITE_URL
from app.core.observability import safe_exception
from app.core.security import db_firestore
from app.services import (
    persistence_guard,
    share_snapshots,
    telegram_notifier,
    watch_service,
)


CONNECTIONS_COLLECTION = "telegram_connections"
CHATS_COLLECTION = "telegram_chats"
LINKS_COLLECTION = "telegram_link_tokens"
DELIVERIES_COLLECTION = "telegram_watch_deliveries"
LINK_TTL_MINUTES = 10
MUTE_HOURS = 24
DELIVERY_RETENTION_DAYS = 90
BOT_USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{5,64}$")
WEBHOOK_SECRET_RE = re.compile(r"^[A-Za-z0-9_-]{16,256}$")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _username() -> str:
    return str(os.environ.get("TELEGRAM_BOT_USERNAME") or "").strip().lstrip("@")


def webhook_secret() -> str:
    return str(os.environ.get("TELEGRAM_WEBHOOK_SECRET") or "").strip()


def is_configured() -> bool:
    return bool(
        telegram_notifier.bot_token()
        and BOT_USERNAME_RE.fullmatch(_username())
        and WEBHOOK_SECRET_RE.fullmatch(webhook_secret())
    )


def verify_webhook_secret(candidate: str) -> bool:
    expected = webhook_secret()
    return bool(expected and hmac.compare_digest(expected, str(candidate or "")))


def ensure_webhook_configured() -> dict:
    """Best-effort startup registration for the per-user bot webhook."""
    if not is_configured():
        return {"status": "skipped_not_configured"}
    result = telegram_notifier.call_bot_api("setWebhook", {
        "url": SITE_URL + "/api/telegram/webhook",
        "secret_token": webhook_secret(),
        "allowed_updates": ["message", "callback_query"],
        "drop_pending_updates": False,
    })
    if result.get("status") != "sent":
        logging.warning("Telegram Watch webhook registration did not succeed: %s", result.get("status"))
    return result


def cleanup_expired_metadata(*, now=None, db=None, max_items=500) -> int:
    """Bound stale link-token and delivery metadata growth."""
    db = db if db is not None else db_firestore
    now = now or utcnow()
    deleted = 0
    cutoffs = (
        (LINKS_COLLECTION, "expires_at", now),
        (DELIVERIES_COLLECTION, "created_at", now - timedelta(days=DELIVERY_RETENTION_DAYS)),
    )
    for collection_name, field, cutoff in cutoffs:
        query = db.collection(collection_name).where(
            filter=FieldFilter(field, "<=", cutoff)
        ).limit(max_items)
        for snap in query.stream():
            snap.reference.delete()
            deleted += 1
    return deleted


def run_startup_maintenance() -> dict:
    """Non-readiness-blocking maintenance invoked once per app start."""
    try:
        cleanup_expired_metadata()
    except Exception as exc:
        logging.error(
            "Telegram metadata cleanup failed on startup category=%s",
            safe_exception(exc),
        )
    return ensure_webhook_configured()


def _digest(value) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _serialize_connection(data: dict | None) -> dict:
    data = data or {}
    linked = bool(data.get("enabled") and data.get("chat_id"))
    configured = is_configured()
    return {
        "configured": configured,
        "linked": linked,
        "connected": bool(configured and linked),
        "telegram_username": str(data.get("telegram_username") or ""),
        "telegram_first_name": str(data.get("telegram_first_name") or ""),
        "linked_at": (
            data["linked_at"].isoformat()
            if isinstance(data.get("linked_at"), datetime) else ""
        ),
    }


def get_connection(uid: str, db=None) -> dict:
    db = db if db is not None else db_firestore
    snap = db.collection(CONNECTIONS_COLLECTION).document(uid).get()
    return _serialize_connection(snap.to_dict() if snap.exists else None)


def create_link(uid: str, *, now=None, db=None) -> dict:
    if not is_configured():
        raise watch_service.WatchError(
            "telegram_not_configured", "Telegram notifications are not configured yet."
        )
    db = db if db is not None else db_firestore
    now = now or utcnow()
    token = secrets.token_urlsafe(24)
    token_ref = db.collection(LINKS_COLLECTION).document(_digest(token))
    token_data = {
        "uid": uid,
        "created_at": now,
        "expires_at": now + timedelta(minutes=LINK_TTL_MINUTES),
    }

    def persist(transaction):
        persistence_guard.ensure_account_write_allowed(
            uid=uid, db=db, transaction=transaction
        )
        transaction.set(token_ref, token_data)

    watch_service._run_transaction(db, persist)
    return {
        "url": f"https://t.me/{quote(_username())}?start={quote(token)}",
        "expires_at": (now + timedelta(minutes=LINK_TTL_MINUTES)).isoformat(),
    }


def _connection_payload(uid: str, chat: dict, sender: dict, now: datetime) -> dict:
    return {
        "uid": uid,
        "chat_id": str(chat.get("id")),
        "telegram_user_id": str(sender.get("id")),
        "telegram_username": str(sender.get("username") or "")[:64],
        "telegram_first_name": str(sender.get("first_name") or "")[:80],
        "enabled": True,
        "linked_at": now,
        "blocked_at": None,
    }


def _consume_link_without_transaction(token_ref, connection_ref, chat_ref, payload, now):
    token_snap = token_ref.get()
    token_data = token_snap.to_dict() if token_snap.exists else None
    if not token_data:
        raise watch_service.WatchError("invalid_token", "This Telegram link is invalid.")
    if not isinstance(token_data.get("expires_at"), datetime) or token_data["expires_at"] < now:
        token_ref.delete()
        raise watch_service.WatchError("expired_token", "This Telegram link has expired.")
    chat_snap = chat_ref.get()
    chat_data = chat_snap.to_dict() if chat_snap.exists else None
    if chat_data and chat_data.get("uid") != payload["uid"]:
        raise watch_service.WatchError(
            "telegram_already_linked", "This Telegram account is already linked elsewhere."
        )
    old_snap = connection_ref.get()
    old_data = old_snap.to_dict() if old_snap.exists else None
    token_ref.delete()
    connection_ref.set(payload)
    chat_ref.set({"uid": payload["uid"], "chat_id": payload["chat_id"]})
    return old_data


def consume_link(token: str, chat: dict, sender: dict, *, now=None, db=None) -> dict:
    db = db if db is not None else db_firestore
    now = now or utcnow()
    if str(chat.get("type") or "") != "private" or not chat.get("id") or not sender.get("id"):
        raise watch_service.WatchError(
            "private_chat_required", "Connect consens.io from a private chat with the bot."
        )
    token_ref = db.collection(LINKS_COLLECTION).document(_digest(token))
    token_snap = token_ref.get()
    token_data = token_snap.to_dict() if token_snap.exists else None
    if not token_data:
        raise watch_service.WatchError("invalid_token", "This Telegram link is invalid.")
    uid = str(token_data.get("uid") or "")
    payload = _connection_payload(uid, chat, sender, now)
    connection_ref = db.collection(CONNECTIONS_COLLECTION).document(uid)
    chat_ref = db.collection(CHATS_COLLECTION).document(_digest(payload["chat_id"]))

    supports_transactions = callable(getattr(db, "run_transaction", None)) or hasattr(
        db, "transaction"
    )
    if not supports_transactions:
        # Compatibility seam for minimal unit-test stores. Production always
        # uses the transactional branch below.
        persistence_guard.ensure_account_write_allowed(uid=uid, db=db)
        old_data = _consume_link_without_transaction(
            token_ref, connection_ref, chat_ref, payload, now,
        )
    else:
        def consume(tx):
            persistence_guard.ensure_account_write_allowed(
                uid=uid, db=db, transaction=tx
            )
            link_snap = token_ref.get(transaction=tx)
            link_data = link_snap.to_dict() if link_snap.exists else None
            if not link_data:
                raise watch_service.WatchError("invalid_token", "This Telegram link is invalid.")
            if not isinstance(link_data.get("expires_at"), datetime) or link_data["expires_at"] < now:
                raise watch_service.WatchError("expired_token", "This Telegram link has expired.")
            current_chat = chat_ref.get(transaction=tx)
            current_chat_data = current_chat.to_dict() if current_chat.exists else None
            if current_chat_data and current_chat_data.get("uid") != uid:
                raise watch_service.WatchError(
                    "telegram_already_linked", "This Telegram account is already linked elsewhere."
                )
            old_connection = connection_ref.get(transaction=tx)
            old_data = old_connection.to_dict() if old_connection.exists else None
            tx.delete(token_ref)
            if old_data and str(old_data.get("chat_id") or "") != payload["chat_id"]:
                tx.delete(db.collection(CHATS_COLLECTION).document(_digest(old_data["chat_id"])))
            tx.set(connection_ref, payload)
            tx.set(chat_ref, {"uid": uid, "chat_id": payload["chat_id"]})
            return old_data

        old_data = watch_service._run_transaction(db, consume)
    if old_data and str(old_data.get("chat_id") or "") != payload["chat_id"] and not supports_transactions:
        db.collection(CHATS_COLLECTION).document(_digest(old_data["chat_id"])).delete()
    return _serialize_connection(payload)


def disconnect(uid: str, db=None) -> dict:
    db = db if db is not None else db_firestore
    ref = db.collection(CONNECTIONS_COLLECTION).document(uid)
    snap = ref.get()
    data = snap.to_dict() if snap.exists else None
    if data and data.get("chat_id"):
        db.collection(CHATS_COLLECTION).document(_digest(data["chat_id"])).delete()
    ref.delete()
    return _serialize_connection(None)


def delete_user_data(uid: str, db=None) -> None:
    """Idempotently remove connection, pending links and delivery metadata."""
    db = db if db is not None else db_firestore
    disconnect(uid, db=db)
    for collection_name in (LINKS_COLLECTION, DELIVERIES_COLLECTION):
        collection = db.collection(collection_name)
        try:
            query = collection.where(filter=FieldFilter("uid", "==", uid))
        except TypeError:
            query = collection.where("uid", "==", uid)
        for snap in query.stream():
            snap.reference.delete()


def _connection_by_chat(chat_id, db=None) -> dict | None:
    db = db if db is not None else db_firestore
    chat_id = str(chat_id or "")
    mapping = db.collection(CHATS_COLLECTION).document(_digest(chat_id)).get()
    mapping_data = mapping.to_dict() if mapping.exists else None
    if not mapping_data:
        return None
    connection = db.collection(CONNECTIONS_COLLECTION).document(mapping_data["uid"]).get()
    data = connection.to_dict() if connection.exists else None
    if not data or not data.get("enabled") or str(data.get("chat_id")) != chat_id:
        return None
    return data


def _watch_url(watch: dict) -> str:
    if watch.get("visibility") == "private":
        return SITE_URL + "/app/watches"
    return SITE_URL + share_snapshots.share_path(
        watch.get("share_slug") or "", watch["share_id"],
    )


def _escape(value, *, limit: int = 1_200) -> str:
    """Escape first, cut second — and never leave half an entity behind,
    which Telegram would reject as broken markup."""
    escaped = html.escape(" ".join(str(value or "").split()), quote=False)
    if len(escaped) <= limit:
        return escaped
    return re.sub(r"&[#\w]*$", "", escaped[:limit]).rstrip() + "…"


def _quote_block(text: str, *, limit: int = 1_200) -> str:
    """Telegram's own collapse: long text folds behind "expand" in the chat."""
    escaped = html.escape(str(text or "").strip(), quote=False)
    expandable = " expandable" if len(escaped) > 220 else ""
    if len(escaped) > limit:
        escaped = re.sub(r"&[#\w]*$", "", escaped[:limit]).rstrip() + "…"
    return f"<blockquote{expandable}>{escaped}</blockquote>"


def _facts_line(watch: dict, result: dict) -> str:
    """One line of numbers: agreement movement and how far models moved."""
    score = result.get("agreement_score")
    old_score = watch.get("last_agreement_score")
    parts = []
    if isinstance(score, (int, float)) and isinstance(old_score, (int, float)):
        delta = int(score) - int(old_score)
        movement = "unchanged" if not delta else f"{'+' if delta > 0 else '−'}{abs(delta)}"
        parts.append(f"Agreement {int(old_score)} → {int(score)} ({movement})")
    elif isinstance(score, (int, float)):
        parts.append(f"Agreement {int(score)}/100")
    position = result.get("opinion_map") or {}
    label = " ".join(str(position.get("shift_label") or "").split())
    shift = position.get("shift_score")
    if label:
        parts.append(
            f"Positions {label}"
            + (f" ({int(shift)}/100)" if isinstance(shift, (int, float)) else "")
        )
    return " · ".join(parts)


def _strip_markup(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", str(text or "")))


def _notification_text(kind: str, watch: dict, result: dict) -> str:
    """Headline, then what changed, then the numbers, then the question.

    Same order as the e-mails: the first two lines carry the news, the
    question sits at the bottom in a quote that Telegram folds when long.
    """
    question = str(watch.get("question") or "Consensus Watch")
    if kind == "condition":
        title = "🎯 Watch condition met"
        change_label = "Why it triggered"
        detail = str(result.get("condition_reason") or result.get("change_summary")
                     or "The condition is now met.")
    elif kind == "every_run":
        title = "🧭 Consensus Watch updated"
        change_label = "What changed"
        detail = str(
            result.get("change_summary")
            or ("This check found a material change." if result.get("changed")
                else "No material change since the last check.")
        )
    elif kind == "paused_error":
        title = "⚠️ Consensus Watch paused"
        change_label = "What happened"
        detail = "Three consecutive checks failed. Open the dashboard to review or resume this watch."
    else:
        title = "🔄 Consensus changed"
        change_label = "What changed"
        detail = str(result.get("change_summary") or "The consensus changed materially.")

    blocks = [f"<b>{title}</b>", f"<b>{change_label}</b>\n{_escape(detail)}"]
    # A failed run has no numbers worth quoting — they would be the old ones.
    facts = "" if kind == "paused_error" else _facts_line(watch, result)
    if facts:
        blocks.append(_escape(facts))
    if kind == "every_run" and str(result.get("consensus") or "").strip():
        blocks.append("<b>New consensus</b>\n" + _quote_block(result["consensus"]))
    blocks.append("<b>Question</b>\n" + _quote_block(question, limit=900))
    return "\n\n".join(blocks)


def _watch_keyboard(watch_id: str, watch: dict) -> dict:
    return {"inline_keyboard": [
        [{"text": "Open watch", "url": _watch_url(watch)}],
        [
            {"text": "Mute 24h", "callback_data": f"wm:{watch_id}"},
            {"text": "Pause", "callback_data": f"wp:{watch_id}"},
        ],
    ]}


def _claim_delivery(delivery_id: str, data: dict, db) -> bool:
    ref = db.collection(DELIVERIES_COLLECTION).document(delivery_id)

    def claim(transaction):
        persistence_guard.ensure_account_write_allowed(
            uid=str(data.get("uid") or ""), db=db, transaction=transaction
        )
        snap = ref.get(transaction=transaction)
        if snap.exists:
            return False
        transaction.set(ref, data)
        return True

    return bool(watch_service._run_transaction(db, claim))


def send_watch_notification(watch_id: str, run_id: str, kind: str, watch: dict,
                            result: dict, *, now=None, db=None) -> bool:
    db = db if db is not None else db_firestore
    now = now or utcnow()
    if not watch.get("telegram_enabled"):
        return False
    muted_until = watch.get("telegram_muted_until")
    if isinstance(muted_until, datetime) and muted_until > now:
        return False
    connection_ref = db.collection(CONNECTIONS_COLLECTION).document(watch["owner_uid"])
    connection_snap = connection_ref.get()
    connection = connection_snap.to_dict() if connection_snap.exists else None
    if not connection or not connection.get("enabled") or not connection.get("chat_id"):
        return False
    delivery_id = _digest(f"{watch_id}:{run_id}:{kind}")
    if not _claim_delivery(delivery_id, {
        "uid": watch["owner_uid"], "watch_id": watch_id, "run_id": run_id,
        "kind": kind, "status": "sending", "created_at": now,
    }, db):
        return False
    text = _notification_text(kind, watch, result)
    keyboard = _watch_keyboard(watch_id, watch)
    result_status = telegram_notifier.send_bot_message(
        connection["chat_id"], text, reply_markup=keyboard, parse_mode="HTML",
    )
    # A rejected entity must never cost the user the alert itself: resend the
    # same message as plain text if Telegram refuses the markup.
    if result_status.get("http_status") == 400 or result_status.get("error_code") == 400:
        logging.warning("Telegram Watch message rejected as HTML; retrying as plain text")
        result_status = telegram_notifier.send_bot_message(
            connection["chat_id"], _strip_markup(text), reply_markup=keyboard,
        )
    updates = {
        "status": result_status.get("status") or "failed",
        "attempted_at": now,
    }
    if result_status.get("status") == "sent":
        updates["sent_at"] = utcnow()
    db.collection(DELIVERIES_COLLECTION).document(delivery_id).update(updates)
    if result_status.get("http_status") == 403 or result_status.get("error_code") == 403:
        connection_ref.update({"enabled": False, "blocked_at": utcnow()})
    return result_status.get("status") == "sent"


def send_test(uid: str, db=None) -> dict:
    db = db if db is not None else db_firestore
    snap = db.collection(CONNECTIONS_COLLECTION).document(uid).get()
    data = snap.to_dict() if snap.exists else None
    if not data or not data.get("enabled") or not data.get("chat_id"):
        raise watch_service.WatchError("telegram_not_linked", "Connect Telegram first.")
    result = telegram_notifier.send_bot_message(
        data["chat_id"],
        "✅ consens.io is connected\n\nFuture Consensus Watch alerts will arrive here.",
        reply_markup={"inline_keyboard": [[{"text": "Open Watches", "url": SITE_URL + "/app/watches"}]]},
    )
    if result.get("status") != "sent":
        raise watch_service.WatchError("telegram_delivery_failed", "Telegram test message could not be delivered.")
    return {"status": "sent"}


def mute_watch(uid: str, watch_id: str, *, now=None, db=None) -> datetime:
    db = db if db is not None else db_firestore
    now = now or utcnow()
    ref, _data = watch_service._owned_watch(uid, watch_id, db)  # owner check is authoritative
    until = now + timedelta(hours=MUTE_HOURS)
    ref.update({"telegram_muted_until": until})
    return until


def _answer_callback(callback_id: str, text: str) -> None:
    telegram_notifier.call_bot_api("answerCallbackQuery", {
        "callback_query_id": callback_id, "text": text[:200],
    })


def _handle_callback(callback: dict, db) -> None:
    callback_id = str(callback.get("id") or "")
    data = str(callback.get("data") or "")
    sender = callback.get("from") or {}
    message = callback.get("message") or {}
    chat = message.get("chat") or {}
    connection = _connection_by_chat(chat.get("id"), db=db)
    if not connection or str(connection.get("telegram_user_id")) != str(sender.get("id")):
        _answer_callback(callback_id, "This Telegram account is not connected to consens.io.")
        return
    uid = connection["uid"]
    action, separator, watch_id = data.partition(":")
    if not separator or not watch_id:
        _answer_callback(callback_id, "Unknown action.")
        return
    try:
        if action == "wm":
            mute_watch(uid, watch_id, db=db)
            _answer_callback(callback_id, "Telegram alerts muted for 24 hours.")
        elif action == "wp":
            watch_service._owned_watch(uid, watch_id, db)
            _answer_callback(callback_id, "Confirm pause below.")
            telegram_notifier.send_bot_message(chat["id"], "Pause this Consensus Watch?", reply_markup={
                "inline_keyboard": [[
                    {"text": "Confirm pause", "callback_data": f"wpc:{watch_id}"},
                    {"text": "Cancel", "callback_data": f"wx:{watch_id}"},
                ]]
            })
        elif action == "wpc":
            watch_service.pause_watch(uid, watch_id, db=db)
            _answer_callback(callback_id, "Watch paused.")
        elif action == "wx":
            _answer_callback(callback_id, "Pause canceled.")
        else:
            _answer_callback(callback_id, "Unknown action.")
    except watch_service.WatchError as exc:
        _answer_callback(callback_id, exc.message)


def handle_update(update: dict, db=None) -> None:
    db = db if db is not None else db_firestore
    callback = update.get("callback_query")
    if isinstance(callback, dict):
        _handle_callback(callback, db)
        return
    message = update.get("message") or {}
    text = str(message.get("text") or "").strip()
    if not text.startswith("/start"):
        return
    parts = text.split(maxsplit=1)
    if len(parts) != 2 or not parts[1].strip():
        telegram_notifier.send_bot_message(
            (message.get("chat") or {}).get("id"),
            "Open consens.io Settings or the Watches dashboard to connect this bot.",
        )
        return
    try:
        connection = consume_link(
            parts[1].strip(), message.get("chat") or {}, message.get("from") or {}, db=db,
        )
    except watch_service.WatchError as exc:
        telegram_notifier.send_bot_message((message.get("chat") or {}).get("id"), f"Unable to connect: {exc.message}")
        return
    name = connection.get("telegram_first_name") or "there"
    telegram_notifier.send_bot_message(
        (message.get("chat") or {}).get("id"),
        f"✅ Telegram connected\n\nHi {name}. You can now enable Telegram on any Consensus Watch.",
        reply_markup={"inline_keyboard": [[{"text": "Open Watches", "url": SITE_URL + "/app/watches"}]]},
    )
