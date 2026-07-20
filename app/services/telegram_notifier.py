"""Best-effort Telegram notifications for server-side maintenance workflows."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_ADMIN_URL = "https://www.consens.io/admin#seo"


def bot_token() -> str:
    return str(os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()


def call_bot_api(method: str, payload: dict, *, timeout: int = 30) -> dict:
    """Call one Telegram Bot API method without leaking credentials.

    The structured result lets user-facing notification flows distinguish a
    blocked bot (HTTP 403) from temporary network failures. Maintenance
    notifications keep their existing best-effort semantics.
    """
    token = bot_token()
    attempted_at = datetime.now(timezone.utc).isoformat()
    if not token:
        return {"status": "skipped_not_configured", "attempted_at": attempted_at}
    request = Request(
        f"https://api.telegram.org/bot{token}/{method}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
        decoded = json.loads(raw or b"{}")
    except HTTPError as exc:
        logging.warning("Telegram Bot API %s failed with HTTP %s", method, exc.code)
        return {
            "status": "failed_http", "http_status": int(exc.code),
            "attempted_at": attempted_at,
        }
    except (URLError, TimeoutError, OSError, ValueError):
        logging.warning("Telegram Bot API %s failed with a network/response error", method)
        return {"status": "failed_network", "attempted_at": attempted_at}
    if decoded.get("ok") is False:
        return {
            "status": "failed_api",
            "error_code": decoded.get("error_code"),
            "retry_after": ((decoded.get("parameters") or {}).get("retry_after")),
            "attempted_at": attempted_at,
        }
    return {
        "status": "sent", "attempted_at": attempted_at,
        "sent_at": datetime.now(timezone.utc).isoformat(),
        "result": decoded.get("result"),
    }


def send_bot_message(chat_id, text: str, *, reply_markup: dict | None = None) -> dict:
    payload = {
        "chat_id": str(chat_id),
        "text": str(text or "")[:4096],
        "disable_web_page_preview": True,
    }
    if reply_markup:
        payload["reply_markup"] = reply_markup
    return call_bot_api("sendMessage", payload)


def _group_count(review: dict, name: str) -> int:
    return len(((review.get("groups") or {}).get(name) or []))


def _review_message(review: dict) -> str:
    status = str(review.get("status") or "unknown")
    pages = list(review.get("pages") or [])
    decisions = review.get("editorial_decisions") or {}
    editorial_total = _group_count(review, "manual_improvement")
    editorial_open = max(0, editorial_total - len(decisions))
    prompt_pending = bool(
        review.get("proposed_topic_brief")
        and review.get("topic_brief_decision", "pending") == "pending"
    )
    summary = str(review.get("summary") or "No summary available.").strip()
    admin_url = str(os.environ.get("SEO_ADMIN_URL") or DEFAULT_ADMIN_URL).strip()
    lines = [
        f"SEO review {status}",
        "",
        summary[:1_200],
        "",
        f"Pages reviewed: {len(pages)}",
        f"Editorial decisions open: {editorial_open}",
        f"Publisher prompt decision: {'required' if prompt_pending else 'none'}",
        "",
        admin_url,
    ]
    return "\n".join(lines)


def send_seo_review_notification(review: dict) -> dict:
    """Notify after every terminal SEO review without failing the review itself."""
    configured_token = bot_token()
    chat_id = str(os.environ.get("TELEGRAM_CHAT_ID") or "").strip()
    attempted_at = datetime.now(timezone.utc).isoformat()
    if not configured_token or not chat_id:
        return {
            "status": "skipped_not_configured",
            "attempted_at": attempted_at,
        }

    payload = {
        "chat_id": chat_id,
        "text": _review_message(review),
        "disable_web_page_preview": True,
    }
    result = call_bot_api("sendMessage", payload)
    result.pop("result", None)
    return result
