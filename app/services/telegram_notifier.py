"""Best-effort Telegram notifications for server-side maintenance workflows."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_ADMIN_URL = "https://www.consens.io/admin#seo"


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
    bot_token = str(os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
    chat_id = str(os.environ.get("TELEGRAM_CHAT_ID") or "").strip()
    attempted_at = datetime.now(timezone.utc).isoformat()
    if not bot_token or not chat_id:
        return {
            "status": "skipped_not_configured",
            "attempted_at": attempted_at,
        }

    payload = json.dumps({
        "chat_id": chat_id,
        "text": _review_message(review),
        "disable_web_page_preview": True,
    }).encode("utf-8")
    request = Request(
        f"https://api.telegram.org/bot{bot_token}/sendMessage",
        data=payload,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            response.read()
    except HTTPError as exc:
        logging.warning("SEO review Telegram notification failed with HTTP %s", exc.code)
        return {
            "status": "failed_http",
            "http_status": int(exc.code),
            "attempted_at": attempted_at,
        }
    except (URLError, TimeoutError, OSError):
        logging.warning("SEO review Telegram notification failed with a network error")
        return {
            "status": "failed_network",
            "attempted_at": attempted_at,
        }
    return {
        "status": "sent",
        "attempted_at": attempted_at,
        "sent_at": datetime.now(timezone.utc).isoformat(),
    }
