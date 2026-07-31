"""Best-effort Telegram notifications for server-side maintenance workflows."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_ADMIN_URL = "https://www.consens.io/admin#seo"
_CRITICAL_DEDUPE_SECONDS = 10 * 60
_CRITICAL_RATE_WINDOW_SECONDS = 10 * 60
_CRITICAL_RATE_LIMIT = 10
_critical_lock = threading.Lock()
_critical_seen: dict[str, float] = {}
_critical_sent_at: deque[float] = deque()

_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+"),
    re.compile(
        r"(?i)([?&](?:access_token|api_key|id_token|key|password|secret|token)=)[^&#\s]+"
    ),
    re.compile(
        r"(?i)((?:api[_-]?key|id[_-]?token|password|secret|token)\s*[:=]\s*)[^\s,;]+"
    ),
    re.compile(r"\b(?:sk[-_][A-Za-z0-9_-]{12,}|cns_live_[A-Za-z0-9_-]{12,})\b"),
    re.compile(r"\bAIza[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\b\d{6,}:[A-Za-z0-9_-]{20,}\b"),
)


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


def _scrub_alert_text(value, *, limit: int) -> str:
    text = str(value or "").replace("\x00", "").strip()
    for pattern in _SECRET_PATTERNS:
        if pattern.groups:
            text = pattern.sub(r"\1[redacted]", text)
        else:
            text = pattern.sub("[redacted]", text)
    return text[:limit]


def _critical_chat_id() -> str:
    return str(
        os.environ.get("CRITICAL_ERROR_TELEGRAM_CHAT_ID")
        or os.environ.get("TELEGRAM_CHAT_ID")
        or ""
    ).strip()


def _reserve_critical_delivery(fingerprint: str) -> str:
    """Bound alert storms per process and collapse identical failures."""
    now = time.monotonic()
    cutoff = now - _CRITICAL_RATE_WINDOW_SECONDS
    with _critical_lock:
        while _critical_sent_at and _critical_sent_at[0] <= cutoff:
            _critical_sent_at.popleft()
        for key, seen_at in list(_critical_seen.items()):
            if seen_at <= now - _CRITICAL_DEDUPE_SECONDS:
                _critical_seen.pop(key, None)
        if fingerprint in _critical_seen:
            return "deduplicated"
        if len(_critical_sent_at) >= _CRITICAL_RATE_LIMIT:
            return "rate_limited"
        _critical_seen[fingerprint] = now
        _critical_sent_at.append(now)
    return "reserved"


def _critical_error_message(report: Mapping) -> str:
    source = _scrub_alert_text(report.get("source") or "server", limit=40)
    error_type = _scrub_alert_text(report.get("type") or "unexpected_error", limit=80)
    phase = _scrub_alert_text(report.get("phase"), limit=80)
    message = _scrub_alert_text(report.get("message") or "No message", limit=700)
    path = _scrub_alert_text(report.get("path"), limit=300)
    details = _scrub_alert_text(report.get("details"), limit=1_200)
    stack = _scrub_alert_text(report.get("stack"), limit=1_800)
    environment = _scrub_alert_text(
        os.environ.get("RENDER_SERVICE_NAME") or os.environ.get("ENVIRONMENT") or "production",
        limit=80,
    )
    lines = [
        "🚨 consens.io critical error",
        f"Source: {source}",
        f"Type: {error_type}",
        f"Environment: {environment}",
        f"Time: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
    ]
    if phase:
        lines.append(f"Phase: {phase}")
    if path:
        lines.append(f"Path: {path}")
    lines.extend(("", message))
    if details:
        lines.extend(("", f"Details: {details}"))
    if stack:
        lines.extend(("", f"Stack: {stack}"))
    return "\n".join(lines)[:4096]


def send_critical_error_notification(report: Mapping) -> dict:
    """Send a redacted, deduplicated operational alert without raising."""
    attempted_at = datetime.now(timezone.utc).isoformat()
    chat_id = _critical_chat_id()
    if not bot_token() or not chat_id:
        return {"status": "skipped_not_configured", "attempted_at": attempted_at}

    source = _scrub_alert_text(report.get("source") or "server", limit=40)
    error_type = _scrub_alert_text(report.get("type") or "unexpected_error", limit=80)
    phase = _scrub_alert_text(report.get("phase"), limit=80)
    message = _scrub_alert_text(report.get("message") or "No message", limit=500)
    path = _scrub_alert_text(report.get("path"), limit=200)
    fingerprint = "\x1f".join((source, error_type, phase, path, message))
    reservation = _reserve_critical_delivery(fingerprint)
    if reservation != "reserved":
        return {"status": reservation, "attempted_at": attempted_at}

    try:
        result = send_bot_message(chat_id, _critical_error_message(report))
    except Exception:
        # This function is called from exception paths and must never create a
        # second failure or log secrets from a Telegram request URL.
        logging.warning("Critical Telegram notification failed safely")
        return {"status": "failed_safely", "attempted_at": attempted_at}
    result.pop("result", None)
    return result


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
