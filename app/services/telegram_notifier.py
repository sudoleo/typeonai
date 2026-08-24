"""Best-effort Telegram notifications for server-side maintenance workflows."""

from __future__ import annotations

import hashlib
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

from app.core.observability import safe_exception


DEFAULT_ADMIN_URL = "https://www.consens.io/admin#seo"
_CRITICAL_DEDUPE_SECONDS = 10 * 60
_CRITICAL_RATE_WINDOW_SECONDS = 10 * 60
_CRITICAL_RATE_LIMIT = 10
_critical_lock = threading.Lock()
_critical_seen: dict[str, float] = {}
_critical_sent_at: deque[float] = deque()
_REGISTRATION_DEDUPE_SECONDS = 24 * 60 * 60
_registration_lock = threading.Lock()
_registration_seen: dict[str, float] = {}

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
        logging.warning(
            "Telegram Bot API %s failed category=%s",
            method,
            safe_exception(exc),
        )
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


def send_bot_message(chat_id, text: str, *, reply_markup: dict | None = None,
                     parse_mode: str = "") -> dict:
    payload = {
        "chat_id": str(chat_id),
        "text": str(text or "")[:4096],
        "disable_web_page_preview": True,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode
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


def _new_user_registration_message(registration_method: str) -> str:
    method = _scrub_alert_text(registration_method or "unknown", limit=60)
    environment_name = (
        os.environ.get("RENDER_SERVICE_NAME")
        or os.environ.get("ENVIRONMENT")
        or "production"
    )
    environment = _scrub_alert_text(environment_name, limit=80)
    return "\n".join((
        "👤 consens.io new user registered",
        f"Method: {method}",
        f"Environment: {environment}",
        f"Time: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
    ))


def _reserve_registration_delivery(dedupe_key: str) -> tuple[str, str]:
    if not dedupe_key:
        return "reserved", ""
    fingerprint = hashlib.sha256(str(dedupe_key).encode("utf-8")).hexdigest()
    now = time.monotonic()
    with _registration_lock:
        for key, seen_at in list(_registration_seen.items()):
            if seen_at <= now - _REGISTRATION_DEDUPE_SECONDS:
                _registration_seen.pop(key, None)
        if fingerprint in _registration_seen:
            return "deduplicated", fingerprint
        _registration_seen[fingerprint] = now
    return "reserved", fingerprint


def _release_registration_delivery(fingerprint: str) -> None:
    if not fingerprint:
        return
    with _registration_lock:
        _registration_seen.pop(fingerprint, None)


def send_new_user_registration_notification(
    registration_method: str,
    dedupe_key: str = "",
) -> dict:
    """Send a PII-free registration alert without affecting sign-up."""
    attempted_at = datetime.now(timezone.utc).isoformat()
    chat_id = _critical_chat_id()
    if not bot_token() or not chat_id:
        return {"status": "skipped_not_configured", "attempted_at": attempted_at}

    reservation, fingerprint = _reserve_registration_delivery(dedupe_key)
    if reservation != "reserved":
        return {"status": reservation, "attempted_at": attempted_at}

    try:
        result = send_bot_message(
            chat_id,
            _new_user_registration_message(registration_method),
        )
    except Exception:
        _release_registration_delivery(fingerprint)
        logging.warning("New-user Telegram notification failed safely")
        return {"status": "failed_safely", "attempted_at": attempted_at}
    if result.get("status") != "sent":
        _release_registration_delivery(fingerprint)
    result.pop("result", None)
    return result


def _group_count(review: dict, name: str) -> int:
    return len(((review.get("groups") or {}).get(name) or []))


def _review_findings(review: dict) -> list[str]:
    findings = review.get("findings") or {}
    lines = []
    for marker, key in (("+", "positive"), ("-", "negative")):
        for item in (findings.get(key) or [])[:3]:
            text = str(item or "").strip()
            if text:
                lines.append(f"{marker} {text[:300]}")
    return lines


def _review_delta_line(review: dict) -> str:
    delta = review.get("delta") or {}
    if not delta.get("comparable"):
        return "Change since last review: no comparable previous run"
    changed = delta.get("changed") or []
    new_pages = delta.get("new_pages") or []
    if not changed and not new_pages:
        return "Change since last review: none"
    parts = [
        f"{str(item.get('title') or item.get('page_id') or '')[:60]}: "
        f"{item.get('from')} -> {item.get('to')}"
        for item in changed[:3]
    ]
    if len(changed) > 3:
        parts.append(f"+{len(changed) - 3} more")
    if new_pages:
        parts.append(f"{len(new_pages)} new")
    return "Change since last review: " + "; ".join(parts)


def _review_judge_line(review: dict) -> str:
    judge_error = str(review.get("judge_error") or "").strip()
    if review.get("judge_called"):
        return "Portfolio judge: answered"
    if judge_error:
        return f"Portfolio judge: FAILED - {judge_error[:200]}"
    return "Portfolio judge: not called"


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
    counts = review.get("status_counts") or {}
    status_line = ", ".join(f"{name} {count}" for name, count in counts.items())
    lines = [
        f"SEO review {status}",
        "",
        summary[:1_200],
    ]
    findings = _review_findings(review)
    if findings:
        lines.extend(["", *findings])
    lines.extend([
        "",
        _review_delta_line(review),
        f"Pages reviewed: {len(pages)}" + (f" ({status_line})" if status_line else ""),
        _review_judge_line(review),
        f"Editorial decisions open: {editorial_open}",
        f"Publisher prompt decision: {'required' if prompt_pending else 'none'}",
        "",
        admin_url,
    ])
    return "\n".join(lines)


def send_seo_review_notification(review: dict) -> dict:
    """Notify after every terminal SEO review without failing the review itself."""
    configured_token = bot_token()
    # Accept the same chat id the critical alerts use. A deployment that only
    # set CRITICAL_ERROR_TELEGRAM_CHAT_ID used to drop every review silently.
    chat_id = _critical_chat_id()
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
