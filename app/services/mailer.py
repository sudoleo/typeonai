"""Small async SMTP service for Consensus Watch notifications."""

from __future__ import annotations

import asyncio
import html
import logging
import os
import re
import smtplib
import ssl
from email.message import EmailMessage
from datetime import datetime, timezone

from app.core.observability import safe_exception
from app.services.llm.mock_llm import mock_llm_enabled


def _smtp_config():
    values = {
        "host": os.environ.get("SMTP_HOST", "").strip(),
        "user": os.environ.get("SMTP_USER", "").strip(),
        "password": os.environ.get("SMTP_PASSWORD", "").strip(),
        "sender": os.environ.get("MAIL_FROM", "").strip(),
    }
    try:
        values["port"] = int(os.environ.get("SMTP_PORT", "587"))
    except ValueError:
        values["port"] = 587
    return values


def is_configured() -> bool:
    config = _smtp_config()
    return bool(config["host"] and config["sender"])


def _deliver(message: EmailMessage) -> bool:
    config = _smtp_config()
    # Last line of defence: a mock instance often inherits the real SMTP
    # credentials from .env, so never let fixture content reach a recipient.
    if mock_llm_enabled():
        logging.info("Mail skipped: MOCK_LLM=1")
        return False
    if not is_configured():
        logging.info("Consensus Watch mail skipped: SMTP_HOST/MAIL_FROM not configured")
        return False
    try:
        if config["port"] == 465:
            with smtplib.SMTP_SSL(config["host"], config["port"], context=ssl.create_default_context(), timeout=30) as smtp:
                if config["user"]:
                    smtp.login(config["user"], config["password"])
                smtp.send_message(message)
        else:
            with smtplib.SMTP(config["host"], config["port"], timeout=30) as smtp:
                smtp.ehlo()
                smtp.starttls(context=ssl.create_default_context())
                smtp.ehlo()
                if config["user"]:
                    smtp.login(config["user"], config["password"])
                smtp.send_message(message)
        return True
    except Exception as exc:
        logging.error(
            "Consensus Watch mail delivery failed category=%s",
            safe_exception(exc),
        )
        return False


async def send_message(message: EmailMessage) -> bool:
    return await asyncio.to_thread(_deliver, message)


def _base_message(recipient: str, subject: str, plain: str, html_body: str) -> EmailMessage:
    message = EmailMessage()
    message["From"] = _smtp_config()["sender"] or "consens.io"
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(plain)
    message.add_alternative(html_body, subtype="html")
    return message


# ---------------------------------------------------------------------------
# Shared building blocks
#
# Every notification answers the same questions in the same order: what
# changed, by how much, and what was asked. The helpers below keep that
# structure identical across all mails (and mirror what telegram_watch.py
# sends), so the first three lines are enough to understand the message.
# Long questions are clipped like the pages clamp them — the full text is
# always one click away.
# ---------------------------------------------------------------------------

INK = "#172033"
INK_SOFT = "#3d4759"
MUTED = "#667085"
PANEL = "#f3f6fb"
BORDER = "#d8deea"
ACCENT = "#335cff"
QUESTION_PREVIEW_CHARS = 200

_EYEBROW_STYLE = (
    f"margin:0;font-size:12px;font-weight:700;letter-spacing:.08em;"
    f"text-transform:uppercase;color:{MUTED}"
)
_FONT_STACK = "-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif"


# Typographic characters would force the whole plain-text part into base64,
# where URLs stop being clickable in text-only clients. The plain half stays
# ASCII wherever we control the wording.
_PLAIN_ASCII = str.maketrans({
    "→": "->", "←": "<-", "−": "-", "–": "-", "—": "-", "…": "...", "·": "-",
    "↑": "up", "↓": "down",
})


def _normalized(value) -> str:
    return " ".join(str(value or "").split())


def _ascii(text: str) -> str:
    return str(text or "").translate(_PLAIN_ASCII)


def _question_view(question, *, limit: int = QUESTION_PREVIEW_CHARS) -> tuple[str, bool]:
    """Collapse a long question to a readable opening; True = it was cut."""
    text = _normalized(question)
    if len(text) <= limit:
        return text, False
    cut = text[:limit]
    space = cut.rfind(" ")
    if space > limit * 0.6:
        cut = cut[:space]
    return cut.rstrip(" ,;:.–-") + "…", True


def _split_lead(summary) -> tuple[str, str]:
    """First sentence carries the change; the rest is detail."""
    text = _normalized(summary)
    if not text:
        return "", ""
    match = re.search(r"(?<=[.!?])\s+", text)
    if not match or match.start() > 180:
        return text, ""
    return text[:match.start()], text[match.end():]


def _score_fact(old_score, new_score) -> dict | None:
    old_known = isinstance(old_score, (int, float))
    new_known = isinstance(new_score, (int, float))
    if not new_known:
        return None
    if not old_known:
        return {"label": "Agreement", "value": f"{int(new_score)}/100", "note": "First measurement"}
    delta = int(new_score) - int(old_score)
    note = "unchanged" if not delta else f"{'+' if delta > 0 else '−'}{abs(delta)} points"
    return {
        "label": "Agreement",
        "value": f"{int(old_score)} → {int(new_score)}",
        "note": note,
    }


def _direction_fact(direction) -> dict | None:
    """Model movement from the opinion map (Stable / Evolving / Turning)."""
    if not isinstance(direction, dict):
        return None
    label = _normalized(direction.get("shift_label") or direction.get("label"))
    score = direction.get("shift_score")
    if not label:
        return None
    return {
        "label": "Model positions",
        "value": label,
        "note": f"{int(score)}/100 movement" if isinstance(score, (int, float)) else "",
    }


def _severity_fact(changed: bool, severity) -> dict:
    if not changed:
        return {"label": "Content", "value": "No material change", "note": ""}
    level = _normalized(severity).lower()
    return {
        "label": "Content",
        "value": "Major change" if level == "major" else "Minor change",
        "note": "",
    }


def _facts_html(facts: list) -> str:
    """One quiet strip of two or three numbers, side by side."""
    cells = [fact for fact in facts if fact]
    if not cells:
        return ""
    width = 100 // len(cells)
    rendered = "".join(
        f'<td style="padding:12px 14px;vertical-align:top;width:{width}%">'
        f'<div style="font-size:11px;font-weight:700;letter-spacing:.07em;'
        f'text-transform:uppercase;color:{MUTED}">{html.escape(fact["label"])}</div>'
        f'<div style="margin-top:3px;font-size:18px;font-weight:700;color:{INK}">'
        f'{html.escape(str(fact["value"]))}</div>'
        + (
            f'<div style="margin-top:1px;font-size:12px;color:{MUTED}">'
            f'{html.escape(str(fact["note"]))}</div>' if fact.get("note") else ""
        )
        + "</td>"
        for fact in cells
    )
    return (
        f'<div style="margin:18px 0;background:{PANEL};border-radius:12px">'
        '<table role="presentation" cellpadding="0" cellspacing="0" border="0" '
        f'style="width:100%;border-collapse:collapse"><tr>{rendered}</tr></table></div>'
    )


def _facts_plain(facts: list) -> str:
    lines = []
    for fact in facts:
        if not fact:
            continue
        note = f" ({fact['note']})" if fact.get("note") else ""
        lines.append(_ascii(f"{fact['label']}: {fact['value']}{note}"))
    return "\n".join(lines)


def _question_html(question, url: str, *, label: str = "Question") -> str:
    text, truncated = _question_view(question)
    more = (
        f'<p style="margin:6px 0 0;font-size:13px"><a href="{html.escape(url)}" '
        f'style="color:{MUTED}">Read the full question</a></p>'
        if truncated and url else ""
    )
    return (
        f'<p style="{_EYEBROW_STYLE}">{html.escape(label)}</p>'
        f'<p style="margin:5px 0 0;font-size:17px;font-weight:600;line-height:1.4;color:{INK}">'
        f"{html.escape(text)}</p>{more}"
    )


def _question_plain(question, *, label: str = "QUESTION") -> str:
    text, _truncated = _question_view(question)
    return _ascii(f"{label}\n{text}")


def _change_block_html(label: str, summary, *, notable: bool = True) -> str:
    """The heart of the mail: what actually changed, not buried in prose."""
    lead, rest = _split_lead(summary)
    if not lead:
        return ""
    rule = ACCENT if notable else BORDER
    detail = (
        f'<p style="margin:8px 0 0;font-size:15px;color:{INK_SOFT}">{html.escape(rest)}</p>'
        if rest else ""
    )
    return (
        f'<div style="margin:18px 0;padding:2px 0 2px 14px;border-left:3px solid {rule}">'
        f'<p style="{_EYEBROW_STYLE}">{html.escape(label)}</p>'
        f'<p style="margin:6px 0 0;font-size:17px;font-weight:600;line-height:1.45;color:{INK}">'
        f"{html.escape(lead)}</p>{detail}</div>"
    )


def _change_block_plain(label: str, summary) -> str:
    lead, rest = _split_lead(summary)
    if not lead:
        return ""
    return _ascii(f"{label.upper()}\n{lead}" + (f"\n{rest}" if rest else ""))


def _button_html(url: str, label: str) -> str:
    return (
        f'<p style="margin:22px 0 0"><a href="{html.escape(url)}" '
        f'style="display:inline-block;background:{ACCENT};color:#fff;text-decoration:none;'
        f'padding:12px 18px;border-radius:8px;font-weight:600">{html.escape(label)}</a></p>'
    )


def _shell_html(*, eyebrow: str, heading: str, preheader: str, body: str,
                footer: str, width: int = 620) -> str:
    return (
        f'<!doctype html><html><body style="margin:0;background:#ffffff;'
        f'font-family:{_FONT_STACK};color:{INK};line-height:1.55">'
        f'<div style="display:none;max-height:0;overflow:hidden;opacity:0;color:transparent">'
        f"{html.escape(preheader)}</div>"
        f'<div style="max-width:{width}px;margin:auto;padding:24px">'
        f'<p style="{_EYEBROW_STYLE}">{html.escape(eyebrow)}</p>'
        f'<h1 style="margin:4px 0 0;font-size:23px;line-height:1.3;color:{INK}">'
        f"{html.escape(heading)}</h1>"
        f"{body}"
        f'<p style="font-size:12px;color:{MUTED};margin-top:32px">{footer}</p>'
        "</div></body></html>"
    )


def build_change_message(*, recipient: str, question: str, old_score, new_score,
                         summary: str, share_url: str, unsubscribe_url: str,
                         severity: str = "major", direction=None) -> EmailMessage:
    clipped_question = _normalized(question)
    subject_question = clipped_question[:72] + ("…" if len(clipped_question) > 72 else "")
    subject = f"Consensus changed: {subject_question}"
    lead, _rest = _split_lead(summary)
    facts = [
        _score_fact(old_score, new_score),
        _direction_fact(direction),
        _severity_fact(True, severity),
    ]
    plain = (
        "Consensus Watch detected a material change.\n\n"
        + _change_block_plain("What changed", summary) + "\n\n"
        + _facts_plain(facts) + "\n\n"
        + _question_plain(question) + "\n\n"
        f"View history: {share_url}\nUnsubscribe: {unsubscribe_url}\n"
    )
    body = (
        _change_block_html("What changed", summary)
        + _facts_html(facts)
        + _question_html(question, share_url)
        + _button_html(share_url, "See what changed")
    )
    html_body = _shell_html(
        eyebrow="Consensus Watch · Material change",
        heading="The consensus changed",
        preheader=lead or "A watched question moved.",
        body=body,
        footer=(
            "You received this service message because you enabled Consensus Watch. "
            f'<a href="{html.escape(unsubscribe_url)}">Pause this watch</a>.'
        ),
    )
    return _base_message(recipient, subject, plain, html_body)


def build_run_message(*, recipient: str, question: str, agreement_score,
                      consensus: str, changed: bool, severity: str,
                      summary: str, share_url: str, unsubscribe_url: str,
                      old_score=None, direction=None) -> EmailMessage:
    """Full-content notification for users who opted into every successful run."""
    clipped_question = _normalized(question)
    subject_question = clipped_question[:72] + ("…" if len(clipped_question) > 72 else "")
    subject = f"New consensus: {subject_question}"
    consensus_text = str(consensus or "").strip()
    change_text = summary if changed and summary else (
        "This check found a material change." if changed
        else "No material content change since the last check."
    )
    facts = [
        _score_fact(old_score, agreement_score),
        _direction_fact(direction),
        _severity_fact(changed, severity),
    ]
    plain = (
        "Consensus Watch completed a new check.\n\n"
        + _change_block_plain("What changed", change_text) + "\n\n"
        + _facts_plain(facts) + "\n\n"
        + _question_plain(question) + "\n\n"
        f"NEW CONSENSUS\n\n{consensus_text}\n\n"
        f"View history: {share_url}\nUnsubscribe: {unsubscribe_url}\n"
    )
    safe_consensus = html.escape(consensus_text).replace("\n", "<br>")
    body = (
        _change_block_html("What changed", change_text, notable=bool(changed))
        + _facts_html(facts)
        + _question_html(question, share_url)
        + f'<div style="margin-top:20px;padding:18px;border:1px solid {BORDER};border-radius:12px">'
        f'<p style="{_EYEBROW_STYLE}">New consensus</p>'
        f'<div style="margin-top:8px;color:{INK_SOFT}">{safe_consensus}</div></div>'
        + _button_html(share_url, "View watch page and history")
    )
    html_body = _shell_html(
        eyebrow="Consensus Watch · Completed check",
        heading="A new consensus is in",
        preheader=_split_lead(change_text)[0],
        body=body,
        footer=(
            "You chose to receive every new consensus result. "
            f'<a href="{html.escape(unsubscribe_url)}">Pause this watch</a>.'
        ),
        width=680,
    )
    return _base_message(recipient, subject, plain, html_body)


def build_condition_message(*, recipient: str, question: str, condition: str,
                            reason: str, agreement_score, consensus: str,
                            share_url: str, unsubscribe_url: str,
                            old_score=None, direction=None) -> EmailMessage:
    """Notification emitted once when a user-defined condition becomes true."""
    clipped_condition = _normalized(condition)[:500]
    clipped_reason = _normalized(reason)[:400]
    subject_condition = clipped_condition[:72] + ("…" if len(clipped_condition) > 72 else "")
    subject = f"Watch condition met: {subject_condition}"
    consensus_text = str(consensus or "").strip()
    facts = [
        _score_fact(old_score, agreement_score),
        _direction_fact(direction),
    ]
    plain = (
        "Your Consensus Watch condition is now met.\n\n"
        + _change_block_plain("Why it triggered", clipped_reason) + "\n\n"
        f"CONDITION\n{clipped_condition}\n\n"
        + _facts_plain(facts) + "\n\n"
        + _question_plain(question) + "\n\n"
        f"NEW CONSENSUS\n\n{consensus_text}\n\n"
        f"Open watch page: {share_url}\nPause this watch: {unsubscribe_url}\n"
    )
    safe_consensus = html.escape(consensus_text).replace("\n", "<br>")
    body = (
        _change_block_html("Why it triggered", clipped_reason)
        + f'<div style="margin:18px 0;padding:14px 16px;background:#eef8f1;border-radius:12px">'
        f'<p style="{_EYEBROW_STYLE}">Your condition</p>'
        f'<p style="margin:5px 0 0;color:{INK}">{html.escape(clipped_condition)}</p></div>'
        + _facts_html(facts)
        + _question_html(question, share_url)
        + f'<div style="margin-top:20px;padding:18px;border:1px solid {BORDER};border-radius:12px">'
        f'<p style="{_EYEBROW_STYLE}">New consensus</p>'
        f'<div style="margin-top:8px;color:{INK_SOFT}">{safe_consensus}</div></div>'
        + _button_html(share_url, "Open watch page")
    )
    html_body = _shell_html(
        eyebrow="Consensus Watch · Condition met",
        heading="Your watch condition is met",
        preheader=_split_lead(clipped_reason)[0] or clipped_condition,
        body=body,
        footer=(
            "This message was sent because your watch condition became true. "
            "It will not repeat while the condition remains true. "
            f'<a href="{html.escape(unsubscribe_url)}">Pause this watch</a>.'
        ),
        width=680,
    )
    return _base_message(recipient, subject, plain, html_body)


def build_follow_confirm_message(*, recipient: str, question: str,
                                 confirm_url: str, share_url: str) -> EmailMessage:
    """Double-Opt-in: einmalige Bestätigungs-Mail für Seiten-Follower."""
    clipped_question = _normalized(question)
    subject_question = clipped_question[:72] + ("…" if len(clipped_question) > 72 else "")
    subject = f"Confirm: follow \"{subject_question}\""
    plain = (
        "Confirm that you want to follow this question on consens.io.\n\n"
        + _question_plain(question) + "\n\n"
        "You will get one e-mail whenever the AI consensus shifts materially.\n"
        f"Confirm: {confirm_url}\n\n"
        f"Page: {share_url}\n"
        "If you did not request this, simply ignore this e-mail — nothing is stored.\n"
    )
    body = (
        f'<div style="margin-top:16px">{_question_html(question, share_url)}</div>'
        f'<p style="margin:14px 0 0;color:{INK_SOFT}">You will get one e-mail whenever the '
        "AI consensus shifts materially — no account needed.</p>"
        + _button_html(confirm_url, "Confirm and follow")
    )
    html_body = _shell_html(
        eyebrow="consens.io · Confirmation needed",
        heading="Follow this question?",
        preheader="One click confirms; we only write when the consensus moves.",
        body=body,
        footer=(
            "If you did not request this, simply ignore this e-mail — nothing is stored. "
            f'<a href="{html.escape(share_url)}">Open the page</a>.'
        ),
    )
    return _base_message(recipient, subject, plain, html_body)


def build_topic_follow_confirm_message(*, recipient: str, title: str,
                                       confirm_url: str, topic_url: str) -> EmailMessage:
    """Double opt-in confirmation for the independent curated Topics area."""
    clipped_title = " ".join(str(title or "").split())
    subject_title = clipped_title[:72] + ("…" if len(clipped_title) > 72 else "")
    subject = f'Confirm: follow "{subject_title}"'
    plain = (
        "Confirm that you want to follow this topic on consens.io.\n\n"
        f"{clipped_title}\n\n"
        "You will receive an e-mail when the curated consensus changes materially.\n"
        f"Confirm: {confirm_url}\n\n"
        f"Topic: {topic_url}\n"
        "If you did not request this, ignore this e-mail; nothing is stored.\n"
    )
    safe_title = html.escape(clipped_title)
    safe_confirm = html.escape(confirm_url)
    safe_topic = html.escape(topic_url)
    html_body = f"""<!doctype html><html><body style="font-family:Arial,sans-serif;color:#172033;line-height:1.55">
<div style="max-width:620px;margin:auto;padding:24px"><h1 style="font-size:22px">Follow this topic?</h1>
<p style="font-size:17px;font-weight:600">{safe_title}</p>
<p>You will receive an e-mail when the curated AI consensus changes materially.</p>
<p><a href="{safe_confirm}" style="display:inline-block;background:#335cff;color:#fff;text-decoration:none;padding:12px 18px;border-radius:8px">Confirm and follow</a></p>
<p style="font-size:12px;color:#667085;margin-top:32px">If you did not request this, ignore this e-mail; nothing is stored. <a href="{safe_topic}">Open the topic</a>.</p>
</div></body></html>"""
    return _base_message(recipient, subject, plain, html_body)


def build_topic_change_message(*, recipient: str, title: str, question: str,
                               old_score, new_score, change_type: str,
                               summary: str, topic_url: str,
                               unsubscribe_url: str) -> EmailMessage:
    """Material-change notification for a confirmed Topic follower."""
    clipped_title = _normalized(title)
    subject_title = clipped_title[:72] + ("…" if len(clipped_title) > 72 else "")
    subject = f"Topic update: {subject_title}"
    lead, _rest = _split_lead(summary)
    facts = [
        _score_fact(old_score, new_score),
        {"label": "Change", "value": _normalized(change_type).title() or "Material", "note": ""},
    ]
    plain = (
        f"The curated consensus changed ({change_type}).\n\n{clipped_title}\n\n"
        + _change_block_plain("What changed", summary) + "\n\n"
        + _facts_plain(facts) + "\n\n"
        + _question_plain(question) + "\n\n"
        f"Open the timeline: {topic_url}\nUnfollow: {unsubscribe_url}\n"
    )
    body = (
        _change_block_html("What changed", summary)
        + _facts_html(facts)
        + _question_html(question, topic_url, label="Tracked question")
        + _button_html(topic_url, "Open the timeline")
    )
    html_body = _shell_html(
        eyebrow="consens.io · Curated topic update",
        heading=clipped_title,
        preheader=lead or "The curated consensus changed.",
        body=body,
        footer=(
            "You follow this curated topic on consens.io. "
            f'<a href="{html.escape(unsubscribe_url)}">Unfollow</a>.'
        ),
    )
    return _base_message(recipient, subject, plain, html_body)


def build_follower_change_message(*, recipient: str, question: str, old_score,
                                  new_score, summary: str, share_url: str,
                                  unsubscribe_url: str, severity: str = "major",
                                  direction=None) -> EmailMessage:
    """Änderungs-Mail an bestätigte Seiten-Follower (nicht den Watch-Owner)."""
    clipped_question = _normalized(question)
    subject_question = clipped_question[:72] + ("…" if len(clipped_question) > 72 else "")
    subject = f"The AI consensus shifted: {subject_question}"
    lead, _rest = _split_lead(summary)
    facts = [
        _score_fact(old_score, new_score),
        _direction_fact(direction),
        _severity_fact(True, severity),
    ]
    plain = (
        "A question you follow on consens.io changed materially.\n\n"
        + _change_block_plain("What changed", summary) + "\n\n"
        + _facts_plain(facts) + "\n\n"
        + _question_plain(question) + "\n\n"
        f"See what changed: {share_url}\nUnfollow: {unsubscribe_url}\n"
    )
    body = (
        _change_block_html("What changed", summary)
        + _facts_html(facts)
        + _question_html(question, share_url)
        + _button_html(share_url, "See what changed")
    )
    html_body = _shell_html(
        eyebrow="consens.io · Question you follow",
        heading="The AI consensus shifted",
        preheader=lead or "A question you follow moved.",
        body=body,
        footer=(
            "You follow this question on consens.io. "
            f'<a href="{html.escape(unsubscribe_url)}">Unfollow</a>.'
        ),
    )
    return _base_message(recipient, subject, plain, html_body)


def build_paused_message(*, recipient: str, question: str, share_url: str,
                         unsubscribe_url: str) -> EmailMessage:
    subject = "Consensus Watch paused after repeated errors"
    plain = (
        "Your watch was paused after three failed checks.\n\n"
        + _question_plain(question) + f"\n\n{share_url}\nUnsubscribe: {unsubscribe_url}\n"
    )
    body = (
        f'<p style="margin:14px 0 0;color:{INK_SOFT}">We could not complete three consecutive '
        "checks, so this watch was paused automatically. Nothing was lost — resuming it "
        "continues the same history.</p>"
        f'<div style="margin-top:16px">{_question_html(question, share_url)}</div>'
        + _button_html(share_url, "Open the consensus page")
    )
    html_body = _shell_html(
        eyebrow="Consensus Watch · Paused",
        heading="This watch was paused",
        preheader="Three consecutive checks failed.",
        body=body,
        footer=f'<a href="{html.escape(unsubscribe_url)}">Pause/unsubscribe</a>',
    )
    return _base_message(recipient, subject, plain, html_body)


def _brief_status_label(status: str) -> str:
    return {
        "active": "Active",
        "paused": "Paused",
        "paused_error": "Paused after errors",
    }.get(status, "Paused")


def _brief_summaries(item: dict) -> list:
    return [
        _normalized(point.get("change_summary"))
        for point in (item.get("new_points") or [])
        if point.get("notable") and point.get("change_summary")
    ]


def _brief_score_line(item: dict) -> str:
    score = item.get("score")
    if not isinstance(score, (int, float)):
        return "No check completed yet"
    previous = item.get("previous_score")
    if isinstance(previous, (int, float)) and int(previous) != int(score):
        arrow = "↑" if score > previous else "↓"
        return f"{int(score)}/100 agreement ({arrow} from {int(previous)})"
    return f"{int(score)}/100 agreement"


def build_brief_message(*, recipient: str, date_label: str, items: list,
                        changes_count: int, site_url: str,
                        unsubscribe_url: str) -> EmailMessage:
    """Daily digest over all watches of one user (no per-watch mail replaced)."""
    watch_count = len(items)
    subject = (
        f"Morning brief: {changes_count} change{'s' if changes_count != 1 else ''} "
        f"across {watch_count} watch{'es' if watch_count != 1 else ''}"
        if changes_count else
        f"Morning brief: your {watch_count} watch{'es' if watch_count != 1 else ''}, no material changes"
    )
    plain_rows, html_rows = [], []
    # Watches that actually moved come first: the digest is scanned top-down
    # and a quiet week should never bury the one line that matters.
    for item in sorted(items, key=lambda entry: not _brief_summaries(entry)):
        question, _truncated = _question_view(item.get("question"), limit=150)
        plain_question = _ascii(question)
        status_label = _brief_status_label(str(item.get("status") or ""))
        score_line = _brief_score_line(item)
        url = site_url + str(item.get("share_path") or "")
        schedule = str(item.get("interval") or "").capitalize()
        if item.get("interval") == "weekly" and item.get("run_weekday"):
            schedule += f" on {str(item['run_weekday']).capitalize()}"
        if item.get("run_time") and item.get("timezone"):
            schedule += f" at {item['run_time']} ({item['timezone']})"
        summaries = _brief_summaries(item)
        plain = _ascii(f"- {plain_question}\n  {status_label} · {score_line} · {schedule}\n")
        for summary in summaries[:3]:
            plain += f"  CHANGED: {summary}\n"
        if not summaries:
            plain += "  No material change.\n"
        plain += f"  {url}\n"
        plain_rows.append(plain)

        safe_question = html.escape(question)
        safe_url = html.escape(url)
        change_html = "".join(
            f'<p style="margin:9px 0 0;padding-left:11px;border-left:3px solid {ACCENT};'
            f'font-size:14px;color:{INK}"><strong>Changed.</strong> {html.escape(summary)}</p>'
            for summary in summaries[:3]
        )
        html_rows.append(
            f'<div style="margin:0 0 14px;padding:14px 16px;border:1px solid {BORDER};border-radius:12px">'
            f'<p style="margin:0;font-weight:600"><a href="{safe_url}" '
            f'style="color:{INK};text-decoration:none">{safe_question}</a></p>'
            f'<p style="margin:6px 0 0;font-size:13px;color:{MUTED}">{html.escape(status_label)} · '
            f"{html.escape(score_line)} · {html.escape(schedule)}</p>"
            f"{change_html}"
            f"</div>"
        )
    plain = (
        f"Your Consensus Watch morning brief - {date_label}\n\n"
        + ("\n".join(plain_rows) if plain_rows else "You have no watches yet.\n")
        + f"\nOpen consens.io: {site_url}/app\nUnsubscribe from this brief: {unsubscribe_url}\n"
    )
    intro = (
        f"{changes_count} notable change{'s' if changes_count != 1 else ''} since your last brief."
        if changes_count else "No material changes since your last brief."
    )
    body = (
        f'<p style="margin:6px 0 18px;color:{MUTED}">{html.escape(date_label)} · '
        f"{html.escape(intro)}</p>"
        + "".join(html_rows)
        + _button_html(f"{site_url}/app", "Open your watch dashboard")
    )
    html_body = _shell_html(
        eyebrow="Consensus Watch · Morning brief",
        heading="Your consensus morning brief",
        preheader=intro,
        body=body,
        footer=(
            "You receive this daily digest because you enabled the Morning Brief. "
            f'<a href="{html.escape(unsubscribe_url)}">Unsubscribe from the brief</a> '
            "— individual watch alerts are unaffected."
        ),
        width=680,
    )
    return _base_message(recipient, subject, plain, html_body)


def build_test_message(*, recipient: str) -> EmailMessage:
    """Small delivery probe used only by the authenticated admin endpoint."""
    sent_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    plain = (
        "Consensus Watch e-mail delivery is configured correctly.\n\n"
        f"This test was requested from the admin dashboard at {sent_at}.\n"
        "No watch was executed and no schedule was changed.\n"
    )
    html_body = f"""<!doctype html><html><body style="font-family:Arial,sans-serif;color:#172033;line-height:1.55">
<div style="max-width:620px;margin:auto;padding:24px"><h1 style="font-size:22px">Consensus Watch test successful</h1>
<p>The application connected to SMTP and submitted this message successfully.</p>
<p style="color:#667085">Requested from the admin dashboard at {html.escape(sent_at)}. No watch was executed and no schedule was changed.</p>
</div></body></html>"""
    return _base_message(recipient, "Consensus Watch e-mail test", plain, html_body)
