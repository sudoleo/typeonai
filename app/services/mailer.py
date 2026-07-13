"""Small async SMTP service for Consensus Watch notifications."""

from __future__ import annotations

import asyncio
import html
import logging
import os
import smtplib
import ssl
from email.message import EmailMessage
from datetime import datetime, timezone


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
    except Exception:
        logging.exception("Consensus Watch mail delivery failed")
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


def build_change_message(*, recipient: str, question: str, old_score, new_score,
                         summary: str, share_url: str, unsubscribe_url: str) -> EmailMessage:
    clipped_question = " ".join(str(question or "").split())
    subject_question = clipped_question[:72] + ("…" if len(clipped_question) > 72 else "")
    subject = f"Consensus changed: {subject_question}"
    plain = (
        f"Consensus Watch detected a material change.\n\n{clipped_question}\n"
        f"Agreement score: {old_score} -> {new_score}\n{summary}\n\n"
        f"View history: {share_url}\nUnsubscribe: {unsubscribe_url}\n"
    )
    safe = {key: html.escape(str(value)) for key, value in {
        "question": clipped_question, "old": old_score, "new": new_score,
        "summary": summary, "share": share_url, "unsubscribe": unsubscribe_url,
    }.items()}
    html_body = f"""<!doctype html><html><body style="font-family:Arial,sans-serif;color:#172033;line-height:1.55">
<div style="max-width:620px;margin:auto;padding:24px"><h1 style="font-size:22px">Consensus Watch update</h1>
<p style="font-size:17px;font-weight:600">{safe['question']}</p>
<div style="padding:18px;background:#f3f6fb;border-radius:12px;font-size:20px"><span style="color:#667085">{safe['old']}</span> → <strong>{safe['new']}</strong> agreement</div>
<p>{safe['summary']}</p><p><a href="{safe['share']}" style="display:inline-block;background:#335cff;color:#fff;text-decoration:none;padding:12px 18px;border-radius:8px">View consensus history</a></p>
<p style="font-size:12px;color:#667085;margin-top:32px">You received this service message because you enabled Consensus Watch. <a href="{safe['unsubscribe']}">Pause this watch</a>.</p>
</div></body></html>"""
    return _base_message(recipient, subject, plain, html_body)


def build_run_message(*, recipient: str, question: str, agreement_score,
                      consensus: str, changed: bool, severity: str,
                      summary: str, share_url: str, unsubscribe_url: str) -> EmailMessage:
    """Full-content notification for users who opted into every successful run."""
    clipped_question = " ".join(str(question or "").split())
    subject_question = clipped_question[:72] + ("…" if len(clipped_question) > 72 else "")
    subject = f"New consensus: {subject_question}"
    consensus_text = str(consensus or "").strip()
    change_line = (
        f"Content change detected ({severity}): {summary}"
        if changed else "No material content change was detected."
    )
    plain = (
        f"Consensus Watch completed a new check.\n\n{clipped_question}\n"
        f"Agreement score: {agreement_score}/100\n{change_line}\n\n"
        f"NEW CONSENSUS\n\n{consensus_text}\n\n"
        f"View history: {share_url}\nUnsubscribe: {unsubscribe_url}\n"
    )
    safe_question = html.escape(clipped_question)
    safe_consensus = html.escape(consensus_text).replace("\n", "<br>")
    safe_summary = html.escape(summary or "")
    safe_share = html.escape(share_url)
    safe_unsubscribe = html.escape(unsubscribe_url)
    change_html = (
        f"<p><strong>Content change detected ({html.escape(severity)}):</strong> {safe_summary}</p>"
        if changed else "<p style='color:#667085'>No material content change was detected.</p>"
    )
    html_body = f"""<!doctype html><html><body style="font-family:Arial,sans-serif;color:#172033;line-height:1.55">
<div style="max-width:680px;margin:auto;padding:24px"><h1 style="font-size:22px">New Consensus Watch result</h1>
<p style="font-size:17px;font-weight:600">{safe_question}</p>
<div style="padding:16px;background:#f3f6fb;border-radius:12px"><strong style="font-size:22px">{int(agreement_score)}/100</strong> agreement</div>
{change_html}<div style="margin-top:20px;padding:18px;border:1px solid #d8deea;border-radius:12px"><h2 style="font-size:16px;margin-top:0">New consensus</h2><div>{safe_consensus}</div></div>
<p><a href="{safe_share}" style="display:inline-block;background:#335cff;color:#fff;text-decoration:none;padding:12px 18px;border-radius:8px">View watch page and history</a></p>
<p style="font-size:12px;color:#667085;margin-top:32px">You chose to receive every new consensus result. <a href="{safe_unsubscribe}">Pause this watch</a>.</p>
</div></body></html>"""
    return _base_message(recipient, subject, plain, html_body)


def build_condition_message(*, recipient: str, question: str, condition: str,
                            reason: str, agreement_score, consensus: str,
                            share_url: str, unsubscribe_url: str) -> EmailMessage:
    """Notification emitted once when a user-defined condition becomes true."""
    clipped_question = " ".join(str(question or "").split())
    clipped_condition = " ".join(str(condition or "").split())[:500]
    clipped_reason = " ".join(str(reason or "").split())[:400]
    subject_condition = clipped_condition[:72] + ("…" if len(clipped_condition) > 72 else "")
    subject = f"Watch condition met: {subject_condition}"
    consensus_text = str(consensus or "").strip()
    plain = (
        f"Your Consensus Watch condition is now met.\n\n"
        f"QUESTION\n{clipped_question}\n\nCONDITION\n{clipped_condition}\n\n"
        f"WHY IT TRIGGERED\n{clipped_reason}\n\nAgreement score: {agreement_score}/100\n\n"
        f"NEW CONSENSUS\n\n{consensus_text}\n\n"
        f"Open watch page: {share_url}\nPause this watch: {unsubscribe_url}\n"
    )
    safe = {key: html.escape(str(value)) for key, value in {
        "question": clipped_question,
        "condition": clipped_condition,
        "reason": clipped_reason,
        "score": agreement_score,
        "consensus": consensus_text,
        "share": share_url,
        "unsubscribe": unsubscribe_url,
    }.items()}
    safe["consensus"] = safe["consensus"].replace("\n", "<br>")
    html_body = f"""<!doctype html><html><body style="font-family:Arial,sans-serif;color:#172033;line-height:1.55">
<div style="max-width:680px;margin:auto;padding:24px"><h1 style="font-size:22px">Your watch condition is met</h1>
<p style="font-size:17px;font-weight:600">{safe['question']}</p>
<div style="padding:18px;background:#eef8f1;border-radius:12px"><strong>Condition</strong><br>{safe['condition']}</div>
<p><strong>Why it triggered:</strong> {safe['reason']}</p>
<div style="padding:14px;background:#f3f6fb;border-radius:12px"><strong>{safe['score']}/100</strong> agreement</div>
<div style="margin-top:20px;padding:18px;border:1px solid #d8deea;border-radius:12px"><h2 style="font-size:16px;margin-top:0">New consensus</h2><div>{safe['consensus']}</div></div>
<p><a href="{safe['share']}" style="display:inline-block;background:#335cff;color:#fff;text-decoration:none;padding:12px 18px;border-radius:8px">Open watch page</a></p>
<p style="font-size:12px;color:#667085;margin-top:32px">This message was sent because your watch condition became true. It will not repeat while the condition remains true. <a href="{safe['unsubscribe']}">Pause this watch</a>.</p>
</div></body></html>"""
    return _base_message(recipient, subject, plain, html_body)


def build_paused_message(*, recipient: str, question: str, share_url: str,
                         unsubscribe_url: str) -> EmailMessage:
    clipped = " ".join(str(question or "").split())
    subject = "Consensus Watch paused after repeated errors"
    plain = f"Your watch was paused after three failed checks.\n\n{clipped}\n{share_url}\nUnsubscribe: {unsubscribe_url}\n"
    html_body = (
        "<!doctype html><html><body style='font-family:Arial,sans-serif;color:#172033;line-height:1.55'>"
        "<div style='max-width:620px;margin:auto;padding:24px'><h1>Watch paused</h1>"
        f"<p>{html.escape(clipped)}</p><p>We could not complete three consecutive checks, so this watch was paused automatically.</p>"
        f"<p><a href='{html.escape(share_url)}'>Open the consensus page</a></p>"
        f"<p style='font-size:12px;color:#667085'><a href='{html.escape(unsubscribe_url)}'>Pause/unsubscribe</a></p></div></body></html>"
    )
    return _base_message(recipient, subject, plain, html_body)


def _brief_status_label(status: str) -> str:
    return {
        "active": "Active",
        "paused": "Paused",
        "paused_error": "Paused after errors",
    }.get(status, "Paused")


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
    for item in items:
        question = " ".join(str(item.get("question") or "").split())
        status_label = _brief_status_label(str(item.get("status") or ""))
        score_line = _brief_score_line(item)
        url = site_url + str(item.get("share_path") or "")
        schedule = str(item.get("interval") or "").capitalize()
        if item.get("run_time") and item.get("timezone"):
            schedule += f" at {item['run_time']} ({item['timezone']})"
        summaries = [
            " ".join(str(point.get("change_summary") or "").split())
            for point in (item.get("new_points") or [])
            if point.get("notable") and point.get("change_summary")
        ]
        plain = f"- {question}\n  {status_label} · {score_line} · {schedule}\n"
        for summary in summaries[:3]:
            plain += f"  Change: {summary}\n"
        plain += f"  {url}\n"
        plain_rows.append(plain)

        safe_question = html.escape(question)
        safe_url = html.escape(url)
        change_html = "".join(
            f"<p style='margin:6px 0 0;font-size:13px'><strong>Change:</strong> {html.escape(summary)}</p>"
            for summary in summaries[:3]
        )
        html_rows.append(
            f"<div style='margin:0 0 14px;padding:14px 16px;border:1px solid #d8deea;border-radius:12px'>"
            f"<p style='margin:0;font-weight:600'><a href='{safe_url}' style='color:#172033;text-decoration:none'>{safe_question}</a></p>"
            f"<p style='margin:6px 0 0;font-size:13px;color:#667085'>{html.escape(status_label)} · {html.escape(score_line)} · {html.escape(schedule)}</p>"
            f"{change_html}"
            f"</div>"
        )
    plain = (
        f"Your Consensus Watch morning brief — {date_label}\n\n"
        + ("\n".join(plain_rows) if plain_rows else "You have no watches yet.\n")
        + f"\nOpen consens.io: {site_url}/app\nUnsubscribe from this brief: {unsubscribe_url}\n"
    )
    intro = (
        f"{changes_count} notable change{'s' if changes_count != 1 else ''} since your last brief."
        if changes_count else "No material changes since your last brief."
    )
    html_body = f"""<!doctype html><html><body style="font-family:Arial,sans-serif;color:#172033;line-height:1.55">
<div style="max-width:680px;margin:auto;padding:24px"><h1 style="font-size:22px">Your consensus morning brief</h1>
<p style="color:#667085;margin-top:-6px">{html.escape(date_label)} · {html.escape(intro)}</p>
{''.join(html_rows)}
<p><a href="{html.escape(site_url)}/app" style="display:inline-block;background:#335cff;color:#fff;text-decoration:none;padding:12px 18px;border-radius:8px">Open your watch dashboard</a></p>
<p style="font-size:12px;color:#667085;margin-top:32px">You receive this daily digest because you enabled the Morning Brief. <a href="{html.escape(unsubscribe_url)}">Unsubscribe from the brief</a> — individual watch alerts are unaffected.</p>
</div></body></html>"""
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
