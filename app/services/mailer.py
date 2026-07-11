"""Small async SMTP service for Consensus Watch notifications."""

from __future__ import annotations

import asyncio
import html
import logging
import os
import smtplib
import ssl
from email.message import EmailMessage


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
