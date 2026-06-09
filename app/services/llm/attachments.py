"""Validierung und Aufbereitung von Datei-Anhängen (PDF/Bilder) für die Provider-Requests."""
from __future__ import annotations

import base64
import io
import logging
from fastapi import HTTPException

logger = logging.getLogger(__name__)

MAX_ATTACHMENTS = 2
MAX_ATTACHMENT_BYTES = 5 * 1024 * 1024  # 5 MB pro Datei
MAX_PDF_EXTRACT_CHARS = 24000

ALLOWED_ATTACHMENT_MIMES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/webp",
}

IMAGE_MIMES = {"image/png", "image/jpeg", "image/webp"}

# Provider, die Bilder bzw. PDFs nativ als Content-Block verarbeiten können.
# Alle anderen erhalten einen Text-Fallback (PDF-Extraktion bzw. Hinweis).
PROVIDER_IMAGE_SUPPORT = {"openai", "anthropic", "gemini", "grok"}
PROVIDER_PDF_SUPPORT = {"openai", "anthropic", "gemini"}


def _sniff_mime(raw: bytes) -> str | None:
    if raw.startswith(b"%PDF"):
        return "application/pdf"
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return None


def parse_attachments(data: dict, is_pro: bool) -> list[dict]:
    """Liest und validiert `attachments` aus dem Request-Body.

    Gibt eine Liste von {name, mime, data (base64), raw (bytes)} zurück.
    """
    raw_list = data.get("attachments")
    if not raw_list:
        return []

    if not is_pro:
        raise HTTPException(
            status_code=403,
            detail="File uploads are exclusively available for Pro users.",
        )

    if not isinstance(raw_list, list):
        raise HTTPException(status_code=400, detail="Invalid attachments format.")
    if len(raw_list) > MAX_ATTACHMENTS:
        raise HTTPException(
            status_code=400,
            detail=f"A maximum of {MAX_ATTACHMENTS} attachments is allowed.",
        )

    parsed = []
    for item in raw_list:
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail="Invalid attachment entry.")

        name = str(item.get("name") or "attachment")[:200]
        b64_data = item.get("data")
        if not isinstance(b64_data, str) or not b64_data.strip():
            raise HTTPException(status_code=400, detail=f"Attachment '{name}' has no data.")

        # Data-URL-Präfix tolerieren
        if b64_data.startswith("data:"):
            b64_data = b64_data.split(",", 1)[-1]

        try:
            raw = base64.b64decode(b64_data, validate=True)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Attachment '{name}' is not valid base64.")

        if len(raw) > MAX_ATTACHMENT_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"Attachment '{name}' exceeds the {MAX_ATTACHMENT_BYTES // (1024 * 1024)} MB size limit.",
            )

        mime = _sniff_mime(raw)
        if mime is None or mime not in ALLOWED_ATTACHMENT_MIMES:
            raise HTTPException(
                status_code=400,
                detail=f"Attachment '{name}' has an unsupported file type. Allowed: PDF, PNG, JPG, WebP.",
            )

        parsed.append({
            "name": name,
            "mime": mime,
            "data": b64_data,
            "raw": raw,
        })

    return parsed


def extract_pdf_text(raw: bytes) -> str | None:
    """Serverseitige Textextraktion als Fallback für Provider ohne natives PDF-Verständnis."""
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.warning("pypdf is not installed; PDF text extraction unavailable.")
        return None

    try:
        reader = PdfReader(io.BytesIO(raw))
        chunks = []
        total = 0
        for page in reader.pages:
            text = page.extract_text() or ""
            if not text.strip():
                continue
            chunks.append(text)
            total += len(text)
            if total >= MAX_PDF_EXTRACT_CHARS:
                break
        combined = "\n".join(chunks).strip()
        if not combined:
            return None
        return combined[:MAX_PDF_EXTRACT_CHARS]
    except Exception as exc:
        logger.warning("PDF text extraction failed: %s", exc)
        return None


def attachment_fallback_text(attachment: dict, *, include_images_note: bool = True) -> str:
    """Baut den Text-Fallback für einen Anhang (für Provider ohne native Unterstützung)."""
    name = attachment.get("name", "attachment")
    mime = attachment.get("mime", "")

    if mime == "application/pdf":
        text = extract_pdf_text(attachment.get("raw", b""))
        if text:
            return (
                f"--- Attached document: {name} (extracted text) ---\n"
                f"{text}\n"
                f"--- End of document: {name} ---"
            )
        return (
            f"[The user attached the PDF '{name}', but its text could not be extracted. "
            "Mention that you could not read the document if it is relevant to the question.]"
        )

    if include_images_note and mime in IMAGE_MIMES:
        return (
            f"[The user attached the image '{name}', but this model cannot view images. "
            "Mention that you could not see the image if it is relevant to the question.]"
        )

    return ""


def build_attachment_question_suffix(attachments: list[dict], provider_key: str) -> str:
    """Sammelt alle Text-Fallbacks, die für den Provider nötig sind."""
    if not attachments:
        return ""

    parts = []
    for att in attachments:
        mime = att.get("mime", "")
        if mime == "application/pdf" and provider_key in PROVIDER_PDF_SUPPORT:
            continue
        if mime in IMAGE_MIMES and provider_key in PROVIDER_IMAGE_SUPPORT:
            continue
        fallback = attachment_fallback_text(att)
        if fallback:
            parts.append(fallback)

    if not parts:
        return ""
    return "\n\n" + "\n\n".join(parts)


def native_attachments_for_provider(attachments: list[dict], provider_key: str) -> list[dict]:
    """Filtert die Anhänge, die der Provider nativ als Content-Block erhält."""
    if not attachments:
        return []

    native = []
    for att in attachments:
        mime = att.get("mime", "")
        if mime == "application/pdf" and provider_key in PROVIDER_PDF_SUPPORT:
            native.append(att)
        elif mime in IMAGE_MIMES and provider_key in PROVIDER_IMAGE_SUPPORT:
            native.append(att)
    return native
