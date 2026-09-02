"""Validierung und Aufbereitung von Datei-Anhängen (PDF/Word/Text/Bilder) für die Provider-Requests."""
from __future__ import annotations

import base64
import io
import logging
import xml.etree.ElementTree as ET
import zipfile
from fastapi import HTTPException

from app.core.observability import safe_exception

logger = logging.getLogger(__name__)

MAX_ATTACHMENTS = 2
MAX_ATTACHMENT_BYTES = 5 * 1024 * 1024  # 5 MB pro Datei (Eingang)
MAX_ATTACHMENT_BASE64_CHARS = 4 * ((MAX_ATTACHMENT_BYTES + 2) // 3)

# Was nach der Aufbereitung noch an die Provider gehen darf. Ein Anhang geht
# an bis zu sechs Familien GLEICHZEITIG raus und base64 legt ein Drittel
# obendrauf: aus 5 MB werden 40 MB Ausgang plus die Bildtokens in jedem der
# sechs Prompts. Deshalb wird jedes Bild auf Providergroesse gebracht
# (shrink_image) und ein grosses PDF geht nur noch als extrahierter Text raus.
IMAGE_MAX_EDGE = 1568           # laengste Kante nach dem Verkleinern
IMAGE_TARGET_BYTES = 900_000    # Zielgroesse der Neukodierung
IMAGE_JPEG_QUALITIES = (85, 70, 55)
MAX_IMAGE_BYTES = 1_500_000     # harte Grenze NACH dem Verkleinern
IMAGE_MAX_PIXELS = 40_000_000   # Schutz vor kleinen Dateien mit riesiger Dekodierung
PDF_NATIVE_MAX_BYTES = 2 * 1024 * 1024
MAX_ATTACHMENT_TOTAL_BYTES = 6 * 1024 * 1024
MAX_PDF_EXTRACT_CHARS = 24000
MAX_TEXT_EXTRACT_CHARS = 24000
MAX_DOCX_ENTRIES = 256
MAX_DOCX_TOTAL_UNCOMPRESSED_BYTES = 20 * 1024 * 1024
MAX_DOCX_ENTRY_UNCOMPRESSED_BYTES = 8 * 1024 * 1024
MAX_DOCX_COMPRESSION_RATIO = 100
DOCX_READ_CHUNK_BYTES = 64 * 1024

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
TEXT_MIME = "text/plain"

ALLOWED_ATTACHMENT_MIMES = {
    "application/pdf",
    DOCX_MIME,
    TEXT_MIME,
    "image/png",
    "image/jpeg",
    "image/webp",
}

IMAGE_MIMES = {"image/png", "image/jpeg", "image/webp"}

ATTACHMENT_TYPES_LABEL = "PDF, Word (.docx), text (.txt/.md/.csv), PNG, JPG, WebP"

# Der Browser meldet für dieselbe Textdatei je nach Betriebssystem
# "text/markdown", "text/csv" oder gar nichts. Der Lauf selbst stört sich nicht
# daran (dort entscheiden die BYTES, siehe _sniff_mime), die Metadaten kommen
# aber als Client-Angabe an. Ohne diese Tabelle fiel eine .csv still aus dem
# Bookmark heraus, obwohl der Lauf mit ihr funktioniert hat.
ATTACHMENT_MIME_ALIASES = {
    "text/markdown": TEXT_MIME,
    "text/x-markdown": TEXT_MIME,
    "text/csv": TEXT_MIME,
    "application/csv": TEXT_MIME,
    "image/jpg": "image/jpeg",
}


def normalize_attachment_mime(raw) -> str | None:
    """Client-MIME auf einen der erlaubten Typen bringen (sonst None)."""
    mime = str(raw or "").split(";", 1)[0].strip().lower()
    mime = ATTACHMENT_MIME_ALIASES.get(mime, mime)
    return mime if mime in ALLOWED_ATTACHMENT_MIMES else None


def normalize_attachment_meta(raw) -> list[dict]:
    """Anhang-Angaben auf reine Metadaten reduzieren (Name/Typ/Größe).

    Dateidaten werden bewusst verworfen: in Firestore landen nie Datei-Bytes
    (Dokument-Limit 1 MiB, Kosten). Was bleibt, ist das, was eine gespeicherte
    Frage über ihre Anhänge erzählen können muss.
    """
    if not isinstance(raw, list):
        return []

    normalized = []
    for item in raw:
        if len(normalized) >= MAX_ATTACHMENTS:
            break
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()[:200]
        mime = normalize_attachment_mime(item.get("mime"))
        if not name or not mime:
            continue
        try:
            size = max(0, int(item.get("size") or 0))
        except (TypeError, ValueError):
            size = 0
        normalized.append({"name": name, "mime": mime, "size": size})
    return normalized

# Provider, die Bilder bzw. PDFs nativ als Content-Block verarbeiten können.
# Alle anderen erhalten einen Text-Fallback (PDF-Extraktion bzw. Hinweis).
PROVIDER_IMAGE_SUPPORT = {
    "openai", "anthropic", "gemini", "grok", "kimi", "glm", "meta"
}
# Meta steht bewusst nicht hier: Muse Spark 1.3 verarbeitet PDFs nativ, das
# freie Muse Glimmer 30B nur Text und Bilder. Bis die PDF-Faehigkeit wie bei
# Anhaengen auf Modellebene aufgeloest wird, bekommt die ganze Familie den
# Text-Fallback (PDF-Extraktion) statt eines Blocks, den Glimmer ablehnt.
PROVIDER_PDF_SUPPORT = {"openai", "anthropic", "gemini"}

_DOCX_ALLOWED_EXACT = {"[Content_Types].xml"}
_DOCX_ALLOWED_PREFIXES = ("_rels/", "docProps/", "word/", "customXml/")


class InvalidDocx(ValueError):
    pass


class ImagePixelLimitExceeded(ValueError):
    pass


def _validate_docx_archive(raw: bytes) -> zipfile.ZipInfo:
    """Validate the ZIP directory without expanding attacker-controlled data."""
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entries = archive.infolist()
            if not entries or len(entries) > MAX_DOCX_ENTRIES:
                raise InvalidDocx("invalid entry count")
            total_uncompressed = 0
            document_info = None
            for info in entries:
                name = info.filename
                normalized_parts = name.replace("\\", "/").split("/")
                if (
                    not name
                    or name.startswith(("/", "\\"))
                    or "\\" in name
                    or ".." in normalized_parts
                    or (
                        name not in _DOCX_ALLOWED_EXACT
                        and not name.startswith(_DOCX_ALLOWED_PREFIXES)
                    )
                ):
                    raise InvalidDocx("unexpected ZIP entry")
                if info.flag_bits & 0x1:
                    raise InvalidDocx("encrypted ZIP entries are not allowed")
                if info.is_dir():
                    continue
                if info.file_size > MAX_DOCX_ENTRY_UNCOMPRESSED_BYTES:
                    raise InvalidDocx("ZIP entry expansion exceeds budget")
                total_uncompressed += info.file_size
                if total_uncompressed > MAX_DOCX_TOTAL_UNCOMPRESSED_BYTES:
                    raise InvalidDocx("DOCX expansion exceeds total budget")
                if info.file_size:
                    if info.compress_size <= 0:
                        raise InvalidDocx("invalid compressed size")
                    if info.file_size / info.compress_size > MAX_DOCX_COMPRESSION_RATIO:
                        raise InvalidDocx("ZIP compression ratio exceeds budget")
                if name == "word/document.xml":
                    document_info = info
            if document_info is None:
                raise InvalidDocx("word/document.xml is missing")
            return document_info
    except InvalidDocx:
        raise
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise InvalidDocx("invalid DOCX ZIP") from exc


def _sniff_mime(raw: bytes) -> str | None:
    if raw.startswith(b"%PDF"):
        return "application/pdf"
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    if raw.startswith(b"PK\x03\x04") and _is_docx(raw):
        return DOCX_MIME
    if _looks_like_text(raw):
        return TEXT_MIME
    return None


def _is_docx(raw: bytes) -> bool:
    """DOCX ist ein ZIP mit word/document.xml — andere ZIPs (xlsx, pptx, ...)
    werden bewusst nicht akzeptiert."""
    try:
        _validate_docx_archive(raw)
        return True
    except InvalidDocx:
        return False


def _looks_like_text(raw: bytes) -> bool:
    """UTF-8-dekodierbar und ohne Null-Bytes: deckt .txt/.md/.csv ab."""
    if not raw.strip() or b"\x00" in raw:
        return False
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def shrink_image(raw: bytes, mime: str) -> tuple[bytes, str]:
    """Bild auf Providergroesse bringen: laengste Kante ``IMAGE_MAX_EDGE``,
    Zielgroesse ``IMAGE_TARGET_BYTES``.

    Kleine Bilder bleiben unangetastet -- ein Screenshot soll seine
    PNG-Schaerfe behalten. Alles darueber wird skaliert und als JPEG neu
    kodiert: ein 4-MB-Handyfoto traegt bei 1568 px keine Information weniger,
    kostet aber ein Vielfaches, weil es in JEDEN der sechs Prompts geht.

    Bei jedem Fehler (kein Pillow, kaputte Datei, Decompression Bomb) bleibt
    das Original stehen; die Groessenpruefung im Aufrufer faengt es dann ab --
    lieber eine ehrliche Absage als ein stiller Riesen-Upload.
    """
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow fehlt nur in Alt-Umgebungen
        logger.warning("Pillow is not installed; image attachments are not shrunk.")
        return raw, mime

    try:
        source = Image.open(io.BytesIO(raw))
    except Image.DecompressionBombError as exc:
        raise ImagePixelLimitExceeded("image pixel budget exceeded") from exc
    except Exception as exc:
        logger.info("Image attachment could not be decoded category=%s", safe_exception(exc))
        return raw, mime

    if source.width * source.height > IMAGE_MAX_PIXELS:
        source.close()
        raise ImagePixelLimitExceeded("image pixel budget exceeded")

    try:
        source.load()
    except Image.DecompressionBombError as exc:
        source.close()
        raise ImagePixelLimitExceeded("image pixel budget exceeded") from exc
    except Exception as exc:
        source.close()
        logger.info("Image attachment could not be decoded category=%s", safe_exception(exc))
        return raw, mime

    with source:
        if max(source.size) <= IMAGE_MAX_EDGE and len(raw) <= IMAGE_TARGET_BYTES:
            return raw, mime

        image = source
        ratio = IMAGE_MAX_EDGE / max(image.size)
        if ratio < 1:
            image = image.resize(
                (max(1, round(image.width * ratio)), max(1, round(image.height * ratio))),
                Image.LANCZOS,
            )

        # JPEG kennt keinen Alphakanal. Fuer das Modell zaehlt das sichtbare
        # Bild, also kommt die Transparenz auf Weiss.
        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGBA")
            flattened = Image.new("RGB", image.size, (255, 255, 255))
            flattened.paste(image, mask=image.split()[-1])
            image = flattened
        elif image.mode != "RGB":
            image = image.convert("RGB")

        smallest = None
        for quality in IMAGE_JPEG_QUALITIES:
            buffer = io.BytesIO()
            try:
                image.save(buffer, format="JPEG", quality=quality, optimize=True)
            except Exception as exc:
                logger.warning(
                    "Image attachment could not be re-encoded category=%s", safe_exception(exc)
                )
                return raw, mime
            encoded = buffer.getvalue()
            if smallest is None or len(encoded) < len(smallest):
                smallest = encoded
            if len(encoded) <= IMAGE_TARGET_BYTES:
                break

    if smallest is None or len(smallest) >= len(raw):
        return raw, mime
    return smallest, "image/jpeg"


def pdf_text_for(attachment: dict) -> str | None:
    """Extrahierter PDF-Text, einmal pro Anhang. Der Text wird von bis zu
    sechs Familien gebraucht und pypdf ist zu teuer fuer sechs Laeufe."""
    if "pdf_text" not in attachment:
        attachment["pdf_text"] = extract_pdf_text(attachment.get("raw", b""))
    return attachment["pdf_text"]


def pdf_goes_native(attachment: dict) -> bool:
    """Ob das PDF als Datei an die Provider geht -- oder nur als Text.

    Nativ heisst: die kompletten Bytes gehen base64-kodiert an jede
    PDF-faehige Familie. Ueber ``PDF_NATIVE_MAX_BYTES`` wiegt das den Aufwand
    nicht auf, solange sich Text extrahieren laesst. Laesst er sich NICHT
    extrahieren (Scan), geht die Datei weiter nativ raus: sonst bekaeme das
    Modell gar nichts.
    """
    if len(attachment.get("raw", b"")) <= PDF_NATIVE_MAX_BYTES:
        return True
    return not pdf_text_for(attachment)


def parse_attachments(data: dict, attachments_allowed: bool) -> list[dict]:
    """Liest und validiert `attachments` aus dem Request-Body.

    `attachments_allowed` kommt aus den Entitlements der Kontostufe (Plus und
    Pro) -- bewusst kein `is_pro` mehr: Anhaenge kosten nur so viel wie das
    antwortende Modell, und Plus faehrt ohnehin die guenstige Modellauswahl.

    Gibt eine Liste von {name, mime, data (base64), raw (bytes)} zurück.
    """
    raw_list = data.get("attachments")
    if not raw_list:
        return []

    if not attachments_allowed:
        raise HTTPException(
            status_code=403,
            detail="File uploads need Plus or Pro.",
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

        # Reject before allocating the decoded byte buffer. validate=True below
        # intentionally disallows whitespace, so the encoded length is exact.
        if len(b64_data) > MAX_ATTACHMENT_BASE64_CHARS:
            raise HTTPException(
                status_code=400,
                detail=f"Attachment '{name}' exceeds the encoded size limit.",
            )
        try:
            b64_data.encode("ascii")
        except UnicodeEncodeError:
            raise HTTPException(
                status_code=400,
                detail=f"Attachment '{name}' is not valid base64.",
            ) from None

        try:
            raw = base64.b64decode(b64_data, validate=True)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Attachment '{name}' is not valid base64.")

        if len(raw) > MAX_ATTACHMENT_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"Attachment '{name}' exceeds the {MAX_ATTACHMENT_BYTES // (1024 * 1024)} MB size limit.",
            )

        if raw.startswith(b"PK\x03\x04"):
            try:
                _validate_docx_archive(raw)
            except InvalidDocx:
                raise HTTPException(
                    status_code=400,
                    detail=f"Attachment '{name}' is not a safe Word document.",
                ) from None

        mime = _sniff_mime(raw)
        if mime is None or mime not in ALLOWED_ATTACHMENT_MIMES:
            raise HTTPException(
                status_code=400,
                detail=f"Attachment '{name}' has an unsupported file type. Allowed: {ATTACHMENT_TYPES_LABEL}.",
            )

        if mime in IMAGE_MIMES:
            try:
                shrunk, shrunk_mime = shrink_image(raw, mime)
            except ImagePixelLimitExceeded:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Image '{name}' has too many pixels to process safely. "
                        "Please resize it before attaching."
                    ),
                ) from None
            if len(shrunk) > MAX_IMAGE_BYTES:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Image '{name}' could not be prepared for the models. "
                        f"Please attach an image under {MAX_IMAGE_BYTES // 1024} KB "
                        "or save it as PNG/JPG first."
                    ),
                )
            if shrunk is not raw:
                logger.info(
                    "Attachment image shrunk: %s -> %s bytes",
                    len(raw),
                    len(shrunk),
                )
                raw = shrunk
                mime = shrunk_mime
                b64_data = base64.b64encode(raw).decode("ascii")
                stem, dot, _extension = name.rpartition(".")
                name = (stem if dot and stem else name) + ".jpg"

        parsed.append({
            "name": name,
            "mime": mime,
            "data": b64_data,
            "raw": raw,
        })

    # Die Einzelgrenzen sagen nichts ueber die Summe: zwei Dateien am oberen
    # Rand ergeben zusammen wieder einen Prompt, der jede Familie erschlaegt.
    total = sum(len(att["raw"]) for att in parsed)
    if total > MAX_ATTACHMENT_TOTAL_BYTES:
        raise HTTPException(
            status_code=400,
            detail=(
                "The attachments are too large together. The limit is "
                f"{MAX_ATTACHMENT_TOTAL_BYTES // (1024 * 1024)} MB per question."
            ),
        )

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
        logger.warning(
            "PDF text extraction failed category=%s", safe_exception(exc)
        )
        return None


def extract_docx_text(raw: bytes) -> str | None:
    """Extrahiert den Absatztext aus word/document.xml (kein python-docx nötig)."""
    W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    try:
        document_info = _validate_docx_archive(raw)
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            with archive.open(document_info, "r") as source:
                chunks = []
                expanded = 0
                while True:
                    chunk = source.read(DOCX_READ_CHUNK_BYTES)
                    if not chunk:
                        break
                    expanded += len(chunk)
                    if expanded > MAX_DOCX_ENTRY_UNCOMPRESSED_BYTES:
                        raise InvalidDocx("document.xml expansion exceeds budget")
                    chunks.append(chunk)
            document_xml = b"".join(chunks)
        lowered = document_xml.lower()
        if b"<!doctype" in lowered or b"<!entity" in lowered:
            raise InvalidDocx("DTD/entity declarations are not allowed")
        root = ET.fromstring(document_xml)
        paragraphs = []
        total = 0
        for paragraph in root.iter(f"{W_NS}p"):
            runs = [node.text for node in paragraph.iter(f"{W_NS}t") if node.text]
            if not runs:
                continue
            text = "".join(runs)
            paragraphs.append(text)
            total += len(text)
            if total >= MAX_TEXT_EXTRACT_CHARS:
                break
        combined = "\n".join(paragraphs).strip()
        if not combined:
            return None
        return combined[:MAX_TEXT_EXTRACT_CHARS]
    except Exception as exc:
        logger.warning(
            "DOCX text extraction failed category=%s", safe_exception(exc)
        )
        return None


def attachment_fallback_text(attachment: dict, *, include_images_note: bool = True) -> str:
    """Baut den Text-Fallback für einen Anhang (für Provider ohne native Unterstützung)."""
    name = attachment.get("name", "attachment")
    mime = attachment.get("mime", "")

    if mime == DOCX_MIME:
        text = extract_docx_text(attachment.get("raw", b""))
        if text:
            return (
                f"--- Attached document: {name} (extracted text) ---\n"
                f"{text}\n"
                f"--- End of document: {name} ---"
            )
        return (
            f"[The user attached the Word document '{name}', but its text could not be extracted. "
            "Mention that you could not read the document if it is relevant to the question.]"
        )

    if mime == TEXT_MIME:
        raw = attachment.get("raw", b"")
        try:
            text = raw.decode("utf-8").strip()
        except UnicodeDecodeError:
            text = ""
        if text:
            return (
                f"--- Attached file: {name} ---\n"
                f"{text[:MAX_TEXT_EXTRACT_CHARS]}\n"
                f"--- End of file: {name} ---"
            )
        return (
            f"[The user attached the file '{name}', but it appears to be empty.]"
        )

    if mime == "application/pdf":
        text = pdf_text_for(attachment)
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
        if mime == "application/pdf" and provider_key in PROVIDER_PDF_SUPPORT and pdf_goes_native(att):
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
        if mime == "application/pdf" and provider_key in PROVIDER_PDF_SUPPORT and pdf_goes_native(att):
            native.append(att)
        elif mime in IMAGE_MIMES and provider_key in PROVIDER_IMAGE_SUPPORT:
            native.append(att)
    return native
