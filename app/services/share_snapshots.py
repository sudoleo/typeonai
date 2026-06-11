"""Share-Snapshots: serverseitige Persistenz von Konsens-Ergebnissen.

Ablauf:
  1. /consensus persistiert das fertige Ergebnis als ``pending_results/{result_id}``
     (24h TTL) und liefert die ``result_id`` im Final-Event mit.
  2. POST /api/share kopiert das Pending-Ergebnis serverseitig nach
     ``shares/{share_id}`` – der Client kann keinen eigenen Inhalt publizieren.

Öffentliche Payloads werden ausschließlich über ``public_share_payload()``
gebaut (Whitelist) – ``owner_uid``, Report-Zähler und Moderations-Flags
verlassen den Server nie.
"""

import hashlib
import logging
import re
import secrets
import string
import unicodedata
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

from firebase_admin import firestore

from app.core.security import db_firestore

PENDING_COLLECTION = "pending_results"
SHARES_COLLECTION = "shares"

# Hinweis Betrieb: Für pending_results sollte zusätzlich eine Firestore-
# TTL-Policy auf dem Feld "expires_at" konfiguriert werden; bis dahin räumt
# cleanup_expired_pending() beim App-Start (täglicher Render-Restart) auf.
PENDING_TTL_HOURS = 24
SHARE_DAILY_LIMIT = 20

MAX_QUESTION_CHARS = 2000
MAX_CONSENSUS_CHARS = 100_000
MAX_DIFFERENCES_TEXT_CHARS = 50_000
MAX_SOURCES = 50

# Qualitätsfilter v1 (Etappe 3 macht das über die limits-Config steuerbar).
QUALITY_MIN_CONSENSUS_CHARS = 600
QUALITY_MIN_SOURCES = 2
QUALITY_MIN_MODELS = 3
QUALITY_QUESTION_MIN_CHARS = 15
QUALITY_QUESTION_MAX_CHARS = 300

# Bewusst ohne '-' und '_': die ID ist das letzte '-'-Segment der URL
# (/s/{slug}-{id}); ein Bindestrich in der ID würde das Parsen brechen.
_ID_ALPHABET = string.ascii_letters + string.digits
SHARE_ID_LENGTH = 16  # 62^16 ≈ 95 Bit – nicht erratbar (IDOR-Schutz)
_SHARE_ID_RE = re.compile(r"^[A-Za-z0-9]{%d}$" % SHARE_ID_LENGTH)

MAX_SLUG_CHARS = 60

PROVIDER_ORDER = ("OpenAI", "Mistral", "Anthropic", "Gemini", "DeepSeek", "Grok")
# Anzeige-Labels analog zu buildConsensusCitation() im Frontend.
PROVIDER_CITATION_LABELS = {
    "OpenAI": "OpenAI",
    "Mistral": "Mistral",
    "Anthropic": "Anthropic Claude",
    "Gemini": "Google Gemini",
    "DeepSeek": "DeepSeek",
    "Grok": "Grok",
}

_UMLAUT_MAP = str.maketrans({
    "ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss",
    "Ä": "ae", "Ö": "oe", "Ü": "ue",
})

_SOURCE_ID_RE = re.compile(r"^S?(\d{1,4})$", re.IGNORECASE)


class ShareError(Exception):
    """Fehler mit stabilem Code, den der Router auf HTTP-Status mappt."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


def _utcnow():
    return datetime.now(timezone.utc)


def _clip(value, limit):
    text = str(value if value is not None else "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip()


def generate_share_id():
    return "".join(secrets.choice(_ID_ALPHABET) for _ in range(SHARE_ID_LENGTH))


def is_valid_share_id(share_id):
    return bool(isinstance(share_id, str) and _SHARE_ID_RE.match(share_id))


def slugify_question(question):
    text = str(question or "").translate(_UMLAUT_MAP)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if len(text) > MAX_SLUG_CHARS:
        text = text[:MAX_SLUG_CHARS]
        if "-" in text:
            text = text.rsplit("-", 1)[0]
    return text or "consensus"


def split_slug_id(slug_id):
    """Zerlegt '{slug}-{id}' – der Slug darf selbst Bindestriche enthalten."""
    raw = str(slug_id or "")
    if "-" not in raw:
        return "", raw
    slug, share_id = raw.rsplit("-", 1)
    return slug, share_id


def share_path(slug, share_id):
    return "/s/%s-%s" % (slug, share_id)


def question_hash(question):
    normalized = re.sub(r"\s+", " ", str(question or "").strip().lower()).rstrip("?!. ")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _source_sort_key(source):
    match = _SOURCE_ID_RE.match(str(source.get("id") or ""))
    return int(match.group(1)) if match else 10_000


def sanitize_sources(model_sources):
    """Vereinigt die per-Modell-Quellen zu einer deduplizierten Liste.

    Die IDs ("S1", "S2", …) sind bereits die global nummerierten IDs aus dem
    Frontend, auf die sich die [S1]-Tags im Konsens-Markdown beziehen.
    Nur http(s)-URLs werden übernommen.
    """
    if isinstance(model_sources, dict):
        ordered_keys = [k for k in PROVIDER_ORDER if k in model_sources]
        ordered_keys += [k for k in model_sources if k not in PROVIDER_ORDER]
        candidates = []
        for key in ordered_keys:
            value = model_sources.get(key)
            if isinstance(value, list):
                candidates.extend(value)
    elif isinstance(model_sources, list):
        candidates = model_sources
    else:
        candidates = []

    sanitized = []
    seen_urls = set()
    seen_ids = set()
    for item in candidates:
        if not isinstance(item, dict):
            continue
        url = _clip(item.get("url"), 2000)
        try:
            scheme = urlsplit(url).scheme.lower()
        except ValueError:
            continue
        if scheme not in ("http", "https"):
            continue
        url_key = url.lower().rstrip("/")
        if url_key in seen_urls:
            continue

        raw_id = _clip(item.get("id"), 10)
        id_match = _SOURCE_ID_RE.match(raw_id)
        source_id = "S%d" % int(id_match.group(1)) if id_match else ""
        if source_id and source_id in seen_ids:
            source_id = ""

        seen_urls.add(url_key)
        if source_id:
            seen_ids.add(source_id)
        sanitized.append({
            "id": source_id,
            "title": _clip(item.get("title"), 300) or url,
            "url": url,
            "provider": _clip(item.get("provider"), 40),
        })
        if len(sanitized) >= MAX_SOURCES:
            break

    sanitized.sort(key=_source_sort_key)
    return sanitized


# Modell-Label aus dem Client (Option-Text des Model-Pickers): nur harmlose
# Zeichen, insbesondere kein Komma – die Zitation joint mit ", ", ein Komma im
# Label könnte dort zusätzliche Modelle vortäuschen.
_MODEL_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._:()/+&-]{0,79}$")


def sanitize_model_labels(model_labels, included_providers=None):
    """Labels nur für tatsächlich beteiligte Provider, sonst Provider-Fallback.

    Schlüssel außerhalb von included_providers werden verworfen (begrenzt
    damit auch die Anzahl), Werte gegen Zeichen-Whitelist + Längenlimit
    geprüft; ungültige Labels fallen still auf den Provider-Namen zurück.
    """
    if not isinstance(model_labels, dict):
        return {}
    allowed = set(PROVIDER_ORDER)
    if included_providers is not None:
        allowed &= {str(p) for p in included_providers}
    sanitized = {}
    for provider in PROVIDER_ORDER:
        if provider not in allowed:
            continue
        label = model_labels.get(provider)
        if not isinstance(label, str):
            continue
        cleaned = re.sub(r"\s+", " ", re.sub(r"[\x00-\x1f\x7f]", "", label)).strip()
        if _MODEL_LABEL_RE.match(cleaned):
            sanitized[provider] = cleaned
    return sanitized


def build_included_models(included_providers, model_labels=None):
    labels = sanitize_model_labels(model_labels, included_providers)
    included = {str(p) for p in (included_providers or [])}
    result = []
    for provider in PROVIDER_ORDER:
        if provider not in included:
            continue
        display = PROVIDER_CITATION_LABELS[provider]
        model_name = labels.get(provider)
        result.append("%s: %s" % (display, model_name) if model_name else display)
    return result


def _sanitize_str_list(value, item_limit, max_items):
    if not isinstance(value, list):
        return []
    return [_clip(item, item_limit) for item in value[:max_items] if isinstance(item, str) and item.strip()]


def sanitize_differences_data(data):
    """Whitelist-Validierung des strukturierten Differences-JSON.

    Übernimmt nur bekannte Schlüssel mit erwarteten Typen und kappt Längen –
    unbekannte Felder (oder API-Metadaten) gelangen nie in den Snapshot.
    """
    if not isinstance(data, dict):
        return None

    claims = []
    for claim in (data.get("claims") or [])[:30]:
        if not isinstance(claim, dict):
            continue
        anchor = _clip(claim.get("anchor"), 500)
        if not anchor:
            continue
        dissent = []
        for entry in (claim.get("dissent") or [])[:6]:
            if isinstance(entry, dict) and entry.get("model"):
                dissent.append({
                    "model": _clip(entry.get("model"), 40),
                    "quote": _clip(entry.get("quote"), 500),
                })
        claims.append({
            "anchor": anchor,
            "agree": _sanitize_str_list(claim.get("agree"), 40, 12),
            "dissent": dissent,
        })

    differences = []
    for diff in (data.get("differences") or [])[:20]:
        if not isinstance(diff, dict):
            continue
        claim_text = _clip(diff.get("claim"), 500)
        positions = []
        for position in (diff.get("positions") or [])[:6]:
            if not isinstance(position, dict):
                continue
            positions.append({
                "stance": _clip(position.get("stance"), 500),
                "models": _sanitize_str_list(position.get("models"), 40, 12),
                "quote": _clip(position.get("quote"), 500),
            })
        if not claim_text or not positions:
            continue
        differences.append({
            "claim": claim_text,
            "type": _clip(diff.get("type"), 40),
            "positions": positions,
            "verify": _clip(diff.get("verify"), 500),
        })

    return {
        "claims": claims,
        "differences": differences,
        "best_model": _clip(data.get("best_model"), 40),
        "models_compared": _sanitize_str_list(data.get("models_compared"), 40, 12),
    }


def compute_index_eligible(question, consensus_md, sources, included_models):
    question = str(question or "")
    return bool(
        len(str(consensus_md or "")) >= QUALITY_MIN_CONSENSUS_CHARS
        and len(sources or []) >= QUALITY_MIN_SOURCES
        and len(included_models or []) >= QUALITY_MIN_MODELS
        and QUALITY_QUESTION_MIN_CHARS <= len(question) <= QUALITY_QUESTION_MAX_CHARS
    )


def build_pending_result(uid, question, consensus_md, differences_data,
                         differences_text, model_sources, included_providers,
                         model_labels, consensus_model):
    """Baut das pending_results-Dokument; None, wenn Pflichtfelder fehlen."""
    question = _clip(question, MAX_QUESTION_CHARS)
    consensus_md = str(consensus_md or "").strip()
    if not uid or not question or not consensus_md:
        return None
    if len(consensus_md) > MAX_CONSENSUS_CHARS:
        consensus_md = consensus_md[:MAX_CONSENSUS_CHARS].rstrip() + "\n\n*[truncated]*"

    return {
        "schema_version": 1,
        "owner_uid": uid,
        "question": question,
        "consensus_md": consensus_md,
        "differences_data": sanitize_differences_data(differences_data),
        "differences_text": _clip(differences_text, MAX_DIFFERENCES_TEXT_CHARS),
        "sources": sanitize_sources(model_sources),
        "included_models": build_included_models(included_providers, model_labels),
        "consensus_model": _clip(consensus_model, 80),
        "answered_at": _utcnow().isoformat(),
    }


def save_pending_result(payload, db=None):
    db = db if db is not None else db_firestore
    result_id = generate_share_id()
    doc = dict(payload)
    doc["created_at"] = firestore.SERVER_TIMESTAMP
    doc["expires_at"] = _utcnow() + timedelta(hours=PENDING_TTL_HOURS)
    db.collection(PENDING_COLLECTION).document(result_id).set(doc)
    return result_id


def persist_pending_result(uid, question, consensus_md, differences_data,
                           differences_text, model_sources, included_providers,
                           model_labels, consensus_model, db=None):
    """Best-effort-Persistenz aus /consensus heraus: Fehler dürfen den
    Konsens-Stream nie beeinträchtigen, daher wird hier alles geschluckt."""
    try:
        payload = build_pending_result(
            uid, question, consensus_md, differences_data, differences_text,
            model_sources, included_providers, model_labels, consensus_model,
        )
        if payload is None:
            return None
        return save_pending_result(payload, db=db)
    except Exception:
        logging.exception("persist_pending_result failed")
        return None


def consume_daily_share_quota(uid, db=None, limit=SHARE_DAILY_LIMIT):
    """Firestore-Zähler pro UID und Tag (slowapi kann nur IP/in-memory)."""
    db = db if db is not None else db_firestore
    ref = db.collection("users").document(uid).collection("counters").document("shares_daily")
    today = _utcnow().strftime("%Y-%m-%d")
    transaction = db.transaction()

    @firestore.transactional
    def _consume(tx):
        snap = ref.get(transaction=tx)
        data = snap.to_dict() if snap.exists else {}
        data = data or {}
        count = data.get("count", 0) if data.get("date") == today else 0
        if not isinstance(count, int) or count < 0:
            count = 0
        if count >= limit:
            return False
        tx.set(ref, {"date": today, "count": count + 1})
        return True

    return _consume(transaction)


def get_share(share_id, db=None):
    if not is_valid_share_id(share_id):
        return None
    db = db if db is not None else db_firestore
    snap = db.collection(SHARES_COLLECTION).document(share_id).get()
    if not snap.exists:
        return None
    return snap.to_dict() or {}


def create_share_from_pending(uid, result_id, db=None, consume_quota=None):
    """Kopiert ein Pending-Ergebnis in einen unveränderlichen Share-Snapshot.

    Wirft ShareError mit code in {"not_found", "forbidden", "quota_exceeded"}.
    Idempotent: Wiederholtes Teilen desselben Ergebnisses liefert den
    bestehenden aktiven Link zurück, ohne das Tageskontingent zu belasten.
    """
    db = db if db is not None else db_firestore
    if not is_valid_share_id(result_id):
        raise ShareError("not_found", "Result not found or expired.")

    pending_ref = db.collection(PENDING_COLLECTION).document(result_id)
    snap = pending_ref.get()
    if not snap.exists:
        raise ShareError("not_found", "Result not found or expired.")
    pending = snap.to_dict() or {}

    expires_at = pending.get("expires_at")
    if isinstance(expires_at, datetime) and expires_at < _utcnow():
        try:
            pending_ref.delete()
        except Exception:
            logging.exception("expired pending_result cleanup failed")
        raise ShareError("not_found", "Result not found or expired.")

    if pending.get("owner_uid") != uid:
        raise ShareError("forbidden", "You can only share your own results.")

    existing_id = pending.get("share_id")
    if existing_id:
        existing = get_share(existing_id, db=db)
        if existing is not None and existing.get("status") == "active":
            return {"share_id": existing_id, "slug": existing.get("slug") or "", "created": False}

    if consume_quota is None:
        def consume_quota():
            return consume_daily_share_quota(uid, db=db)
    if not consume_quota():
        raise ShareError("quota_exceeded", "Daily share limit reached. Please try again tomorrow.")

    question = pending.get("question") or ""
    consensus_md = pending.get("consensus_md") or ""
    sources = pending.get("sources") or []
    included_models = pending.get("included_models") or []
    share_id = generate_share_id()
    slug = slugify_question(question)

    share_doc = {
        "schema_version": 1,
        "slug": slug,
        "status": "active",
        "question": question,
        "consensus_md": consensus_md,
        "differences_data": pending.get("differences_data"),
        "differences_text": pending.get("differences_text") or "",
        "sources": sources,
        "included_models": included_models,
        "consensus_model": pending.get("consensus_model") or "",
        "answered_at": pending.get("answered_at") or "",
        "created_at": firestore.SERVER_TIMESTAMP,
        "question_hash": question_hash(question),
        "owner_uid": uid,
        "source_result_id": result_id,
        "index_eligible": compute_index_eligible(question, consensus_md, sources, included_models),
        "indexed": False,
        "reports_count": 0,
    }
    db.collection(SHARES_COLLECTION).document(share_id).set(share_doc)

    try:
        pending_ref.update({"share_id": share_id})
    except Exception:
        logging.exception("pending_result share_id backlink failed")

    return {"share_id": share_id, "slug": slug, "created": True}


def revoke_share(share_id, uid, is_admin=False, db=None):
    db = db if db is not None else db_firestore
    data = get_share(share_id, db=db)
    if data is None:
        raise ShareError("not_found", "Share not found.")
    if data.get("owner_uid") != uid and not is_admin:
        raise ShareError("forbidden", "You can only revoke your own shares.")
    db.collection(SHARES_COLLECTION).document(share_id).update({
        "status": "revoked",
        "revoked_at": firestore.SERVER_TIMESTAMP,
    })


REPORT_REASONS = ("inaccurate", "harmful", "spam", "copyright", "other")


def report_share(share_id, reason, db=None):
    """Besucher-Report: Zähler + Grund-Aggregat, bewusst ohne IP/UA-Speicherung.

    Read-modify-write statt Transaktion: bei Reports ist ein verlorenes
    Increment unter Race-Bedingungen verschmerzbar. Auto-noindex ab 5 Reports
    und die Admin-Review-Priorisierung folgen in Etappe 3.
    """
    db = db if db is not None else db_firestore
    data = get_share(share_id, db=db)
    if data is None or data.get("status") != "active":
        raise ShareError("not_found", "Share not found.")
    if reason not in REPORT_REASONS:
        reason = "other"

    reasons = data.get("report_reasons")
    reasons = dict(reasons) if isinstance(reasons, dict) else {}
    reasons[reason] = (reasons.get(reason) or 0) + 1
    count = data.get("reports_count")
    count = count + 1 if isinstance(count, int) and count >= 0 else 1

    db.collection(SHARES_COLLECTION).document(share_id).update({
        "reports_count": count,
        "report_reasons": reasons,
        "last_reported_at": firestore.SERVER_TIMESTAMP,
    })
    return count


def list_shares_for_owner(uid, db=None, max_items=200):
    db = db if db is not None else db_firestore
    docs = db.collection(SHARES_COLLECTION).where("owner_uid", "==", uid).stream()
    shares = []
    for doc in docs:
        data = doc.to_dict() or {}
        created_at = data.get("created_at")
        shares.append({
            "share_id": doc.id,
            "path": share_path(data.get("slug") or "", doc.id),
            "question": _clip(data.get("question"), 200),
            "status": data.get("status") or "active",
            "created_at": created_at.isoformat() if isinstance(created_at, datetime) else "",
        })
        if len(shares) >= max_items:
            break
    shares.sort(key=lambda item: item["created_at"], reverse=True)
    return shares


def public_share_payload(data):
    """Whitelist-Serializer für alles, was die öffentliche Seite sehen darf.

    Bewusst KEIN doc.to_dict()-Durchreichen: owner_uid, reports_count,
    indexed/index_eligible, source_result_id usw. bleiben serverintern.
    """
    created_at = data.get("created_at")
    return {
        "slug": data.get("slug") or "",
        "question": data.get("question") or "",
        "consensus_md": data.get("consensus_md") or "",
        "differences_data": data.get("differences_data"),
        "differences_text": data.get("differences_text") or "",
        "sources": data.get("sources") or [],
        "included_models": data.get("included_models") or [],
        "consensus_model": data.get("consensus_model") or "",
        "answered_at": data.get("answered_at") or "",
        "created_at": created_at.isoformat() if isinstance(created_at, datetime) else "",
    }


def build_citation(payload, canonical_url):
    """Server-Pendant zu buildConsensusCitation() – mit kanonischer Share-URL."""
    included_models = payload.get("included_models") or []
    if not included_models:
        return ""

    date_iso = payload.get("answered_at") or payload.get("created_at") or ""
    date_str = date_iso[:10] if len(date_iso) >= 10 else _utcnow().strftime("%Y-%m-%d")

    parts = ["consens.io. (%s)." % date_str]
    question = (payload.get("question") or "").strip()
    parts.append('Consensus answer to "%s".' % question if question else "Consensus answer.")
    parts.append("Models consulted: %s." % ", ".join(included_models))
    consensus_model = (payload.get("consensus_model") or "").strip()
    if consensus_model:
        parts.append("Consensus model: %s." % consensus_model)
    links = [s.get("url") for s in (payload.get("sources") or []) if s.get("url")]
    if links:
        parts.append("Sources: %s" % ", ".join(links))
    parts.append("Retrieved from %s" % canonical_url)
    return " ".join(parts)


def cleanup_expired_pending(db=None, max_docs=500):
    """Aufräum-Fallback beim App-Start (zusätzlich zur Firestore-TTL-Policy)."""
    db = db if db is not None else db_firestore
    docs = (
        db.collection(PENDING_COLLECTION)
        .where("expires_at", "<", _utcnow())
        .limit(max_docs)
        .stream()
    )
    deleted = 0
    for doc in docs:
        doc.reference.delete()
        deleted += 1
    if deleted:
        logging.info("cleanup_expired_pending: removed %d expired pending results", deleted)
    return deleted
