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

from cachetools import TTLCache
from firebase_admin import firestore

import app.core.config as cfg
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

# Ab so vielen Reports wird ein indexierter Share automatisch auf noindex
# gesetzt und für die Admin-Review priorisiert (kein Auto-Unpublish).
AUTO_NOINDEX_REPORTS = 5

# Hard-Delete-Frist für widerrufene Shares (DSGVO-Zusage in den Rechtstexten).
REVOKED_RETENTION_DAYS = 30

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
# Provider-Logos analog zum Model-Picker auf /app (static/icons/chat_icons/).
PROVIDER_ICONS = {
    "OpenAI": "chatgpt.png",
    "Mistral": "mistral.png",
    "Anthropic": "claude.png",
    "Gemini": "gemini-icon.png",
    "DeepSeek": "deepseek.png",
    "Grok": "grok.png",
}
# Reverse-Map: Zitations-Label -> Provider-Key (Labels sind eindeutig).
_CITATION_LABEL_TO_PROVIDER = {v: k for k, v in PROVIDER_CITATION_LABELS.items()}

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
        # Der Option-Text aus dem Frontend trägt teils ein Badge-Suffix
        # (" · New", "• Pro"). Für die Zitation nur den reinen Modellnamen
        # behalten – sonst scheitert die Whitelist am "·" und der Name fiele
        # ganz auf den Provider-Namen zurück.
        cleaned = re.split(r"\s*[·•]\s*", cleaned, maxsplit=1)[0].strip()
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


# Konsens-Engine-Key -> konkretes Modell (deterministisch, analog zu
# consensus_engine.run_consensus). Der Snapshot speichert nur den Engine-Key,
# das konkrete Modell wird hier rekonstruiert – gilt auch für Alt-Snapshots.
_CONSENSUS_ENGINE_MODELS = {
    "OpenAI": cfg.DEFAULT_OPENAI_MODEL,
    "OpenAI-Pro": "gpt-5.5",
    "Mistral": cfg.DEFAULT_MISTRAL_MODEL,
    "Mistral-Pro": cfg.MISTRAL_PRO_MODEL,
    "Anthropic": cfg.DEFAULT_ANTHROPIC_MODEL,
    "Anthropic-Pro": cfg.ANTHROPIC_PRO_MODEL,
    "Gemini": cfg.GEMINI_FLASH_MODEL,
    "Gemini-Pro": cfg.GEMINI_PRO_MODEL,
    cfg.GEMINI_FRONTIER_LOW_MODEL: cfg.GEMINI_FRONTIER_LOW_MODEL,
    "DeepSeek": cfg.DEFAULT_DEEPSEEK_MODEL,
    "DeepSeek-Pro": "deepseek-v4-pro",
    "Grok": cfg.DEFAULT_GROK_MODEL,
    "Grok-Pro": "grok-4.3",
}


def consensus_model_view(consensus_model):
    """Strukturierte Ansicht des Konsens-Modells (Icon + konkretes Modell).

    Der gespeicherte Wert ist ein Engine-Key wie "Anthropic" oder
    "Gemini-Pro"; daraus werden Provider (für das Logo), das konkrete Modell
    (lesbares Label) und das Pro-Flag rekonstruiert. Unbekannte Werte bleiben
    ohne Icon erhalten.
    """
    raw = str(consensus_model or "").strip()
    if not raw:
        return None
    is_pro = raw.endswith("-Pro")
    base = raw[:-4] if is_pro else raw
    provider = base if base in PROVIDER_ICONS else None
    if provider is None:
        low = raw.lower()
        for key in PROVIDER_ICONS:
            if key.lower() in low:
                provider = key
                break
    icon = PROVIDER_ICONS.get(provider) if provider else None
    model_id = _CONSENSUS_ENGINE_MODELS.get(raw)
    model_label = cfg.get_model_label(model_id) if model_id else ""
    return {
        "provider": provider or "",
        "label": PROVIDER_CITATION_LABELS.get(provider) if provider else raw,
        "model": model_label,
        "icon": ("/static/icons/chat_icons/%s" % icon) if icon else "",
        "pro": is_pro,
    }


def consulted_models_view(included_models):
    """Strukturierte Ansicht der konsultierten Modelle für die Share-Seite.

    Macht aus den flachen Zitations-Strings ("Anthropic Claude: opus-4")
    wieder Provider-Label, Modellname und passendes Logo, damit das Template
    je Anbieter ein Icon-Chip rendern kann. Unbekannte Einträge bleiben ohne
    Icon erhalten (graceful fallback).
    """
    view = []
    for entry in included_models or []:
        label, _, model_name = str(entry).partition(": ")
        label = label.strip()
        provider = _CITATION_LABEL_TO_PROVIDER.get(label)
        icon = PROVIDER_ICONS.get(provider) if provider else None
        view.append({
            "provider": provider or "",
            "label": label,
            "model": model_name.strip(),
            "icon": ("/static/icons/chat_icons/%s" % icon) if icon else "",
        })
    return view


def _sanitize_str_list(value, item_limit, max_items):
    if not isinstance(value, list):
        return []
    return [_clip(item, item_limit) for item in value[:max_items] if isinstance(item, str) and item.strip()]


def _coerce_bounded_int(value, lo, hi):
    if isinstance(value, bool):
        return lo
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return lo
    return max(lo, min(hi, parsed))


_RESOLVE_OUTCOMES = {"resolved", "standoff", "mutual_revision"}
_RESOLVE_DECISIONS = {"maintain", "revise", "error"}


def _sanitize_resolution(raw):
    """Whitelist der persistierten Resolve-Runde eines Widerspruchs.

    Übernimmt nur bekannte Outcomes/Decisions mit gekappten Texten; ohne
    verwertbares Ergebnis wird nichts persistiert (None)."""
    if not isinstance(raw, dict):
        return None
    outcome = str(raw.get("outcome") or "").strip()
    if outcome not in _RESOLVE_OUTCOMES:
        return None
    results = []
    for entry in (raw.get("results") or [])[:6]:
        if not isinstance(entry, dict) or not entry.get("model"):
            continue
        decision = str(entry.get("decision") or "").strip()
        if decision not in _RESOLVE_DECISIONS:
            decision = "error"
        results.append({
            "model": _clip(entry.get("model"), 40),
            "decision": decision,
            "position": _clip(entry.get("position"), 500),
            "reason": _clip(entry.get("reason"), 500),
        })
    if not results:
        return None
    return {"outcome": outcome, "results": results}


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
        entry = {
            "claim": claim_text,
            "type": _clip(diff.get("type"), 40),
            "severity": _clip(diff.get("severity"), 20),
            "positions": positions,
            "verify": _clip(diff.get("verify"), 500),
        }
        # Ergebnis einer Resolve-Runde (Pro): bleibt am Widerspruch hängen,
        # damit Bookmarks den gelösten Zustand wieder anzeigen können.
        resolution = _sanitize_resolution(diff.get("resolution"))
        if resolution:
            entry["resolution"] = resolution
        differences.append(entry)

    result = {
        "claims": claims,
        "differences": differences,
        "best_model": _clip(data.get("best_model"), 40),
        "models_compared": _sanitize_str_list(data.get("models_compared"), 40, 12),
    }

    agreement = data.get("agreement")
    if isinstance(agreement, dict):
        result["agreement"] = {
            "score": _coerce_bounded_int(agreement.get("score"), 0, 100),
            "level": _clip(agreement.get("level"), 20),
            "model_count": _coerce_bounded_int(agreement.get("model_count"), 0, 12),
            "major_contradictions": _coerce_bounded_int(agreement.get("major_contradictions"), 0, 50),
            "minor_contradictions": _coerce_bounded_int(agreement.get("minor_contradictions"), 0, 50),
            "emphases": _coerce_bounded_int(agreement.get("emphases"), 0, 50),
        }

    # Judge-Metadaten (Provider/Modell/Stufe, keine Texte): bleiben im
    # Snapshot, damit Bookmarks/Shares die Judge-Fußnote anzeigen können.
    judges = data.get("judges")
    if isinstance(judges, dict):
        sanitized_judges = {}
        for role in ("differences", "adjudicator"):
            entry = judges.get(role)
            if isinstance(entry, dict) and entry.get("provider"):
                sanitized_judges[role] = {
                    "provider": _clip(entry.get("provider"), 40),
                    "model": _clip(entry.get("model"), 80),
                    "tier": _clip(entry.get("tier"), 20),
                }
        if sanitized_judges:
            result["judges"] = sanitized_judges

    return result


def _quality_limit(key, fallback):
    try:
        value = int(cfg.LIMITS.get(key, fallback))
    except (TypeError, ValueError):
        return fallback
    return value if value >= 0 else fallback


def compute_index_eligible(question, consensus_md, sources, included_models):
    """Qualitätsfilter aus der limits-Config (Admin-UI); nur Eligibility-
    Anzeige – ``indexed`` setzt ausschließlich der Admin."""
    question = str(question or "")
    return bool(
        len(str(consensus_md or "")) >= _quality_limit("share_min_consensus_chars", 600)
        and len(sources or []) >= _quality_limit("share_min_sources", 2)
        and len(included_models or []) >= _quality_limit("share_min_models", 3)
        and _quality_limit("share_question_min_chars", 15)
            <= len(question)
            <= _quality_limit("share_question_max_chars", 300)
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


# In-Process-TTL-Cache für die öffentliche /s/-Seite (kein CDN, eine Render-
# Instanz). Misses werden nicht gecacht, damit frisch erstellte Shares sofort
# sichtbar sind; Revoke/Block/Auto-noindex invalidieren explizit.
SHARE_CACHE_TTL_SECONDS = 300
_share_cache = TTLCache(maxsize=1024, ttl=SHARE_CACHE_TTL_SECONDS)
# Dedup-Canonical-Lookups (question_hash -> Ziel oder None) separat cachen;
# jede Moderation kann Canonical-Ziele ändern, daher wird er mit invalidiert.
_canonical_cache = TTLCache(maxsize=1024, ttl=SHARE_CACHE_TTL_SECONDS)
# "Verwandte Fragen"-Vorschläge (share_id -> Liste). Längere TTL, da nur aus
# dem selten wechselnden Index-Set gespeist; bei Moderation mit-invalidiert.
RELATED_CACHE_TTL_SECONDS = 900
_related_cache = TTLCache(maxsize=512, ttl=RELATED_CACHE_TTL_SECONDS)


def get_share_cached(share_id, db=None):
    if share_id in _share_cache:
        return _share_cache[share_id]
    data = get_share(share_id, db=db)
    if data is not None:
        _share_cache[share_id] = data
    return data


def invalidate_share_cache(share_id=None):
    if share_id is None:
        _share_cache.clear()
    else:
        _share_cache.pop(share_id, None)
    _canonical_cache.clear()
    _related_cache.clear()


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
        "indexed": False,
        "revoked_at": firestore.SERVER_TIMESTAMP,
    })
    # Scheduler-Metadaten sofort entfernen; kompakte History bleibt bis zum
    # regulaeren Share-Hard-Delete erhalten.
    from app.services.watch_service import delete_watches_for_share
    delete_watches_for_share(share_id, db=db)
    invalidate_share_cache(share_id)


MODERATION_ACTIONS = ("block", "unblock")


def moderate_share(share_id, action=None, indexed=None, db=None):
    """Admin-Moderation: Status blocken/freigeben und/oder ``indexed`` setzen.

    ``indexed`` wird ausschließlich hier (also durch den Admin) auf True
    gesetzt – nirgends sonst im Code. Jede Moderation gilt als Review und
    nimmt den Share aus der priorisierten Review-Liste.
    """
    db = db if db is not None else db_firestore
    data = get_share(share_id, db=db)
    if data is None:
        raise ShareError("not_found", "Share not found.")
    if action is not None and action not in MODERATION_ACTIONS:
        raise ShareError("bad_request", "Unknown moderation action.")
    if action is None and indexed is None:
        raise ShareError("bad_request", "Nothing to moderate: pass action and/or indexed.")

    status = data.get("status")
    updates = {"needs_review": False, "reviewed_at": firestore.SERVER_TIMESTAMP}
    if action == "block":
        if status == "revoked":
            raise ShareError("bad_request", "Share is already revoked.")
        updates["status"] = "blocked"
        updates["indexed"] = False
        updates["blocked_at"] = firestore.SERVER_TIMESTAMP
        status = "blocked"
    elif action == "unblock":
        if status != "blocked":
            raise ShareError("bad_request", "Only blocked shares can be unblocked.")
        updates["status"] = "active"
        status = "active"

    if indexed is not None:
        if bool(indexed) and status != "active":
            raise ShareError("bad_request", "Only active shares can be indexed.")
        updates["indexed"] = bool(indexed)

    db.collection(SHARES_COLLECTION).document(share_id).update(updates)
    invalidate_share_cache(share_id)
    merged = dict(data)
    merged.update(updates)
    return merged


def list_shares_for_admin(db=None, only_reported=False, max_items=500):
    """Moderationsliste: priorisiert needs_review, dann Report-Anzahl.

    Bewusst ein Collection-Scan mit Limit statt Firestore-Range-Query –
    das Share-Volumen ist klein (privates Projekt, 20/Tag/UID-Quota).
    """
    db = db if db is not None else db_firestore
    docs = db.collection(SHARES_COLLECTION).limit(max_items).stream()
    shares = []
    for doc in docs:
        data = doc.to_dict() or {}
        reports = data.get("reports_count")
        reports = reports if isinstance(reports, int) and reports > 0 else 0
        if only_reported and not reports:
            continue
        created_at = data.get("created_at")
        last_reported = data.get("last_reported_at")
        shares.append({
            "share_id": doc.id,
            "path": share_path(data.get("slug") or "", doc.id),
            "question": _clip(data.get("question"), 200),
            "status": data.get("status") or "active",
            "owner_uid": data.get("owner_uid") or "",
            "reports_count": reports,
            "report_reasons": data.get("report_reasons") if isinstance(data.get("report_reasons"), dict) else {},
            "needs_review": bool(data.get("needs_review")),
            "indexed": bool(data.get("indexed")),
            "index_eligible": bool(data.get("index_eligible")),
            "created_at": created_at.isoformat() if isinstance(created_at, datetime) else "",
            "last_reported_at": last_reported.isoformat() if isinstance(last_reported, datetime) else "",
        })
    shares.sort(key=lambda item: (
        not item["needs_review"],
        -item["reports_count"],
        item["created_at"],
    ))
    return shares


def cleanup_revoked_shares(db=None, max_docs=500):
    """30-Tage-Hard-Delete für widerrufene Shares (Aufruf beim App-Start,
    kein eigener Scheduler – der tägliche Render-Restart reicht)."""
    db = db if db is not None else db_firestore
    cutoff = _utcnow() - timedelta(days=REVOKED_RETENTION_DAYS)
    docs = (
        db.collection(SHARES_COLLECTION)
        .where("status", "==", "revoked")
        .limit(max_docs)
        .stream()
    )
    deleted = 0
    for doc in docs:
        data = doc.to_dict() or {}
        revoked_at = data.get("revoked_at")
        if isinstance(revoked_at, datetime) and revoked_at < cutoff:
            # Alte Unit-Test-Doubles modellieren keine Subcollections; der
            # reale Firestore-DocumentReference tut es.
            if hasattr(doc.reference, "collection"):
                for history_doc in doc.reference.collection("watch_history").stream():
                    history_doc.reference.delete()
            doc.reference.delete()
            invalidate_share_cache(doc.id)
            deleted += 1
    if deleted:
        logging.info("cleanup_revoked_shares: hard-deleted %d revoked shares", deleted)
    return deleted


def find_canonical_share(question_hash_value, db=None, max_candidates=50):
    """Dedup-Ziel für rel=canonical: ältester aktiver UND indexierter Share
    mit derselben normalisierten Frage. None, wenn es keinen gibt – ein
    Canonical darf nie auf eine noindex-Seite zeigen."""
    if not question_hash_value:
        return None
    use_cache = db is None
    if use_cache and question_hash_value in _canonical_cache:
        return _canonical_cache[question_hash_value]
    db = db if db is not None else db_firestore
    docs = (
        db.collection(SHARES_COLLECTION)
        .where("question_hash", "==", question_hash_value)
        .stream()
    )
    best = None
    for doc in docs:
        data = doc.to_dict() or {}
        if data.get("status") != "active" or not data.get("indexed"):
            continue
        created_at = data.get("created_at")
        created_key = created_at.isoformat() if isinstance(created_at, datetime) else "9999"
        candidate = {"share_id": doc.id, "slug": data.get("slug") or "", "created_key": created_key}
        if best is None or candidate["created_key"] < best["created_key"]:
            best = candidate
        max_candidates -= 1
        if max_candidates <= 0:
            break
    if use_cache:
        _canonical_cache[question_hash_value] = best
    return best


def list_indexed_share_urls(db=None, max_items=1000):
    """URLs für sitemap-shares.xml: nur indexed == True und status == active."""
    db = db if db is not None else db_firestore
    docs = (
        db.collection(SHARES_COLLECTION)
        .where("indexed", "==", True)
        .limit(max_items)
        .stream()
    )
    urls = []
    for doc in docs:
        data = doc.to_dict() or {}
        if data.get("status") != "active":
            continue
        created_at = data.get("created_at")
        urls.append({
            "path": share_path(data.get("slug") or "", doc.id),
            "lastmod": created_at.strftime("%Y-%m-%d") if isinstance(created_at, datetime) else "",
        })
    urls.sort(key=lambda item: item["path"])
    return urls


# Häufige Füllwörter (EN/DE), die für die Themen-Ähnlichkeit nichts beitragen.
_RELATED_STOPWORDS = frozenset("""
the and for are was were can could should would will what which who how why when
where does did with without from into about over under not yes vs versus this that
these those your our their its his her been being have has had any all more most
der die das und oder ist sind war wie warum wann wer wem wen ein eine einen einem
einer fuer von zu im in am an mit ohne auf ueber unter nicht kein keine ja auch
""".split())


def _question_tokens(question):
    """Themen-Tokens einer Frage für die Ähnlichkeitswertung (lower, entstoppt)."""
    tokens = re.findall(r"[a-z0-9]+", str(question or "").lower())
    return {t for t in tokens if len(t) >= 3 and t not in _RELATED_STOPWORDS}


def list_related_shares(exclude_share_id, question, db=None, limit=4, scan_limit=400):
    """Vorschläge "verwandte Fragen" für die öffentliche Share-Seite.

    Bewusst NUR indexierte UND aktive Shares (admin-freigegeben) – nie
    noindex/private/gesperrte Snapshots. Relevanz über Token-Überlappung der
    Frage, Tie-Break und Fallback über Aktualität. Read-only, gecacht.
    """
    use_cache = db is None and bool(exclude_share_id)
    if use_cache and exclude_share_id in _related_cache:
        return _related_cache[exclude_share_id]

    db = db if db is not None else db_firestore
    query_tokens = _question_tokens(question)
    docs = (
        db.collection(SHARES_COLLECTION)
        .where("indexed", "==", True)
        .limit(scan_limit)
        .stream()
    )

    candidates = []
    for doc in docs:
        if doc.id == exclude_share_id:
            continue
        data = doc.to_dict() or {}
        if data.get("status") != "active":
            continue
        candidate_question = data.get("question") or ""
        overlap = len(query_tokens & _question_tokens(candidate_question)) if query_tokens else 0
        created_at = data.get("created_at")
        created_key = created_at.isoformat() if isinstance(created_at, datetime) else ""
        candidates.append({
            "path": share_path(data.get("slug") or "", doc.id),
            "question": _clip(candidate_question, 200),
            "models_count": len(data.get("included_models") or []),
            "overlap": overlap,
            "created_key": created_key,
        })

    # Beste Überlappung zuerst, bei Gleichstand die neuesten.
    candidates.sort(key=lambda item: (item["overlap"], item["created_key"]), reverse=True)
    related = [
        {"path": c["path"], "question": c["question"], "models_count": c["models_count"]}
        for c in candidates[:limit]
    ]
    if use_cache:
        _related_cache[exclude_share_id] = related
    return related


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

    updates = {
        "reports_count": count,
        "report_reasons": reasons,
        "last_reported_at": firestore.SERVER_TIMESTAMP,
    }
    # Ab der Schwelle: kein Auto-Unpublish, aber raus aus dem Index und
    # priorisiert in die Admin-Review (Re-Indexierung nur durch den Admin).
    if count >= AUTO_NOINDEX_REPORTS:
        updates["needs_review"] = True
        if data.get("indexed"):
            updates["indexed"] = False
            logging.warning(
                "report_share: auto-noindex for %s after %d reports", share_id, count
            )

    db.collection(SHARES_COLLECTION).document(share_id).update(updates)
    if "indexed" in updates:
        invalidate_share_cache(share_id)
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


def list_watch_history(share_id, db=None, max_items=100):
    """Whitelist compact public history points; never expose rerun text."""
    db = db if db is not None else db_firestore
    ref = db.collection(SHARES_COLLECTION).document(share_id).collection("watch_history")
    points = []
    for doc in ref.stream():
        data = doc.to_dict() or {}
        ts = data.get("ts")
        score = data.get("agreement_score")
        if not isinstance(ts, datetime) or not isinstance(score, (int, float)):
            continue
        points.append({
            "ts": ts,
            "agreement_score": max(0, min(100, int(score))),
            "verdict": _clip(data.get("verdict"), 80),
            "changed": bool(data.get("changed")),
            "severity": _clip(data.get("severity"), 10),
            "change_summary": _clip(data.get("change_summary"), 400),
        })
    points.sort(key=lambda item: item["ts"])
    return points[-max_items:]


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
