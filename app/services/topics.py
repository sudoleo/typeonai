"""Curated public Topics with immutable, versioned consensus snapshots.

Topics are deliberately independent from user-owned shares and Consensus
Watches. A topic document contains editorial configuration and denormalized
latest-state fields; every published run lives in ``topics/{id}/runs`` and is
never modified after creation.
"""

from __future__ import annotations

import hashlib
import re
import secrets
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

import app.core.config as cfg
from app.core.security import db_firestore
from app.services.watch_service import WatchError, parse_token_payload, sign_token_payload


TOPICS_COLLECTION = "topics"
FOLLOWERS_COLLECTION = "topic_followers"
DELIVERIES_COLLECTION = "topic_follower_deliveries"
MAX_FOLLOWERS_PER_TOPIC = 500
CONFIRM_TOKEN_MAX_AGE_DAYS = 3

TOPIC_STATUSES = {"active", "paused", "archived"}
UPDATE_INTERVALS = {"manual", "daily", "weekly", "biweekly", "monthly"}
CHANGE_TYPES = {"stable", "minor", "major"}
EVIDENCE_TYPES = {"x", "official", "github", "documentation", "press"}
PROVIDER_ORDER = ("openai", "mistral", "anthropic", "gemini", "deepseek", "grok")
PROVIDER_LABELS = {
    "openai": "OpenAI",
    "mistral": "Mistral",
    "anthropic": "Anthropic",
    "gemini": "Google Gemini",
    "deepseek": "DeepSeek",
    "grok": "Grok",
}
TOPIC_RUN_LEASE_MINUTES = 30
_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]{2,}$")


class TopicError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _clean(value, *, limit: int, required: bool = False, label: str = "Value") -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if required and not text:
        raise TopicError("bad_request", f"{label} is required.")
    if len(text) > limit:
        raise TopicError("bad_request", f"{label} must be at most {limit} characters.")
    return text


def _clean_multiline(value, *, limit: int, label: str) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(text) > limit:
        raise TopicError("bad_request", f"{label} must be at most {limit} characters.")
    return text


def normalize_slug(value) -> str:
    slug = str(value or "").strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")
    if not slug or len(slug) > 80 or not _SLUG_RE.fullmatch(slug):
        raise TopicError("bad_request", "Enter a valid topic slug.")
    return slug


def _iso(value) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    return str(value or "")


def _public_datetime(value):
    return _iso(value) if value else ""


def _valid_url(value) -> str:
    url = str(value or "").strip()
    try:
        parsed = urlsplit(url)
    except ValueError:
        parsed = None
    if (
        not parsed
        or parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or len(url) > 1500
    ):
        raise TopicError("bad_request", "Each evidence item needs a valid http(s) URL.")
    return url


def normalize_models(value) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TopicError("bad_request", "Models must be a list.")
    models = []
    for item in value[:20]:
        model = _clean(item, limit=120, label="Model")
        if model and model not in models:
            models.append(model)
    return models


def _default_provider_models() -> dict[str, str]:
    configured = cfg.get_watch_models(False)
    return {
        provider: model
        for provider in PROVIDER_ORDER
        if (model := cfg.canonical_model_id(configured.get(provider), provider))
        and model in cfg.get_ordered_models(provider)
    }


def normalize_provider_models(value, *, fallback=None) -> dict[str, str]:
    """Validate the explicit answer-model plan stored with a Topic.

    Older Topic drafts stored free-form display labels in ``models``. They
    remain readable, while new/updated Topics get a concrete provider->model
    mapping so the Run button has an executable configuration.
    """
    if value is None:
        value = fallback if isinstance(fallback, dict) else _default_provider_models()
    if not isinstance(value, dict):
        raise TopicError("bad_request", "Topic answer models must be an object.")
    selected = {}
    for provider in PROVIDER_ORDER:
        raw = value.get(provider)
        if raw in (None, ""):
            continue
        model = cfg.canonical_model_id(raw, provider)
        if model not in cfg.get_ordered_models(provider):
            raise TopicError(
                "bad_request",
                f"{PROVIDER_LABELS[provider]} model is not available.",
            )
        selected[provider] = model
    if len(selected) < 2:
        raise TopicError("bad_request", "Select at least two Topic answer models.")
    return selected


def topic_model_labels(provider_models: dict) -> list[str]:
    return [
        f"{PROVIDER_LABELS[provider]} · {cfg.get_model_label(model)}"
        for provider, model in provider_models.items()
        if provider in PROVIDER_LABELS
    ]


def normalize_run_config(value, *, previous=None) -> dict:
    data = value if isinstance(value, dict) else {}
    prior = previous if isinstance(previous, dict) else {}
    provider_models = normalize_provider_models(
        data.get("provider_models"),
        fallback=prior.get("provider_models"),
    )
    return {
        "provider_models": provider_models,
        # Topic runs always ask the provider tools to research current sources.
        # Keeping this explicit makes the Firestore document self-explanatory.
        "collect_sources": True,
    }


def next_run_at(update_interval: str, *, now=None):
    now = now or utcnow()
    days = {
        "daily": 1,
        "weekly": 7,
        "biweekly": 14,
        "monthly": 30,
    }.get(str(update_interval or ""))
    return now + timedelta(days=days) if days else None


def normalize_source_rules(value) -> dict:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise TopicError("bad_request", "Source rules must be an object.")
    allowed = value.get("allowed_types") or list(EVIDENCE_TYPES)
    if not isinstance(allowed, list):
        raise TopicError("bad_request", "Allowed source types must be a list.")
    allowed_types = []
    for item in allowed:
        item = str(item or "").strip().lower()
        if item not in EVIDENCE_TYPES:
            raise TopicError("bad_request", f"Unsupported evidence type: {item}")
        if item not in allowed_types:
            allowed_types.append(item)
    domains = value.get("preferred_domains") or []
    if not isinstance(domains, list):
        raise TopicError("bad_request", "Preferred domains must be a list.")
    preferred_domains = []
    for item in domains[:30]:
        domain = _clean(item, limit=120, label="Preferred domain").lower()
        domain = re.sub(r"^https?://", "", domain).strip("/")
        if domain and domain not in preferred_domains:
            preferred_domains.append(domain)
    return {
        "allowed_types": allowed_types,
        "preferred_domains": preferred_domains,
        "notes": _clean_multiline(value.get("notes"), limit=3000, label="Source rules"),
    }


def normalize_evidence(value) -> list[dict]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TopicError("bad_request", "Evidence must be a list.")
    evidence = []
    seen = set()
    for raw in value[:80]:
        if not isinstance(raw, dict):
            raise TopicError("bad_request", "Each evidence item must be an object.")
        url = _valid_url(raw.get("url"))
        if url in seen:
            continue
        kind = str(raw.get("type") or "").strip().lower()
        if kind not in EVIDENCE_TYPES:
            raise TopicError("bad_request", f"Unsupported evidence type: {kind}")
        evidence.append({
            "id": f"S{len(evidence) + 1}",
            "type": kind,
            "title": _clean(raw.get("title"), limit=240, required=True, label="Evidence title"),
            "url": url,
            "publisher": _clean(raw.get("publisher"), limit=120, label="Publisher"),
            "published_at": _clean(raw.get("published_at"), limit=40, label="Evidence date"),
            "excerpt": _clean_multiline(raw.get("excerpt"), limit=600, label="Evidence excerpt"),
        })
        seen.add(url)
    return evidence


def normalize_opinion_changes(value) -> list[dict]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TopicError("bad_request", "Opinion changes must be a list.")
    changes = []
    for raw in value[:30]:
        if not isinstance(raw, dict):
            raise TopicError("bad_request", "Each opinion change must be an object.")
        summary = _clean(raw.get("summary"), limit=400, required=True, label="Opinion change")
        changes.append({
            "model": _clean(raw.get("model"), limit=120, label="Model"),
            "from": _clean(raw.get("from"), limit=180, label="Previous position"),
            "to": _clean(raw.get("to"), limit=180, label="New position"),
            "summary": summary,
        })
    return changes


def normalize_topic_input(data: dict, *, existing: dict | None = None) -> dict:
    if not isinstance(data, dict):
        raise TopicError("bad_request", "Topic data must be an object.")
    previous = existing or {}
    title = _clean(
        data.get("title", previous.get("title")),
        limit=140,
        required=True,
        label="Topic title",
    )
    slug = normalize_slug(data.get("slug", previous.get("slug") or title))
    status = str(data.get("status", previous.get("status") or "active")).lower()
    if status not in TOPIC_STATUSES:
        raise TopicError("bad_request", "Invalid topic status.")
    interval = str(
        data.get("update_interval", previous.get("update_interval") or "weekly")
    ).lower()
    if interval not in UPDATE_INTERVALS:
        raise TopicError("bad_request", "Invalid update interval.")
    source_rules = normalize_source_rules(
        data.get("source_rules", previous.get("source_rules") or {})
    )
    run_config = normalize_run_config(
        data.get("run_config"),
        previous=previous.get("run_config"),
    )
    evidence = normalize_evidence(
        data.get("evidence", previous.get("evidence") or [])
    )
    disallowed = sorted({
        item["type"] for item in evidence
        if item["type"] not in source_rules["allowed_types"]
    })
    if disallowed:
        raise TopicError(
            "bad_request",
            "Evidence uses source types disabled by the Topic rules: "
            + ", ".join(disallowed),
        )
    return {
        "title": title,
        "slug": slug,
        "lead_question": _clean(
            data.get("lead_question", previous.get("lead_question")),
            limit=500,
            required=True,
            label="Lead question",
        ),
        "category": _clean(
            data.get("category", previous.get("category")),
            limit=80,
            required=True,
            label="Category",
        ),
        "summary": _clean_multiline(
            data.get("summary", previous.get("summary")),
            limit=1200,
            label="Topic summary",
        ),
        "status": status,
        "update_interval": interval,
        "run_config": run_config,
        # Denormalized public labels. Executable IDs live in run_config.
        "models": topic_model_labels(run_config["provider_models"]),
        "source_rules": source_rules,
        # Optional editorial seed sources remain supported, but generated runs
        # no longer require or inherit them as their evidence set.
        "evidence": evidence,
        "seo": {
            "title": _clean(
                (data.get("seo") or {}).get(
                    "title", (previous.get("seo") or {}).get("title")
                ),
                limit=160,
                label="SEO title",
            ),
            "description": _clean(
                (data.get("seo") or {}).get(
                    "description", (previous.get("seo") or {}).get("description")
                ),
                limit=300,
                label="SEO description",
            ),
            "noindex": bool(
                (data.get("seo") or {}).get(
                    "noindex", (previous.get("seo") or {}).get("noindex", False)
                )
            ),
        },
    }


def _slug_in_use(slug: str, *, exclude_id: str = "", db=None) -> bool:
    db = db if db is not None else db_firestore
    for doc in db.collection(TOPICS_COLLECTION).where("slug", "==", slug).stream():
        if doc.id != exclude_id:
            return True
    return False


def _topic_from_doc(doc) -> dict | None:
    if not getattr(doc, "exists", False):
        return None
    data = doc.to_dict() or {}
    return {"id": doc.id, **data}


def get_topic(topic_id: str, *, db=None) -> dict | None:
    db = db if db is not None else db_firestore
    return _topic_from_doc(db.collection(TOPICS_COLLECTION).document(topic_id).get())


def get_topic_by_slug(slug: str, *, db=None) -> dict | None:
    db = db if db is not None else db_firestore
    normalized = normalize_slug(slug)
    for doc in db.collection(TOPICS_COLLECTION).where("slug", "==", normalized).stream():
        return _topic_from_doc(doc)
    return None


def create_topic(data: dict, *, actor_uid: str, db=None, now=None) -> dict:
    db = db if db is not None else db_firestore
    now = now or utcnow()
    normalized = normalize_topic_input(data)
    if _slug_in_use(normalized["slug"], db=db):
        raise TopicError("conflict", "This topic slug is already in use.")
    topic_id = secrets.token_urlsafe(15).replace("-", "").replace("_", "")[:20]
    stored = {
        **normalized,
        "created_at": now,
        "updated_at": now,
        "created_by": actor_uid,
        "updated_by": actor_uid,
        "latest_run_id": "",
        "latest_run_at": None,
        "latest_agreement_score": None,
        "latest_change_type": "",
        "latest_change_summary": "",
        "run_count": 0,
        "next_run_at": (
            next_run_at(normalized["update_interval"], now=now)
            if normalized["status"] == "active" else None
        ),
        "claimed_until": None,
        "current_run_id": "",
        "last_run_status": "never",
        "last_run_error": "",
        "consecutive_failures": 0,
    }
    db.collection(TOPICS_COLLECTION).document(topic_id).set(stored)
    return {"id": topic_id, **stored}


def update_topic(
    topic_id: str, data: dict, *, actor_uid: str, db=None, now=None
) -> dict:
    db = db if db is not None else db_firestore
    now = now or utcnow()
    existing = get_topic(topic_id, db=db)
    if not existing:
        raise TopicError("not_found", "Topic not found.")
    normalized = normalize_topic_input(data, existing=existing)
    if _slug_in_use(normalized["slug"], exclude_id=topic_id, db=db):
        raise TopicError("conflict", "This topic slug is already in use.")
    updates = {
        **normalized,
        "updated_at": now,
        "updated_by": actor_uid,
    }
    if (
        normalized["update_interval"] != existing.get("update_interval")
        or normalized["status"] != existing.get("status")
        or (
            normalized["status"] == "active"
            and normalized["update_interval"] != "manual"
            and not isinstance(existing.get("next_run_at"), datetime)
        )
    ):
        updates["next_run_at"] = (
            next_run_at(normalized["update_interval"], now=now)
            if normalized["status"] == "active" else None
        )
        updates["claimed_until"] = None
        updates["current_run_id"] = ""
    db.collection(TOPICS_COLLECTION).document(topic_id).set(updates, merge=True)
    return {**existing, **updates}


def _normalize_score(value) -> int:
    if isinstance(value, bool):
        raise TopicError("bad_request", "Agreement score must be between 0 and 100.")
    try:
        score = int(value)
    except (TypeError, ValueError):
        raise TopicError("bad_request", "Agreement score must be between 0 and 100.")
    if score < 0 or score > 100:
        raise TopicError("bad_request", "Agreement score must be between 0 and 100.")
    return score


def create_run(
    topic_id: str, data: dict, *, actor_uid: str, db=None, now=None, run_id: str = ""
) -> dict:
    """Create one immutable snapshot and advance only the topic's latest pointer."""
    db = db if db is not None else db_firestore
    now = now or utcnow()
    topic = get_topic(topic_id, db=db)
    if not topic:
        raise TopicError("not_found", "Topic not found.")
    if topic.get("status") == "archived":
        raise TopicError("conflict", "Archived topics cannot receive new runs.")
    change_type = str(data.get("change_type") or "stable").lower()
    if change_type not in CHANGE_TYPES:
        raise TopicError("bad_request", "Invalid change type.")
    consensus_md = _clean_multiline(
        data.get("consensus_md"), limit=40_000, label="Consensus"
    )
    if not consensus_md:
        raise TopicError("bad_request", "Consensus is required.")
    observed_at = data.get("observed_at") or now
    if isinstance(observed_at, str):
        try:
            observed_at = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
        except ValueError:
            raise TopicError("bad_request", "Observed date must be ISO-8601.") from None
    if not isinstance(observed_at, datetime):
        raise TopicError("bad_request", "Observed date must be ISO-8601.")
    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=timezone.utc)
    run_id = (
        str(run_id or "").strip()
        or secrets.token_urlsafe(12).replace("-", "").replace("_", "")[:16]
    )
    change_summary = _clean_multiline(
        data.get("change_summary"), limit=1200, label="Change summary"
    )
    if change_type in {"minor", "major"} and not change_summary:
        raise TopicError("bad_request", "Material changes need a change summary.")
    evidence = normalize_evidence(
        data["evidence"] if "evidence" in data else topic.get("evidence") or []
    )
    models = normalize_models(
        data["models"] if "models" in data else topic.get("models") or []
    )
    source_rules = normalize_source_rules(topic.get("source_rules") or {})
    if len(models) < 2:
        raise TopicError("bad_request", "A Topic snapshot needs at least two models.")
    if any(item["type"] not in source_rules["allowed_types"] for item in evidence):
        raise TopicError("bad_request", "Snapshot evidence violates the Topic source rules.")
    run = {
        "topic_id": topic_id,
        "version": int(topic.get("run_count") or 0) + 1,
        "observed_at": observed_at.astimezone(timezone.utc),
        "created_at": now,
        "created_by": actor_uid,
        "consensus_md": consensus_md,
        "agreement_score": _normalize_score(data.get("agreement_score")),
        "change_type": change_type,
        "change_summary": change_summary,
        "opinion_changes": normalize_opinion_changes(data.get("opinion_changes")),
        "evidence": evidence,
        "models": models,
        "source_rules": source_rules,
        "run_mode": str(data.get("run_mode") or "manual")[:20],
        "differences_data": (
            data.get("differences_data")
            if isinstance(data.get("differences_data"), dict) else {}
        ),
        "opinion_map": (
            data.get("opinion_map")
            if isinstance(data.get("opinion_map"), dict) else {}
        ),
        "topic_state": {
            "title": topic.get("title"),
            "slug": topic.get("slug"),
            "lead_question": topic.get("lead_question"),
            "category": topic.get("category"),
        },
    }
    topic_ref = db.collection(TOPICS_COLLECTION).document(topic_id)
    topic_ref.collection("runs").document(run_id).set(run)
    topic_ref.set({
        "latest_run_id": run_id,
        "latest_run_at": run["observed_at"],
        "latest_agreement_score": run["agreement_score"],
        "latest_change_type": change_type,
        "latest_change_summary": run["change_summary"],
        "run_count": run["version"],
        "next_run_at": (
            next_run_at(topic.get("update_interval") or "manual", now=now)
            if topic.get("status") == "active" else None
        ),
        "claimed_until": None,
        "current_run_id": "",
        "last_run_status": "success",
        "last_run_error": "",
        "consecutive_failures": 0,
        "updated_at": now,
        "updated_by": actor_uid,
    }, merge=True)
    return {"id": run_id, **run}


def list_runs(topic_id: str, *, db=None, max_items: int = 100) -> list[dict]:
    db = db if db is not None else db_firestore
    ref = db.collection(TOPICS_COLLECTION).document(topic_id).collection("runs")
    runs = [{"id": doc.id, **(doc.to_dict() or {})} for doc in ref.stream()]
    runs.sort(
        key=lambda item: (
            item.get("observed_at") if isinstance(item.get("observed_at"), datetime)
            else datetime.min.replace(tzinfo=timezone.utc),
            item.get("version") or 0,
        )
    )
    return runs[-max_items:]


def get_run(topic_id: str, run_id: str, *, db=None) -> dict | None:
    db = db if db is not None else db_firestore
    if not str(run_id or "").strip():
        return None
    doc = (
        db.collection(TOPICS_COLLECTION).document(topic_id)
        .collection("runs").document(run_id).get()
    )
    return _topic_from_doc(doc)


def list_due_topic_ids(*, now=None, db=None, max_items=50) -> list[str]:
    db = db if db is not None else db_firestore
    now = now or utcnow()
    due = []
    for doc in db.collection(TOPICS_COLLECTION).where("status", "==", "active").stream():
        data = doc.to_dict() or {}
        scheduled = data.get("next_run_at")
        if isinstance(scheduled, datetime) and scheduled <= now:
            due.append(doc.id)
        if len(due) >= max_items:
            break
    return due


def _claim_topic_transaction(transaction, ref, now: datetime, *, force: bool):
    snap = ref.get(transaction=transaction)
    data = snap.to_dict() if snap.exists else None
    if not data:
        return None, "not_found"
    if data.get("status") != "active":
        return None, "not_active"
    claimed_until = data.get("claimed_until")
    if isinstance(claimed_until, datetime) and claimed_until > now:
        return None, "claimed"
    scheduled = data.get("next_run_at")
    if not force and (not isinstance(scheduled, datetime) or scheduled > now):
        return None, "not_due"
    run_id = secrets.token_urlsafe(12).replace("-", "").replace("_", "")[:16]
    updates = {
        "claimed_until": now + timedelta(minutes=TOPIC_RUN_LEASE_MINUTES),
        "current_run_id": run_id,
        "last_run_status": "running",
        "last_run_error": "",
    }
    transaction.update(ref, updates)
    return {"id": ref.id, **data, **updates}, "claimed"


def claim_topic_run(topic_id: str, *, force=False, now=None, db=None) -> dict:
    from firebase_admin import firestore

    db = db if db is not None else db_firestore
    now = now or utcnow()
    ref = db.collection(TOPICS_COLLECTION).document(topic_id)
    transaction = db.transaction()

    @firestore.transactional
    def claim(tx):
        return _claim_topic_transaction(tx, ref, now, force=force)

    claimed, reason = claim(transaction)
    if claimed:
        return claimed
    messages = {
        "not_found": ("not_found", "Topic not found."),
        "not_active": ("conflict", "Only active Topics can run."),
        "claimed": ("conflict", "This Topic already has a run in progress."),
        "not_due": ("conflict", "This Topic is not due yet."),
    }
    code, message = messages.get(reason, ("conflict", "Topic run could not start."))
    raise TopicError(code, message)


def fail_topic_run(topic_id: str, message: str, *, now=None, db=None) -> None:
    db = db if db is not None else db_firestore
    now = now or utcnow()
    topic = get_topic(topic_id, db=db) or {}
    failures = int(topic.get("consecutive_failures") or 0) + 1
    db.collection(TOPICS_COLLECTION).document(topic_id).set({
        "claimed_until": None,
        "current_run_id": "",
        "last_run_status": "failed",
        "last_run_error": _clean(message, limit=500, label="Run error"),
        "consecutive_failures": failures,
        "next_run_at": next_run_at(
            topic.get("update_interval") or "manual", now=now
        ),
        "updated_at": now,
    }, merge=True)


def topic_public_view(topic: dict) -> dict:
    return {
        "id": topic["id"],
        "title": str(topic.get("title") or ""),
        "slug": str(topic.get("slug") or ""),
        "lead_question": str(topic.get("lead_question") or ""),
        "category": str(topic.get("category") or ""),
        "summary": str(topic.get("summary") or ""),
        "status": str(topic.get("status") or ""),
        "update_interval": str(topic.get("update_interval") or ""),
        "models": list(topic.get("models") or []),
        "seo": dict(topic.get("seo") or {}),
        "latest_run_id": str(topic.get("latest_run_id") or ""),
        "latest_run_at": _public_datetime(topic.get("latest_run_at")),
        "latest_agreement_score": topic.get("latest_agreement_score"),
        "latest_change_type": str(topic.get("latest_change_type") or ""),
        "latest_change_summary": str(topic.get("latest_change_summary") or ""),
        "run_count": int(topic.get("run_count") or 0),
        "created_at": _public_datetime(topic.get("created_at")),
        "updated_at": _public_datetime(topic.get("updated_at")),
    }


def run_public_view(run: dict) -> dict:
    return {
        "id": run["id"],
        "version": int(run.get("version") or 0),
        "observed_at": _public_datetime(run.get("observed_at")),
        "consensus_md": str(run.get("consensus_md") or ""),
        "agreement_score": run.get("agreement_score"),
        "change_type": str(run.get("change_type") or "stable"),
        "change_summary": str(run.get("change_summary") or ""),
        "opinion_changes": list(run.get("opinion_changes") or []),
        "evidence": list(run.get("evidence") or []),
        "models": list(run.get("models") or []),
        "opinion_map": dict(run.get("opinion_map") or {}),
        "topic_state": dict(run.get("topic_state") or {}),
    }


def list_public_topics(*, db=None) -> list[dict]:
    db = db if db is not None else db_firestore
    topics = []
    for doc in db.collection(TOPICS_COLLECTION).stream():
        topic = _topic_from_doc(doc)
        if not topic or topic.get("status") not in {"active", "paused"}:
            continue
        if not topic.get("latest_run_id"):
            continue
        topics.append(topic_public_view(topic))
    topics.sort(key=lambda item: item.get("latest_run_at") or "", reverse=True)
    return topics


def list_admin_topics(*, db=None) -> list[dict]:
    db = db if db is not None else db_firestore
    items = []
    for doc in db.collection(TOPICS_COLLECTION).stream():
        topic = _topic_from_doc(doc)
        if topic:
            item = {**topic}
            for key in (
                "created_at", "updated_at", "latest_run_at", "next_run_at",
                "claimed_until",
            ):
                item[key] = _public_datetime(item.get(key))
            items.append(item)
    items.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
    return items


def list_indexed_topic_urls(*, db=None) -> list[dict]:
    return [{
        "path": f"/topics/{topic['slug']}",
        "lastmod": (topic.get("latest_run_at") or topic.get("updated_at") or "")[:10],
    } for topic in list_public_topics(db=db) if not (topic.get("seo") or {}).get("noindex")]


def normalize_email(value) -> str:
    email = str(value or "").strip().lower()
    if not email or len(email) > 254 or not _EMAIL_RE.fullmatch(email):
        raise TopicError("invalid_email", "Enter a valid e-mail address.")
    return email


def follower_id(topic_id: str, email: str) -> str:
    return hashlib.sha256(f"{topic_id}:{email}".encode("utf-8")).hexdigest()[:32]


def _followable_topic(topic_id: str, db) -> dict:
    topic = get_topic(topic_id, db=db)
    if (
        not topic
        or topic.get("status") not in {"active", "paused"}
        or not topic.get("latest_run_id")
    ):
        raise TopicError("not_found", "This topic cannot be followed.")
    return topic


def make_confirm_token(topic_id: str, email: str, *, now=None) -> str:
    return sign_token_payload(
        {"tt": "topic", "tid": topic_id, "em": email},
        now=now,
        max_age_days=CONFIRM_TOKEN_MAX_AGE_DAYS,
    )


def make_unsubscribe_token(topic_id: str, email: str, *, now=None) -> str:
    return sign_token_payload(
        {"tt": "topic", "tid": topic_id, "em": email, "un": 1}, now=now
    )


def request_follow(topic_id: str, email, *, db=None) -> dict:
    db = db if db is not None else db_firestore
    email = normalize_email(email)
    topic = _followable_topic(topic_id, db)
    doc_id = follower_id(topic_id, email)
    exists = db.collection(FOLLOWERS_COLLECTION).document(doc_id).get().exists
    return {
        "email": email,
        "title": topic["title"],
        "slug": topic["slug"],
        "token": "" if exists else make_confirm_token(topic_id, email),
    }


def _parse_topic_token(token: str, *, unsubscribe: bool) -> tuple[str, str]:
    try:
        payload = parse_token_payload(token)
    except WatchError as exc:
        raise TopicError(exc.code, exc.message) from exc
    if payload.get("tt") != "topic" or bool(payload.get("un")) != unsubscribe:
        raise TopicError("invalid_token", "This link is invalid.")
    return str(payload.get("tid") or ""), normalize_email(payload.get("em"))


def confirm_follow(token: str, *, db=None, now=None) -> dict:
    db = db if db is not None else db_firestore
    topic_id, email = _parse_topic_token(token, unsubscribe=False)
    topic = _followable_topic(topic_id, db)
    ref = db.collection(FOLLOWERS_COLLECTION).document(follower_id(topic_id, email))
    if not ref.get().exists:
        count = sum(
            1 for _ in db.collection(FOLLOWERS_COLLECTION)
            .where("topic_id", "==", topic_id).stream()
        )
        if count >= MAX_FOLLOWERS_PER_TOPIC:
            raise TopicError("limit_reached", "This topic has reached its follower limit.")
        ref.set({"topic_id": topic_id, "email": email, "created_at": now or utcnow()})
    return {"topic_id": topic_id, "email": email, "title": topic["title"], "slug": topic["slug"]}


def unsubscribe_follow(token: str, *, db=None) -> dict:
    db = db if db is not None else db_firestore
    topic_id, email = _parse_topic_token(token, unsubscribe=True)
    db.collection(FOLLOWERS_COLLECTION).document(follower_id(topic_id, email)).delete()
    return {"topic_id": topic_id, "email": email}


def list_followers(topic_id: str, *, db=None) -> list[dict]:
    db = db if db is not None else db_firestore
    followers = []
    for doc in (
        db.collection(FOLLOWERS_COLLECTION).where("topic_id", "==", topic_id).stream()
    ):
        data = doc.to_dict() or {}
        if data.get("email"):
            followers.append({"id": doc.id, **data})
    return followers[:MAX_FOLLOWERS_PER_TOPIC]


def delivery_id(topic_id: str, run_id: str, follower_doc_id: str) -> str:
    raw = f"{topic_id}:{run_id}:{follower_doc_id}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def claim_delivery(topic_id: str, run_id: str, follower_doc_id: str, *, db=None) -> bool:
    """Best-effort delivery dedupe. Failed sends delete the claim for retry."""
    db = db if db is not None else db_firestore
    ref = db.collection(DELIVERIES_COLLECTION).document(
        delivery_id(topic_id, run_id, follower_doc_id)
    )
    if ref.get().exists:
        return False
    ref.set({
        "topic_id": topic_id,
        "run_id": run_id,
        "follower_id": follower_doc_id,
        "status": "sending",
        "created_at": utcnow(),
    })
    return True


def finish_delivery(
    topic_id: str, run_id: str, follower_doc_id: str, *, success: bool, db=None
) -> None:
    db = db if db is not None else db_firestore
    ref = db.collection(DELIVERIES_COLLECTION).document(
        delivery_id(topic_id, run_id, follower_doc_id)
    )
    if success:
        ref.set({"status": "sent", "sent_at": utcnow()}, merge=True)
    else:
        ref.delete()
