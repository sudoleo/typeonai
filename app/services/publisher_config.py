"""Firestore configuration for the scheduled Consensus publisher."""

from __future__ import annotations

from datetime import datetime, timezone

from app.core.security import db_firestore
from app.services import watch_service


CONFIG_COLLECTION = "app_config"
CONFIG_DOCUMENT = "scheduled_consensus_publisher"
TOPIC_BRIEF_MAX_CHARS = 6_000

DEFAULT_TOPIC_BRIEF = (
    "Choose one timely, evidence-rich topic that real people are actively searching for "
    "in science, technology, economics, environment, or society. Favor a specific, "
    "question-shaped angle with clear search demand and a gap in existing coverage — "
    "an underserved query rather than a broad, already-saturated subject. Prefer topics "
    "inside a fresh news or debate window where opinion is still forming.\n\n"
    "The topic must have a concrete use case: name the reader who would run it and the "
    "question or decision the answer actually helps them with. It should genuinely "
    "benefit from comparing multiple AI models — where the models are likely to disagree "
    "or hedge, so that both the consensus and the dissent are informative — and support "
    "a substantial answer backed by several credible web sources.\n\n"
    "Avoid personal medical, legal, or financial advice, sensationalism, pure opinion "
    "polls, and purely speculative topics with no verifiable grounding."
)

DEFAULT_CONFIG = {
    "enabled": True,
    "topic_brief": DEFAULT_TOPIC_BRIEF,
    "auto_index": True,
    "weekly_watch_enabled": True,
    "watch_weekday": "tuesday",
    "watch_time": "09:00",
    "watch_timezone": "Europe/Berlin",
}


class PublisherConfigError(ValueError):
    pass


def _document(db):
    return db.collection(CONFIG_COLLECTION).document(CONFIG_DOCUMENT)


def normalize_config(data: dict | None) -> dict:
    incoming = data if isinstance(data, dict) else {}
    config = dict(DEFAULT_CONFIG)
    for field in ("enabled", "auto_index", "weekly_watch_enabled"):
        if field in incoming:
            if not isinstance(incoming[field], bool):
                raise PublisherConfigError(f"{field} must be a boolean")
            config[field] = incoming[field]

    brief = str(incoming.get("topic_brief", config["topic_brief"]) or "").strip()
    if not brief:
        raise PublisherConfigError("topic_brief must not be empty")
    if len(brief) > TOPIC_BRIEF_MAX_CHARS:
        raise PublisherConfigError(
            f"topic_brief must contain at most {TOPIC_BRIEF_MAX_CHARS} characters"
        )
    config["topic_brief"] = brief

    weekday = str(incoming.get("watch_weekday", config["watch_weekday"]) or "").strip().lower()
    run_time = str(incoming.get("watch_time", config["watch_time"]) or "").strip()
    timezone_name = str(
        incoming.get("watch_timezone", config["watch_timezone"]) or ""
    ).strip()
    try:
        run_time, timezone_name = watch_service.validate_run_schedule(run_time, timezone_name)
        weekday = watch_service.validate_run_weekday(
            weekday, "weekly", has_run_time=bool(run_time)
        )
    except watch_service.WatchError as exc:
        raise PublisherConfigError(exc.message) from exc
    config.update(
        watch_weekday=weekday,
        watch_time=run_time,
        watch_timezone=timezone_name,
    )
    return config


def get_config(*, db=None) -> dict:
    db = db if db is not None else db_firestore
    ref = _document(db)
    snap = ref.get()
    if not snap.exists:
        config = dict(DEFAULT_CONFIG)
        now = datetime.now(timezone.utc)
        ref.set({**config, "created_at": now, "updated_at": now, "updated_by": "default"})
        return config
    return normalize_config(snap.to_dict())


def save_config(data: dict, *, updated_by: str, db=None) -> dict:
    db = db if db is not None else db_firestore
    config = normalize_config(data)
    _document(db).set(
        {
            **config,
            "updated_at": datetime.now(timezone.utc),
            "updated_by": str(updated_by or "")[:128],
        },
        merge=True,
    )
    return config


def public_config(config: dict) -> dict:
    """Add immutable execution facts that the Admin UI and publisher can display."""
    return {
        **normalize_config(config),
        "watch_interval": "weekly",
        "watch_model_tier": "free",
        "excluded_providers": ["deepseek"],
    }
