"""Firestore configuration for the scheduled Consensus publisher."""

from __future__ import annotations

from datetime import datetime, timezone

from app.core.security import db_firestore
from app.services import watch_service


CONFIG_COLLECTION = "app_config"
CONFIG_DOCUMENT = "scheduled_consensus_publisher"
TOPIC_BRIEF_MAX_CHARS = 6_000
DEFAULT_MAX_ACTIVE_PUBLISHER_WATCHES = 12

# Kept here as a read-only product fact for the Admin SEO review. The standalone
# publisher script intentionally carries the same constant because it has no
# application-package dependency at runtime.
SEARCH_OPPORTUNITY_RULES = (
    "Search-opportunity requirements:\n"
    "- Work in the high-current AI product/news lane. Prefer a named model, company, feature, "
    "plan, coding tool, release, leak, or rumor with a new signal from roughly the last seven days.\n"
    "- Favor queries shaped like release/availability checks, rumor verification, product-name "
    "clarification, or what a just-announced change means for users.\n"
    "- Use web search to compare at least five candidate queries before choosing. Reject a "
    "candidate when its exact search intent is already answered by many established, high-ranking "
    "news, government, legal, or corporate explainer pages. Choose the candidate with the best "
    "combination of freshness, plausible search demand, low exact-intent competition, and sources.\n"
    "- Do not select government policy, grants, federal/state law, regulation, enforcement, "
    "elections, or broad societal impact as the main intent.\n"
    "- Do not invent a release or rumor. A speculative question needs at least one current, "
    "checkable signal and must make uncertainty explicit."
)

LEGACY_DEFAULT_TOPIC_BRIEF = (
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

DEFAULT_TOPIC_BRIEF = (
    "Choose one highly current, evidence-rich AI topic that people are beginning to search "
    "for now. Focus on named AI models, products, features, subscriptions, developer tools, "
    "release timing, availability, surprising product behavior, or a credible emerging "
    "rumor. Favor a narrow exact-intent query while the news window is still young and "
    "dedicated coverage is sparse.\n\n"
    "The question should help someone verify a claim, understand what just changed, or "
    "decide whether to wait for or use a specific AI product. It must benefit from comparing "
    "multiple AI models and support a substantial answer from several credible web sources. "
    "Careful speculation is welcome only when it is clearly framed and anchored in official "
    "announcements, documentation, changelogs, observed product behavior, or credible reporting.\n\n"
    "Avoid government policy, regulation, legislation, elections, broad evergreen explainers, "
    "generic AI trend pieces, personal medical/legal/financial advice, sensationalism, and "
    "unsupported rumors."
)

DEFAULT_CONFIG = {
    "enabled": True,
    "topic_brief": DEFAULT_TOPIC_BRIEF,
    "auto_index": True,
    "weekly_watch_enabled": True,
    "watch_weekday": "tuesday",
    "watch_time": "09:00",
    "watch_timezone": "Europe/Berlin",
    "max_active_publisher_watches": DEFAULT_MAX_ACTIVE_PUBLISHER_WATCHES,
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
    if brief == LEGACY_DEFAULT_TOPIC_BRIEF:
        brief = DEFAULT_TOPIC_BRIEF
    if not brief:
        raise PublisherConfigError("topic_brief must not be empty")
    if len(brief) > TOPIC_BRIEF_MAX_CHARS:
        raise PublisherConfigError(
            f"topic_brief must contain at most {TOPIC_BRIEF_MAX_CHARS} characters"
        )
    config["topic_brief"] = brief

    try:
        max_watches = int(incoming.get(
            "max_active_publisher_watches", config["max_active_publisher_watches"]
        ))
    except (TypeError, ValueError):
        raise PublisherConfigError("max_active_publisher_watches must be an integer") from None
    if not 1 <= max_watches <= 100:
        raise PublisherConfigError("max_active_publisher_watches must be between 1 and 100")
    config["max_active_publisher_watches"] = max_watches

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
        "search_opportunity_rules": SEARCH_OPPORTUNITY_RULES,
    }
