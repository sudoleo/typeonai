"""Small, privacy-conscious dossiers for pages observed by the SEO service."""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from bs4 import BeautifulSoup

from app.services import share_snapshots, topics
from app.services.public_markdown import markdown_to_plaintext


TEMPLATES_ROOT = Path(__file__).resolve().parents[2] / "templates"
MAX_CONTENT_SUMMARY_CHARS = 900
MAX_SHARE_CONTENT_REPRESENTATION_CHARS = 3_200

STATIC_TEMPLATE_BY_PATH = {
    "/": "landing.html",
    "/ai-model-comparison": "ai-model-comparison.html",
    "/consensus-engine": "consensus-engine.html",
    "/questions": "questions.html",
    "/topics": "topics.html",
    "/benchmark": "benchmark.html",
    "/about": "about.html",
}


def _clip(value, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _iso(value) -> str | None:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value or "").strip()
    return text or None


def _static_summary(soup: BeautifulSoup, meta_description: str) -> str:
    main = soup.find("main") or soup.body
    parts: list[str] = []
    if main:
        for element in main.find_all(["h1", "h2", "p"], limit=8):
            text = _clip(element.get_text(" ", strip=True), 300)
            if text and "{{" not in text and "{%" not in text:
                parts.append(text)
    return _clip(" ".join(parts) or meta_description, MAX_CONTENT_SUMMARY_CHARS)


def _share_meta_description(data: dict, history: list[dict]) -> str:
    differences = data.get("differences_data")
    differences = differences if isinstance(differences, dict) else {}
    agreement = differences.get("agreement")
    agreement = agreement if isinstance(agreement, dict) else {}
    model_count = (
        len(differences.get("models_compared") or [])
        or len(data.get("included_models") or [])
    )
    contradiction_count = sum(
        1 for item in (differences.get("differences") or [])
        if isinstance(item, dict) and item.get("type") == "contradiction"
    )
    latest_score = history[-1].get("agreement_score") if history else agreement.get("score")
    bits = []
    if model_count:
        bits.append(f"{model_count} AI models answered independently")
    if isinstance(latest_score, (int, float)):
        bits.append(f"agreement {int(latest_score)}/100")
    if contradiction_count:
        bits.append(
            f"{contradiction_count} contradiction"
            f"{'s' if contradiction_count != 1 else ''}"
        )
    if history and isinstance(history[0].get("ts"), datetime):
        bits.append(f"tracked since {history[0]['ts'].strftime('%b %Y')}")
    if not bits:
        return markdown_to_plaintext(data.get("consensus_md"), limit=160)
    prefix = " · ".join(bits) + ". "
    excerpt = markdown_to_plaintext(
        data.get("consensus_md"), limit=max(40, 160 - len(prefix))
    )
    return prefix + excerpt


def build_static_dossier(url: str, lastmod=None) -> dict:
    path = urlsplit(url).path or "/"
    template_name = STATIC_TEMPLATE_BY_PATH.get(path)
    uncertainties: list[str] = []
    title = meta_description = content_summary = ""
    if not template_name:
        uncertainties.append("static_template_unknown")
    else:
        try:
            source = (TEMPLATES_ROOT / template_name).read_text(encoding="utf-8")
            soup = BeautifulSoup(source, "html.parser")
            title = _clip(soup.title.get_text(" ", strip=True) if soup.title else "", 300)
            meta = soup.find("meta", attrs={"name": "description"})
            meta_description = _clip(meta.get("content") if meta else "", 500)
            content_summary = _static_summary(soup, meta_description)
        except (OSError, UnicodeError):
            uncertainties.append("static_template_unreadable")
    if not title:
        uncertainties.append("missing_title")
    if not meta_description:
        uncertainties.append("missing_meta_description")
    return {
        "schema_version": 1,
        "published_at": None,
        "last_content_change_at": _iso(lastmod),
        "title": title,
        "meta_description": meta_description,
        "content_summary": content_summary,
        "content_representation": None,
        "source_freshness": {"source_count": 0, "snapshot_at": None},
        "watch_freshness": {"last_checked_at": None, "last_material_change_at": None},
        "technical_uncertainties": sorted(set(uncertainties)),
    }


def build_share_dossier(share_id: str, *, db=None) -> dict:
    data = share_snapshots.get_share(share_id, db=db)
    if not data:
        return {
            "schema_version": 1,
            "published_at": None,
            "last_content_change_at": None,
            "title": "",
            "meta_description": "",
            "content_summary": "",
            "content_representation": None,
            "source_freshness": {"source_count": 0, "snapshot_at": None},
            "watch_freshness": {"last_checked_at": None, "last_material_change_at": None},
            "technical_uncertainties": ["share_snapshot_unavailable"],
        }

    question = _clip(data.get("question"), 300)
    consensus = markdown_to_plaintext(data.get("consensus_md"), limit=3_000)
    created_at = data.get("created_at")
    last_checked_at = data.get("last_watch_run_at")
    history = share_snapshots.list_watch_history(share_id, db=db, max_items=100)
    material_changes = [point.get("ts") for point in history if point.get("changed")]
    last_material_change = material_changes[-1] if material_changes else created_at
    sources = data.get("sources") if isinstance(data.get("sources"), list) else []
    source_snapshot_at = data.get("answered_at") or created_at
    watch_prefix = "Consensus Watch: " if isinstance(last_checked_at, datetime) else ""
    title = _clip(f"{watch_prefix}{question} | consens.io", 300) if question else ""
    meta_description = _share_meta_description(data, history)
    representation = _clip(
        f"Question: {question}\nConsensus excerpt: {consensus}",
        MAX_SHARE_CONTENT_REPRESENTATION_CHARS,
    )
    uncertainties = []
    if not title:
        uncertainties.append("missing_title")
    if not meta_description:
        uncertainties.append("missing_meta_description")
    if not isinstance(created_at, datetime):
        uncertainties.append("missing_publish_timestamp")
    return {
        "schema_version": 1,
        "published_at": _iso(created_at),
        "last_content_change_at": _iso(last_material_change),
        "title": title,
        "meta_description": meta_description,
        "content_summary": _clip(consensus, MAX_CONTENT_SUMMARY_CHARS),
        # The journal never receives this field. It is deliberately bounded in
        # the page dossier so an optional content judge has useful context
        # without persisting the full share answer a second time.
        "content_representation": representation,
        "source_freshness": {
            "source_count": len(sources),
            "snapshot_at": _iso(source_snapshot_at),
        },
        "watch_freshness": {
            "last_checked_at": _iso(last_checked_at),
            "last_material_change_at": _iso(last_material_change),
        },
        "technical_uncertainties": sorted(set(uncertainties)),
    }


def build_topic_dossier(topic_id: str, *, db=None) -> dict:
    topic = topics.get_topic(topic_id, db=db)
    if not topic:
        return {
            "schema_version": 1,
            "published_at": None,
            "last_content_change_at": None,
            "title": "",
            "meta_description": "",
            "content_summary": "",
            "content_representation": None,
            "source_freshness": {"source_count": 0, "snapshot_at": None},
            "watch_freshness": {"last_checked_at": None, "last_material_change_at": None},
            "technical_uncertainties": ["topic_unavailable"],
        }
    latest = topics.get_run(topic_id, topic.get("latest_run_id") or "", db=db)
    consensus = markdown_to_plaintext((latest or {}).get("consensus_md"), limit=3_000)
    seo = topic.get("seo") if isinstance(topic.get("seo"), dict) else {}
    evidence = (latest or {}).get("evidence")
    evidence = evidence if isinstance(evidence, list) else []
    title = _clip(
        seo.get("title") or f"{topic.get('title', '')} Consensus Timeline | consens.io",
        300,
    )
    description = _clip(
        seo.get("description")
        or f"{topic.get('lead_question', '')} Versioned consensus and evidence timeline.",
        500,
    )
    return {
        "schema_version": 1,
        "published_at": _iso(topic.get("created_at")),
        "last_content_change_at": _iso(
            (latest or {}).get("observed_at") or topic.get("updated_at")
        ),
        "title": title,
        "meta_description": description,
        "content_summary": _clip(consensus, MAX_CONTENT_SUMMARY_CHARS),
        "content_representation": _clip(
            f"Topic: {topic.get('title', '')}\nQuestion: {topic.get('lead_question', '')}"
            f"\nConsensus excerpt: {consensus}",
            MAX_SHARE_CONTENT_REPRESENTATION_CHARS,
        ),
        "source_freshness": {
            "source_count": len(evidence),
            "snapshot_at": _iso((latest or {}).get("observed_at")),
        },
        "watch_freshness": {
            "last_checked_at": _iso((latest or {}).get("observed_at")),
            "last_material_change_at": _iso((latest or {}).get("observed_at")),
        },
        "technical_uncertainties": [],
    }


def journal_summary(page: dict) -> dict:
    """Return the whitelisted, raw-content-free dossier stored per judgement."""
    dossier = page.get("dossier") if isinstance(page.get("dossier"), dict) else {}
    return {
        "url": _clip(page.get("url"), 500),
        "origin": _clip(page.get("origin"), 40),
        "first_seen_at": _iso(page.get("first_seen_at")),
        "published_at": _iso(dossier.get("published_at")),
        "last_content_change_at": _iso(dossier.get("last_content_change_at")),
        "title": _clip(dossier.get("title"), 300),
        "meta_description": _clip(dossier.get("meta_description"), 500),
        "content_summary": _clip(dossier.get("content_summary"), MAX_CONTENT_SUMMARY_CHARS),
        "source_freshness": dossier.get("source_freshness") or {},
        "watch_freshness": dossier.get("watch_freshness") or {},
        "technical_uncertainties": list(dossier.get("technical_uncertainties") or [])[:10],
    }
