"""Public and admin routes for the independent curated Topics area."""

import asyncio
import html
import json
import logging
import re
from datetime import datetime
from typing import Optional
from urllib.parse import urlsplit

from fastapi import APIRouter, BackgroundTasks, Body, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

from app.api.routers.pages import SITE_URL
from app.core.rate_limit import limiter
from app.core.security import extract_id_token, is_user_admin, verify_user_token
from app.services import favicons, mailer, topic_runner, topics
from app.services.public_markdown import (
    markdown_to_plaintext,
    render_public_markdown,
    source_site_name,
)


router = APIRouter()
templates = Jinja2Templates(directory="templates")

_STATUS_BY_CODE = {
    "not_found": 404,
    "bad_request": 400,
    "conflict": 409,
    "invalid_email": 400,
    "invalid_token": 400,
    "expired_token": 410,
    "limit_reached": 429,
}


def _raise_topic(exc: topics.TopicError):
    raise HTTPException(
        status_code=_STATUS_BY_CODE.get(exc.code, 400), detail=exc.message
    )


def _require_admin(request: Request, data: Optional[dict] = None) -> str:
    token = extract_id_token(request, data or {})
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        uid = verify_user_token(token)
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Authentication failed") from exc
    if not is_user_admin(uid):
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return uid


def _date_label(value: str) -> str:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%b %d, %Y")
    except (TypeError, ValueError):
        return str(value or "")[:10]


_VIDEO_HOST_RE = re.compile(r"(?:^|\.)(youtube\.com|youtu\.be|vimeo\.com)$")


def _x_handle(url: str) -> str:
    """Best-effort @handle from an X/Twitter status URL (x.com/<handle>/status/..)."""
    try:
        parts = [p for p in urlsplit(str(url or "")).path.split("/") if p]
    except ValueError:
        return ""
    if parts and parts[0].lower() not in {"i", "intent", "search", "hashtag"}:
        handle = re.sub(r"[^A-Za-z0-9_]", "", parts[0])[:30]
        return f"@{handle}" if handle else ""
    return ""


def _is_video_url(url: str) -> bool:
    try:
        host = (urlsplit(str(url or "")).hostname or "").lower()
    except ValueError:
        return False
    host = re.sub(r"^www\.", "", host)
    return bool(_VIDEO_HOST_RE.search(host))


def _host_of(url) -> str:
    try:
        host = (urlsplit(str(url or "")).hostname or "").lower()
    except ValueError:
        return ""
    return re.sub(r"^www\.", "", host)


def _enrich_evidence(item: dict) -> None:
    """Attach self-hosted embed hints (no third-party requests at view time)."""
    url = item.get("url")
    item["site"] = source_site_name(url)
    item["host"] = _host_of(url)
    item["is_video"] = _is_video_url(url)
    handle = _x_handle(url) if item.get("type") == "x" else ""
    item["handle"] = handle
    seed = str(item.get("publisher") or handle.lstrip("@") or item.get("site") or "X")
    initial = next((ch for ch in seed if ch.isalnum()), "X")
    item["initial"] = initial.upper()
    # Auto-generated runs sometimes carry a bare citation index ("7") as the
    # title. Suppress those so the card stays clean (favicon + site + badge).
    title = str(item.get("title") or "").strip()
    item["show_title"] = bool(title) and not title.isdigit()


def _admin_topic_view(topic: dict) -> dict:
    result = dict(topic)
    for key in (
        "created_at", "updated_at", "latest_run_at", "next_run_at",
        "claimed_until",
    ):
        value = result.get(key)
        if isinstance(value, datetime):
            result[key] = value.isoformat()
    return result


def _message_page(
    heading: str, message: str, status_code: int, *, href: str = "/topics"
) -> HTMLResponse:
    content = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<meta name='robots' content='noindex'><title>Topics | consens.io</title>"
        "<style>body{font-family:Inter,system-ui,sans-serif;max-width:680px;margin:12vh auto;"
        "padding:24px;color:#172033}a{color:#335cff}</style></head><body>"
        f"<p style='color:#335cff;font-weight:700'>consens.io Topics</p>"
        f"<h1>{html.escape(heading)}</h1><p>{html.escape(message)}</p>"
        f"<p><a href='{html.escape(href)}'>Open Topics</a></p></body></html>"
    )
    return HTMLResponse(
        content,
        status_code=status_code,
        headers={"X-Robots-Tag": "noindex, noarchive"},
    )


@router.get("/topics", response_class=HTMLResponse)
async def topics_hub(request: Request):
    try:
        entries = await asyncio.to_thread(topics.list_public_topics)
    except Exception:
        logging.exception("topics_hub failed")
        entries = []
    categories = sorted({entry["category"] for entry in entries if entry["category"]})
    jsonld = {
        "@context": "https://schema.org",
        "@type": "CollectionPage",
        "@id": SITE_URL + "/topics",
        "url": SITE_URL + "/topics",
        "name": "AI Consensus Topics",
        "description": (
            "Curated AI topics with versioned consensus timelines, agreement "
            "scores, opinion changes, and time-matched evidence."
        ),
        "mainEntity": {
            "@type": "ItemList",
            "numberOfItems": len(entries),
            "itemListElement": [{
                "@type": "ListItem",
                "position": index,
                "url": f"{SITE_URL}/topics/{entry['slug']}",
                "name": entry["title"],
            } for index, entry in enumerate(entries, start=1)],
        },
    }
    response = templates.TemplateResponse("topics.html", {
        "request": request,
        "entries": entries,
        "categories": categories,
        "jsonld": json.dumps(jsonld, ensure_ascii=False).replace("</", "<\\/"),
    })
    response.headers["Cache-Control"] = "public, max-age=60, s-maxage=300"
    return response


@router.get("/sitemap-topics.xml")
async def sitemap_topics():
    try:
        urls = await asyncio.to_thread(topics.list_indexed_topic_urls)
    except Exception:
        logging.exception("sitemap_topics failed")
        raise HTTPException(status_code=500, detail="Error building topic sitemap")
    items = []
    for item in urls:
        lastmod = (
            f"\n    <lastmod>{html.escape(item['lastmod'])}</lastmod>"
            if item.get("lastmod") else ""
        )
        items.append(
            "  <url>\n"
            f"    <loc>{SITE_URL}{html.escape(item['path'])}</loc>{lastmod}\n"
            "    <changefreq>weekly</changefreq>\n"
            "    <priority>0.8</priority>\n"
            "  </url>"
        )
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + "\n".join(items)
        + "\n</urlset>\n"
    )
    return Response(content=xml, media_type="application/xml")


@router.get("/topics/{slug}", response_class=HTMLResponse)
async def topic_page(
    request: Request, slug: str, version: str = Query(default="", max_length=40)
):
    try:
        topic = await asyncio.to_thread(topics.get_topic_by_slug, slug)
    except topics.TopicError:
        raise HTTPException(status_code=404, detail="Topic not found")
    except Exception:
        logging.exception("topic_page lookup failed")
        raise HTTPException(status_code=500, detail="Error loading topic")
    if (
        not topic
        or topic.get("status") not in {"active", "paused"}
        or not topic.get("latest_run_id")
    ):
        raise HTTPException(status_code=404, detail="Topic not found")

    runs_raw = await asyncio.to_thread(topics.list_runs, topic["id"])
    runs = [topics.run_public_view(run) for run in runs_raw]
    selected = next(
        (run for run in runs if run["id"] == (version or topic["latest_run_id"])),
        None,
    )
    if not selected:
        raise HTTPException(status_code=404, detail="Topic version not found")
    current = selected["id"] == topic["latest_run_id"]
    for run in runs:
        run["date_display"] = _date_label(run["observed_at"])
        run["consensus_excerpt"] = markdown_to_plaintext(
            run["consensus_md"], limit=190
        )
        run["is_selected"] = run["id"] == selected["id"]
        for item in run["evidence"]:
            _enrich_evidence(item)
    runs_desc = list(reversed(runs))
    selected["consensus_html"] = render_public_markdown(
        selected["consensus_md"], selected["evidence"]
    )
    for change in selected["opinion_changes"]:
        change["summary_html"] = render_public_markdown(change.get("summary"))

    public_topic = topics.topic_public_view(topic)
    canonical_url = f"{SITE_URL}/topics/{public_topic['slug']}"
    page_url = canonical_url + (f"?version={selected['id']}" if version else "")
    seo = public_topic.get("seo") or {}
    title = seo.get("title") or f"{public_topic['title']} Consensus Timeline"
    meta_description = seo.get("description") or (
        f"{public_topic['lead_question']} Current agreement "
        f"{selected['agreement_score']}/100, with a versioned consensus timeline "
        "and time-matched evidence."
    )
    robots = "noindex, follow" if seo.get("noindex") or version else "index, follow"
    citations = [item["url"] for run in runs for item in run["evidence"]]
    jsonld = {
        "@context": "https://schema.org",
        "@type": "Article",
        "headline": title[:110],
        "description": meta_description[:300],
        "datePublished": runs[0]["observed_at"] if runs else selected["observed_at"],
        "dateModified": runs[-1]["observed_at"] if runs else selected["observed_at"],
        "mainEntityOfPage": {"@type": "WebPage", "@id": canonical_url},
        "author": {"@type": "Organization", "name": "consens.io", "url": SITE_URL},
        "publisher": {
            "@type": "Organization",
            "name": "consens.io",
            "logo": {"@type": "ImageObject", "url": SITE_URL + "/static/favicon-square.png"},
        },
        "citation": list(dict.fromkeys(citations))[:30],
        "isAccessibleForFree": True,
    }
    response = templates.TemplateResponse("topic.html", {
        "request": request,
        "topic": public_topic,
        "selected": selected,
        "runs": runs_desc,
        "canonical_url": canonical_url,
        "page_url": page_url,
        "page_title": title,
        "meta_description": meta_description,
        "robots_meta": robots,
        "jsonld": json.dumps(jsonld, ensure_ascii=False).replace("</", "<\\/"),
        "is_current": current,
    })
    response.headers["X-Robots-Tag"] = robots
    response.headers["Cache-Control"] = (
        "public, max-age=31536000, immutable"
        if version else "public, max-age=60, s-maxage=300, stale-while-revalidate=300"
    )
    return response


_GLOBE_SVG = (
    b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
    b'stroke="currentColor" stroke-width="1.6" stroke-linecap="round" '
    b'stroke-linejoin="round"><circle cx="12" cy="12" r="9"/>'
    b'<path d="M3 12h18M12 3c2.5 2.5 3.5 6 3.5 9s-1 6.5-3.5 9c-2.5-2.5-3.5-6-3.5-9'
    b'S9.5 5.5 12 3z"/></svg>'
)


@router.get("/api/topics/favicon")
@limiter.limit("300/minute")
async def topic_favicon(request: Request, d: str = Query(default="", max_length=253)):
    data, content_type = await asyncio.to_thread(favicons.get_favicon, d)
    if not data:
        return Response(
            content=_GLOBE_SVG,
            media_type="image/svg+xml",
            headers={"Cache-Control": "public, max-age=86400"},
        )
    return Response(
        content=data,
        media_type=content_type or "image/png",
        headers={"Cache-Control": "public, max-age=604800, immutable"},
    )


@router.post("/api/topics/{slug}/follow")
@limiter.limit("3/minute")
async def follow_topic(request: Request, slug: str, data: dict = Body(default={})):
    try:
        topic = await asyncio.to_thread(topics.get_topic_by_slug, slug)
        if not topic:
            raise topics.TopicError("not_found", "This topic cannot be followed.")
        pending = await asyncio.to_thread(
            topics.request_follow, topic["id"], data.get("email")
        )
    except topics.TopicError as exc:
        _raise_topic(exc)
    except Exception:
        logging.exception("follow_topic failed")
        raise HTTPException(status_code=500, detail="Error processing follow request")
    if pending["token"]:
        if not mailer.is_configured():
            raise HTTPException(status_code=503, detail="E-mail delivery is not configured.")
        topic_url = f"{SITE_URL}/topics/{pending['slug']}"
        confirm_url = SITE_URL + "/topic-follow/confirm?token=" + pending["token"]
        sent = await mailer.send_message(mailer.build_topic_follow_confirm_message(
            recipient=pending["email"],
            title=pending["title"],
            confirm_url=confirm_url,
            topic_url=topic_url,
        ))
        if not sent:
            raise HTTPException(
                status_code=502,
                detail="Confirmation e-mail could not be sent. Please try again later.",
            )
    return {
        "status": "success",
        "message": "Check your inbox and confirm to start following.",
    }


@router.get("/topic-follow/confirm", response_class=HTMLResponse)
@limiter.limit("20/minute")
async def topic_follow_confirm(request: Request, token: str = ""):
    try:
        result = await asyncio.to_thread(topics.confirm_follow, token)
        return _message_page(
            "You're following this topic",
            f"We will e-mail you when the curated consensus on “{result['title']}” changes materially.",
            200,
            href=f"/topics/{result['slug']}",
        )
    except Exception as exc:
        message = getattr(exc, "message", "This confirmation link is invalid or expired.")
        code = _STATUS_BY_CODE.get(getattr(exc, "code", ""), 400)
        return _message_page("Unable to confirm", message, code)


@router.get("/topic-follow/unsubscribe", response_class=HTMLResponse)
@limiter.limit("20/minute")
async def topic_follow_unsubscribe(request: Request, token: str = ""):
    try:
        await asyncio.to_thread(topics.unsubscribe_follow, token)
        return _message_page(
            "Topic unfollowed",
            "You will no longer receive curated consensus updates for this topic.",
            200,
        )
    except Exception as exc:
        message = getattr(exc, "message", "This unsubscribe link is invalid or expired.")
        code = _STATUS_BY_CODE.get(getattr(exc, "code", ""), 400)
        return _message_page("Unable to unfollow", message, code)


async def _notify_topic_followers(topic: dict, run: dict, old_score) -> None:
    try:
        await topic_runner.notify_topic_followers(topic, run, old_score)
    except Exception:
        logging.exception("Topic follower notification failed")


@router.get("/api/admin/topics")
@limiter.limit("30/minute")
async def admin_list_topics(request: Request):
    _require_admin(request)
    try:
        items = await asyncio.to_thread(topics.list_admin_topics)
        return {"status": "success", "topics": items}
    except Exception:
        logging.exception("admin_list_topics failed")
        raise HTTPException(status_code=500, detail="Failed to load topics")


@router.post("/api/admin/topics")
@limiter.limit("20/minute")
async def admin_create_topic(request: Request, data: dict = Body(...)):
    actor_uid = _require_admin(request, data)
    payload = {key: value for key, value in data.items() if key != "id_token"}
    try:
        topic = await asyncio.to_thread(
            topics.create_topic, payload, actor_uid=actor_uid
        )
        return {"status": "success", "topic": _admin_topic_view(topic)}
    except topics.TopicError as exc:
        _raise_topic(exc)
    except Exception:
        logging.exception("admin_create_topic failed")
        raise HTTPException(status_code=500, detail="Failed to create topic")


@router.get("/api/admin/topics/{topic_id}")
@limiter.limit("30/minute")
async def admin_get_topic(request: Request, topic_id: str):
    _require_admin(request)
    topic = await asyncio.to_thread(topics.get_topic, topic_id)
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    runs = await asyncio.to_thread(topics.list_runs, topic_id)
    return {
        "status": "success",
        "topic": _admin_topic_view(topic),
        "runs": [topics.run_public_view(run) for run in reversed(runs)],
        "follower_count": len(await asyncio.to_thread(topics.list_followers, topic_id)),
    }


@router.put("/api/admin/topics/{topic_id}")
@limiter.limit("20/minute")
async def admin_update_topic(
    request: Request, topic_id: str, data: dict = Body(...)
):
    actor_uid = _require_admin(request, data)
    payload = {key: value for key, value in data.items() if key != "id_token"}
    try:
        topic = await asyncio.to_thread(
            topics.update_topic, topic_id, payload, actor_uid=actor_uid
        )
        return {"status": "success", "topic": _admin_topic_view(topic)}
    except topics.TopicError as exc:
        _raise_topic(exc)
    except Exception:
        logging.exception("admin_update_topic failed")
        raise HTTPException(status_code=500, detail="Failed to update topic")


@router.post("/api/admin/topics/{topic_id}/runs")
@limiter.limit("10/minute")
async def admin_create_topic_run(
    request: Request,
    background_tasks: BackgroundTasks,
    topic_id: str,
    data: dict = Body(...),
):
    actor_uid = _require_admin(request, data)
    payload = {key: value for key, value in data.items() if key != "id_token"}
    try:
        topic_before = await asyncio.to_thread(topics.get_topic, topic_id)
        if not topic_before:
            raise topics.TopicError("not_found", "Topic not found.")
        old_score = topic_before.get("latest_agreement_score")
        if str(payload.get("consensus_md") or "").strip():
            # Explicit legacy/editorial import. Normal admin runs send an empty
            # body and execute the configured models plus source collection.
            run = await asyncio.to_thread(
                topics.create_run, topic_id, payload, actor_uid=actor_uid
            )
        else:
            run = await asyncio.to_thread(
                topic_runner.run_topic_now, topic_id, actor_uid=actor_uid
            )
        topic_after = await asyncio.to_thread(topics.get_topic, topic_id)
        should_notify = (
            run["change_type"] in {"minor", "major"} and mailer.is_configured()
        )
        if should_notify:
            background_tasks.add_task(
                _notify_topic_followers, topic_after, run, old_score
            )
        return {
            "status": "success",
            "run": topics.run_public_view(run),
            "notifications_queued": should_notify,
            "notification_warning": (
                "Run saved, but SMTP is not configured; follower e-mails were not queued."
                if run["change_type"] in {"minor", "major"} and not should_notify
                else ""
            ),
        }
    except topics.TopicError as exc:
        _raise_topic(exc)
    except Exception:
        logging.exception("admin_create_topic_run failed")
        raise HTTPException(status_code=500, detail="Topic run failed")
