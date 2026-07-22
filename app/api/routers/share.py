import json
import logging
import re
from datetime import datetime, timezone
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Request, Body, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from app.core.rate_limit import limiter
from app.core.security import verify_user_token, extract_id_token, is_user_admin
from app.api.routers.pages import SITE_URL
from app.services import og_image
from app.services import share_snapshots as snapshots
from app.services import watch_service
from app.services.share_snapshots import ShareError
from app.services.public_markdown import (
    render_public_markdown,
    markdown_to_plaintext,
    source_site_name,
)

# Browser 5 min, (zukünftiges) CDN 1 Tag; Invalidierung bei Revoke/Block läuft
# über den In-Process-Cache + kurze max-age, ein CDN gibt es bewusst nicht.
SHARE_CACHE_CONTROL = "public, max-age=300, s-maxage=86400, stale-while-revalidate=86400"

templates = Jinja2Templates(directory="templates")

router = APIRouter()

_SHARE_ERROR_STATUS = {
    "not_found": 404,
    "forbidden": 403,
    "quota_exceeded": 429,
}

# "BestModel: Grok" (ggf. mit Markdown-Sternchen) im Differences-Freitext.
_BEST_MODEL_RE = re.compile(
    r"^\s*\**\s*Best\s*Model\s*\**\s*[:\-]\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _build_watch_history_view(points):
    if not points:
        return None
    width, height = 640, 190
    left, right, top, bottom = 38, 18, 18, 30
    plot_w, plot_h = width - left - right, height - top - bottom
    count = len(points)
    coords = []
    for index, point in enumerate(points):
        x = left + (plot_w * index / (count - 1) if count > 1 else plot_w / 2)
        y = top + plot_h * (100 - point["agreement_score"]) / 100
        previous_score = points[index - 1]["agreement_score"] if index else None
        score_event = previous_score is not None and abs(point["agreement_score"] - previous_score) >= 15
        trigger = point.get("trigger")
        if trigger not in {"stable", "changed"}:
            trigger = "changed" if point.get("changed") or score_event else "stable"
        coords.append({
            **point,
            "trigger": trigger,
            "x": round(x, 1),
            "y": round(y, 1),
            "score_event": score_event,
        })
    path = " ".join(
        ("M" if index == 0 else "L") + f" {point['x']} {point['y']}"
        for index, point in enumerate(coords)
    )
    events = [point for point in reversed(coords) if point["changed"] or point["score_event"]]
    mapped_points = [point for point in coords if point.get("opinion_map")]
    position_view = None
    if mapped_points:
        latest_map = mapped_points[-1]["opinion_map"]
        providers = []
        for point in mapped_points:
            for model in point["opinion_map"].get("models") or []:
                provider = model.get("provider")
                if provider and provider not in providers:
                    providers.append(provider)
        trajectories = []
        for provider in providers:
            cells = []
            for point in mapped_points:
                model = next((
                    item for item in point["opinion_map"].get("models") or []
                    if item.get("provider") == provider
                ), None)
                cells.append({
                    "date": point["ts"].strftime("%Y-%m-%d"),
                    "score": model.get("movement_score") if model else None,
                    "moved": bool(model and model.get("moved")),
                    "summary": model.get("summary") if model else "",
                })
            trajectories.append({"provider": provider, "cells": cells})
        position_view = {
            "dates": [point["ts"].strftime("%b %d") for point in mapped_points],
            "trajectories": trajectories,
            "dimensions": latest_map.get("dimensions") or [],
            "shift_score": latest_map.get("shift_score"),
            "shift_label": latest_map.get("shift_label") or "New baseline",
        }
    return {
        "width": width,
        "height": height,
        "path": path,
        "points": coords,
        "events": events,
        "start_date": points[0]["ts"].strftime("%Y-%m-%d"),
        "end_date": points[-1]["ts"].strftime("%Y-%m-%d"),
        "latest_score": points[-1]["agreement_score"],
        "position_map": position_view,
    }


def _build_watch_drift_view(history_points, selected_run_id="", query_first=False):
    """Human-readable drift state for the current or selected Watch version."""
    if selected_run_id == "original":
        return {
            "trigger": "stable",
            "label": "Original baseline",
            "summary": "This is the consensus captured when tracking started.",
            "score_delta": None,
            "direction_shift": None,
            "baseline_summary": "",
        }
    if not history_points:
        return {
            "trigger": "stable",
            "label": "Baseline established",
            "summary": "The next check will show whether the consensus moved.",
            "score_delta": None,
            "direction_shift": None,
            "baseline_summary": "",
        }
    index = len(history_points) - 1
    if selected_run_id:
        for candidate, point in enumerate(history_points):
            if point.get("run_id") == selected_run_id:
                index = candidate
                break
    point = history_points[index]
    if query_first and index == 0:
        return {
            "trigger": "stable",
            "label": "Baseline established",
            "summary": "The first scheduled consensus is ready.",
            "score_delta": None,
            "direction_shift": (point.get("opinion_map") or {}).get("shift_score"),
            "baseline_summary": "",
            "baseline_changed": False,
            "checked_at": point.get("ts"),
        }
    previous_score = history_points[index - 1]["agreement_score"] if index else None
    score = point.get("agreement_score")
    score_delta = (
        int(score) - int(previous_score)
        if isinstance(score, (int, float)) and isinstance(previous_score, (int, float))
        else None
    )
    trigger = point.get("trigger")
    if trigger not in {"stable", "changed"}:
        trigger = "changed" if point.get("changed") or (
            score_delta is not None and abs(score_delta) >= 15
        ) else "stable"
    position = point.get("opinion_map") or {}
    return {
        "trigger": trigger,
        "label": "Changed since last check" if trigger == "changed" else "Stable since last check",
        "summary": (
            point.get("change_summary")
            or ("The consensus moved materially." if trigger == "changed"
                else "No material change was detected in the latest check.")
        ),
        "score_delta": score_delta,
        "direction_shift": position.get("shift_score"),
        "direction_label": position.get("shift_label") or "",
        "baseline_summary": point.get("baseline_summary") or "",
        "baseline_changed": bool(point.get("baseline_changed")),
        "checked_at": point.get("ts"),
    }


def _build_watch_versions_view(history_points, selected_id, latest_id, page_path):
    versions = []
    for point in reversed(history_points):
        run_id = str(point.get("run_id") or "")
        if not run_id or not point.get("has_snapshot"):
            continue
        versions.append({
            "run_id": run_id,
            "date": point["ts"].strftime("%Y-%m-%d %H:%M UTC"),
            "url": f"{page_path}?version={run_id}",
            "trigger": point.get("trigger") or ("changed" if point.get("changed") else "stable"),
            "is_selected": run_id == selected_id,
            "is_current": run_id == latest_id,
        })
    return versions


def _watch_datetime_view(value, timezone_name=""):
    if not isinstance(value, datetime):
        return {"iso": "", "display": ""}
    display_zone = "UTC"
    normalized = value.astimezone(timezone.utc)
    if timezone_name:
        try:
            normalized = value.astimezone(ZoneInfo(timezone_name))
            display_zone = timezone_name
        except (ZoneInfoNotFoundError, ValueError):
            pass
    return {
        "iso": value.astimezone(timezone.utc).isoformat(),
        "display": normalized.strftime("%Y-%m-%d %H:%M ") + display_zone,
    }


def _build_watch_page_meta(meta, history_points):
    if not meta and not history_points:
        return None
    meta = meta or {}
    status = meta.get("status") or "history"
    labels = {
        "active": "Active",
        "paused": "Paused",
        "paused_error": "Paused after errors",
        "history": "Archived history",
    }
    last_run = meta.get("last_run_at")
    if not isinstance(last_run, datetime) and history_points:
        last_run = history_points[-1].get("ts")
    timezone_name = str(meta.get("timezone") or "")
    run_time = str(meta.get("run_time") or "")
    run_weekday = str(meta.get("run_weekday") or "")
    interval = str(meta.get("interval") or "")
    schedule_label = interval.capitalize()
    if interval == "weekly" and run_weekday:
        schedule_label += f" on {run_weekday.capitalize()}"
    if run_time and timezone_name:
        schedule_label += f" at {run_time} ({timezone_name})"
    return {
        "status": status,
        "status_label": labels.get(status, "Paused"),
        "is_active": status == "active",
        "interval": interval,
        "interval_label": interval.capitalize(),
        "schedule_label": schedule_label,
        "last_run": _watch_datetime_view(last_run, timezone_name),
        "next_run": _watch_datetime_view(meta.get("next_run_at"), timezone_name),
        "created": _watch_datetime_view(meta.get("created_at"), timezone_name),
    }


def _score_stats(data):
    """(agreement-dict, model_count, contradiction_count) aus einem Share-Doc."""
    differences_data = data.get("differences_data")
    differences_data = differences_data if isinstance(differences_data, dict) else {}
    agreement = differences_data.get("agreement")
    agreement = agreement if isinstance(agreement, dict) else {}
    model_count = (
        len(differences_data.get("models_compared") or [])
        or len(data.get("included_models") or [])
    )
    contradiction_count = sum(
        1 for d in (differences_data.get("differences") or [])
        if isinstance(d, dict) and d.get("type") == "contradiction"
    )
    return agreement, model_count, contradiction_count


def _resolve_display_version(data, watch_page, requested_version, latest_run_id, share_id):
    """Resolve the only snapshot allowed to supply visible answer content."""
    original = {
        "id": "original",
        "kind": "original",
        "data": data,
        "snapshot": None,
        "fallback_notice": "",
    }
    if not watch_page or requested_version == "original":
        return original
    if requested_version:
        snapshot = snapshots.get_watch_version(share_id, requested_version)
        if snapshot is None:
            return None
        return {
            "id": requested_version,
            "kind": "historical",
            "data": {**data, **snapshot},
            "snapshot": snapshot,
            "fallback_notice": "",
        }
    if not latest_run_id:
        return original
    snapshot = snapshots.get_watch_version(share_id, latest_run_id)
    if snapshot is None:
        original["kind"] = "fallback"
        original["fallback_notice"] = (
            "The latest Watch check has no complete saved version. "
            "Showing the original consensus and all of its original metadata instead."
        )
        return original
    return {
        "id": latest_run_id,
        "kind": "current",
        "data": {**data, **snapshot},
        "snapshot": snapshot,
        "fallback_notice": "",
    }


def _require_uid(request, data):
    id_token = extract_id_token(request, data)
    if not id_token:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        return verify_user_token(id_token)
    except Exception:
        raise HTTPException(status_code=401, detail="Authentication failed")


def _raise_share_error(exc):
    raise HTTPException(status_code=_SHARE_ERROR_STATUS.get(exc.code, 400), detail=exc.message)


@router.post("/api/share")
@limiter.limit("5/minute")
async def create_share(request: Request, data: dict = Body(...)):
    uid = _require_uid(request, data)
    result_id = str(data.get("result_id") or "").strip()
    if not result_id:
        raise HTTPException(status_code=400, detail="Missing required field: result_id")

    try:
        result = snapshots.create_share_from_pending(uid, result_id, visibility="public")
    except ShareError as exc:
        _raise_share_error(exc)
    except Exception:
        logging.exception("create_share failed")
        raise HTTPException(status_code=500, detail="Error creating share link")

    path = snapshots.share_path(result["slug"], result["share_id"])
    return {
        "status": "success",
        "share_id": result["share_id"],
        "path": path,
        "url": SITE_URL + path,
        "created": result["created"],
    }


@router.delete("/api/share/{share_id}")
@limiter.limit("10/minute")
async def delete_share(request: Request, share_id: str, data: dict = Body(default={})):
    uid = _require_uid(request, data)
    try:
        snapshots.revoke_share(share_id, uid, is_admin=is_user_admin(uid))
    except ShareError as exc:
        _raise_share_error(exc)
    except Exception:
        logging.exception("delete_share failed")
        raise HTTPException(status_code=500, detail="Error revoking share link")
    return {"status": "success", "message": "Share link revoked."}


@router.get("/api/my/shares")
@limiter.limit("20/minute")
async def my_shares(request: Request):
    uid = _require_uid(request, {})
    try:
        shares = snapshots.list_shares_for_owner(uid)
    except Exception:
        logging.exception("my_shares failed")
        raise HTTPException(status_code=500, detail="Error loading shares")
    return {"status": "success", "shares": shares, "site_url": SITE_URL}


@router.post("/api/share/{share_id}/indexing-request")
@limiter.limit("10/minute")
async def request_indexing(request: Request, share_id: str, data: dict = Body(default={})):
    """Owner nominiert die eigene öffentliche Seite für den Google-Index.

    Setzt nur ``index_requested`` – die eigentliche Freigabe (``indexed``)
    bleibt eine Admin-Entscheidung in der Moderations-UI.
    """
    uid = _require_uid(request, data)
    want = data.get("want", True)
    if not isinstance(want, bool):
        raise HTTPException(status_code=400, detail="want must be a boolean")
    try:
        state = snapshots.request_share_indexing(share_id, uid, want=want)
    except ShareError as exc:
        _raise_share_error(exc)
    except Exception:
        logging.exception("request_indexing failed")
        raise HTTPException(status_code=500, detail="Error updating listing request")
    return {"status": "success", **state}


@router.post("/api/share/{share_id}/report")
@limiter.limit("3/minute")
async def report_share(request: Request, share_id: str, data: dict = Body(default={})):
    # Bewusst ohne Auth (Besucher sollen melden können) und ohne IP/UA-
    # Speicherung; Missbrauchsschutz nur über das Rate-Limit.
    reason = str(data.get("reason") or "other")
    try:
        snapshots.report_share(share_id, reason)
    except ShareError as exc:
        _raise_share_error(exc)
    except Exception:
        logging.exception("report_share failed")
        raise HTTPException(status_code=500, detail="Error reporting page")
    return {"status": "success", "message": "Thanks, this page has been reported for review."}


def _unavailable_response(request, status_code, heading, message):
    response = templates.TemplateResponse(
        "share_unavailable.html",
        {"request": request, "heading": heading, "message": message},
        status_code=status_code,
    )
    response.headers["X-Robots-Tag"] = "noindex, noarchive"
    return response


@router.get("/sitemap-shares.xml")
@limiter.limit("30/minute")
async def sitemap_shares(request: Request):
    """Nur vom Admin indexierte UND aktive Shares – alles andere ist noindex
    und hat in der Sitemap nichts verloren."""
    try:
        urls = snapshots.list_indexed_share_urls()
    except Exception:
        logging.exception("sitemap_shares failed")
        raise HTTPException(status_code=500, detail="Error building sitemap")

    entries = "\n".join(
        "  <url>\n"
        f"    <loc>{SITE_URL}{item['path']}</loc>\n"
        + (f"    <lastmod>{item['lastmod']}</lastmod>\n" if item["lastmod"] else "")
        + "    <changefreq>monthly</changefreq>\n"
        "  </url>"
        for item in urls
    )
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{entries}\n"
        "</urlset>\n"
    )
    return Response(
        content=xml,
        media_type="application/xml",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/questions", response_class=HTMLResponse)
@limiter.limit("30/minute")
async def questions_hub(request: Request):
    """Öffentliche Hub-Seite: verlinkt alle indexierten Share-Seiten intern.

    SEO-Zweck: indexierte Shares hingen bisher nur in der Sitemap ("verwaiste
    Seiten") – der Hub gibt ihnen einen Crawl-Pfad aus der Hauptnavigation.
    """
    try:
        entries = snapshots.list_hub_shares()
    except Exception:
        logging.exception("list_hub_shares failed")
        entries = []

    watch_count = sum(1 for e in entries if e["is_watch"])
    jsonld = {
        "@context": "https://schema.org",
        "@type": "CollectionPage",
        "@id": SITE_URL + "/questions",
        "url": SITE_URL + "/questions",
        "name": "AI Consensus Library",
        "description": (
            "Questions answered by multiple AI models independently, "
            "cross-checked and scored for agreement."
        ),
        "isPartOf": {"@type": "WebSite", "name": "consens.io", "url": SITE_URL + "/"},
        "mainEntity": {
            "@type": "ItemList",
            "numberOfItems": len(entries),
            "itemListElement": [
                {
                    "@type": "ListItem",
                    "position": index + 1,
                    "name": entry["question"][:110],
                    "url": SITE_URL + entry["path"],
                }
                for index, entry in enumerate(entries[:50])
            ],
        },
    }
    jsonld_html = json.dumps(jsonld, ensure_ascii=False).replace("</", "<\\/")

    response = templates.TemplateResponse("questions.html", {
        "request": request,
        "entries": entries,
        "total_count": len(entries),
        "watch_count": watch_count,
        "jsonld": jsonld_html,
    })
    response.headers["Cache-Control"] = "public, max-age=300"
    return response


@router.get("/s/{slug_id}/og.png")
@limiter.limit("60/minute")
async def share_og_card(request: Request, slug_id: str):
    """Generierte Open-Graph-Karte (PNG) einer öffentlichen Share-Seite."""
    _slug, share_id = snapshots.split_slug_id(slug_id)
    if not snapshots.is_valid_share_id(share_id) or not og_image.is_available():
        raise HTTPException(status_code=404, detail="Not found")
    try:
        data = snapshots.get_share_cached(share_id)
    except Exception:
        logging.exception("share_og_card lookup failed")
        raise HTTPException(status_code=500, detail="Error loading share")
    if (data is None or data.get("status") != "active"
            or str(data.get("visibility") or "public") == "private"):
        raise HTTPException(status_code=404, detail="Not found")

    try:
        history_points = snapshots.list_watch_history(share_id)
    except Exception:
        logging.exception("share_og_card history failed")
        history_points = []
    try:
        watch_meta = watch_service.get_public_watch_meta(share_id)
    except Exception:
        watch_meta = None
    watch_page = _build_watch_page_meta(watch_meta, history_points)
    latest_run_id = str(
        (watch_meta or {}).get("last_successful_run_id")
        or data.get("latest_watch_run_id") or ""
    )
    try:
        display_version = _resolve_display_version(data, watch_page, "", latest_run_id, share_id)
    except Exception:
        display_version = None
    display_data = display_version["data"] if display_version else data
    agreement, model_count, contradiction_count = _score_stats(display_data)
    score = agreement.get("score")
    if not isinstance(score, (int, float)):
        score = display_data.get("agreement_score")
    display_payload = snapshots.public_share_payload(display_data)
    checked_label = (
        "Answered " + display_payload["answered_at"][:10]
        if display_payload["answered_at"] else ""
    )
    png = og_image.share_card_png(
        share_id,
        question=str(data.get("question") or ""),
        score=score if isinstance(score, (int, float)) else None,
        model_count=model_count,
        contradiction_count=contradiction_count,
        history_scores=[],
        checked_label=checked_label,
    )
    if png is None:
        raise HTTPException(status_code=404, detail="Not found")
    return Response(
        content=png,
        media_type="image/png",
        headers={
            "Cache-Control": "public, max-age=3600, stale-while-revalidate=86400",
            "X-Robots-Tag": "noindex",
        },
    )


@router.get("/s/{slug_id}", response_class=HTMLResponse)
@limiter.limit("30/minute")
async def share_page(request: Request, slug_id: str):
    slug, share_id = snapshots.split_slug_id(slug_id)
    if not snapshots.is_valid_share_id(share_id):
        return _unavailable_response(
            request, 404, "Page not found",
            "This shared consensus does not exist.",
        )

    try:
        data = snapshots.get_share_cached(share_id)
    except Exception:
        logging.exception("share_page lookup failed")
        raise HTTPException(status_code=500, detail="Error loading shared page")

    if data is None:
        return _unavailable_response(
            request, 404, "Page not found",
            "This shared consensus does not exist.",
        )
    if data.get("status") != "active":
        # Widerrufen oder gesperrt: dauerhaft weg (410 Gone).
        return _unavailable_response(
            request, 410, "This page is gone",
            "This shared consensus has been removed by its creator or by consens.io.",
        )

    is_private = str(data.get("visibility") or "public") == "private"
    if is_private:
        token = extract_id_token(request, {})
        try:
            viewer_uid = verify_user_token(token) if token else ""
        except Exception:
            viewer_uid = ""
        if not viewer_uid or viewer_uid != data.get("owner_uid"):
            return _unavailable_response(
                request, 403, "Private watch page",
                "This page is private. Sign in with its owner account to view it.",
            )

    canonical_slug = data.get("slug") or ""
    if not is_private and slug != canonical_slug:
        return RedirectResponse(
            url=snapshots.share_path(canonical_slug, share_id), status_code=301
        )

    page_path = snapshots.share_path("" if is_private else canonical_slug, share_id)
    page_url = SITE_URL + page_path

    # Shared Pages stay immutable. Only a page with Watch metadata overlays an
    # immutable version snapshot at render time; the underlying Share document
    # remains the original baseline and keeps its existing API semantics.
    try:
        history_points = snapshots.list_watch_history(share_id)
    except Exception:
        logging.exception("list_watch_history failed")
        history_points = []
    try:
        current_watch_meta = watch_service.get_public_watch_meta(share_id)
    except Exception:
        logging.exception("get_public_watch_meta failed")
        current_watch_meta = None
    watch_page = _build_watch_page_meta(current_watch_meta, history_points)

    requested_version = str(request.query_params.get("version") or "").strip()
    latest_run_id = str(
        (current_watch_meta or {}).get("last_successful_run_id")
        or data.get("latest_watch_run_id")
        or ""
    )
    if watch_page and requested_version and requested_version != "original":
        if not re.fullmatch(r"[A-Za-z0-9]{8,64}", requested_version):
            return _unavailable_response(
                request, 404, "Version not found",
                "This Consensus Watch version does not exist.",
            )
    try:
        display_version = _resolve_display_version(
            data, watch_page, requested_version, latest_run_id, share_id,
        )
    except Exception:
        logging.exception("resolve Watch display version failed")
        display_version = None
    if display_version is None:
        if requested_version:
            return _unavailable_response(
                request, 404, "Version not found",
                "This Consensus Watch version does not exist.",
            )
        display_version = {
            "id": "original", "kind": "fallback", "data": data, "snapshot": None,
            "fallback_notice": (
                "The latest Watch version could not be loaded. "
                "Showing the original consensus and all of its original metadata instead."
            ),
        }

    selected_run_id = display_version["id"]
    selected_version = display_version["snapshot"]
    display_data = display_version["data"]
    payload = snapshots.public_share_payload(display_data)
    consensus_html = render_public_markdown(payload["consensus_md"], payload["sources"])
    watch_awaiting_first_run = bool(
        watch_page and data.get("awaiting_first_watch_run") and not selected_version
    )

    # Indexierung: nur wenn der Admin "indexed" gesetzt hat (nie automatisch).
    is_indexed = bool(data.get("indexed")) and not is_private
    robots_meta = "index, follow" if is_indexed else "noindex, nofollow" if is_private else "noindex, follow"

    # Canonical-Dedup über question_hash: Nicht indexierte Duplikate zeigen
    # auf den ältesten aktiven UND indexierten Share derselben Frage. Ein
    # Canonical zeigt nie auf eine noindex-Seite; ohne Ziel: selbst-kanonisch.
    canonical_url = page_url
    if not is_indexed and data.get("question_hash"):
        try:
            canonical_target = snapshots.find_canonical_share(data["question_hash"])
        except Exception:
            logging.exception("find_canonical_share failed")
            canonical_target = None
        if canonical_target and canonical_target["share_id"] != share_id:
            canonical_url = SITE_URL + snapshots.share_path(
                canonical_target["slug"], canonical_target["share_id"]
            )

    sources_view = []
    for source in payload["sources"]:
        match = re.match(r"^S(\d+)$", str(source.get("id") or ""))
        url = source.get("url") or ""
        try:
            domain = re.sub(r"^www\.", "", urlsplit(url).hostname or "")
        except ValueError:
            domain = ""
        # Nichtssagende Titel (leer, nur eine Zahl oder die nackte URL)
        # fallen auf die Domain zurück, damit Quellen zuordenbar bleiben.
        title = str(source.get("title") or "").strip()
        if not title or title == url or re.fullmatch(r"\d+", title):
            title = domain or url
        site_label = source_site_name(url)
        sources_view.append({
            "num": match.group(1) if match else "",
            "id": source.get("id") or "",
            "title": title,
            "url": url,
            "domain": domain,
            "site": site_label,
        })

    # Differences: strukturierte Karten, sonst Freitext-Fallback (markdown-
    # gerendert). Beides read-only aus dem Snapshot – keine LLM-Calls.
    differences_data = payload["differences_data"] if isinstance(payload["differences_data"], dict) else {}
    differences = differences_data.get("differences") or []
    best_model_display = str(differences_data.get("best_model") or "").strip()
    differences_fallback_html = ""
    if not differences and payload["differences_text"]:
        # Die technische "BestModel: X"-Zeile aus dem Freitext ziehen und
        # stattdessen als gestaltetes Element unter den Differences anzeigen.
        fallback_text = str(payload["differences_text"])
        match = _BEST_MODEL_RE.search(fallback_text)
        if match:
            if not best_model_display:
                best_model_display = match.group(1).strip()
            fallback_text = _BEST_MODEL_RE.sub("", fallback_text).strip()
        differences_fallback_html = render_public_markdown(
            fallback_text, payload["sources"]
        )
    model_count = (
        len(differences_data.get("models_compared") or [])
        or len(payload["included_models"])
    )
    contradiction_count = sum(1 for d in differences if d.get("type") == "contradiction")

    # "Verwandte Fragen": nur indexierte, aktive Shares (read-only, gecacht).
    related_shares = []
    if not is_private:
        try:
            related_shares = snapshots.list_related_shares(share_id, payload["question"])
        except Exception:
            logging.exception("list_related_shares failed")

    watch_history = _build_watch_history_view(history_points)
    watch_drift = _build_watch_drift_view(
        history_points,
        selected_run_id,
        query_first=bool(data.get("watch_query_only")),
    ) if watch_page else None
    watch_versions = _build_watch_versions_view(
        history_points, selected_run_id, latest_run_id, page_path,
    ) if watch_page else []

    date_iso = payload["answered_at"] or payload["created_at"]
    published_iso = date_iso
    # Auch strukturierte Zeitangaben bleiben an die autoritative Anzeigeversion
    # gebunden; kompakte History-Metadaten dürfen hier nicht einsickern.
    modified_iso = date_iso

    # Verdict-Scoreboard: der datendichte Einstieg über der Konsens-Antwort.
    agreement_data = differences_data.get("agreement")
    agreement_data = agreement_data if isinstance(agreement_data, dict) else {}
    base_score = agreement_data.get("score")
    if not isinstance(base_score, (int, float)):
        base_score = display_data.get("agreement_score")
    latest_score = base_score if isinstance(base_score, (int, float)) else None
    scoreboard = {
        "score": int(latest_score) if isinstance(latest_score, (int, float)) else None,
        "level": str(agreement_data.get("level") or ""),
        "model_count": model_count,
        "contradiction_count": contradiction_count,
        "source_count": len(sources_view),
        "checks": 0,
        "tracked_since": "",
        "last_checked": date_iso[:10] if date_iso else "",
        "spark": None,
    }

    # Meta-Description datengeführt statt AI-Textanfang: liest sich im SERP
    # wie ein Datenprodukt, der Konsens-Auszug folgt dahinter.
    description_bits = []
    if model_count:
        description_bits.append(f"{model_count} AI models answered independently")
    if scoreboard["score"] is not None:
        description_bits.append(f"agreement {scoreboard['score']}/100")
    if contradiction_count:
        description_bits.append(
            f"{contradiction_count} contradiction{'s' if contradiction_count != 1 else ''}"
        )
    if scoreboard["tracked_since"]:
        description_bits.append(f"tracked since {scoreboard['tracked_since']}")
    if description_bits:
        prefix = " · ".join(description_bits) + ". "
        excerpt = markdown_to_plaintext(
            payload["consensus_md"], limit=max(40, 160 - len(prefix))
        )
        meta_description = prefix + excerpt
    else:
        meta_description = (
            f"Scheduled Consensus Watch for: {payload['question']}"[:160]
            if watch_awaiting_first_run
            else markdown_to_plaintext(payload["consensus_md"], limit=160)
        )

    # Generierte OG-Karte (Scoreboard als Bild) statt Favicon, wenn möglich.
    og_is_card = og_image.is_available() and not is_private
    og_image_url = (
        page_url + "/og.png" if og_is_card
        else SITE_URL + "/static/favicon-square.png"
    )

    jsonld = {
        "@context": "https://schema.org",
        "@type": "Article",
        "headline": payload["question"][:110],
        "description": meta_description,
        "datePublished": published_iso or date_iso,
        "dateModified": modified_iso,
        "mainEntityOfPage": {"@type": "WebPage", "@id": page_url},
        "author": {"@type": "Organization", "name": "consens.io", "url": SITE_URL},
        "publisher": {
            "@type": "Organization",
            "name": "consens.io",
            "logo": {"@type": "ImageObject", "url": SITE_URL + "/static/favicon-square.png"},
        },
        "citation": [s["url"] for s in sources_view if s["url"]][:10],
        "isAccessibleForFree": True,
    }
    if og_is_card:
        jsonld["image"] = og_image_url
    # "</" escapen, damit Snapshot-Inhalte das <script>-Element nie schließen können.
    jsonld_html = json.dumps(jsonld, ensure_ascii=False).replace("</", "<\\/")

    response = templates.TemplateResponse("share.html", {
        "request": request,
        "share_id": share_id,
        "is_private": is_private,
        "question": payload["question"],
        "consensus_html": consensus_html,
        "differences": differences,
        "differences_fallback_html": differences_fallback_html,
        "best_model_display": best_model_display,
        "has_differences_view": bool(differences or differences_fallback_html or differences_data),
        "model_count": model_count,
        "contradiction_count": contradiction_count,
        "scoreboard": scoreboard,
        "can_follow": bool(
            not is_private and watch_page and watch_page["is_active"]
            and not watch_awaiting_first_run
        ),
        "sources": sources_view,
        "related_shares": related_shares,
        "watch_history": watch_history,
        "watch_page": watch_page,
        "watch_awaiting_first_run": watch_awaiting_first_run,
        "watch_drift": watch_drift,
        "watch_versions": watch_versions,
        "watch_selected_version": {
            "id": selected_run_id,
            "kind": display_version["kind"],
            "is_original": display_version["kind"] in {"original", "fallback"},
            "is_current": display_version["kind"] == "current",
            "is_historical": bool(requested_version),
            "fallback_notice": display_version["fallback_notice"],
            "current_url": page_path,
            "original_url": page_path + "?version=original",
        } if watch_page else None,
        "included_models": payload["included_models"],
        "consulted_models": snapshots.consulted_models_view(payload["included_models"]),
        "consensus_model": payload["consensus_model"],
        "consensus_model_view": snapshots.consensus_model_view(payload["consensus_model"]),
        "date_display": date_iso[:10] if date_iso else "",
        "canonical_url": canonical_url,
        "page_url": page_url,
        "robots_meta": robots_meta,
        "meta_description": meta_description,
        "og_image": og_image_url,
        "og_is_card": og_is_card,
        "jsonld": jsonld_html,
        # Zitation immer mit der eigenen URL der Seite (nicht dem Dedup-Canonical).
        "citation_text": snapshots.build_citation(
            payload,
            page_url + (f"?version={selected_run_id}" if requested_version else ""),
        ),
    })
    response.headers["X-Robots-Tag"] = robots_meta
    if is_private:
        response.headers["Cache-Control"] = "private, no-store"
    elif watch_page and requested_version:
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    elif watch_page:
        response.headers["Cache-Control"] = (
            "public, max-age=60, s-maxage=300, stale-while-revalidate=300"
        )
    else:
        response.headers["Cache-Control"] = SHARE_CACHE_CONTROL
    return response
