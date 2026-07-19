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
        coords.append({**point, "x": round(x, 1), "y": round(y, 1), "score_event": score_event})
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


def _mini_spark(history_points, width=120, height=34):
    """Kompakte Sparkline fürs Scoreboard – auf den echten Score-Bereich
    skaliert (min. 20 Punkte Spanne), sonst wirkt jede Kurve flach."""
    scores = [p["agreement_score"] for p in history_points][-16:]
    if len(scores) < 2:
        return None
    low, high = min(scores), max(scores)
    span = max(20, high - low + 10)
    lo = max(0, min(100 - span, (low + high) / 2 - span / 2))
    pad = 3
    step = (width - 2 * pad) / (len(scores) - 1)
    coords = [
        (round(pad + index * step, 1),
         round(pad + (height - 2 * pad) * (1 - (score - lo) / span), 1))
        for index, score in enumerate(scores)
    ]
    path = " ".join(
        ("M" if index == 0 else "L") + f" {x} {y}" for index, (x, y) in enumerate(coords)
    )
    return {"width": width, "height": height, "path": path,
            "last_x": coords[-1][0], "last_y": coords[-1][1]}


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

    agreement, model_count, contradiction_count = _score_stats(data)
    try:
        history_points = snapshots.list_watch_history(share_id)
    except Exception:
        logging.exception("share_og_card history failed")
        history_points = []
    history_scores = [p["agreement_score"] for p in history_points]
    score = history_scores[-1] if history_scores else agreement.get("score")
    checked_label = ""
    if history_points:
        checked_label = "Tracked since " + history_points[0]["ts"].strftime("%b %Y")
    png = og_image.share_card_png(
        share_id,
        question=str(data.get("question") or ""),
        score=score if isinstance(score, (int, float)) else None,
        model_count=model_count,
        contradiction_count=contradiction_count,
        history_scores=history_scores,
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

    payload = snapshots.public_share_payload(data)
    page_url = SITE_URL + snapshots.share_path("" if is_private else canonical_slug, share_id)
    consensus_html = render_public_markdown(payload["consensus_md"], payload["sources"])

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

    try:
        history_points = snapshots.list_watch_history(share_id)
    except Exception:
        logging.exception("list_watch_history failed")
        history_points = []
    watch_history = _build_watch_history_view(history_points)
    try:
        current_watch_meta = watch_service.get_public_watch_meta(share_id)
    except Exception:
        logging.exception("get_public_watch_meta failed")
        current_watch_meta = None
    watch_page = _build_watch_page_meta(current_watch_meta, history_points)

    date_iso = payload["answered_at"] or payload["created_at"]
    # Watch-Seiten sind lebende Dokumente: der letzte Run ist das echte
    # dateModified (Freshness-Signal für Google), nicht das Erstelldatum.
    modified_iso = (
        history_points[-1]["ts"].isoformat() if history_points else date_iso
    )

    # Verdict-Scoreboard: der datendichte Einstieg über der Konsens-Antwort.
    agreement_data = differences_data.get("agreement")
    agreement_data = agreement_data if isinstance(agreement_data, dict) else {}
    base_score = agreement_data.get("score")
    latest_score = (
        watch_history["latest_score"] if watch_history
        else base_score if isinstance(base_score, (int, float)) else None
    )
    scoreboard = {
        "score": int(latest_score) if isinstance(latest_score, (int, float)) else None,
        "level": str(agreement_data.get("level") or ""),
        "model_count": model_count,
        "contradiction_count": contradiction_count,
        "source_count": len(sources_view),
        "checks": len(history_points),
        "tracked_since": history_points[0]["ts"].strftime("%b %Y") if history_points else "",
        "last_checked": (
            history_points[-1]["ts"].strftime("%Y-%m-%d") if history_points
            else (date_iso[:10] if date_iso else "")
        ),
        "spark": _mini_spark(history_points),
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
        meta_description = markdown_to_plaintext(payload["consensus_md"], limit=160)

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
        "datePublished": date_iso,
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
        "can_follow": bool(not is_private and watch_page and watch_page["is_active"]),
        "sources": sources_view,
        "related_shares": related_shares,
        "watch_history": watch_history,
        "watch_page": watch_page,
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
        "citation_text": snapshots.build_citation(payload, page_url),
    })
    response.headers["X-Robots-Tag"] = robots_meta
    response.headers["Cache-Control"] = "private, no-store" if is_private else SHARE_CACHE_CONTROL
    return response
