import json
import logging
import re
from urllib.parse import urlsplit

from fastapi import APIRouter, Request, Body, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from app.core.rate_limit import limiter
from app.core.security import verify_user_token, extract_id_token, is_user_admin
from app.api.routers.pages import SITE_URL
from app.services import share_snapshots as snapshots
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
        result = snapshots.create_share_from_pending(uid, result_id)
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

    canonical_slug = data.get("slug") or ""
    if slug != canonical_slug:
        return RedirectResponse(
            url=snapshots.share_path(canonical_slug, share_id), status_code=301
        )

    payload = snapshots.public_share_payload(data)
    page_url = SITE_URL + snapshots.share_path(canonical_slug, share_id)
    consensus_html = render_public_markdown(payload["consensus_md"], payload["sources"])

    # Indexierung: nur wenn der Admin "indexed" gesetzt hat (nie automatisch).
    is_indexed = bool(data.get("indexed"))
    robots_meta = "index, follow" if is_indexed else "noindex, follow"

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
    try:
        related_shares = snapshots.list_related_shares(share_id, payload["question"])
    except Exception:
        logging.exception("list_related_shares failed")
        related_shares = []

    date_iso = payload["answered_at"] or payload["created_at"]
    meta_description = markdown_to_plaintext(payload["consensus_md"], limit=160)

    jsonld = {
        "@context": "https://schema.org",
        "@type": "Article",
        "headline": payload["question"][:110],
        "description": meta_description,
        "datePublished": date_iso,
        "dateModified": date_iso,
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
    # "</" escapen, damit Snapshot-Inhalte das <script>-Element nie schließen können.
    jsonld_html = json.dumps(jsonld, ensure_ascii=False).replace("</", "<\\/")

    response = templates.TemplateResponse("share.html", {
        "request": request,
        "share_id": share_id,
        "question": payload["question"],
        "consensus_html": consensus_html,
        "differences": differences,
        "differences_fallback_html": differences_fallback_html,
        "best_model_display": best_model_display,
        "has_differences_view": bool(differences or differences_fallback_html or differences_data),
        "model_count": model_count,
        "contradiction_count": contradiction_count,
        "sources": sources_view,
        "related_shares": related_shares,
        "included_models": payload["included_models"],
        "consensus_model": payload["consensus_model"],
        "date_display": date_iso[:10] if date_iso else "",
        "canonical_url": canonical_url,
        "page_url": page_url,
        "robots_meta": robots_meta,
        "meta_description": meta_description,
        "og_image": SITE_URL + "/static/favicon-square.png",
        "jsonld": jsonld_html,
        # Zitation immer mit der eigenen URL der Seite (nicht dem Dedup-Canonical).
        "citation_text": snapshots.build_citation(payload, page_url),
    })
    response.headers["X-Robots-Tag"] = robots_meta
    response.headers["Cache-Control"] = SHARE_CACHE_CONTROL
    return response
