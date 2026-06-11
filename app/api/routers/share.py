import logging
import re

from fastapi import APIRouter, Request, Body, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.core.rate_limit import limiter
from app.core.security import verify_user_token, extract_id_token, is_user_admin
from app.api.routers.pages import SITE_URL
from app.services import share_snapshots as snapshots
from app.services.share_snapshots import ShareError
from app.services.public_markdown import render_public_markdown

templates = Jinja2Templates(directory="templates")

router = APIRouter()

_SHARE_ERROR_STATUS = {
    "not_found": 404,
    "forbidden": 403,
    "quota_exceeded": 429,
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


def _unavailable_response(request, status_code, heading, message):
    response = templates.TemplateResponse(
        "share_unavailable.html",
        {"request": request, "heading": heading, "message": message},
        status_code=status_code,
    )
    response.headers["X-Robots-Tag"] = "noindex, noarchive"
    return response


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
        data = snapshots.get_share(share_id)
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
    canonical_url = SITE_URL + snapshots.share_path(canonical_slug, share_id)
    consensus_html = render_public_markdown(payload["consensus_md"], payload["sources"])

    sources_view = []
    for source in payload["sources"]:
        match = re.match(r"^S(\d+)$", str(source.get("id") or ""))
        sources_view.append({
            "num": match.group(1) if match else "",
            "id": source.get("id") or "",
            "title": source.get("title") or source.get("url") or "",
            "url": source.get("url") or "",
        })

    date_iso = payload["answered_at"] or payload["created_at"]
    response = templates.TemplateResponse("share.html", {
        "request": request,
        "question": payload["question"],
        "consensus_html": consensus_html,
        "sources": sources_view,
        "included_models": payload["included_models"],
        "consensus_model": payload["consensus_model"],
        "date_display": date_iso[:10] if date_iso else "",
        "canonical_url": canonical_url,
        "citation_text": snapshots.build_citation(payload, canonical_url),
    })
    # Etappe 1: hartes noindex; die Qualitätsfilter-/indexed-Logik kommt in Etappe 3.
    response.headers["X-Robots-Tag"] = "noindex, follow"
    return response
