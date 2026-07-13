import html
import logging

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import HTMLResponse

from app.core.rate_limit import limiter
from app.core.security import extract_id_token, is_user_pro, verify_user_token
from app.services import watch_brief, watch_service


router = APIRouter()
_STATUS_BY_CODE = {"not_found": 404, "forbidden": 403, "pro_required": 403, "limit_reached": 429, "already_exists": 409, "invalid_token": 400, "expired_token": 410}


def _uid(request: Request, data: dict) -> str:
    token = extract_id_token(request, data)
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        return verify_user_token(token)
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Authentication failed") from exc


def _raise(exc: watch_service.WatchError):
    raise HTTPException(status_code=_STATUS_BY_CODE.get(exc.code, 400), detail=exc.message)


@router.post("/api/watch")
@limiter.limit("5/minute")
async def create_watch(request: Request, data: dict = Body(...)):
    uid = _uid(request, data)
    if "visibility" not in data:
        raise HTTPException(status_code=400, detail="Choose whether the watch page is private or public.")
    try:
        watch = watch_service.create_watch(
            uid,
            interval=data.get("interval"),
            email_mode=data.get("email_mode", "changes_only"),
            condition=data.get("condition", ""),
            visibility=data.get("visibility", "public"),
            run_time=data.get("run_time", ""),
            timezone_name=data.get("timezone", ""),
            is_pro=is_user_pro(uid),
            result_id=data.get("result_id"),
            share_id=data.get("share_id"),
        )
    except watch_service.WatchError as exc:
        _raise(exc)
    except Exception:
        logging.exception("create_watch failed")
        raise HTTPException(status_code=500, detail="Error creating watch")
    return {"status": "success", "watch": watch}


@router.get("/api/my/watches")
@limiter.limit("20/minute")
async def my_watches(request: Request):
    uid = _uid(request, {})
    try:
        return {"status": "success", "watches": watch_service.list_watches(uid, include_history=True)}
    except Exception:
        logging.exception("my_watches failed")
        raise HTTPException(status_code=500, detail="Error loading watches")


@router.patch("/api/watch/{watch_id}")
@limiter.limit("10/minute")
async def patch_watch(request: Request, watch_id: str, data: dict = Body(...)):
    uid = _uid(request, data)
    try:
        watch = watch_service.update_watch(uid, watch_id, data, is_user_pro(uid))
    except watch_service.WatchError as exc:
        _raise(exc)
    except Exception:
        logging.exception("patch_watch failed")
        raise HTTPException(status_code=500, detail="Error updating watch")
    return {"status": "success", "watch": watch}


@router.delete("/api/watch/{watch_id}")
@limiter.limit("10/minute")
async def remove_watch(request: Request, watch_id: str, data: dict = Body(default={})):
    uid = _uid(request, data)
    try:
        watch_service.delete_watch(uid, watch_id)
    except watch_service.WatchError as exc:
        _raise(exc)
    except Exception:
        logging.exception("delete_watch failed")
        raise HTTPException(status_code=500, detail="Error deleting watch")
    return {"status": "success"}


@router.get("/api/my/watch-brief")
@limiter.limit("20/minute")
async def my_watch_brief(request: Request):
    uid = _uid(request, {})
    try:
        return {"status": "success", "brief": watch_brief.get_brief(uid)}
    except Exception:
        logging.exception("my_watch_brief failed")
        raise HTTPException(status_code=500, detail="Error loading brief settings")


@router.patch("/api/my/watch-brief")
@limiter.limit("10/minute")
async def patch_watch_brief(request: Request, data: dict = Body(...)):
    uid = _uid(request, data)
    changes = {key: value for key, value in data.items() if key != "id_token"}
    try:
        brief = watch_brief.update_brief(uid, changes)
    except watch_service.WatchError as exc:
        _raise(exc)
    except Exception:
        logging.exception("patch_watch_brief failed")
        raise HTTPException(status_code=500, detail="Error updating brief settings")
    return {"status": "success", "brief": brief}


def _unsubscribe_page(heading: str, message: str, status_code: int) -> HTMLResponse:
    content = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<meta name='robots' content='noindex'><title>Consensus Watch</title></head>"
        "<body style='font-family:system-ui;max-width:640px;margin:12vh auto;padding:24px'>"
        f"<h1>{html.escape(heading)}</h1><p>{html.escape(message)}</p><p><a href='/'>Return to consens.io</a></p></body></html>"
    )
    return HTMLResponse(content, status_code=status_code, headers={"X-Robots-Tag": "noindex, noarchive"})


@router.get("/watch/unsubscribe", response_class=HTMLResponse)
@limiter.limit("20/minute")
async def unsubscribe(request: Request, token: str = ""):
    try:
        watch_service.unsubscribe(token)
        heading, message, status_code = "Watch paused", "You will no longer receive updates for this consensus watch.", 200
    except watch_service.WatchError as exc:
        heading, message, status_code = "Unable to unsubscribe", exc.message, _STATUS_BY_CODE.get(exc.code, 400)
    return _unsubscribe_page(heading, message, status_code)


@router.get("/watch/brief/unsubscribe", response_class=HTMLResponse)
@limiter.limit("20/minute")
async def unsubscribe_brief(request: Request, token: str = ""):
    try:
        watch_brief.unsubscribe_brief(token)
        heading, message, status_code = "Morning brief disabled", "You will no longer receive the daily watch digest. Individual watch alerts are unaffected.", 200
    except watch_service.WatchError as exc:
        heading, message, status_code = "Unable to unsubscribe", exc.message, _STATUS_BY_CODE.get(exc.code, 400)
    return _unsubscribe_page(heading, message, status_code)
