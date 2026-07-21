import asyncio
import html
import logging

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import HTMLResponse

from app.core import config as cfg
from app.core.rate_limit import limiter
from app.core.security import extract_id_token, is_user_pro, verify_user_token
from app.api.routers.pages import SITE_URL
from app.services import mailer, telegram_watch, watch_brief, watch_followers, watch_service


router = APIRouter()
_STATUS_BY_CODE = {
    "not_found": 404, "forbidden": 403, "pro_required": 403,
    "limit_reached": 429, "already_exists": 409, "invalid_token": 400,
    "expired_token": 410, "invalid_email": 400, "not_watched": 404,
    "telegram_not_configured": 503, "telegram_not_linked": 409,
    "telegram_already_linked": 409, "telegram_delivery_failed": 502,
}


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
    if data.get("telegram_enabled") is True and not telegram_watch.get_connection(uid).get("connected"):
        raise HTTPException(status_code=409, detail="Connect Telegram before enabling it for a watch.")
    try:
        watch = watch_service.create_watch(
            uid,
            interval=data.get("interval"),
            email_mode=data.get("email_mode", "changes_only"),
            email_enabled=data.get("email_enabled", True),
            telegram_enabled=data.get("telegram_enabled", False),
            condition=data.get("condition", ""),
            visibility=data.get("visibility", "public"),
            run_weekday=data.get("run_weekday", ""),
            run_time=data.get("run_time", ""),
            timezone_name=data.get("timezone", ""),
            is_pro=is_user_pro(uid),
            result_id=data.get("result_id"),
            share_id=data.get("share_id"),
            question=data.get("question"),
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
        watches = watch_service.list_watches(uid, include_history=True)
        is_pro = is_user_pro(uid)
        active_count = sum(1 for watch in watches if watch.get("status") == "active")
        active_limit = cfg.get_watch_active_limit(is_pro)
        return {
            "status": "success",
            "watches": watches,
            "limits": {
                "plan": "pro" if is_pro else "free",
                "active_count": active_count,
                "active_limit": active_limit,
                "remaining": max(0, active_limit - active_count),
                "paused_count": len(watches) - active_count,
                "daily_available": is_pro,
            },
        }
    except Exception:
        logging.exception("my_watches failed")
        raise HTTPException(status_code=500, detail="Error loading watches")


@router.patch("/api/watch/{watch_id}")
@limiter.limit("10/minute")
async def patch_watch(request: Request, watch_id: str, data: dict = Body(...)):
    uid = _uid(request, data)
    changes = {key: value for key, value in data.items() if key != "id_token"}
    if changes.get("telegram_enabled") is True and not telegram_watch.get_connection(uid).get("connected"):
        raise HTTPException(status_code=409, detail="Connect Telegram before enabling it for a watch.")
    try:
        watch = watch_service.update_watch(uid, watch_id, changes, is_user_pro(uid))
    except watch_service.WatchError as exc:
        _raise(exc)
    except Exception:
        logging.exception("patch_watch failed")
        raise HTTPException(status_code=500, detail="Error updating watch")
    return {"status": "success", "watch": watch}


@router.get("/api/my/telegram")
@limiter.limit("20/minute")
async def my_telegram(request: Request):
    uid = _uid(request, {})
    try:
        return {"status": "success", "telegram": telegram_watch.get_connection(uid)}
    except Exception:
        logging.exception("my_telegram failed")
        raise HTTPException(status_code=500, detail="Error loading Telegram connection")


@router.post("/api/my/telegram/link")
@limiter.limit("5/minute")
async def create_telegram_link(request: Request, data: dict = Body(default={})):
    uid = _uid(request, data)
    try:
        link = telegram_watch.create_link(uid)
    except watch_service.WatchError as exc:
        _raise(exc)
    except Exception:
        logging.exception("create_telegram_link failed")
        raise HTTPException(status_code=500, detail="Error creating Telegram link")
    return {"status": "success", **link}


@router.post("/api/my/telegram/test")
@limiter.limit("3/minute")
async def test_telegram(request: Request, data: dict = Body(default={})):
    uid = _uid(request, data)
    try:
        result = await asyncio.to_thread(telegram_watch.send_test, uid)
    except watch_service.WatchError as exc:
        _raise(exc)
    except Exception:
        logging.exception("test_telegram failed")
        raise HTTPException(status_code=500, detail="Error testing Telegram connection")
    return {"status": "success", "delivery": result}


@router.delete("/api/my/telegram")
@limiter.limit("5/minute")
async def disconnect_telegram(request: Request, data: dict = Body(default={})):
    uid = _uid(request, data)
    try:
        state = telegram_watch.disconnect(uid)
    except Exception:
        logging.exception("disconnect_telegram failed")
        raise HTTPException(status_code=500, detail="Error disconnecting Telegram")
    return {"status": "success", "telegram": state}


@router.post("/api/telegram/webhook")
async def telegram_webhook(request: Request, data: dict = Body(default={})):
    if not telegram_watch.verify_webhook_secret(
        request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
    ):
        raise HTTPException(status_code=403, detail="Invalid Telegram webhook secret")
    try:
        await asyncio.to_thread(telegram_watch.handle_update, data)
    except Exception:
        # Return 200 so one malformed update cannot create an endless Telegram
        # retry loop; action-level errors are acknowledged inside the handler.
        logging.exception("telegram_webhook update failed")
    return {"ok": True}


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
    try:
        watch_brief.disable_if_no_watches(uid)
    except Exception:
        # Die Watch ist bereits gelöscht; Brief-Cleanup darf dem Client keinen
        # falschen Delete-Fehler melden. Das Aktivierungs-Gate bleibt intakt.
        logging.exception("Morning Brief cleanup after final watch failed")
    return {"status": "success"}


@router.get("/api/my/watch-brief")
@limiter.limit("20/minute")
async def my_watch_brief(request: Request):
    uid = _uid(request, {})
    try:
        return {
            "status": "success",
            "brief": watch_brief.get_brief(uid),
            "has_watches": watch_brief.has_watches(uid),
        }
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


def _unsubscribe_page(heading: str, message: str, status_code: int,
                      link_href: str = "/", link_label: str = "Return to consens.io") -> HTMLResponse:
    content = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<meta name='robots' content='noindex'><title>Consensus Watch</title></head>"
        "<body style='font-family:system-ui;max-width:640px;margin:12vh auto;padding:24px'>"
        f"<h1>{html.escape(heading)}</h1><p>{html.escape(message)}</p>"
        f"<p><a href='{html.escape(link_href)}'>{html.escape(link_label)}</a></p></body></html>"
    )
    return HTMLResponse(content, status_code=status_code, headers={"X-Robots-Tag": "noindex, noarchive"})


@router.post("/api/share/{share_id}/follow")
@limiter.limit("3/minute")
async def follow_share(request: Request, share_id: str, data: dict = Body(default={})):
    """Besucher-Follow (Double-Opt-in): sendet nur eine Bestätigungs-Mail.

    Bewusst ohne Auth (Besucher!) und ohne IP/UA-Speicherung; persistiert
    wird erst beim Klick auf den Bestätigungslink. Die Antwort ist immer
    generisch, damit Adressen nicht enumeriert werden können.
    """
    try:
        pending = watch_followers.request_follow(share_id, data.get("email"))
    except watch_service.WatchError as exc:
        _raise(exc)
    except Exception:
        logging.exception("follow_share failed")
        raise HTTPException(status_code=500, detail="Error processing follow request")
    if pending["token"]:
        if not mailer.is_configured():
            raise HTTPException(status_code=503, detail="E-mail delivery is not configured.")
        share_url = SITE_URL + "/s/" + share_id
        confirm_url = SITE_URL + "/watch/follow/confirm?token=" + pending["token"]
        sent = await mailer.send_message(mailer.build_follow_confirm_message(
            recipient=pending["email"], question=pending["question"],
            confirm_url=confirm_url, share_url=share_url,
        ))
        if not sent:
            raise HTTPException(status_code=502, detail="Confirmation e-mail could not be sent. Please try again later.")
    return {"status": "success", "message": "Check your inbox and confirm to start following."}


@router.get("/watch/follow/confirm", response_class=HTMLResponse)
@limiter.limit("20/minute")
async def follow_confirm(request: Request, token: str = ""):
    try:
        result = watch_followers.confirm_follow(token)
        question = " ".join(str(result.get("question") or "").split())[:120]
        return _unsubscribe_page(
            "You're following this question",
            f"We will e-mail you when the AI consensus on \"{question}\" shifts materially.",
            200, link_href=result.get("share_path") or "/", link_label="Open the consensus page",
        )
    except watch_service.WatchError as exc:
        return _unsubscribe_page("Unable to confirm", exc.message, _STATUS_BY_CODE.get(exc.code, 400))


@router.get("/watch/follow/unsubscribe", response_class=HTMLResponse)
@limiter.limit("20/minute")
async def follow_unsubscribe(request: Request, token: str = ""):
    try:
        watch_followers.unsubscribe_follow(token)
        return _unsubscribe_page(
            "Unfollowed", "You will no longer receive updates for this question.", 200,
        )
    except watch_service.WatchError as exc:
        return _unsubscribe_page("Unable to unfollow", exc.message, _STATUS_BY_CODE.get(exc.code, 400))


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
