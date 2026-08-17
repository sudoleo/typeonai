import logging
from typing import Literal, Optional
from firebase_admin import auth, firestore
from fastapi import APIRouter, Request, Body, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from app.core.rate_limit import limiter
from app.core.observability import safe_exception
import app.core.config as cfg
from app.core.security import (
    TierStatusUnavailable,
    verify_user_token,
    extract_id_token,
    is_user_pro,
    invalidate_tier_cache,
    db_firestore,
)
from app.services.usage_repository import (
    FirestoreUsageRepository,
    UsageLimits,
    UsageRunNotFound,
    UsageTransitionError,
)
from app.services.account_deletion import FirestoreAccountDeletion
from app.services import memory_edit, persistence_guard, user_memory
from app.services.user_memory import FirestoreUserMemoryRepository

router = APIRouter()
run_usage_repository = FirestoreUsageRepository(db_firestore)
account_deletion = FirestoreAccountDeletion(db_firestore)
user_memory_repository = FirestoreUserMemoryRepository(db_firestore)
memory_edit_repository = memory_edit.FirestoreMemoryEditRepository(db_firestore)
memory_edit_service = memory_edit.MemoryEditService(memory_edit_repository)


class UsageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id_token: str = Field(min_length=1, max_length=8192)


def _run_limits(is_pro: bool) -> UsageLimits:
    return UsageLimits(
        total=cfg.get_consensus_run_limit(is_pro),
        deep_think=cfg.get_deep_think_run_limit(is_pro),
    )

@router.get("/user_status")
@limiter.limit("20/minute")
def get_user_status(request: Request):
    """
    Prüft den Status des Nutzers (Free vs. Pro) basierend auf dem ID-Token.
    Wird beim Seiten-Load (checkUserStatusOnLoad) aufgerufen.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = auth_header.split(" ")[1]

    try:
        # 1. UID verifizieren
        uid = verify_user_token(token)
        
        # 2. Status aus Firestore holen
        try:
            pro_status = is_user_pro(uid)
        except TierStatusUnavailable:
            raise HTTPException(
                status_code=503,
                detail="Account tier is temporarily unavailable. Please retry.",
            ) from None

        # 3. Limits basierend auf Status setzen
        limit_regular = cfg.get_consensus_run_limit(pro_status)
        limit_deep = cfg.get_deep_think_run_limit(pro_status)

        return {
            "uid": uid,
            "is_pro": pro_status,
            "limit": limit_regular,
            "deep_limit": limit_deep
        }

    except HTTPException:
        raise
    except Exception as exc:
        logging.error("User status check failed category=%s", safe_exception(exc))
        raise HTTPException(status_code=401, detail="Authentication failed")

@router.post("/usage")
@limiter.limit("20/minute")
def get_usage_post(request: Request, data: UsageRequest):
    """
    Liefert den persistenten Run-Stand des aktuellen UTC-Tags zurück.
    """
    token = data.id_token
    
    try:
        uid = verify_user_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Authentication failed")
    
    # 1. Status prüfen
    try:
        pro_status = is_user_pro(uid)
    except TierStatusUnavailable:
        raise HTTPException(
            status_code=503,
            detail="Account tier is temporarily unavailable. Please retry.",
        ) from None

    # 2. Limits festlegen
    limits = _run_limits(pro_status)

    # 3. Persistenten UTC-Tagesstand abrufen. Ein einzelnes Tagesdokument
    #    enthaelt Total- und Deep-Think-Bucket konsistent zusammen.
    snapshot = run_usage_repository.snapshot(uid, limits)

    return {
        "remaining": snapshot.total.remaining,
        "deep_remaining": snapshot.deep_think.remaining,
        "is_pro": pro_status,
        "total_limit": snapshot.total.limit,
        "deep_total_limit": snapshot.deep_think.limit,
        "reserved": snapshot.total.reserved,
        "consumed": snapshot.total.consumed,
        "utc_date": snapshot.utc_date,
    }


@router.post("/usage/run/release")
@limiter.limit("20/minute")
def release_usage_run(request: Request, data: dict = Body(...)):
    token = extract_id_token(request, data)
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        uid = verify_user_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Authentication failed")

    key = data.get("usage_run_key")
    if not isinstance(key, str) or not key.strip():
        raise HTTPException(status_code=400, detail="Missing usage_run_key")
    try:
        result = run_usage_repository.release(uid, key.strip())
    except UsageRunNotFound:
        raise HTTPException(status_code=404, detail="Usage run not found") from None
    except UsageTransitionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None

    return {
        "status": result.status.value,
        "remaining": result.snapshot.total.remaining,
        "deep_remaining": result.snapshot.deep_think.remaining,
        "total_limit": result.snapshot.total.limit,
        "deep_total_limit": result.snapshot.deep_think.limit,
        "utc_date": result.utc_date,
    }

class UserMemoryRequest(BaseModel):
    """Das selbst geschriebene Profil samt grosser, manueller Notiz.

    Die Grenzen hier sind Missbrauchsschutz; die verbindlichen Laengen setzt
    ``user_memory.sanitize_profile``. Grosszuegigere API-Werte verhindern 422er
    fuer Text, der nach der Normalisierung laengst passt.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    role: str = Field(default="", max_length=1000)
    focus: str = Field(default="", max_length=1000)
    style: str = Field(default="", max_length=1000)
    constraints: str = Field(default="", max_length=1000)
    # ``None`` unterscheidet alte Browser, die das additive Feld noch nicht
    # kennen, von einem bewussten Leeren durch die aktuelle UI (``""``).
    notes: Optional[str] = Field(default=None, max_length=30_000)


class UserMemoryEditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    client_request_id: str = Field(min_length=8, max_length=128)
    source_kind: str = Field(min_length=1, max_length=32)
    selected_text: str = Field(min_length=1, max_length=memory_edit.MAX_SOURCE_CHARS)
    intent: Literal["add", "correct"] = "correct"
    # Das autoritative Limit kommt aus der Admin-Konfiguration; der groessere
    # Pydantic-Cap verhindert nur unbeschraenkte Request-Bodies.
    correction: str = Field(min_length=1, max_length=2_000)


class UserMemoryUndoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    revision_id: str = Field(pattern=r"^[0-9a-f]{32}$")


def _memory_uid(request: Request) -> str:
    token = extract_id_token(request, {})
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        return verify_user_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Authentication failed") from None


def _memory_tier(uid: str) -> bool:
    try:
        return is_user_pro(uid)
    except TierStatusUnavailable:
        raise HTTPException(
            status_code=503,
            detail="Account tier is temporarily unavailable. Please retry.",
        ) from None


def _memory_response(profile: dict, *, is_pro: bool) -> dict:
    notes_limit = cfg.get_memory_char_limit(is_pro)
    return {
        "status": "success",
        "memory": profile,
        "limits": {
            "field_chars": user_memory.MAX_FIELD_CHARS,
            "notes_chars": notes_limit,
            "profile_chars": notes_limit + (
                user_memory.MAX_PROFILE_CHARS - user_memory.MAX_NOTES_CHARS
            ),
        },
    }


@router.get("/api/my/memory")
@limiter.limit("30/minute")
def get_user_memory(request: Request):
    uid = _memory_uid(request)
    is_pro = _memory_tier(uid)
    try:
        try:
            profile = user_memory_repository.get(
                uid, max_notes_chars=cfg.get_memory_char_limit(is_pro)
            )
        except TypeError:
            profile = user_memory_repository.get(uid)
    except Exception as exc:
        logging.error("user memory read failed category=%s", safe_exception(exc))
        raise HTTPException(status_code=503, detail="Memory is temporarily unavailable.") from None
    return _memory_response(profile, is_pro=is_pro)


@router.put("/api/my/memory")
@limiter.limit("20/minute")
def put_user_memory(request: Request, payload: UserMemoryRequest):
    uid = _memory_uid(request)
    is_pro = _memory_tier(uid)
    try:
        try:
            profile = user_memory_repository.save(
                uid,
                payload.model_dump(),
                max_notes_chars=cfg.get_memory_char_limit(is_pro),
            )
        except TypeError:
            profile = user_memory_repository.save(uid, payload.model_dump())
    except persistence_guard.AccountDeletionInProgress:
        raise HTTPException(status_code=403, detail="This account is being deleted.") from None
    except Exception as exc:
        logging.error("user memory write failed category=%s", safe_exception(exc))
        raise HTTPException(status_code=503, detail="Memory could not be saved.") from None
    return _memory_response(profile, is_pro=is_pro)


_MEMORY_EDIT_HTTP_STATUS = {
    "disabled": 503,
    "provider_unavailable": 503,
    "provider_failed": 502,
    "invalid_model_patch": 502,
    "daily_limit": 429,
    "minute_limit": 429,
    "global_limit": 429,
    "edit_in_progress": 409,
    "idempotency_conflict": 409,
    "revision_conflict": 409,
    "undo_expired": 409,
    "revision_not_found": 404,
}


def _raise_memory_edit_error(exc: memory_edit.MemoryEditError):
    raise HTTPException(
        status_code=_MEMORY_EDIT_HTTP_STATUS.get(exc.code, 422),
        detail={"error_code": exc.code, "message": exc.safe_message},
    ) from None


@router.post("/api/my/memory/edit")
@limiter.limit("10/minute")
def edit_user_memory(request: Request, payload: UserMemoryEditRequest):
    uid = _memory_uid(request)
    is_pro = _memory_tier(uid)
    try:
        result = memory_edit_service.edit(
            uid,
            is_pro=is_pro,
            client_request_id=payload.client_request_id,
            source_kind=payload.source_kind,
            selected_text=payload.selected_text,
            correction=payload.correction,
            intent=payload.intent,
            config=cfg.get_memory_edit_config(),
        )
    except persistence_guard.AccountDeletionInProgress:
        raise HTTPException(status_code=403, detail="This account is being deleted.") from None
    except memory_edit.MemoryEditError as exc:
        _raise_memory_edit_error(exc)
    if result.get("status") == "reserved":
        return JSONResponse(status_code=202, content={"status": "processing"})
    return result


@router.post("/api/my/memory/undo")
@limiter.limit("10/minute")
def undo_user_memory(request: Request, payload: UserMemoryUndoRequest):
    uid = _memory_uid(request)
    is_pro = _memory_tier(uid)
    try:
        return memory_edit_repository.undo(
            uid,
            payload.revision_id,
            memory_limit=cfg.get_memory_char_limit(is_pro),
        )
    except persistence_guard.AccountDeletionInProgress:
        raise HTTPException(status_code=403, detail="This account is being deleted.") from None
    except memory_edit.MemoryEditError as exc:
        _raise_memory_edit_error(exc)


@router.post("/delete_account")
@limiter.limit("3/minute")
def delete_account(request: Request, data: dict = Body(default={})):
    """
    Löscht den Account vollständig (DSGVO Art. 17): Auth-Account, users-Dokument
    inkl. Bookmarks, Usage-Daten, Einträgen in pro_waitlist und feedback.
    allow_unverified=True, damit auch unbestätigte Accounts gelöscht werden können.
    """
    id_token = extract_id_token(request, data)
    if not id_token:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        # Account deletion is a deliberately rare sensitive action. Pay the
        # live Firebase revocation/disabled-user lookup here instead of on every
        # ordinary app read.
        uid = verify_user_token(
            id_token,
            allow_unverified=True,
            check_revoked=True,
        )
    except Exception:
        raise HTTPException(status_code=401, detail="Authentication failed")
    try:
        user = auth.get_user(uid)
        account_deletion.start(uid, email=str(user.email or ""))
    except Exception as exc:
        logging.error(
            "delete_account: durable deletion job could not start category=%s",
            safe_exception(exc),
        )
        raise HTTPException(
            status_code=503,
            detail="Account deletion could not be started safely. Please try again.",
        ) from None

    try:
        errors = account_deletion.cleanup_uid(uid)
    except Exception as exc:
        # start() has already persisted the fail-closed job. A transient read
        # or cleanup coordinator outage must therefore remain an honest 202;
        # the maintenance loop will retry the same idempotent job.
        logging.error(
            "delete_account: cleanup attempt failed category=%s",
            safe_exception(exc),
        )
        errors = ["cleanup_coordinator"]
    invalidate_tier_cache(uid)
    if errors:
        return JSONResponse(
            status_code=202,
            content={
                "status": "cleanup_pending",
                "cleanup_pending": True,
                "message": (
                    "Account access is blocked. Remaining personal-data cleanup "
                    "is queued and will be retried automatically."
                ),
            },
        )
    return {"status": "deleted", "cleanup_pending": False}


@router.post("/track-interest")
@limiter.limit("5/minute")
def track_interest(request: Request, data: dict = Body(...)):
    """
    Speichert das Interesse an der Pro-Version in der DB.
    """
    token = data.get("id_token")
    source = data.get("source", "unknown")
    
    if not token:
         raise HTTPException(status_code=401, detail="Authentication failed")

    try:
        uid = verify_user_token(token)
        if is_user_pro(uid):
            raise HTTPException(status_code=409, detail="Pro access is already active.")
        user_email = auth.get_user(uid).email
        waitlist = db_firestore.collection("pro_waitlist")
        request_ref = waitlist.document(uid)
        existing = request_ref.get()
        legacy_pending = False
        if not existing.exists:
            legacy_pending = any(
                waitlist.where("uid", "==", uid).limit(1).stream()
            )
        if existing.exists or legacy_pending:
            return {
                "status": "pending",
                "already_requested": True,
                "message": "Your Pro beta request is already pending.",
            }

        source = str(source or "unknown")[:80]
        if source not in {"pro_beta_modal", "unknown"}:
            source = "other"
        interest_data = {
            "uid": uid,
            "email": user_email,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "source": source,
            "status": "pending",
        }
        created = persistence_guard.create_pro_beta_request(
            uid=uid,
            doc_ref=request_ref,
            payload=interest_data,
            db=db_firestore,
        )
        if not created:
            return {
                "status": "pending",
                "already_requested": True,
                "message": "Your Pro beta request is already pending.",
            }
        return {
            "status": "success",
            "already_requested": False,
            "message": "Your Pro beta request has been received.",
        }

    except HTTPException:
        raise
    except Exception as exc:
        logging.error("Pro beta request failed category=%s", safe_exception(exc))
        raise HTTPException(
            status_code=503,
            detail="Could not request Pro access. Please try again later.",
        ) from exc
