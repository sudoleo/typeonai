"""Versioned, API-key-authenticated asynchronous Consensus API."""

import logging
from datetime import datetime
from typing import Any, Literal, Optional, Union

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field

import app.core.config as cfg
from app.core.security import db_firestore, is_user_admin, is_user_pro
from app.api.routers.pages import SITE_URL
from app.core.rate_limit import (
    ApiUidRateLimitExceeded,
    api_key_rate_key,
    api_uid_limiter,
    limiter,
)
from app.services.api_account_cleanup import (
    ApiAccountInactive,
    ApiAccountStatusUnavailable,
    FirestoreApiAccountCleanup,
)
from app.services.api_consensus_runner import (
    api_run_repository,
    build_server_model_plan,
    fail_expired_run,
    reserve_run,
    schedule_run,
    validate_server_credentials,
)
from app.services.api_key_repository import (
    DEFAULT_API_KEY_SCOPES,
    FirestoreApiKeyRepository,
    InvalidApiKey,
)
from app.services.api_run_repository import (
    ApiRunConflict,
    ApiRunNotFound,
    ApiRunTransitionError,
)
from app.services.llm.base import count_words
from app.services import publisher_config, share_snapshots, watch_service
from app.services.share_snapshots import ShareError
from app.services.usage_repository import UsageLimitExceeded, UsageRunConflict


router = APIRouter(prefix="/api/v1", tags=["Consensus API"])
api_key_repository = FirestoreApiKeyRepository(db_firestore)
api_account_cleanup = FirestoreApiAccountCleanup(db_firestore)
api_key_header = APIKeyHeader(
    name="X-API-Key",
    scheme_name="ConsensusApiKey",
    description="User-bound Consensus API key. The plaintext value is shown only when issued.",
    auto_error=False,
)


class ConsensusRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    question: str = Field(min_length=1, max_length=8_000)
    deep_think: bool = False


class RunError(BaseModel):
    code: str
    message: str


class ConsensusRunResponse(BaseModel):
    run_id: str
    status: Literal["accepted", "reserved", "running", "succeeded", "failed"]
    deep_think: bool
    accepted_at: datetime
    expires_at: Optional[datetime] = None
    reserved_at: Optional[datetime] = None
    running_at: Optional[datetime] = None
    succeeded_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[RunError] = None


class ApiErrorResponse(BaseModel):
    error: Union[str, dict[str, Any]]


class ShareIndexingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    indexed: bool


class ApiShareResponse(BaseModel):
    share_id: str
    url: str
    path: str
    question: str
    status: str
    visibility: Literal["public", "private"]
    index_eligible: bool
    indexed: bool
    indexing_status: Literal["noindex", "requested", "indexed"]
    robots: Literal["noindex, nofollow", "noindex, follow", "index, follow"]
    in_sitemap: bool
    created: Optional[bool] = None


class ApiShareListResponse(BaseModel):
    shares: list[ApiShareResponse]


class ApiPublisherConfigResponse(BaseModel):
    enabled: bool
    topic_brief: str
    auto_index: bool
    weekly_watch_enabled: bool
    watch_weekday: str
    watch_time: str
    watch_timezone: str
    watch_interval: Literal["weekly"]
    watch_model_tier: Literal["free"]
    excluded_providers: list[Literal["deepseek"]]


def authenticate_api_identity(api_key: Optional[str], required_scope: str = "consensus:run"):
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    try:
        identity = api_key_repository.authenticate(api_key)
    except InvalidApiKey:
        raise HTTPException(status_code=401, detail="Invalid API key") from None
    ensure_api_account_active(identity.uid)
    scopes = tuple(getattr(identity, "scopes", DEFAULT_API_KEY_SCOPES))
    if required_scope not in scopes:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "insufficient_scope",
                "message": f"API key requires scope: {required_scope}",
            },
        )
    return identity


def ensure_api_account_active(uid: str) -> None:
    """Fail closed for locally blocked, deleted, disabled or unverified users."""
    try:
        api_account_cleanup.ensure_active(uid)
    except ApiAccountInactive:
        raise HTTPException(status_code=401, detail="Invalid API key") from None
    except ApiAccountStatusUnavailable:
        logging.exception("Consensus API account status is unavailable")
        raise HTTPException(
            status_code=503, detail="Consensus API authentication is temporarily unavailable"
        ) from None


def enforce_uid_rate_limit(uid: str, operation: str, limit: int) -> None:
    try:
        api_uid_limiter.check(uid, operation, limit)
    except ApiUidRateLimitExceeded:
        raise HTTPException(
            status_code=429,
            detail="Consensus API rate limit exceeded",
            headers={"Retry-After": "60"},
        ) from None


def _require_api_admin(identity) -> None:
    if not is_user_admin(identity.uid):
        raise HTTPException(
            status_code=403,
            detail={
                "code": "admin_required",
                "message": "Scheduled Publisher access requires an admin account.",
            },
        )


@router.get(
    "/publisher/config",
    response_model=ApiPublisherConfigResponse,
    responses={401: {"model": ApiErrorResponse}, 403: {"model": ApiErrorResponse}},
)
@limiter.limit("30/minute")
@limiter.limit("20/minute", key_func=api_key_rate_key)
def get_api_publisher_config(
    request: Request,
    api_key: Optional[str] = Security(api_key_header),
):
    identity = authenticate_api_identity(api_key, "share:write")
    _require_api_admin(identity)
    try:
        return publisher_config.public_config(publisher_config.get_config())
    except Exception:
        logging.exception("Consensus API publisher configuration failed")
        raise HTTPException(
            status_code=503, detail="Publisher configuration is temporarily unavailable"
        ) from None


@router.post(
    "/consensus/runs",
    response_model=ConsensusRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        401: {"model": ApiErrorResponse},
        400: {"model": ApiErrorResponse},
        403: {"model": ApiErrorResponse},
        409: {"model": ApiErrorResponse},
        422: {"model": ApiErrorResponse},
        429: {"model": ApiErrorResponse},
        503: {"model": ApiErrorResponse},
    },
)
@limiter.limit("30/minute")
@limiter.limit("10/minute", key_func=api_key_rate_key)
def create_consensus_run(
    request: Request,
    payload: ConsensusRunRequest,
    response: Response,
    api_key: Optional[str] = Security(api_key_header),
    idempotency_key: str = Header(..., alias="Idempotency-Key"),
    publisher_mode: bool = Header(False, alias="X-Consensus-Publisher"),
):
    """Accept and reserve one logical run; provider execution is asynchronous."""
    identity = authenticate_api_identity(api_key)
    if publisher_mode:
        _require_api_admin(identity)
    enforce_uid_rate_limit(identity.uid, "create", 10)
    if not idempotency_key.strip():
        raise HTTPException(status_code=400, detail="Missing Idempotency-Key header")

    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty")
    request_payload = {"question": question, "deep_think": payload.deep_think}
    if publisher_mode:
        request_payload["publisher_mode"] = True
    try:
        existing = api_run_repository.get_by_idempotency(
            uid=identity.uid,
            idempotency_key=idempotency_key.strip(),
            request_payload=request_payload,
        )
    except ApiRunConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    if existing is not None and existing.get("status") != "accepted":
        if existing.get("status") == "reserved":
            if schedule_run(existing["run_id"]) is False:
                raise HTTPException(
                    status_code=503,
                    detail="Consensus API worker capacity is temporarily exhausted",
                    headers={"Retry-After": "5"},
                )
        response.headers["Location"] = f"/api/v1/consensus/runs/{existing['run_id']}"
        response.headers["Retry-After"] = "2"
        return _public_run(existing)

    if existing is not None:
        # A crash may leave the tiny accepted->reserved window open. Continue
        # with the exact server plan and tier snapshot stored at acceptance.
        run = existing
        model_plan = run.get("model_plan") or {}
    else:
        is_pro = is_user_pro(identity.uid)
        if payload.deep_think and not is_pro:
            raise HTTPException(status_code=403, detail="Deep Think requires a Pro account")
        max_words = cfg.get_word_limit(is_pro, payload.deep_think)
        if count_words(question) > max_words:
            raise HTTPException(status_code=400, detail=f"Input exceeds word limit of {max_words}")
        try:
            model_plan = build_server_model_plan(
                deep_think=payload.deep_think,
                is_pro=is_pro,
                excluded_providers=("deepseek",) if publisher_mode else (),
            )
        except HTTPException:
            raise
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from None
        except Exception:
            logging.exception("Consensus API server model plan is unavailable")
            raise HTTPException(
                status_code=503, detail="Consensus API is temporarily unavailable"
            ) from None
        try:
            run, _created = api_run_repository.create_or_get(
                uid=identity.uid,
                api_key_id=identity.key_id,
                idempotency_key=idempotency_key.strip(),
                request_payload=request_payload,
                model_plan=model_plan,
                is_pro=is_pro,
            )
        except ApiRunConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

    try:
        validate_server_credentials(model_plan)
    except Exception:
        logging.exception("Consensus API server credentials are unavailable")
        raise HTTPException(
            status_code=503, detail="Consensus API is temporarily unavailable"
        ) from None

    if run.get("status") == "accepted":
        try:
            run, _usage = reserve_run(run)
        except UsageLimitExceeded as exc:
            api_run_repository.delete_accepted(run["run_id"])
            raise HTTPException(
                status_code=429,
                detail=(
                    "Deep Think quota is exhausted for this UTC day"
                    if exc.limiting_bucket == "deep_think"
                    else "Run quota is exhausted for this UTC day"
                ),
            ) from None
        except UsageRunConflict as exc:
            api_run_repository.delete_accepted(run["run_id"])
            raise HTTPException(status_code=409, detail=str(exc)) from None
        except Exception:
            logging.exception("Consensus API usage reservation failed")
            # Keep the accepted run and any possibly committed reservation.
            # A retry with the same Idempotency-Key can safely finish this
            # step; deleting/releasing here would make that key terminal.
            raise HTTPException(
                status_code=503, detail="Run reservation is temporarily unavailable"
            ) from None

    if run.get("status") == "reserved":
        if schedule_run(run["run_id"]) is False:
            raise HTTPException(
                status_code=503,
                detail="Consensus API worker capacity is temporarily exhausted",
                headers={"Retry-After": "5"},
            )

    response.headers["Location"] = f"/api/v1/consensus/runs/{run['run_id']}"
    response.headers["Retry-After"] = "2"
    return _public_run(run)


@router.get(
    "/consensus/runs/{run_id}",
    response_model=ConsensusRunResponse,
    responses={401: {"model": ApiErrorResponse}, 404: {"model": ApiErrorResponse}},
)
@limiter.limit("240/minute")
@limiter.limit("120/minute", key_func=api_key_rate_key)
def get_consensus_run(
    request: Request,
    run_id: str,
    api_key: Optional[str] = Security(api_key_header),
):
    identity = authenticate_api_identity(api_key)
    enforce_uid_rate_limit(identity.uid, "get", 120)
    try:
        run = api_run_repository.get_for_uid(run_id, identity.uid)
        if run.get("status") == "running" and fail_expired_run(run_id):
            run = api_run_repository.get_for_uid(run_id, identity.uid)
    except ApiRunNotFound:
        raise HTTPException(status_code=404, detail="Run not found") from None
    return _public_run(run)


@router.delete(
    "/consensus/runs/{run_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        401: {"model": ApiErrorResponse},
        404: {"model": ApiErrorResponse},
        409: {"model": ApiErrorResponse},
    },
)
@limiter.limit("30/minute")
@limiter.limit("20/minute", key_func=api_key_rate_key)
def delete_consensus_run(
    request: Request,
    run_id: str,
    api_key: Optional[str] = Security(api_key_header),
):
    """Delete a terminal run and its idempotency mapping before TTL expiry."""
    identity = authenticate_api_identity(api_key)
    enforce_uid_rate_limit(identity.uid, "delete", 20)
    try:
        deleted = api_run_repository.delete_terminal_for_uid(run_id, identity.uid)
    except ApiRunNotFound:
        raise HTTPException(status_code=404, detail="Run not found") from None
    except ApiRunTransitionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    if not deleted:
        raise HTTPException(status_code=404, detail="Run not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/consensus/runs/{run_id}/share",
    response_model=ApiShareResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        401: {"model": ApiErrorResponse},
        403: {"model": ApiErrorResponse},
        404: {"model": ApiErrorResponse},
        409: {"model": ApiErrorResponse},
        429: {"model": ApiErrorResponse},
    },
)
@limiter.limit("20/minute")
@limiter.limit("10/minute", key_func=api_key_rate_key)
def publish_consensus_run(
    request: Request,
    run_id: str,
    response: Response,
    api_key: Optional[str] = Security(api_key_header),
):
    """Publish one succeeded owned run as a public immutable share."""
    identity = authenticate_api_identity(api_key, "share:write")
    enforce_uid_rate_limit(identity.uid, "share_create", 10)
    try:
        run = api_run_repository.get_for_uid(run_id, identity.uid)
        publication = share_snapshots.create_share_from_api_run(identity.uid, run)
        share = share_snapshots.get_share(publication["share_id"])
    except ApiRunNotFound:
        raise HTTPException(status_code=404, detail="Run not found") from None
    except ShareError as exc:
        _raise_api_share_error(exc)
    except Exception:
        logging.exception("Consensus API publication failed: %s", run_id)
        raise HTTPException(status_code=500, detail="Publication failed") from None
    if not publication["created"]:
        response.status_code = status.HTTP_200_OK
    response.headers["Location"] = publication["path"] if "path" in publication else (
        share_snapshots.share_path(publication["slug"], publication["share_id"])
    )
    return _public_api_share(publication["share_id"], share or {}, created=publication["created"])


@router.get(
    "/shares",
    response_model=ApiShareListResponse,
    responses={401: {"model": ApiErrorResponse}, 403: {"model": ApiErrorResponse}},
)
@limiter.limit("60/minute")
@limiter.limit("30/minute", key_func=api_key_rate_key)
def list_api_shares(
    request: Request,
    limit: int = Query(default=20, ge=1, le=200),
    api_key: Optional[str] = Security(api_key_header),
):
    identity = authenticate_api_identity(api_key, "share:write")
    enforce_uid_rate_limit(identity.uid, "share_list", 30)
    # The repository sorts after its bounded Firestore scan; scan the full API
    # maximum first so `limit=20` actually means the 20 newest rows.
    rows = share_snapshots.list_shares_for_owner(identity.uid, max_items=200)[:limit]
    return {"shares": [_public_api_share_row(row) for row in rows]}


@router.get(
    "/shares/{share_id}",
    response_model=ApiShareResponse,
    responses={
        401: {"model": ApiErrorResponse},
        403: {"model": ApiErrorResponse},
        404: {"model": ApiErrorResponse},
    },
)
@limiter.limit("120/minute")
@limiter.limit("60/minute", key_func=api_key_rate_key)
def get_api_share(
    request: Request,
    share_id: str,
    api_key: Optional[str] = Security(api_key_header),
):
    identity = authenticate_api_identity(api_key, "share:write")
    enforce_uid_rate_limit(identity.uid, "share_get", 60)
    share = _owned_api_share(share_id, identity.uid)
    return _public_api_share(share_id, share)


@router.post(
    "/shares/{share_id}/watch",
    response_model=dict[str, Any],
    responses={
        401: {"model": ApiErrorResponse},
        403: {"model": ApiErrorResponse},
        404: {"model": ApiErrorResponse},
        409: {"model": ApiErrorResponse},
        429: {"model": ApiErrorResponse},
    },
)
@limiter.limit("20/minute")
@limiter.limit("10/minute", key_func=api_key_rate_key)
def create_api_publisher_watch(
    request: Request,
    share_id: str,
    api_key: Optional[str] = Security(api_key_header),
):
    """Idempotently attach the configured weekly Free-tier Watch to a page."""
    identity = authenticate_api_identity(api_key, "share:write")
    _require_api_admin(identity)
    enforce_uid_rate_limit(identity.uid, "publisher_watch", 10)
    try:
        config = publisher_config.get_config()
        if not config["weekly_watch_enabled"]:
            raise HTTPException(
                status_code=409, detail="Weekly watches are disabled in Publisher configuration"
            )
        watch = watch_service.create_watch(
            identity.uid,
            share_id=share_id,
            interval="weekly",
            email_mode="changes_only",
            visibility="public",
            run_weekday=config["watch_weekday"],
            run_time=config["watch_time"],
            timezone_name=config["watch_timezone"],
            is_pro=is_user_pro(identity.uid),
            model_tier="free",
            return_existing=True,
            bypass_active_limit=True,
            excluded_providers=("deepseek",),
        )
    except HTTPException:
        raise
    except watch_service.WatchError as exc:
        status_by_code = {
            "not_found": 404,
            "forbidden": 404,
            "limit_reached": 429,
            "already_exists": 409,
        }
        raise HTTPException(
            status_code=status_by_code.get(exc.code, 400),
            detail={"code": exc.code, "message": exc.message},
        ) from None
    except Exception:
        logging.exception("Consensus API publisher Watch creation failed: %s", share_id)
        raise HTTPException(status_code=500, detail="Failed to create weekly Watch") from None
    return {"status": "success", "watch": watch}


@router.put(
    "/shares/{share_id}/indexing",
    response_model=ApiShareResponse,
    responses={
        401: {"model": ApiErrorResponse},
        403: {"model": ApiErrorResponse},
        404: {"model": ApiErrorResponse},
        409: {"model": ApiErrorResponse},
        422: {"model": ApiErrorResponse},
    },
)
@limiter.limit("20/minute")
@limiter.limit("10/minute", key_func=api_key_rate_key)
def set_api_share_indexing(
    request: Request,
    share_id: str,
    payload: ShareIndexingRequest,
    api_key: Optional[str] = Security(api_key_header),
):
    identity = authenticate_api_identity(api_key, "share:index")
    enforce_uid_rate_limit(identity.uid, "share_index", 10)
    if not is_user_admin(identity.uid):
        raise HTTPException(
            status_code=403,
            detail={
                "code": "admin_required",
                "message": "Direct indexing requires an admin account.",
            },
        )
    try:
        share = share_snapshots.set_api_share_indexing(
            share_id,
            identity.uid,
            indexed=payload.indexed,
            actor_key_id=identity.key_id,
        )
    except ShareError as exc:
        _raise_api_share_error(exc)
    return _public_api_share(share_id, share)


@router.delete(
    "/shares/{share_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        401: {"model": ApiErrorResponse},
        403: {"model": ApiErrorResponse},
        404: {"model": ApiErrorResponse},
    },
)
@limiter.limit("20/minute")
@limiter.limit("10/minute", key_func=api_key_rate_key)
def delete_api_share(
    request: Request,
    share_id: str,
    api_key: Optional[str] = Security(api_key_header),
):
    identity = authenticate_api_identity(api_key, "share:write")
    enforce_uid_rate_limit(identity.uid, "share_delete", 10)
    try:
        share_snapshots.revoke_share(share_id, identity.uid)
    except ShareError as exc:
        _raise_api_share_error(exc)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _owned_api_share(share_id: str, uid: str) -> dict:
    share = share_snapshots.get_share(share_id)
    if share is None or share.get("owner_uid") != uid:
        raise HTTPException(status_code=404, detail="Share not found")
    return share


def _public_api_share(share_id: str, share: dict, *, created: Optional[bool] = None) -> dict:
    slug = str(share.get("slug") or "")
    visibility = str(share.get("visibility") or "public")
    path = share_snapshots.share_path("" if visibility == "private" else slug, share_id)
    indexed = (
        visibility == "public"
        and bool(share.get("indexed"))
        and share.get("status") == "active"
    )
    requested = bool(share.get("index_requested")) and not indexed
    indexing_status = "indexed" if indexed else "requested" if requested else "noindex"
    result = {
        "share_id": share_id,
        "url": SITE_URL + path,
        "path": path,
        "question": str(share.get("question") or ""),
        "status": str(share.get("status") or "active"),
        "visibility": visibility,
        "index_eligible": bool(share.get("index_eligible")),
        "indexed": indexed,
        "indexing_status": indexing_status,
        "robots": (
            "index, follow"
            if indexed
            else "noindex, nofollow"
            if visibility == "private"
            else "noindex, follow"
        ),
        "in_sitemap": indexed,
    }
    if created is not None:
        result["created"] = created
    return result


def _public_api_share_row(row: dict) -> dict:
    share_id = str(row.get("share_id") or "")
    visibility = str(row.get("visibility") or "public")
    indexed = (
        visibility == "public"
        and bool(row.get("indexed"))
        and row.get("status") == "active"
    )
    requested = bool(row.get("index_requested")) and not indexed
    path = str(row.get("path") or share_snapshots.share_path("", share_id))
    return {
        "share_id": share_id,
        "url": SITE_URL + path,
        "path": path,
        "question": str(row.get("question") or ""),
        "status": str(row.get("status") or "active"),
        "visibility": visibility,
        "index_eligible": bool(row.get("index_eligible")),
        "indexed": indexed,
        "indexing_status": "indexed" if indexed else "requested" if requested else "noindex",
        "robots": (
            "index, follow"
            if indexed
            else "noindex, nofollow"
            if visibility == "private"
            else "noindex, follow"
        ),
        "in_sitemap": indexed,
    }


def _raise_api_share_error(exc: ShareError):
    status_by_code = {
        "not_found": 404,
        "forbidden": 404,
        "quota_exceeded": 429,
        "run_not_succeeded": 409,
        "share_not_active": 409,
        "conflict": 409,
        "duplicate": 409,
        "bad_result": 422,
        "index_quality_failed": 422,
        "bad_request": 400,
    }
    detail = {"code": exc.code, "message": exc.message, **exc.details}
    raise HTTPException(status_code=status_by_code.get(exc.code, 400), detail=detail)


def _public_run(run: dict) -> dict:
    request = run.get("request") or {}
    return {
        "run_id": run["run_id"],
        "status": run["status"],
        "deep_think": bool(request.get("deep_think")),
        "accepted_at": run["accepted_at"],
        "expires_at": run.get("expires_at"),
        "reserved_at": run.get("reserved_at"),
        "running_at": run.get("running_at"),
        "succeeded_at": run.get("succeeded_at"),
        "failed_at": run.get("failed_at"),
        "result": run.get("result") if run.get("status") == "succeeded" else None,
        "error": run.get("error") if run.get("status") == "failed" else None,
    }
