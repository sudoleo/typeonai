"""Versioned, API-key-authenticated asynchronous Consensus API."""

import logging
from datetime import datetime
from typing import Any, Literal, Optional, Union

from fastapi import APIRouter, Header, HTTPException, Request, Response, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field

import app.core.config as cfg
from app.core.security import db_firestore, is_user_pro
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
    FirestoreApiKeyRepository,
    InvalidApiKey,
)
from app.services.api_run_repository import (
    ApiRunConflict,
    ApiRunNotFound,
    ApiRunTransitionError,
)
from app.services.llm.base import count_words
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


def authenticate_api_identity(api_key: Optional[str]):
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    try:
        identity = api_key_repository.authenticate(api_key)
    except InvalidApiKey:
        raise HTTPException(status_code=401, detail="Invalid API key") from None
    ensure_api_account_active(identity.uid)
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
):
    """Accept and reserve one logical run; provider execution is asynchronous."""
    identity = authenticate_api_identity(api_key)
    enforce_uid_rate_limit(identity.uid, "create", 10)
    if not idempotency_key.strip():
        raise HTTPException(status_code=400, detail="Missing Idempotency-Key header")

    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty")
    request_payload = {"question": question, "deep_think": payload.deep_think}
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
                deep_think=payload.deep_think, is_pro=is_pro
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
