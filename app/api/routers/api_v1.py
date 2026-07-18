"""Versioned, API-key-authenticated asynchronous Consensus API."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, Header, HTTPException, Response, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field

import app.core.config as cfg
from app.core.security import db_firestore, is_user_pro
from app.services.api_consensus_runner import (
    api_run_repository,
    build_server_model_plan,
    reserve_run,
    schedule_run,
    validate_server_credentials,
)
from app.services.api_key_repository import (
    FirestoreApiKeyRepository,
    InvalidApiKey,
)
from app.services.api_run_repository import ApiRunConflict, ApiRunNotFound
from app.services.llm.base import count_words
from app.services.usage_repository import UsageLimitExceeded, UsageRunConflict


router = APIRouter(prefix="/api/v1", tags=["Consensus API"])
api_key_repository = FirestoreApiKeyRepository(db_firestore)
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
    reserved_at: datetime | None = None
    running_at: datetime | None = None
    succeeded_at: datetime | None = None
    failed_at: datetime | None = None
    result: dict[str, Any] | None = None
    error: RunError | None = None


class ApiErrorResponse(BaseModel):
    error: str | dict[str, Any]


def require_api_identity(api_key: str | None = Security(api_key_header)):
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    try:
        return api_key_repository.authenticate(api_key)
    except InvalidApiKey:
        raise HTTPException(status_code=401, detail="Invalid API key") from None


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
def create_consensus_run(
    payload: ConsensusRunRequest,
    response: Response,
    identity=Security(require_api_identity),
    idempotency_key: str = Header(..., alias="Idempotency-Key"),
):
    """Accept and reserve one logical run; provider execution is asynchronous."""
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
            schedule_run(existing["run_id"])
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
        # Duplicate submissions may enqueue more than one local task; the
        # Firestore reserved->running claim lets exactly one task proceed.
        schedule_run(run["run_id"])

    response.headers["Location"] = f"/api/v1/consensus/runs/{run['run_id']}"
    response.headers["Retry-After"] = "2"
    return _public_run(run)


@router.get(
    "/consensus/runs/{run_id}",
    response_model=ConsensusRunResponse,
    responses={401: {"model": ApiErrorResponse}, 404: {"model": ApiErrorResponse}},
)
def get_consensus_run(run_id: str, identity=Security(require_api_identity)):
    try:
        run = api_run_repository.get_for_uid(run_id, identity.uid)
        if run.get("status") == "running" and api_run_repository.fail_if_lease_expired(run_id):
            run = api_run_repository.get_for_uid(run_id, identity.uid)
    except ApiRunNotFound:
        raise HTTPException(status_code=404, detail="Run not found") from None
    return _public_run(run)


def _public_run(run: dict) -> dict:
    request = run.get("request") or {}
    return {
        "run_id": run["run_id"],
        "status": run["status"],
        "deep_think": bool(request.get("deep_think")),
        "accepted_at": run["accepted_at"],
        "reserved_at": run.get("reserved_at"),
        "running_at": run.get("running_at"),
        "succeeded_at": run.get("succeeded_at"),
        "failed_at": run.get("failed_at"),
        "result": run.get("result") if run.get("status") == "succeeded" else None,
        "error": run.get("error") if run.get("status") == "failed" else None,
    }
