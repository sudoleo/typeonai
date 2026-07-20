import asyncio
import os
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY

# Init Environment
load_dotenv()
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", "gen-lang-client-0234219247-53b2b1c0e355.json")
logging.basicConfig(level=logging.INFO)

from app.core.security import CustomSecurityMiddleware, db_firestore
from app.core.rate_limit import limiter

# Import routers
from app.api.routers import auth, users, bookmarks, chat, pages, admin, share, watch, api_v1
from app.core.config import load_models_from_db
from app.services.api_account_cleanup import FirestoreApiAccountCleanup
from app.services.api_consensus_runner import (
    api_run_maintenance_loop,
    recover_persisted_runs,
)
from app.services.share_snapshots import cleanup_expired_pending, cleanup_revoked_shares
from app.services.watch_scheduler import watch_scheduler_loop
from app.services.watch_service import backfill_publisher_watch_lineage
from app.services.seo_weekly_review import seo_review_scheduler_loop


def _startup_job_timeout_seconds() -> int:
    try:
        configured = int(os.environ.get("STARTUP_JOB_TIMEOUT_SECONDS", "15"))
    except (TypeError, ValueError):
        configured = 15
    return max(5, min(configured, 60))


STARTUP_JOB_TIMEOUT_SECONDS = _startup_job_timeout_seconds()


def _run_startup_jobs_blocking(jobs):
    for name, func in jobs:
        try:
            func()
        except Exception:
            logging.exception("%s failed on startup", name)


async def _run_startup_jobs(jobs):
    """Bound all blocking Firestore startup work by one readiness deadline."""
    try:
        await asyncio.wait_for(
            asyncio.to_thread(_run_startup_jobs_blocking, jobs),
            timeout=STARTUP_JOB_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logging.error(
            "Firestore startup maintenance timed out after %ss; "
            "startup continues with safe defaults",
            STARTUP_JOB_TIMEOUT_SECONDS,
        )

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail-closed Account-Tombstones bleiben bestehen; nur ihre idempotente
    # Datenbereinigung wird nach transienten Firestore-Fehlern wiederholt.
    api_account_cleanup = FirestoreApiAccountCleanup(db_firestore)
    await _run_startup_jobs((
        # Modell-Defaults bleiben bei einem Timeout aktiv.
        ("load_models_from_db", load_models_from_db),
        # TTL-Fallbacks; der tägliche Render-Restart triggert sie regelmäßig.
        ("cleanup_expired_pending", cleanup_expired_pending),
        ("cleanup_revoked_shares", cleanup_revoked_shares),
        ("blocked Consensus API account cleanup retry", api_account_cleanup.retry_pending),
        # reserved->running bleibt transaktional und wird zusätzlich vom
        # 60-Sekunden-Maintenance-Loop wieder aufgenommen.
        ("recover_persisted_runs", recover_persisted_runs),
    ))
    lineage_backfill_task = asyncio.create_task(
        asyncio.to_thread(backfill_publisher_watch_lineage),
        name="publisher-watch-lineage-backfill",
    )
    watch_task = asyncio.create_task(watch_scheduler_loop(), name="consensus-watch-scheduler")
    seo_review_task = asyncio.create_task(
        seo_review_scheduler_loop(), name="seo-weekly-review-scheduler"
    )
    api_maintenance_task = asyncio.create_task(
        api_run_maintenance_loop(), name="consensus-api-maintenance"
    )
    api_account_cleanup_task = asyncio.create_task(
        api_account_cleanup.retry_loop(), name="consensus-api-account-cleanup"
    )
    try:
        yield
    finally:
        watch_task.cancel()
        seo_review_task.cancel()
        api_maintenance_task.cancel()
        api_account_cleanup_task.cancel()
        lineage_backfill_task.cancel()
        try:
            await watch_task
        except asyncio.CancelledError:
            pass
        try:
            await seo_review_task
        except asyncio.CancelledError:
            pass
        try:
            await api_account_cleanup_task
        except asyncio.CancelledError:
            pass
        try:
            await api_maintenance_task
        except asyncio.CancelledError:
            pass
        try:
            await lineage_backfill_task
        except asyncio.CancelledError:
            pass

app = FastAPI(
    title="consens.io API",
    version="1.0.0",
    description="Asynchronous, user-bound Consensus runs.",
    lifespan=lifespan,
)

# Add Custom Security Middleware
app.add_middleware(CustomSecurityMiddleware)

# Add Rate Limiter state
app.state.limiter = limiter

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Exception Handlers
@app.exception_handler(HTTPException)
async def handle_http_exception(request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

@app.exception_handler(RequestValidationError)
async def handle_validation_exception(request, exc: RequestValidationError):
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Validation failed", "details": exc.errors()},
    )

# Include Routers
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(bookmarks.router)
app.include_router(chat.router)
app.include_router(pages.router)
app.include_router(admin.router)
app.include_router(share.router)
app.include_router(watch.router)
app.include_router(api_v1.router)
