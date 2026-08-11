import asyncio
import os
import logging
import traceback
from contextlib import asynccontextmanager

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from starlette.background import BackgroundTask
from starlette.status import HTTP_422_UNPROCESSABLE_CONTENT

# Init Environment
load_dotenv()
if os.environ.get("E2E_TEST_MODE") != "1":
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", "gen-lang-client-0234219247-53b2b1c0e355.json")
logging.basicConfig(level=logging.INFO)

from app.core.security import CustomSecurityMiddleware, db_firestore
from app.core.background_tasks import (
    mark_task_disabled,
    supervise_background_task,
    task_health_snapshot,
)
from app.core.request_limits import RequestBodyLimitMiddleware
from app.core.e2e_profile import e2e_test_mode_enabled
from app.core.rate_limit import limiter

# Import routers
from app.api.routers import (
    admin,
    api_v1,
    auth,
    bookmarks,
    chat,
    chat_history,
    client_errors,
    pages,
    share,
    topics,
    users,
    watch,
)
from app.core.config import load_models_from_db
from app.services.api_account_cleanup import FirestoreApiAccountCleanup
from app.services.account_deletion import FirestoreAccountDeletion
from app.services.api_consensus_runner import (
    api_run_maintenance_loop,
)
from app.services.llm.mock_llm import mock_llm_enabled
from app.services.retention_maintenance import retention_maintenance_loop
from app.services.topic_runner import topic_scheduler_loop
from app.services.watch_scheduler import watch_scheduler_loop
from app.services.watch_service import backfill_publisher_watch_lineage
from app.services.seo_weekly_review import seo_review_scheduler_loop
from app.services.telegram_watch import run_startup_maintenance as telegram_startup_maintenance
from app.services.telegram_notifier import send_critical_error_notification


def _load_startup_configuration() -> None:
    """Load the only readiness-critical document with its SDK-side 5s budget."""
    try:
        load_models_from_db()
    except Exception:
        logging.exception("load_models_from_db failed on startup; using code defaults")

async def _disabled_loop() -> None:
    return None


def _scheduler_task(loop_factory, name: str):
    """Background writers stay off in MOCK_LLM instances. They share the
    production Firestore with the live deployment, so a local mock server would
    claim due schedule slots and publish fixture answers as real snapshots."""
    if mock_llm_enabled():
        logging.info("%s not started: MOCK_LLM=1", name)
        mark_task_disabled(name, "MOCK_LLM=1")
        return asyncio.create_task(_disabled_loop(), name=name)
    return _supervised_task(loop_factory, name)


def _supervised_task(loop_factory, name: str, *, restart: bool = True):
    return asyncio.create_task(
        supervise_background_task(
            name,
            loop_factory,
            restart=restart,
            alert=send_critical_error_notification,
        ),
        name=name,
    )


def _one_shot_task(func, name: str):
    async def run_once():
        await asyncio.to_thread(func)

    return _supervised_task(run_once, name, restart=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if e2e_test_mode_enabled():
        # Request-level persistence still runs against the isolated emulator,
        # but no cleanup, recovery, backfill, webhook or scheduler writer is
        # meaningful (or allowed) in the browser-test process.
        logging.info("E2E profile active: all lifespan maintenance is disabled")
        yield
        return

    # Fail-closed Account-Tombstones bleiben bestehen; nur ihre idempotente
    # Datenbereinigung wird nach transienten Firestore-Fehlern wiederholt.
    api_account_cleanup = FirestoreApiAccountCleanup(db_firestore)
    account_deletion = FirestoreAccountDeletion(db_firestore)
    _load_startup_configuration()
    lineage_backfill_task = _one_shot_task(
        backfill_publisher_watch_lineage, "publisher-watch-lineage-backfill"
    )
    telegram_webhook_task = _one_shot_task(
        telegram_startup_maintenance, "telegram-watch-startup-maintenance"
    )
    watch_task = _scheduler_task(watch_scheduler_loop, "consensus-watch-scheduler")
    topic_task = _scheduler_task(topic_scheduler_loop, "topic-scheduler")
    seo_review_task = _scheduler_task(
        seo_review_scheduler_loop, "seo-weekly-review-scheduler"
    )
    api_maintenance_task = _scheduler_task(
        api_run_maintenance_loop, "consensus-api-maintenance"
    )
    retention_task = _supervised_task(
        retention_maintenance_loop, "retention-maintenance"
    )
    api_account_cleanup_task = _supervised_task(
        api_account_cleanup.retry_loop, "consensus-api-account-cleanup"
    )
    account_deletion_task = _supervised_task(
        account_deletion.retry_loop, "full-account-deletion-cleanup"
    )
    tasks = (
        watch_task,
        topic_task,
        seo_review_task,
        api_maintenance_task,
        retention_task,
        api_account_cleanup_task,
        account_deletion_task,
        lineage_backfill_task,
        telegram_webhook_task,
    )
    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

app = FastAPI(
    title="consens.io API",
    version="1.0.0",
    description="Asynchronous, user-bound Consensus runs.",
    lifespan=lifespan,
)


@app.get("/health/maintenance", include_in_schema=False)
def maintenance_health():
    tasks = task_health_snapshot()
    degraded = any(
        item.get("state") in {"failed", "restarting"} for item in tasks.values()
    )
    return {
        "status": "degraded" if degraded else "ok",
        "tasks": tasks,
    }

# Add Custom Security Middleware
app.add_middleware(CustomSecurityMiddleware)
app.add_middleware(RequestBodyLimitMiddleware)

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
    """422 mit einer garantiert serialisierbaren Fehlerliste.

    exc.errors() enthaelt bei Pydantic v2 zwei Felder, die hier nicht
    hingehoeren: "ctx" traegt das ROHE Exception-Objekt (jeder Validator, der
    wie ueberall in diesem Projekt ValueError wirft, liess json.dumps damit
    platzen -- die Antwort war dann ein 500er statt eines 422ers), und "input"
    spiegelt den eingesendeten Wert zurueck, also potentiell einen Key oder ein
    Token aus einem abgelehnten Feld. Deshalb werden nur Ort, Typ und Meldung
    uebernommen, jeweils als reiner String.
    """
    details = []
    for error in exc.errors():
        location = error.get("loc") or ()
        details.append({
            "loc": [str(part) for part in location],
            "type": str(error.get("type") or ""),
            "msg": str(error.get("msg") or ""),
        })
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_CONTENT,
        content={"error": "Validation failed", "details": details},
    )

@app.exception_handler(Exception)
async def handle_unexpected_exception(request, exc: Exception):
    logging.exception(
        "Unhandled request exception for %s %s",
        request.method,
        request.url.path,
        exc_info=exc,
    )
    report = {
        "source": "server",
        "type": type(exc).__name__,
        "phase": "request",
        "message": str(exc) or "Unhandled server exception",
        "path": f"{request.method} {request.url.path}",
        "stack": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
        background=BackgroundTask(send_critical_error_notification, report),
    )

# Include Routers
#
# Nur /api/v1 ist die dokumentierte, oeffentliche Consensus-API und erscheint
# in /docs bzw. /openapi.json. Alle uebrigen Router sind interne App- und
# Admin-Endpunkte: sie sind zwar einzeln autorisiert, muessen einem Angreifer
# aber nicht als fertige Landkarte samt Parametern serviert werden.
for internal_router in (
    auth.router,
    users.router,
    bookmarks.router,
    chat.router,
    chat_history.router,
    client_errors.router,
    pages.router,
    admin.router,
    share.router,
    watch.router,
    topics.router,
):
    app.include_router(internal_router, include_in_schema=False)

app.include_router(api_v1.router)
