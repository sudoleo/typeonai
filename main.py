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

from app.core.security import CustomSecurityMiddleware
from app.core.rate_limit import limiter

# Import routers
from app.api.routers import auth, users, bookmarks, chat, pages, admin, share
from app.core.config import load_models_from_db
from app.services.share_snapshots import cleanup_expired_pending

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models from db on startup
    load_models_from_db()
    # Abgelaufene pending_results aufräumen (Fallback zur Firestore-TTL-Policy;
    # der tägliche Render-Restart triggert das regelmäßig)
    try:
        cleanup_expired_pending()
    except Exception:
        logging.exception("cleanup_expired_pending failed on startup")
    yield

app = FastAPI(lifespan=lifespan)

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
