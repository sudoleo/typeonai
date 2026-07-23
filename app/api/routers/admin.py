import asyncio
import logging
from typing import Literal, Optional
from firebase_admin import auth
from fastapi import APIRouter, Request, Body, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.core.security import extract_id_token, verify_user_token, is_user_admin, db_firestore
from app.core.rate_limit import limiter
import app.core.config as cfg
from app.core.config import apply_limits, get_limits_config, load_models_from_db
from app.services import share_snapshots as snapshots
from app.services import (
    mailer,
    publisher_config,
    seo_data,
    seo_recommendation,
    seo_weekly_review,
    watch_scheduler,
    watch_service,
)
from app.services.share_snapshots import ShareError
from app.services.api_key_repository import (
    ApiKeyNotFound,
    FirestoreApiKeyRepository,
)
from app.services.api_account_cleanup import FirestoreApiAccountCleanup

router = APIRouter()
api_key_repository = FirestoreApiKeyRepository(db_firestore)
api_account_cleanup = FirestoreApiAccountCleanup(db_firestore)
seo_data_service = seo_data.SeoDataService(db_firestore)
seo_recommendation_service = seo_recommendation.SeoRecommendationService(db_firestore)
seo_weekly_review_service = seo_weekly_review.default_service


class AdminIssueApiKeyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    uid: str = Field(min_length=1, max_length=128)
    label: str = Field(default="", max_length=80)
    scopes: list[Literal["consensus:run", "share:write", "share:index"]] = Field(
        default_factory=lambda: ["consensus:run", "share:write"]
    )


class AdminPublisherConfigRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    enabled: bool
    topic_brief: str = Field(min_length=1, max_length=publisher_config.TOPIC_BRIEF_MAX_CHARS)
    auto_index: bool
    weekly_watch_enabled: bool
    watch_weekday: Literal[
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
    ]
    watch_time: str = Field(min_length=5, max_length=5)
    watch_timezone: str = Field(min_length=1, max_length=64)
    max_active_publisher_watches: int = Field(default=12, ge=1, le=100)


class AdminSeoReviewConfigRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    enabled: bool
    interval_days: int = Field(default=7, ge=1, le=90)
    run_time: str = Field(default="09:00", min_length=5, max_length=5)
    timezone: str = Field(default="Europe/Berlin", min_length=1, max_length=64)


class AdminSeoReviewActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    group: Optional[Literal[
        "keep_indexed", "pause_watch_only", "resume_watch", "noindex_only",
        "noindex_and_pause_watch", "delete_candidate", "manual_improvement",
    ]] = None
    page_ids: list[str] = Field(default_factory=list, max_length=100)
    apply_all: bool = False
    confirm_delete: bool = False


class AdminSeoEditorialDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    page_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: Literal[
        "keep_as_is", "create_successor", "investigate", "noindex", "delete",
        "edit_static_page",
    ]
    note: str = Field(default="", max_length=500)


def _require_admin(request, data):
    id_token = extract_id_token(request, data)
    if not id_token:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        uid = verify_user_token(id_token)
    except Exception:
        raise HTTPException(status_code=401, detail="Authentication failed")
    if not is_user_admin(uid):
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return uid


_SHARE_ERROR_STATUS = {"not_found": 404, "bad_request": 400}


@router.get("/api/admin/seo")
async def admin_get_seo_overview(request: Request):
    _require_admin(request, {})
    try:
        result = await asyncio.to_thread(seo_data_service.overview)
        result["content_judge"] = seo_recommendation_service.content_judge.status()
        result["weekly_review"] = await asyncio.to_thread(seo_weekly_review_service.status)
        return result
    except Exception:
        logging.exception("admin_get_seo_overview failed")
        raise HTTPException(status_code=500, detail="Failed to load SEO data")


@router.post("/api/admin/seo/check")
async def admin_check_seo_connection(request: Request, data: dict = Body(default={})):
    _require_admin(request, data)
    try:
        return await asyncio.to_thread(seo_data_service.check_connection)
    except Exception:
        # Do not include exception details here: configuration failures may
        # originate while opening the secret file. The admin receives only the
        # stable sanitized error below.
        logging.error("admin_check_seo_connection failed safely")
        raise HTTPException(status_code=500, detail="Search Console connection check failed safely")


@router.post("/api/admin/seo/collect")
async def admin_collect_seo_data(request: Request, data: dict = Body(default={})):
    _require_admin(request, data)
    try:
        return await asyncio.to_thread(seo_data_service.collect)
    except seo_data.CollectionAlreadyRunning:
        raise HTTPException(status_code=409, detail="SEO collection is already running")
    except Exception:
        logging.exception("admin_collect_seo_data failed")
        raise HTTPException(status_code=500, detail="SEO collection failed safely")


_SEO_RECOMMENDATION_STATUS = {
    "not_found": 404,
    "llm_not_configured": 409,
    "content_judge_not_applicable": 409,
    "invalid_llm_response": 502,
    "unsafe_llm_response": 502,
}


def _raise_seo_recommendation_error(exc):
    raise HTTPException(
        status_code=_SEO_RECOMMENDATION_STATUS.get(exc.code, 400),
        detail=exc.safe_message,
    )


def _raise_seo_review_error(exc):
    status = {
        "not_found": 404,
        "delete_confirmation_required": 409,
        "topic_brief_changed": 409,
        "state_changed": 409,
        "recommendation_stale": 409,
    }.get(exc.code, 400)
    raise HTTPException(status_code=status, detail=exc.safe_message)


@router.get("/api/admin/seo/review")
async def admin_get_seo_weekly_review(request: Request):
    _require_admin(request, {})
    try:
        return await asyncio.to_thread(seo_weekly_review_service.status)
    except Exception:
        logging.exception("admin_get_seo_weekly_review failed")
        raise HTTPException(status_code=500, detail="Failed to load weekly SEO review")


@router.put("/api/admin/seo/review/config")
async def admin_save_seo_weekly_review_config(
    request: Request, data: AdminSeoReviewConfigRequest = Body(...)
):
    _require_admin(request, {})
    try:
        return await asyncio.to_thread(
            seo_weekly_review_service.save_config,
            enabled=data.enabled,
            interval_days=data.interval_days,
            run_time=data.run_time,
            timezone_name=data.timezone,
        )
    except seo_weekly_review.ReviewError as exc:
        _raise_seo_review_error(exc)
    except Exception:
        logging.exception("admin_save_seo_weekly_review_config failed")
        raise HTTPException(status_code=500, detail="Failed to save weekly SEO review configuration")


@router.post("/api/admin/seo/review/run")
@limiter.limit("3/minute")
async def admin_run_seo_weekly_review(request: Request, data: dict = Body(default={})):
    _require_admin(request, data)
    try:
        return await asyncio.to_thread(seo_weekly_review_service.run, force=True)
    except seo_weekly_review.ReviewAlreadyRunning:
        raise HTTPException(status_code=409, detail="A weekly SEO review is already running")
    except Exception:
        logging.exception("admin_run_seo_weekly_review failed")
        raise HTTPException(status_code=500, detail="Weekly SEO review failed safely")


@router.post("/api/admin/seo/reviews/{run_id}/preview")
async def admin_preview_seo_review_actions(
    request: Request, run_id: str, data: AdminSeoReviewActionRequest = Body(...)
):
    _require_admin(request, {})
    try:
        return await asyncio.to_thread(
            seo_weekly_review_service.preview,
            run_id,
            group=data.group,
            page_ids=data.page_ids,
            apply_all=data.apply_all,
        )
    except seo_weekly_review.ReviewError as exc:
        _raise_seo_review_error(exc)


@router.post("/api/admin/seo/reviews/{run_id}/apply")
@limiter.limit("10/minute")
async def admin_apply_seo_review_actions(
    request: Request, run_id: str, data: AdminSeoReviewActionRequest = Body(...)
):
    admin_uid = _require_admin(request, {})
    try:
        return await asyncio.to_thread(
            seo_weekly_review_service.apply,
            run_id,
            admin_uid=admin_uid,
            group=data.group,
            page_ids=data.page_ids,
            apply_all=data.apply_all,
            confirm_delete=data.confirm_delete,
        )
    except seo_weekly_review.ReviewError as exc:
        _raise_seo_review_error(exc)


@router.post("/api/admin/seo/reviews/{run_id}/topic-brief/accept")
async def admin_accept_seo_review_topic_brief(
    request: Request, run_id: str, data: dict = Body(default={})
):
    admin_uid = _require_admin(request, data)
    try:
        return await asyncio.to_thread(
            seo_weekly_review_service.accept_topic_brief, run_id, admin_uid=admin_uid
        )
    except seo_weekly_review.ReviewError as exc:
        _raise_seo_review_error(exc)


@router.post("/api/admin/seo/reviews/{run_id}/topic-brief/reject")
async def admin_reject_seo_review_topic_brief(
    request: Request, run_id: str, data: dict = Body(default={})
):
    admin_uid = _require_admin(request, data)
    try:
        return await asyncio.to_thread(
            seo_weekly_review_service.reject_topic_brief, run_id, admin_uid=admin_uid
        )
    except seo_weekly_review.ReviewError as exc:
        _raise_seo_review_error(exc)


@router.post("/api/admin/seo/reviews/{run_id}/editorial-decision")
async def admin_record_seo_editorial_decision(
    request: Request, run_id: str, data: AdminSeoEditorialDecisionRequest = Body(...)
):
    admin_uid = _require_admin(request, {})
    try:
        return await asyncio.to_thread(
            seo_weekly_review_service.record_editorial_decision,
            run_id,
            page_id=data.page_id,
            decision=data.decision,
            note=data.note,
            admin_uid=admin_uid,
        )
    except seo_weekly_review.ReviewError as exc:
        _raise_seo_review_error(exc)


@router.post("/api/admin/seo/pages/{page_id}/recommendation")
async def admin_generate_seo_recommendation(
    request: Request, page_id: str, data: dict = Body(default={})
):
    _require_admin(request, data)
    try:
        return await asyncio.to_thread(seo_recommendation_service.generate, page_id)
    except seo_recommendation.SeoRecommendationError as exc:
        _raise_seo_recommendation_error(exc)
    except Exception:
        logging.exception("admin_generate_seo_recommendation failed")
        raise HTTPException(status_code=500, detail="SEO recommendation failed safely")


@router.post("/api/admin/seo/pages/{page_id}/content-judge")
@limiter.limit("5/minute")
async def admin_ask_seo_content_judge(
    request: Request, page_id: str, data: dict = Body(default={})
):
    _require_admin(request, data)
    try:
        return await asyncio.to_thread(
            seo_recommendation_service.ask_content_judge, page_id
        )
    except seo_recommendation.SeoRecommendationError as exc:
        _raise_seo_recommendation_error(exc)
    except Exception:
        logging.exception("admin_ask_seo_content_judge failed")
        raise HTTPException(status_code=500, detail="SEO content judge failed safely")


@router.post("/api/admin/api-keys", status_code=201)
@limiter.limit("10/minute")
def admin_issue_api_key(
    request: Request, data: AdminIssueApiKeyRequest = Body(...)
):
    """Issue a user-bound key. The plaintext secret is returned exactly once."""
    admin_uid = _require_admin(request, {})
    try:
        user = auth.get_user(data.uid.strip())
        if user.disabled:
            raise HTTPException(status_code=409, detail="Cannot issue a key for a disabled user")
        if not user.email_verified:
            raise HTTPException(status_code=409, detail="Cannot issue a key for an unverified user")
        if api_account_cleanup.is_blocked(data.uid.strip()):
            raise HTTPException(status_code=409, detail="API access is blocked for this account")
        if "share:index" in data.scopes and not is_user_admin(data.uid.strip()):
            raise HTTPException(
                status_code=409,
                detail="Direct indexing scope can only be issued to an admin UID",
            )
        return api_key_repository.issue(
            data.uid.strip(),
            label=data.label.strip(),
            created_by=admin_uid,
            scopes=data.scopes,
        )
    except HTTPException:
        raise
    except auth.UserNotFoundError:
        raise HTTPException(status_code=404, detail="User not found") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except Exception:
        logging.exception("admin_issue_api_key failed")
        raise HTTPException(status_code=500, detail="Failed to issue API key")


@router.get("/api/admin/api-keys")
@limiter.limit("30/minute")
def admin_list_api_keys(request: Request, uid: Optional[str] = None):
    _require_admin(request, {})
    try:
        return {"keys": api_key_repository.list(uid=uid.strip() if uid else None)}
    except Exception:
        logging.exception("admin_list_api_keys failed")
        raise HTTPException(status_code=500, detail="Failed to list API keys")


@router.delete("/api/admin/api-keys/{key_id}")
@limiter.limit("20/minute")
def admin_revoke_api_key(request: Request, key_id: str):
    _require_admin(request, {})
    try:
        key = api_key_repository.revoke(key_id)
    except ApiKeyNotFound:
        raise HTTPException(status_code=404, detail="API key not found") from None
    except Exception:
        logging.exception("admin_revoke_api_key failed")
        raise HTTPException(status_code=500, detail="Failed to revoke API key")
    return {"key_id": key_id, "status": key["status"]}


@router.get("/api/admin/publisher-config")
@limiter.limit("30/minute")
def admin_get_publisher_config(request: Request):
    _require_admin(request, {})
    try:
        return {"config": publisher_config.public_config(publisher_config.get_config())}
    except Exception:
        logging.exception("admin_get_publisher_config failed")
        raise HTTPException(status_code=500, detail="Failed to load publisher configuration")


@router.put("/api/admin/publisher-config")
@limiter.limit("20/minute")
def admin_update_publisher_config(
    request: Request, data: AdminPublisherConfigRequest = Body(...)
):
    admin_uid = _require_admin(request, {})
    try:
        config = publisher_config.save_config(data.model_dump(), updated_by=admin_uid)
    except publisher_config.PublisherConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except Exception:
        logging.exception("admin_update_publisher_config failed")
        raise HTTPException(status_code=500, detail="Failed to save publisher configuration")
    return {"status": "success", "config": publisher_config.public_config(config)}


@router.get("/api/admin/watches")
def admin_list_watches(request: Request):
    _require_admin(request, {})
    try:
        watches = watch_service.list_watches_for_admin()
    except Exception:
        logging.exception("admin_list_watches failed")
        raise HTTPException(status_code=500, detail="Failed to load watches")
    return {
        "status": "success",
        "smtp_configured": mailer.is_configured(),
        "watches": watches,
    }


@router.post("/api/admin/watches/{watch_id}/run")
async def admin_run_watch(request: Request, watch_id: str, data: dict = Body(default={})):
    _require_admin(request, data)
    try:
        watch = await asyncio.to_thread(watch_service.queue_watch_run, watch_id)
        watch_scheduler.wake_watch_scheduler()
    except watch_service.WatchError as exc:
        status = 404 if exc.code == "not_found" else 409
        raise HTTPException(status_code=status, detail=exc.message)
    except Exception:
        logging.exception("admin_run_watch failed")
        raise HTTPException(status_code=500, detail="Failed to start watch")
    return {"status": "success", "watch": watch, "run_requested": True}


@router.post("/api/admin/watches/test-email")
async def admin_send_watch_test_email(request: Request, data: dict = Body(default={})):
    uid = _require_admin(request, data)
    if not mailer.is_configured():
        raise HTTPException(status_code=503, detail="SMTP_HOST and MAIL_FROM must be configured")
    try:
        user = await asyncio.to_thread(auth.get_user, uid)
        recipient = getattr(user, "email", None)
        if not getattr(user, "email_verified", False) or not recipient:
            raise HTTPException(status_code=409, detail="The admin account needs a verified e-mail address")
        accepted = await mailer.send_message(mailer.build_test_message(recipient=recipient))
        if not accepted:
            raise HTTPException(status_code=502, detail="SMTP did not accept the test message")
    except HTTPException:
        raise
    except Exception:
        logging.exception("admin_send_watch_test_email failed")
        raise HTTPException(status_code=500, detail="Failed to send watch test e-mail")
    return {"status": "success", "recipient": recipient}


@router.get("/api/admin/shares")
def admin_list_shares(request: Request, filter: str = "reported"):
    _require_admin(request, {})
    try:
        shares = snapshots.list_shares_for_admin(only_reported=(filter != "all"))
    except Exception:
        logging.exception("admin_list_shares failed")
        raise HTTPException(status_code=500, detail="Failed to load shares")
    return {"status": "success", "shares": shares, "site_url": snapshots_site_url()}


def snapshots_site_url():
    # Lazy-Import, um den Router-Importgraphen (pages -> LLM-SDKs) nicht in
    # jeden admin.py-Import zu ziehen.
    from app.api.routers.pages import SITE_URL
    return SITE_URL


@router.post("/api/admin/shares/{share_id}/moderate")
def admin_moderate_share(request: Request, share_id: str, data: dict = Body(...)):
    admin_uid = _require_admin(request, data)
    action = data.get("action")
    indexed = data.get("indexed")
    if indexed is not None and not isinstance(indexed, bool):
        raise HTTPException(status_code=400, detail="indexed must be a boolean")
    try:
        result = snapshots.moderate_share(
            share_id,
            action=action,
            indexed=indexed,
            actor_uid=admin_uid,
            source="admin_ui",
        )
    except ShareError as exc:
        raise HTTPException(status_code=_SHARE_ERROR_STATUS.get(exc.code, 400), detail=exc.message)
    except Exception:
        logging.exception("admin_moderate_share failed")
        raise HTTPException(status_code=500, detail="Failed to moderate share")
    return {
        "status": "success",
        "share": {
            "share_id": share_id,
            "share_status": result.get("status"),
            "indexed": bool(result.get("indexed")),
            "index_eligible": bool(result.get("index_eligible")),
        },
    }


@router.delete("/api/admin/shares/{share_id}")
@limiter.limit("10/minute")
def admin_delete_share(request: Request, share_id: str):
    _require_admin(request, {})
    try:
        deleted = snapshots.hard_delete_share(share_id)
    except ShareError as exc:
        raise HTTPException(status_code=_SHARE_ERROR_STATUS.get(exc.code, 400), detail=exc.message)
    except Exception:
        logging.exception("admin_delete_share failed")
        raise HTTPException(status_code=500, detail="Failed to delete share")
    return {"status": "success", "deleted": deleted}


@router.get("/api/admin/benchmark/runs")
def admin_list_benchmark_runs(request: Request):
    _require_admin(request, {})
    from app.services import benchmark_reports
    try:
        runs = benchmark_reports.list_runs_with_disk_fallback()
    except Exception:
        logging.exception("admin_list_benchmark_runs failed")
        raise HTTPException(status_code=500, detail="Failed to load benchmark runs")
    return {"status": "success", "runs": runs}


@router.get("/api/admin/benchmark/runs/{run_id}")
def admin_get_benchmark_run(request: Request, run_id: str):
    _require_admin(request, {})
    from app.services import benchmark_reports
    try:
        run = benchmark_reports.get_run_with_disk_fallback(run_id)
    except Exception:
        logging.exception("admin_get_benchmark_run failed")
        raise HTTPException(status_code=500, detail="Failed to load benchmark run")
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"status": "success", "run": run}


PROVIDER_KEYS = ("openai", "mistral", "anthropic", "gemini", "deepseek", "grok")


def _ordered_unique(items, drop=None, ensure=None) -> list:
    """Dedupliziert unter Erhalt der Reihenfolge, entfernt `drop` und haengt
    fehlende Pflichtmodelle aus `ensure` (in dieser Reihenfolge) hinten an."""
    drop = set(drop or ())
    seen = set()
    out = []
    for item in items or []:
        value = str(item).strip()
        if value and value not in drop and value not in seen:
            seen.add(value)
            out.append(value)
    for value in (ensure or ()):
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def normalize_models_document(data: dict) -> dict:
    normalized = dict(data or {})
    normalized["grok"] = cfg.canonical_model_ids(normalized.get("grok"), "grok")
    # Provider-Listen behalten ihre (Admin-)Reihenfolge bei. Entfernte interne
    # Modell-IDs werden bei jedem Admin-Read/-Save aus Legacy-Dokumenten bereinigt.
    for provider in PROVIDER_KEYS:
        normalized[provider] = _ordered_unique(
            normalized.get(provider), drop=cfg.REMOVED_MODEL_IDS
        )

    normalized["mistral"] = _ordered_unique(
        normalized.get("mistral"), drop=cfg.DEPRECATED_MISTRAL_MODELS,
    )
    normalized["deepseek"] = _ordered_unique(
        normalized.get("deepseek"), drop=cfg.DEPRECATED_DEEPSEEK_MODELS,
    )

    configured_models = set().union(
        *(set(normalized.get(provider) or []) for provider in PROVIDER_KEYS)
    )
    premium = {
        cfg.canonical_model_id(model)
        for model in (normalized.get("premium") or [])
    }
    premium.difference_update(cfg.REMOVED_MODEL_IDS)
    premium.difference_update(cfg.DEPRECATED_MISTRAL_MODELS)
    premium.difference_update(cfg.DEPRECATED_DEEPSEEK_MODELS)
    premium.difference_update(cfg.DEPRECATED_GROK_MODELS)
    premium.intersection_update(configured_models)
    normalized["premium"] = sorted(premium)

    # Free-Default je Provider: nur gueltig, wenn das Modell im Provider gelistet
    # und nicht Premium ist (sonst saehe ein Free-Nutzer einen Sperr-Default).
    incoming_defaults = normalized.get("defaults") or {}
    clean_defaults = {}
    for provider in PROVIDER_KEYS:
        chosen = cfg.canonical_model_id(incoming_defaults.get(provider), provider)
        allowed = list(normalized.get(provider) or [])
        if not (chosen and chosen in allowed and chosen not in premium):
            base = cfg._BASE_FREE_DEFAULTS.get(provider)
            chosen = next(
                (
                    candidate for candidate in (base, *allowed)
                    if candidate in allowed and candidate not in premium
                ),
                "",
            )
        if chosen:
            clean_defaults[provider] = chosen
    normalized["defaults"] = clean_defaults

    # Watch-Antwortmodelle: je Tier hoechstens ein Modell pro Provider. Free
    # darf keine Premium-Modelle verwenden. Bei Legacy-Dokumenten ohne
    # dieses Feld wird das bisherige Drei-Modell-Setup eingeblendet.
    incoming_watch = normalized.get("watch_models")
    incoming_watch = incoming_watch if isinstance(incoming_watch, dict) else {}
    clean_watch = {}
    for tier in ("free", "pro"):
        supplied = incoming_watch.get(tier)
        source = supplied if isinstance(supplied, dict) else cfg._BASE_WATCH_MODELS_BY_TIER[tier]
        tier_models = {}
        for provider in PROVIDER_KEYS:
            chosen = cfg.canonical_model_id(source.get(provider), provider)
            if not chosen or chosen not in set(normalized.get(provider) or []):
                continue
            if tier == "free" and chosen in premium:
                continue
            tier_models[provider] = chosen
        if len(tier_models) < 2:
            tier_models = {}
            for provider in PROVIDER_KEYS:
                candidates = (
                    clean_defaults.get(provider),
                    cfg._BASE_WATCH_MODELS_BY_TIER[tier].get(provider),
                    *(normalized.get(provider) or []),
                )
                chosen = next(
                    (
                        candidate for candidate in candidates
                        if candidate in set(normalized.get(provider) or [])
                        and (tier == "pro" or candidate not in premium)
                    ),
                    "",
                )
                if chosen:
                    tier_models[provider] = chosen
                if len(tier_models) >= 3:
                    break
        clean_watch[tier] = tier_models
    normalized["watch_models"] = clean_watch

    allowed_direct_consensus = set()
    for provider in PROVIDER_KEYS:
        allowed_direct_consensus.update(normalized.get(provider) or [])

    # Vollstaendige Model-Sets je Picker-Preset. Fast/Balanced bleiben auch bei
    # Admin-Konfiguration Free-faehig; High Quality (ID: thorough) ist Pro-only.
    incoming_presets = normalized.get("preset_models")
    incoming_presets = incoming_presets if isinstance(incoming_presets, dict) else {}
    preset_definitions = {
        preset["id"]: preset for preset in cfg.CONSENSUS_PRESET_DEFINITIONS
    }
    clean_presets = {}
    for preset_id, base in cfg._BASE_CONSENSUS_PRESET_MODELS.items():
        supplied = incoming_presets.get(preset_id)
        supplied = supplied if isinstance(supplied, dict) else {}
        pro_only = bool(preset_definitions[preset_id]["pro_only"])
        clean = {}
        for provider in PROVIDER_KEYS:
            chosen = cfg.canonical_model_id(supplied.get(provider), provider)
            deprecated = (
                cfg.DEPRECATED_CONSENSUS_PRESET_MODELS
                .get(preset_id, {})
                .get(provider, set())
            )
            if chosen in deprecated:
                chosen = ""
            allowed = list(normalized.get(provider) or [])
            if chosen not in allowed or (not pro_only and chosen in premium):
                candidates = (
                    cfg.canonical_model_id(base.get(provider), provider),
                    clean_defaults.get(provider),
                    *allowed,
                )
                chosen = next(
                    (
                        candidate for candidate in candidates
                        if candidate in allowed
                        and (pro_only or candidate not in premium)
                    ),
                    "",
                )
            clean[provider] = chosen

        consensus_model = cfg.canonical_model_id(supplied.get("consensus"))
        consensus_valid = (
            consensus_model in cfg.CONSENSUS_ENGINE_ALIASES
            or consensus_model in allowed_direct_consensus
        )
        consensus_locked = (
            consensus_model.endswith("-Pro")
            or consensus_model in premium
        )
        if not consensus_valid or (not pro_only and consensus_locked):
            fallback = cfg.canonical_model_id(base["consensus"])
            fallback_valid = (
                fallback in cfg.CONSENSUS_ENGINE_ALIASES
                or fallback in allowed_direct_consensus
            )
            if (
                fallback_valid
                and (pro_only or not (
                    fallback.endswith("-Pro") or fallback in premium
                ))
            ):
                consensus_model = fallback
            else:
                consensus_model = "Gemini"
        clean["consensus"] = consensus_model
        clean_presets[preset_id] = clean
    normalized["preset_models"] = clean_presets

    consensus = [
        str(model).strip()
        for model in (normalized.get("consensus") or cfg.DEFAULT_CONSENSUS_MODELS)
        if str(model or "").strip()
    ]
    normalized_consensus = []
    for model in consensus:
        if model in normalized_consensus:
            continue
        if model in cfg.CONSENSUS_ENGINE_ALIASES or model in allowed_direct_consensus:
            normalized_consensus.append(model)
    for preset in clean_presets.values():
        if preset["consensus"] not in normalized_consensus:
            normalized_consensus.append(preset["consensus"])

    # Deep-Think-Modell: jeder gueltige Consensus-Wert (Alias oder Modell-ID
    # aus den Provider-Listen); ungueltige Werte fallen auf die Basis zurueck.
    # Das gewaehlte Modell muss in der Consensus-Liste bleiben, weil Deep Think
    # die Synthese fest daran koppelt.
    chosen_deep = str(normalized.get("deep_think_model") or "").strip()
    if not chosen_deep or (
        chosen_deep not in cfg.CONSENSUS_ENGINE_ALIASES
        and chosen_deep not in allowed_direct_consensus
    ):
        fallback = cfg._BASE_DEEP_THINK_CONSENSUS_MODEL
        chosen_deep = (
            fallback
            if fallback in allowed_direct_consensus
            else "Gemini"
        )
    if chosen_deep not in normalized_consensus:
        normalized_consensus.append(chosen_deep)
    normalized["deep_think_model"] = chosen_deep

    normalized["consensus"] = normalized_consensus

    # Differences-Judges je Provider: Firestore speichert immer eine vollstaendige
    # gueltige Zuordnung. Fallbacks stammen aus derselben Providerliste statt
    # unsichtbar ein hardcodiertes Modell wieder einzuschleusen.
    for field in ("judge_models", "judge_models_pro"):
        incoming_judges = normalized.get(field) or {}
        clean_judges = {}
        for provider in PROVIDER_KEYS:
            chosen = cfg.canonical_model_id(incoming_judges.get(provider), provider)
            allowed = list(normalized.get(provider) or [])
            if not (chosen and chosen in allowed):
                base_map = (
                    cfg._BASE_PRO_JUDGE_BY_PROVIDER
                    if field == "judge_models_pro"
                    else cfg._BASE_DIFFERENCES_JUDGE_BY_PROVIDER
                )
                candidates = (
                    base_map.get(provider),
                    clean_defaults.get(provider),
                    *allowed,
                )
                chosen = next(
                    (candidate for candidate in candidates if candidate in allowed),
                    "",
                )
            if chosen:
                clean_judges[provider] = chosen
        normalized[field] = clean_judges

    # Mapping Engine-Familie -> bevorzugte Judge-Familie: nur bekannte
    # Provider, nie die eigene Familie (Anti-Self-Judging). Fehlende Eintraege
    # bedeuten Auto (Prioritaetsliste).
    incoming_families = normalized.get("judge_families") or {}
    clean_families = {}
    for engine_provider in PROVIDER_KEYS:
        chosen = str(incoming_families.get(engine_provider) or "").strip()
        if chosen in PROVIDER_KEYS and chosen != engine_provider:
            clean_families[engine_provider] = chosen
    normalized["judge_families"] = clean_families

    return normalized


def _server_enforced_models() -> dict:
    """Kompatibilitaetsfeld fuer das Admin-Frontend.

    Providerlisten und Premium-Zuordnung kommen vollstaendig aus Firestore.
    Interne Deep-Think-/Alias-Fallbacks muessen nicht im normalen Picker stehen
    und werden deshalb nicht mehr als unsichtbare Pflichtzeilen erzwungen.
    """
    return {provider: [] for provider in PROVIDER_KEYS}


def _model_dependencies(data: dict) -> dict:
    """Liefert rein informative In-use-Gruende je Provider/Modell.

    Anders als das fruehere Required-Flag sperrt dies keine Zeile. Das Admin-UI
    kann eine Referenz zuerst umstellen oder beim Entfernen direkt sehen, welche
    abhaengige Auswahl noch angepasst werden muss.
    """
    dependencies = {provider: {} for provider in PROVIDER_KEYS}

    def add(provider, model, reason):
        if model not in set(data.get(provider) or []):
            return
        dependencies[provider].setdefault(model, [])
        if reason not in dependencies[provider][model]:
            dependencies[provider][model].append(reason)

    for provider, model in (data.get("defaults") or {}).items():
        add(provider, model, "Free default")
    for preset_id, preset in (data.get("preset_models") or {}).items():
        for provider in PROVIDER_KEYS:
            add(provider, preset.get(provider), f"{preset_id} preset")
    for tier, models in (data.get("watch_models") or {}).items():
        for provider, model in (models or {}).items():
            add(provider, model, f"{tier} Watch")
    for field, label in (
        ("judge_models", "Standard judge"),
        ("judge_models_pro", "Pro judge"),
    ):
        for provider, model in (data.get(field) or {}).items():
            add(provider, model, label)
    deep_model = data.get("deep_think_model")
    if deep_model not in cfg.CONSENSUS_ENGINE_ALIASES:
        for provider in PROVIDER_KEYS:
            add(provider, deep_model, "Deep Think")
    return dependencies


def _admin_meta(data: dict) -> dict:
    """Metadaten fuer Alias-Aufloesung, Labels und sichtbare Abhaengigkeiten."""
    aliases = {
        alias: {
            "provider": provider,
            "model": api_model,
            "label": cfg.get_model_label(api_model),
        }
        for alias, (provider, api_model) in cfg.CONSENSUS_ENGINE_ALIASES.items()
    }
    enforced = _server_enforced_models()
    labels = {}
    for models in enforced.values():
        for model in models:
            labels[model] = cfg.get_model_label(model)
    for model in cfg.ALL_ALLOWED_MODELS:
        labels.setdefault(model, cfg.get_model_label(model))
    return {
        "aliases": aliases,
        "enforced": enforced,
        "dependencies": _model_dependencies(data),
        "labels": labels,
        "deep_think_fallback": cfg._BASE_DEEP_THINK_CONSENSUS_MODEL,
        "judge_defaults": dict(cfg._BASE_DIFFERENCES_JUDGE_BY_PROVIDER),
        "judge_pro_defaults": dict(cfg._BASE_PRO_JUDGE_BY_PROVIDER),
        "judge_priority": list(cfg.JUDGE_FAMILY_PRIORITY),
        "preset_definitions": list(cfg.CONSENSUS_PRESET_DEFINITIONS),
    }


def _validate_admin_models_input(data: dict, normalized: dict) -> None:
    """Verhindert stille Server-Korrekturen bei Admin-Saves.

    Legacy-Dokumente duerfen beim GET normalisiert/migriert werden. Ein neuer
    Admin-POST muss dagegen bereits intern konsistent sein; sonst erhaelt die UI
    einen konkreten 400er statt eines scheinbar erfolgreichen Saves, nach dem
    Werte verschwinden oder wieder auftauchen.
    """
    premium = set(normalized["premium"])
    for provider in PROVIDER_KEYS:
        models = normalized[provider]
        if not models:
            raise HTTPException(
                status_code=400,
                detail=f"Configure at least one {provider} model.",
            )
        chosen_default = cfg.canonical_model_id(
            (data.get("defaults") or {}).get(provider), provider
        )
        if (
            not chosen_default
            or chosen_default not in models
            or chosen_default in premium
        ):
            raise HTTPException(
                status_code=400,
                detail=f"Select a valid non-Premium default for {provider}.",
            )

    configured_models = set().union(
        *(set(normalized[provider]) for provider in PROVIDER_KEYS)
    )
    invalid_premium = {
        cfg.canonical_model_id(model)
        for model in (data.get("premium") or [])
        if cfg.canonical_model_id(model) not in configured_models
    }
    if invalid_premium:
        raise HTTPException(
            status_code=400,
            detail="Premium contains models missing from provider lists: "
            + ", ".join(sorted(invalid_premium)),
        )

    incoming_consensus = [
        cfg.canonical_model_id(model)
        for model in (data.get("consensus") or [])
        if str(model or "").strip()
    ]
    if incoming_consensus != normalized["consensus"][:len(incoming_consensus)]:
        raise HTTPException(
            status_code=400,
            detail="Consensus contains an unknown or removed model.",
        )

    incoming_deep = cfg.canonical_model_id(data.get("deep_think_model"))
    if not incoming_deep or incoming_deep != normalized["deep_think_model"]:
        raise HTTPException(
            status_code=400,
            detail="Select a valid Deep Think consensus model.",
        )

    for field, label in (
        ("judge_models", "Standard judge"),
        ("judge_models_pro", "Pro judge"),
    ):
        incoming = data.get(field)
        if not isinstance(incoming, dict):
            raise HTTPException(status_code=400, detail=f"{field} must be a provider mapping")
        for provider in PROVIDER_KEYS:
            chosen = cfg.canonical_model_id(incoming.get(provider), provider)
            if not chosen or normalized[field].get(provider) != chosen:
                raise HTTPException(
                    status_code=400,
                    detail=f"Select a valid {label} for {provider}.",
                )

@router.get("/api/admin/models")
def get_models(request: Request):
    # This endpoint can be accessed by the admin frontend to get current models
    _require_admin(request, {})
    try:
        doc_ref = db_firestore.collection("app_config").document("models")
        doc = doc_ref.get()
        if doc.exists:
            raw_data = doc.to_dict() or {}
            data = normalize_models_document(raw_data)
            apply_limits(data.get("limits"))
            data["limits"] = get_limits_config()
            if data != raw_data:
                # Persistente Schema-/Legacy-Bereinigung: der Admin sieht nicht
                # nur eine geschoente Response, sondern Firestore wird zur
                # kanonischen, vom Runtime-Loader konsumierten Quelle.
                doc_ref.set(data)
        else:
            from app.core.config import ALLOWED_OPENAI_MODELS, ALLOWED_MISTRAL_MODELS, ALLOWED_ANTHROPIC_MODELS, ALLOWED_GEMINI_MODELS, ALLOWED_DEEPSEEK_MODELS, ALLOWED_GROK_MODELS, PREMIUM_MODELS
            data = {
                "openai": list(ALLOWED_OPENAI_MODELS),
                "mistral": list(ALLOWED_MISTRAL_MODELS),
                "anthropic": list(ALLOWED_ANTHROPIC_MODELS),
                "gemini": list(ALLOWED_GEMINI_MODELS),
                "deepseek": list(ALLOWED_DEEPSEEK_MODELS),
                "grok": list(ALLOWED_GROK_MODELS),
                "premium": list(PREMIUM_MODELS),
                "consensus": list(cfg.ALLOWED_CONSENSUS_MODELS),
                "preset_models": cfg.get_consensus_preset_models(),
                "defaults": dict(cfg.FREE_DEFAULT_MODEL_BY_PROVIDER),
                "watch_models": {
                    tier: dict(models) for tier, models in cfg.WATCH_MODELS_BY_TIER.items()
                },
                "deep_think_model": cfg.get_deep_think_consensus_model(),
                "judge_models": cfg.get_judge_models(),
                "judge_models_pro": cfg.get_pro_judge_models(),
                "judge_families": cfg.get_judge_families(),
                "limits": get_limits_config()
            }
        data["meta"] = _admin_meta(data)
        return data
    except Exception as e:
        logging.error(f"Error fetching models: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch models")

@router.post("/api/admin/models")
def update_models(request: Request, data: dict = Body(...)):
    _require_admin(request, data)
    required_keys = ["openai", "mistral", "anthropic", "gemini", "deepseek", "grok", "premium"]
    for k in required_keys:
        if k not in data or not isinstance(data[k], list):
            raise HTTPException(status_code=400, detail=f"Missing or invalid format for {k}. Must be a list of strings.")

    try:
        apply_limits(data.get("limits"))
        normalized = normalize_models_document(data)
        _validate_admin_models_input(data, normalized)
        incoming_presets = data.get("preset_models")
        if not isinstance(incoming_presets, dict):
            raise HTTPException(status_code=400, detail="preset_models must contain Fast, Balanced and High Quality mappings")
        for preset in cfg.CONSENSUS_PRESET_DEFINITIONS:
            preset_id = preset["id"]
            supplied = incoming_presets.get(preset_id)
            if not isinstance(supplied, dict):
                raise HTTPException(status_code=400, detail=f"preset_models.{preset_id} must be a model mapping")
            missing = [key for key in (*PROVIDER_KEYS, "consensus") if not supplied.get(key)]
            if missing:
                raise HTTPException(
                    status_code=400,
                    detail=f"preset_models.{preset_id} is missing: {', '.join(missing)}",
                )
            for key in (*PROVIDER_KEYS, "consensus"):
                chosen = (
                    cfg.canonical_model_id(supplied.get(key), key)
                    if key in PROVIDER_KEYS
                    else cfg.canonical_model_id(supplied.get(key))
                )
                if normalized["preset_models"][preset_id].get(key) != chosen:
                    raise HTTPException(
                        status_code=400,
                        detail=f"preset_models.{preset_id}.{key} is invalid for this tier",
                    )
        incoming_watch = data.get("watch_models")
        if not isinstance(incoming_watch, dict):
            raise HTTPException(status_code=400, detail="watch_models must contain free and pro mappings")
        for tier in ("free", "pro"):
            if not isinstance(incoming_watch.get(tier), dict):
                raise HTTPException(status_code=400, detail=f"watch_models.{tier} must be a provider mapping")
            for provider, chosen in incoming_watch[tier].items():
                canonical = cfg.canonical_model_id(chosen, provider)
                if normalized["watch_models"][tier].get(provider) != canonical:
                    raise HTTPException(
                        status_code=400,
                        detail=f"watch_models.{tier}.{provider} is invalid for this tier",
                    )
            if len(normalized["watch_models"][tier]) < 2:
                raise HTTPException(status_code=400, detail=f"Select at least two valid {tier} Watch models")
        doc_ref = db_firestore.collection("app_config").document("models")
        doc_ref.set({
            "openai": normalized["openai"],
            "mistral": normalized["mistral"],
            "anthropic": normalized["anthropic"],
            "gemini": normalized["gemini"],
            "deepseek": normalized["deepseek"],
            "grok": normalized["grok"],
            "premium": normalized["premium"],
            "consensus": normalized["consensus"],
            "preset_models": normalized["preset_models"],
            "defaults": normalized["defaults"],
            "watch_models": normalized["watch_models"],
            "deep_think_model": normalized["deep_think_model"],
            "judge_models": normalized["judge_models"],
            "judge_models_pro": normalized["judge_models_pro"],
            "judge_families": normalized["judge_families"],
            "limits": get_limits_config()
        })

        # Refresh the cache in config.py
        load_models_from_db()

        return {"status": "success", "message": "Configuration updated successfully."}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error updating models: {e}")
        raise HTTPException(status_code=500, detail="Failed to update models")
