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
from app.services import mailer, publisher_config, seo_data, watch_scheduler, watch_service
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
        return await asyncio.to_thread(seo_data_service.overview)
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
    # Provider-Listen behalten ihre (Admin-)Reihenfolge bei, damit die normalen
    # Picker exakt diese Anordnung anzeigen. Pflichtmodelle werden hinten ergaenzt.
    provider_frontier = {
        "openai": cfg.OPENAI_FRONTIER_LOW_MODEL,
        "anthropic": cfg.ANTHROPIC_FRONTIER_LOW_MODEL,
        "gemini": cfg.GEMINI_FRONTIER_LOW_MODEL,
        "grok": cfg.GROK_FRONTIER_LOW_MODEL,
    }
    for provider, frontier_model in provider_frontier.items():
        normalized[provider] = _ordered_unique(normalized.get(provider), ensure=(frontier_model,))

    normalized["anthropic"] = _ordered_unique(
        normalized.get("anthropic"), ensure=(cfg.DEFAULT_ANTHROPIC_MODEL, cfg.ANTHROPIC_PRO_MODEL)
    )
    normalized["mistral"] = _ordered_unique(
        normalized.get("mistral"), drop=cfg.DEPRECATED_MISTRAL_MODELS,
        ensure=(cfg.DEFAULT_MISTRAL_MODEL, cfg.MISTRAL_PRO_MODEL),
    )
    normalized["deepseek"] = _ordered_unique(
        normalized.get("deepseek"), drop=cfg.DEPRECATED_DEEPSEEK_MODELS,
        ensure=tuple(cfg.REQUIRED_DEEPSEEK_MODELS),
    )

    premium = set(normalized.get("premium") or [])
    premium.difference_update(cfg.FRONTIER_LOW_MODELS)
    premium.difference_update(cfg.DEPRECATED_DEEPSEEK_MODELS)
    premium.difference_update(cfg.DEPRECATED_GROK_MODELS)
    premium.update(cfg.EARLY_AND_PRO_MODELS)
    premium.update(cfg.REQUIRED_PRO_MODELS)
    normalized["premium"] = sorted(premium)

    # Free-Default je Provider: nur gueltig, wenn das Modell im Provider gelistet
    # und weder Premium noch Early ist (sonst saehe ein Free-Nutzer einen Sperr-Default).
    incoming_defaults = normalized.get("defaults") or {}
    clean_defaults = {}
    for provider in PROVIDER_KEYS:
        chosen = cfg.canonical_model_id(incoming_defaults.get(provider), provider)
        allowed = set(normalized.get(provider) or [])
        if chosen and chosen in allowed and chosen not in premium and chosen not in cfg.EARLY_MODELS:
            clean_defaults[provider] = chosen
    normalized["defaults"] = clean_defaults

    # Watch-Antwortmodelle: je Tier hoechstens ein Modell pro Provider. Free
    # darf keine Premium-/Early-Modelle verwenden. Bei Legacy-Dokumenten ohne
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
            if tier == "free" and (chosen in premium or chosen in cfg.EARLY_MODELS):
                continue
            tier_models[provider] = chosen
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
            if chosen not in set(normalized.get(provider) or []):
                chosen = base[provider]
            if not pro_only and (chosen in premium or chosen in cfg.EARLY_MODELS):
                chosen = base[provider]
            clean[provider] = chosen

        consensus_model = cfg.canonical_model_id(supplied.get("consensus"))
        consensus_valid = (
            consensus_model in cfg.CONSENSUS_ENGINE_ALIASES
            or consensus_model in allowed_direct_consensus
        )
        consensus_locked = (
            consensus_model.endswith("-Pro")
            or consensus_model in premium
            or consensus_model in cfg.EARLY_MODELS
        )
        if not consensus_valid or (not pro_only and consensus_locked):
            consensus_model = base["consensus"]
        clean["consensus"] = consensus_model
        clean_presets[preset_id] = clean
    normalized["preset_models"] = clean_presets

    # Nur die tatsaechlich konfigurierten Preset-Modelle muessen in den
    # Provider-Listen bleiben. Dadurch kann der Admin alte Basiswerte nach dem
    # Umstellen eines Presets wirklich umbenennen oder entfernen.
    for provider in PROVIDER_KEYS:
        selected = [preset[provider] for preset in clean_presets.values()]
        normalized[provider] = _ordered_unique(
            normalized.get(provider), ensure=selected
        )

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
    if cfg.GEMINI_FRONTIER_LOW_MODEL not in normalized_consensus:
        normalized_consensus.insert(0, cfg.GEMINI_FRONTIER_LOW_MODEL)
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
        chosen_deep = cfg._BASE_DEEP_THINK_CONSENSUS_MODEL
    if chosen_deep not in normalized_consensus:
        normalized_consensus.append(chosen_deep)
    normalized["deep_think_model"] = chosen_deep

    normalized["consensus"] = normalized_consensus

    # Differences-Judges je Provider (Standard- und Pro-Stufe): nur gueltig,
    # wenn das Modell im Provider gelistet und keine interne Frontier-Low-ID
    # ist. Ungueltige Eintraege werden verworfen (zur Laufzeit greift dann die
    # Basis, siehe apply_judge_models/apply_pro_judge_models).
    for field in ("judge_models", "judge_models_pro"):
        incoming_judges = normalized.get(field) or {}
        clean_judges = {}
        for provider in PROVIDER_KEYS:
            chosen = cfg.canonical_model_id(incoming_judges.get(provider), provider)
            allowed = set(normalized.get(provider) or [])
            if chosen and chosen in allowed and chosen not in cfg.FRONTIER_LOW_MODELS:
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
    """Modelle, die der Server je Provider immer wieder ergaenzt — im Admin-UI
    als 'Required' markiert, damit klar ist, warum sie nicht entfernbar sind."""
    enforced = {
        "openai": [cfg.DEFAULT_OPENAI_MODEL, cfg.OPENAI_FRONTIER_LOW_MODEL],
        "mistral": [cfg.DEFAULT_MISTRAL_MODEL, cfg.MISTRAL_PRO_MODEL],
        "anthropic": [cfg.DEFAULT_ANTHROPIC_MODEL, cfg.ANTHROPIC_PRO_MODEL, cfg.ANTHROPIC_FRONTIER_LOW_MODEL],
        "gemini": [cfg.DEFAULT_GEMINI_MODEL, cfg.GEMINI_35_FLASH_MODEL, cfg.GEMINI_FRONTIER_LOW_MODEL],
        "deepseek": sorted(cfg.REQUIRED_DEEPSEEK_MODELS),
        "grok": [cfg.DEFAULT_GROK_MODEL, cfg.GROK_FRONTIER_LOW_MODEL, "grok-4.3"],
    }
    return enforced


def _admin_meta() -> dict:
    """Metadaten fuer das Admin-UI: Alias-Aufloesung, server-erzwungene
    Modelle, Early-Set und Labels — macht das implizite Server-Verhalten
    (ensure/drop in normalize_models_document) im UI sichtbar."""
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
        "early": sorted(cfg.EARLY_MODELS),
        "premium_enforced": sorted(cfg.REQUIRED_PRO_MODELS | cfg.EARLY_AND_PRO_MODELS),
        "labels": labels,
        "consensus_forced_first": cfg.GEMINI_FRONTIER_LOW_MODEL,
        "deep_think_fallback": cfg._BASE_DEEP_THINK_CONSENSUS_MODEL,
        # Interne Frontier-Low-IDs sind keine direkt aufrufbaren API-Modelle
        # und deshalb als Judge nicht waehlbar.
        "frontier_low": sorted(cfg.FRONTIER_LOW_MODELS),
        "judge_defaults": dict(cfg._BASE_DIFFERENCES_JUDGE_BY_PROVIDER),
        "judge_pro_defaults": dict(cfg._BASE_PRO_JUDGE_BY_PROVIDER),
        "judge_priority": list(cfg.JUDGE_FAMILY_PRIORITY),
        "preset_definitions": list(cfg.CONSENSUS_PRESET_DEFINITIONS),
    }

@router.get("/api/admin/models")
def get_models(request: Request):
    # This endpoint can be accessed by the admin frontend to get current models
    _require_admin(request, {})
    try:
        doc_ref = db_firestore.collection("app_config").document("models")
        doc = doc_ref.get()
        if doc.exists:
            data = normalize_models_document(doc.to_dict())
            apply_limits(data.get("limits"))
            data["limits"] = get_limits_config()
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
        data["meta"] = _admin_meta()
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
        incoming_watch = data.get("watch_models")
        if not isinstance(incoming_watch, dict):
            raise HTTPException(status_code=400, detail="watch_models must contain free and pro mappings")
        for tier in ("free", "pro"):
            if not isinstance(incoming_watch.get(tier), dict):
                raise HTTPException(status_code=400, detail=f"watch_models.{tier} must be a provider mapping")
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
