import logging
from fastapi import APIRouter, Request, Body, HTTPException

from app.core.security import extract_id_token, verify_user_token, is_user_admin, db_firestore
import app.core.config as cfg
from app.core.config import apply_limits, get_limits_config, load_models_from_db
from app.services import share_snapshots as snapshots
from app.services.share_snapshots import ShareError

router = APIRouter()


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
    _require_admin(request, data)
    action = data.get("action")
    indexed = data.get("indexed")
    if indexed is not None and not isinstance(indexed, bool):
        raise HTTPException(status_code=400, detail="indexed must be a boolean")
    try:
        result = snapshots.moderate_share(share_id, action=action, indexed=indexed)
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
    premium.update(cfg.EARLY_AND_PRO_MODELS)
    premium.update(cfg.REQUIRED_PRO_MODELS)
    normalized["premium"] = sorted(premium)

    # Free-Default je Provider: nur gueltig, wenn das Modell im Provider gelistet
    # und weder Premium noch Early ist (sonst saehe ein Free-Nutzer einen Sperr-Default).
    incoming_defaults = normalized.get("defaults") or {}
    clean_defaults = {}
    for provider in PROVIDER_KEYS:
        chosen = str(incoming_defaults.get(provider) or "").strip()
        allowed = set(normalized.get(provider) or [])
        if chosen and chosen in allowed and chosen not in premium and chosen not in cfg.EARLY_MODELS:
            clean_defaults[provider] = chosen
    normalized["defaults"] = clean_defaults

    allowed_direct_consensus = set()
    for provider in PROVIDER_KEYS:
        allowed_direct_consensus.update(normalized.get(provider) or [])
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
    normalized["consensus"] = normalized_consensus
    return normalized

@router.get("/api/admin/models")
def get_models(request: Request):
    # This endpoint can be accessed by the admin frontend to get current models
    id_token = extract_id_token(request, {})
    if not id_token:
        raise HTTPException(status_code=401, detail="Authentication required")
        
    try:
        uid = verify_user_token(id_token)
        if not is_user_admin(uid):
            raise HTTPException(status_code=403, detail="Admin privileges required")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=401, detail="Authentication failed")
        
    try:
        doc_ref = db_firestore.collection("app_config").document("models")
        doc = doc_ref.get()
        if doc.exists:
            data = normalize_models_document(doc.to_dict())
            apply_limits(data.get("limits"))
            data["limits"] = get_limits_config()
            return data
        else:
            from app.core.config import ALLOWED_OPENAI_MODELS, ALLOWED_MISTRAL_MODELS, ALLOWED_ANTHROPIC_MODELS, ALLOWED_GEMINI_MODELS, ALLOWED_DEEPSEEK_MODELS, ALLOWED_GROK_MODELS, PREMIUM_MODELS
            return {
                "openai": list(ALLOWED_OPENAI_MODELS),
                "mistral": list(ALLOWED_MISTRAL_MODELS),
                "anthropic": list(ALLOWED_ANTHROPIC_MODELS),
                "gemini": list(ALLOWED_GEMINI_MODELS),
                "deepseek": list(ALLOWED_DEEPSEEK_MODELS),
                "grok": list(ALLOWED_GROK_MODELS),
                "premium": list(PREMIUM_MODELS),
                "consensus": list(cfg.ALLOWED_CONSENSUS_MODELS),
                "defaults": dict(cfg.FREE_DEFAULT_MODEL_BY_PROVIDER),
                "limits": get_limits_config()
            }
    except Exception as e:
        logging.error(f"Error fetching models: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch models")

@router.post("/api/admin/models")
def update_models(request: Request, data: dict = Body(...)):
    id_token = extract_id_token(request, data)
    if not id_token:
        raise HTTPException(status_code=401, detail="Authentication required")
        
    try:
        uid = verify_user_token(id_token)
        if not is_user_admin(uid):
            raise HTTPException(status_code=403, detail="Admin privileges required")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=401, detail="Authentication failed")
        
    required_keys = ["openai", "mistral", "anthropic", "gemini", "deepseek", "grok", "premium"]
    for k in required_keys:
        if k not in data or not isinstance(data[k], list):
            raise HTTPException(status_code=400, detail=f"Missing or invalid format for {k}. Must be a list of strings.")
            
    try:
        apply_limits(data.get("limits"))
        normalized = normalize_models_document(data)
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
            "defaults": normalized["defaults"],
            "limits": get_limits_config()
        })
        
        # Refresh the cache in config.py
        load_models_from_db()
        
        return {"status": "success", "message": "Configuration updated successfully."}
    except Exception as e:
        logging.error(f"Error updating models: {e}")
        raise HTTPException(status_code=500, detail="Failed to update models")
