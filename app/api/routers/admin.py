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


def normalize_models_document(data: dict) -> dict:
    normalized = dict(data or {})
    provider_frontier = {
        "openai": cfg.OPENAI_FRONTIER_LOW_MODEL,
        "anthropic": cfg.ANTHROPIC_FRONTIER_LOW_MODEL,
        "gemini": cfg.GEMINI_FRONTIER_LOW_MODEL,
        "grok": cfg.GROK_FRONTIER_LOW_MODEL,
    }
    for provider, frontier_model in provider_frontier.items():
        models = set(normalized.get(provider) or [])
        models.add(frontier_model)
        normalized[provider] = sorted(models)

    anthropic_models = set(normalized.get("anthropic") or [])
    anthropic_models.update({cfg.DEFAULT_ANTHROPIC_MODEL, cfg.ANTHROPIC_PRO_MODEL})
    normalized["anthropic"] = sorted(anthropic_models)

    mistral_models = set(normalized.get("mistral") or [])
    mistral_models.difference_update(cfg.DEPRECATED_MISTRAL_MODELS)
    mistral_models.add(cfg.DEFAULT_MISTRAL_MODEL)
    mistral_models.add(cfg.MISTRAL_PRO_MODEL)
    normalized["mistral"] = sorted(mistral_models)

    deepseek_models = set(normalized.get("deepseek") or [])
    deepseek_models.difference_update(cfg.DEPRECATED_DEEPSEEK_MODELS)
    deepseek_models.update(cfg.REQUIRED_DEEPSEEK_MODELS)
    normalized["deepseek"] = sorted(deepseek_models)

    premium = set(normalized.get("premium") or [])
    premium.difference_update(cfg.FRONTIER_LOW_MODELS)
    premium.difference_update(cfg.DEPRECATED_DEEPSEEK_MODELS)
    premium.update(cfg.EARLY_AND_PRO_MODELS)
    premium.update(cfg.REQUIRED_PRO_MODELS)
    normalized["premium"] = sorted(premium)
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
            "limits": get_limits_config()
        })
        
        # Refresh the cache in config.py
        load_models_from_db()
        
        return {"status": "success", "message": "Configuration updated successfully."}
    except Exception as e:
        logging.error(f"Error updating models: {e}")
        raise HTTPException(status_code=500, detail="Failed to update models")
