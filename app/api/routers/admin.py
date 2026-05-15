import logging
from fastapi import APIRouter, Request, Body, HTTPException

from app.core.security import extract_id_token, verify_user_token, is_user_admin, db_firestore
from app.core.config import load_models_from_db

router = APIRouter()

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
            return doc.to_dict()
        else:
            from app.core.config import ALLOWED_OPENAI_MODELS, ALLOWED_MISTRAL_MODELS, ALLOWED_ANTHROPIC_MODELS, ALLOWED_GEMINI_MODELS, ALLOWED_DEEPSEEK_MODELS, ALLOWED_GROK_MODELS, PREMIUM_MODELS
            return {
                "openai": list(ALLOWED_OPENAI_MODELS),
                "mistral": list(ALLOWED_MISTRAL_MODELS),
                "anthropic": list(ALLOWED_ANTHROPIC_MODELS),
                "gemini": list(ALLOWED_GEMINI_MODELS),
                "deepseek": list(ALLOWED_DEEPSEEK_MODELS),
                "grok": list(ALLOWED_GROK_MODELS),
                "premium": list(PREMIUM_MODELS)
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
        doc_ref = db_firestore.collection("app_config").document("models")
        doc_ref.set({
            "openai": data["openai"],
            "mistral": data["mistral"],
            "anthropic": data["anthropic"],
            "gemini": data["gemini"],
            "deepseek": data["deepseek"],
            "grok": data["grok"],
            "premium": data["premium"]
        })
        
        # Refresh the cache in config.py
        load_models_from_db()
        
        return {"status": "success", "message": "Models updated successfully."}
    except Exception as e:
        logging.error(f"Error updating models: {e}")
        raise HTTPException(status_code=500, detail="Failed to update models")
