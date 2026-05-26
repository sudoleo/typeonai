import os
import logging
from datetime import datetime, timedelta

import openai
from mistralai import Mistral
import google.generativeai as genai
from fastapi import APIRouter, Request, Body, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
import requests

from app.core.rate_limit import limiter
from app.core.security import verify_user_token, is_valid_session, extract_id_token, db_firestore
from firebase_admin import firestore
from app.core.state import last_feedback_time
import app.core.config as cfg

# To be supplied by main.py dependency injection or imported 
# We'll import templates from main or setup a generic one here.
from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="templates")

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    token = request.cookies.get("session") or request.headers.get("Authorization", "").removeprefix("Bearer ")
    force_landing = request.query_params.get("landing") == "1"
    if token and is_valid_session(token) and not force_landing:
        return RedirectResponse(url="/app")
    return templates.TemplateResponse("landing.html", {"request": request})

@router.get("/privacy", response_class=HTMLResponse)
def privacy(req: Request):
    return templates.TemplateResponse("privacy.html", {"request": req})

@router.get("/imprint", response_class=HTMLResponse)
def imprint(req: Request):
    return templates.TemplateResponse("imprint.html", {"request": req})

@router.get("/about", response_class=HTMLResponse)
def about(req: Request):
    return templates.TemplateResponse("about.html", {"request": req})

@router.get("/app", response_class=HTMLResponse)
async def read_root(request: Request):
    firebase_config = {
        "firebase_api_key": os.environ.get("FIREBASE_API_KEY"),
        "firebase_auth_domain": os.environ.get("FIREBASE_AUTH_DOMAIN"),
        "firebase_project_id": os.environ.get("FIREBASE_PROJECT_ID"),
        "firebase_storage_bucket": os.environ.get("FIREBASE_STORAGE_BUCKET"),
        "firebase_messaging_sender_id": os.environ.get("FIREBASE_MESSAGING_SENDER_ID"),
        "firebase_app_id": os.environ.get("FIREBASE_APP_ID")
    }
    from app.core.config import ALLOWED_OPENAI_MODELS, ALLOWED_MISTRAL_MODELS, ALLOWED_ANTHROPIC_MODELS, ALLOWED_GEMINI_MODELS, ALLOWED_DEEPSEEK_MODELS, ALLOWED_GROK_MODELS, PREMIUM_MODELS
    
    def model_picker_sort(model_name: str):
        return cfg.model_picker_sort_key(model_name)

    models = {
        "openai": sorted(list(ALLOWED_OPENAI_MODELS), key=model_picker_sort),
        "mistral": sorted(list(ALLOWED_MISTRAL_MODELS), key=model_picker_sort),
        "anthropic": sorted(list(ALLOWED_ANTHROPIC_MODELS), key=model_picker_sort),
        "gemini": sorted(list(ALLOWED_GEMINI_MODELS), key=model_picker_sort),
        "deepseek": sorted(list(ALLOWED_DEEPSEEK_MODELS), key=model_picker_sort),
        "grok": sorted(list(ALLOWED_GROK_MODELS), key=model_picker_sort),
        "premium": list(PREMIUM_MODELS - cfg.EARLY_FREE_MODELS)
    }
    model_metadata = cfg.get_model_picker_metadata()
    model_labels = {model_id: meta["label"] for model_id, meta in model_metadata.items()}
    model_badges = {model_id: meta["badge"] for model_id, meta in model_metadata.items() if meta["badge"]}

    return templates.TemplateResponse("index.html", {
        "request": request, 
        "free_limit": cfg.get_usage_limit(False),
        "limits": cfg.get_limits_config(),
        "models": models,
        "default_models": cfg.FREE_DEFAULT_MODEL_BY_PROVIDER,
        "pro_default_models": cfg.DEFAULT_MODEL_BY_PROVIDER,
        "consensus_default_models": cfg.DEFAULT_MODEL_BY_PROVIDER,
        "model_labels": model_labels,
        "model_badges": model_badges,
        "frontier_models": list(cfg.EARLY_FREE_MODELS),
        **firebase_config
    })

@router.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    firebase_config = {
        "firebase_api_key": os.environ.get("FIREBASE_API_KEY"),
        "firebase_auth_domain": os.environ.get("FIREBASE_AUTH_DOMAIN"),
        "firebase_project_id": os.environ.get("FIREBASE_PROJECT_ID"),
        "firebase_storage_bucket": os.environ.get("FIREBASE_STORAGE_BUCKET"),
        "firebase_messaging_sender_id": os.environ.get("FIREBASE_MESSAGING_SENDER_ID"),
        "firebase_app_id": os.environ.get("FIREBASE_APP_ID")
    }
    return templates.TemplateResponse("admin.html", {
        "request": request,
        **firebase_config
    })

@router.post("/feedback")
@limiter.limit("3/minute")
async def submit_feedback(request: Request, data: dict = Body(...)):
    message = data.get("message")
    email = data.get("email")
    id_token = extract_id_token(request, data)
    
    if not id_token:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        uid = verify_user_token(id_token)
    except Exception as e:
        raise HTTPException(status_code=401, detail="Authentication failed")
    
    now = datetime.utcnow()
    last_time = last_feedback_time.get(uid)
    if last_time and now - last_time < timedelta(seconds=30):
        raise HTTPException(status_code=429, detail="Please wait a few seconds before sending feedback again.")
    
    last_feedback_time[uid] = now

    if not message or message.strip() == "":
        raise HTTPException(status_code=400, detail="Feedback message must not be empty.")

    feedback_data = {
        "message": message,
        "email": email,
        "uid": uid,
        "ip_address": request.client.host,
        "timestamp": now
    }

    try:
        db_firestore.collection("feedback").add(feedback_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error when saving the feedback.")
    
    return {"status": "success", "message": "Feedback has been successfully submitted."}

ALLOWED_VOTE_TYPES = {"BestModel"}

@router.post("/vote")
@limiter.limit("3/minute")
async def record_vote(request: Request, data: dict = Body(...)):
    id_token = extract_id_token(request, data)
    model = data.get("model")
    vote_type = data.get("vote_type")

    if not id_token or not model or not vote_type:
        raise HTTPException(status_code=400, detail="Missing required fields: id_token, model or vote_type.")
    if vote_type not in ALLOWED_VOTE_TYPES:
        raise HTTPException(status_code=400, detail="Invalid vote type provided.")
    if model not in cfg.VALID_LEADERBOARD_MODELS:
         logging.warning(f"Invalid vote attempt for model '{model}'")
         raise HTTPException(status_code=400, detail="Invalid model name.")

    try:
        uid = verify_user_token(id_token)
    except Exception as e:
        raise HTTPException(status_code=401, detail="Authentication failed: " + str(e))

    try:
        doc_ref = db_firestore.collection("leaderboard").document(model)
        doc_ref.set({ vote_type: firestore.Increment(1) }, merge=True)
        return {"status": "success", "message": f"{vote_type} vote recorded for {model}"}
    except Exception as e:
        logging.exception("vote update failed")
        raise HTTPException(status_code=500, detail="Internal error")


def is_valid(key):
    return key is not None and len(key) > 10

@router.post("/check_keys")
@limiter.limit("3/minute")
async def check_keys(request: Request, data: dict = Body(...)):
    try:
        openai_key = data.get("openai_key")
        mistral_key = data.get("mistral_key")
        anthropic_key = data.get("anthropic_key")
        gemini_key = data.get("gemini_key")
        deepseek_key = data.get("deepseek_key")
        grok_key = data.get("grok_key")
        
        results = {}
        
        # OpenAI Handshake
        try:
            if openai_key and len(openai_key) > 10:
                client = openai.OpenAI(api_key=openai_key)
                response = client.chat.completions.create(
                    model=cfg.DEFAULT_MODEL_BY_PROVIDER["openai"],
                    messages=[
                        {"role": "system", "content": "ping"},
                        {"role": "user", "content": "ping"}
                    ],
                    max_completion_tokens=5
                )
                results["OpenAI"] = "valid"
            else:
                results["OpenAI"] = "invalid"
        except Exception as e:
            results["OpenAI"] = f"invalid: {str(e)}"
        
        # Mistral Handshake
        try:
            if mistral_key and len(mistral_key) > 10:
                client = Mistral(api_key=mistral_key)
                response = client.chat.complete(
                    model=cfg.DEFAULT_MODEL_BY_PROVIDER["mistral"],
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=5
                )
                results["Mistral"] = "valid"
            else:
                results["Mistral"] = "invalid"
        except Exception as e:
            results["Mistral"] = f"invalid: {str(e)}"
            
        # Anthropic Handshake
        try:
            if anthropic_key and len(anthropic_key) > 10:
                url = "https://api.anthropic.com/v1/messages"
                headers = {
                    "x-api-key": anthropic_key,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"
                }
                payload = {
                    "model": cfg.DEFAULT_MODEL_BY_PROVIDER["anthropic"],
                    "max_tokens": 5,
                    "messages": [{"role": "user", "content": "ping"}]
                }
                resp = requests.post(url, json=payload, headers=headers)
                if resp.status_code == 200:
                    results["Anthropic"] = "valid"
                else:
                    results["Anthropic"] = f"invalid: {resp.status_code}"
            else:
                results["Anthropic"] = "invalid"
        except Exception as e:
            results["Anthropic"] = f"invalid: {str(e)}"
            
        # Gemini Handshake
        try:
            if gemini_key and len(gemini_key) > 10:
                genai.configure(api_key=gemini_key)
                model = genai.GenerativeModel(cfg.DEFAULT_MODEL_BY_PROVIDER["gemini"])
                resp = model.generate_content("ping", generation_config={"max_output_tokens": 5})
                results["Gemini"] = "valid"
            else:
                results["Gemini"] = "invalid"
        except Exception as e:
            results["Gemini"] = f"invalid: {str(e)}"

        # DeepSeek Handshake
        try:
            if deepseek_key and len(deepseek_key) > 10:
                client = openai.OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com")
                response = client.chat.completions.create(
                    model=cfg.DEFAULT_MODEL_BY_PROVIDER["deepseek"],
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=5
                )
                results["DeepSeek"] = "valid"
            else:
                results["DeepSeek"] = "invalid"
        except Exception as e:
            results["DeepSeek"] = f"invalid: {str(e)}"
            
        # Grok Handshake
        try:
            if grok_key and len(grok_key) > 10:
                client = openai.OpenAI(api_key=grok_key, base_url="https://api.x.ai/v1")
                response = client.chat.completions.create(
                    model=cfg.DEFAULT_MODEL_BY_PROVIDER["grok"],
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=5
                )
                results["Grok"] = "valid"
            else:
                results["Grok"] = "invalid"
        except Exception as e:
            results["Grok"] = f"invalid: {str(e)}"
            
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking keys: {str(e)}")
