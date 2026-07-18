import os
import logging
from datetime import datetime, timedelta

import openai
from mistralai import Mistral
import google.generativeai as genai
from fastapi import APIRouter, Request, Body, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
import requests

from app.core.rate_limit import limiter
from app.core.security import verify_user_token, extract_id_token, db_firestore
from firebase_admin import firestore
from app.core.state import last_feedback_time
import app.core.config as cfg

# To be supplied by main.py dependency injection or imported 
# We'll import templates from main or setup a generic one here.
from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="templates")

router = APIRouter()

SITE_URL = "https://www.consens.io"
SITEMAP_URLS = (
    {"loc": f"{SITE_URL}/", "lastmod": "2026-06-03", "changefreq": "weekly", "priority": "1.0"},
    {"loc": f"{SITE_URL}/ai-model-comparison", "lastmod": "2026-06-03", "changefreq": "monthly", "priority": "0.8"},
    {"loc": f"{SITE_URL}/consensus-engine", "lastmod": "2026-07-09", "changefreq": "monthly", "priority": "0.8"},
    {"loc": f"{SITE_URL}/benchmark", "lastmod": "2026-06-30", "changefreq": "monthly", "priority": "0.7"},
    {"loc": f"{SITE_URL}/about", "lastmod": "2026-06-03", "changefreq": "monthly", "priority": "0.6"},
)


@router.get("/robots.txt", response_class=PlainTextResponse)
def robots_txt():
    return "\n".join([
        "User-agent: *",
        "Allow: /",
        f"Sitemap: {SITE_URL}/sitemap.xml",
        "",
    ])


@router.get("/sitemap.xml")
def sitemap_xml():
    """Sitemap-Index: statische Seiten + vom Admin indexierte Share-Seiten."""
    sitemaps = "\n".join(
        "  <sitemap>\n"
        f"    <loc>{SITE_URL}{path}</loc>\n"
        "  </sitemap>"
        for path in ("/sitemap-pages.xml", "/sitemap-shares.xml")
    )
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{sitemaps}\n"
        "</sitemapindex>\n"
    )
    return Response(content=xml, media_type="application/xml")


@router.get("/sitemap-pages.xml")
def sitemap_pages_xml():
    urls = "\n".join(
        [
            "  <url>\n"
            f"    <loc>{item['loc']}</loc>\n"
            f"    <lastmod>{item['lastmod']}</lastmod>\n"
            f"    <changefreq>{item['changefreq']}</changefreq>\n"
            f"    <priority>{item['priority']}</priority>\n"
            "  </url>"
            for item in SITEMAP_URLS
        ]
    )
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{urls}\n"
        "</urlset>\n"
    )
    return Response(content=xml, media_type="application/xml")

@router.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    return templates.TemplateResponse("landing.html", {"request": request})

@router.get("/privacy", response_class=HTMLResponse)
def privacy(req: Request):
    response = templates.TemplateResponse("privacy.html", {"request": req})
    response.headers["X-Robots-Tag"] = "noindex, noarchive"
    return response

@router.get("/imprint", response_class=HTMLResponse)
def imprint(req: Request):
    response = templates.TemplateResponse("imprint.html", {"request": req})
    response.headers["X-Robots-Tag"] = "noindex, noarchive"
    return response

@router.get("/terms", response_class=HTMLResponse)
def terms(req: Request):
    response = templates.TemplateResponse("terms.html", {"request": req})
    response.headers["X-Robots-Tag"] = "noindex, noarchive"
    return response

@router.get("/about", response_class=HTMLResponse)
def about(req: Request):
    return templates.TemplateResponse("about.html", {"request": req})

@router.get("/ai-model-comparison", response_class=HTMLResponse)
def ai_model_comparison(req: Request):
    return templates.TemplateResponse("ai-model-comparison.html", {"request": req})

@router.get("/consensus-engine", response_class=HTMLResponse)
def consensus_engine_page(req: Request):
    return templates.TemplateResponse("consensus-engine.html", {"request": req})

@router.get("/benchmark", response_class=HTMLResponse)
def benchmark(req: Request):
    return templates.TemplateResponse("benchmark.html", {"request": req})

@router.get("/app", response_class=HTMLResponse)
# Deep-Link auf das Watch-Dashboard: gleiche App-Shell, das Frontend öffnet
# die Watch-Seite anhand des Pfads (watch.js).
@router.get("/app/watches", response_class=HTMLResponse)
async def read_root(request: Request):
    firebase_config = {
        "firebase_api_key": os.environ.get("FIREBASE_API_KEY"),
        "firebase_auth_domain": os.environ.get("FIREBASE_AUTH_DOMAIN"),
        "firebase_project_id": os.environ.get("FIREBASE_PROJECT_ID"),
        "firebase_storage_bucket": os.environ.get("FIREBASE_STORAGE_BUCKET"),
        "firebase_messaging_sender_id": os.environ.get("FIREBASE_MESSAGING_SENDER_ID"),
        "firebase_app_id": os.environ.get("FIREBASE_APP_ID")
    }
    from app.core.config import PREMIUM_MODELS

    # Reihenfolge je Provider kommt aus der Admin-Konfiguration (get_ordered_models);
    # ohne Override deterministischer Auto-Sort.
    models = {
        "openai": cfg.get_ordered_models("openai"),
        "mistral": cfg.get_ordered_models("mistral"),
        "anthropic": cfg.get_ordered_models("anthropic"),
        "gemini": cfg.get_ordered_models("gemini"),
        "deepseek": cfg.get_ordered_models("deepseek"),
        "grok": cfg.get_ordered_models("grok"),
        "premium": list(PREMIUM_MODELS - cfg.EARLY_MODELS)
    }
    model_metadata = cfg.get_model_picker_metadata()
    model_labels = {model_id: meta["label"] for model_id, meta in model_metadata.items()}
    model_badges = {model_id: meta["badge"] for model_id, meta in model_metadata.items() if meta["badge"]}
    consensus_models = [
        {
            "value": model,
            "label": cfg.get_consensus_model_label(model),
            "badge": cfg.get_consensus_model_badge(model),
            "is_premium": cfg.is_premium_consensus_model(model),
            "is_frontier": model in cfg.EARLY_MODELS,
        }
        for model in cfg.ALLOWED_CONSENSUS_MODELS
    ]

    response = templates.TemplateResponse("index.html", {
        "request": request, 
        "free_limit": cfg.get_consensus_run_limit(False),
        "limits": cfg.get_limits_config(),
        "models": models,
        "default_models": cfg.FREE_DEFAULT_MODEL_BY_PROVIDER,
        "pro_default_models": cfg.DEFAULT_MODEL_BY_PROVIDER,
        "early_default_models": cfg.EARLY_DEFAULT_MODEL_BY_PROVIDER,
        "consensus_default_models": cfg.DEFAULT_MODEL_BY_PROVIDER,
        "consensus_models": consensus_models,
        "consensus_presets": cfg.get_consensus_presets(),
        "default_consensus_preset": cfg.DEFAULT_CONSENSUS_PRESET,
        "deep_think_consensus_model": cfg.get_deep_think_consensus_model(),
        "model_labels": model_labels,
        "model_badges": model_badges,
        "frontier_models": list(cfg.EARLY_MODELS),
        **firebase_config
    })
    response.headers["X-Robots-Tag"] = "noindex, follow"
    return response

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

@router.get("/admin/benchmark", response_class=HTMLResponse)
async def admin_benchmark_page(request: Request):
    firebase_config = {
        "firebase_api_key": os.environ.get("FIREBASE_API_KEY"),
        "firebase_auth_domain": os.environ.get("FIREBASE_AUTH_DOMAIN"),
        "firebase_project_id": os.environ.get("FIREBASE_PROJECT_ID"),
        "firebase_storage_bucket": os.environ.get("FIREBASE_STORAGE_BUCKET"),
        "firebase_messaging_sender_id": os.environ.get("FIREBASE_MESSAGING_SENDER_ID"),
        "firebase_app_id": os.environ.get("FIREBASE_APP_ID")
    }
    return templates.TemplateResponse("admin_benchmark.html", {
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

    # Datenminimierung: keine IP-Adresse speichern, Spam-Schutz läuft über
    # Rate-Limit und das 30-Sekunden-Fenster pro UID.
    feedback_data = {
        "message": message,
        "email": email,
        "uid": uid,
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
    model = cfg.LEADERBOARD_MODEL_ALIASES.get(model, model)
    if model not in cfg.VALID_LEADERBOARD_MODELS:
         logging.warning(f"Invalid vote attempt for model '{model}'")
         raise HTTPException(status_code=400, detail="Invalid model name.")

    try:
        uid = verify_user_token(id_token)
    except Exception:
        raise HTTPException(status_code=401, detail="Authentication failed")

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
    id_token = extract_id_token(request, data)
    if not id_token:
        raise HTTPException(status_code=401, detail="Please log in to test and use your own API keys.")
    try:
        verify_user_token(id_token)
    except Exception:
        raise HTTPException(status_code=401, detail="Authentication failed")

    try:
        openai_key = data.get("openai_key")
        mistral_key = data.get("mistral_key")
        anthropic_key = data.get("anthropic_key")
        gemini_key = data.get("gemini_key")
        deepseek_key = data.get("deepseek_key")
        grok_key = data.get("grok_key")

        submitted_keys = [openai_key, mistral_key, anthropic_key, gemini_key, deepseek_key, grok_key]
        if not any(is_valid(key) for key in submitted_keys):
            raise HTTPException(status_code=400, detail="Enter at least one API key to test.")
        
        results = {}
        
        # OpenAI Handshake
        try:
            if openai_key and len(openai_key) > 10:
                client = openai.OpenAI(api_key=openai_key, timeout=15)
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
            logging.warning("OpenAI key check failed: %s", e)
            results["OpenAI"] = "invalid"
        
        # Mistral Handshake
        try:
            if mistral_key and len(mistral_key) > 10:
                client = Mistral(api_key=mistral_key, timeout_ms=15000)
                response = client.chat.complete(
                    model=cfg.DEFAULT_MODEL_BY_PROVIDER["mistral"],
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=5
                )
                results["Mistral"] = "valid"
            else:
                results["Mistral"] = "invalid"
        except Exception as e:
            logging.warning("Mistral key check failed: %s", e)
            results["Mistral"] = "invalid"
            
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
                resp = requests.post(url, json=payload, headers=headers, timeout=15)
                if resp.status_code == 200:
                    results["Anthropic"] = "valid"
                else:
                    results["Anthropic"] = "invalid"
            else:
                results["Anthropic"] = "invalid"
        except Exception as e:
            logging.warning("Anthropic key check failed: %s", e)
            results["Anthropic"] = "invalid"
            
        # Gemini Handshake
        try:
            if gemini_key and len(gemini_key) > 10:
                genai.configure(api_key=gemini_key)
                model = genai.GenerativeModel(cfg.DEFAULT_MODEL_BY_PROVIDER["gemini"])
                resp = model.generate_content(
                    "ping",
                    generation_config={"max_output_tokens": 5},
                    request_options={"timeout": 15},
                )
                results["Gemini"] = "valid"
            else:
                results["Gemini"] = "invalid"
        except Exception as e:
            logging.warning("Gemini key check failed: %s", e)
            results["Gemini"] = "invalid"

        # DeepSeek Handshake
        try:
            if deepseek_key and len(deepseek_key) > 10:
                client = openai.OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com", timeout=15)
                response = client.chat.completions.create(
                    model=cfg.DEFAULT_MODEL_BY_PROVIDER["deepseek"],
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=5
                )
                results["DeepSeek"] = "valid"
            else:
                results["DeepSeek"] = "invalid"
        except Exception as e:
            logging.warning("DeepSeek key check failed: %s", e)
            results["DeepSeek"] = "invalid"
            
        # Grok Handshake
        try:
            if grok_key and len(grok_key) > 10:
                client = openai.OpenAI(api_key=grok_key, base_url="https://api.x.ai/v1", timeout=15)
                response = client.chat.completions.create(
                    model=cfg.DEFAULT_MODEL_BY_PROVIDER["grok"],
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=5
                )
                results["Grok"] = "valid"
            else:
                results["Grok"] = "invalid"
        except Exception as e:
            logging.warning("Grok key check failed: %s", e)
            results["Grok"] = "invalid"
            
        return {"results": results}
        
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Error checking API keys")
        raise HTTPException(status_code=500, detail="Error checking API keys.")
