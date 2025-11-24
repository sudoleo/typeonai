import os
from fastapi import FastAPI, Query, Request, HTTPException, Body
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
import openai
import requests
import base64, re
import time, logging
import random
from mistralai import Mistral
from dotenv import load_dotenv
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, auth, firestore
from typing import Optional
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from slowapi import Limiter
from slowapi.util import get_remote_address

class CustomSecurityMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                csp = (
                    "default-src 'self' https://fonts.googleapis.com https://fonts.gstatic.com https://cdn.jsdelivr.net https://www.gstatic.com; "
                    "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://www.gstatic.com https://apis.google.com https://accounts.google.com; "
                    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                    "img-src 'self' data: https://lh3.googleusercontent.com https:; "
                    "connect-src 'self' "
                    "https://firestore.googleapis.com "
                    "https://*.firebaseio.com "
                    "https://identitytoolkit.googleapis.com "
                    "https://securetoken.googleapis.com "
                    "https://firebaseinstallations.googleapis.com "
                    "https://content-firebaseappcheck.googleapis.com "
                    "https://www.gstatic.com "
                    "https://*.gstatic.com "
                    "https://apis.google.com "
                    "https://accounts.google.com "
                    "https://www.googleapis.com "
                    "https://*.googleapis.com "
                    "https://firebasestorage.googleapis.com "
                    "https://api.openai.com https://api.mistral.ai https://api.anthropic.com "
                    "https://api.x.ai https://api.deepseek.com https://api.exa.ai "
                    "https://cdn.jsdelivr.net; "
                    "frame-src 'self' https://accounts.google.com https://*.google.com https://*.gstatic.com https://*.firebaseapp.com https://*.web.app;"
                )
                headers[b"Content-Security-Policy"] = csp.encode("utf-8")
                headers[b"X-Content-Type-Options"] = b"nosniff"
                headers[b"X-Frame-Options"] = b"DENY"
                headers[b"Strict-Transport-Security"] = b"max-age=31536000; includeSubDomains"
                headers[b"Referrer-Policy"] = b"no-referrer-when-downgrade"
                message["headers"] = list(headers.items())
            await send(message)

        await self.app(scope, receive, send_wrapper)

app = FastAPI()
# Füge die Middleware direkt nach der App-Initialisierung hinzu
app.add_middleware(CustomSecurityMiddleware)

load_dotenv()

logging.basicConfig(level=logging.INFO)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

    
@app.exception_handler(HTTPException)
async def handle_http_exception(request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

@app.exception_handler(RequestValidationError)
async def handle_validation_exception(request, exc: RequestValidationError):
    # Angenehmer JSON-Körper für Validierungsfehler
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Validation failed", "details": exc.errors()},
    )

FREE_USAGE_LIMIT = 25
MAX_WORDS = 500
DEEP_SEARCH_MAX_WORDS = 1000
MAX_TOKENS = 2048
DEEP_SEARCH_MAX_TOKENS = 4096
CONSENSUS_MAX_TOKENS = 4096
DIFFERENCES_MAX_TOKENS = 1024
REASONING_EFFORT_FOR_DEEP = "low"
GEMINI_MAX_TOKENS = 2048
GEMINI_DEEP_MAX_TOKENS = 4096

PRO_USAGE_LIMIT = 500
PRO_DEEP_SEARCH_LIMIT = 50

# Modelle, die pro Anbieter erlaubt sind
ALLOWED_OPENAI_MODELS = {
    "gpt-5-nano",
    "gpt-5-mini",
    "gpt-4.1",
    "gpt-4o",
    "gpt-3.5-turbo",
}

ALLOWED_MISTRAL_MODELS = {
    "mistral-large-latest",
    "mistral-medium-latest",
    "mistral-small-latest",
}

ALLOWED_ANTHROPIC_MODELS = {
    "claude-haiku-4-5",
    "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-haiku-20241022",
}

ALLOWED_GEMINI_MODELS = {
    "gemini-2.5-flash",
    "gemini-2.0-flash",
}

ALLOWED_DEEPSEEK_MODELS = {
    "deepseek-chat",
}

ALLOWED_GROK_MODELS = {
    "grok-4-fast-non-reasoning-latest",
}

# Modelle, die pro Anbieter erlaubt sind
# WICHTIG: Hier müssen AUCH die Premium-Modelle rein, damit validate_model sie erkennt.

ALLOWED_OPENAI_MODELS = {
    "gpt-5-nano",
    "gpt-5-mini",
    "gpt-4.1",
    "gpt-4o",
    "gpt-3.5-turbo",
    # Premium Modelle hinzufügen:
    "gpt-5",
    "gpt-5-chat-latest",
}

ALLOWED_MISTRAL_MODELS = {
    "mistral-large-latest",
    "mistral-medium-latest",
    "mistral-small-latest",
}

ALLOWED_ANTHROPIC_MODELS = {
    "claude-haiku-4-5",
    "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-haiku-20241022",
    # Premium Modelle hinzufügen:
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-5", # Fallback falls Frontend Kurzform sendet
}

ALLOWED_GEMINI_MODELS = {
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    # Premium Modelle hinzufügen:
    "gemini-2.5-pro",
}

ALLOWED_DEEPSEEK_MODELS = {
    "deepseek-chat",
    # Premium Modelle hinzufügen:
    "deepseek-reasoner",
}

ALLOWED_GROK_MODELS = {
    "grok-4-fast-non-reasoning-latest",
    # Premium Modelle hinzufügen:
    "grok-4-latest",
    "grok-3-latest",
}

PREMIUM_MODELS = {
    # OpenAI High-End
    "gpt-5",
    "gpt-5-chat-latest",

    # Anthropic High-End
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-5",
    
    # Gemini Pro
    "gemini-2.5-pro",
    
    # DeepSeek Reasoner
    "deepseek-reasoner",
    
    # Grok
    "grok-4-latest",
    "grok-3-latest",
}

def get_system_prompt() -> str:
    # Aktuelles Datum in deiner Zeitzone (z.B. Europe/Berlin)
    now = datetime.now(ZoneInfo("Europe/Berlin"))
    today_str = now.strftime("%Y-%m-%d")

    return (
        f"Today is {today_str}. "
        "Please respond briefly and precisely, focusing only on the essentials."
    )

DEEP_THINK_PROMPT = "Deep Think: Focus as hard as you can! But only on the essentials."

usage_counter = {}  # { uid: anzahl_anfragen }
deep_search_usage = {}  # { uid: anzahl_deep_search_anfragen }

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gen-lang-client-0234219247-53b2b1c0e355.json"

# ganz oben in app.py, direkt nach deinen Imports
def is_valid_session(token: str) -> bool:
    """
    Prüft, ob das übergebene Firebase-ID-Token gültig ist.
    Gibt True zurück, wenn verify_user_token() keinen Fehler wirft.
    """
    try:
        verify_user_token(token)
        return True
    except Exception:
        return False

def count_words(text: str) -> int:
    return len(text.strip().split())

def validate_model(model: str, allowed: set, provider: str, is_pro: bool = False):
    """
    Prüft, ob das Modell existiert UND ob der Nutzer berechtigt ist.
    """
    # 1. Ist das Modell überhaupt technisch erlaubt?
    if model and model not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' is not allowed for {provider}."
        )
    
    # 2. Ist es ein Premium-Modell und der Nutzer ist NICHT Pro?
    if model in PREMIUM_MODELS and not is_pro:
        raise HTTPException(
            status_code=403, # Forbidden
            detail=f"The model '{model}' is reserved for Premium users. Please upgrade your plan."
        )
    
def query_openai(
    question: str,
    api_key: str,
    deep_search: bool = False,
    system_prompt: str = None,
    model_override: str = None
) -> str:
    if system_prompt is None:
        system_prompt = get_system_prompt()
    if deep_search:
        system_prompt += "\n" + DEEP_THINK_PROMPT

    client = openai.OpenAI(api_key=api_key)

    # Modell-Entscheidung: search_mode ist entfernt – nur deep_search & override steuern
    model_to_use = "gpt-5" if deep_search else (model_override or "gpt-5-mini")

    print(f"[MODEL] OpenAI -> {model_to_use} | deep_search={deep_search} | override={model_override}")

    user_msg = {"role": "user", "content": question}

    # ===== Deep Reasoning: Responses API =====
    if deep_search:
        try:
            resp = client.responses.create(
                model=model_to_use,
                reasoning={"effort": REASONING_EFFORT_FOR_DEEP},
                input=[
                    {"role": "system", "content": system_prompt},
                    user_msg,
                ],
                max_output_tokens=DEEP_SEARCH_MAX_TOKENS,
            )

            if getattr(resp, "status", None) == "incomplete":
                details = getattr(resp, "incomplete_details", None)
                reason = getattr(details, "reason", None) if details else None
                prefix = "[Info] answer not finished"
                if reason == "max_output_tokens":
                    prefix += f" (max_output_tokens={DEEP_SEARCH_MAX_TOKENS} reached)."
                partial = getattr(resp, "output_text", None)
                return (prefix + "\n\n" + partial) if partial else prefix

            if hasattr(resp, "output_text") and resp.output_text:
                return resp.output_text.strip()

            if hasattr(resp, "content") and resp.content:
                try:
                    text_parts = []
                    for block in resp.content:
                        if getattr(block, "type", "") in ("output_text", "text"):
                            text_parts.append(getattr(block, "text", "") or getattr(block, "content", ""))
                    if text_parts:
                        return "".join(text_parts).strip()
                except Exception:
                    pass

            return "Error: Empty response payload."

        except Exception as e:
            # Fallback: Chat Completions (ohne explicit reasoning)
            try:
                cmpl = client.chat.completions.create(
                    model=model_to_use,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        user_msg
                    ],
                    # GPT-5 nutzt max_completion_tokens
                    max_completion_tokens=DEEP_SEARCH_MAX_TOKENS
                )
                return cmpl.choices[0].message.content.strip()
            except Exception as e2:
                return f"Error with OpenAI (deep reasoning): {e} | fallback error: {e2}"

    # ===== Normal: Chat Completions =====
    try:
        cmpl = client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": system_prompt},
                user_msg
            ],
            max_completion_tokens=MAX_TOKENS
        )
        return cmpl.choices[0].message.content.strip()
    except Exception as e:
        return f"Error with OpenAI: {e}"


def query_mistral(question: str, api_key: str, system_prompt: str = None, deep_search: bool = False, model_override: str = None) -> str:
    """Fragt die Mistral API zu der gegebenen Frage unter Verwendung des übergebenen API Keys ohne Limit."""
    if system_prompt is None:
        system_prompt = get_system_prompt()

    if deep_search:
        system_prompt += "\n" + DEEP_THINK_PROMPT

    # Setze max_tokens basierend auf dem deep_search Flag
    max_tokens = DEEP_SEARCH_MAX_TOKENS if deep_search else MAX_TOKENS

    try:
        client = Mistral(api_key=api_key)
        model = model_override if (model_override and not deep_search) else "mistral-large-latest"

        print(f"[MODEL] Mistral -> {model} | deep_search={deep_search} | override={model_override}")

        response = client.chat.complete(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error with Mistral: {str(e)}"


def query_claude(question: str, api_key: str, system_prompt: str = None, deep_search: bool = False, model_override: str = None) -> str:
    """Fragt die Anthropic API (Claude) zu der gegebenen Frage unter Verwendung des übergebenen API Keys ohne Limit.
       Da die Anthropic API ein Token-Limit erwartet, setzen wir einen sehr hohen Wert ein."""
    if system_prompt is None:
        system_prompt = get_system_prompt()

    if deep_search:
        system_prompt += "\n" + DEEP_THINK_PROMPT

    # Setze max_tokens basierend auf dem deep_search Flag
    max_tokens = DEEP_SEARCH_MAX_TOKENS if deep_search else MAX_TOKENS

    try:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        payload = {
            "model": model_override if (model_override and not deep_search) else "claude-sonnet-4-5-20250929",
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": question}]
        }

        print(f"[MODEL] Claude -> {payload['model']} | deep_search={deep_search} | override={model_override}")

        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if "content" in data and isinstance(data["content"], list) and len(data["content"]) > 0:
                return data["content"][0]["text"]
            else:
                return "Error: No response found in the API response."
        else:
            return f"Error with Anthropic: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error with Anthropic: {str(e)}"

def query_gemini(
    question: str,
    user_api_key: Optional[str] = None,
    deep_search: bool = False,
    system_prompt: str = None,
    model_override: str = None,
    max_output_tokens: Optional[int] = None,
) -> str:
    if system_prompt is None:
        system_prompt = get_system_prompt()
    if deep_search:
        system_prompt += "\n" + DEEP_THINK_PROMPT

    # (A) API-Key setzen (bevorzugt expliziter Key)
    try:
        if user_api_key and user_api_key.strip():
            genai.configure(api_key=user_api_key.strip())
        elif os.environ.get("DEVELOPER_GEMINI_API_KEY"):
            genai.configure(api_key=os.environ["DEVELOPER_GEMINI_API_KEY"])
        else:
            genai.configure()  # falls Service Account genutzt wird
    except Exception as e:
        return f"Error with Gemini: configuration failed: {e}"

    # (B) Modell & Config
    model_name = "gemini-2.5-pro" if deep_search else (model_override or "gemini-2.5-flash")

    print(f"[MODEL] Gemini -> {model_name} | deep_search={deep_search} | override={model_override}")

    try:
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt,
            safety_settings=[{"category":"HARM_CATEGORY_HARASSMENT","threshold":"BLOCK_ONLY_HIGH"}],
        )

        # Tokenlimit defensiv erhöhen, falls nichts übergeben wurde
        eff_max = int(max_output_tokens) if max_output_tokens is not None else (4096 if deep_search else 2048)
        generation_config = {
            "max_output_tokens": eff_max,
            "temperature": 0.2,
        }

        # Große Eingaben leicht kappen, damit mehr Budget fürs Output bleibt
        # (kein Helper – inline, nur bei extrem langen Fragen)
        if question and len(question) > 12000:
            question = question[:12000] + " … [truncated]"

        # Anfrage
        base_content = "Do not ask any questions.\n---\n" + question
        resp = model.generate_content(base_content, generation_config=generation_config)

        # (C) Text sicher auslesen oder finish_reason erklären
        txt = (getattr(resp, "text", None) or "").strip()
        if txt:
            return txt

        cand = (getattr(resp, "candidates", []) or [None])[0]
        fr = getattr(cand, "finish_reason", None)
        frs = str(fr)

        # 2 = MAX_TOKENS → genau dein Fehlerfall
        if frs in ("2", "MAX_TOKENS", "FinishReason.MAX_TOKENS"):
            return "Error with Gemini: hit max tokens before producing text. Raise max_output_tokens or trim input."

        # Häufige weitere Gründe
        if frs in ("3", "SAFETY", "FinishReason.SAFETY"):
            return "Error with Gemini: response was blocked by safety filters."
        if frs in ("4", "RECITATION", "FinishReason.RECITATION"):
            return "Error with Gemini: response suppressed by recitation policy."

        return f"Error with Gemini: empty response payload (finish_reason={frs})."

    except Exception as e:
        return f"Error with Gemini: {e}"

    
def query_deepseek(question: str, api_key: str, system_prompt: str = None, deep_search: bool = False, model_override: str = None) -> str:
    """Fragt DeepSeek zu der gegebenen Frage unter Verwendung des übergebenen API Keys."""
    if system_prompt is None:
        system_prompt = get_system_prompt()

    if deep_search:
        system_prompt += "\n" + DEEP_THINK_PROMPT

    # Setze max_tokens basierend auf dem deep_search Flag
    max_tokens = DEEP_SEARCH_MAX_TOKENS if deep_search else MAX_TOKENS

    try:
        client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        model_to_use = "deepseek-reasoner" if deep_search else (model_override or "deepseek-chat")
        print(f"[MODEL] DeepSeek -> {model_to_use} | deep_search={deep_search} | override={model_override}")
        response = client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            stream=False,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error with DeepSeek: {str(e)}"
    
def query_grok(question: str, api_key: str, system_prompt: str = None, deep_search: bool = False, model_override: str = None) -> str:
    """Fragt die Grok API zu der gegebenen Frage unter Verwendung des übergebenen API Keys."""
    if system_prompt is None:
        system_prompt = get_system_prompt()

    if deep_search:
        system_prompt += "\n" + DEEP_THINK_PROMPT

    # Setze max_tokens basierend auf dem deep_search Flag
    max_tokens = DEEP_SEARCH_MAX_TOKENS if deep_search else MAX_TOKENS

    try:
        client = openai.OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        model_to_use = "grok-4-latest" if deep_search else (model_override or "grok-4-fast-non-reasoning-latest")

        print(f"[MODEL] Grok -> {model_to_use} | deep_search={deep_search} | override={model_override}")

        response = client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error with Grok: {str(e)}"
        

def exa_search(query: str, num_results: int = 5):
    search_url = "https://api.exa.ai/search"
    headers = {"Content-Type": "application/json", "x-api-key": os.getenv("DEVELOPER_EXA_API_KEY")}
    payload = {"query": query, "num_results": num_results}

    resp = requests.post(search_url, json=payload, headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])
    if not results:
        return {"results": []}

    ids = [r["id"] for r in results]

    # Inhalte holen
    contents_resp = requests.post(
        "https://api.exa.ai/contents",
        json={"ids": ids, "contents": {"max_characters": 1200, "include_html": False}},
        headers=headers,
        timeout=10
    )
    contents_resp.raise_for_status()
    contents_data = contents_resp.json()
    contents_by_id = {c["id"]: c for c in contents_data.get("results", contents_data.get("contents", []))}

    merged = []
    for r in results:
        cid = r["id"]
        c = contents_by_id.get(cid, {})
        merged.append({
            "id": r["id"],
            "title": r.get("title"),
            "url": r.get("url"),
            "text": c.get("text") or c.get("content") or c.get("snippet") or ""
        })
    return {"results": merged}


def clean_exa_text(raw: str) -> str:
    if not raw:
        return ""
    text = raw.replace("\r", "\n").strip()

    # Navigation Müll raus
    drop_prefixes = (
        "[Skip to", "- [Skip to", "[Jump to", "- [Jump to",
        "[LIVING ROOM IDEAS", "[HALLWAY IDEAS"
    )
    lines = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            continue
        if any(s.startswith(p) for p in drop_prefixes):
            continue
        lines.append(s)

    text = " ".join(lines)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def build_evidence_block(exa_results, max_sources: int = 5):
    sources = []
    for i, r in enumerate(exa_results.get("results", [])[:max_sources], start=1):
        cleaned = clean_exa_text(r.get("text", ""))
        extract = cleaned[:800]
        sources.append({
            "id": f"S{i}",
            "title": r["title"],
            "url": r["url"],
            "extract": extract
        })

    block_lines = ["Relevant web sources:"]
    for s in sources:
        block_lines.append(f"[{s['id']}] {s['title']} – {s['url']}\n{s['extract']}\n")

    return "\n".join(block_lines), sources

def prepare_prompt_with_websearch(question: str, search_mode: bool, base_system_prompt: str):
    if not search_mode:
        return base_system_prompt, None

    exa_key = os.getenv("DEVELOPER_EXA_API_KEY")
    if not exa_key:
        raise HTTPException(status_code=500, detail="Exa API key missing")

    raw = exa_search(question, num_results=6)
    evidence_block, sources = build_evidence_block(raw)

    enriched_prompt = f"""
You are an AI assistant that must ground its answers in the web sources below.

Guidelines:
- Use ONLY the information from the sources as factual evidence.
- Cite sources as [S1], [S2], etc.
- If information is missing, say so.

Web sources:
{evidence_block}

Original instructions:
{base_system_prompt}
""".strip()

    return enriched_prompt, sources

def query_consensus(
    question: str,
    answer_openai: str,
    answer_mistral: str,
    answer_claude: str,
    answer_gemini: str,
    answer_deepseek: str,
    answer_grok: str,
    best_model: str,
    excluded_models: list,
    consensus_model: str,
    api_keys: dict,
    search_mode: bool = False
) -> str:
    """
    Konsolidiert die Antworten der 6 Haupt-LLMs zu einer Konsensantwort.
    Web Search (search_mode) bedeutet: Antworten basieren schon auf Web-Context,
    aber Exa ist KEIN eigenes Modell mehr.
    """
    prompt_parts = []
    if search_mode:
        prompt_parts.append(
            "Note: The following responses are based on additional web context. "
            "If URLs are present in the model answers, include the most relevant ones in the final answer.\n\n"
        )

    prompt_parts.append(
        f"Please provide your answer in the same language as the user's question. "
        f"The question is: {question}\n\n"
    )

    if "OpenAI" not in excluded_models and answer_openai:
        prompt_parts.append(f"Response from GPT (OpenAI): {answer_openai}\n\n")
    if "Mistral" not in excluded_models and answer_mistral:
        prompt_parts.append(f"Response from Mistral: {answer_mistral}\n\n")
    if "Anthropic" not in excluded_models and answer_claude:
        prompt_parts.append(f"Response from Claude: {answer_claude}\n\n")
    if "Gemini" not in excluded_models and answer_gemini:
        prompt_parts.append(f"Response from Gemini: {answer_gemini}\n\n")
    if "DeepSeek" not in excluded_models and answer_deepseek:
        prompt_parts.append(f"Response from DeepSeek: {answer_deepseek}\n\n")
    if "Grok" not in excluded_models and answer_grok:
        prompt_parts.append(f"Response from Grok: {answer_grok}\n\n")

    if best_model:
        prompt_parts.append(
            f"The user marked the Answer from the Model: {best_model} as the best one. "
            "You receive multiple expert opinions on a specific question. "
            "Your task is to combine these responses into a comprehensive, correct, and coherent answer. "
            "Note: Experts can also make mistakes. Therefore, try to identify and exclude possible errors by comparing the answers. "
            "Structure the answer clearly and coherently. "
            "Provide only the final, balanced answer."
        )
    else:
        prompt_parts.append(
            "You receive multiple expert opinions on a specific question. "
            "Treat all expert opinions equally. Do not focus on the answer of one model. "
            "Your task is to combine these responses into a comprehensive, correct, and coherent answer. "
            "Note: Experts can also make mistakes. Therefore, try to identify and exclude possible errors by comparing the answers. "
            "Structure the answer clearly and coherently."
            "Provide only the final, balanced answer."
        )

    consensus_prompt = "".join(prompt_parts)

    try:
        if consensus_model == "OpenAI":
            client = openai.OpenAI(api_key=api_keys.get("OpenAI"))
            model_to_use = "gpt-5-mini"
            response = client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": ""},
                    {"role": "user", "content": consensus_prompt}
                ],
                max_completion_tokens=CONSENSUS_MAX_TOKENS
            )
            return response.choices[0].message.content.strip()

        elif consensus_model == "Mistral":
            client = Mistral(api_key=api_keys.get("Mistral"))
            response = client.chat.complete(
                model="mistral-large-latest",
                messages=[
                    {"role": "system", "content": ""},
                    {"role": "user", "content": consensus_prompt}
                ],
                max_tokens=CONSENSUS_MAX_TOKENS
            )
            return response.choices[0].message.content.strip()

        elif consensus_model == "Anthropic":
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": api_keys.get("Anthropic"),
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            payload = {
                "model": "claude-haiku-4-5",
                "max_tokens": 2048,
                "system": "",
                "messages": [{"role": "user", "content": consensus_prompt}]
            }
            response = requests.post(url, json=payload, headers=headers)
            if response.status_code == 200:
                data = response.json()
                if "content" in data and isinstance(data["content"], list) and len(data["content"]) > 0:
                    return data["content"][0]["text"]
                else:
                    return "Error: No response found in the API response."
            else:
                return f"Error with Anthropic: {response.status_code} - {response.text}"

        elif consensus_model == "Gemini":
            gemini_key = api_keys.get("Gemini")
            if gemini_key and gemini_key.strip() != "":
                genai.configure(api_key=gemini_key)
            else:
                genai.configure()

            model = genai.GenerativeModel("gemini-2.5-flash")
            generation_config = {"max_output_tokens": int(CONSENSUS_MAX_TOKENS)}

            response = model.generate_content(
                consensus_prompt,
                generation_config=generation_config
            )

            return (response.text or "").strip() or "Error: Empty response payload."

        elif consensus_model == "DeepSeek":
            client = openai.OpenAI(api_key=api_keys.get("DeepSeek"), base_url="https://api.deepseek.com")
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": " "},
                    {"role": "user", "content": consensus_prompt}
                ],
                max_tokens=CONSENSUS_MAX_TOKENS
            )
            return response.choices[0].message.content.strip()

        elif consensus_model == "Grok":
            client = openai.OpenAI(api_key=api_keys.get("Grok"), base_url="https://api.x.ai/v1")
            response = client.chat.completions.create(
                model="grok-4-fast-non-reasoning-latest",
                messages=[
                    {"role": "system", "content": " "},
                    {"role": "user", "content": consensus_prompt}
                ],
                max_tokens=CONSENSUS_MAX_TOKENS
            )
            return response.choices[0].message.content.strip()

        else:
            return "Invalid consensus model selected."
    except Exception as e:
        return f"Consensus error: {str(e)}"
    

def query_differences(
    answer_openai: str,
    answer_mistral: str,
    answer_claude: str,
    answer_gemini: str,
    answer_deepseek: str,
    answer_grok: str,
    consensus_answer: str,
    api_keys: dict,
    differences_model: str
) -> str:
    """
    Extrahiert die Unterschiede zwischen den Antworten der 6 Hauptmodelle,
    anonymisiert die Modellnamen und ordnet das bestbewertete Modell anschließend wieder zu.
    Web Search ist bereits in den Antworten eingebacken, Exa selbst taucht hier nicht mehr auf.
    """

    model_answers = [
        ("OpenAI",   answer_openai),
        ("Mistral",  answer_mistral),
        ("Claude",   answer_claude),
        ("Gemini",   answer_gemini),
        ("DeepSeek", answer_deepseek),
        ("Grok",     answer_grok),
    ]

    # Leere Antworten filtern
    model_answers = [(n, a) for (n, a) in model_answers if a]

    if not model_answers:
        return "Error in comparison: no model responses available."

    random.shuffle(model_answers)

    anon_map = {}
    lines = []
    labels = []
    for idx, (name, text) in enumerate(model_answers):
        label = chr(ord("A") + idx)      # A, B, C, ...
        anon_label = f"Model {label}"
        anon_map[anon_label] = name
        labels.append(anon_label)
        lines.append(f"- {anon_label}: {(text or '')[:4000]}")

    responses_text = "\n".join(lines)

    if len(labels) > 1:
        allowed_list = ", ".join(labels[:-1]) + " or " + labels[-1]
    else:
        allowed_list = labels[0]
    best_models_instruction = f"Choose from one of the following labels: {allowed_list}."

    differences_prompt = (
        "Analyze the LLM responses and assess how strongly they differ from each other. "
        "If all models respond almost identically, the consensus is very credible. "
        "If there are only linguistic variations, it is largely credible. "
        "If there are content nuances, it is partially credible. "
        "If there are clear contradictions, it is hardly or not credible."
        "Respond with one of the following sentences:\n\n"

        "- 'The consensus answer is **very** credible.'\n"
        "- 'The consensus answer is **largely** credible.'\n"
        "- 'The consensus answer is **partially** credible.'\n"
        "- 'The consensus answer is **hardly** credible.'\n"
        "- 'The consensus answer is **not** credible.'\n\n"

        "After the sentence, include a separator line and a **very brief explanation** of why these differences are relevant.\n\n"
        "Consensus answer:\n" + consensus_answer + "\n\n"
        "Model responses:\n" + responses_text + "\n\n"
        "Finally, subjectively determine which model provided the best answer. "
        + best_models_instruction + "\n"
        "Include your decision at the end of the response on a separate line, "
        "starting with 'BestModel:' followed by the **anonymized** model name.\n"

        "Response format:\n"
        "[Credibility statement]\n"
        "\n"
        "_____________\n"
        "\n"
        "[Very brief explanation of why these differences affect credibility.]\n\n"
        "(Info: Mark the model closest to the consensus as Best Model)\n"
        "BestModel: [Model name]"
    )

    try:
        if differences_model == "OpenAI":
            client = openai.OpenAI(api_key=api_keys.get("OpenAI"))
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "Answer in the exact same language as the Model responses."},
                    {"role": "user", "content": differences_prompt}
                ],
                max_tokens=DIFFERENCES_MAX_TOKENS
            )
            result = response.choices[0].message.content.strip()

        elif differences_model == "Mistral":
            client = Mistral(api_key=api_keys.get("Mistral"))
            response = client.chat.complete(
                model="mistral-large-latest",
                messages=[
                    {"role": "system", "content": "Answer in the exact same language as the Model responses."},
                    {"role": "user", "content": differences_prompt}
                ],
                max_tokens=DIFFERENCES_MAX_TOKENS
            )
            result = response.choices[0].message.content.strip()

        elif differences_model == "Anthropic":
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": api_keys.get("Anthropic"),
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            payload = {
                "model": "claude-haiku-4-5",
                "max_tokens": 1024,
                "system": "Answer in the exact same language as the Model responses.",
                "messages": [{"role": "user", "content": differences_prompt}]
            }
            resp = requests.post(url, json=payload, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                result = data["content"][0]["text"] if data.get("content") else ""
            else:
                return f"Error with Anthropic: {resp.status_code} - {resp.text}"

        elif differences_model == "Gemini":
            try:
                if api_keys.get("Gemini"):
                    genai.configure(api_key=api_keys["Gemini"])
                elif os.environ.get("DEVELOPER_GEMINI_API_KEY"):
                    genai.configure(api_key=os.environ["DEVELOPER_GEMINI_API_KEY"])
                else:
                    genai.configure()

                model = genai.GenerativeModel(
                    model_name="gemini-2.5-flash",
                    system_instruction="Answer in the exact same language as the Model responses.",
                    safety_settings=[{"category":"HARM_CATEGORY_HARASSMENT","threshold":"BLOCK_ONLY_HIGH"}],
                    generation_config={"max_output_tokens": int(DIFFERENCES_MAX_TOKENS), "temperature": 0.2}
                )

                resp = model.generate_content(differences_prompt)
                result = (getattr(resp, "text", None) or "").strip()
                if not result:
                    cand = (getattr(resp, "candidates", []) or [None])[0]
                    fr = getattr(cand, "finish_reason", None)
                    frs = str(fr)
                    if frs in ("2","FinishReason.SAFETY","SAFETY"):
                        return "Error with Gemini (differences): response was blocked by safety filters."
                    if frs in ("3","FinishReason.MAX_TOKENS","MAX_TOKENS"):
                        return "Error with Gemini (differences): hit max tokens before returning text."
                    return f"Error with Gemini (differences): empty candidate (finish_reason={frs})."

            except Exception as e:
                return f"Error with Gemini (differences): {e}"

        elif differences_model == "DeepSeek":
            client = openai.OpenAI(api_key=api_keys.get("DeepSeek"), base_url="https://api.deepseek.com")
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "Answer in the exact same language as the Model responses."},
                    {"role": "user", "content": differences_prompt}
                ],
                max_tokens=DIFFERENCES_MAX_TOKENS
            )
            result = response.choices[0].message.content.strip()

        elif differences_model == "Grok":
            client = openai.OpenAI(api_key=api_keys.get("Grok"), base_url="https://api.x.ai/v1")
            response = client.chat.completions.create(
                model="grok-4-fast-non-reasoning-latest",
                messages=[
                    {"role": "system", "content": "Answer in the exact same language as the Model responses."},
                    {"role": "user", "content": differences_prompt}
                ],
                max_tokens=DIFFERENCES_MAX_TOKENS
            )
            result = response.choices[0].message.content.strip()

        else:
            return "Invalid model selected for difference comparison."

    except Exception as e:
        return f"Error in comparison: {e}"

    if not result:
        return "Error in comparison: empty result from differences engine."

    # BestModel-Zeile rückübersetzen
    match = re.search(r"BestModel:\s*Model\s*([A-Z])", result)
    if match:
        anon_label = f"Model {match.group(1)}"
        real_name = anon_map.get(anon_label, anon_label)
        result = re.sub(
            r"BestModel:\s*Model\s*[A-Z]",
            f"BestModel: {real_name}",
            result
        )

    return result

# Initialisiere Firebase Admin (Beispiel, passe den Pfad zu deinem Service Account an)
cred = credentials.Certificate("consensai-firebase-adminsdk-fbsvc-9064a77134.json")
firebase_admin.initialize_app(cred)

# Erstelle einen Firestore-Client
db_firestore = firestore.client()

def verify_user_token(token: str, allow_unverified: bool = False) -> str:
    """
    Verifiziert das Firebase-ID-Token. Standardmäßig NUR verifizierte E-Mails zulassen.
    Mit allow_unverified=True kann man Endpoints wie /confirm-registration erlauben.
    """
    try:
        decoded_token = auth.verify_id_token(token, clock_skew_seconds=5)
        if not allow_unverified and not decoded_token.get("email_verified", False):
            raise Exception("Email not verified")
        return decoded_token["uid"]
    except Exception as e:
        # Detailiert loggen, aber nach außen nichts leaken
        logging.error(f"verify_user_token failed: {e}")
        # Nur eine generische Exception werfen – der Aufrufer entscheidet über HTTP-Status
        raise Exception("Invalid token")
    

def extract_id_token(request: Request, data: dict) -> Optional[str]:
    raw = data.get("id_token")
    # Leere oder "null"/"undefined" wie "kein Token" behandeln
    if raw is not None and str(raw).strip().lower() in {"", "null", "undefined"}:
        raw = None
    if raw:
        return raw
    # Fallback: Authorization-Header akzeptieren
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[len("Bearer "):].strip()
        if token:
            return token
    # Fallback: Cookie (falls du den mal setzt)
    cookie_token = request.cookies.get("session")
    if cookie_token:
        return cookie_token
    return None

def is_user_pro(uid: str) -> bool:
    """
    Liest aus Firestore, ob das Feld 'tier' auf 'premium' (oder 'pro') steht.
    """
    try:
        doc_ref = db_firestore.collection("users").document(uid)
        doc = doc_ref.get()
        
        if doc.exists:
            data = doc.to_dict()
            tier = data.get("tier", "").lower()
            # Prüfen auf "premium" (wie im Text beschrieben) oder "pro" zur Sicherheit
            return tier in ["premium", "pro"]
        return False
    except Exception as e:
        logging.error(f"Pro-Check Fehler für {uid}: {e}")
        return False

# 1) Landingpage unter '/'
@app.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    # Lies das Token aus dem Cookie (oder Authorization-Header), je nachdem wo du es speicherst
    token = request.cookies.get("session") or request.headers.get("Authorization", "").removeprefix("Bearer ")
    
    if token and is_valid_session(token):
        # eingeloggter Nutzer kommt direkt in die App
        return RedirectResponse(url="/app")
    
    # sonst Landingpage
    return templates.TemplateResponse("landing.html", {"request": request})

@app.get("/privacy", response_class=HTMLResponse)
def privacy(req: Request):
    return templates.TemplateResponse("privacy.html", {"request": req})

@app.get("/imprint", response_class=HTMLResponse)
def privacy(req: Request):
    return templates.TemplateResponse("imprint.html", {"request": req})

@app.get("/about", response_class=HTMLResponse)
def privacy(req: Request):
    return templates.TemplateResponse("about.html", {"request": req})

@app.get("/app", response_class=HTMLResponse)
async def read_root(request: Request):
    firebase_config = {
        "firebase_api_key": os.environ.get("FIREBASE_API_KEY"),
        "firebase_auth_domain": os.environ.get("FIREBASE_AUTH_DOMAIN"),
        "firebase_project_id": os.environ.get("FIREBASE_PROJECT_ID"),
        "firebase_storage_bucket": os.environ.get("FIREBASE_STORAGE_BUCKET"),
        "firebase_messaging_sender_id": os.environ.get("FIREBASE_MESSAGING_SENDER_ID"),
        "firebase_app_id": os.environ.get("FIREBASE_APP_ID")
    }
    return templates.TemplateResponse("index.html", {"request": request, "free_limit": FREE_USAGE_LIMIT, **firebase_config})

@app.get("/user_status")
@limiter.limit("20/minute")
async def get_user_status(request: Request):
    """
    Prüft den Status des Nutzers (Free vs. Pro) basierend auf dem ID-Token.
    Wird beim Seiten-Load (checkUserStatusOnLoad) aufgerufen.
    """
    # Token aus dem Authorization Header holen (Bearer <token>)
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = auth_header.split(" ")[1]

    try:
        # 1. UID verifizieren
        uid = verify_user_token(token)
        
        # 2. Status aus Firestore holen
        pro_status = is_user_pro(uid)

        # 3. Limits basierend auf Status setzen
        limit_regular = PRO_USAGE_LIMIT if pro_status else FREE_USAGE_LIMIT
        limit_deep = PRO_DEEP_SEARCH_LIMIT if pro_status else 0  # Free User haben 0 Deep Search

        return {
            "uid": uid,
            "is_pro": pro_status,
            "limit": limit_regular,
            "deep_limit": limit_deep
        }

    except Exception as e:
        logging.error(f"User status check failed: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")

@app.get("/bookmarks")
@limiter.limit("20/minute")
async def load_bookmarks(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication failed")

    id_token = auth_header.split(" ")[1]
    try:
        uid = verify_user_token(id_token)
    except Exception as e:
        logging.error(f"/bookmarks auth failed: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")
    
    try:
        bookmarks_ref = db_firestore.collection("users").document(uid).collection("bookmarks")
        query_ref = bookmarks_ref.order_by("timestamp", direction=firestore.Query.DESCENDING)
        docs = query_ref.stream()
        bookmarks = []
        for doc in docs:
            bookmark_data = doc.to_dict()
            bookmark_data["id"] = doc.id
            bookmarks.append(bookmark_data)
        return {"status": "success", "bookmarks": bookmarks}
    except Exception as e:
        logging.error(f"Error loading bookmarks for uid={uid}: {e}")
        raise HTTPException(status_code=500, detail="Error loading bookmarks")

# Globales Dictionary zum Speichern der IP-Adressen registrierter Nutzer
registered_ips = {}  # { ip_address: uid }

@app.post("/register")
@limiter.limit("3/minute")
async def register_user(request: Request, data: dict = Body(...)):    
    email = data.get("email")
    password = data.get("password")
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password must be provided.")

    try:
        # Überprüfe, ob die E-Mail bereits existiert
        try:
            existing_user = auth.get_user_by_email(email)
            # Falls kein Fehler auftritt, existiert der Nutzer bereits
            raise HTTPException(status_code=400, detail="This email is already registered.")
        except firebase_admin.auth.UserNotFoundError:
            # Keine Registrierung mit dieser E-Mail gefunden, also weiter
            pass

        user = auth.create_user(email=email, password=password)
        custom_token = auth.create_custom_token(user.uid)
        custom_token_str = custom_token.decode("utf-8")
        return {"uid": user.uid, "email": user.email, "customToken": custom_token_str}

    except HTTPException:
        # bereits bewusst gesetzte Meldungen (z.B. "already registered") durchreichen
        raise
    except Exception as e:
        logging.error(f"/register failed for {email}: {e}")
        # generische Meldung an den Client
        raise HTTPException(status_code=400, detail="Registration failed. Please try again later.")
    

@app.post("/confirm-registration")
async def confirm_registration(request: Request, data: dict = Body(...)):
    token = data.get("id_token")
    if not token:
        raise HTTPException(status_code=400, detail="Authentication failed")

    try:
        uid = verify_user_token(token, allow_unverified=True)
        user = auth.get_user(uid)
    except Exception as e:
        logging.error(f"/confirm-registration token error: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")

    if not user.email_verified:
        # Diese Info ist okay, weil sie nichts über Passwort / Existenz aussagt
        raise HTTPException(status_code=400, detail="E-mail address not yet verified.")

    ip_address = request.client.host

    if ip_address in registered_ips and registered_ips[ip_address] != uid:
        raise HTTPException(status_code=400, detail="Only one confirmed account per user/IP is allowed.")

    registered_ips[ip_address] = uid
    return {"status": "registered", "ip": ip_address}

    
@app.post("/usage")
@limiter.limit("20/minute")
async def get_usage_post(request: Request):
    """
    Liefert die verbleibenden Anfragen dynamisch zurück.
    Rechnet: (Limit_basierend_auf_Tier) - (Bisherige_Nutzung).
    """
    data = await request.json()
    token = data.get("id_token")
    
    try:
        uid = verify_user_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Authentication failed")
    
    # 1. Status prüfen
    pro_status = is_user_pro(uid)

    # 2. Limits festlegen
    limit_regular = PRO_USAGE_LIMIT if pro_status else FREE_USAGE_LIMIT
    limit_deep = PRO_DEEP_SEARCH_LIMIT if pro_status else 0

    # 3. Verbrauch abrufen
    current_usage = usage_counter.get(uid, 0)
    current_deep_usage = deep_search_usage.get(uid, 0)

    # 4. Verbleibend berechnen (verhindert negative Zahlen in der UI, falls mal überzogen wurde)
    remaining = int(limit_regular - current_usage)
    deep_remaining = int(limit_deep - current_deep_usage)

    return {
        "remaining": remaining,
        "deep_remaining": deep_remaining,
        "is_pro": pro_status,
        "total_limit": limit_regular
    }
    
# Globales Dictionary zum Speichern des letzten Feedback-Zeitstempels pro Nutzer
last_feedback_time = {}
    
@app.post("/feedback")
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
    
    # Restlicher Code bleibt unverändert...
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


ALLOWED_VOTE_TYPES = {"best", "exclude", "BestModel"}

@app.post("/vote")
@limiter.limit("3/minute")
async def record_vote(request: Request, data: dict = Body(...)):
    id_token = extract_id_token(request, data)
    model = data.get("model")
    vote_type = data.get("vote_type")

    if not id_token or not model or not vote_type:
        raise HTTPException(status_code=400, detail="Missing required fields: id_token, model or vote_type.")
    if vote_type not in ALLOWED_VOTE_TYPES:
        raise HTTPException(status_code=400, detail="Invalid vote type provided.")

    try:
        uid = verify_user_token(id_token)
    except Exception as e:
        raise HTTPException(status_code=401, detail="Authentication failed: " + str(e))

    try:
        doc_ref = db_firestore.collection("leaderboard").document(model)
        # Wichtig: set(..., merge=True) statt update()
        doc_ref.set({ vote_type: firestore.Increment(1) }, merge=True)
        return {"status": "success", "message": f"{vote_type} vote recorded for {model}"}
    except Exception as e:
        # hilfreicher log
        logging.exception("vote update failed")
        raise HTTPException(status_code=500, detail="Internal error")

@app.post("/bookmark")
@limiter.limit("20/minute")
async def save_bookmark(request: Request, data: dict = Body(...)):
    id_token     = data.get("id_token")
    question     = data.get("question")
    response_text= data.get("response")
    modelName    = data.get("modelName")
    mode         = data.get("mode")
    
    if not (id_token and question and response_text and modelName):
        raise HTTPException(status_code=400, detail="Missing required fields.")
    
    try:
        uid = verify_user_token(id_token)
    except Exception as e:
        raise HTTPException(status_code=401, detail="Authentication failed: " + str(e))
    
    # Berechne die Dokument-ID wie gehabt
    raw_id = base64.b64encode(question.encode()).decode()
    doc_id = re.sub(r'[^a-zA-Z0-9]', '_', raw_id)[:50]
    
    dataToMerge = {
        "query": question,
        "timestamp": firestore.SERVER_TIMESTAMP,
        "mode": mode,
        "responses": { modelName: response_text }
    }
    
    try:
        # Speichern (merge)
        doc_ref = (
            db_firestore
            .collection("users")
            .document(uid)
            .collection("bookmarks")
            .document(doc_id)
        )
        # speichern (merge)
        doc_ref.set(dataToMerge, merge=True)

        # **Neu:** direkt danach auslesen
        snap = doc_ref.get()
        bm = snap.to_dict()
        bm["id"] = snap.id

        return {
            "status":  "success",
            "message": f"Bookmark for {modelName} saved.",
            "bookmark": bm
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error saving bookmark: " + str(e))
    

@app.post("/bookmark/consensus")
@limiter.limit("3/minute")
async def save_bookmark_consensus(request: Request, data: dict = Body(...)):
    id_token = extract_id_token(request, data)
    question = data.get("question")
    consensusText = data.get("consensusText")
    differencesText = data.get("differencesText")
    
    if not id_token or not question or consensusText is None or differencesText is None:
        raise HTTPException(status_code=400, detail="Missing required fields.")
    
    try:
        uid = verify_user_token(id_token)
    except Exception as e:
        raise HTTPException(status_code=401, detail="Authentication failed: " + str(e))
    
    # Berechne Dokument-ID (wie oben)
    doc_id = base64.b64encode(question.encode()).decode()
    doc_id = re.sub(r'[^a-zA-Z0-9]', '_', doc_id)[:50]
    
    dataToMerge = {
        "responses": {
            "consensus": consensusText,
            "differences": differencesText
        }
    }
    
    try:
        db_firestore.collection("users").document(uid).collection("bookmarks").document(doc_id).set(dataToMerge, merge=True)
        return {"status": "success", "message": "Consensus and differences saved."}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error saving consensus: " + str(e))
    
    
@app.delete("/bookmark")
async def delete_bookmark(data: dict):
    id_token = data.get("id_token")
    bookmark_id = data.get("bookmarkId")
    
    if not id_token or not bookmark_id:
        raise HTTPException(status_code=400, detail="Missing required fields.")
    
    try:
        uid = verify_user_token(id_token)
    except Exception as e:
        raise HTTPException(status_code=401, detail="Authentication failed: " + str(e))
    
    try:
        db_firestore.collection("users").document(uid).collection("bookmarks").document(bookmark_id).delete()
        return {"status": "success", "message": "Bookmark deleted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error deleting bookmark: " + str(e))

    
# --- NEU: Tracking Endpoint in main.py ---

@app.post("/track-interest")
@limiter.limit("5/minute")
async def track_interest(request: Request, data: dict = Body(...)):
    """
    Speichert das Interesse an der Pro-Version in der DB.
    """
    token = data.get("id_token")
    source = data.get("source", "unknown")
    
    if not token:
         raise HTTPException(status_code=401, detail="Authentication failed")

    try:
        # 1. User verifizieren (deine existierende Funktion nutzen)
        uid = verify_user_token(token)
        user_email = auth.get_user(uid).email
        
        # 2. Daten vorbereiten
        interest_data = {
            "uid": uid,
            "email": user_email,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "source": source,
            "ip": request.client.host, # IP auch nützlich für Spamschutz
            "user_agent": request.headers.get("user-agent")
        }
        
        # 3. In "pro_waitlist" Collection schreiben (Backend Admin SDK hat immer Schreibrechte)
        db_firestore.collection("pro_waitlist").add(interest_data)
        
        return {"status": "success", "message": "Interest tracked"}

    except Exception as e:
        logging.error(f"Tracking error for token prefix {token[:10]}...: {e}")
        return {"status": "error", "detail": str(e)}
    

@app.post("/ask_openai")
@limiter.limit("5/minute")
async def ask_openai_post(request: Request, data: dict = Body(...)):
    question = data.get("question")

    # deep_search robust konvertieren
    deep_search_raw = data.get("deep_search", False)
    deep_search = True if str(deep_search_raw).lower() == "true" else False

    max_words_limit = DEEP_SEARCH_MAX_WORDS if deep_search else MAX_WORDS
    if count_words(question) > max_words_limit:
        raise HTTPException(status_code=400, detail=f"Input exceeds word limit of {max_words_limit}.")

    system_prompt = data.get("system_prompt")
    id_token = extract_id_token(request, data)
    api_key = data.get("api_key")
    model = data.get("model")

    # --- 1. Auth & Status Check ---
    is_pro_user = False
    uid = None

    if id_token:
        try:
            uid = verify_user_token(id_token)
            is_pro_user = is_user_pro(uid)
        except Exception:
            raise HTTPException(status_code=401, detail="Authentication failed")

    # --- 2. Deep Search Check (Strictly Pro Only) ---
    if deep_search and not is_pro_user:
        raise HTTPException(
            status_code=403, 
            detail="Deep Think is exclusively available for Pro users."
        )

    # --- 3. Modell Validierung ---
    validate_model(model, ALLOWED_OPENAI_MODELS, "OpenAI", is_pro=is_pro_user)

    active_count = data.get("active_count", 1)

    # --- 4. Quota Management ---
    if uid:
        increment = 1.0 / active_count
        
        # Limits festlegen basierend auf Status
        limit_regular = PRO_USAGE_LIMIT if is_pro_user else FREE_USAGE_LIMIT
        limit_deep = PRO_DEEP_SEARCH_LIMIT if is_pro_user else 0  # Free hat 0 Zugriff

        current_usage = usage_counter.get(uid, 0)
        current_deep_usage = deep_search_usage.get(uid, 0)

        # Prüfung Regular Usage
        if current_usage + increment > limit_regular:
            msg = "Pro usage limit reached." if is_pro_user else "Free usage limit reached. Upgrade to Pro."
            return {
                "error": msg,
                "free_usage_remaining": 0,
                "deep_remaining": int(limit_deep - current_deep_usage)
            }

        # Prüfung Deep Search Usage (nur relevant wenn deep_search=True)
        if deep_search:
            if current_deep_usage + increment > limit_deep:
                return {
                    "error": "Your Deep Think quota is exhausted for this period.",
                    "free_usage_remaining": int(limit_regular - current_usage),
                    "deep_remaining": 0
                }

        # Zähler erhöhen
        usage_counter[uid] = current_usage + increment
        if deep_search:
            deep_search_usage[uid] = current_deep_usage + increment

        # API Call
        developer_api_key = os.environ.get("DEVELOPER_OPENAI_API_KEY")
        if not developer_api_key:
            raise HTTPException(status_code=500, detail="Server error: API key missing")

        answer = query_openai(
            question, developer_api_key, deep_search=deep_search,
            system_prompt=system_prompt, model_override=model
        )

        # Remaining berechnen
        remaining_regular = int(limit_regular - usage_counter[uid])
        remaining_deep = int(limit_deep - deep_search_usage.get(uid, 0))

        return {
            "response": answer,
            "free_usage_remaining": remaining_regular,
            "deep_remaining": remaining_deep,
            "key_used": "Developer API Key"
        }

    # --- 5. Eigener API Key (Bypass Limits) ---
    elif api_key:
        answer = query_openai(
            question, api_key, deep_search=deep_search,
            system_prompt=system_prompt, model_override=model
        )
        return {
            "response": answer,
            "free_usage_remaining": "Unlimited",
            "deep_remaining": "Unlimited",
            "key_used": "User API Key"
        }

    else:
        raise HTTPException(status_code=400, detail="No auth provided.")


@app.post("/ask_mistral")
@limiter.limit("5/minute")
async def ask_mistral_post(request: Request, data: dict = Body(...)):
    question = data.get("question")
    deep_search_raw = data.get("deep_search", False)
    deep_search = True if str(deep_search_raw).lower() == "true" else False
    max_words_limit = DEEP_SEARCH_MAX_WORDS if deep_search else MAX_WORDS

    if count_words(question) > max_words_limit:
        raise HTTPException(status_code=400, detail=f"Input exceeds word limit.")

    system_prompt = data.get("system_prompt")
    id_token = extract_id_token(request, data)
    api_key = data.get("api_key")
    model = data.get("model")

    is_pro_user = False
    uid = None

    if id_token:
        try:
            uid = verify_user_token(id_token)
            is_pro_user = is_user_pro(uid)
        except Exception:
            raise HTTPException(status_code=401, detail="Authentication failed")

    if deep_search and not is_pro_user:
        raise HTTPException(status_code=403, detail="Deep Think is exclusively available for Pro users.")

    validate_model(model, ALLOWED_MISTRAL_MODELS, "Mistral", is_pro=is_pro_user)

    active_count = data.get("active_count", 1)

    if uid:
        increment = 1.0 / active_count
        limit_regular = PRO_USAGE_LIMIT if is_pro_user else FREE_USAGE_LIMIT
        limit_deep = PRO_DEEP_SEARCH_LIMIT if is_pro_user else 0

        current_usage = usage_counter.get(uid, 0)
        current_deep_usage = deep_search_usage.get(uid, 0)

        if current_usage + increment > limit_regular:
             return {"error": "Usage limit reached.", "free_usage_remaining": 0, "deep_remaining": int(limit_deep - current_deep_usage)}

        if deep_search and (current_deep_usage + increment > limit_deep):
             return {"error": "Deep Think quota exhausted.", "free_usage_remaining": int(limit_regular - current_usage), "deep_remaining": 0}

        usage_counter[uid] = current_usage + increment
        if deep_search:
            deep_search_usage[uid] = current_deep_usage + increment

        developer_api_key = os.environ.get("DEVELOPER_MISTRAL_API_KEY")
        if not developer_api_key:
             raise HTTPException(status_code=500, detail="Server error: API key missing")
        
        answer = query_mistral(question, developer_api_key, system_prompt, deep_search=deep_search, model_override=model)
        
        return {
            "response": answer,
            "free_usage_remaining": int(limit_regular - usage_counter[uid]),
            "deep_remaining": int(limit_deep - deep_search_usage.get(uid, 0)),
            "key_used": "Developer API Key"
        }

    elif api_key:
        answer = query_mistral(question, api_key, system_prompt, deep_search=deep_search, model_override=model)
        return {"response": answer, "free_usage_remaining": "Unlimited", "deep_remaining": "Unlimited", "key_used": "User API Key"}
    else:
        raise HTTPException(status_code=400, detail="No auth provided.")


@app.post("/ask_claude")
@limiter.limit("3/minute")
async def ask_claude_post(request: Request, data: dict = Body(...)):
    question = data.get("question")
    deep_search_raw = data.get("deep_search", False)
    deep_search = True if str(deep_search_raw).lower() == "true" else False
    max_words_limit = DEEP_SEARCH_MAX_WORDS if deep_search else MAX_WORDS

    if count_words(question) > max_words_limit:
        raise HTTPException(status_code=400, detail="Input exceeds word limit.")

    system_prompt = data.get("system_prompt")
    id_token = extract_id_token(request, data)
    api_key = data.get("api_key")
    model = data.get("model")

    is_pro_user = False
    uid = None

    if id_token:
        try:
            uid = verify_user_token(id_token)
            is_pro_user = is_user_pro(uid)
        except Exception:
            raise HTTPException(status_code=401, detail="Authentication failed")

    if deep_search and not is_pro_user:
        raise HTTPException(status_code=403, detail="Deep Think is exclusively available for Pro users.")

    validate_model(model, ALLOWED_ANTHROPIC_MODELS, "Anthropic", is_pro=is_pro_user)

    active_count = data.get("active_count", 1)

    if uid:
        increment = 1.0 / active_count
        limit_regular = PRO_USAGE_LIMIT if is_pro_user else FREE_USAGE_LIMIT
        limit_deep = PRO_DEEP_SEARCH_LIMIT if is_pro_user else 0

        current_usage = usage_counter.get(uid, 0)
        current_deep_usage = deep_search_usage.get(uid, 0)

        if current_usage + increment > limit_regular:
             return {"error": "Usage limit reached.", "free_usage_remaining": 0, "deep_remaining": int(limit_deep - current_deep_usage)}

        if deep_search and (current_deep_usage + increment > limit_deep):
             return {"error": "Deep Think quota exhausted.", "free_usage_remaining": int(limit_regular - current_usage), "deep_remaining": 0}

        usage_counter[uid] = current_usage + increment
        if deep_search:
            deep_search_usage[uid] = current_deep_usage + increment

        developer_api_key = os.environ.get("DEVELOPER_ANTHROPIC_API_KEY")
        if not developer_api_key:
             raise HTTPException(status_code=500, detail="Server error: API key missing")

        answer = query_claude(question, developer_api_key, system_prompt, deep_search=deep_search, model_override=model)
        
        return {
            "response": answer,
            "free_usage_remaining": int(limit_regular - usage_counter[uid]),
            "deep_remaining": int(limit_deep - deep_search_usage.get(uid, 0)),
            "key_used": "Developer API Key"
        }

    elif api_key:
        answer = query_claude(question, api_key, system_prompt, deep_search=deep_search, model_override=model)
        return {"response": answer, "free_usage_remaining": "Unlimited", "deep_remaining": "Unlimited", "key_used": "User API Key"}
    else:
        raise HTTPException(status_code=400, detail="No auth provided.")


@app.post("/ask_gemini")
@limiter.limit("3/minute")
async def ask_gemini_post(request: Request, data: dict = Body(...)):
    question = data.get("question")
    deep_search_raw = data.get("deep_search", False)
    deep_search = True if str(deep_search_raw).lower() == "true" else False
    max_words_limit = DEEP_SEARCH_MAX_WORDS if deep_search else MAX_WORDS
    max_tokens = GEMINI_DEEP_MAX_TOKENS if deep_search else GEMINI_MAX_TOKENS

    if count_words(question) > max_words_limit:
        raise HTTPException(status_code=400, detail="Input exceeds word limit.")

    system_prompt = data.get("system_prompt")
    use_own_keys = str(data.get("useOwnKeys", "false")).lower() == "true"
    id_token = extract_id_token(request, data)
    api_key = (data.get("api_key") or data.get("gemini_key"))
    model = data.get("model")

    is_pro_user = False
    uid = None

    if id_token:
        try:
            uid = verify_user_token(id_token)
            is_pro_user = is_user_pro(uid)
        except Exception:
            raise HTTPException(status_code=401, detail="Authentication failed")

    if deep_search and not is_pro_user:
        raise HTTPException(status_code=403, detail="Deep Think is exclusively available for Pro users.")

    validate_model(model, ALLOWED_GEMINI_MODELS, "Gemini", is_pro=is_pro_user)

    active_count = data.get("active_count", 1)

    if uid:
        increment = 1.0 / active_count
        limit_regular = PRO_USAGE_LIMIT if is_pro_user else FREE_USAGE_LIMIT
        limit_deep = PRO_DEEP_SEARCH_LIMIT if is_pro_user else 0

        current_usage = usage_counter.get(uid, 0)
        current_deep_usage = deep_search_usage.get(uid, 0)

        if current_usage + increment > limit_regular:
             return {"error": "Usage limit reached.", "free_usage_remaining": 0, "deep_remaining": int(limit_deep - current_deep_usage)}

        if deep_search and (current_deep_usage + increment > limit_deep):
             return {"error": "Deep Think quota exhausted.", "free_usage_remaining": int(limit_regular - current_usage), "deep_remaining": 0}

        usage_counter[uid] = current_usage + increment
        if deep_search:
            deep_search_usage[uid] = current_deep_usage + increment

        # Entscheidung: Key oder Service Account
        if use_own_keys:
            if not (api_key and api_key.strip()):
                raise HTTPException(status_code=400, detail="Missing user API key for Gemini.")
            answer = query_gemini(question, user_api_key=api_key.strip(), deep_search=deep_search, system_prompt=system_prompt, model_override=model, max_output_tokens=max_tokens)
            key_info = "User API Key"
        else:
            # Service Account (SaaS Budget)
            answer = query_gemini(question, user_api_key=None, deep_search=deep_search, system_prompt=system_prompt, model_override=model, max_output_tokens=max_tokens)
            key_info = "Service Account"

        return {
            "response": answer,
            "free_usage_remaining": int(limit_regular - usage_counter[uid]),
            "deep_remaining": int(limit_deep - deep_search_usage.get(uid, 0)),
            "key_used": key_info
        }

    else:
        # Guest mit eigenem Key
        if not (api_key and api_key.strip()):
            raise HTTPException(status_code=400, detail="No credentials provided.")
        answer = query_gemini(question, user_api_key=api_key.strip(), deep_search=deep_search, system_prompt=system_prompt, model_override=model, max_output_tokens=max_tokens)
        return {"response": answer, "key_used": "User API Key"}


@app.post("/ask_deepseek")
@limiter.limit("3/minute")
async def ask_deepseek_post(request: Request, data: dict = Body(...)):
    question = data.get("question")
    deep_search_raw = data.get("deep_search", False)
    deep_search = True if str(deep_search_raw).lower() == "true" else False
    max_words_limit = DEEP_SEARCH_MAX_WORDS if deep_search else MAX_WORDS

    if count_words(question) > max_words_limit:
        raise HTTPException(status_code=400, detail="Input exceeds word limit.")

    system_prompt = data.get("system_prompt")
    id_token = extract_id_token(request, data)
    api_key = data.get("api_key")
    model = data.get("model")

    is_pro_user = False
    uid = None

    if id_token:
        try:
            uid = verify_user_token(id_token)
            is_pro_user = is_user_pro(uid)
        except Exception:
            raise HTTPException(status_code=401, detail="Authentication failed")

    if deep_search and not is_pro_user:
        raise HTTPException(status_code=403, detail="Deep Think is exclusively available for Pro users.")

    validate_model(model, ALLOWED_DEEPSEEK_MODELS, "DeepSeek", is_pro=is_pro_user)

    active_count = data.get("active_count", 1)

    if uid:
        increment = 1.0 / active_count
        limit_regular = PRO_USAGE_LIMIT if is_pro_user else FREE_USAGE_LIMIT
        limit_deep = PRO_DEEP_SEARCH_LIMIT if is_pro_user else 0

        current_usage = usage_counter.get(uid, 0)
        current_deep_usage = deep_search_usage.get(uid, 0)

        if current_usage + increment > limit_regular:
             return {"error": "Usage limit reached.", "free_usage_remaining": 0, "deep_remaining": int(limit_deep - current_deep_usage)}

        if deep_search and (current_deep_usage + increment > limit_deep):
             return {"error": "Deep Think quota exhausted.", "free_usage_remaining": int(limit_regular - current_usage), "deep_remaining": 0}

        usage_counter[uid] = current_usage + increment
        if deep_search:
            deep_search_usage[uid] = current_deep_usage + increment

        developer_api_key = os.environ.get("DEVELOPER_DEEPSEEK_API_KEY")
        if not developer_api_key:
             raise HTTPException(status_code=500, detail="Server error: API key missing")

        answer = query_deepseek(question, developer_api_key, system_prompt, deep_search=deep_search, model_override=model)
        
        return {
            "response": answer,
            "free_usage_remaining": int(limit_regular - usage_counter[uid]),
            "deep_remaining": int(limit_deep - deep_search_usage.get(uid, 0)),
            "key_used": "Developer API Key"
        }

    elif api_key:
        answer = query_deepseek(question, api_key, system_prompt, deep_search=deep_search, model_override=model)
        return {"response": answer, "free_usage_remaining": "Unlimited", "deep_remaining": "Unlimited", "key_used": "User API Key"}
    else:
        raise HTTPException(status_code=400, detail="No auth provided.")


@app.post("/ask_grok")
@limiter.limit("3/minute")
async def ask_grok_post(request: Request, data: dict = Body(...)):
    question = data.get("question")
    deep_search_raw = data.get("deep_search", False)
    deep_search = True if str(deep_search_raw).lower() == "true" else False
    max_words_limit = DEEP_SEARCH_MAX_WORDS if deep_search else MAX_WORDS

    if count_words(question) > max_words_limit:
        raise HTTPException(status_code=400, detail="Input exceeds word limit.")

    system_prompt = data.get("system_prompt")
    id_token = extract_id_token(request, data)
    api_key = data.get("api_key")
    model = data.get("model")

    is_pro_user = False
    uid = None

    if id_token:
        try:
            uid = verify_user_token(id_token)
            is_pro_user = is_user_pro(uid)
        except Exception:
            raise HTTPException(status_code=401, detail="Authentication failed")

    if deep_search and not is_pro_user:
        raise HTTPException(status_code=403, detail="Deep Think is exclusively available for Pro users.")

    validate_model(model, ALLOWED_GROK_MODELS, "Grok", is_pro=is_pro_user)

    active_count = data.get("active_count", 1)

    if uid:
        increment = 1.0 / active_count
        limit_regular = PRO_USAGE_LIMIT if is_pro_user else FREE_USAGE_LIMIT
        limit_deep = PRO_DEEP_SEARCH_LIMIT if is_pro_user else 0

        current_usage = usage_counter.get(uid, 0)
        current_deep_usage = deep_search_usage.get(uid, 0)

        if current_usage + increment > limit_regular:
             return {"error": "Usage limit reached.", "free_usage_remaining": 0, "deep_remaining": int(limit_deep - current_deep_usage)}

        if deep_search and (current_deep_usage + increment > limit_deep):
             return {"error": "Deep Think quota exhausted.", "free_usage_remaining": int(limit_regular - current_usage), "deep_remaining": 0}

        usage_counter[uid] = current_usage + increment
        if deep_search:
            deep_search_usage[uid] = current_deep_usage + increment

        developer_api_key = os.environ.get("DEVELOPER_GROK_API_KEY")
        if not developer_api_key:
             raise HTTPException(status_code=500, detail="Server error: API key missing")

        answer = query_grok(question, developer_api_key, system_prompt, deep_search=deep_search, model_override=model)
        
        return {
            "response": answer,
            "free_usage_remaining": int(limit_regular - usage_counter[uid]),
            "deep_remaining": int(limit_deep - deep_search_usage.get(uid, 0)),
            "key_used": "Developer API Key"
        }

    elif api_key:
        answer = query_grok(question, api_key, system_prompt, deep_search=deep_search, model_override=model)
        return {"response": answer, "free_usage_remaining": "Unlimited", "deep_remaining": "Unlimited", "key_used": "User API Key"}
    else:
        raise HTTPException(status_code=400, detail="No auth provided.")

@app.post("/prepare")
async def prepare(request: Request, data: dict = Body(...)):
    question = (data.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Missing 'question' in request body.")

    # search_mode robust von String → Bool
    search_mode_raw = data.get("search_mode", False)
    search_mode = True if str(search_mode_raw).lower() == "true" else bool(search_mode_raw)

    base_system_prompt = data.get("system_prompt") or get_system_prompt()

    # Wenn Web Search nicht aktiv ist: keine Exa-Suche, keine Quota-Prüfung nötig
    if not search_mode:
        return {
            "system_prompt": base_system_prompt,
            "sources": []
        }

    # Ab hier: Web Search aktiv → Auth + FREE_USAGE_LIMIT-Check
    id_token = extract_id_token(request, data)
    if not id_token:
        # Frontend wertet das wie gehabt als error aus
        raise HTTPException(status_code=401, detail="Authentication required for web search.")

    try:
        uid = verify_user_token(id_token)
    except Exception:
        raise HTTPException(status_code=401, detail="Authentication failed for web search.")

    # Nur prüfen, NICHT erhöhen – Exa zählt nicht am usage_counter
    current_usage = usage_counter.get(uid, 0)
    if current_usage >= FREE_USAGE_LIMIT:
        # Ganz wichtig: Exa wird hier NICHT aufgerufen
        raise HTTPException(
            status_code=403,
            detail="Your free quota is exhausted. Web search is only available with your own API keys."
        )

    # Quota ok → Exa-Suche durchführen und Evidence-Block bauen
    final_prompt, sources = prepare_prompt_with_websearch(
        question=question,
        search_mode=search_mode,
        base_system_prompt=base_system_prompt
    )

    return {
        "system_prompt": final_prompt,
        "sources": sources or []
    }


@app.post("/consensus")
@limiter.limit("3/minute")
async def consensus(request: Request, data: dict = Body(...)):
    id_token = extract_id_token(request, data)
    use_own_keys = data.get("useOwnKeys", False)
    # Neuer Parameter: search_mode
    search_mode = data.get("search_mode", False)
    
    if id_token:
        if not use_own_keys:
            try:
                uid = verify_user_token(id_token)
            except Exception:
                raise HTTPException(status_code=401, detail="Invalid token")
            current_usage = usage_counter.get(uid, 0)
            if current_usage >= FREE_USAGE_LIMIT:
                raise HTTPException(
                    status_code=403,
                    detail="Your free quota has been used up. Please store your own API keys."
                )
            usage_counter[uid] = current_usage + 1
    else:
        use_own_keys = True

    # Parameter extrahieren
    question        = data.get("question")
    answer_openai   = data.get("answer_openai")
    answer_mistral  = data.get("answer_mistral")
    answer_claude   = data.get("answer_claude")
    answer_gemini   = data.get("answer_gemini")
    answer_deepseek = data.get("answer_deepseek")
    answer_grok     = data.get("answer_grok")
    best_model      = data.get("best_model", "")
    consensus_model = data.get("consensus_model")
    excluded_models = data.get("excluded_models", [])

    # API Keys setzen: Bei useOwnKeys werden die vom Nutzer übermittelten Keys genutzt,
    # andernfalls wird für fehlende Keys auf die Developer Keys zurückgegriffen.
    api_keys = {}
    if use_own_keys:
        api_keys["OpenAI"] = data.get("openai_key")
        api_keys["Mistral"] = data.get("mistral_key")
        api_keys["Anthropic"] = data.get("anthropic_key")
        api_keys["Gemini"] = data.get("gemini_key")
        api_keys["DeepSeek"] = data.get("deepseek_key")
        api_keys["Grok"] = data.get("grok_key")
        api_keys["Exa"] = data.get("exa_key")

    else:
        api_keys["OpenAI"] = data.get("openai_key") or os.environ.get("DEVELOPER_OPENAI_API_KEY")
        api_keys["Mistral"] = data.get("mistral_key") or os.environ.get("DEVELOPER_MISTRAL_API_KEY")
        api_keys["Anthropic"] = data.get("anthropic_key") or os.environ.get("DEVELOPER_ANTHROPIC_API_KEY")
        api_keys["Gemini"] = data.get("gemini_key") or os.environ.get("DEVELOPER_GEMINI_API_KEY")
        api_keys["DeepSeek"] = data.get("deepseek_key") or os.environ.get("DEVELOPER_DEEPSEEK_API_KEY")
        api_keys["Grok"] = data.get("grok_key") or os.environ.get("DEVELOPER_GROK_API_KEY")
        api_keys["Exa"] = data.get("exa_key") or os.environ.get("DEVELOPER_EXA_API_KEY")

    # Validierung der erforderlichen Parameter (nur für Modelle, die nicht ausgeschlossen wurden)
    missing = []
    if not question:
        missing.append("question")
    if not consensus_model:
        missing.append("consensus_model")

    if "OpenAI" not in excluded_models and not answer_openai:
        missing.append("OpenAI")
    if "Mistral" not in excluded_models and not answer_mistral:
        missing.append("Mistral")
    if "Anthropic" not in excluded_models and not answer_claude:
        missing.append("Anthropic")
    if "Gemini" not in excluded_models and not answer_gemini:
        missing.append("Gemini")
    if "DeepSeek" not in excluded_models and not answer_deepseek:
        missing.append("DeepSeek")
    if "Grok" not in excluded_models and not answer_grok:
        missing.append("Grok")

    if missing:
        raise HTTPException(status_code=400, detail="Missing parameters: " + ", ".join(missing))
    
    # Engine-Key-Check (wichtig, um 401 der Engine zu vermeiden)
    engine = consensus_model
    engine_key_map = {
        "OpenAI": "OpenAI",
        "Mistral": "Mistral",
        "Anthropic": "Anthropic",
        "Gemini": "Gemini",
        "DeepSeek": "DeepSeek",
        "Grok": "Grok",
    }
    need_key_for = engine_key_map.get(engine)
    if need_key_for:
        if engine == "Gemini":
            # Erlaube drei Varianten:
            # 1) expliziter Key (User- oder Dev-Key),
            # 2) Dev-Key aus ENV,
            # 3) Service Account via GOOGLE_APPLICATION_CREDENTIALS, aber NUR wenn nicht useOwnKeys
            has_explicit_key = bool(api_keys.get("Gemini"))
            has_dev_key      = bool(os.environ.get("DEVELOPER_GEMINI_API_KEY"))
            using_service_acct = (not use_own_keys) and bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))

            if not (has_explicit_key or has_dev_key or using_service_acct):
                raise HTTPException(
                    status_code=400,
                    detail=("Missing credentials for selected consensus engine: Gemini. "
                            "Provide a Gemini API key or configure a Service Account on the server.")
                )
        else:
            if not api_keys.get(need_key_for):
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing API key for selected consensus engine: {engine}."
                )

    if best_model and best_model in excluded_models:
        raise HTTPException(status_code=400, detail="The answer marked as best must not be excluded.")

    consensus_answer = query_consensus(
        question,
        answer_openai,
        answer_mistral,
        answer_claude,
        answer_gemini,
        answer_deepseek,
        answer_grok,
        best_model,
        excluded_models,
        consensus_model,
        api_keys,
        search_mode
    )

    differences = query_differences(
        answer_openai,
        answer_mistral,
        answer_claude,
        answer_gemini,
        answer_deepseek,
        answer_grok,
        consensus_answer,
        api_keys,
        differences_model=consensus_model
    )

    return {"consensus_response": consensus_answer, "differences": differences}


def is_valid(key):
    # Example validation: Key is considered valid if it is present and longer than 10 characters.
    return key is not None and len(key) > 10


@app.post("/check_keys")
@limiter.limit("3/minute")
async def check_keys(request: Request, data: dict = Body(...)):
    try:
        openai_key = data.get("openai_key")
        mistral_key = data.get("mistral_key")
        anthropic_key = data.get("anthropic_key")
        gemini_key = data.get("gemini_key")
        deepseek_key = data.get("deepseek_key")
        grok_key = data.get("grok_key")
        exa_key = data.get("exa_key")
        
        results = {}
        
        # OpenAI Handshake (minimaler Chat-Request) – neue Syntax
        try:
            if openai_key and len(openai_key) > 10:
                client = openai.OpenAI(api_key=openai_key)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "ping"},
                        {"role": "user", "content": "ping"}
                    ]
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
                model = "mistral-large-latest"
                _ = client.chat.complete(
                    model=model,
                    messages=[{"role": "user", "content": "ping"}]
                )
                results["Mistral"] = "valid"
            else:
                results["Mistral"] = "invalid"
        except Exception as e:
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
                    "model": "claude-haiku-4-5",
                    "max_tokens": 8192,
                    "system": "",
                    "messages": [{"role": "user", "content": "ping"}]
                }
                response = requests.post(url, json=payload, headers=headers)
                if response.status_code == 200:
                    results["Anthropic"] = "valid"
                else:
                    results["Anthropic"] = "invalid"
            else:
                results["Anthropic"] = "invalid"
        except Exception as e:
            results["Anthropic"] = "invalid"

        # DeepSeek Handshake
        try:
            deepseek_key = data.get("deepseek_key")
            if deepseek_key and len(deepseek_key) > 10:
                client = openai.OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com")
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "ping"},
                        {"role": "user", "content": "ping"}
                    ],
                    stream=False
                )
                results["DeepSeek"] = "valid"
            else:
                results["DeepSeek"] = "invalid"
        except Exception as e:
            results["DeepSeek"] = "invalid"

        # Grok Handshake
        try:
            grok_key = data.get("grok_key")
            if grok_key and len(grok_key) > 10:
                client = openai.OpenAI(api_key=grok_key, base_url="https://api.x.ai/v1")
                response = client.chat.completions.create(
                    model="grok-3-latest",
                    messages=[
                        {"role": "system", "content": "ping"},
                        {"role": "user", "content": "ping"}
                    ]
                )
                results["Grok"] = "valid"
            else:
                results["Grok"] = "invalid"
        except Exception as e:
            results["Grok"] = "invalid"
        
        # Gemini Handshake
        try:
            gemini_key = data.get("gemini_key")
            if gemini_key and len(gemini_key) > 10:
                genai.configure(api_key=gemini_key)
                model = genai.GenerativeModel("gemini-1.5-pro-latest")
                _ = model.generate_content("ping")
                results["Gemini"] = "valid"
            else:
                results["Gemini"] = "invalid"
        except Exception as e:
            results["Gemini"] = "invalid"

        # Exa Handshake
        try:
            exa_key = data.get("exa_key")
            if exa_key and len(exa_key) > 10:
                client = openai.OpenAI(api_key=exa_key, base_url="https://api.exa.ai")
                response = client.chat.completions.create(
                    model="exa",  # Alternativ "exa-pro" falls gewünscht
                    messages=[
                        {"role": "system", "content": "ping"},
                        {"role": "user", "content": "ping"}
                    ],
                    extra_body={"text": True}
                )
                results["Exa"] = "valid"
            else:
                results["Exa"] = "invalid"
        except Exception as e:
            results["Exa"] = "invalid"
        
        return {"results": results}

    except Exception as overall_error:
        return {"results": {"error": str(overall_error)}}