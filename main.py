import os
from fastapi import FastAPI, Query, Request, HTTPException, Body
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import openai
import requests
import base64, re
import time, logging
from mistralai import Mistral
from dotenv import load_dotenv
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, auth, firestore
from typing import Optional
from datetime import datetime, timedelta
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
                    "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://www.gstatic.com; "
                    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                    "img-src 'self' data:; "
                    "connect-src 'self' https://firestore.googleapis.com https://*.firebaseio.com https://identitytoolkit.googleapis.com https://securetoken.googleapis.com https://apis.google.com https://api.openai.com https://api.mistral.ai https://api.anthropic.com https://api.x.ai https://api.deepseek.com https://api.perplexity.ai https://api.exa.ai;"
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

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

FREE_USAGE_LIMIT = 25
MAX_WORDS = 300
DEEP_SEARCH_MAX_WORDS = 1000
MAX_TOKENS = 1024
DEEP_SEARCH_MAX_TOKENS = 2048
CONSENSUS_MAX_TOKENS = 2048

DEFAULT_SYSTEM_PROMPT = "Please respond briefly and precisely, focusing only on the essentials."
DEEP_THINK_PROMPT = "Deep Think: Please provide a deep, detailed analysis. Focus as hard as you can!"

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

def query_openai(question: str, api_key: str, search_mode: bool = False, deep_search: bool = False, system_prompt: str = None) -> str:
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    if deep_search:
        system_prompt += "\n" + DEEP_THINK_PROMPT

    max_tokens = DEEP_SEARCH_MAX_TOKENS if deep_search else MAX_TOKENS

    try:
        client = openai.OpenAI(api_key=api_key)
        # Wähle das Modell abhängig von search_mode und deep_search:
        if search_mode:
            model_to_use = "gpt-4o-search-preview"
        elif deep_search:
            model_to_use = "o3-mini"
        else:
            model_to_use = "gpt-4.1"

        # Verwende den korrekten Parameter basierend auf dem Modell
        if model_to_use == "o3-mini":
            response = client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                max_completion_tokens=max_tokens
            )
        else:
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
        return f"Error with OpenAI: {str(e)}"

def query_mistral(question: str, api_key: str, system_prompt: str = None, deep_search: bool = False) -> str:
    """Fragt die Mistral API zu der gegebenen Frage unter Verwendung des übergebenen API Keys ohne Limit."""
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    if deep_search:
        system_prompt += "\n" + DEEP_THINK_PROMPT

    # Setze max_tokens basierend auf dem deep_search Flag
    max_tokens = DEEP_SEARCH_MAX_TOKENS if deep_search else MAX_TOKENS

    try:
        client = Mistral(api_key=api_key)
        model = "mistral-large-latest"
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


def query_claude(question: str, api_key: str, system_prompt: str = None, deep_search: bool = False) -> str:
    """Fragt die Anthropic API (Claude) zu der gegebenen Frage unter Verwendung des übergebenen API Keys ohne Limit.
       Da die Anthropic API ein Token-Limit erwartet, setzen wir einen sehr hohen Wert ein."""
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT

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
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": question}]
        }
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if "content" in data and isinstance(data["content"], list) and len(data["content"]) > 0:
                return data["content"][0]["text"]
            else:
                return "Error: No response found in the API response."
        else:
            return f"Error with Anthropice: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error with Anthropic: {str(e)}"

def query_gemini(question: str, user_api_key: Optional[str] = None, search_mode: bool = False, deep_search: bool = False, system_prompt: str = None) -> str:
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    if deep_search:
        system_prompt += "\n" + DEEP_THINK_PROMPT

    try:
        # Wenn ein eigener API Key übergeben wurde, verwende ihn.
        # Andernfalls – bei eingeloggten Nutzern – wird der Service-Account-Key aus der JSON genutzt.
        if user_api_key and user_api_key.strip():
            genai.configure(api_key=user_api_key)
        else:
            genai.configure()
        
        # Je nach Search Mode den passenden Modellnamen wählen:
        model_name = "gemini-1.5-pro-002" if search_mode else "gemini-1.5-pro-latest"
        model = genai.GenerativeModel(model_name)
        
        base_content = "Do not ask any questions.\n" + system_prompt + "\n---\n" + question
        if search_mode:
            # Hinweis im Prompt, der auch die Links (über Retrieval) liefern soll
            base_content += "\nPlease include clickable links to the sources used at the end of your answer."
            response = model.generate_content(
                contents=base_content,
                tools={"google_search_retrieval": {"dynamic_retrieval_config": {"mode": "MODE_DYNAMIC", "dynamic_threshold": 0.5}}}
            )
        else:
            response = model.generate_content(base_content)
        
        answer_text = response.text.strip()
        # Im Search Mode: Falls Grounding‑Metadata vorhanden, extrahiere die Links
        if search_mode and hasattr(response.candidates[0], "grounding_metadata") and response.candidates[0].grounding_metadata:
            grounding = response.candidates[0].grounding_metadata
            if hasattr(grounding, "grounding_chunks") and grounding.grounding_chunks:
                formatted_links = [f"[{chunk.web.title}]({chunk.web.uri})" for chunk in grounding.grounding_chunks]
                answer_text += "\n\n" + "\n".join(formatted_links)
        return answer_text
    except Exception as e:
        return f"Error with Gemini: {str(e)}"

    
def query_deepseek(question: str, api_key: str, system_prompt: str = None, deep_search: bool = False) -> str:
    """Fragt DeepSeek zu der gegebenen Frage unter Verwendung des übergebenen API Keys."""
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    if deep_search:
        system_prompt += "\n" + DEEP_THINK_PROMPT

    # Setze max_tokens basierend auf dem deep_search Flag
    max_tokens = DEEP_SEARCH_MAX_TOKENS if deep_search else MAX_TOKENS

    try:
        client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        # Wähle das Modell: Falls deep_search aktiviert ist, nutze "deepseek-reasoner", sonst "deepseek-chat"
        model_to_use = "deepseek-reasoner" if deep_search else "deepseek-chat"
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
    
def query_grok(question: str, api_key: str, system_prompt: str = None, deep_search: bool = False) -> str:
    """Fragt die Grok API zu der gegebenen Frage unter Verwendung des übergebenen API Keys."""
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    if deep_search:
        system_prompt += "\n" + DEEP_THINK_PROMPT

    # Setze max_tokens basierend auf dem deep_search Flag
    max_tokens = DEEP_SEARCH_MAX_TOKENS if deep_search else MAX_TOKENS

    try:
        client = openai.OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        response = client.chat.completions.create(
            model="grok-2-latest",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error with Grok: {str(e)}"
    
    
def query_exa(question: str, api_key: str, search_mode: bool, system_prompt: str = None, deep_search: bool = False) -> str:
    if not search_mode:
        # Falls der Search Mode nicht aktiv ist, wird Exa nicht angefragt.
        return ""
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    if deep_search:
        system_prompt += "\n" + DEEP_THINK_PROMPT
    
    # Setze max_tokens basierend auf dem deep_search Flag
    max_tokens = DEEP_SEARCH_MAX_TOKENS if deep_search else MAX_TOKENS

    # Wähle das Model abhängig vom deep_search Flag
    model_to_use = "exa-pro" if deep_search else "exa"

    try:
        client = openai.OpenAI(api_key=api_key, base_url="https://api.exa.ai")
        response = client.chat.completions.create(
            model=model_to_use,  # Alternativ: "exa-pro", falls gewünscht
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            extra_body={"text": True},  # Stellt sicher, dass als Text geantwortet wird
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error with Exa: {str(e)}"
    

def query_perplexity(question: str, api_key: str, search_mode: bool, system_prompt: str = None, deep_search: bool = False) -> str:
    # Falls der Search Mode nicht aktiv ist, wird Perplexity nicht angefragt.
    if not search_mode:
        return ""
    if system_prompt is None:
        system_prompt = (
            "You are an artificial intelligence assistant and you need to "
            "Answer shortly."
        )

    if deep_search:
        system_prompt += "\n" + DEEP_THINK_PROMPT
    
    # Setze max_tokens basierend auf dem deep_search Flag
    max_tokens = DEEP_SEARCH_MAX_TOKENS if deep_search else MAX_TOKENS

    # Wähle das Model abhängig vom deep_search Flag
    model_to_use = "sonar-pro" if deep_search else "sonar"

    try:
        # Erstelle den Client mit der Perplexity-URL
        client = openai.OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        response = client.chat.completions.create(
            model=model_to_use,
            messages=messages,
            max_tokens=max_tokens,
        )
        
        # Extrahiere den Antworttext und die citations (sofern vorhanden)
        result_text = response.choices[0].message.content.strip()
        citations = getattr(response, "citations", [])
        
        # Ersetze Zitations-Marker [1], [2], ... durch HTML-Links
        def replace_citation(match):
            # Die Ziffern in den Klammern sind 1-indexiert, während die Liste 0-indexiert ist.
            index = int(match.group(1)) - 1
            if 0 <= index < len(citations):
                url = citations[index]
                return f'<a href="{url}" target="_blank">[{match.group(1)}]</a>'
            return match.group(0)
        
        processed_text = re.sub(r'\[([0-9]+)\]', replace_citation, result_text)
        return processed_text
    except Exception as e:
        return f"Error with Perplexity: {str(e)}"


def query_consensus(question: str, answer_openai: str, answer_mistral: str, answer_claude: str, 
                    answer_gemini: str, answer_deepseek: str, answer_grok: str, answer_exa: str, answer_perplexity: str,
                    best_model: str, excluded_models: list, consensus_model: str, 
                    api_keys: dict, search_mode: bool = False) -> str:
    """
    Uses a model to consolidate responses from multiple models into a consistent consensus answer using chain-of-thought logic.
    The API keys provided are retrieved from the 'api_keys' dictionary.
    """
    prompt_parts = []
    if search_mode:
        prompt_parts.append("Note: The following responses are based on current real-time data. Please include links from the responses in your final consensus answer.\n\n")
    prompt_parts.append(f"Please provide your answer in the same language as the user's question. The question is: {question}\n\n")
    if "OpenAI" not in excluded_models and answer_openai:
        prompt_parts.append(f"Response from GPT-4o: {answer_openai}\n\n")
    if "Mistral" not in excluded_models and answer_mistral:
        prompt_parts.append(f"Response from mistral-large-latest: {answer_mistral}\n\n")
    if "Anthropic" not in excluded_models and answer_claude:
        prompt_parts.append(f"Response from claude-3-5-sonnet: {answer_claude}\n\n")
    if "Gemini" not in excluded_models and answer_gemini:
        prompt_parts.append(f"Response from gemini-pro: {answer_gemini}\n\n")
    if "DeepSeek" not in excluded_models and answer_deepseek:
        prompt_parts.append(f"Response from deepseek-chat: {answer_deepseek}\n\n")
    if "Grok" not in excluded_models and answer_grok:
        prompt_parts.append(f"Response from Grok: {answer_grok}\n\n")
    if "Exa" not in excluded_models and answer_exa:
        prompt_parts.append(f"Response from Exa: {answer_exa}\n\n")
    if "Perplexity" not in excluded_models and answer_perplexity:
        prompt_parts.append(f"Response from Perplexity: {answer_perplexity}\n\n")

    if best_model:
        prompt_parts.append(
            f"The user marked the Answer from the Model: {best_model} as the best one. "
            "You receive four expert opinions on a specific question. "
            "Your task is to combine these responses into a comprehensive, correct, and coherent answer. "
            "Note: Experts can also make mistakes. Therefore, try to identify and exclude possible errors by comparing the answers. "
            "If the answers strongly contradict each other at any point, logically determine which variant is most plausible. "
            "Structure the answer clearly and coherently. "
            "Provide only the final, balanced answer. Adapt the "
        )
    else:
        prompt_parts.append(
            "You receive four expert opinions on a specific question. "
            "Treat all expert opinions equally. Do not focus on the answer of one model. "
            "Your task is to combine these responses into a comprehensive, correct, and coherent answer. "
            "Note: Experts can also make mistakes. Therefore, try to identify and exclude possible errors by comparing the answers. "
            "If the answers strongly contradict each other at any point, logically determine which variant is most plausible. "
            "Structure the answer clearly and coherently. "
            "Provide only the final, balanced answer."
        )

    consensus_prompt = "".join(prompt_parts)
    
    try:
        if consensus_model == "OpenAI":
            client = openai.OpenAI(api_key=api_keys.get("OpenAI"))
            model_to_use = "gpt-4.1" if search_mode else "gpt-4.1"
            response = client.chat.completions.create(
                        model=model_to_use,
                        messages=[
                            {"role": "system", "content": ""},
                            {"role": "user", "content": consensus_prompt}
                        ],
                        max_tokens=CONSENSUS_MAX_TOKENS
                    )
            return response.choices[0].message.content.strip()
        
        elif consensus_model == "Mistral":
            client = Mistral(api_key=api_keys.get("Mistral"))
            model = "mistral-large-latest"
            response = client.chat.complete(
                model=model,
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
                "model": "claude-3-5-sonnet-20241022",
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
                genai.configure()  # Service-Account-Modus
            model = genai.GenerativeModel("gemini-1.5-pro-latest")
            # Schalte auch hier zwischen Search und normalem Modus um:
            if search_mode:
                response = model.generate_content(
                    consensus_prompt,
                    max_tokens=CONSENSUS_MAX_TOKENS,
                    tools={"google_search_retrieval": {"dynamic_retrieval_config": {"mode": "MODE_DYNAMIC", "dynamic_threshold": 0.5}}}
                )
            else:
                response = model.generate_content(consensus_prompt)
            return response.text.strip()

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
                model="grok-2-latest",
                messages=[
                    {"role": "system", "content": " "},
                    {"role": "user", "content": consensus_prompt}
                ],
                max_tokens=CONSENSUS_MAX_TOKENS
            )
            return response.choices[0].message.content.strip()
        
        elif consensus_model == "Exa":
            client = openai.OpenAI(api_key=api_keys.get("Exa"), base_url="https://api.exa.ai")
            response = client.chat.completions.create(
                model="exa",  # Alternativ "exa-pro", falls gewünscht
                messages=[
                    {"role": "system", "content": ""},
                    {"role": "user", "content": consensus_prompt}
                ],
                extra_body={"text": True},
                max_tokens=CONSENSUS_MAX_TOKENS
            )
            return response.choices[0].message.content.strip()
        
        elif consensus_model == "Perplexity":
            client = openai.OpenAI(api_key=api_keys.get("Perplexity"), base_url="https://api.perplexity.ai")
            response = client.chat.completions.create(
                model="sonar",
                messages=[
                    {"role": "system", "content": ""},
                    {"role": "user", "content": consensus_prompt}
                ],
                max_tokens=CONSENSUS_MAX_TOKENS
            )
            return response.choices[0].message.content.strip()

        else:
            return "Invalid consensus model selected."
    except Exception as e:
        return f"Consensus error: {str(e)}"
    

def query_differences(answer_openai: str, answer_mistral: str, answer_claude: str, answer_gemini: str, answer_deepseek: str, answer_grok: str, answer_exa: str, answer_perplexity: str, consensus_answer: str, api_keys: dict, differences_model: str, search_mode: bool = False) -> str:
    """
    Extrahiert die Unterschiede zwischen den vier Expertenantworten mittels des angegebenen Konsens‑Modells.
    """
    # Je nach Modus nur die relevanten Antworten in den Prompt aufnehmen:
    if search_mode in [True, "true", "True"]:
        responses_text = (
            f"- GPT-4o: {answer_openai}\n"
            f"- Gemini: {answer_gemini}\n"
            f"- Exa: {answer_exa}\n"
            f"- Perplexity: {answer_perplexity}\n"
        )
        best_models_instruction = "Choose from one of the following models: OpenAI or Gemini."
    else:
        responses_text = (
            f"- GPT-4o: {answer_openai}\n"
            f"- Mistral: {answer_mistral}\n"
            f"- Claude: {answer_claude}\n"
            f"- Gemini: {answer_gemini}\n"
            f"- DeepSeek: {answer_deepseek}\n"
            f"- Grok: {answer_grok}\n"
        )
        best_models_instruction = "Choose from one of the following models: Anthropic, Gemini, Mistral, or OpenAI."

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

        "Model responses:\n" + responses_text + "\n"

        "Finally, subjectively determine which model provided the best answer. " + best_models_instruction + " "
        "Include your decision at the end of the response on a separate line, starting with 'BestModel:' followed by the model name.\n\n"

        "Response format:\n"
        "[Credibility statement]\n"
        "\n"
        "_____________\n"
        "\n"
        "[Very brief explanation of why these differences affect credibility.]\n\n"
        "(Info: Mark the model closest to the consensus as Best Model)"
        "BestModel: [Model name]"
    )

    try:
        if differences_model == "OpenAI":
            client = openai.OpenAI(api_key=api_keys.get("OpenAI"))
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "Answer in the exact same Langugage as the Model responses: "},
                    {"role": "user", "content": differences_prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        elif differences_model == "Mistral":
            client = Mistral(api_keys.get("Mistral"))
            response = client.chat.complete(
                model="mistral-large-latest",
                messages=[
                    {"role": "system", "content": "Answer in the exact same Langugage as the Model responses:  "},
                    {"role": "user", "content": differences_prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        elif differences_model == "Anthropic":
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": api_keys.get("Anthropic"),
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            payload = {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 8192,
                "system": "Answer in the exact same Langugage as the Model responses:  ",
                "messages": [{"role": "user", "content": differences_prompt}]
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
            
        elif differences_model == "Gemini":
            gemini_key = api_keys.get("Gemini")
            if gemini_key and gemini_key.strip() != "":
                genai.configure(api_key=gemini_key)
            else:
                genai.configure()
            model = genai.GenerativeModel("gemini-1.5-pro-latest")
            response = model.generate_content(differences_prompt)
            return response.text.strip()
        
        elif differences_model == "DeepSeek":
            client = openai.OpenAI(api_key=api_keys.get("DeepSeek"), base_url="https://api.deepseek.com")
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "Answer in the exact same Langugage as the Model responses:  "},
                    {"role": "user", "content": differences_prompt}
                ],
                stream=False
            )
            return response.choices[0].message.content.strip()
        
        elif differences_model == "Grok":
            client = openai.OpenAI(api_key=api_keys.get("Grok"), base_url="https://api.x.ai/v1")
            response = client.chat.completions.create(
                model="grok-2-latest",
                messages=[
                    {"role": "system", "content": "Answer in the exact same Langugage as the Model responses: "},
                    {"role": "user", "content": differences_prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        
        elif differences_model == "Exa":
            client = openai.OpenAI(api_key=api_keys.get("Exa"), base_url="https://api.exa.ai")
            response = client.chat.completions.create(
                model="exa",  # oder "exa-pro" falls gewünscht
                messages=[
                    {"role": "system", "content": "Answer in the exact same language as the Model responses:"},
                    {"role": "user", "content": differences_prompt}
                ],
                extra_body={"text": True},
                max_tokens=CONSENSUS_MAX_TOKENS
            )
            return response.choices[0].message.content.strip()
        
        elif differences_model == "Perplexity":
            client = openai.OpenAI(api_key=api_keys.get("Perplexity"), base_url="https://api.perplexity.ai")
            response = client.chat.completions.create(
                model="sonar",
                messages=[
                    {"role": "system", "content": "Answer in the exact same language as the Model responses:"},
                    {"role": "user", "content": differences_prompt}
                ],
                max_tokens=CONSENSUS_MAX_TOKENS
            )
            return response.choices[0].message.content.strip()

        else:
            return "Invalid model selected for difference comparison."
    except Exception as e:
        return f"Error in comparison: {str(e)}"
    
# Initialisiere Firebase Admin (Beispiel, passe den Pfad zu deinem Service Account an)
cred = credentials.Certificate("consensai-firebase-adminsdk-fbsvc-9064a77134.json")
firebase_admin.initialize_app(cred)

# Erstelle einen Firestore-Client
db_firestore = firestore.client()

def verify_user_token(token: str) -> str:
    """
    Verifiziert das Firebase-ID-Token mit etwas Clock-Skew-Toleranz
    und gibt die uid zurück.
    """
    try:
        # Erlaube z.B. bis zu 5 Sekunden Drift
        decoded_token = auth.verify_id_token(token, clock_skew_seconds=5)
        return decoded_token["uid"]
    except Exception as e:
        # Logge den Originalfehler inkl. Uhrzeit zum Debuggen
        import time, logging
        now = int(time.time())
        logging.error(f"verify_user_token failed (server time={now}): {e}")
        raise Exception("Invalid token: " + str(e))


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

@app.get("/bookmarks")
@limiter.limit("20/minute")  # Beispiel: maximal 10 Anfragen pro Minute pro IP
async def load_bookmarks(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    
    id_token = auth_header.split(" ")[1]
    try:
        uid = verify_user_token(id_token)
    except Exception as e:
        raise HTTPException(status_code=401, detail="Authentication failed: " + str(e))
    
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
        raise HTTPException(status_code=500, detail="Error loading bookmarks: " + str(e))

# Globales Dictionary zum Speichern der IP-Adressen registrierter Nutzer
registered_ips = {}  # { ip_address: uid }

@app.post("/register")
@limiter.limit("1/minute")  # Beispiel: maximal 10 Registrierungen pro Minute pro IP
async def register_user(request: Request, data: dict = Body(...)):
    ip_address = request.client.host
    # Prüfe, ob diese IP-Adresse bereits einen Account registriert hat
    if ip_address in registered_ips:
        raise HTTPException(status_code=400, detail="Only one account per user is allowed.")
    
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

        # Erstelle den Nutzer über Firebase Admin
        user = auth.create_user(email=email, password=password)
        # Erzeuge ein Custom Token für den neuen Nutzer
        custom_token = auth.create_custom_token(user.uid)
        # Das Token ist ein Bytes-Objekt – in einen String konvertieren
        custom_token_str = custom_token.decode("utf-8")
        return {"uid": user.uid, "email": user.email, "customToken": custom_token_str}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

@app.post("/confirm-registration")
async def confirm_registration(request: Request, data: dict = Body(...)):
    """
    Einmaliger Aufruf beim ersten Login nach erfolgreicher E‑Mail‑Bestätigung.
    Trägt die IP in registered_ips ein, wenn emailVerified == True.
    """
    token = data.get("id_token")
    if not token:
        raise HTTPException(status_code=400, detail="id_token fehlt.")

    try:
        uid = verify_user_token(token)
        user = auth.get_user(uid)
    except Exception as e:
        raise HTTPException(status_code=401, detail="Token ungültig: " + str(e))

    if not user.email_verified:
        raise HTTPException(status_code=400, detail="E‑Mail noch nicht bestätigt.")

    ip_address = request.client.host
    if ip_address in registered_ips:
        # falls schon eingetragen, nichts tun
        return {"status": "already_registered"}

    registered_ips[ip_address] = uid
    return {"status": "registered", "ip": ip_address}

    
@app.post("/usage")
@limiter.limit("20/minute")  # Beispiel: maximal 20 Anfragen pro Minute pro IP
async def get_usage_post(request: Request):
    # JSON-Daten aus dem Request abrufen
    data = await request.json()
    
    token = data.get("id_token")
    try:
        uid = verify_user_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Authentication failed")
    
    current_usage = usage_counter.get(uid, 0)
    remaining = FREE_USAGE_LIMIT - current_usage

    current_deep_usage = deep_search_usage.get(uid, 0)
    deep_remaining = 12 - current_deep_usage  # Deep Search Limit: 12

    return {"remaining": remaining, "deep_remaining": deep_remaining}
    
# Globales Dictionary zum Speichern des letzten Feedback-Zeitstempels pro Nutzer
last_feedback_time = {}
    
@app.post("/feedback")
@limiter.limit("3/minute")
async def submit_feedback(request: Request, data: dict = Body(...)):
    message = data.get("message")
    email = data.get("email")
    id_token = data.get("id_token")
    
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
    # Extrahiere Parameter
    id_token = data.get("id_token")
    model = data.get("model")
    vote_type = data.get("vote_type")
    
    # Parameter-Validierung
    if not id_token or not model or not vote_type:
        raise HTTPException(status_code=400, detail="Missing required fields: id_token, model or vote_type.")
    
    if vote_type not in ALLOWED_VOTE_TYPES:
        raise HTTPException(status_code=400, detail="Invalid vote type provided.")

    # Authentifizierung
    try:
        uid = verify_user_token(id_token)
    except Exception as e:
        raise HTTPException(status_code=401, detail="Authentication failed: " + str(e))
    
    # Optional: Hier könnte ein Rate-Limiting-Check ergänzt werden, um Mehrfachvotings in kurzer Zeit zu unterbinden.
    
    # Aktualisiere den Vote in Firestore (Leaderboard)
    try:
        doc_ref = db_firestore.collection("leaderboard").document(model)
        # Verwende ein atomic update (Increment) statt eines direkten setDoc-Aufrufs
        doc_ref.update({
            vote_type: firestore.Increment(1)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error updating vote: " + str(e))
    
    return {"status": "success", "message": f"{vote_type} vote recorded for {model}"}

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
    id_token = data.get("id_token")
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

    
@app.post("/ask_openai")
@limiter.limit("5/minute")  # Beispiel: maximal 20 Anfragen pro Minute pro IP
async def ask_openai_post(request: Request, data: dict = Body(...)):
    question = data.get("question")
    # Konvertiere deep_search vor der Wortzählung
    deep_search_raw = data.get("deep_search", False)
    deep_search = True if str(deep_search_raw).lower() == "true" else False
    # Wähle das passende Wortlimit basierend auf deep_search
    max_words_limit = DEEP_SEARCH_MAX_WORDS if deep_search else MAX_WORDS

    if count_words(question) > max_words_limit:
        raise HTTPException(
            status_code=400,
            detail=f"Die Eingabe überschreitet das erlaubte Wortlimit von {max_words_limit} Wörtern."
        )
    system_prompt = data.get("system_prompt")

    id_token = data.get("id_token")
    api_key = data.get("api_key")
    # Lese den Status der Toggle-Switches aus
    search_mode = data.get("search_mode", False)
    active_count = data.get("active_count", 1)
    
    if id_token:
        try:
            uid = verify_user_token(id_token)
        except Exception:
            raise HTTPException(status_code=401, detail="Authentication failed")
        increment = 1.0 / active_count

        # --- Zuerst beide Quota-Checks durchführen ---
        current_usage = usage_counter.get(uid, 0)
        if current_usage + increment > FREE_USAGE_LIMIT:
            # Auch im Fehlerfall die aktuellen Zählerwerte zurückgeben:
            deep_remaining = 12 - deep_search_usage.get(uid, 0)
            return {
                "error": "Your free quota is exhausted. Please provide your own API keys.",
                "free_usage_remaining": 0,
                "deep_remaining": deep_remaining
            }

        if deep_search:
            current_deep_usage = deep_search_usage.get(uid, 0)
            if current_deep_usage + increment > 12:
                free_remaining = FREE_USAGE_LIMIT - current_usage
                # Hier werden beide Werte mitgegeben, auch wenn deep search nicht verfügbar ist:
                return {
                    "error": "Your free deep search quota is exhausted. Please store your own API keys.",
                    "free_usage_remaining": free_remaining,
                    "deep_remaining": 0
                }

        # --- Jetzt beide Zähler inkrementieren, da beide Checks bestanden sind ---
        usage_counter[uid] = current_usage + increment
        if deep_search:
            deep_search_usage[uid] = current_deep_usage + increment

        developer_api_key = os.environ.get("DEVELOPER_OPENAI_API_KEY")
        if not developer_api_key:
            raise HTTPException(status_code=500, detail="Server error: API key not configured")
        answer = query_openai(
            question,
            developer_api_key,
            search_mode=search_mode,
            deep_search=deep_search,
            system_prompt=system_prompt
        )
        free_remaining = FREE_USAGE_LIMIT - usage_counter[uid]
        deep_remaining = 12 - deep_search_usage.get(uid, 0)
        return {"response": answer, "free_usage_remaining": free_remaining, "deep_remaining": deep_remaining, "key_used": "Developer API Key"}
    elif api_key:
        answer = query_openai(
            question,
            api_key,
            search_mode=search_mode,
            deep_search=deep_search,
            system_prompt=system_prompt
        )
        # Auch hier beide Werte immer mitgeben
        return {
            "response": answer,
            "free_usage_remaining": FREE_USAGE_LIMIT,
            "deep_remaining": 12,
            "key_used": "User API Key"
        }
    else:
        raise HTTPException(status_code=400, detail="No authentication parameter (id_token or api_key) specified")

# Angepasster Endpoint für Mistral
@app.post("/ask_mistral")
@limiter.limit("5/minute")
async def ask_mistral_post(request: Request, data: dict = Body(...)):
    question = data.get("question")
    # Konvertiere deep_search in einen Boolean
    deep_search_raw = data.get("deep_search", False)
    deep_search = True if str(deep_search_raw).lower() == "true" else False
    # Wähle das passende Wortlimit: DEEP_SEARCH_MAX_WORDS, wenn deep_search aktiv ist, sonst MAX_WORDS
    max_words_limit = DEEP_SEARCH_MAX_WORDS if deep_search else MAX_WORDS

    if count_words(question) > max_words_limit:
        raise HTTPException(
            status_code=400,
            detail=f"Die Eingabe überschreitet das erlaubte Wortlimit von {max_words_limit} Wörtern."
        )
    system_prompt = data.get("system_prompt")
    id_token = data.get("id_token")
    api_key = data.get("api_key")
    active_count = data.get("active_count", 1)
    
    if id_token:
        try:
            uid = verify_user_token(id_token)
        except Exception:
            raise HTTPException(status_code=401, detail="Authentication failed")

        increment = 1.0 / active_count

        # --- Zuerst beide Quota-Checks durchführen ---
        current_usage = usage_counter.get(uid, 0)
        if current_usage + increment > FREE_USAGE_LIMIT:
            # Auch im Fehlerfall die aktuellen Zählerwerte zurückgeben:
            deep_remaining = 12 - deep_search_usage.get(uid, 0)
            return {
                "error": "Your free quota is exhausted. Please provide your own API keys.",
                "free_usage_remaining": 0,
                "deep_remaining": deep_remaining
            }

        if deep_search:
            current_deep_usage = deep_search_usage.get(uid, 0)
            if current_deep_usage + increment > 12:
                free_remaining = FREE_USAGE_LIMIT - current_usage
                # Hier werden beide Werte mitgegeben, auch wenn deep search nicht verfügbar ist:
                return {
                    "error": "Your free deep search quota is exhausted. Please store your own API keys.",
                    "free_usage_remaining": free_remaining,
                    "deep_remaining": 0
                }

        # --- Jetzt beide Zähler inkrementieren, da beide Checks bestanden sind ---
        usage_counter[uid] = current_usage + increment
        if deep_search:
            deep_search_usage[uid] = current_deep_usage + increment

        developer_api_key = os.environ.get("DEVELOPER_MISTRAL_API_KEY")
        if not developer_api_key:
            raise HTTPException(status_code=500, detail="Server error: API key not configured")
        answer = query_mistral(question, developer_api_key, system_prompt, deep_search=deep_search)
        free_remaining = FREE_USAGE_LIMIT - usage_counter[uid]
        deep_remaining = 12 - deep_search_usage.get(uid, 0)
        return {"response": answer, "free_usage_remaining": free_remaining, "deep_remaining": deep_remaining, "key_used": "Developer API Key"}
    elif api_key:
        answer = query_mistral(question, api_key, system_prompt, deep_search=deep_search)
        # Auch hier beide Werte immer mitgeben
        return {
            "response": answer,
            "free_usage_remaining": FREE_USAGE_LIMIT,
            "deep_remaining": 12,
            "key_used": "User API Key"
        }
    else:
        raise HTTPException(status_code=400, detail="No id_token or api_key specified.")


# Angepasster Endpoint für Anthropic
@app.post("/ask_claude")
@limiter.limit("3/minute")
async def ask_claude_post(request: Request, data: dict = Body(...)):
    question = data.get("question")
    # Konvertiere deep_search in einen Boolean
    deep_search_raw = data.get("deep_search", False)
    deep_search = True if str(deep_search_raw).lower() == "true" else False
    # Wähle das entsprechende Wortlimit
    max_words_limit = DEEP_SEARCH_MAX_WORDS if deep_search else MAX_WORDS

    if count_words(question) > max_words_limit:
        raise HTTPException(
            status_code=400,
            detail=f"Die Eingabe überschreitet das erlaubte Wortlimit von {max_words_limit} Wörtern."
        )
    system_prompt = data.get("system_prompt")
    id_token = data.get("id_token")
    api_key = data.get("api_key")
    active_count = data.get("active_count", 1)
    
    if id_token:
        try:
            uid = verify_user_token(id_token)
        except Exception:
            raise HTTPException(status_code=401, detail="Authentication failed")

        increment = 1.0 / active_count

        # --- Zuerst beide Quota-Checks durchführen ---
        current_usage = usage_counter.get(uid, 0)
        if current_usage + increment > FREE_USAGE_LIMIT:
            # Auch im Fehlerfall die aktuellen Zählerwerte zurückgeben:
            deep_remaining = 12 - deep_search_usage.get(uid, 0)
            return {
                "error": "Your free quota is exhausted. Please provide your own API keys.",
                "free_usage_remaining": 0,
                "deep_remaining": deep_remaining
            }

        if deep_search:
            current_deep_usage = deep_search_usage.get(uid, 0)
            if current_deep_usage + increment > 12:
                free_remaining = FREE_USAGE_LIMIT - current_usage
                # Hier werden beide Werte mitgegeben, auch wenn deep search nicht verfügbar ist:
                return {
                    "error": "Your free deep search quota is exhausted. Please store your own API keys.",
                    "free_usage_remaining": free_remaining,
                    "deep_remaining": 0
                }

        # --- Jetzt beide Zähler inkrementieren, da beide Checks bestanden sind ---
        usage_counter[uid] = current_usage + increment
        if deep_search:
            deep_search_usage[uid] = current_deep_usage + increment

        developer_api_key = os.environ.get("DEVELOPER_ANTHROPIC_API_KEY")
        if not developer_api_key:
            raise HTTPException(status_code=500, detail="Server error: API key not configured")
        answer = query_claude(
            question,
            developer_api_key,
            system_prompt,
            deep_search=deep_search
        )
        free_remaining = FREE_USAGE_LIMIT - usage_counter[uid]
        deep_remaining = 12 - deep_search_usage.get(uid, 0)
        return {"response": answer, "free_usage_remaining": free_remaining, "deep_remaining": deep_remaining, "key_used": "Developer API Key"}
    elif api_key:
        answer = query_claude(
            question,
            api_key,
            system_prompt,
            deep_search=deep_search
        )
        # Auch hier beide Werte immer mitgeben
        return {
            "response": answer,
            "free_usage_remaining": FREE_USAGE_LIMIT,
            "deep_remaining": 12,
            "key_used": "User API Key"
        }
    else:
        raise HTTPException(status_code=400, detail="No id_token or api_key specified.")

@app.post("/ask_gemini")
@limiter.limit("3/minute")
async def ask_gemini_post(request: Request, data: dict = Body(...)):
    question = data.get("question")
    # Konvertiere deep_search in einen Boolean
    deep_search_raw = data.get("deep_search", False)
    deep_search = True if str(deep_search_raw).lower() == "true" else False
    # Wähle das entsprechende Wortlimit
    max_words_limit = DEEP_SEARCH_MAX_WORDS if deep_search else MAX_WORDS

    if count_words(question) > max_words_limit:
        raise HTTPException(
            status_code=400,
            detail=f"Die Eingabe überschreitet das erlaubte Wortlimit von {max_words_limit} Wörtern."
        )
    system_prompt = data.get("system_prompt")
    use_own_keys = data.get("useOwnKeys", False)
    if isinstance(use_own_keys, str):
        use_own_keys = use_own_keys.lower() == "true"
    id_token = data.get("id_token")
    api_key = data.get("api_key")  # von der Sidebar
    active_count = data.get("active_count", 1)
    search_mode = data.get("search_mode", False)
    
    if id_token:
        try:
            uid = verify_user_token(id_token)
        except Exception:
            raise HTTPException(status_code=401, detail="Authentication failed")
        
        increment = 1.0 / active_count

        # Zuerst: Prüfe, ob genügend allgemeine Free Requests vorhanden sind
        current_usage = usage_counter.get(uid, 0)
        if current_usage + increment > FREE_USAGE_LIMIT:
            # Auch im Fehlerfall die aktuellen Zählerwerte zurückgeben:
            deep_remaining = 12 - deep_search_usage.get(uid, 0)
            return {
                "error": "Your free quota is exhausted. Please provide your own API keys.",
                "free_usage_remaining": 0,
                "deep_remaining": deep_remaining
            }
        
        # Falls Deep Search angefordert wurde, auch den Deep Search Quota prüfen
        if deep_search:
            current_deep_usage = deep_search_usage.get(uid, 0)
            if current_deep_usage + increment > 12:
                free_remaining = FREE_USAGE_LIMIT - current_usage
                # Hier werden beide Werte mitgegeben, auch wenn deep search nicht verfügbar ist:
                return {
                    "error": "Your free deep search quota is exhausted. Please store your own API keys.",
                    "free_usage_remaining": free_remaining,
                    "deep_remaining": 0
                }
        
        # Beide Quota-Checks bestanden – jetzt beide Zähler erhöhen
        usage_counter[uid] = current_usage + increment
        if deep_search:
            deep_search_usage[uid] = current_deep_usage + increment

        if use_own_keys:
            if not (api_key and api_key.strip()):
                raise HTTPException(status_code=400, detail="Please log in or store your own API keys.")
            answer = query_gemini(question, api_key.strip(), search_mode, system_prompt)
            key_used = "User API Key"
        else:
            answer = query_gemini(question, None, search_mode, system_prompt)
            key_used = "Service Account"
        
        free_remaining = FREE_USAGE_LIMIT - usage_counter[uid]
        deep_remaining = 12 - deep_search_usage.get(uid, 0)
        return {"response": answer, "free_usage_remaining": free_remaining, "deep_remaining": deep_remaining, "key_used": key_used}
    else:
        # Nicht eingeloggte Nutzer müssen einen eigenen API Key bereitstellen.
        if not (api_key and api_key.strip()):
            raise HTTPException(status_code=400, detail="Please log in or store your own API keys.")
        answer = query_gemini(question, api_key.strip(), search_mode, system_prompt)
        return {"response": answer, "key_used": "User API Key"}

# Angepasster Endpoint für DeepSeek
@app.post("/ask_deepseek")
@limiter.limit("3/minute")
async def ask_deepseek_post(request: Request, data: dict = Body(...)):
    question = data.get("question")
    # Konvertiere deep_search in einen Boolean
    deep_search_raw = data.get("deep_search", False)
    deep_search = True if str(deep_search_raw).lower() == "true" else False
    # Wähle das entsprechende Wortlimit
    max_words_limit = DEEP_SEARCH_MAX_WORDS if deep_search else MAX_WORDS

    if count_words(question) > max_words_limit:
        raise HTTPException(
            status_code=400,
            detail=f"Die Eingabe überschreitet das erlaubte Wortlimit von {max_words_limit} Wörtern."
        )
    system_prompt = data.get("system_prompt")
    id_token = data.get("id_token")
    api_key = data.get("api_key")
    active_count = data.get("active_count", 1)
    
    if id_token:
        try:
            uid = verify_user_token(id_token)
        except Exception:
            raise HTTPException(status_code=401, detail="Authentication failed")

        increment = 1.0 / active_count

        # --- Zuerst beide Quota-Checks durchführen ---
        current_usage = usage_counter.get(uid, 0)
        if current_usage + increment > FREE_USAGE_LIMIT:
            # Auch im Fehlerfall die aktuellen Zählerwerte zurückgeben:
            deep_remaining = 12 - deep_search_usage.get(uid, 0)
            return {
                "error": "Your free quota is exhausted. Please provide your own API keys.",
                "free_usage_remaining": 0,
                "deep_remaining": deep_remaining
            }

        if deep_search:
            current_deep_usage = deep_search_usage.get(uid, 0)
            if current_deep_usage + increment > 12:
                free_remaining = FREE_USAGE_LIMIT - current_usage
                # Hier werden beide Werte mitgegeben, auch wenn deep search nicht verfügbar ist:
                return {
                    "error": "Your free deep search quota is exhausted. Please store your own API keys.",
                    "free_usage_remaining": free_remaining,
                    "deep_remaining": 0
                }

        # --- Jetzt beide Zähler inkrementieren, da beide Checks bestanden sind ---
        usage_counter[uid] = current_usage + increment
        if deep_search:
            deep_search_usage[uid] = current_deep_usage + increment

        developer_api_key = os.environ.get("DEVELOPER_DEEPSEEK_API_KEY")
        if not developer_api_key:
            raise HTTPException(status_code=500, detail="Server error: API key not configured")
        answer = query_deepseek(
            question,
            developer_api_key,
            system_prompt,
            deep_search=deep_search
        )
        free_remaining = FREE_USAGE_LIMIT - usage_counter[uid]
        deep_remaining = 12 - deep_search_usage.get(uid, 0)
        return {"response": answer, "free_usage_remaining": free_remaining, "deep_remaining": deep_remaining, "key_used": "Developer API Key"}
    elif api_key:
        answer = query_deepseek(
            question,
            api_key,
            system_prompt,
            deep_search=deep_search
        )
        # Auch hier beide Werte immer mitgeben
        return {
            "response": answer,
            "free_usage_remaining": FREE_USAGE_LIMIT,
            "deep_remaining": 12,
            "key_used": "User API Key"
        }
    else:
        raise HTTPException(status_code=400, detail="No id_token or api_key specified.")

    
@app.post("/ask_grok")
@limiter.limit("3/minute")
async def ask_grok_post(request: Request, data: dict = Body(...)):
    question = data.get("question")
    # Konvertiere deep_search in einen Boolean
    deep_search_raw = data.get("deep_search", False)
    deep_search = True if str(deep_search_raw).lower() == "true" else False
    # Wähle das entsprechende Wortlimit
    max_words_limit = DEEP_SEARCH_MAX_WORDS if deep_search else MAX_WORDS

    if count_words(question) > max_words_limit:
        raise HTTPException(
            status_code=400,
            detail=f"Die Eingabe überschreitet das erlaubte Wortlimit von {max_words_limit} Wörtern."
        )
    system_prompt = data.get("system_prompt")
    id_token = data.get("id_token")
    api_key = data.get("api_key")
    active_count = data.get("active_count", 1)
    
    if id_token:
        try:
            uid = verify_user_token(id_token)
        except Exception:
            raise HTTPException(status_code=401, detail="Authentication failed")

        increment = 1.0 / active_count

        # --- Zuerst beide Quota-Checks durchführen ---
        current_usage = usage_counter.get(uid, 0)
        if current_usage + increment > FREE_USAGE_LIMIT:
            # Auch im Fehlerfall die aktuellen Zählerwerte zurückgeben:
            deep_remaining = 12 - deep_search_usage.get(uid, 0)
            return {
                "error": "Your free quota is exhausted. Please provide your own API keys.",
                "free_usage_remaining": 0,
                "deep_remaining": deep_remaining
            }

        if deep_search:
            current_deep_usage = deep_search_usage.get(uid, 0)
            if current_deep_usage + increment > 12:
                free_remaining = FREE_USAGE_LIMIT - current_usage
                # Hier werden beide Werte mitgegeben, auch wenn deep search nicht verfügbar ist:
                return {
                    "error": "Your free deep search quota is exhausted. Please store your own API keys.",
                    "free_usage_remaining": free_remaining,
                    "deep_remaining": 0
                }

        # --- Jetzt beide Zähler inkrementieren, da beide Checks bestanden sind ---
        usage_counter[uid] = current_usage + increment
        if deep_search:
            deep_search_usage[uid] = current_deep_usage + increment

        developer_api_key = os.environ.get("DEVELOPER_GROK_API_KEY")
        if not developer_api_key:
            raise HTTPException(status_code=500, detail="Server error: API key not configured")
        answer = query_grok(
            question,
            developer_api_key,
            system_prompt,
            deep_search=deep_search
        )
        free_remaining = FREE_USAGE_LIMIT - usage_counter[uid]
        deep_remaining = 12 - deep_search_usage.get(uid, 0)
        return {"response": answer, "free_usage_remaining": free_remaining, "deep_remaining": deep_remaining, "key_used": "Developer API Key"}
    elif api_key:
        answer = query_grok(
            question,
            api_key,
            system_prompt,
            deep_search=deep_search
        )
        # Auch hier beide Werte immer mitgeben
        return {
            "response": answer,
            "free_usage_remaining": FREE_USAGE_LIMIT,
            "deep_remaining": 12,
            "key_used": "User API Key"
        }
    else:
        raise HTTPException(status_code=400, detail="No id_token or api_key specified.")
    

@app.post("/ask_exa")
@limiter.limit("3/minute")
async def ask_exa_post(request: Request, data: dict = Body(...)):
    """
    Endpoint für Exa im Search Mode.
    Nutzt entweder den Developer-API-Key (bei eingeloggten Nutzern) oder einen vom User gelieferten API-Key.
    """
    question = data.get("question")
    # Konvertiere deep_search in einen Boolean
    deep_search_raw = data.get("deep_search", False)
    deep_search = True if str(deep_search_raw).lower() == "true" else False
    # Wähle das entsprechende Wortlimit
    max_words_limit = DEEP_SEARCH_MAX_WORDS if deep_search else MAX_WORDS

    if count_words(question) > max_words_limit:
        raise HTTPException(
            status_code=400,
            detail=f"Die Eingabe überschreitet das erlaubte Wortlimit von {max_words_limit} Wörtern."
        )
    system_prompt = data.get("system_prompt")
    id_token = data.get("id_token")
    api_key = data.get("api_key")
    active_count = data.get("active_count", 1)
    # Exa wird ausschließlich im Search Mode eingesetzt – daher setzen wir search_mode auf True.
    search_mode = True

    if id_token:
        try:
            uid = verify_user_token(id_token)
        except Exception:
            raise HTTPException(status_code=401, detail="Authentication failed")

        increment = 1.0 / active_count

        # --- Zuerst beide Quota-Checks durchführen ---
        current_usage = usage_counter.get(uid, 0)
        if current_usage + increment > FREE_USAGE_LIMIT:
            # Auch im Fehlerfall die aktuellen Zählerwerte zurückgeben:
            deep_remaining = 12 - deep_search_usage.get(uid, 0)
            return {
                "error": "Your free quota is exhausted. Please provide your own API keys.",
                "free_usage_remaining": 0,
                "deep_remaining": deep_remaining
            }

        if deep_search:
            current_deep_usage = deep_search_usage.get(uid, 0)
            if current_deep_usage + increment > 12:
                free_remaining = FREE_USAGE_LIMIT - current_usage
                # Hier werden beide Werte mitgegeben, auch wenn deep search nicht verfügbar ist:
                return {
                    "error": "Your free deep search quota is exhausted. Please store your own API keys.",
                    "free_usage_remaining": free_remaining,
                    "deep_remaining": 0
                }

        # --- Jetzt beide Zähler inkrementieren, da beide Checks bestanden sind ---
        usage_counter[uid] = current_usage + increment
        if deep_search:
            deep_search_usage[uid] = current_deep_usage + increment

        developer_api_key = os.environ.get("DEVELOPER_EXA_API_KEY")
        if not developer_api_key:
            raise HTTPException(status_code=500, detail="Server error: API key not configured")
        answer = query_exa(
            question,
            developer_api_key,
            search_mode,          # Search mode ist True
            system_prompt,
            deep_search=deep_search
        )
        free_remaining = FREE_USAGE_LIMIT - usage_counter[uid]
        deep_remaining = 12 - deep_search_usage.get(uid, 0)
        return {"response": answer, "free_usage_remaining": free_remaining, "deep_remaining": deep_remaining, "key_used": "Developer API Key"}
    elif api_key:
        answer = query_exa(
            question,
            api_key,
            search_mode,          # Search mode ist True
            system_prompt,
            deep_search=deep_search
        )
        # Auch hier beide Werte immer mitgeben
        return {
            "response": answer,
            "free_usage_remaining": FREE_USAGE_LIMIT,
            "deep_remaining": 12,
            "key_used": "User API Key"
        }
    else:
        raise HTTPException(status_code=400, detail="No authentication parameter (id_token or api_key) specified")


@app.post("/ask_perplexity")
@limiter.limit("3/minute")
async def ask_perplexity_post(request: Request, data: dict = Body(...)):
    """
    Endpoint für Perplexity im Search Mode.
    Nutzt entweder den Developer-API-Key (bei eingeloggten Nutzern) oder einen vom User gelieferten API-Key.
    """
    question = data.get("question")
    # Konvertiere deep_search in einen Boolean
    deep_search_raw = data.get("deep_search", False)
    deep_search = True if str(deep_search_raw).lower() == "true" else False
    # Wähle das entsprechende Wortlimit
    max_words_limit = DEEP_SEARCH_MAX_WORDS if deep_search else MAX_WORDS

    if count_words(question) > max_words_limit:
        raise HTTPException(
            status_code=400,
            detail=f"Die Eingabe überschreitet das erlaubte Wortlimit von {max_words_limit} Wörtern."
        )
    system_prompt = data.get("system_prompt")
    id_token = data.get("id_token")
    api_key = data.get("api_key")
    active_count = data.get("active_count", 1)
    
    # Da Perplexity ausschließlich im Search Mode genutzt wird, setzen wir search_mode auf True.
    search_mode = True
    
    if id_token:
        try:
            uid = verify_user_token(id_token)
        except Exception:
            raise HTTPException(status_code=401, detail="Authentication failed")

        increment = 1.0 / active_count

        # --- Zuerst beide Quota-Checks durchführen ---
        current_usage = usage_counter.get(uid, 0)
        if current_usage + increment > FREE_USAGE_LIMIT:
            # Auch im Fehlerfall die aktuellen Zählerwerte zurückgeben:
            deep_remaining = 12 - deep_search_usage.get(uid, 0)
            return {
                "error": "Your free quota is exhausted. Please provide your own API keys.",
                "free_usage_remaining": 0,
                "deep_remaining": deep_remaining
            }

        if deep_search:
            current_deep_usage = deep_search_usage.get(uid, 0)
            if current_deep_usage + increment > 12:
                free_remaining = FREE_USAGE_LIMIT - current_usage
                # Hier werden beide Werte mitgegeben, auch wenn deep search nicht verfügbar ist:
                return {
                    "error": "Your free deep search quota is exhausted. Please store your own API keys.",
                    "free_usage_remaining": free_remaining,
                    "deep_remaining": 0
                }

        # --- Jetzt beide Zähler inkrementieren, da beide Checks bestanden sind ---
        usage_counter[uid] = current_usage + increment
        if deep_search:
            deep_search_usage[uid] = current_deep_usage + increment

        developer_api_key = os.environ.get("DEVELOPER_PERPLEXITY_API_KEY")
        if not developer_api_key:
            raise HTTPException(status_code=500, detail="Server error: API key not configured")
        answer = query_perplexity(
            question,
            developer_api_key,
            search_mode,
            system_prompt,
            deep_search=deep_search
        )
        free_remaining = FREE_USAGE_LIMIT - usage_counter[uid]
        deep_remaining = 12 - deep_search_usage.get(uid, 0)
        return {"response": answer, "free_usage_remaining": free_remaining, "deep_remaining": deep_remaining, "key_used": "Developer API Key"}
    elif api_key:
        answer = query_perplexity(
            question,
            api_key,
            search_mode,
            system_prompt,
            deep_search=deep_search
        )
        # Auch hier beide Werte immer mitgeben
        return {
            "response": answer,
            "free_usage_remaining": FREE_USAGE_LIMIT,
            "deep_remaining": 12,
            "key_used": "User API Key"
        }
    else:
        raise HTTPException(status_code=400, detail="No authentication parameter (id_token or api_key) specified")


@app.post("/consensus")
@limiter.limit("3/minute")
async def consensus(request: Request, data: dict = Body(...)):
    id_token = data.get("id_token")
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
    question = data.get("question")
    answer_openai   = data.get("answer_openai")
    answer_mistral  = data.get("answer_mistral")
    answer_claude   = data.get("answer_claude")
    answer_gemini   = data.get("answer_gemini")
    answer_deepseek = data.get("answer_deepseek")
    answer_grok     = data.get("answer_grok")
    answer_exa     = data.get("answer_exa")
    answer_perplexity     = data.get("answer_perplexity")
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
        api_keys["Perplexity"] = data.get("perplexity_key")
    else:
        api_keys["OpenAI"] = data.get("openai_key") or os.environ.get("DEVELOPER_OPENAI_API_KEY")
        api_keys["Mistral"] = data.get("mistral_key") or os.environ.get("DEVELOPER_MISTRAL_API_KEY")
        api_keys["Anthropic"] = data.get("anthropic_key") or os.environ.get("DEVELOPER_ANTHROPIC_API_KEY")
        api_keys["Gemini"] = data.get("gemini_key") or os.environ.get("DEVELOPER_GEMINI_API_KEY")
        api_keys["DeepSeek"] = data.get("deepseek_key") or os.environ.get("DEVELOPER_DEEPSEEK_API_KEY")
        api_keys["Grok"] = data.get("grok_key") or os.environ.get("DEVELOPER_GROK_API_KEY")
        api_keys["Exa"] = data.get("exa_key") or os.environ.get("DEVELOPER_EXA_API_KEY")
        api_keys["Perplexity"] = data.get("perplexity_key") or os.environ.get("DEVELOPER_PERPLEXITY_API_KEY")

    # Validierung der erforderlichen Parameter (nur für Modelle, die nicht ausgeschlossen wurden)
    missing = []
    if not question:
        missing.append("question")
    if not consensus_model:
        missing.append("consensus_model")
    # Immer prüfen: OpenAI und Gemini
    if "OpenAI" not in excluded_models:
        if not answer_openai or not api_keys.get("OpenAI"):
            missing.append("OpenAI")
    if "Gemini" not in excluded_models:
        if not answer_gemini or (use_own_keys and not api_keys.get("Gemini")):
            missing.append("Gemini")
    # Bei Search Mode: Exa UND Perplexity prüfen, ansonsten die anderen Modelle
    if search_mode:
        if "Exa" not in excluded_models:
            if not answer_exa or not api_keys.get("Exa"):
                missing.append("Exa")
        if "Perplexity" not in excluded_models:
            if not answer_perplexity or not api_keys.get("Perplexity"):
                missing.append("Perplexity")
    else:
        if "Mistral" not in excluded_models:
            if not answer_mistral or not api_keys.get("Mistral"):
                missing.append("Mistral")
        if "Anthropic" not in excluded_models:
            if not answer_claude or not api_keys.get("Anthropic"):
                missing.append("Anthropic")
        if "DeepSeek" not in excluded_models:
            if not answer_deepseek or not api_keys.get("DeepSeek"):
                missing.append("DeepSeek")
        if "Grok" not in excluded_models:
            if not answer_grok or not api_keys.get("Grok"):
                missing.append("Grok")

    if missing:
        raise HTTPException(status_code=400, detail="Missing parameters: " + ", ".join(missing))

    if best_model and best_model in excluded_models:
        raise HTTPException(status_code=400, detail="The answer marked as best must not be excluded.")

    # Konsens-Antwort generieren
    consensus_answer = query_consensus(
        question, answer_openai, answer_mistral, answer_claude, answer_gemini, answer_deepseek, answer_grok, answer_exa, answer_perplexity,
        best_model, excluded_models, consensus_model, api_keys, search_mode)

    # Unterschiede ermitteln
    differences = query_differences(
        answer_openai, answer_mistral, answer_claude, answer_gemini, answer_deepseek, answer_grok, answer_exa, answer_perplexity,
        consensus_answer, api_keys, differences_model=consensus_model, search_mode=search_mode)

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
        perplexity_key = data.get("perplexity_key")
        
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
                    "model": "claude-3-5-sonnet-20241022",
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
                    model="grok-2-latest",
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

        # Perplexity Handshake
        try:
            if perplexity_key and len(perplexity_key) > 10:
                client = openai.OpenAI(api_key=perplexity_key, base_url="https://api.perplexity.ai")
                response = client.chat.completions.create(
                    model="sonar",
                    messages=[
                        {"role": "system", "content": "ping"},
                        {"role": "user", "content": "ping"}
                    ]
                )
                results["Perplexity"] = "valid"
            else:
                results["Perplexity"] = "invalid"
        except Exception as e:
            results["Perplexity"] = "invalid"
        
        return {"results": results}

    except Exception as overall_error:
        return {"results": {"error": str(overall_error)}}