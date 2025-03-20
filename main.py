
import os
from fastapi import FastAPI, Query, Request, HTTPException, Body
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import openai
import requests
from mistralai import Mistral
from dotenv import load_dotenv
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, auth
from typing import Optional
from datetime import datetime, timedelta

# Load .env if necessary (no longer used here for API keys)
load_dotenv()

app = FastAPI()

# Free usage limit
FREE_USAGE_LIMIT = 25

usage_counter = {}  # { uid: anzahl_anfragen }

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gen-lang-client-0234219247-53b2b1c0e355.json"

# Keine globalen API Keys mehr – diese werden nun via Request übergeben

def query_openai(question: str, api_key: str, search_mode: bool = False) -> str:
    try:
        client = openai.OpenAI(api_key=api_key)
        # Wähle das richtige Modell: falls search_mode aktiv ist, nutze "gpt-4o-search-preview"
        model_to_use = "gpt-4o-search-preview" if search_mode else "gpt-4o"
        response = client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": "Please respond briefly and precisely, focusing only on the essentials."},
                {"role": "user", "content": question}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error with OpenAI: {str(e)}"

def query_mistral(question: str, api_key: str) -> str:
    """Fragt die Mistral API zu der gegebenen Frage unter Verwendung des übergebenen API Keys ohne Limit."""
    try:
        client = Mistral(api_key=api_key)
        model = "mistral-large-latest"
        response = client.chat.complete(
            model=model,
            messages=[
                {"role": "system", "content": "Please respond briefly and precisely, focusing only on the essentials."},
                {"role": "user", "content": question}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error with Mistral: {str(e)}"


def query_claude(question: str, api_key: str) -> str:
    """Fragt die Anthropic API (Claude) zu der gegebenen Frage unter Verwendung des übergebenen API Keys ohne Limit.
       Da die Anthropic API ein Token-Limit erwartet, setzen wir einen sehr hohen Wert ein."""
    try:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 8192,  # Sehr hoher Wert als "unbegrenzt"
            "system": "Please respond briefly and precisely, focusing only on the essentials.",
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
            return f"Error with Anthropic Claude: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error with Anthropic Claude: {str(e)}"

def query_gemini(question: str, user_api_key: Optional[str] = None, search_mode: bool = False) -> str:
    try:
        # Wenn ein eigener API Key übergeben wurde, verwende ihn.
        # Andernfalls – bei eingeloggten Nutzern – wird der Service-Account-Key aus der JSON genutzt.
        if user_api_key and user_api_key.strip():
            genai.configure(api_key=user_api_key)
        else:
            genai.configure()
        
        # Je nach Search Mode den passenden Modellnamen wählen:
        model_name = "models/gemini-1.5-pro-002" if search_mode else "gemini-1.5-pro-latest"
        model = genai.GenerativeModel(model_name)
        
        base_content = "Bitte antworte präzise und stelle keine Rückfragen. " + question
        if search_mode:
            # Hinweis im Prompt, der auch die Links (über Retrieval) liefern soll
            base_content += "\nBitte füge am Ende deiner Antwort klickbare Links zu den verwendeten Quellen ein."
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
        return f"Error with Google Gemini: {str(e)}"

    
def query_deepseek(question: str, api_key: str) -> str:
    """Fragt DeepSeek zu der gegebenen Frage unter Verwendung des übergebenen API Keys."""
    try:
        client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Please respond briefly and precisely, focusing only on the essentials."},
                {"role": "user", "content": question}
            ],
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error with DeepSeek: {str(e)}"
    
def query_grok(question: str, api_key: str) -> str:
    """Fragt die Grok API zu der gegebenen Frage unter Verwendung des übergebenen API Keys."""
    try:
        client = openai.OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        response = client.chat.completions.create(
            model="grok-2-latest",
            messages=[
                {"role": "system", "content": "Please respond briefly and precisely, focusing only on the essentials."},
                {"role": "user", "content": question}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error with Grok: {str(e)}"


def query_consensus(question: str, answer_openai: str, answer_mistral: str, answer_claude: str, 
                    answer_gemini: str, answer_deepseek: str, answer_grok: str,
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
    if "Anthropic Claude" not in excluded_models and answer_claude:
        prompt_parts.append(f"Response from claude-3-5-sonnet: {answer_claude}\n\n")
    if "Google Gemini" not in excluded_models and answer_gemini:
        prompt_parts.append(f"Response from gemini-pro: {answer_gemini}\n\n")
    if "DeepSeek" not in excluded_models and answer_deepseek:
        prompt_parts.append(f"Response from deepseek-chat: {answer_deepseek}\n\n")
    if "Grok" not in excluded_models and answer_grok:
        prompt_parts.append(f"Response from Grok: {answer_grok}\n\n")

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
            model_to_use = "gpt-4o-search-preview" if search_mode else "gpt-4o"
            response = client.chat.completions.create(
                        model=model_to_use,
                        messages=[
                            {"role": "system", "content": ""},
                            {"role": "user", "content": consensus_prompt}
                        ]
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
                ]
            )
            return response.choices[0].message.content.strip()
        elif consensus_model == "Anthropic Claude":
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": api_keys.get("Anthropic Claude"),
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            payload = {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 8192, 
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
                return f"Error with Anthropic Claude: {response.status_code} - {response.text}"
            
        elif consensus_model == "Google Gemini":
            gemini_key = api_keys.get("Google Gemini")
            if gemini_key and gemini_key.strip() != "":
                genai.configure(api_key=gemini_key)
            else:
                genai.configure()  # Service-Account-Modus
            model = genai.GenerativeModel("gemini-1.5-pro-latest")
            # Schalte auch hier zwischen Search und normalem Modus um:
            if search_mode:
                response = model.generate_content(
                    consensus_prompt,
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
                ]
            )
            return response.choices[0].message.content.strip()
        
        elif consensus_model == "Grok":
            client = openai.OpenAI(api_key=api_keys.get("Grok"), base_url="https://api.x.ai/v1")
            response = client.chat.completions.create(
                model="grok-2-latest",
                messages=[
                    {"role": "system", "content": " "},
                    {"role": "user", "content": consensus_prompt}
                ]
            )
            return response.choices[0].message.content.strip()

        else:
            return "Invalid consensus model selected."
    except Exception as e:
        return f"Consensus error: {str(e)}"
    

def query_differences(answer_openai: str, answer_mistral: str, answer_claude: str, answer_gemini: str, answer_deepseek: str, answer_grok: str, consensus_answer: str, api_keys: dict, differences_model: str) -> str:
    """
    Extrahiert die Unterschiede zwischen den vier Expertenantworten mittels des angegebenen Konsens‑Modells.
    """
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

        "Model responses:\n"
        "- GPT-4o: " + answer_openai + "\n"
        "- Mistral: " + answer_mistral + "\n"
        "- Claude: " + answer_claude + "\n"
        "- Gemini: " + answer_gemini + "\n"
        "- DeepSeek: " + answer_deepseek + "\n"
        "- Grok: " + answer_grok + "\n\n"

        "Finally, subjectively determine which model provided the best answer. "
        "Choose from one of the following models: Anthropic, Gemini, Mistral, or OpenAI. "
        "Include your decision at the end of the response on a separate line, starting with 'BestModel:' followed by the model name.\n\n"

        "Response format:\n"
        "[Credibility statement]\n"
        "\n"
        "_____________\n"
        "\n"
        "[Very brief explanation of why these differences affect credibility.]\n\n"
        "BestModel: [Model name]"
    )

    try:
        if differences_model == "OpenAI":
            client = openai.OpenAI(api_key=api_keys.get("OpenAI"))
            response = client.chat.completions.create(
                model="gpt-4o",
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
        elif differences_model == "Anthropic Claude":
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": api_keys.get("Anthropic Claude"),
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
                return f"Error with Anthropic Claude: {response.status_code} - {response.text}"
            
        elif differences_model == "Google Gemini":
            gemini_key = api_keys.get("Google Gemini")
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

        else:
            return "Invalid model selected for difference comparison."
    except Exception as e:
        return f"Error in comparison: {str(e)}"
    
# Initialisiere Firebase Admin (Beispiel, passe den Pfad zu deinem Service Account an)
cred = credentials.Certificate("consensai-firebase-adminsdk-fbsvc-9064a77134.json")
firebase_admin.initialize_app(cred)

def verify_user_token(token: str) -> str:
    """
    Verifiziert das Firebase-ID-Token und gibt die uid zurück.
    """
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token["uid"]
    except Exception as e:
        raise Exception("Invalid token")


@app.get("/", response_class=HTMLResponse)
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

# Globales Dictionary zum Speichern der IP-Adressen registrierter Nutzer
registered_ips = {}  # { ip_address: uid }

@app.post("/register")
async def register_user(request: Request, data: dict):
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
        # Speichere die IP-Adresse als registriert
        registered_ips[ip_address] = user.uid
        # Erzeuge ein Custom Token für den neuen Nutzer
        custom_token = auth.create_custom_token(user.uid)
        # Das Token ist ein Bytes-Objekt – in einen String konvertieren
        custom_token_str = custom_token.decode("utf-8")
        return {"uid": user.uid, "email": user.email, "customToken": custom_token_str}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/usage")
async def get_usage_post(data: dict):
    try:
        token = data.get("id_token")
        uid = verify_user_token(token)
        current_usage = usage_counter.get(uid, 0)
        remaining = FREE_USAGE_LIMIT - current_usage
        return {"remaining": remaining}
    except Exception:
        raise HTTPException(status_code=401, detail="Authentication failed")

@app.post("/ask_openai")
async def ask_openai_post(data: dict = Body(...)):
    question = data.get("question")
    id_token = data.get("id_token")
    api_key = data.get("api_key")
    # Neuer Parameter: search_mode (Standard: False)
    search_mode = data.get("search_mode", False)
    active_count = data.get("active_count", 1)
    
    if id_token:
        try:
            uid = verify_user_token(id_token)
        except Exception:
            raise HTTPException(status_code=401, detail="Authentication failed")
        current_usage = usage_counter.get(uid, 0)
        increment = 1.0 / active_count
        if current_usage + increment > FREE_USAGE_LIMIT:
            return {"error": "Your free quota is exhausted. Please provide your own API keys."}
        usage_counter[uid] = current_usage + increment

        developer_api_key = os.environ.get("DEVELOPER_OPENAI_API_KEY")
        if not developer_api_key:
            raise HTTPException(status_code=500, detail="Server error: API key not configured")
        # Übergabe des search_mode-Flags
        answer = query_openai(question, developer_api_key, search_mode)
        free_remaining = FREE_USAGE_LIMIT - usage_counter[uid]
        return {"response": answer, "free_usage_remaining": free_remaining, "key_used": "Developer API Key"}
    elif api_key:
        answer = query_openai(question, api_key, search_mode)
        return {"response": answer, "key_used": "User API Key"}
    else:
        raise HTTPException(status_code=400, detail="No authentication parameter (id_token or api_key) specified")

# Angepasster Endpoint für Mistral
@app.post("/ask_mistral")
async def ask_mistral_post(data: dict = Body(...)):
    question = data.get("question")
    id_token = data.get("id_token")
    api_key = data.get("api_key")
    active_count = data.get("active_count", 1)
    
    if id_token:
        try:
            uid = verify_user_token(id_token)
        except Exception:
            raise HTTPException(status_code=401, detail="Authentication failed")
        current_usage = usage_counter.get(uid, 0)
        increment = 1.0 / active_count
        if current_usage + increment > FREE_USAGE_LIMIT:
            return {"error": "Your free quota has been used up. Please store your own API keys."}
        usage_counter[uid] = current_usage + increment

        developer_api_key = os.environ.get("DEVELOPER_MISTRAL_API_KEY")
        if not developer_api_key:
            raise HTTPException(status_code=500, detail="Server error: API key not configured")
        answer = query_mistral(question, developer_api_key)
        free_remaining = FREE_USAGE_LIMIT - usage_counter[uid]
        return {"response": answer, "free_usage_remaining": free_remaining, "key_used": "Developer API Key"}
    elif api_key:
        answer = query_mistral(question, api_key)
        return {"response": answer, "key_used": "User API Key"}
    else:
        raise HTTPException(status_code=400, detail="No id_token or api_key specified.")

# Angepasster Endpoint für Anthropic Claude
@app.post("/ask_claude")
async def ask_claude_post(data: dict = Body(...)):
    question = data.get("question")
    id_token = data.get("id_token")
    api_key = data.get("api_key")
    active_count = data.get("active_count", 1)
    
    if id_token:
        try:
            uid = verify_user_token(id_token)
        except Exception:
            raise HTTPException(status_code=401, detail="Authentication failed")
        current_usage = usage_counter.get(uid, 0)
        increment = 1.0 / active_count
        if current_usage + increment > FREE_USAGE_LIMIT:
            return {"error": "Your free quota has been used up. Please store your own API keys."}
        usage_counter[uid] = current_usage + increment

        developer_api_key = os.environ.get("DEVELOPER_ANTHROPIC_API_KEY")
        if not developer_api_key:
            raise HTTPException(status_code=500, detail="Server error: API key not configured")
        answer = query_claude(question, developer_api_key)
        free_remaining = FREE_USAGE_LIMIT - usage_counter[uid]
        return {"response": answer, "free_usage_remaining": free_remaining, "key_used": "Developer API Key"}
    elif api_key:
        answer = query_claude(question, api_key)
        return {"response": answer, "key_used": "User API Key"}
    else:
        raise HTTPException(status_code=400, detail="No id_token or api_key specified.")

@app.post("/ask_gemini")
async def ask_gemini_post(data: dict = Body(...)):
    question = data.get("question")
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
        current_usage = usage_counter.get(uid, 0)
        increment = 1.0 / active_count
        if current_usage + increment > FREE_USAGE_LIMIT:
            return {"error": "Your free quota has been used up. Please store your own API keys."}
        usage_counter[uid] = current_usage + increment

        if use_own_keys:
            if not (api_key and api_key.strip()):
                raise HTTPException(status_code=400, detail="Please log in or store your own API keys.")
            answer = query_gemini(question, api_key.strip(), search_mode)
            key_used = "User API Key"
        else:
            answer = query_gemini(question, None, search_mode)
            key_used = "Service Account"
        free_remaining = FREE_USAGE_LIMIT - usage_counter[uid]
        return {"response": answer, "free_usage_remaining": free_remaining, "key_used": key_used}
    else:
        # Nicht eingeloggte Nutzer müssen einen eigenen API Key bereitstellen.
        if not (api_key and api_key.strip()):
            raise HTTPException(status_code=400, detail="Please log in or store your own API keys.")
        answer = query_gemini(question, api_key.strip(), search_mode)
        return {"response": answer, "key_used": "User API Key"}


# Angepasster Endpoint für DeepSeek
@app.post("/ask_deepseek")
async def ask_deepseek_post(data: dict = Body(...)):
    question = data.get("question")
    id_token = data.get("id_token")
    api_key = data.get("api_key")
    active_count = data.get("active_count", 1)
    
    if id_token:
        try:
            uid = verify_user_token(id_token)
        except Exception:
            raise HTTPException(status_code=401, detail="Authentication failed")
        current_usage = usage_counter.get(uid, 0)
        increment = 1.0 / active_count
        if current_usage + increment > FREE_USAGE_LIMIT:
            return {"error": "Your free quota has been used up. Please store your own API keys."}
        usage_counter[uid] = current_usage + increment

        developer_api_key = os.environ.get("DEVELOPER_DEEPSEEK_API_KEY")
        if not developer_api_key:
            raise HTTPException(status_code=500, detail="Server error: API key not configured")
        answer = query_deepseek(question, developer_api_key)
        free_remaining = FREE_USAGE_LIMIT - usage_counter[uid]
        return {"response": answer, "free_usage_remaining": free_remaining, "key_used": "Developer API Key"}
    elif api_key:
        answer = query_deepseek(question, api_key)
        return {"response": answer, "key_used": "User API Key"}
    else:
        raise HTTPException(status_code=400, detail="No id_token or api_key specified.")
    
@app.post("/ask_grok")
async def ask_grok_post(data: dict = Body(...)):
    question = data.get("question")
    id_token = data.get("id_token")
    api_key = data.get("api_key")
    active_count = data.get("active_count", 1)
    
    if id_token:
        try:
            uid = verify_user_token(id_token)
        except Exception:
            raise HTTPException(status_code=401, detail="Authentication failed")
        current_usage = usage_counter.get(uid, 0)
        increment = 1.0 / active_count
        if current_usage + increment > FREE_USAGE_LIMIT:
            return {"error": "Your free quota has been used up. Please store your own API keys."}
        usage_counter[uid] = current_usage + increment

        developer_api_key = os.environ.get("DEVELOPER_GROK_API_KEY")
        if not developer_api_key:
            raise HTTPException(status_code=500, detail="Server error: API key not configured")
        answer = query_grok(question, developer_api_key)
        free_remaining = FREE_USAGE_LIMIT - usage_counter[uid]
        return {"response": answer, "free_usage_remaining": free_remaining, "key_used": "Developer API Key"}
    elif api_key:
        answer = query_grok(question, api_key)
        return {"response": answer, "key_used": "User API Key"}
    else:
        raise HTTPException(status_code=400, detail="No id_token or api_key specified.")


@app.post("/consensus")
async def consensus(data: dict):
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
    best_model      = data.get("best_model", "")
    consensus_model = data.get("consensus_model")
    excluded_models = data.get("excluded_models", [])

    # API Keys setzen: Bei useOwnKeys werden die vom Nutzer übermittelten Keys genutzt,
    # andernfalls wird für fehlende Keys auf die Developer Keys zurückgegriffen.
    api_keys = {}
    if use_own_keys:
        api_keys["OpenAI"] = data.get("openai_key")
        api_keys["Mistral"] = data.get("mistral_key")
        api_keys["Anthropic Claude"] = data.get("anthropic_key")
        api_keys["Google Gemini"] = data.get("gemini_key")
        api_keys["DeepSeek"] = data.get("deepseek_key")
        api_keys["Grok"] = data.get("grok_key")
    else:
        api_keys["OpenAI"] = data.get("openai_key") or os.environ.get("DEVELOPER_OPENAI_API_KEY")
        api_keys["Mistral"] = data.get("mistral_key") or os.environ.get("DEVELOPER_MISTRAL_API_KEY")
        api_keys["Anthropic Claude"] = data.get("anthropic_key") or os.environ.get("DEVELOPER_ANTHROPIC_API_KEY")
        api_keys["Google Gemini"] = data.get("gemini_key") or os.environ.get("DEVELOPER_GEMINI_API_KEY")
        api_keys["DeepSeek"] = data.get("deepseek_key") or os.environ.get("DEVELOPER_DEEPSEEK_API_KEY")
        api_keys["Grok"] = data.get("grok_key") or os.environ.get("DEVELOPER_GROK_API_KEY")

    # Validierung der erforderlichen Parameter (nur für Modelle, die nicht ausgeschlossen wurden)
    missing = []
    if not question:
        missing.append("question")
    if not consensus_model:
        missing.append("consensus_model")
    if "OpenAI" not in excluded_models:
        if not answer_openai or not api_keys.get("OpenAI"):
            missing.append("OpenAI")
    if "Mistral" not in excluded_models:
        if not answer_mistral or not api_keys.get("Mistral"):
            missing.append("Mistral")
    if "Anthropic Claude" not in excluded_models:
        if not answer_claude or not api_keys.get("Anthropic Claude"):
            missing.append("Anthropic Claude")
    if "Google Gemini" not in excluded_models:
        if use_own_keys:
            if not answer_gemini or not api_keys.get("Google Gemini"):
                missing.append("Google Gemini")
        else:
            if not answer_gemini:
                missing.append("Google Gemini")
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
        question, answer_openai, answer_mistral, answer_claude, answer_gemini, answer_deepseek, answer_grok,
        best_model, excluded_models, consensus_model, api_keys, search_mode)

    # Unterschiede ermitteln
    differences = query_differences(
        answer_openai, answer_mistral, answer_claude, answer_gemini, answer_deepseek, answer_grok,
        consensus_answer, api_keys, differences_model=consensus_model)

    return {"consensus_response": consensus_answer, "differences": differences}


def is_valid(key):
    # Example validation: Key is considered valid if it is present and longer than 10 characters.
    return key is not None and len(key) > 10


@app.post("/check_keys")
async def check_keys(data: dict):
    try:
        openai_key = data.get("openai_key")
        mistral_key = data.get("mistral_key")
        anthropic_key = data.get("anthropic_key")
        gemini_key = data.get("gemini_key")
        deepseek_key = data.get("deepseek_key")
        grok_key = data.get("grok_key")
        
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
                    results["Anthropic Claude"] = "valid"
                else:
                    results["Anthropic Claude"] = "invalid"
            else:
                results["Anthropic Claude"] = "invalid"
        except Exception as e:
            results["Anthropic Claude"] = "invalid"

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
        
        # Google Gemini Handshake
        try:
            gemini_key = data.get("gemini_key")
            if gemini_key and len(gemini_key) > 10:
                genai.configure(api_key=gemini_key)
                model = genai.GenerativeModel("gemini-1.5-pro-latest")
                _ = model.generate_content("ping")
                results["Google Gemini"] = "valid"
            else:
                results["Google Gemini"] = "invalid"
        except Exception as e:
            results["Google Gemini"] = "invalid"
        
        return {"results": results}

    except Exception as overall_error:
        return {"results": {"error": str(overall_error)}}
