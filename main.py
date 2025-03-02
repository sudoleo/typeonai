click_counts = {
    "OpenAI": {"best": 0, "exclude": 0},
    "Mistral": {"best": 0, "exclude": 0},
    "Anthropic Claude": {"best": 0, "exclude": 0},
    "Google Gemini": {"best": 0, "exclude": 0},
}

from fastapi import FastAPI, Query, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import openai
import requests
from mistralai import Mistral
from dotenv import load_dotenv
import google.generativeai as genai

# Lade .env falls nötig (wird hier nicht mehr für API Keys genutzt)
load_dotenv()

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Keine globalen API Keys mehr – diese werden nun via Request übergeben

def query_openai(question: str, api_key: str) -> str:
    """Fragt OpenAI (GPT-4) mit der neuen API-Schnittstelle ohne Limit."""
    try:
        client = openai.OpenAI(api_key=api_key)  # Erstelle OpenAI-Client
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Keine Formatierung bitte."},
                {"role": "user", "content": question}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Fehler bei OpenAI: {str(e)}"


def query_mistral(question: str, api_key: str) -> str:
    """Fragt die Mistral API zu der gegebenen Frage unter Verwendung des übergebenen API Keys ohne Limit."""
    try:
        client = Mistral(api_key=api_key)
        model = "mistral-large-latest"
        response = client.chat.complete(
            model=model,
            messages=[
                {"role": "system", "content": "Keine Formatierung bitte."},
                {"role": "user", "content": question}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Fehler bei Mistral: {str(e)}"


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
            "system": "Keine Formatierung bitte.",
            "messages": [{"role": "user", "content": question}]
        }
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if "content" in data and isinstance(data["content"], list) and len(data["content"]) > 0:
                return data["content"][0]["text"]
            else:
                return "Fehler: Keine Antwort im API-Response gefunden."
        else:
            return f"Fehler bei Anthropic Claude: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Fehler bei Anthropic Claude: {str(e)}"


def query_gemini(question: str, api_key: str) -> str:
    """Fragt Google Gemini zu der gegebenen Frage unter Verwendung des übergebenen API Keys ohne Limit."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        response = model.generate_content(question)
        return response.text.strip()
    except Exception as e:
        return f"Fehler bei Google Gemini: {str(e)}"


def query_consensus(question: str, answer_openai: str, answer_mistral: str, answer_claude: str, answer_gemini: str,
                    best_model: str, excluded_models: list, consensus_model: str, api_keys: dict) -> str:
    """
    Nutzt ein Modell, um die Antworten der Modelle mittels Chain-of-Thought-Logik zu einem konsistenten Konsens zusammenzufassen.
    Die übergebenen API Keys werden dabei aus dem Dictionary 'api_keys' entnommen.
    """
    prompt_parts = [f"Die Frage lautet: {question}\n\n"]
    if "OpenAI" not in excluded_models:
        prompt_parts.append(f"Antwort von GPT-4o: {answer_openai}\n\n")
    if "Mistral" not in excluded_models:
        prompt_parts.append(f"Antwort von mistral-large-latest: {answer_mistral}\n\n")
    if "Anthropic Claude" not in excluded_models:
        prompt_parts.append(f"Antwort von claude-3-5-sonnet: {answer_claude}\n\n")
    if "Google Gemini" not in excluded_models:
        prompt_parts.append(f"Antwort von gemini-pro: {answer_gemini}\n\n")
    
    if best_model:
        prompt_parts.append(
            f"Der Nutzer hat die Antwort von {best_model} als die beste markiert. "
            "Du erhältst vier Meinungen von Experten zu einer bestimmten Frage. "
            "Deine Aufgabe ist es, diese Antworten zu bündeln und zu einer umfassenden, korrekten und kohärenten Antwort zu entwickeln. "
            "Beachte: Auch Experten können sich irren. Versuche daher, durch Abgleich der Antworten mögliche Fehler aufzudecken und auszuschließen. "
            "Falls sich die Antworten in einem Punkt stark widersprechen, überlege logisch, welche Variante am plausibelsten ist. "
            "Strukturiere die Antwort verständlich und schlüssig. "
            "Gib ausschließlich die finale, ausbalancierte Antwort aus."
        )
    else:
        prompt_parts.append(
            "Du erhältst vier Meinungen von Experten zu einer bestimmten Frage. "
            "Deine Aufgabe ist es, diese Antworten zu bündeln und zu einer umfassenden, korrekten und kohärenten Antwort zu entwickeln. "
            "Beachte: Auch Experten können sich irren. Versuche daher, durch Abgleich der Antworten mögliche Fehler aufzudecken und auszuschließen. "
            "Falls sich die Antworten in einem Punkt stark widersprechen, überlege logisch, welche Variante am plausibelsten ist. "
            "Strukturiere die Antwort verständlich und schlüssig. "
            "Gib ausschließlich die finale, ausbalancierte Antwort aus."
        )

    consensus_prompt = "".join(prompt_parts)
    
    try:
        if consensus_model == "OpenAI":
            client = openai.OpenAI(api_key=api_keys.get("OpenAI"))
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Keine Formatierung bitte."},
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
                    {"role": "system", "content": "Keine Formatierung bitte."},
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
                "max_tokens": 8192,  # Sehr hoher Wert als "unbegrenzt"
                "system": "Keine Formatierung bitte.",
                "messages": [{"role": "user", "content": consensus_prompt}]
            }
            response = requests.post(url, json=payload, headers=headers)
            if response.status_code == 200:
                data = response.json()
                if "content" in data and isinstance(data["content"], list) and len(data["content"]) > 0:
                    return data["content"][0]["text"]
                else:
                    return "Fehler: Keine Antwort im API-Response gefunden."
            else:
                return f"Fehler bei Anthropic Claude: {response.status_code} - {response.text}"
        elif consensus_model == "Google Gemini":
            genai.configure(api_key=api_keys.get("Google Gemini"))
            model = genai.GenerativeModel("gemini-1.5-pro-latest")
            response = model.generate_content(consensus_prompt)
            return response.text.strip()
        else:
            return "Ungültiges Konsensmodell ausgewählt."
    except Exception as e:
        return f"Fehler beim Konsens: {str(e)}"
    

def query_differences(answer_openai: str, answer_mistral: str, answer_claude: str, answer_gemini: str, consensus_answer: str, api_keys: dict, differences_model: str) -> str:
    """
    Extrahiert die Unterschiede zwischen den vier Expertenantworten mittels des angegebenen Konsens‑Modells.
    """
    differences_prompt = (
        "Analysiere die Antworten der LLMs und bewerte, wie stark sie voneinander abweichen. "
        "Falls alle Modelle nahezu identisch antworten, ist der Konsens sehr glaubwürdig. "
        "Falls es nur sprachliche Variationen gibt, ist er weitgehend glaubwürdig. "
        "Falls es inhaltliche Nuancen gibt, ist er teilweise glaubwürdig. "
        "Falls es klare Widersprüche gibt, ist er kaum oder nicht glaubwürdig. "
        "Antwort ausschließlich mit einem der folgenden Sätze:\n\n"
        
        "- 'Die Konsens-Antwort ist **sehr** glaubwürdig.'\n"
        "- 'Die Konsens-Antwort ist **weitgehend** glaubwürdig.'\n"
        "- 'Die Konsens-Antwort ist **teilweise** glaubwürdig.'\n"
        "- 'Die Konsens-Antwort ist **kaum** glaubwürdig.'\n"
        "- 'Die Konsens-Antwort ist **nicht** glaubwürdig.'\n\n"

        "Nach dem Satz folgt eine Trennlinie und eine **sehr knappe Erklärung**, warum diese Unterschiede relevant sind.\n\n"
        
        "Konsens-Antwort:\n" + consensus_answer + "\n\n"
        
        "Antworten der Modelle:\n"
        "- GPT-4o: " + answer_openai + "\n"
        "- Mistral: " + answer_mistral + "\n"
        "- Claude: " + answer_claude + "\n"
        "- Gemini: " + answer_gemini + "\n\n"
        
        "Format der Antwort:\n"
        "[Bewertungssatz]\n"
        "\n"
        "_____________\n"
        "\n"
        "[Sehr kurze Erklärung, warum diese Unterschiede die Glaubwürdigkeit beeinflussen.]"
    )

    try:
        if differences_model == "OpenAI":
            client = openai.OpenAI(api_key=api_keys.get("OpenAI"))
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Gib keine Formatierungen aus."},
                    {"role": "user", "content": differences_prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        elif differences_model == "Mistral":
            client = Mistral(api_keys.get("Mistral"))
            response = client.chat.complete(
                model="mistral-large-latest",
                messages=[
                    {"role": "system", "content": "Gib keine Formatierungen aus."},
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
                "system": "Gib keine Formatierungen aus.",
                "messages": [{"role": "user", "content": differences_prompt}]
            }
            response = requests.post(url, json=payload, headers=headers)
            if response.status_code == 200:
                data = response.json()
                if "content" in data and isinstance(data["content"], list) and len(data["content"]) > 0:
                    return data["content"][0]["text"]
                else:
                    return "Fehler: Keine Antwort im API-Response gefunden."
            else:
                return f"Fehler bei Anthropic Claude: {response.status_code} - {response.text}"
        elif differences_model == "Google Gemini":
            genai.configure(api_key=api_keys.get("Google Gemini"))
            model = genai.GenerativeModel("gemini-1.5-pro-latest")
            response = model.generate_content(differences_prompt)
            return response.text.strip()
        else:
            return "Ungültiges Modell für den Unterschiedsvergleich."
    except Exception as e:
        return f"Fehler beim Vergleich: {str(e)}"


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/ask_openai")
async def ask_openai(
    question: str = Query(..., description="Frage für OpenAI"),
    api_key: str = Query(..., description="API Key für OpenAI")
):
    answer = query_openai(question, api_key)
    return {"response": answer}


@app.get("/ask_mistral")
async def ask_mistral(
    question: str = Query(..., description="Frage für Mistral"),
    api_key: str = Query(..., description="API Key für Mistral")
):
    answer = query_mistral(question, api_key)
    return {"response": answer}


@app.get("/ask_claude")
async def ask_claude(
    question: str = Query(..., description="Frage für Anthropic Claude"),
    api_key: str = Query(..., description="API Key für Anthropic Claude")
):
    answer = query_claude(question, api_key)
    return {"response": answer}


@app.get("/ask_gemini")
async def ask_gemini(
    question: str = Query(..., description="Frage für Google Gemini"),
    api_key: str = Query(..., description="API Key für Google Gemini")
):
    answer = query_gemini(question, api_key)
    return {"response": answer}


@app.post("/consensus")
async def consensus(data: dict):
    question = data.get("question")
    answer_openai = data.get("answer_openai")
    answer_mistral = data.get("answer_mistral")
    answer_claude = data.get("answer_claude")
    answer_gemini = data.get("answer_gemini")
    best_model = data.get("best_model", "")
    consensus_model = data.get("consensus_model")
    excluded_models = data.get("excluded_models", [])
    api_keys = {
        "OpenAI": data.get("openai_key"),
        "Mistral": data.get("mistral_key"),
        "Anthropic Claude": data.get("anthropic_key"),
        "Google Gemini": data.get("gemini_key")
    }
    
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
        if not answer_gemini or not api_keys.get("Google Gemini"):
            missing.append("Google Gemini")
    
    if missing:
        raise HTTPException(status_code=400, detail="Fehlende Parameter: " + ", ".join(missing))
    
    if best_model and best_model in excluded_models:
        raise HTTPException(status_code=400, detail="Die als beste markierte Antwort darf nicht ausgeschlossen werden.")
    
    # Konsens-Antwort generieren
    consensus_answer = query_consensus(
        question, answer_openai, answer_mistral, answer_claude, answer_gemini,
        best_model, excluded_models, consensus_model, api_keys
    )
    
    # Unterschiede ermitteln – nun mit Übergabe der Konsens-Antwort
    differences = query_differences(answer_openai, answer_mistral, answer_claude, answer_gemini, consensus_answer, api_keys, differences_model=consensus_model)
    
    return {"consensus_response": consensus_answer, "differences": differences}


def is_valid(key):
    # Beispielhafte Validierung: Key gilt als valide, wenn er vorhanden ist und mehr als 10 Zeichen hat.
    return key is not None and len(key) > 10


@app.post("/check_keys")
async def check_keys(data: dict):
    try:
        openai_key = data.get("openai_key")
        mistral_key = data.get("mistral_key")
        anthropic_key = data.get("anthropic_key")
        gemini_key = data.get("gemini_key")
        
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
        
        # Google Gemini Handshake
        try:
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

@app.post("/record_click")
async def record_click(data: dict):
    model = data.get("model")
    click_type = data.get("click_type")
    if model not in click_counts or click_type not in ["best", "exclude"]:
        raise HTTPException(status_code=400, detail="Ungültige Parameter")
    click_counts[model][click_type] += 1
    return {"status": "success", "click_counts": click_counts}

@app.get("/leaderboard")
async def leaderboard():
    return click_counts