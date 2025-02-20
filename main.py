from fastapi import FastAPI, Query, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import openai
import requests
import os
from mistralai import Mistral
from dotenv import load_dotenv
import google.generativeai as genai

# Lade .env Datei
load_dotenv()

app = FastAPI()

# Templates und statische Dateien einbinden
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# API-Schlüssel sicher aus Umgebungsvariablen abrufen
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

openai.api_key = OPENAI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

def query_openai(question: str) -> str:
    """Fragt OpenAI (GPT-4) zu der gegebenen Frage mit kurzer Antwort-Anweisung."""
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": question}
            ],
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Fehler bei OpenAI: {str(e)}"

def query_mistral(question: str) -> str:
    """Fragt die Mistral API zu der gegebenen Frage."""
    try:
        client = Mistral(api_key=MISTRAL_API_KEY)
        model = "mistral-large-latest"
        response = client.chat.complete(
            model=model,
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": question}
            ],
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Fehler bei Mistral: {str(e)}"

def query_claude(question: str) -> str:
    """Fragt die Anthropic API (Claude) zu der gegebenen Frage mit kurzer Antwort-Anweisung."""
    try:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        payload = {
            "model": "claude-3-5-sonnet-20241022",  # ggf. anpassen
            "max_tokens": 500,
            "system": "",
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

def query_gemini(question: str) -> str:
    """Fragt Google Gemini zu der gegebenen Frage."""
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(question)
        return response.text.strip()
    except Exception as e:
        return f"Fehler bei Google Gemini: {str(e)}"

def query_consensus(question: str, answer_openai: str, answer_mistral: str, answer_claude: str, answer_gemini: str, best_model: str, excluded_models: list, consensus_model: str) -> str:
    """
    Nutzt ein frei wählbares Modell (consensus_model), um die Antworten der Modelle
    (außer der ausgeschlossenen) mithilfe einer Chain-of-Thought-Logik zu einem
    konsistenten, prägnanten Konsens zusammenzufassen.
    Falls eine beste Antwort markiert wurde, wird diese stärker gewichtet.
    """
    prompt_parts = [f"Die Frage lautet: {question}\n\n"]
    if "OpenAI" not in excluded_models:
        prompt_parts.append(f"Antwort von OpenAI: {answer_openai}\n\n")
    if "Mistral" not in excluded_models:
        prompt_parts.append(f"Antwort von Mistral: {answer_mistral}\n\n")
    if "Anthropic Claude" not in excluded_models:
        prompt_parts.append(f"Antwort von Anthropic Claude: {answer_claude}\n\n")
    if "Google Gemini" not in excluded_models:
        prompt_parts.append(f"Antwort von Google Gemini: {answer_gemini}\n\n")
    
    # Wenn ein best_model markiert wurde, diesen Satz einfügen, sonst eine Alternative:
    if best_model:
        prompt_parts.append(
            f"Der Nutzer hat die Antwort von {best_model} als die beste markiert. "
            "Bitte nutze eine Chain-of-Thought-Logik: Überlege dir zunächst in kurzen Stichpunkten, welche Gemeinsamkeiten und Unterschiede in den übergebenen Antworten bestehen. "
            "Fasse danach die wichtigsten Punkte zusammen und formuliere auf Basis dieser Überlegungen einen konsistenten, prägnanten Konsens als finale Antwort. "
            "Gib als Antwort ausschließlich den finalen Konsens aus, ohne die Zwischenschritte darzustellen. "
            "Nutze ausschließlich die übergebenen Informationen."
        )
    else:
        prompt_parts.append(
            "Es wurde keine Antwort als beste markiert. "
            "Bitte nutze eine Chain-of-Thought-Logik: Überlege in kurzen Stichpunkten, welche Gemeinsamkeiten und Unterschiede in den übergebenen Antworten bestehen. "
            "Fasse die wichtigsten Punkte zusammen und formuliere auf Basis dieser Überlegungen einen konsistenten, prägnanten Konsens als finale Antwort. "
            "Gib als Antwort ausschließlich den finalen Konsens aus, ohne die Zwischenschritte darzustellen. "
            "Nutze ausschließlich die übergebenen Informationen."
        )

    consensus_prompt = "".join(prompt_parts)
    
    try:
        if consensus_model == "OpenAI":
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Bitte antworte so kurz und präzise wie möglich."},
                    {"role": "user", "content": consensus_prompt}
                ],
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        elif consensus_model == "Mistral":
            client = Mistral(api_key=MISTRAL_API_KEY)
            model = "mistral-large-latest"  # ggf. anpassen
            response = client.chat.complete(
                model=model,
                messages=[
                    {"role": "system", "content": "Bitte antworte so kurz und präzise wie möglich."},
                    {"role": "user", "content": consensus_prompt}
                ],
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        elif consensus_model == "Anthropic Claude":
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": ANTHROPIC_API_KEY,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            payload = {
                "model": "claude-3-5-sonnet-20241022",  # ggf. anpassen
                "max_tokens": 300,
                "system": "Bitte antworte so kurz und präzise wie möglich.",
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
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(consensus_prompt)
            return response.text.strip()
        else:
            return "Ungültiges Konsensmodell ausgewählt."
    except Exception as e:
        return f"Fehler beim Konsens: {str(e)}"


@app.post("/consensus")
async def consensus(data: dict):
    question = data.get("question")
    answer_openai = data.get("answer_openai")
    answer_mistral = data.get("answer_mistral")
    answer_claude = data.get("answer_claude")
    answer_gemini = data.get("answer_gemini")
    best_model = data.get("best_model", "")  # Optional – Default: leer
    consensus_model = data.get("consensus_model")
    excluded_models = data.get("excluded_models", [])
    if not all([question, answer_openai, answer_mistral, answer_claude, answer_gemini, consensus_model]):
        raise HTTPException(status_code=400, detail="Fehlende Parameter")
    # Entferne die Prüfung, dass best_model nicht ausgeschlossen sein darf – da es optional ist.
    consensus_answer = query_consensus(question, answer_openai, answer_mistral, answer_claude, answer_gemini, best_model, excluded_models, consensus_model)
    return {"consensus_response": consensus_answer}




@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/ask_openai")
async def ask_openai(question: str = Query(..., description="Frage für OpenAI")):
    answer = query_openai(question)
    return {"response": answer}

@app.get("/ask_mistral")
async def ask_mistral(question: str = Query(..., description="Frage für Mistral")):
    answer = query_mistral(question)
    return {"response": answer}

@app.get("/ask_claude")
async def ask_claude(question: str = Query(..., description="Frage für Anthropic Claude")):
    answer = query_claude(question)
    return {"response": answer}

@app.get("/ask_gemini")
async def ask_gemini(question: str = Query(..., description="Frage für Google Gemini")):
    answer = query_gemini(question)
    return {"response": answer}

@app.post("/consensus")
async def consensus(data: dict):
    question = data.get("question")
    answer_openai = data.get("answer_openai")
    answer_mistral = data.get("answer_mistral")
    answer_claude = data.get("answer_claude")
    answer_gemini = data.get("answer_gemini")
    best_model = data.get("best_model")
    consensus_model = data.get("consensus_model")
    excluded_models = data.get("excluded_models", [])
    if not all([question, answer_openai, answer_mistral, answer_claude, answer_gemini, best_model, consensus_model]):
        raise HTTPException(status_code=400, detail="Fehlende Parameter")
    if best_model in excluded_models:
        raise HTTPException(status_code=400, detail="Die als beste markierte Antwort darf nicht ausgeschlossen werden.")
    consensus_answer = query_consensus(question, answer_openai, answer_mistral, answer_claude, answer_gemini, best_model, excluded_models, consensus_model)
    return {"consensus_response": consensus_answer}

def is_valid(key):
    # Dummy-Prüfung: Key gilt als valide, wenn er vorhanden ist und mehr als 10 Zeichen hat.
    return key is not None and len(key) > 10

@app.post("/check_keys")
async def check_keys(data: dict):
    openai_key = data.get("openai_key")
    mistral_key = data.get("mistral_key")
    anthropic_key = data.get("anthropic_key")
    gemini_key = data.get("gemini_key")
    
    results = {
        "OpenAI": "valid" if is_valid(openai_key) else "invalid",
        "Mistral": "valid" if is_valid(mistral_key) else "invalid",
        "Anthropic Claude": "valid" if is_valid(anthropic_key) else "invalid",
        "Google Gemini": "valid" if is_valid(gemini_key) else "invalid"
    }
    
    return {"results": results}
