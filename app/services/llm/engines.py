import os
import openai
import requests
from typing import Optional
import google.generativeai as genai
from mistralai import Mistral

from app.core.config import MAX_TOKENS, DEEP_SEARCH_MAX_TOKENS, REASONING_EFFORT_FOR_DEEP, DEEP_THINK_PROMPT
from app.services.llm.base import get_system_prompt

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
    model_to_use = "gpt-5.5" if deep_search else (model_override or "gpt-5.4-mini")

    print(f"[MODEL] OpenAI -> {model_to_use} | deep_search={deep_search} | override={model_override}")

    user_msg = {"role": "user", "content": question}

    # ===== Single Chat Completions Call =====
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        kwargs = {
            "model": model_to_use,
            "messages": messages,
        }
        
        if "gpt-5" in model_to_use or "o" in model_to_use:
             kwargs["max_completion_tokens"] = DEEP_SEARCH_MAX_TOKENS if deep_search else MAX_TOKENS
        else:
             kwargs["max_tokens"] = DEEP_SEARCH_MAX_TOKENS if deep_search else MAX_TOKENS
             
        if deep_search and ("gpt-5.5" in model_to_use or "gpt-5" in model_to_use):
             kwargs["reasoning_effort"] = REASONING_EFFORT_FOR_DEEP
             
        cmpl = client.chat.completions.create(**kwargs)
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
        model = model_override if (model_override and not deep_search) else ("mistral-large-latest" if deep_search else "mistral-small-latest")

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
            "model": model_override if (model_override and not deep_search) else ("claude-opus-4-7" if deep_search else "claude-haiku-4-5"),
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
    model_name = "gemini-3-pro-preview" if deep_search else (model_override or "gemini-3-flash-preview")

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
        model_to_use = "deepseek-v4-pro" if deep_search else (model_override or "deepseek-v4-flash")
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
        model_to_use = "grok-4.3" if deep_search else (model_override or "grok-4.20-non-reasoning")

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
