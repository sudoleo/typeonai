from __future__ import annotations

import os
import openai
import requests
from typing import Optional
from urllib.parse import quote
import google.auth
from google.auth.transport.requests import Request as GoogleAuthRequest

from app.core.config import (
    MAX_TOKENS,
    DEEP_SEARCH_MAX_TOKENS,
    REASONING_EFFORT_FOR_DEEP,
    DEEP_THINK_PROMPT,
    GEMINI_FLASH_MODEL,
    GEMINI_PRO_MODEL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_MISTRAL_MODEL,
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_GROK_MODEL,
)
from app.services.llm.base import get_system_prompt
from app.services.llm.citations import (
    make_llm_result,
    parse_anthropic_response,
    parse_gemini_response,
    parse_mistral_content,
    parse_openai_response,
    result_text,
)


def _error(provider: str, error: Exception | str):
    return make_llm_result(f"Error with {provider}: {error}", [])


def _openai_responses_call(
    *,
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    question: str,
    max_tokens: int,
    reasoning_effort: str | None = None,
    provider: str,
):
    payload = {
        "model": model,
        "instructions": system_prompt,
        "input": question,
        "tools": [{"type": "web_search"}],
        "tool_choice": "auto",
        "include": ["web_search_call.action.sources"],
        "max_output_tokens": max_tokens,
    }
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}

    resp = requests.post(
        f"{base_url.rstrip('/')}/responses",
        json=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=120,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"{resp.status_code} - {resp.text}")
    return parse_openai_response(resp.json(), provider=provider)


def _mistral_headers(api_key: str):
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _mistral_builtin_tools_unsupported(response: requests.Response) -> bool:
    if response.status_code != 400:
        return False
    try:
        data = response.json()
    except ValueError:
        return "builtin connectors" in response.text.lower()

    message = str(data.get("message") or "").lower()
    code = data.get("code")
    return code == 3004 or "builtin connectors" in message


def _google_adc_headers() -> dict:
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/generative-language"])
    credentials.refresh(GoogleAuthRequest())
    return {
        "Authorization": f"Bearer {credentials.token}",
        "Content-Type": "application/json",
    }


def _gemini_search_tool_unsupported(error_text: str) -> bool:
    text = (error_text or "").lower()
    return (
        "google_search_retrieval is not supported" in text
        or "google_search is not supported" in text
        or "search grounding" in text
        or "google_search tool" in text and "not supported" in text
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

    model_to_use = "gpt-5.5" if deep_search else (model_override or DEFAULT_OPENAI_MODEL)

    print(f"[MODEL] OpenAI -> {model_to_use} | deep_search={deep_search} | override={model_override}")

    try:
        return _openai_responses_call(
            api_key=api_key,
            base_url="https://api.openai.com/v1",
            model=model_to_use,
            system_prompt=system_prompt,
            question=question,
            max_tokens=DEEP_SEARCH_MAX_TOKENS if deep_search else MAX_TOKENS,
            reasoning_effort=REASONING_EFFORT_FOR_DEEP if deep_search else None,
            provider="openai",
        )
    except Exception as e:
        return _error("OpenAI", e)


def query_mistral(question: str, api_key: str, system_prompt: str = None, deep_search: bool = False, model_override: str = None) -> str:
    """Fragt die Mistral API zu der gegebenen Frage unter Verwendung des übergebenen API Keys ohne Limit."""
    if system_prompt is None:
        system_prompt = get_system_prompt()

    if deep_search:
        system_prompt += "\n" + DEEP_THINK_PROMPT

    # Setze max_tokens basierend auf dem deep_search Flag
    max_tokens = DEEP_SEARCH_MAX_TOKENS if deep_search else MAX_TOKENS

    try:
        model = model_override if (model_override and not deep_search) else ("mistral-large-latest" if deep_search else DEFAULT_MISTRAL_MODEL)

        print(f"[MODEL] Mistral -> {model} | deep_search={deep_search} | override={model_override}")

        payload = {
            "model": model,
            "instructions": system_prompt,
            "inputs": question,
            "tools": [{"type": "web_search"}],
            "completion_args": {
                "max_tokens": max_tokens,
                "temperature": 0.2,
            },
            "store": False,
        }
        response = requests.post(
            "https://api.mistral.ai/v1/conversations",
            json=payload,
            headers=_mistral_headers(api_key),
            timeout=120,
        )
        if _mistral_builtin_tools_unsupported(response):
            payload.pop("tools", None)
            response = requests.post(
                "https://api.mistral.ai/v1/conversations",
                json=payload,
                headers=_mistral_headers(api_key),
                timeout=120,
            )
        if response.status_code >= 400:
            raise RuntimeError(f"{response.status_code} - {response.text}")
        data = response.json()
        content = []
        for output in data.get("outputs", []) or []:
            if output.get("type") == "message.output":
                output_content = output.get("content", "")
                if isinstance(output_content, list):
                    content.extend(output_content)
                elif output_content:
                    content.append({"type": "text", "text": str(output_content)})
        if not content:
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return parse_mistral_content(content)
    except Exception as e:
        return _error("Mistral", str(e))


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
            "model": model_override if (model_override and not deep_search) else ("claude-opus-4-7" if deep_search else DEFAULT_ANTHROPIC_MODEL),
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": question}],
            "tools": [{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 5
            }]
        }

        print(f"[MODEL] Claude -> {payload['model']} | deep_search={deep_search} | override={model_override}")

        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            parsed = parse_anthropic_response(data)
            if result_text(parsed):
                return parsed
            else:
                return make_llm_result("Error: No response found in the API response.", [])
        else:
            return _error("Anthropic", f"{response.status_code} - {response.text}")
    except Exception as e:
        return _error("Anthropic", str(e))

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

    model_name = GEMINI_PRO_MODEL if deep_search else (model_override or GEMINI_FLASH_MODEL)
    print(f"[MODEL] Gemini -> {model_name} | deep_search={deep_search} | override={model_override}")

    api_key = (user_api_key or os.environ.get("DEVELOPER_GEMINI_API_KEY") or "").strip()
    eff_max = int(max_output_tokens) if max_output_tokens is not None else (4096 if deep_search else 2048)
    if question and len(question) > 12000:
        question = question[:12000] + " ... [truncated]"
    base_content = "Do not ask any questions.\n---\n" + question

    try:
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": base_content}]}],
            "tools": [{"google_search": {}}],
            "generationConfig": {
                "maxOutputTokens": eff_max,
                "temperature": 0.2,
            },
            "safetySettings": [{
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_ONLY_HIGH",
            }],
        }
        request_kwargs = {
            "json": payload,
            "timeout": 120,
        }
        if api_key:
            request_kwargs["params"] = {"key": api_key}
        else:
            request_kwargs["headers"] = _google_adc_headers()

        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{quote(model_name, safe='')}:generateContent",
            **request_kwargs,
        )
        if response.status_code >= 400 and _gemini_search_tool_unsupported(response.text):
            payload.pop("tools", None)
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{quote(model_name, safe='')}:generateContent",
                **request_kwargs,
            )
        if response.status_code >= 400:
            raise RuntimeError(f"{response.status_code} - {response.text}")
        data = response.json()
        parsed = parse_gemini_response(data)
        if result_text(parsed):
            return parsed
        cand = (data.get("candidates") or [{}])[0]
        fr = cand.get("finishReason")

        frs = str(fr)
        if frs in ("2", "MAX_TOKENS", "FinishReason.MAX_TOKENS"):
            return make_llm_result("Error with Gemini: hit max tokens before producing text. Raise max_output_tokens or trim input.", [])
        if frs in ("3", "SAFETY", "FinishReason.SAFETY"):
            return make_llm_result("Error with Gemini: response was blocked by safety filters.", [])
        if frs in ("4", "RECITATION", "FinishReason.RECITATION"):
            return make_llm_result("Error with Gemini: response suppressed by recitation policy.", [])
        return make_llm_result(f"Error with Gemini: empty response payload (finish_reason={frs}).", [])
    except Exception as e:
        return _error("Gemini", e)

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
        model_to_use = "deepseek-v4-pro" if deep_search else (model_override or DEFAULT_DEEPSEEK_MODEL)
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
        model_to_use = "grok-4.3" if deep_search else (model_override or DEFAULT_GROK_MODEL)

        print(f"[MODEL] Grok -> {model_to_use} | deep_search={deep_search} | override={model_override}")

        return _openai_responses_call(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
            model=model_to_use,
            system_prompt=system_prompt,
            question=question,
            max_tokens=max_tokens,
            reasoning_effort=REASONING_EFFORT_FOR_DEEP if deep_search else None,
            provider="grok",
        )
    except Exception as e:
        return _error("Grok", str(e))

