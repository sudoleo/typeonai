from __future__ import annotations

import os
import logging
import openai
import requests
from typing import Optional
from urllib.parse import quote
import google.auth
from google.auth.transport.requests import Request as GoogleAuthRequest

import app.core.config as cfg
from app.core.config import (
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
    error_text = str(error)
    logging.error("Provider request failed for %s: %s", provider, error_text)
    return {
        "text": "",
        "sources": [],
        "error": f"{provider} could not complete this request. Please try again later.",
        "error_detail": error_text,
    }


def _openai_responses_payload(
    *,
    model: str,
    system_prompt: str,
    question: str,
    max_tokens: int,
    request_config: dict | None = None,
) -> dict:
    payload = {
        "model": model,
        "instructions": system_prompt,
        "input": question,
        "tools": [{"type": "web_search"}],
        "tool_choice": "auto",
        "include": ["web_search_call.action.sources"],
        "max_output_tokens": max_tokens,
    }
    if request_config:
        payload.update(request_config)
    return payload


def _openai_responses_call(
    *,
    api_key: str,
    base_url: str,
    payload: dict,
    provider: str,
):

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


def _merge_nested_config(payload: dict, config: dict | None):
    if not config:
        return
    for key, value in config.items():
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            _merge_nested_config(payload[key], value)
        else:
            payload[key] = value


def build_provider_payload(
    provider: str,
    *,
    question: str = "dry run",
    system_prompt: str | None = None,
    model_override: str | None = None,
    deep_search: bool = False,
    max_output_tokens: int | None = None,
) -> dict:
    if system_prompt is None:
        system_prompt = get_system_prompt()
    if deep_search:
        system_prompt += "\n" + DEEP_THINK_PROMPT

    max_tokens = int(max_output_tokens) if max_output_tokens is not None else cfg.get_output_token_limit(True, deep_search)
    provider_key = provider.lower()

    if provider_key == "openai":
        if deep_search:
            api_model = "gpt-5.5"
            model_config = None
            request_config = {"reasoning": {"effort": REASONING_EFFORT_FOR_DEEP}}
            internal_model = "deep_search:gpt-5.5"
        else:
            internal_model = model_override or DEFAULT_OPENAI_MODEL
            api_model, model_config = cfg.resolve_api_model(model_override, DEFAULT_OPENAI_MODEL, "openai")
            request_config = model_config.low_config if model_config.is_low_reasoning else None
        payload = _openai_responses_payload(
            model=api_model,
            system_prompt=system_prompt,
            question=question,
            max_tokens=max_tokens,
            request_config=request_config,
        )
        return {
            "provider": "openai",
            "endpoint": "responses",
            "internal_model": internal_model,
            "api_model": api_model,
            "is_low_reasoning": bool(model_config and model_config.is_low_reasoning) if not deep_search else True,
            "payload": payload,
        }

    if provider_key == "mistral":
        if deep_search:
            api_model = "mistral-large-latest"
            internal_model = "deep_search:mistral-large-latest"
        else:
            internal_model = model_override or DEFAULT_MISTRAL_MODEL
            api_model, _ = cfg.resolve_api_model(model_override, DEFAULT_MISTRAL_MODEL, "mistral")
        return {
            "provider": "mistral",
            "endpoint": "conversations",
            "internal_model": internal_model,
            "api_model": api_model,
            "is_low_reasoning": False,
            "payload": {
                "model": api_model,
                "instructions": system_prompt,
                "inputs": question,
                "tools": [{"type": "web_search"}],
                "completion_args": {
                    "max_tokens": max_tokens,
                    "temperature": 0.2,
                },
                "store": False,
            },
        }

    if provider_key == "anthropic":
        if deep_search:
            api_model = "claude-opus-4-7"
            model_config = None
            internal_model = "deep_search:claude-opus-4-7"
        else:
            internal_model = model_override or DEFAULT_ANTHROPIC_MODEL
            api_model, model_config = cfg.resolve_api_model(model_override, DEFAULT_ANTHROPIC_MODEL, "anthropic")
        payload = {
            "model": api_model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": question}],
            "tools": [{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 5,
            }],
        }
        if model_config and model_config.is_low_reasoning:
            _merge_nested_config(payload, model_config.low_config)
        return {
            "provider": "anthropic",
            "endpoint": "messages",
            "internal_model": internal_model,
            "api_model": api_model,
            "is_low_reasoning": bool(model_config and model_config.is_low_reasoning),
            "payload": payload,
        }

    if provider_key == "gemini":
        if deep_search:
            api_model = GEMINI_PRO_MODEL
            model_config = None
            internal_model = f"deep_search:{GEMINI_PRO_MODEL}"
        else:
            internal_model = model_override or GEMINI_FLASH_MODEL
            api_model, model_config = cfg.resolve_api_model(model_override, GEMINI_FLASH_MODEL, "gemini")
        if question and len(question) > 12000:
            question = question[:12000] + " ... [truncated]"
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": "Do not ask any questions.\n---\n" + question}]}],
            "tools": [{"google_search": {}}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.2,
            },
            "safetySettings": [{
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_ONLY_HIGH",
            }],
        }
        if model_config and model_config.is_low_reasoning:
            _merge_nested_config(payload, model_config.low_config)
        return {
            "provider": "gemini",
            "endpoint": "generateContent",
            "internal_model": internal_model,
            "api_model": api_model,
            "is_low_reasoning": bool(model_config and model_config.is_low_reasoning),
            "payload": payload,
        }

    if provider_key == "deepseek":
        if deep_search:
            api_model = "deepseek-v4-pro"
            internal_model = "deep_search:deepseek-v4-pro"
        else:
            internal_model = model_override or DEFAULT_DEEPSEEK_MODEL
            api_model, _ = cfg.resolve_api_model(model_override, DEFAULT_DEEPSEEK_MODEL, "deepseek")
        return {
            "provider": "deepseek",
            "endpoint": "chat.completions",
            "internal_model": internal_model,
            "api_model": api_model,
            "is_low_reasoning": False,
            "payload": {
                "model": api_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                "stream": False,
                "max_tokens": max_tokens,
            },
        }

    if provider_key == "grok":
        if deep_search:
            api_model = "grok-4.3"
            model_config = None
            request_config = {"reasoning": {"effort": REASONING_EFFORT_FOR_DEEP}}
            internal_model = "deep_search:grok-4.3"
        else:
            internal_model = model_override or DEFAULT_GROK_MODEL
            api_model, model_config = cfg.resolve_api_model(model_override, DEFAULT_GROK_MODEL, "grok")
            request_config = model_config.low_config if model_config.is_low_reasoning else None
        payload = _openai_responses_payload(
            model=api_model,
            system_prompt=system_prompt,
            question=question,
            max_tokens=max_tokens,
            request_config=request_config,
        )
        return {
            "provider": "grok",
            "endpoint": "responses",
            "internal_model": internal_model,
            "api_model": api_model,
            "is_low_reasoning": bool(model_config and model_config.is_low_reasoning) if not deep_search else True,
            "payload": payload,
        }

    raise ValueError(f"Unsupported provider for payload dry-run: {provider}")


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
    model_override: str = None,
    max_output_tokens: Optional[int] = None,
) -> str:
    max_tokens = int(max_output_tokens) if max_output_tokens is not None else cfg.get_output_token_limit(True, deep_search)
    request_data = build_provider_payload(
        "openai",
        question=question,
        system_prompt=system_prompt,
        model_override=model_override,
        deep_search=deep_search,
        max_output_tokens=max_tokens,
    )

    print(f"[MODEL] OpenAI -> {request_data['api_model']} | deep_search={deep_search} | override={model_override}")

    try:
        return _openai_responses_call(
            api_key=api_key,
            base_url="https://api.openai.com/v1",
            payload=request_data["payload"],
            provider="openai",
        )
    except Exception as e:
        return _error("OpenAI", e)


def query_mistral(
    question: str,
    api_key: str,
    system_prompt: str = None,
    deep_search: bool = False,
    model_override: str = None,
    max_output_tokens: Optional[int] = None,
) -> str:
    """Fragt die Mistral API zu der gegebenen Frage unter Verwendung des übergebenen API Keys ohne Limit."""
    # Setze max_tokens basierend auf dem deep_search Flag
    max_tokens = int(max_output_tokens) if max_output_tokens is not None else cfg.get_output_token_limit(True, deep_search)

    try:
        request_data = build_provider_payload(
            "mistral",
            question=question,
            system_prompt=system_prompt,
            model_override=model_override,
            deep_search=deep_search,
            max_output_tokens=max_tokens,
        )

        print(f"[MODEL] Mistral -> {request_data['api_model']} | deep_search={deep_search} | override={model_override}")

        payload = request_data["payload"]
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


def query_claude(
    question: str,
    api_key: str,
    system_prompt: str = None,
    deep_search: bool = False,
    model_override: str = None,
    max_output_tokens: Optional[int] = None,
) -> str:
    """Fragt die Anthropic API (Claude) zu der gegebenen Frage unter Verwendung des übergebenen API Keys ohne Limit.
       Da die Anthropic API ein Token-Limit erwartet, setzen wir einen sehr hohen Wert ein."""
    # Setze max_tokens basierend auf dem deep_search Flag
    max_tokens = int(max_output_tokens) if max_output_tokens is not None else cfg.get_output_token_limit(True, deep_search)

    try:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        request_data = build_provider_payload(
            "anthropic",
            question=question,
            system_prompt=system_prompt,
            model_override=model_override,
            deep_search=deep_search,
            max_output_tokens=max_tokens,
        )
        payload = request_data["payload"]

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
    max_tokens = int(max_output_tokens) if max_output_tokens is not None else cfg.get_output_token_limit(True, deep_search)
    request_data = build_provider_payload(
        "gemini",
        question=question,
        system_prompt=system_prompt,
        model_override=model_override,
        deep_search=deep_search,
        max_output_tokens=max_tokens,
    )
    model_name = request_data["api_model"]
    print(f"[MODEL] Gemini -> {model_name} | deep_search={deep_search} | override={model_override}")

    api_key = (user_api_key or os.environ.get("DEVELOPER_GEMINI_API_KEY") or "").strip()

    try:
        payload = request_data["payload"]
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

def query_deepseek(
    question: str,
    api_key: str,
    system_prompt: str = None,
    deep_search: bool = False,
    model_override: str = None,
    max_output_tokens: Optional[int] = None,
) -> str:
    """Fragt DeepSeek zu der gegebenen Frage unter Verwendung des übergebenen API Keys."""
    # Setze max_tokens basierend auf dem deep_search Flag
    max_tokens = int(max_output_tokens) if max_output_tokens is not None else cfg.get_output_token_limit(True, deep_search)

    try:
        client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        request_data = build_provider_payload(
            "deepseek",
            question=question,
            system_prompt=system_prompt,
            model_override=model_override,
            deep_search=deep_search,
            max_output_tokens=max_tokens,
        )
        print(f"[MODEL] DeepSeek -> {request_data['api_model']} | deep_search={deep_search} | override={model_override}")
        response = client.chat.completions.create(**request_data["payload"])
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error with DeepSeek: {str(e)}"
    
def query_grok(
    question: str,
    api_key: str,
    system_prompt: str = None,
    deep_search: bool = False,
    model_override: str = None,
    max_output_tokens: Optional[int] = None,
) -> str:
    """Fragt die Grok API zu der gegebenen Frage unter Verwendung des übergebenen API Keys."""
    # Setze max_tokens basierend auf dem deep_search Flag
    max_tokens = int(max_output_tokens) if max_output_tokens is not None else cfg.get_output_token_limit(True, deep_search)

    try:
        request_data = build_provider_payload(
            "grok",
            question=question,
            system_prompt=system_prompt,
            model_override=model_override,
            deep_search=deep_search,
            max_output_tokens=max_tokens,
        )

        print(f"[MODEL] Grok -> {request_data['api_model']} | deep_search={deep_search} | override={model_override}")

        return _openai_responses_call(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
            payload=request_data["payload"],
            provider="grok",
        )
    except Exception as e:
        return _error("Grok", str(e))

