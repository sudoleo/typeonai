from __future__ import annotations

import os
import re
import json
import time
import difflib
import logging
import random
import requests
import openai
import google.auth
from google.auth.transport.requests import Request as GoogleAuthRequest
from urllib.parse import quote

import app.core.config as cfg
from app.core.config import (
    GEMINI_FLASH_MODEL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_MISTRAL_MODEL,
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_GROK_MODEL,
)
from app.services.llm.engines import _merge_nested_config
from app.services.llm.mock_llm import mock_engine_stream, mock_engine_text, mock_llm_enabled

CANONICAL_MODEL_NAMES = {
    "openai": "OpenAI",
    "gpt": "OpenAI",
    "chatgpt": "OpenAI",
    "mistral": "Mistral",
    "anthropic": "Anthropic",
    "claude": "Anthropic",
    "gemini": "Gemini",
    "deepseek": "DeepSeek",
    "grok": "Grok",
}

MAX_SOURCES_PER_EXPERT = 5
MAX_SOURCE_FIELD_CHARS = 180


def _google_adc_headers() -> dict:
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/generative-language"])
    credentials.refresh(GoogleAuthRequest())
    return {
        "Authorization": f"Bearer {credentials.token}",
        "Content-Type": "application/json",
    }


ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"

_PROVIDER_KEY_NAMES = {
    "openai": "OpenAI",
    "mistral": "Mistral",
    "anthropic": "Anthropic",
    "gemini": "Gemini",
    "deepseek": "DeepSeek",
    "grok": "Grok",
}

_OPENAI_COMPAT_BASE_URLS = {
    "deepseek": "https://api.deepseek.com",
    "grok": "https://api.x.ai/v1",
}


def _gemini_engine_key(api_keys: dict) -> str | None:
    return api_keys.get("Gemini") or os.environ.get("DEVELOPER_GEMINI_API_KEY")


def _gemini_engine_payload(
    model_ref: str,
    system: str,
    prompt: str,
    max_tokens: int,
    temperature: float | None = None,
    json_mode: bool = False,
    effort: str | None = None,
) -> tuple[str, dict]:
    """Baut den generateContent-Payload für Consensus-/Differences-Calls.

    Bewusst NICHT über build_provider_payload: dessen Gemini-Pfad kappt die
    Frage bei 12k Zeichen und hängt Chat-Instruktionen an — beides falsch für
    die langen Engine-Prompts. Frontier-Low-Mapping (interne ID -> api_model
    + low_config) läuft über resolve_api_model. effort ("low") kappt das
    Thinking-Budget (thinkingLevel) — genutzt für Judge-Calls, deren Task
    kein tiefes Denken braucht."""
    api_model, model_config = cfg.resolve_api_model(model_ref, GEMINI_FLASH_MODEL, "gemini")
    payload: dict = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": int(max_tokens)},
        "safetySettings": [{
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_ONLY_HIGH",
        }],
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    if temperature is not None:
        payload["generationConfig"]["temperature"] = temperature
    if json_mode:
        payload["generationConfig"]["responseMimeType"] = "application/json"
    if model_config and model_config.is_low_reasoning:
        _merge_nested_config(payload, model_config.low_config)
    if effort:
        _merge_nested_config(payload, {"generationConfig": {"thinkingConfig": {"thinkingLevel": effort}}})
    return api_model, payload


def _gemini_generate_content(api_model: str, payload: dict, api_key: str | None) -> str:
    request_kwargs = {"json": payload, "timeout": 120}
    if api_key and api_key.strip():
        request_kwargs["params"] = {"key": api_key.strip()}
    else:
        request_kwargs["headers"] = _google_adc_headers()

    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{quote(api_model, safe='')}:generateContent",
        **request_kwargs,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Gemini: {response.status_code} - {response.text}")
    data = response.json()
    text_parts = []
    finish_reason = None
    for candidate in data.get("candidates", []) or []:
        if candidate.get("finishReason"):
            finish_reason = candidate["finishReason"]
        content = candidate.get("content") or {}
        for part in content.get("parts", []) or []:
            if part.get("text"):
                text_parts.append(part["text"])
    text = "\n".join(text_parts).strip()
    if not text:
        raise RuntimeError(f"Gemini: empty response payload (finish_reason={finish_reason}).")
    return text


def _coerce_message_text(content) -> str:
    """Mistral liefert message.content je nach Modell als String oder als
    Liste von Chunks (z.B. thinking- und text-Chunks bei Magistral bzw.
    json_mode). Beides zu reinem Text zusammenfuehren."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for chunk in content:
            if isinstance(chunk, str):
                parts.append(chunk)
            elif isinstance(chunk, dict):
                text = chunk.get("text")
                if isinstance(text, str) and chunk.get("type") in (None, "text"):
                    parts.append(text)
        return "".join(parts).strip()
    return str(content or "").strip()


def _mistral_chat_complete(
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float | None = None,
    json_mode: bool = False,
    reasoning_effort: str | None = None,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if model in cfg.MISTRAL_REASONING_MODELS:
        payload["reasoning_effort"] = reasoning_effort or "high"

    response = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        json=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=120,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"{response.status_code} - {response.text}")
    data = response.json()
    message = (data.get("choices") or [{}])[0].get("message") or {}
    return _coerce_message_text(message.get("content"))


def normalize_model_name(model_name: str) -> str:
    key = str(model_name or "").strip()
    if key.endswith("-Pro"):
        key = key[:-4]
    return CANONICAL_MODEL_NAMES.get(key.lower(), key)


def resolve_consensus_engine_model(consensus_model: str):
    """Liefert die Provider-/API-Modell-Konfiguration fuer Consensus-Werte.

    Unterstuetzt die historischen Alias-Werte (z. B. ``Gemini-Pro``) und direkte
    interne Modell-IDs aus ``MODEL_CONFIGS``.
    """
    config = cfg.get_consensus_model_config(consensus_model)
    if not config or not config.provider:
        return None
    return config


# OpenAI-Reasoning-Modelle: gpt-5-Familie und o-Serie (o1/o3/o4) als Wortanfang.
# Bewusst kein Substring-Match - das frühere '"o" in model' traf fast jedes
# Modell (z. B. "gpt-4o") und unterdrückte darüber die Temperatur für alle
# OpenAI-Engines (siehe _effective_temperature).
_OPENAI_REASONING_MODEL_RE = re.compile(r"^(gpt-5|o[134])([.\-]|$)")


def _openai_token_param(model_to_use: str) -> str:
    if _OPENAI_REASONING_MODEL_RE.match(str(model_to_use or "")):
        return "max_completion_tokens"
    return "max_tokens"


# ---------------------------------------------------------------------------
# Einheitlicher Engine-Dispatch: eine Stelle, die für alle sechs Provider den
# Consensus-/Differences-Call ausführt (nicht-streamend und streamend).
# Fehler schlagen als Exception nach außen; die Aufrufer entscheiden über
# Fehlertexte bzw. Retries.
# ---------------------------------------------------------------------------

class _InvalidEngineError(Exception):
    pass


def _effective_temperature(provider: str, api_model: str, temperature: float | None) -> float | None:
    if temperature is None:
        return None
    # Reasoning-Modelle akzeptieren keine (oder ignorieren die) Temperatur.
    if provider == "openai" and _openai_token_param(api_model) == "max_completion_tokens":
        return None
    if provider == "mistral" and api_model in cfg.MISTRAL_REASONING_MODELS:
        return None
    return temperature


def _resolve_engine(engine_model: str) -> tuple[str, str, str] | None:
    """Löst einen Engine-Wert (Alias wie "OpenAI-Pro" oder interne Modell-ID)
    zu (provider, api_model, gemini_model_ref) auf.

    gemini_model_ref ist der Wert für resolve_api_model: bei internen IDs die
    ID selbst (Frontier-Low mappt dort auf api_model + low_config), bei
    Aliassen direkt das API-Modell."""
    config = resolve_consensus_engine_model(engine_model)
    if not config or not config.provider:
        return None
    if engine_model in cfg.CONSENSUS_ENGINE_ALIASES:
        model_ref = config.api_model
    else:
        model_ref = config.internal_id
    return config.provider, config.api_model, model_ref


def _call_engine_text(
    provider: str,
    api_model: str,
    model_ref: str,
    api_keys: dict,
    *,
    system: str,
    prompt: str,
    max_tokens: int,
    temperature: float | None = None,
    json_mode: bool = False,
    effort: str | None = None,
) -> str:
    if mock_llm_enabled():
        # E2E-Suite: deterministische Engine-Antwort; Prompt-Bau, Parsing,
        # Verifikation und Agreement-Score laufen weiterhin echt.
        return mock_engine_text(prompt=prompt, json_mode=json_mode)

    max_tokens = int(max_tokens)
    temperature = _effective_temperature(provider, api_model, temperature)

    if provider in ("openai", "deepseek", "grok"):
        base_url = _OPENAI_COMPAT_BASE_URLS.get(provider)
        api_key = api_keys.get(_PROVIDER_KEY_NAMES[provider])
        client = openai.OpenAI(api_key=api_key, base_url=base_url) if base_url else openai.OpenAI(api_key=api_key)
        token_param = _openai_token_param(api_model) if provider == "openai" else "max_tokens"
        kwargs = {
            "model": api_model,
            "messages": [
                {"role": "system", "content": system or " "},
                {"role": "user", "content": prompt},
            ],
            token_param: max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if effort and provider == "openai" and token_param == "max_completion_tokens":
            # Nur echte OpenAI-Reasoning-Modelle kennen reasoning_effort;
            # DeepSeek/Grok (OpenAI-kompatibel) lehnen den Parameter ab.
            kwargs["reasoning_effort"] = effort
        response = client.chat.completions.create(**kwargs)
        return (response.choices[0].message.content or "").strip()

    if provider == "mistral":
        return _mistral_chat_complete(
            api_keys.get("Mistral"),
            api_model,
            messages=[
                {"role": "system", "content": system or ""},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=json_mode,
            reasoning_effort=effort,
        )

    if provider == "anthropic":
        payload = {
            "model": api_model,
            "max_tokens": max_tokens,
            "system": system or "",
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            payload["temperature"] = temperature
        prefix = ""
        if json_mode:
            # Prefill erzwingt den JSON-Anfang; das "{" gehört zum Ergebnis.
            payload["messages"].append({"role": "assistant", "content": "{"})
            prefix = "{"
        response = requests.post(
            ANTHROPIC_MESSAGES_URL,
            json=payload,
            headers={
                "x-api-key": api_keys.get("Anthropic"),
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            },
            timeout=120,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Anthropic: {response.status_code} - {response.text}")
        data = response.json()
        if not (data.get("content") and isinstance(data["content"], list)):
            raise RuntimeError("Anthropic: empty response payload.")
        return (prefix + (data["content"][0].get("text") or "")).strip()

    if provider == "gemini":
        gemini_model, payload = _gemini_engine_payload(
            model_ref, system, prompt, max_tokens,
            temperature=temperature, json_mode=json_mode, effort=effort,
        )
        return _gemini_generate_content(gemini_model, payload, _gemini_engine_key(api_keys))

    raise _InvalidEngineError(f"Unsupported engine provider: {provider}")


def _stream_engine_text(
    provider: str,
    api_model: str,
    model_ref: str,
    api_keys: dict,
    *,
    system: str,
    prompt: str,
    max_tokens: int,
    temperature: float | None = None,
    json_mode: bool = False,
    effort: str | None = None,
):
    """Streamt Engine-Events: {"type": "delta", "text": ...} für Antworttext
    und {"type": "reasoning"} als Fortschrittsmarker, solange ein
    Reasoning-Modell noch denkt (hält SSE-Verbindungen aktiv und speist den
    "Reasoning"-Indikator im Frontend)."""
    if mock_llm_enabled():
        # E2E-Suite: siehe _call_engine_text.
        for text in mock_engine_stream(prompt=prompt, json_mode=json_mode):
            yield {"type": "delta", "text": text}
        return

    from app.services.llm.streaming import (
        stream_anthropic_text,
        stream_chat_completion_text,
        stream_gemini_payload_text,
        stream_mistral_chat_text,
    )

    max_tokens = int(max_tokens)
    temperature = _effective_temperature(provider, api_model, temperature)
    response_format = {"type": "json_object"} if json_mode else None

    if provider in ("openai", "deepseek", "grok"):
        token_param = _openai_token_param(api_model) if provider == "openai" else "max_tokens"
        yield from stream_chat_completion_text(
            api_key=api_keys.get(_PROVIDER_KEY_NAMES[provider]),
            base_url=_OPENAI_COMPAT_BASE_URLS.get(provider),
            model=api_model,
            messages=[
                {"role": "system", "content": system or " "},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            token_param=token_param,
            temperature=temperature,
            response_format=response_format,
            # Nur echte OpenAI-Reasoning-Modelle kennen reasoning_effort.
            reasoning_effort=effort if (provider == "openai" and token_param == "max_completion_tokens") else None,
        )
    elif provider == "mistral":
        yield from stream_mistral_chat_text(
            api_key=api_keys.get("Mistral"),
            model=api_model,
            messages=[
                {"role": "system", "content": system or ""},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
            reasoning_effort=effort,
        )
    elif provider == "anthropic":
        yield from stream_anthropic_text(
            api_key=api_keys.get("Anthropic"),
            model=api_model,
            system=system or "",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            assistant_prefill="{" if json_mode else None,
        )
    elif provider == "gemini":
        gemini_model, payload = _gemini_engine_payload(
            model_ref, system, prompt, max_tokens,
            temperature=temperature, json_mode=json_mode, effort=effort,
        )
        yield from stream_gemini_payload_text(
            api_model=gemini_model,
            payload=payload,
            api_key=_gemini_engine_key(api_keys),
        )
    else:
        raise _InvalidEngineError(f"Unsupported engine provider: {provider}")


def normalize_excluded_models(excluded_models) -> set:
    if not isinstance(excluded_models, (list, tuple, set)):
        return set()
    return {normalize_model_name(model) for model in excluded_models if model}


def _clip(value, limit=MAX_SOURCE_FIELD_CHARS):
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _sources_for_model(model_sources, model_name):
    if not isinstance(model_sources, dict):
        return []

    normalized_target = normalize_model_name(model_name)
    for key, sources in model_sources.items():
        if normalize_model_name(key) == normalized_target and isinstance(sources, list):
            return sources
    return []


def _format_sources_for_prompt(model_name, model_sources):
    sources = _sources_for_model(model_sources, model_name)
    if not sources:
        return ""

    lines = []
    seen = set()
    for source in sources:
        if not isinstance(source, dict):
            continue
        title = _clip(source.get("title") or source.get("url") or "Source")
        url = _clip(source.get("url") or "")
        source_id = _clip(source.get("id") or "")
        key = (url or title).lower()
        if not key or key in seen:
            continue
        seen.add(key)

        prefix = f"[{source_id}] " if source_id else ""
        suffix = f" - {url}" if url and url != title else ""
        lines.append(f"- {prefix}{title}{suffix}")
        if len(lines) >= MAX_SOURCES_PER_EXPERT:
            break

    if not lines:
        return ""

    omitted = max(0, len(sources) - len(lines))
    if omitted:
        lines.append(f"- ... {omitted} additional source(s) omitted")

    return "Sources for this expert (compact, provenance only):\n" + "\n".join(lines) + "\n"


def _format_expert_opinion(label, model_name, answer, model_sources):
    # model_name ist der echte Modellname (nur für den model_sources-Lookup);
    # im Prompt erscheint ausschließlich das anonyme Label.
    source_section = _format_sources_for_prompt(model_name, model_sources)
    return (
        f"Expert opinion from {label}:\n"
        f"Answer:\n{answer}\n"
        f"{source_section}\n"
    )


def _build_consensus_prompt(
    question: str,
    answer_openai: str,
    answer_mistral: str,
    answer_claude: str,
    answer_gemini: str,
    answer_deepseek: str,
    answer_grok: str,
    excluded_models: list,
    model_sources=None,
    shuffle: bool = True,
) -> str:
    """Baut den Consensus-Prompt. Die Expertenantworten werden wie im
    Differences-Prompt anonymisiert ("Expert A/B/...") und gemischt, damit
    weder Markenname noch Position die Synthese verzerren. Die [S1]-Source-IDs
    in den Antworten bleiben unverändert. shuffle=False liefert die feste
    Reihenfolge OpenAI..Grok (nur für das deterministische
    Benchmark-Prompt-Template, nicht für Live-Calls)."""
    excluded = normalize_excluded_models(excluded_models)

    model_answers = [
        ("OpenAI",    answer_openai),
        ("Mistral",   answer_mistral),
        ("Anthropic", answer_claude),
        ("Gemini",    answer_gemini),
        ("DeepSeek",  answer_deepseek),
        ("Grok",      answer_grok),
    ]
    model_answers = [
        (name, answer) for (name, answer) in model_answers
        if answer and normalize_model_name(name) not in excluded
    ]
    if shuffle:
        random.shuffle(model_answers)

    prompt_parts = []

    prompt_parts.append(
        f"Please provide your answer in the same language as the user's question. "
        f"The question is: {question}\n\n"
    )

    prompt_parts.append(
        "Below are independent expert opinions from different models. "
        "Each source list belongs only to the immediately preceding expert opinion. "
        "Use sources as compact provenance, not as additional opinions. "
        "Do not restate raw source lists in the final answer.\n\n"
    )

    for idx, (name, answer) in enumerate(model_answers):
        label = f"Expert {chr(ord('A') + idx)}"
        prompt_parts.append(_format_expert_opinion(label, name, answer, model_sources))

    user_facing_instruction = (
        "Use the expert-opinion framing only for your internal synthesis. "
        "The final answer is for an end user, so do not mention experts, expert opinions, models, "
        "model responses, consensus mechanics, or that sources disagree. "
        "Resolve disagreements silently where possible. If uncertainty remains important, state it as "
        "ordinary factual uncertainty without referring to the underlying experts or models. "
        "When a central factual claim is directly supported by a cited source in the provided opinions, "
        "include the existing source tag such as [S1] next to that claim. "
        "Use only source tags that were provided in the opinions or their compact source lists; never invent new source IDs. "
        "Use citations sparingly and only where they add verifiability. "
        "Provide only the final, balanced answer. "
        "Do not ask the user any follow-up or clarifying questions; answer directly with the information available."
    )

    prompt_parts.append(
        "You receive multiple expert opinions on a specific question. "
        "Treat all expert opinions equally. Do not focus on the answer of one model. "
        "Your task is to combine these responses into a comprehensive, correct, and coherent answer. "
        "Note: Experts can also make mistakes. Therefore, try to identify and exclude possible errors by comparing the answers. "
        "Structure the answer clearly and coherently. "
        + user_facing_instruction
    )

    return "".join(prompt_parts)


# Konsens-Fehlertexte, an denen Aufrufer (chat.py) einen gescheiterten Lauf
# erkennen: Differences und Share-Persistenz werden dann übersprungen.
CONSENSUS_ERROR_PREFIXES = ("Consensus error:", "Invalid consensus model selected:")
CONSENSUS_MAX_ATTEMPTS = 2
# Niedrige Temperatur für die Synthese (wie DIFFERENCES_TEMPERATURE eine
# bewusste Engine-Einstellung); _effective_temperature filtert sie für
# Reasoning-Modelle, die keine Temperatur akzeptieren, wieder heraus.
CONSENSUS_TEMPERATURE = 0.3

# Freitext für die Differences-Spalte, wenn der Vergleich mangels
# Konsensantwort gar nicht erst gestartet wird.
DIFFERENCES_SKIPPED_TEXT = (
    "The model comparison was skipped because no consensus answer "
    "could be generated. Please try again."
)


def is_consensus_error_text(text) -> bool:
    """True, wenn der Konsens-Text ein Fehler (oder leer) ist."""
    stripped = str(text or "").strip()
    return not stripped or stripped.startswith(CONSENSUS_ERROR_PREFIXES)


def query_consensus(
    question: str,
    answer_openai: str,
    answer_mistral: str,
    answer_claude: str,
    answer_gemini: str,
    answer_deepseek: str,
    answer_grok: str,
    excluded_models: list,
    consensus_model: str,
    api_keys: dict,
    model_sources=None,
) -> str:
    """
    Konsolidiert die Antworten der 6 Haupt-LLMs zu einer Konsensantwort.
    Engine-Auswahl (inkl. Pro-Aliasse) läuft über _resolve_engine.
    """
    consensus_prompt = _build_consensus_prompt(
        question,
        answer_openai,
        answer_mistral,
        answer_claude,
        answer_gemini,
        answer_deepseek,
        answer_grok,
        excluded_models,
        model_sources=model_sources,
    )

    resolved = _resolve_engine(consensus_model)
    if resolved is None:
        return f"Invalid consensus model selected: {consensus_model}"

    # Zwei Versuche auf der gewählten Engine: Provider-Fehler (503, Timeouts,
    # ...) sind oft transient, und ein gescheiterter Konsens macht den
    # gesamten Lauf wertlos. Scheitern beide, folgt ein dritter Versuch auf
    # einem anderen Provider mit verfügbarem Key (wie der Fallback-Judge der
    # Differences-Engine); "Consensus error:" bleibt die letzte Stufe.
    attempts = [resolved] * CONSENSUS_MAX_ATTEMPTS
    fallback = _fallback_judge_engine(resolved[0], api_keys)
    if fallback:
        attempts.append(fallback)

    last_error = "empty response from consensus engine."
    for provider, api_model, model_ref in attempts:
        try:
            result = _call_engine_text(
                provider, api_model, model_ref, api_keys,
                system="",
                prompt=consensus_prompt,
                max_tokens=cfg.CONSENSUS_MAX_TOKENS,
                temperature=CONSENSUS_TEMPERATURE,
            )
        except Exception as e:
            last_error = str(e)
            logging.warning(f"Consensus attempt failed on {provider}/{api_model}: {e}")
            continue
        if result:
            return result
        last_error = "empty response from consensus engine."
    return f"Consensus error: {last_error}"


MAX_DIFF_ANSWER_CHARS = 6000


def _build_differences_prompt(
    answer_openai: str,
    answer_mistral: str,
    answer_claude: str,
    answer_gemini: str,
    answer_deepseek: str,
    answer_grok: str,
    consensus_answer: str,
    excluded_models: list = None,
):
    """Baut den Differences-Prompt. Gibt (prompt, anon_map, answers_by_model)
    zurück oder None, wenn keine Modellantworten vorliegen. answers_by_model
    enthält die (gekappten) Antworttexte je echtem Modellnamen für die
    serverseitige Zitat-Verifikation."""

    excluded = normalize_excluded_models(excluded_models or [])

    model_answers = [
        ("OpenAI",   answer_openai),
        ("Mistral",  answer_mistral),
        ("Anthropic", answer_claude),
        ("Gemini",   answer_gemini),
        ("DeepSeek", answer_deepseek),
        ("Grok",     answer_grok),
    ]

    # Leere und explizit abgewählte Antworten filtern.
    model_answers = [
        (n, a) for (n, a) in model_answers
        if a and normalize_model_name(n) not in excluded
    ]

    if not model_answers:
        return None

    random.shuffle(model_answers)

    anon_map = {}
    answers_by_model = {}
    lines = []
    labels = []
    for idx, (name, text) in enumerate(model_answers):
        label = chr(ord("A") + idx)      # A, B, C, ...
        anon_label = f"Model {label}"
        anon_map[anon_label] = name
        answers_by_model[name] = (text or "")[:MAX_DIFF_ANSWER_CHARS]
        labels.append(anon_label)
        lines.append(f"- {anon_label}: {answers_by_model[name]}")

    responses_text = "\n".join(lines)

    if len(labels) > 1:
        allowed_list = ", ".join(labels[:-1]) + " or " + labels[-1]
    else:
        allowed_list = labels[0]

    differences_prompt = (
        "You compare several anonymized model responses against a consensus answer.\n"
        "Respond with ONLY one JSON object. No prose before or after it, no markdown fences.\n\n"
        "JSON schema:\n"
        "{\n"
        '  "claims": [\n'
        "    {\n"
        '      "anchor": "verbatim excerpt of 5-12 consecutive words copied exactly from the consensus answer",\n'
        '      "agree": ["Model A"],\n'
        '      "dissent": [{"model": "Model B", "quote": "verbatim short quote from that model\'s response"}]\n'
        "    }\n"
        "  ],\n"
        '  "differences": [\n'
        "    {\n"
        '      "claim": "the disputed point in one short sentence",\n'
        '      "type": "contradiction",\n'
        '      "severity": "major",\n'
        '      "positions": [\n'
        '        {"stance": "one short sentence", "models": ["Model A"], "quote": "verbatim short quote"}\n'
        "      ],\n"
        '      "verify": "one short sentence saying what exactly the user should double-check"\n'
        "    }\n"
        "  ],\n"
        '  "best_model": "Model A"\n'
        "}\n\n"
        "Rules:\n"
        "- \"claims\": the 3-6 most central claims of the consensus answer. For each, list under \"agree\" every model "
        "whose response supports it and under \"dissent\" every model whose response contradicts or clearly deviates "
        "from it, with a short verbatim quote. A model that does not address a claim appears in neither list.\n"
        "- \"differences\": substantive disagreements between the model responses. Use an empty list if there are none. "
        "\"type\" is \"contradiction\" when facts or conclusions are incompatible, and \"emphasis\" when models merely "
        "set different focus, omit something, or weight things differently. Be conservative: only incompatible "
        "statements count as a contradiction. \"verify\" is optional.\n"
        "- \"severity\" (only for type \"contradiction\"): \"major\" when the disagreement changes the overall "
        "conclusion, recommendation, or a central fact of the answer; \"minor\" when it concerns a side detail "
        "that leaves the conclusion intact. Omit it for \"emphasis\" differences.\n"
        "- Quotes and anchors must be copied verbatim from the given texts. You may shorten them at the start or end, "
        "but never paraphrase. Keep each quote under 200 characters.\n"
        f"- Use only these model labels: {allowed_list}. Never invent other labels.\n"
        "- Ignore citation markers, source labels, URLs, and source-list noise unless they reveal a real factual "
        "disagreement.\n"
        "- Write \"claim\", \"stance\", and \"verify\" in the same language as the model responses.\n"
        "- \"best_model\": the model whose answer is closest to the consensus answer.\n\n"
        "Consensus answer:\n" + consensus_answer + "\n\n"
        "Model responses:\n" + responses_text + "\n"
    )

    return differences_prompt, anon_map, answers_by_model


MAX_DIFF_CLAIMS = 8
MAX_DIFF_ENTRIES = 6
MAX_DIFF_POSITIONS = 4
MAX_DIFF_QUOTE_CHARS = 300
MAX_DIFF_TEXT_CHARS = 280


def _extract_json_object(raw: str):
    text = str(raw or "").strip()
    if not text:
        return None

    candidates = []
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        candidates.append(fence.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed

    # Abgeschnittene Ausgaben (max_tokens mitten im JSON) reparieren: offene
    # Strings/Arrays/Objekte schließen, halbe Werte am Ende verwerfen.
    if start != -1:
        repaired = _repair_truncated_json(text[start:])
        if repaired is not None:
            return repaired
    return None


def _close_open_json(text: str):
    """Schließt offene Strings/Klammern am Ende eines JSON-Fragments.
    Liefert None, wenn das Fragment so nicht schließbar ist (der Aufrufer
    kürzt dann weiter)."""
    stack = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                return None
            open_ch = stack.pop()
            if (open_ch == "{") != (ch == "}"):
                return None
    repaired = text
    if escaped:
        repaired = repaired[:-1]
    if in_string:
        repaired += '"'
    stripped = repaired.rstrip()
    if stripped.endswith(","):
        stripped = stripped[:-1].rstrip()
    if stripped.endswith(":"):
        # Schlüssel ohne Wert: hier nicht reparierbar.
        return None
    return stripped + "".join("}" if ch == "{" else "]" for ch in reversed(stack))


def _repair_truncated_json(fragment: str):
    text = fragment
    for _ in range(40):
        repaired = _close_open_json(text)
        if repaired is not None:
            try:
                parsed = json.loads(repaired)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict):
                logging.info("Differences engine output was truncated; repaired JSON tail.")
                return parsed
        cut = max(text.rfind(","), text.rfind("{"), text.rfind("["))
        if cut <= 0:
            return None
        text = text[:cut]
    return None


def _real_model_names(labels, anon_map: dict) -> list:
    names = []
    for label in labels if isinstance(labels, list) else []:
        real = anon_map.get(str(label or "").strip())
        if not real:
            logging.warning(f"Differences engine used unknown model label: {label!r}")
            continue
        if real not in names:
            names.append(real)
    return names


def _normalize_claims(raw_claims, anon_map: dict) -> list:
    claims = []
    for entry in raw_claims if isinstance(raw_claims, list) else []:
        if not isinstance(entry, dict):
            continue
        anchor = _clip(entry.get("anchor"), MAX_DIFF_TEXT_CHARS)
        if not anchor:
            continue

        agree = _real_model_names(entry.get("agree"), anon_map)
        dissent = []
        for item in entry.get("dissent") if isinstance(entry.get("dissent"), list) else []:
            if not isinstance(item, dict):
                continue
            real = _real_model_names([item.get("model")], anon_map)
            if not real:
                continue
            dissent.append({
                "model": real[0],
                "quote": _clip(item.get("quote"), MAX_DIFF_QUOTE_CHARS),
            })

        # Doppelnennungen auflösen: Abweichler verdrängen die Zustimmung.
        dissent_models = {item["model"] for item in dissent}
        agree = [name for name in agree if name not in dissent_models]
        if not agree and not dissent:
            continue

        claims.append({"anchor": anchor, "agree": agree, "dissent": dissent})
        if len(claims) >= MAX_DIFF_CLAIMS:
            break
    return claims


def _normalize_differences(raw_differences, anon_map: dict) -> list:
    differences = []
    for entry in raw_differences if isinstance(raw_differences, list) else []:
        if not isinstance(entry, dict):
            continue
        claim = _clip(entry.get("claim"), MAX_DIFF_TEXT_CHARS)
        if not claim:
            continue

        positions = []
        for item in entry.get("positions") if isinstance(entry.get("positions"), list) else []:
            if not isinstance(item, dict):
                continue
            models = _real_model_names(item.get("models"), anon_map)
            if not models:
                continue
            positions.append({
                "stance": _clip(item.get("stance"), MAX_DIFF_TEXT_CHARS),
                "models": models,
                "quote": _clip(item.get("quote"), MAX_DIFF_QUOTE_CHARS),
            })
            if len(positions) >= MAX_DIFF_POSITIONS:
                break
        if not positions:
            continue

        diff_type = str(entry.get("type") or "").strip().lower()
        if diff_type != "contradiction":
            diff_type = "emphasis"

        severity = ""
        if diff_type == "contradiction":
            severity = str(entry.get("severity") or "").strip().lower()
            if severity != "minor":
                # Konservativer Default: unklare Schwere zählt als gewichtiger
                # Widerspruch, damit fehlende Severity nichts beschönigt.
                severity = "major"

        differences.append({
            "claim": claim,
            "type": diff_type,
            "severity": severity,
            "positions": positions,
            "verify": _clip(entry.get("verify"), MAX_DIFF_TEXT_CHARS),
        })
        if len(differences) >= MAX_DIFF_ENTRIES:
            break
    return differences


# ---------------------------------------------------------------------------
# Agreement-Score: eine transparente 0-100-Zahl aus Claims und Differences.
# Ersetzt die alte "Anzahl Widersprüche"-Heuristik als einzige Quelle für die
# Credibility-Stufe (Freitext-Satz UND Frontend-Verdict speisen sich daraus).
# ---------------------------------------------------------------------------

AGREEMENT_LEVEL_THRESHOLDS = [
    (85, "very"),
    (65, "largely"),
    (40, "partially"),
    (20, "hardly"),
]

_CREDIBILITY_SENTENCES = {
    "very": "The consensus answer is **very** credible.",
    "largely": "The consensus answer is **largely** credible.",
    "partially": "The consensus answer is **partially** credible.",
    "hardly": "The consensus answer is **hardly** credible.",
    "not": "The consensus answer is **not** credible.",
}


def compute_agreement_score(data: dict) -> dict:
    """Berechnet den Agreement-Score (0-100) samt Level.

    Basis ist die mittlere Zustimmungsquote über die Claims; Widersprüche
    ziehen nach Schwere ab (major > minor > emphasis). Caps erhalten die
    etablierte Stufen-Semantik: "very" gibt es nur ohne jede Differenz,
    Major-Widersprüche deckeln auf "partially"/"hardly", und wenige
    verglichene Modelle deckeln das erreichbare Vertrauen."""
    claims = data.get("claims") or []
    differences = data.get("differences") or []
    model_count = len(data.get("models_compared") or [])

    ratios = []
    for claim in claims:
        agree = len(claim.get("agree") or [])
        dissent = len(claim.get("dissent") or [])
        if agree + dissent:
            ratios.append(agree / (agree + dissent))
    base = sum(ratios) / len(ratios) if ratios else 1.0

    contradictions = [d for d in differences if d.get("type") == "contradiction"]
    major = sum(1 for d in contradictions if d.get("severity") != "minor")
    minor = len(contradictions) - major
    emphases = len(differences) - len(contradictions)

    score = base - 0.25 * major - 0.10 * minor - 0.05 * emphases

    caps = [1.0]
    if differences:
        caps.append(0.84)   # "very" nur bei völlig differenzfreiem Vergleich
    if major >= 2:
        caps.append(0.39)   # mehrere Major-Widersprüche: höchstens "hardly"
    elif major == 1:
        caps.append(0.64)   # ein Major-Widerspruch: höchstens "partially"
    if model_count == 3:
        caps.append(0.90)
    elif model_count == 2:
        caps.append(0.75)   # 2 Modelle können kein "very" belegen
    elif model_count <= 1:
        caps.append(0.50)
    score = max(0.0, min(score, *caps))

    score_pct = int(round(score * 100))
    level = "not"
    for threshold, name in AGREEMENT_LEVEL_THRESHOLDS:
        if score_pct >= threshold:
            level = name
            break

    return {
        "score": score_pct,
        "level": level,
        "model_count": model_count,
        "major_contradictions": major,
        "minor_contradictions": minor,
        "emphases": emphases,
    }


def _legacy_differences_text(data: dict) -> str:
    """Synthetisiert aus den strukturierten Daten den bisherigen Freitext
    (Credibility-Satz, Bullets, BestModel-Zeile), damit Bookmarks,
    Credibility-Frame und Leaderboard-Vote unverändert funktionieren.
    Der Credibility-Satz leitet sich aus dem Agreement-Score ab, damit
    Freitext und strukturierte Auswertung nie divergieren."""
    differences = data.get("differences") or []
    agreement = data.get("agreement") or compute_agreement_score(data)
    credibility = _CREDIBILITY_SENTENCES.get(agreement.get("level"), _CREDIBILITY_SENTENCES["partially"])

    lines = [credibility, "", "_____________", ""]
    if differences:
        for diff in differences[:2]:
            lines.append(f"- {_clip(diff.get('claim'), 120)}")
    else:
        lines.append("- No substantive contradictions between the responses.")

    best_model = data.get("best_model")
    if best_model:
        lines.extend(["", f"BestModel: {best_model}"])

    return "\n".join(lines)


def _looks_like_json(raw: str) -> bool:
    text = str(raw or "").lstrip()
    return text.startswith("{") or text.startswith("```")


# ---------------------------------------------------------------------------
# Serverseitige Zitat-Verifikation: Anchors gegen die Konsensantwort, Quotes
# gegen die jeweilige Modellantwort. Spiegelbildlich zur Suche im Frontend
# (consensus-insights.js): Whitespace kollabieren, Anführungszeichen
# vereinheitlichen, Ellipsen an den Rändern ignorieren.
# ---------------------------------------------------------------------------

_QUOTE_CHARS = set("“”„‘’«»\"")
_ELLIPSIS_EDGE_RE = re.compile(r"^(?:\.{3}|…)\s*|\s*(?:\.{3}|…)$")
FUZZY_MATCH_MIN_CHARS = 15
FUZZY_MATCH_MIN_RATIO = 0.6


def _normalize_with_offsets(text: str):
    """Normalisiert einen Text und liefert je normalisiertem Zeichen den
    Original-Offset, damit Treffer auf den Originaltext abgebildet werden."""
    norm_chars = []
    offsets = []
    for i, ch in enumerate(str(text or "")):
        c = ch.lower()
        if c in _QUOTE_CHARS:
            c = '"'
        if c.isspace():
            if not norm_chars or norm_chars[-1] == " ":
                continue
            c = " "
        norm_chars.append(c)
        offsets.append(i)
    while norm_chars and norm_chars[-1] == " ":
        norm_chars.pop()
        offsets.pop()
    return "".join(norm_chars), offsets


def _normalize_needle(text: str) -> str:
    norm, _ = _normalize_with_offsets(_ELLIPSIS_EDGE_RE.sub("", str(text or "")))
    return norm


def _locate_span(haystack: str, hay_norm: str, hay_offsets: list, needle: str):
    """Sucht ein (LLM-)Zitat im Originaltext: erst exakt auf normalisierter
    Basis, dann fuzzy über difflib. Liefert den Original-Ausschnitt oder None."""
    needle_norm = _normalize_needle(needle)
    if not needle_norm:
        return None
    idx = hay_norm.find(needle_norm)
    if idx != -1:
        start = hay_offsets[idx]
        end = hay_offsets[idx + len(needle_norm) - 1] + 1
        return haystack[start:end]
    if len(needle_norm) >= FUZZY_MATCH_MIN_CHARS:
        matcher = difflib.SequenceMatcher(None, hay_norm, needle_norm, autojunk=False)
        match = matcher.find_longest_match(0, len(hay_norm), 0, len(needle_norm))
        if match.size >= max(FUZZY_MATCH_MIN_CHARS, int(len(needle_norm) * FUZZY_MATCH_MIN_RATIO)):
            start = hay_offsets[match.a]
            end = hay_offsets[match.a + match.size - 1] + 1
            return haystack[start:end].strip()
    return None


def _verify_differences_data(data: dict, consensus_answer: str, model_answers: dict) -> None:
    """Ersetzt gefundene Anchors/Quotes durch den Originaltext (hilft dem
    Frontend-Matching) und leert Quotes, die in der jeweiligen Modellantwort
    nicht auffindbar sind - halluzinierte Zitate werden so nie angezeigt."""
    prepared = {}

    def _find(key: str, text: str, needle: str):
        if not text:
            return None
        if key not in prepared:
            norm, offsets = _normalize_with_offsets(text)
            prepared[key] = (norm, offsets)
        norm, offsets = prepared[key]
        return _locate_span(text, norm, offsets, needle)

    consensus_text = str(consensus_answer or "")
    for claim in data.get("claims") or []:
        span = _find("__consensus__", consensus_text, claim.get("anchor"))
        if span:
            claim["anchor"] = _clip(span, MAX_DIFF_TEXT_CHARS)
        else:
            logging.info(f"Differences anchor not found in consensus answer: {claim.get('anchor')!r}")
        for item in claim.get("dissent") or []:
            if not item.get("quote"):
                continue
            span = _find(item["model"], model_answers.get(item["model"]) or "", item["quote"])
            if span:
                item["quote"] = _clip(span, MAX_DIFF_QUOTE_CHARS)
            else:
                logging.info(f"Dropping unverifiable dissent quote for {item['model']}: {item['quote']!r}")
                item["quote"] = ""

    for diff in data.get("differences") or []:
        for position in diff.get("positions") or []:
            if not position.get("quote"):
                continue
            span = None
            for model in position.get("models") or []:
                span = _find(model, model_answers.get(model) or "", position["quote"])
                if span:
                    break
            if span:
                position["quote"] = _clip(span, MAX_DIFF_QUOTE_CHARS)
            else:
                logging.info(f"Dropping unverifiable position quote for {position.get('models')}: {position['quote']!r}")
                position["quote"] = ""


def parse_differences_payload(raw: str, anon_map: dict, consensus_answer: str = None, model_answers: dict = None):
    """Parst die JSON-Ausgabe des Differences-Calls und übersetzt die
    anonymisierten Labels zurück. Gibt (data | None, legacy_text) zurück.

    Bei unparsbarer Ausgabe ist data None; sieht die Rohausgabe nach JSON aus,
    ist legacy_text leer (kein Roh-JSON an den Nutzer), sonst der Rohtext mit
    rückübersetzter BestModel-Zeile (Alt-Verhalten für Prosa-Ausgaben).
    Mit consensus_answer/model_answers werden Anchors und Quotes serverseitig
    verifiziert."""
    parsed = _extract_json_object(raw)
    if parsed is None or not (
        isinstance(parsed.get("claims"), list) and isinstance(parsed.get("differences"), list)
    ):
        # Auch reparierte, aber strukturell unvollständige Objekte (z. B. ohne
        # "differences"-Liste) gelten als unparsbar: fehlende Widersprüche
        # dürfen nicht als "keine Widersprüche" durchgehen.
        text = str(raw or "").strip()
        if _looks_like_json(text):
            return None, ""
        return None, _translate_best_model(text, anon_map)

    best_label = str(parsed.get("best_model") or "").strip()
    best_model = anon_map.get(best_label, "")
    if best_label and not best_model:
        logging.warning(f"Differences engine hallucinated best_model label: {best_label!r}")

    data = {
        "claims": _normalize_claims(parsed.get("claims"), anon_map),
        "differences": _normalize_differences(parsed.get("differences"), anon_map),
        "best_model": best_model,
        "models_compared": sorted(anon_map.values()),
    }
    if consensus_answer or model_answers:
        _verify_differences_data(data, consensus_answer or "", model_answers or {})
    data["agreement"] = compute_agreement_score(data)
    return data, _legacy_differences_text(data)


def _translate_best_model(result: str, anon_map: dict) -> str:
    """Übersetzt die anonymisierte BestModel-Zeile zurück auf den echten Modellnamen."""
    match = re.search(r"BestModel:\s*Model\s*([A-Z])", result)
    if match:
        anon_label = f"Model {match.group(1)}"
        # Sicherstellen, dass wir den echten Namen haben
        if anon_label in anon_map:
            real_name = anon_map[anon_label]
            result = re.sub(
                r"BestModel:\s*Model\s*[A-Z]",
                f"BestModel: {real_name}",
                result
            )
        else:
            logging.warning(f"LLM hallucinated ID {anon_label} in differences.")

    return result


# ---------------------------------------------------------------------------
# Differences-Engine: Judge-Policy, Attempt-Plan und die beiden Einstiege
# (query_differences / stream_differences).
# ---------------------------------------------------------------------------

# Standard-Judges: das günstige Default-Modell des jeweiligen Providers.
# Die Judge-Stufe folgt der gewählten Consensus-Engine (Standard-Engine ->
# Standard-Judge, Pro-Engine -> Pro-Judge über die bestehenden Engine-Aliasse);
# die Judge-FAMILIE ist dabei immer eine andere als die der Engine, siehe
# _resolve_differences_engine.
DIFFERENCES_JUDGE_MODEL_BY_PROVIDER = {
    "openai": DEFAULT_OPENAI_MODEL,
    "mistral": DEFAULT_MISTRAL_MODEL,
    "anthropic": DEFAULT_ANTHROPIC_MODEL,
    "gemini": GEMINI_FLASH_MODEL,
    "deepseek": DEFAULT_DEEPSEEK_MODEL,
    "grok": DEFAULT_GROK_MODEL,
}

# Familien-Priorität für die Judge-Wahl: primärer Differences-Judge und
# Fallback-Judge nehmen die erste Familie mit verfügbarem Key, die nicht die
# der Consensus-Engine ist. Wird auch vom Consensus-Fallback (dritter Versuch
# auf einem anderen Provider) genutzt.
_FALLBACK_JUDGE_PRIORITY = ["gemini", "openai", "mistral", "deepseek", "grok", "anthropic"]

DIFFERENCES_SYSTEM_PROMPT = "Answer in the exact same language as the Model responses."
DIFFERENCES_TEMPERATURE = 0.2
DIFFERENCES_RETRY_SUFFIX = (
    "\n\nIMPORTANT: Return exactly ONE complete, syntactically valid JSON object "
    "matching the schema above. No prose, no markdown fences, no trailing text."
)


def _provider_key_available(provider: str, api_keys: dict) -> bool:
    if provider == "gemini":
        return bool(_gemini_engine_key(api_keys))
    return bool(api_keys.get(_PROVIDER_KEY_NAMES[provider]))


def _judge_tier(differences_model: str) -> str:
    """Judge-Stufe der gewählten Consensus-Engine: Pro-Engines bekommen einen
    Pro-Judge, alles andere (inkl. Early-/Frontier-Low-Engines) den günstigen
    Standard-Judge."""
    return "pro" if cfg.is_premium_consensus_model(differences_model) else "standard"


def _standard_judge_engine(provider: str):
    judge = DIFFERENCES_JUDGE_MODEL_BY_PROVIDER[provider]
    return provider, judge, judge


def _judge_engine(provider: str, tier: str):
    """(provider, api_model, model_ref) für den Judge einer Familie in der
    gewünschten Stufe. Die Pro-Stufe löst über den bestehenden Engine-Alias
    "<Familie>-Pro" auf (keine eigenen Modell-Konstanten); scheitert die
    Auflösung, bleibt der Standard-Judge."""
    if tier == "pro":
        resolved = _resolve_engine(_PROVIDER_KEY_NAMES[provider] + "-Pro")
        if resolved is not None:
            return resolved
    return _standard_judge_engine(provider)


def _judge_families(consensus_provider: str, api_keys: dict, count: int) -> list:
    """Die ersten `count` Judge-Familien aus der Priorität, die (a) nicht die
    Familie der Consensus-Engine sind und (b) einen verfügbaren Key haben."""
    families = []
    for provider in _FALLBACK_JUDGE_PRIORITY:
        if provider == consensus_provider:
            continue
        if not _provider_key_available(provider, api_keys):
            continue
        families.append(provider)
        if len(families) >= count:
            break
    return families


def _resolve_differences_engine(differences_model: str, api_keys: dict):
    """Primärer Differences-Judge für die gewählte Consensus-Engine.

    Die Judge-Familie ist immer eine ANDERE als die der Consensus-Engine:
    der Judge bewertet die Konsensantwort und darf nicht das Modell sein,
    das sie geschrieben hat (Self-Judging-Bias). Die frühere
    Same-Family-Policy ist damit bewusst aufgegeben. Nur wenn keine fremde
    Familie einen verfügbaren Key hat, fällt die Wahl fail-open auf den
    Standard-Judge der eigenen Familie zurück (ein fehlender Fremd-Key darf
    den Lauf nicht brechen; der Standard-Judge ist dann wenigstens nicht das
    Pro-Modell, das die Konsensantwort geschrieben haben kann).

    Gibt ((provider, api_model, model_ref), tier) zurück, None bei
    ungültiger Engine."""
    resolved = _resolve_engine(differences_model)
    if resolved is None:
        return None
    tier = _judge_tier(differences_model)
    families = _judge_families(resolved[0], api_keys, count=1)
    if families:
        return _judge_engine(families[0], tier), tier
    logging.warning(
        f"No cross-family judge key available for engine {differences_model}; "
        "falling back to the same-family standard judge."
    )
    return _standard_judge_engine(resolved[0]), "standard"


def _fallback_judge_engine(exclude_provider: str, api_keys: dict):
    for provider in _FALLBACK_JUDGE_PRIORITY:
        if provider == exclude_provider:
            continue
        if not _provider_key_available(provider, api_keys):
            continue
        judge = DIFFERENCES_JUDGE_MODEL_BY_PROVIDER[provider]
        return provider, judge, judge
    return None


def _differences_attempts(differences_model: str, api_keys: dict):
    """Attempt-Plan für den Differences-Judge. None bei ungültiger Engine.

    Einträge sind ((provider, api_model, model_ref), is_retry, tier):
    primärer Judge (Fremd-Familie, Stufe der Engine), Retry, dann die nächste
    Fremd-Familie in derselben Stufe. Die Pro-Stufe fail-opent zuletzt auf
    einen Standard-Judge; gibt es keine zweite Fremd-Familie, ist der
    Standard-Judge der eigenen Familie die letzte Stufe — Robustheit geht
    als letztes Mittel vor Unabhängigkeit."""
    resolved = _resolve_engine(differences_model)
    if resolved is None:
        return None
    consensus_provider = resolved[0]
    primary, tier = _resolve_differences_engine(differences_model, api_keys)
    attempts = [(primary, False, tier), (primary, True, tier)]

    families = _judge_families(consensus_provider, api_keys, count=2)
    if not families:
        # Primär ist bereits der eigene Standard-Judge (Fail-open ohne
        # Fremd-Key); mehr Stufen gibt es nicht.
        return attempts

    if len(families) > 1:
        attempts.append((_judge_engine(families[1], tier), True, tier))
        if tier == "pro":
            attempts.append((_standard_judge_engine(families[1]), True, "standard"))
    else:
        if tier == "pro":
            attempts.append((_standard_judge_engine(families[0]), True, "standard"))
        attempts.append((_standard_judge_engine(consensus_provider), True, "standard"))
    return attempts


def _judge_metadata(provider: str, api_model: str, tier: str, attempts: int = 0, duration_ms: int = 0) -> dict:
    """Transparenz-Metadaten des Judges, der das Ergebnis TATSÄCHLICH geliefert
    hat (nach einem Fallback also nicht der geplante primäre Judge). Landet als
    differences_data["judges"]["differences"] im Payload, Snapshot und in der
    anonymen Telemetrie (nur Metadaten, keine Texte). Der Schlüssel
    "adjudicator" ist für eine spätere Adjudicator-Runde reserviert.
    attempts = Nummer des erfolgreichen Versuchs (1 = kein Retry nötig),
    duration_ms = Dauer nur dieses Versuchs."""
    return {
        "provider": _PROVIDER_KEY_NAMES.get(provider, provider),
        "model": api_model,
        "tier": tier,
        "attempts": int(attempts),
        "duration_ms": int(duration_ms),
    }


def _judge_effort(judge_tier: str) -> str | None:
    """Thinking-Kappung für Judge-Calls: Der Judge-Task (Zitate verbatim
    extrahieren und vergleichen) braucht kein tiefes Denken. Pro-Judges sind
    Reasoning-Modelle, deren unbegrenztes Thinking den Differences-Schritt
    minutenlang verzögert und das Token-Budget des JSON auffressen kann —
    daher effort "low" (Gemini thinkingLevel, OpenAI/Mistral reasoning_effort).
    Standard-Judges sind günstige, schnelle Modelle und bleiben unangetastet."""
    return "low" if judge_tier == "pro" else None


def query_differences(
    answer_openai: str,
    answer_mistral: str,
    answer_claude: str,
    answer_gemini: str,
    answer_deepseek: str,
    answer_grok: str,
    consensus_answer: str,
    api_keys: dict,
    differences_model: str,
    excluded_models: list = None,
) -> tuple:
    """
    Extrahiert die Unterschiede zwischen den Antworten der 6 Hauptmodelle,
    anonymisiert die Modellnamen und ordnet das bestbewertete Modell anschließend wieder zu.
    Läuft mit Structured Output, JSON-Repair, einem Retry und Fallback-Judge;
    der Judge ist immer eine andere Modellfamilie als die Consensus-Engine
    (siehe _resolve_differences_engine) und wird in data["judges"] ausgewiesen.
    Gibt (legacy_text, structured_data | None) zurück.
    """
    built = _build_differences_prompt(
        answer_openai,
        answer_mistral,
        answer_claude,
        answer_gemini,
        answer_deepseek,
        answer_grok,
        consensus_answer,
        excluded_models=excluded_models,
    )
    if built is None:
        return "Error in comparison: no model responses available.", None

    differences_prompt, anon_map, answers_by_model = built

    attempts = _differences_attempts(differences_model, api_keys)
    if attempts is None:
        return "Invalid model selected for difference comparison.", None

    prose_fallback = None
    last_error = "empty result from differences engine."
    for attempt_no, ((provider, api_model, model_ref), is_retry, judge_tier) in enumerate(attempts, start=1):
        attempt_prompt = differences_prompt + (DIFFERENCES_RETRY_SUFFIX if is_retry else "")
        attempt_started = time.monotonic()
        try:
            raw = _call_engine_text(
                provider, api_model, model_ref, api_keys,
                system=DIFFERENCES_SYSTEM_PROMPT,
                prompt=attempt_prompt,
                max_tokens=cfg.DIFFERENCES_MAX_TOKENS,
                temperature=DIFFERENCES_TEMPERATURE,
                json_mode=True,
                effort=_judge_effort(judge_tier),
            )
        except Exception as e:
            last_error = str(e)
            logging.warning(
                f"Differences attempt {attempt_no} failed on {provider}/{api_model} "
                f"after {time.monotonic() - attempt_started:.1f}s: {e}"
            )
            continue
        duration_ms = int((time.monotonic() - attempt_started) * 1000)
        if not raw:
            last_error = "empty result from differences engine."
            continue

        data, legacy_text = parse_differences_payload(
            raw, anon_map,
            consensus_answer=consensus_answer,
            model_answers=answers_by_model,
        )
        if data is not None:
            data["judges"] = {"differences": _judge_metadata(
                provider, api_model, judge_tier,
                attempts=attempt_no, duration_ms=duration_ms,
            )}
            return legacy_text, data
        if prose_fallback is None and legacy_text and not _looks_like_json(raw):
            prose_fallback = legacy_text
        last_error = "unparsable output from differences engine."
        logging.warning(
            f"Differences output unparsable on {provider}/{api_model} "
            f"(attempt {attempt_no}, {duration_ms} ms)"
        )

    if prose_fallback:
        return prose_fallback, None
    return f"Error in comparison: {last_error}", None


# ---------------------------------------------------------------------------
# Streaming-Varianten: liefern {"type": "delta", "text": ...} Events und am
# Ende {"type": "final", "text": <Gesamttext>}. Fehler werden - wie bei den
# nicht-streamenden Varianten - als Fehlertext im final-Event transportiert.
# ---------------------------------------------------------------------------

def _stream_consensus_engine(consensus_model: str, api_keys: dict, consensus_prompt: str):
    resolved = _resolve_engine(consensus_model)
    if resolved is None:
        raise _InvalidEngineError(f"Invalid consensus model selected: {consensus_model}")
    provider, api_model, model_ref = resolved
    # Bewusst ohne effort-Kappung: die Consensus-Synthese ist das
    # Kernprodukt, ein Pro-Modell darf hier voll denken.
    yield from _stream_engine_text(
        provider, api_model, model_ref, api_keys,
        system="",
        prompt=consensus_prompt,
        max_tokens=cfg.CONSENSUS_MAX_TOKENS,
        temperature=CONSENSUS_TEMPERATURE,
    )


def stream_consensus(
    question: str,
    answer_openai: str,
    answer_mistral: str,
    answer_claude: str,
    answer_gemini: str,
    answer_deepseek: str,
    answer_grok: str,
    excluded_models: list,
    consensus_model: str,
    api_keys: dict,
    model_sources=None,
):
    consensus_prompt = _build_consensus_prompt(
        question,
        answer_openai,
        answer_mistral,
        answer_claude,
        answer_gemini,
        answer_deepseek,
        answer_grok,
        excluded_models,
        model_sources=model_sources,
    )

    resolved = _resolve_engine(consensus_model)
    if resolved is None:
        yield {"type": "final", "text": f"Invalid consensus model selected: {consensus_model}", "error": True}
        return

    # Versuchsplan wie im nicht-streamenden Pfad: zweimal die gewählte Engine,
    # danach einmal ein Fallback-Provider mit verfügbarem Key. Deltas eines
    # gescheiterten Versuchs sind unkritisch: das final-Event ersetzt den
    # gerenderten Konsens-Inhalt im Frontend vollständig (injectMarkdown).
    engine_models = [consensus_model] * CONSENSUS_MAX_ATTEMPTS
    fallback = _fallback_judge_engine(resolved[0], api_keys)
    if fallback:
        # Fallback-Judge ist eine interne Modell-ID und damit selbst ein
        # gültiger Engine-Wert für _stream_consensus_engine.
        engine_models.append(fallback[2])

    last_error = "empty response from consensus engine."
    for engine_model in engine_models:
        parts = []
        try:
            for event in _stream_consensus_engine(engine_model, api_keys, consensus_prompt):
                if event.get("type") == "reasoning":
                    yield {"type": "reasoning"}
                    continue
                text = event.get("text") or ""
                parts.append(text)
                yield {"type": "delta", "text": text}
        except _InvalidEngineError as e:
            yield {"type": "final", "text": str(e), "error": True}
            return
        except Exception as e:
            last_error = str(e)
            logging.warning(f"Consensus stream attempt failed on {engine_model}: {e}")
            continue

        final_text = "".join(parts).strip()
        if final_text:
            yield {"type": "final", "text": final_text}
            return
        last_error = "empty response from consensus engine."

    yield {"type": "final", "text": f"Consensus error: {last_error}", "error": True}


def stream_differences(
    answer_openai: str,
    answer_mistral: str,
    answer_claude: str,
    answer_gemini: str,
    answer_deepseek: str,
    answer_grok: str,
    consensus_answer: str,
    api_keys: dict,
    differences_model: str,
    excluded_models: list = None,
):
    built = _build_differences_prompt(
        answer_openai,
        answer_mistral,
        answer_claude,
        answer_gemini,
        answer_deepseek,
        answer_grok,
        consensus_answer,
        excluded_models=excluded_models,
    )
    if built is None:
        yield {"type": "final", "text": "Error in comparison: no model responses available.", "data": None}
        return

    differences_prompt, anon_map, answers_by_model = built

    attempts = _differences_attempts(differences_model, api_keys)
    if attempts is None:
        yield {"type": "final", "text": "Invalid model selected for difference comparison.", "data": None}
        return

    prose_fallback = None
    last_error = "empty result from differences engine."
    for attempt_no, ((provider, api_model, model_ref), is_retry, judge_tier) in enumerate(attempts, start=1):
        attempt_prompt = differences_prompt + (DIFFERENCES_RETRY_SUFFIX if is_retry else "")
        attempt_started = time.monotonic()
        parts = []
        try:
            for event in _stream_engine_text(
                provider, api_model, model_ref, api_keys,
                system=DIFFERENCES_SYSTEM_PROMPT,
                prompt=attempt_prompt,
                max_tokens=cfg.DIFFERENCES_MAX_TOKENS,
                temperature=DIFFERENCES_TEMPERATURE,
                json_mode=True,
                effort=_judge_effort(judge_tier),
            ):
                if event.get("type") == "reasoning":
                    # Marker, solange der Judge noch denkt: hält die
                    # SSE-Verbindung aktiv und speist den Frontend-Indikator.
                    yield {"type": "reasoning"}
                    continue
                text = event.get("text") or ""
                parts.append(text)
                # Roh-JSON wird im Frontend nicht gerendert; die Deltas halten
                # nur die SSE-Verbindung aktiv (auch während der Retries).
                yield {"type": "delta", "text": text}
        except Exception as e:
            last_error = str(e)
            logging.warning(
                f"Differences stream attempt {attempt_no} failed on {provider}/{api_model} "
                f"after {time.monotonic() - attempt_started:.1f}s: {e}"
            )
            continue

        duration_ms = int((time.monotonic() - attempt_started) * 1000)
        raw = "".join(parts).strip()
        if not raw:
            last_error = "empty result from differences engine."
            continue

        data, legacy_text = parse_differences_payload(
            raw, anon_map,
            consensus_answer=consensus_answer,
            model_answers=answers_by_model,
        )
        if data is not None:
            data["judges"] = {"differences": _judge_metadata(
                provider, api_model, judge_tier,
                attempts=attempt_no, duration_ms=duration_ms,
            )}
            yield {"type": "final", "text": legacy_text, "data": data}
            return
        if prose_fallback is None and legacy_text and not _looks_like_json(raw):
            prose_fallback = legacy_text
        last_error = "unparsable output from differences engine."
        logging.warning(
            f"Differences stream output unparsable on {provider}/{api_model} "
            f"(attempt {attempt_no}, {duration_ms} ms)"
        )

    if prose_fallback:
        yield {"type": "final", "text": prose_fallback, "data": None}
        return
    yield {"type": "final", "text": f"Error in comparison: {last_error}", "data": None}
