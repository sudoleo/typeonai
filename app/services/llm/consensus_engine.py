from __future__ import annotations

import os
import re
import json
import logging
import random
import requests
import openai
import google.generativeai as genai
import google.auth
from google.auth.transport.requests import Request as GoogleAuthRequest
from urllib.parse import quote

import app.core.config as cfg
from app.core.config import (
    GEMINI_FLASH_MODEL,
    GEMINI_PRO_MODEL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_MISTRAL_MODEL,
    MISTRAL_PRO_MODEL,
    DEFAULT_ANTHROPIC_MODEL,
    ANTHROPIC_PRO_MODEL,
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_GROK_MODEL,
)
from app.services.llm.engines import build_provider_payload

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


def _gemini_generate_content_rest(prompt: str, api_key: str | None, model_override: str, max_tokens: int) -> str:
    request_data = build_provider_payload(
        "gemini",
        question=prompt,
        system_prompt="",
        model_override=model_override,
        max_output_tokens=max_tokens,
    )
    payload = request_data["payload"]
    payload.pop("tools", None)
    request_kwargs = {"json": payload, "timeout": 120}
    if api_key and api_key.strip():
        request_kwargs["params"] = {"key": api_key.strip()}
    else:
        request_kwargs["headers"] = _google_adc_headers()

    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{quote(request_data['api_model'], safe='')}:generateContent",
        **request_kwargs,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"{response.status_code} - {response.text}")
    data = response.json()
    text_parts = []
    for candidate in data.get("candidates", []) or []:
        content = candidate.get("content") or {}
        for part in content.get("parts", []) or []:
            if part.get("text"):
                text_parts.append(part["text"])
    return "\n".join(text_parts).strip() or "Error: Empty response payload."


def _mistral_chat_complete(api_key: str, model: str, messages: list[dict], max_tokens: int) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if model in cfg.MISTRAL_REASONING_MODELS:
        payload["reasoning_effort"] = "high"

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
    return (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()


def normalize_model_name(model_name: str) -> str:
    key = str(model_name or "").strip()
    if key.endswith("-Pro"):
        key = key[:-4]
    return CANONICAL_MODEL_NAMES.get(key.lower(), key)


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


def _format_expert_opinion(label, answer, model_sources):
    source_section = _format_sources_for_prompt(label, model_sources)
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
) -> str:
    excluded = normalize_excluded_models(excluded_models)
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

    if "OpenAI" not in excluded and answer_openai:
        prompt_parts.append(_format_expert_opinion("OpenAI", answer_openai, model_sources))
    if "Mistral" not in excluded and answer_mistral:
        prompt_parts.append(_format_expert_opinion("Mistral", answer_mistral, model_sources))
    if "Anthropic" not in excluded and answer_claude:
        prompt_parts.append(_format_expert_opinion("Anthropic", answer_claude, model_sources))
    if "Gemini" not in excluded and answer_gemini:
        prompt_parts.append(_format_expert_opinion("Gemini", answer_gemini, model_sources))
    if "DeepSeek" not in excluded and answer_deepseek:
        prompt_parts.append(_format_expert_opinion("DeepSeek", answer_deepseek, model_sources))
    if "Grok" not in excluded and answer_grok:
        prompt_parts.append(_format_expert_opinion("Grok", answer_grok, model_sources))

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
        "Provide only the final, balanced answer."
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
    Unterscheidet jetzt zwischen Standard- und Pro-Modellen.
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

    try:
        # --- OPENAI ---
        # Prüft auf "OpenAI" (Standard) oder "OpenAI-Pro" (Premium)
        if consensus_model in ["OpenAI", "OpenAI-Pro"]:
            client = openai.OpenAI(api_key=api_keys.get("OpenAI"))
            # WICHTIG: Hier wird das Modell gewählt
            model_to_use = "gpt-5.5" if consensus_model == "OpenAI-Pro" else DEFAULT_OPENAI_MODEL
            
            kwargs = {
                "model": model_to_use,
                "messages": [
                    {"role": "system", "content": " "},
                    {"role": "user", "content": consensus_prompt}
                ]
            }
            if "gpt-5" in model_to_use or "o" in model_to_use:
                kwargs["max_completion_tokens"] = cfg.CONSENSUS_MAX_TOKENS
            else:
                kwargs["max_tokens"] = cfg.CONSENSUS_MAX_TOKENS
                
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()

        # --- MISTRAL ---
        elif consensus_model in ["Mistral", "Mistral-Pro"]:
            model_to_use = MISTRAL_PRO_MODEL if consensus_model == "Mistral-Pro" else DEFAULT_MISTRAL_MODEL

            return _mistral_chat_complete(
                api_keys.get("Mistral"),
                model_to_use,
                messages=[
                    {"role": "system", "content": ""},
                    {"role": "user", "content": consensus_prompt}
                ],
                max_tokens=cfg.CONSENSUS_MAX_TOKENS,
            )

        # --- ANTHROPIC ---
        elif consensus_model in ["Anthropic", "Anthropic-Pro"]:
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": api_keys.get("Anthropic"),
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            model_to_use = ANTHROPIC_PRO_MODEL if consensus_model == "Anthropic-Pro" else DEFAULT_ANTHROPIC_MODEL
            
            payload = {
                "model": model_to_use,
                "max_tokens": cfg.CONSENSUS_MAX_TOKENS,
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

        # --- GEMINI ---
        elif consensus_model in ["Gemini", "Gemini-Pro", cfg.GEMINI_FRONTIER_LOW_MODEL]:
            gemini_key = api_keys.get("Gemini")
            if consensus_model == cfg.GEMINI_FRONTIER_LOW_MODEL:
                return _gemini_generate_content_rest(
                    consensus_prompt,
                    gemini_key,
                    cfg.GEMINI_FRONTIER_LOW_MODEL,
                    int(cfg.CONSENSUS_MAX_TOKENS),
                )

            if gemini_key and gemini_key.strip() != "":
                genai.configure(api_key=gemini_key)
            else:
                genai.configure()

            # Flash vs Pro
            model_name = GEMINI_PRO_MODEL if consensus_model == "Gemini-Pro" else GEMINI_FLASH_MODEL
            
            model = genai.GenerativeModel(model_name)
            generation_config = {"max_output_tokens": int(cfg.CONSENSUS_MAX_TOKENS)}

            response = model.generate_content(
                consensus_prompt,
                generation_config=generation_config
            )
            return (response.text or "").strip() or "Error: Empty response payload."

        # --- DEEPSEEK ---
        elif consensus_model in ["DeepSeek", "DeepSeek-Pro"]:
            client = openai.OpenAI(api_key=api_keys.get("DeepSeek"), base_url="https://api.deepseek.com")
            # Chat vs Reasoner
            model_to_use = "deepseek-v4-pro" if consensus_model == "DeepSeek-Pro" else DEFAULT_DEEPSEEK_MODEL
            
            response = client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": " "},
                    {"role": "user", "content": consensus_prompt}
                ],
                max_tokens=cfg.CONSENSUS_MAX_TOKENS
            )
            return response.choices[0].message.content.strip()

        # --- GROK ---
        elif consensus_model in ["Grok", "Grok-Pro"]:
            client = openai.OpenAI(api_key=api_keys.get("Grok"), base_url="https://api.x.ai/v1")
            # Fast vs Latest (Strong)
            model_to_use = "grok-4.3" if consensus_model == "Grok-Pro" else DEFAULT_GROK_MODEL
            
            response = client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": " "},
                    {"role": "user", "content": consensus_prompt}
                ],
                max_tokens=cfg.CONSENSUS_MAX_TOKENS
            )
            return response.choices[0].message.content.strip()

        else:
            return f"Invalid consensus model selected: {consensus_model}"
    except Exception as e:
        return f"Consensus error: {str(e)}"
    

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
    """Baut den Differences-Prompt. Gibt (prompt, anon_map) zurück oder None,
    wenn keine Modellantworten vorliegen."""

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

    return differences_prompt, anon_map


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

        differences.append({
            "claim": claim,
            "type": diff_type,
            "positions": positions,
            "verify": _clip(entry.get("verify"), MAX_DIFF_TEXT_CHARS),
        })
        if len(differences) >= MAX_DIFF_ENTRIES:
            break
    return differences


def _legacy_differences_text(data: dict) -> str:
    """Synthetisiert aus den strukturierten Daten den bisherigen Freitext
    (Credibility-Satz, Bullets, BestModel-Zeile), damit Bookmarks,
    Credibility-Frame und Leaderboard-Vote unverändert funktionieren."""
    differences = data.get("differences") or []
    contradictions = [d for d in differences if d.get("type") == "contradiction"]

    if not differences:
        credibility = "The consensus answer is **very** credible."
    elif not contradictions:
        credibility = "The consensus answer is **largely** credible."
    elif len(contradictions) == 1:
        credibility = "The consensus answer is **partially** credible."
    else:
        credibility = "The consensus answer is **hardly** credible."

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


def parse_differences_payload(raw: str, anon_map: dict):
    """Parst die JSON-Ausgabe des Differences-Calls und übersetzt die
    anonymisierten Labels zurück. Gibt (data | None, legacy_text) zurück;
    bei unparsbarer Ausgabe ist data None und legacy_text der Rohtext
    (mit rückübersetzter BestModel-Zeile, falls vorhanden)."""
    parsed = _extract_json_object(raw)
    if parsed is None:
        return None, _translate_best_model(str(raw or "").strip(), anon_map)

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

    differences_prompt, anon_map = built

    try:
        # OPENAI
        if differences_model in ["OpenAI", "OpenAI-Pro"]:
            client = openai.OpenAI(api_key=api_keys.get("OpenAI"))
            model_to_use = "gpt-5.5" if differences_model == "OpenAI-Pro" else DEFAULT_OPENAI_MODEL
            
            kwargs = {
                "model": model_to_use,
                "messages": [
                    {"role": "system", "content": "Answer in the exact same language as the Model responses."},
                    {"role": "user", "content": differences_prompt}
                ]
            }
            if "gpt-5" in model_to_use or "o" in model_to_use:
                kwargs["max_completion_tokens"] = cfg.DIFFERENCES_MAX_TOKENS
            else:
                kwargs["max_tokens"] = cfg.DIFFERENCES_MAX_TOKENS
                
            response = client.chat.completions.create(**kwargs)
            result = response.choices[0].message.content.strip()

        # MISTRAL
        elif differences_model in ["Mistral", "Mistral-Pro"]:
            model_to_use = MISTRAL_PRO_MODEL if differences_model == "Mistral-Pro" else DEFAULT_MISTRAL_MODEL
            result = _mistral_chat_complete(
                api_keys.get("Mistral"),
                model_to_use,
                messages=[
                    {"role": "system", "content": "Answer in the exact same language as the Model responses."},
                    {"role": "user", "content": differences_prompt}
                ],
                max_tokens=cfg.DIFFERENCES_MAX_TOKENS,
            )

        elif differences_model in ["Anthropic", "Anthropic-Pro"]:
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": api_keys.get("Anthropic"),
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            model_to_use = ANTHROPIC_PRO_MODEL if differences_model == "Anthropic-Pro" else DEFAULT_ANTHROPIC_MODEL
            payload = {
                "model": model_to_use,
                "max_tokens": cfg.DIFFERENCES_MAX_TOKENS,
                "system": "Answer in the exact same language as the Model responses.",
                "messages": [{"role": "user", "content": differences_prompt}]
            }
            resp = requests.post(url, json=payload, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                result = data["content"][0]["text"] if data.get("content") else ""
            else:
                return f"Error with Anthropic: {resp.status_code} - {resp.text}", None

        elif differences_model in ["Gemini", "Gemini-Pro", cfg.GEMINI_FRONTIER_LOW_MODEL]:
            try:
                if differences_model == cfg.GEMINI_FRONTIER_LOW_MODEL:
                    result = _gemini_generate_content_rest(
                        differences_prompt,
                        api_keys.get("Gemini"),
                        cfg.GEMINI_FRONTIER_LOW_MODEL,
                        int(cfg.DIFFERENCES_MAX_TOKENS),
                    )
                else:
                    if api_keys.get("Gemini"):
                        genai.configure(api_key=api_keys["Gemini"])
                    elif os.environ.get("DEVELOPER_GEMINI_API_KEY"):
                        genai.configure(api_key=os.environ["DEVELOPER_GEMINI_API_KEY"])
                    else:
                        genai.configure()

                    model = genai.GenerativeModel(
                        model_name=GEMINI_FLASH_MODEL,
                        system_instruction="Answer in the exact same language as the Model responses.",
                        safety_settings=[{"category":"HARM_CATEGORY_HARASSMENT","threshold":"BLOCK_ONLY_HIGH"}],
                        generation_config={"max_output_tokens": int(cfg.DIFFERENCES_MAX_TOKENS), "temperature": 0.2}
                    )

                    resp = model.generate_content(differences_prompt)
                    result = (getattr(resp, "text", None) or "").strip()
                    if not result:
                        cand = (getattr(resp, "candidates", []) or [None])[0]
                        fr = getattr(cand, "finish_reason", None)
                        frs = str(fr)
                        if frs in ("2","FinishReason.SAFETY","SAFETY"):
                            return "Error with Gemini (differences): response was blocked by safety filters.", None
                        if frs in ("3","FinishReason.MAX_TOKENS","MAX_TOKENS"):
                            return "Error with Gemini (differences): hit max tokens before returning text.", None
                        return f"Error with Gemini (differences): empty candidate (finish_reason={frs}).", None

            except Exception as e:
                return f"Error with Gemini (differences): {e}", None

        elif differences_model in ["DeepSeek", "DeepSeek-Pro"]:
            client = openai.OpenAI(api_key=api_keys.get("DeepSeek"), base_url="https://api.deepseek.com")
            response = client.chat.completions.create(
                model=DEFAULT_DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": "Answer in the exact same language as the Model responses."},
                    {"role": "user", "content": differences_prompt}
                ],
                max_tokens=cfg.DIFFERENCES_MAX_TOKENS
            )
            result = response.choices[0].message.content.strip()

        elif differences_model in ["Grok", "Grok-Pro"]:
            client = openai.OpenAI(api_key=api_keys.get("Grok"), base_url="https://api.x.ai/v1")
            response = client.chat.completions.create(
                model=DEFAULT_GROK_MODEL,
                messages=[
                    {"role": "system", "content": "Answer in the exact same language as the Model responses."},
                    {"role": "user", "content": differences_prompt}
                ],
                max_tokens=cfg.DIFFERENCES_MAX_TOKENS
            )
            result = response.choices[0].message.content.strip()

        else:
            return "Invalid model selected for difference comparison.", None

    except Exception as e:
        return f"Error in comparison: {e}", None

    if not result:
        return "Error in comparison: empty result from differences engine.", None

    # JSON parsen, Labels rückübersetzen, Legacy-Text synthetisieren
    data, legacy_text = parse_differences_payload(result, anon_map)
    return legacy_text, data


# ---------------------------------------------------------------------------
# Streaming-Varianten: liefern {"type": "delta", "text": ...} Events und am
# Ende {"type": "final", "text": <Gesamttext>}. Fehler werden – wie bei den
# nicht-streamenden Varianten – als Fehlertext im final-Event transportiert.
# ---------------------------------------------------------------------------

class _InvalidEngineError(Exception):
    pass


def _stream_consensus_engine(consensus_model: str, api_keys: dict, consensus_prompt: str):
    from app.services.llm.streaming import (
        stream_anthropic_text,
        stream_chat_completion_text,
        stream_gemini_prompt_text,
        stream_gemini_text,
        stream_mistral_chat_text,
    )

    max_tokens = int(cfg.CONSENSUS_MAX_TOKENS)

    if consensus_model in ["OpenAI", "OpenAI-Pro"]:
        model_to_use = "gpt-5.5" if consensus_model == "OpenAI-Pro" else DEFAULT_OPENAI_MODEL
        token_param = "max_completion_tokens" if ("gpt-5" in model_to_use or "o" in model_to_use) else "max_tokens"
        yield from stream_chat_completion_text(
            api_key=api_keys.get("OpenAI"),
            model=model_to_use,
            messages=[
                {"role": "system", "content": " "},
                {"role": "user", "content": consensus_prompt}
            ],
            max_tokens=max_tokens,
            token_param=token_param,
        )

    elif consensus_model in ["Mistral", "Mistral-Pro"]:
        model_to_use = MISTRAL_PRO_MODEL if consensus_model == "Mistral-Pro" else DEFAULT_MISTRAL_MODEL
        yield from stream_mistral_chat_text(
            api_key=api_keys.get("Mistral"),
            model=model_to_use,
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": consensus_prompt}
            ],
            max_tokens=max_tokens,
        )

    elif consensus_model in ["Anthropic", "Anthropic-Pro"]:
        model_to_use = ANTHROPIC_PRO_MODEL if consensus_model == "Anthropic-Pro" else DEFAULT_ANTHROPIC_MODEL
        yield from stream_anthropic_text(
            api_key=api_keys.get("Anthropic"),
            model=model_to_use,
            system="",
            prompt=consensus_prompt,
            max_tokens=max_tokens,
        )

    elif consensus_model in ["Gemini", "Gemini-Pro", cfg.GEMINI_FRONTIER_LOW_MODEL]:
        gemini_key = api_keys.get("Gemini")
        if consensus_model == cfg.GEMINI_FRONTIER_LOW_MODEL:
            yield from stream_gemini_prompt_text(
                api_key=gemini_key,
                model_override=cfg.GEMINI_FRONTIER_LOW_MODEL,
                prompt=consensus_prompt,
                max_tokens=max_tokens,
            )
        else:
            model_name = GEMINI_PRO_MODEL if consensus_model == "Gemini-Pro" else GEMINI_FLASH_MODEL
            yield from stream_gemini_text(
                api_key=gemini_key,
                model=model_name,
                prompt=consensus_prompt,
                max_tokens=max_tokens,
            )

    elif consensus_model in ["DeepSeek", "DeepSeek-Pro"]:
        model_to_use = "deepseek-v4-pro" if consensus_model == "DeepSeek-Pro" else DEFAULT_DEEPSEEK_MODEL
        yield from stream_chat_completion_text(
            api_key=api_keys.get("DeepSeek"),
            base_url="https://api.deepseek.com",
            model=model_to_use,
            messages=[
                {"role": "system", "content": " "},
                {"role": "user", "content": consensus_prompt}
            ],
            max_tokens=max_tokens,
        )

    elif consensus_model in ["Grok", "Grok-Pro"]:
        model_to_use = "grok-4.3" if consensus_model == "Grok-Pro" else DEFAULT_GROK_MODEL
        yield from stream_chat_completion_text(
            api_key=api_keys.get("Grok"),
            base_url="https://api.x.ai/v1",
            model=model_to_use,
            messages=[
                {"role": "system", "content": " "},
                {"role": "user", "content": consensus_prompt}
            ],
            max_tokens=max_tokens,
        )

    else:
        raise _InvalidEngineError(f"Invalid consensus model selected: {consensus_model}")


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

    parts = []
    try:
        for text in _stream_consensus_engine(consensus_model, api_keys, consensus_prompt):
            parts.append(text)
            yield {"type": "delta", "text": text}
    except _InvalidEngineError as e:
        yield {"type": "final", "text": str(e)}
        return
    except Exception as e:
        yield {"type": "final", "text": f"Consensus error: {str(e)}"}
        return

    final_text = "".join(parts).strip()
    if not final_text:
        final_text = "Consensus error: empty response from consensus engine."
    yield {"type": "final", "text": final_text}


def _stream_differences_engine(differences_model: str, api_keys: dict, differences_prompt: str):
    from app.services.llm.streaming import (
        stream_anthropic_text,
        stream_chat_completion_text,
        stream_gemini_prompt_text,
        stream_gemini_text,
        stream_mistral_chat_text,
    )

    max_tokens = int(cfg.DIFFERENCES_MAX_TOKENS)
    system_text = "Answer in the exact same language as the Model responses."

    if differences_model in ["OpenAI", "OpenAI-Pro"]:
        model_to_use = "gpt-5.5" if differences_model == "OpenAI-Pro" else DEFAULT_OPENAI_MODEL
        token_param = "max_completion_tokens" if ("gpt-5" in model_to_use or "o" in model_to_use) else "max_tokens"
        yield from stream_chat_completion_text(
            api_key=api_keys.get("OpenAI"),
            model=model_to_use,
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": differences_prompt}
            ],
            max_tokens=max_tokens,
            token_param=token_param,
        )

    elif differences_model in ["Mistral", "Mistral-Pro"]:
        model_to_use = MISTRAL_PRO_MODEL if differences_model == "Mistral-Pro" else DEFAULT_MISTRAL_MODEL
        yield from stream_mistral_chat_text(
            api_key=api_keys.get("Mistral"),
            model=model_to_use,
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": differences_prompt}
            ],
            max_tokens=max_tokens,
        )

    elif differences_model in ["Anthropic", "Anthropic-Pro"]:
        model_to_use = ANTHROPIC_PRO_MODEL if differences_model == "Anthropic-Pro" else DEFAULT_ANTHROPIC_MODEL
        yield from stream_anthropic_text(
            api_key=api_keys.get("Anthropic"),
            model=model_to_use,
            system=system_text,
            prompt=differences_prompt,
            max_tokens=max_tokens,
        )

    elif differences_model in ["Gemini", "Gemini-Pro", cfg.GEMINI_FRONTIER_LOW_MODEL]:
        gemini_key = api_keys.get("Gemini") or os.environ.get("DEVELOPER_GEMINI_API_KEY")
        if differences_model == cfg.GEMINI_FRONTIER_LOW_MODEL:
            yield from stream_gemini_prompt_text(
                api_key=gemini_key,
                model_override=cfg.GEMINI_FRONTIER_LOW_MODEL,
                prompt=differences_prompt,
                max_tokens=max_tokens,
            )
        else:
            yield from stream_gemini_text(
                api_key=gemini_key,
                model=GEMINI_FLASH_MODEL,
                prompt=differences_prompt,
                max_tokens=max_tokens,
                system=system_text,
                temperature=0.2,
            )

    elif differences_model in ["DeepSeek", "DeepSeek-Pro"]:
        yield from stream_chat_completion_text(
            api_key=api_keys.get("DeepSeek"),
            base_url="https://api.deepseek.com",
            model=DEFAULT_DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": differences_prompt}
            ],
            max_tokens=max_tokens,
        )

    elif differences_model in ["Grok", "Grok-Pro"]:
        yield from stream_chat_completion_text(
            api_key=api_keys.get("Grok"),
            base_url="https://api.x.ai/v1",
            model=DEFAULT_GROK_MODEL,
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": differences_prompt}
            ],
            max_tokens=max_tokens,
        )

    else:
        raise _InvalidEngineError("Invalid model selected for difference comparison.")


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

    differences_prompt, anon_map = built

    parts = []
    try:
        for text in _stream_differences_engine(differences_model, api_keys, differences_prompt):
            parts.append(text)
            # Roh-JSON wird im Frontend nicht gerendert; die Deltas halten nur
            # die SSE-Verbindung aktiv.
            yield {"type": "delta", "text": text}
    except _InvalidEngineError as e:
        yield {"type": "final", "text": str(e), "data": None}
        return
    except Exception as e:
        yield {"type": "final", "text": f"Error in comparison: {str(e)}", "data": None}
        return

    result = "".join(parts).strip()
    if not result:
        yield {"type": "final", "text": "Error in comparison: empty result from differences engine.", "data": None}
        return

    data, legacy_text = parse_differences_payload(result, anon_map)
    yield {"type": "final", "text": legacy_text, "data": data}
