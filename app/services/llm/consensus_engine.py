from __future__ import annotations

import os
import re
import logging
import random
import requests
import openai
from mistralai import Mistral
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
    DEFAULT_ANTHROPIC_MODEL,
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
    model_sources=None,
) -> str:
    """
    Konsolidiert die Antworten der 6 Haupt-LLMs zu einer Konsensantwort.
    Unterscheidet jetzt zwischen Standard- und Pro-Modellen.
    """
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

    if best_model:
        prompt_parts.append(
            f"The user marked the Answer from the Model: {best_model} as the best one. "
            "You receive multiple expert opinions on a specific question. "
            "Your task is to combine these responses into a comprehensive, correct, and coherent answer. "
            "Note: Experts can also make mistakes. Therefore, try to identify and exclude possible errors by comparing the answers. "
            "Structure the answer clearly and coherently. "
            + user_facing_instruction
        )
    else:
        prompt_parts.append(
            "You receive multiple expert opinions on a specific question. "
            "Treat all expert opinions equally. Do not focus on the answer of one model. "
            "Your task is to combine these responses into a comprehensive, correct, and coherent answer. "
            "Note: Experts can also make mistakes. Therefore, try to identify and exclude possible errors by comparing the answers. "
            "Structure the answer clearly and coherently. "
            + user_facing_instruction
        )

    consensus_prompt = "".join(prompt_parts)

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
            client = Mistral(api_key=api_keys.get("Mistral"))
            # Standard & Pro nutzen aktuell beide 'large', ansonsten hier anpassen
            model_to_use = "mistral-large-latest" if consensus_model == "Mistral-Pro" else DEFAULT_MISTRAL_MODEL
            
            response = client.chat.complete(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": ""},
                    {"role": "user", "content": consensus_prompt}
                ],
                max_tokens=cfg.CONSENSUS_MAX_TOKENS
            )
            return response.choices[0].message.content.strip()

        # --- ANTHROPIC ---
        elif consensus_model in ["Anthropic", "Anthropic-Pro"]:
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": api_keys.get("Anthropic"),
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            # Haiku vs Sonnet 4.5
            model_to_use = "claude-opus-4-7" if consensus_model == "Anthropic-Pro" else DEFAULT_ANTHROPIC_MODEL
            
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
) -> str:
    """
    Extrahiert die Unterschiede zwischen den Antworten der 6 Hauptmodelle,
    anonymisiert die Modellnamen und ordnet das bestbewertete Modell anschließend wieder zu.
    """

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
        "If there are clear contradictions, it is hardly or not credible. "
        "Respond with one of the following sentences:\n\n"

        "- 'The consensus answer is **very** credible.'\n"
        "- 'The consensus answer is **largely** credible.'\n"
        "- 'The consensus answer is **partially** credible.'\n"
        "- 'The consensus answer is **hardly** credible.'\n"
        "- 'The consensus answer is **not** credible.'\n\n"

        "After the sentence, include a separator line and a concise differences summary. "
        "Keep it short and scannable: maximum two bullet points, each under 18 words. "
        "Mention only the most important substantive deviations. Do not write long paragraphs or restate every model. "
        "Ignore citation markers, source labels, URLs, and source-list noise unless they reveal a real factual disagreement.\n\n"
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
        "- [Most important difference, under 18 words.]\n"
        "- [Optional second important difference, under 18 words.]\n\n"
        "(Info: Mark the model closest to the consensus as Best Model)\n"
        "BestModel: [Model name]"
    )

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
            client = Mistral(api_key=api_keys.get("Mistral"))
            response = client.chat.complete(
                model=DEFAULT_MISTRAL_MODEL,
                messages=[
                    {"role": "system", "content": "Answer in the exact same language as the Model responses."},
                    {"role": "user", "content": differences_prompt}
                ],
                max_tokens=cfg.DIFFERENCES_MAX_TOKENS
            )
            result = response.choices[0].message.content.strip()

        elif differences_model in ["Anthropic", "Anthropic-Pro"]:
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": api_keys.get("Anthropic"),
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            payload = {
                "model": DEFAULT_ANTHROPIC_MODEL,
                "max_tokens": cfg.DIFFERENCES_MAX_TOKENS,
                "system": "Answer in the exact same language as the Model responses.",
                "messages": [{"role": "user", "content": differences_prompt}]
            }
            resp = requests.post(url, json=payload, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                result = data["content"][0]["text"] if data.get("content") else ""
            else:
                return f"Error with Anthropic: {resp.status_code} - {resp.text}"

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
                            return "Error with Gemini (differences): response was blocked by safety filters."
                        if frs in ("3","FinishReason.MAX_TOKENS","MAX_TOKENS"):
                            return "Error with Gemini (differences): hit max tokens before returning text."
                        return f"Error with Gemini (differences): empty candidate (finish_reason={frs})."

            except Exception as e:
                return f"Error with Gemini (differences): {e}"

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
            return "Invalid model selected for difference comparison."

    except Exception as e:
        return f"Error in comparison: {e}"

    if not result:
        return "Error in comparison: empty result from differences engine."

    # BestModel-Zeile rückübersetzen
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
