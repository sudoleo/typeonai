"""Deterministische LLM-Mocks fuer die Playwright-E2E-Suite (MOCK_LLM=1).

Kein Produktionspfad: alle Hooks pruefen mock_llm_enabled() und sind No-ops,
solange die Env-Variable nicht gesetzt ist. Die Mocks setzen bewusst an den
untersten Seams an (Provider-Call in _run_ask, Engine-Dispatch in
consensus_engine), damit SSE-Packing, JSON-Parsing, Anchor-/Quote-
Verifikation und Agreement-Score im Test echt durchlaufen.

Fixture-Vertrag (wichtig fuer die Zitat-Verifikation in consensus_engine):
- Jede Modellantwort ist EINE Zeile (der Differences-Prompt listet Antworten
  zeilenweise als "- Model X: ...", mehrzeilige Antworten wuerden das
  Label-Parsing des Mocks brechen).
- Alle Antworten ausser Grok enthalten SHARED_FACT woertlich; Grok enthaelt
  DISSENT_FACT. Anchors/Quotes im Mock-Differences-JSON sind Substrings
  dieser Saetze, sonst leert _verify_differences_data sie.
"""

import json
import os
import re
import time


def mock_llm_enabled() -> bool:
    return os.environ.get("MOCK_LLM") == "1"


def _delay_seconds() -> float:
    """Delta-Delay, damit Streaming im Test als Zwischenzustand sichtbar ist."""
    try:
        return max(0.0, float(os.environ.get("MOCK_LLM_DELAY_MS", "25"))) / 1000.0
    except ValueError:
        return 0.025


SHARED_FACT = "The Eiffel Tower is located in Paris and was completed in 1889."
DISSENT_FACT = "The Eiffel Tower was completed in 1887, two years earlier than commonly stated."

MOCK_MODEL_ANSWERS = {
    "OpenAI": f"**OpenAI mock answer.** {SHARED_FACT} It stands about 330 metres tall.",
    "Mistral": f"**Mistral mock answer.** {SHARED_FACT} It was built for the World's Fair.",
    "Anthropic": f"**Claude mock answer.** {SHARED_FACT} Gustave Eiffel's company built it.",
    "Gemini": f"**Gemini mock answer.** {SHARED_FACT} It is visited by millions of people each year.",
    "DeepSeek": f"**DeepSeek mock answer.** {SHARED_FACT} It was the world's tallest structure until 1930.",
    "Grok": f"**Grok mock answer.** {DISSENT_FACT} It stands about 330 metres tall.",
}

MOCK_CONSENSUS_TEXT = (
    "## Mock consensus\n\n"
    f"{SHARED_FACT} It stands about 330 metres tall and was built for the World's Fair.\n\n"
    "One model dates the completion to 1887, which contradicts the other responses."
)


def _chunks(text: str, size: int = 12):
    for start in range(0, len(text), size):
        yield text[start:start + size]


def mock_ask_stream(provider_label: str, question: str):
    """Ersatz fuer provider.stream_fn: liefert dieselben StreamEvents
    ({type: delta/final}), die streaming_model_response erwartet."""
    text = MOCK_MODEL_ANSWERS.get(provider_label, f"**{provider_label} mock answer.** {SHARED_FACT}")
    delay = _delay_seconds()
    for chunk in _chunks(text):
        if delay:
            time.sleep(delay)
        yield {"type": "delta", "text": chunk}
    yield {"type": "final", "result": {"text": text, "sources": []}}


def mock_ask_result(provider_label: str, question: str):
    """Ersatz fuer provider.query_fn (nicht-streamender Pfad)."""
    text = MOCK_MODEL_ANSWERS.get(provider_label, f"**{provider_label} mock answer.** {SHARED_FACT}")
    return {"text": text, "sources": []}


def _build_mock_differences_json(prompt: str) -> str:
    """Baut das Judge-JSON aus dem echten Differences-Prompt.

    Die Anonymisierung (Model A/B/...) wird pro Aufruf zufaellig gemischt;
    der Mock ermittelt das dissentierende Label daher aus dem Prompt selbst:
    es ist die Antwortzeile, die den DISSENT_FACT-Marker "1887" enthaelt.
    """
    labeled = re.findall(r"^- (Model [A-Z]): (.*)$", prompt, flags=re.MULTILINE)
    labels = [label for label, _ in labeled]
    dissent_label = next((label for label, text in labeled if "1887" in text), None)
    agree = [label for label in labels if label != dissent_label]

    claims = [
        {
            "anchor": "The Eiffel Tower is located in Paris",
            "agree": agree or labels,
            "dissent": [],
        },
        {
            "anchor": "was completed in 1889",
            "agree": agree or labels,
            "dissent": (
                [{"model": dissent_label, "quote": "was completed in 1887"}]
                if dissent_label else []
            ),
        },
    ]

    differences = []
    if dissent_label and agree:
        differences.append({
            "claim": "Completion year of the Eiffel Tower",
            "type": "contradiction",
            "severity": "major",
            "positions": [
                {"stance": "Completed in 1889", "models": agree[:2], "quote": "was completed in 1889"},
                {"stance": "Completed in 1887", "models": [dissent_label], "quote": "was completed in 1887"},
            ],
            "verify": "Check the completion year of the Eiffel Tower.",
        })

    return json.dumps({
        "claims": claims,
        "differences": differences,
        "best_model": (agree or labels)[0] if labels else "Model A",
    })


def _mock_engine_output(prompt: str, json_mode: bool) -> str:
    if json_mode:
        if "Compare the OLD and NEW consensus answers" in prompt:
            return json.dumps({"changed": False, "severity": "minor", "change_summary": "No material change."})
        if '"claims"' in prompt:
            return _build_mock_differences_json(prompt)
        # Fremder Structured-Output-Call (z. B. Resolve-Runde): neutrales,
        # schema-kompatibles Minimal-JSON statt Differences-Payload.
        return json.dumps({"decision": "maintain", "position": "Mock position.", "reason": "Mock reason."})
    return MOCK_CONSENSUS_TEXT


def mock_engine_text(prompt: str, json_mode: bool) -> str:
    """Ersatz fuer consensus_engine._call_engine_text."""
    return _mock_engine_output(prompt, json_mode)


def mock_engine_stream(prompt: str, json_mode: bool):
    """Ersatz fuer consensus_engine._stream_engine_text (yieldet Text-Chunks)."""
    text = _mock_engine_output(prompt, json_mode)
    delay = _delay_seconds()
    for chunk in _chunks(text, size=24):
        if delay:
            time.sleep(delay)
        yield chunk
