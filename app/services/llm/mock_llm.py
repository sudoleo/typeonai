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
  DISSENT_FACT. Quotes im Mock-Differences-JSON sind Substrings dieser Saetze,
  sonst leert _verify_differences_data sie.
- Differences zeigen ueber die Satznummer ("s") in die Konsensantwort, der
  Coverage-Judge ueber die Satz-ID ("s3"). Beide liest der Mock aus dem
  nummerierten Konsenstext im Prompt (_mock_sentence_number).
- Der Coverage-Mock deckt die verbindliche ID-Liste VOLLSTAENDIG ab; sonst
  laeuft im Test die Nachforderung an, die es in der Fixture nicht gibt.
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


def _mock_sentence_number(prompt: str, needle: str, default: int) -> int:
    """Nummer des Konsens-Satzes, der `needle` enthaelt.

    Der Judge referenziert Saetze inzwischen ueber ihre Nummer statt ueber eine
    Abschrift (siehe _enumerate_consensus_sentences). Der Mock liest die
    Nummern aus dem echten Prompt aus, damit die Fixture nicht an einer
    hartkodierten Zaehlung haengt, sobald sich MOCK_CONSENSUS_TEXT aendert.
    """
    _, _, numbered = prompt.partition("Consensus answer (sentences numbered):")
    # Bis zur naechsten Marke, nicht bis zum Zeilenende: mehrere Saetze koennen
    # in derselben Zeile stehen.
    for number, text in re.findall(r"\[(\d+)\]\s*([^\[\n]*)", numbered):
        if needle in text:
            return int(number)
    return default


def _mock_labels(prompt: str):
    """(labels, dissentierendes Label) aus dem echten Prompt.

    Die Anonymisierung (Model A/B/...) wird pro Aufruf zufaellig gemischt;
    der Mock ermittelt das dissentierende Label daher aus dem Prompt selbst:
    es ist die Antwortzeile, die den DISSENT_FACT-Marker "1887" enthaelt.
    """
    labeled = re.findall(r"^- (Model [A-Z]): (.*)$", prompt, flags=re.MULTILINE)
    labels = [label for label, _ in labeled]
    dissent_label = next((label for label, text in labeled if "1887" in text), None)
    return labels, dissent_label


def _build_mock_coverage_json(prompt: str) -> str:
    """Baut das Coverage-JSON aus dem echten Coverage-Prompt.

    Deckt die verbindliche ID-Liste vollstaendig ab - genau das prueft der
    Server danach. Der dritte Konsens-Satz ("One model dates ...") wird
    bewusst als `context_only` eingestuft: so laeuft im Test auch der Pfad
    durch, auf dem ein Satz ausdruecklich KEINE Marke bekommt.
    """
    labels, dissent_label = _mock_labels(prompt)
    _, _, listed = prompt.partition("Binding list of sentence ids (one entry each, in this order):")
    ids = re.findall(r'"(s\d+)"', listed.split("\n\n", 1)[0])

    disputed = f"s{_mock_sentence_number(prompt, '1889', 1)}"
    meta = f"s{_mock_sentence_number(prompt, 'One model dates', 3)}"

    sentences = []
    for key in ids:
        if key == meta:
            sentences.append({
                "id": key,
                "classification": "context_only",
                "models": {label: "not_addressed" for label in labels},
                "counter_quotes": [],
            })
            continue
        contradicts = key == disputed and bool(dissent_label)
        sentences.append({
            "id": key,
            "classification": "claim",
            "models": {
                label: ("contradicts" if contradicts and label == dissent_label else "supports")
                for label in labels
            },
            "counter_quotes": (
                [{"model": dissent_label, "quote": "was completed in 1887"}]
                if contradicts else []
            ),
        })
    return json.dumps({"sentences": sentences})


def _build_mock_differences_json(prompt: str) -> str:
    """Baut das Judge-JSON aus dem echten Differences-Prompt."""
    labels, dissent_label = _mock_labels(prompt)
    agree = [label for label in labels if label != dissent_label]

    disputed_sentence = _mock_sentence_number(prompt, "1889", 1)

    differences = []
    if dissent_label and agree:
        differences.append({
            "claim": "Completion year of the Eiffel Tower",
            # Zeigt bewusst auf denselben Satz wie der dissentierende Claim:
            # dort darf nur EIN sichtbares Signal stehen (Badge statt Marker).
            "s": disputed_sentence,
            "type": "contradiction",
            "severity": "major",
            "positions": [
                {"stance": "Completed in 1889", "models": agree[:2], "quote": "was completed in 1889"},
                {"stance": "Completed in 1887", "models": [dissent_label], "quote": "was completed in 1887"},
            ],
            "verify": "Check the completion year of the Eiffel Tower.",
        })

    return json.dumps({
        "differences": differences,
        "best_model": (agree or labels)[0] if labels else "Model A",
    })


def _mock_engine_output(prompt: str, json_mode: bool) -> str:
    if json_mode:
        if "Compare the OLD and NEW consensus answers" in prompt:
            return json.dumps({"changed": False, "severity": "minor", "change_summary": "No material change."})
        if '"current_question"' in prompt:
            # Frage-Aufloesung vor dem Fan-out (chat_context.ChatMemoryCompressor).
            return json.dumps({
                "depends_on_previous_turn": True,
                "resolved_question": "Mock resolved follow-up question.",
            })
        if '"counter_quotes"' in prompt:
            return _build_mock_coverage_json(prompt)
        if '"differences"' in prompt:
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
