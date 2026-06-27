"""Eingefrorene Benchmark-Konfiguration (E6).

Alle Werte, die den finalen 98-Fragen-Snapshot reproduzierbar machen, stehen
hier explizit – **unabhaengig** von ``load_models_from_db()`` (Firestore kann die
``MODEL_CONFIGS`` beim Startup ueberschreiben; der Benchmark darf davon nicht
abhaengen, siehe Plan §2).

Die sechs Modell-Pins (``internal_id`` + aufgeloestes ``api_model``) sind hier
hart eingefroren. ``assert_pins_match_config()`` prueft beim Dry-Run/Manifest, ob
die produktive ``resolve_api_model``-Aufloesung noch mit den Pins uebereinstimmt –
Drift wird damit sichtbar, statt still zu passieren.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import app.core.config as cfg

# --- Pfade -----------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "benchmark"
SMOKE_MANIFEST = DATA_DIR / "mmlu_pro_smoke_v1.json"
PILOT_MANIFEST = DATA_DIR / "mmlu_pro_pilot_v1.json"
SAMPLE_MANIFEST = DATA_DIR / "mmlu_pro_sample_v1.json"
RUNS_DIR = DATA_DIR / "runs"

# --- MMLU-Pro Quelle -------------------------------------------------------

HF_REPO_ID = "TIGER-Lab/MMLU-Pro"
HF_REPO_TYPE = "dataset"
HF_TEST_GLOB = "data/test-*.parquet"

# --- Sampling (E3 / E3b) ---------------------------------------------------

PILOT_SIZE = 5
SMOKE_SIZE = 1
QUESTIONS_PER_CATEGORY = 7
EXPECTED_CATEGORIES = 14
FINAL_SIZE = QUESTIONS_PER_CATEGORY * EXPECTED_CATEGORIES  # 98
PILOT_SEED = 20260627
FINAL_SEED = 98140314

# --- Fixierter, reproduzierbarer closed-book System-Prompt -----------------
# Bewusst OHNE Tagesdatum (anders als app.services.llm.base.get_system_prompt()),
# damit der Lauf reproduzierbar bleibt (Plan §6).

SYSTEM_PROMPT = (
    "You are answering a single multiple-choice question from a closed-book exam. "
    "Rely only on your own knowledge. Do not use web search, external tools, or any "
    "outside sources. Reason as briefly as possible, then finish your reply with a "
    "final line in exactly this format:\n"
    "The answer is (X).\n"
    "where X is the single letter of the option you choose."
)

# --- Run-Einstellungen (E6: nach dem Pilot final einfrieren) ---------------

TEMPERATURE = 0.2  # nur Mistral/Gemini-Payloads tragen Temperatur (build_provider_payload);
# OpenAI/Anthropic/DeepSeek/Grok nutzen Provider-Defaults (im Manifest dokumentiert).
OUTPUT_TOKEN_LIMIT = 4096
INCLUDE_SYNTH_ALONE = True  # vierte Vergleichsgroesse "Synthesizer allein" (Plan §10)
DEFAULT_LABEL_MODE = "names"  # Hauptpfad mit Modellnamen (E5)


@dataclass(frozen=True)
class BenchmarkModel:
    """Ein eingefrorener Modell-Pin fuer den Benchmark."""

    provider: str
    internal_id: str
    api_model: str
    reasoning: str  # beschreibend; effektive Payload-Settings stehen im Manifest


# Die sechs Modelle = regulaere hochwertige Modellpfade fuer Smoke/Pilot/Final,
# ohne frontier-low Aliase. api_model ist hart eingefroren.
MODELS: tuple[BenchmarkModel, ...] = (
    BenchmarkModel("openai", "gpt-5.5", "gpt-5.5", "provider_default"),
    BenchmarkModel("mistral", cfg.MISTRAL_PRO_MODEL, cfg.MISTRAL_PRO_MODEL, "provider_default"),
    BenchmarkModel("anthropic", cfg.ANTHROPIC_PRO_MODEL, cfg.ANTHROPIC_PRO_MODEL, "provider_default"),
    BenchmarkModel("gemini", cfg.GEMINI_PRO_MODEL, cfg.GEMINI_PRO_MODEL, "provider_default"),
    BenchmarkModel("deepseek", cfg.DEEPSEEK_PRO_MODEL, cfg.DEEPSEEK_PRO_MODEL, "provider_default"),
    BenchmarkModel("grok", "grok-4.3", "grok-4.3", "provider_default"),
)

# Consensus-Synthese-Modell (Pin): regulaerer Gemini-3.1-Pro-Preview-Pfad.
# Synthesizer-alone nutzt exakt denselben Pin.
CONSENSUS_MODEL = cfg.GEMINI_PRO_MODEL

# Provider-Name -> API-Key-Schluessel, wie query_consensus ihn erwartet.
PROVIDER_API_KEY_NAME = {
    "openai": "OpenAI",
    "mistral": "Mistral",
    "anthropic": "Anthropic",
    "gemini": "Gemini",
    "deepseek": "DeepSeek",
    "grok": "Grok",
}

# --- Pricing-Tabelle (Plan §10) --------------------------------------------
# USD pro 1M Tokens je api_model. **Manuell gepflegte Schaetzungen** – im Repo
# existiert keine Pricing-Quelle; Kosten sind daher Naeherungen.

PRICING_USD_PER_1M: dict[str, dict[str, float]] = {
    "gpt-5.5": {"input": 1.25, "output": 10.0},
    cfg.MISTRAL_PRO_MODEL: {"input": 0.40, "output": 2.0},
    cfg.ANTHROPIC_PRO_MODEL: {"input": 5.0, "output": 25.0},
    cfg.GEMINI_PRO_MODEL: {"input": 1.25, "output": 10.0},
    cfg.DEEPSEEK_PRO_MODEL: {"input": 0.28, "output": 0.42},
    "grok-4.3": {"input": 3.0, "output": 15.0},
}


def resolve_pins() -> list[dict]:
    """Loest die Modell-Pins ueber die produktive ``resolve_api_model`` auf
    (zur Drift-Pruefung). Gibt je Modell den frozen + den aktuell aufgeloesten
    ``api_model`` zurueck."""
    rows = []
    for model in MODELS:
        resolved, _ = cfg.resolve_api_model(
            model.internal_id, cfg.DEFAULT_MODEL_BY_PROVIDER[model.provider], model.provider
        )
        rows.append(
            {
                "provider": model.provider,
                "internal_id": model.internal_id,
                "frozen_api_model": model.api_model,
                "resolved_api_model": resolved,
                "reasoning": model.reasoning,
                "matches": resolved == model.api_model,
            }
        )
    return rows


def assert_pins_match_config() -> None:
    """Bricht ab, falls die produktive Modellaufloesung von den eingefrorenen
    Pins abweicht (sichtbar gemachte Drift, Plan E6)."""
    mismatches = [row for row in resolve_pins() if not row["matches"]]
    if mismatches:
        details = ", ".join(
            f"{row['provider']}:{row['internal_id']} frozen={row['frozen_api_model']} "
            f"resolved={row['resolved_api_model']}"
            for row in mismatches
        )
        raise RuntimeError(f"Benchmark model pins drifted from config: {details}")
