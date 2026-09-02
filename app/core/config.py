from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any
from dotenv import load_dotenv

from app.core.entitlements import (
    TIER_FREE,
    TIER_PLUS,
    TIER_PRO,
    normalize_tier,
)

load_dotenv()


@dataclass(frozen=True)
class ModelConfig:
    internal_id: str
    provider: str
    api_model: str
    label: str
    is_free: bool = True
    is_pro: bool = False
    is_low_reasoning: bool = False
    accepts_attachments: bool = True
    request_config: dict[str, Any] = field(default_factory=dict)

DEFAULT_LIMITS = {
    # Persistente, run-basierte UTC-Tageslimits. Das Total zaehlt jeden
    # serverfinanzierten logischen Run genau einmal (Follow-ups eingeschlossen);
    # Deep Think fuehrt zusaetzlich ein separates Teilkontingent.
    # Drei Runs erlaubten einen Test, keine Gewohnheit — seit 2026-08-04 zwoelf.
    "free_consensus_run_limit": 12,
    # Plus laeuft auf denselben guenstigen Modellen wie Free und darf deshalb
    # das groesste Kontingent haben - mehr als Pro, dessen Runs ein Vielfaches
    # kosten koennen.
    "plus_consensus_run_limit": 750,
    "pro_consensus_run_limit": 500,
    "free_deep_think_run_limit": 0,
    "pro_deep_think_run_limit": 50,
    "free_max_words": 500,
    "plus_max_words": 500,
    "pro_max_words": 500,
    "free_deep_search_max_words": 0,
    "pro_deep_search_max_words": 1000,
    "free_max_tokens": 4096,
    "plus_max_tokens": 4096,
    "pro_max_tokens": 4096,
    "free_deep_search_max_tokens": 0,
    "pro_deep_search_max_tokens": 8192,
    "consensus_max_tokens": 8192,
    "differences_max_tokens": 8192,
    # Der Coverage-Judge schreibt eine Zeile pro Konsens-Satz und Modell
    # (bis zu 80 Saetze x 8 Familien). Das ist mehr Ausgabe als beim
    # Differences-Judge, aber auf einem guenstigen Standard-Judge.
    "coverage_max_tokens": 12288,
    # Serverseitige Eingabe-Caps fuer /consensus: Antworten/Frage kommen vom
    # Client und muessen begrenzt werden, bevor sie in den Engine-Prompt
    # fliessen (Kosten-/Abuse-Schutz). Grosszuegig gewaehlt, damit legitime
    # Deep-Search-Antworten (8192 Output-Tokens) nie gekappt werden.
    "consensus_max_answer_chars": 40_000,
    "consensus_max_question_chars": 8_000,
    # Serverseitige Caps fuer den Follow-up-Kontext (previous_question +
    # previous_consensus kommen vom Client). Bewusst enger als die
    # /consensus-Caps: der Kontext geht bei jeder Follow-up-Frage in alle
    # /ask_*-Prompts gleichzeitig ein (Kostenkontrolle).
    "followup_max_question_chars": 4_000,
    "followup_max_consensus_chars": 12_000,
    # Qualitätsfilter für index_eligible von Share-Snapshots (Etappe 3):
    # steuert nur die Eligibility-Anzeige, indexed setzt weiterhin der Admin.
    "share_min_consensus_chars": 600,
    "share_min_sources": 2,
    "share_min_models": 3,
    "share_question_min_chars": 15,
    "share_question_max_chars": 300,
    "watch_free_active_limit": 1,
    "watch_plus_active_limit": 3,
    "watch_pro_active_limit": 5,
    "watch_max_runs_per_day": 50,
    # 1 = taegliches Intervall bleibt Pro vorbehalten, 0 = auch Free darf
    # taeglich pruefen lassen (Boolean als 0/1, damit es ins numerische
    # Admin-Limits-Raster passt).
    "watch_daily_interval_requires_pro": 1,
    # 1 = Plus darf das taegliche Intervall auch dann, wenn es fuer Free
    # gesperrt ist. Greift nur, solange die Zeile darueber 1 ist.
    "watch_plus_daily_interval_allowed": 1,
}

# Persistente Admin-Konfiguration fuer explizite User-Memory-Edits. Diese
# Defaults sind absichtlich konservativ und werden fuer jedes fehlende oder
# ungueltige Firestore-Feld einzeln verwendet.
DEFAULT_MEMORY_EDIT_CONFIG = {
    "memory_edit_enabled": True,
    "memory_edit_model": "gpt-5.6-luna",
    "memory_free_chars": 12_000,
    "memory_plus_chars": 18_000,
    "memory_pro_chars": 24_000,
    "memory_free_ai_edits_daily": 5,
    "memory_plus_ai_edits_daily": 15,
    "memory_pro_ai_edits_daily": 30,
    "memory_ai_edits_per_minute": 5,
    "memory_edit_input_chars": 500,
    "memory_edit_output_tokens": 150,
    "memory_edit_timeout_seconds": 10,
    "memory_global_calls_daily": 5_000,
}

MEMORY_EDIT_CONFIG = DEFAULT_MEMORY_EDIT_CONFIG.copy()

LIMITS = DEFAULT_LIMITS.copy()
_RUNTIME_CONFIG_LOCK = threading.RLock()

MAX_WORDS = LIMITS["free_max_words"]
DEEP_SEARCH_MAX_WORDS = LIMITS["pro_deep_search_max_words"]
MAX_TOKENS = LIMITS["pro_max_tokens"]
DEEP_SEARCH_MAX_TOKENS = LIMITS["pro_deep_search_max_tokens"]
CONSENSUS_MAX_TOKENS = LIMITS["consensus_max_tokens"]
DIFFERENCES_MAX_TOKENS = LIMITS["differences_max_tokens"]
COVERAGE_MAX_TOKENS = LIMITS["coverage_max_tokens"]
REASONING_EFFORT_FOR_DEEP = "low"
# Zentrale Reasoning-Policy fuer alle festen Laufarten. Modellbezogene
# Overrides stehen weiter unten in MODEL_REQUEST_CONFIG. Aufrufer sollen
# keine String-Literale mehr verteilen: So kann das Admin-Dashboard denselben
# autoritativen Stand anzeigen, den die Requests tatsaechlich verwenden.
REASONING_EFFORT_FOR_JUDGE = "low"
REASONING_EFFORT_FOR_JUDGE_BY_PROVIDER = {"mistral": "none"}
REASONING_EFFORT_FOR_MEMORY_EDIT = "none"
REASONING_EFFORT_FOR_SEO_REVIEW = "medium"
REASONING_EFFORT_FOR_PUBLISHER_SCREEN = "low"
GEMINI_MAX_TOKENS = MAX_TOKENS
GEMINI_DEEP_MAX_TOKENS = DEEP_SEARCH_MAX_TOKENS
DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"
OPENAI_LUNA_MODEL = "gpt-5.6-luna"
OPENAI_SOL_MODEL = "gpt-5.6-sol"
DEFAULT_MISTRAL_MODEL = "mistral-small-latest"
MISTRAL_PRO_MODEL = "mistral-medium-3-5"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5"
ANTHROPIC_PRO_MODEL = "claude-opus-4-8"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_FLASH_MODEL = DEFAULT_GEMINI_MODEL
GEMINI_36_FLASH_MODEL = "gemini-3.6-flash"
GEMINI_35_FLASH_MODEL = "gemini-3.5-flash"
GEMINI_PRO_MODEL = "gemini-3.1-pro-preview"
DEEPSEEK_FLASH_MODEL = "deepseek-v4-flash"
DEEPSEEK_PRO_MODEL = "deepseek-v4-pro"
# Basis-Default wie bei jeder anderen Familie das guenstige Modell. Stand hier
# als einzige Familie auf dem Pro-Modell, was drei Stellen still verteuert hat:
# die Consensus-Engine "DeepSeek" (als free ausgewiesen, lief aber auf Pro und
# damit auf demselben Modell wie "DeepSeek-Pro"), den Standard-Judge der
# Differences-Engine und den Fallback in engines.py fuer Aufrufe ohne
# ausdrueckliche Modellwahl. Die Antwortmodelle waren nie betroffen: Presets
# fast/balanced und der Free-Default zeigen laengst auf Flash. Wo Pro gewollt
# ist, steht es ausdruecklich da -- Preset "thorough", Alias "DeepSeek-Pro",
# judge_models_pro und die Benchmark-Matrix.
DEFAULT_DEEPSEEK_MODEL = DEEPSEEK_FLASH_MODEL
GROK_NO_REASONING_MODEL = "grok-4.3-no-reasoning"
GROK_FAST_MODEL = GROK_NO_REASONING_MODEL
DEFAULT_GROK_MODEL = "grok-4.20-non-reasoning"
# Modelle hinter den "<Familie>-Pro"-Aliassen, soweit sie nicht schon oben
# stehen. Bewusst benannt, damit die Provider-Registry ohne Stringliterale
# auskommt.
OPENAI_PRO_MODEL = "gpt-5.5"
GROK_PRO_MODEL = "grok-4.3"
KIMI_BASE_MODEL = "kimi-k2.6"
KIMI_PRO_MODEL = "kimi-k3"
GLM_BASE_MODEL = "glm-5.3-flash"
GLM_PRO_MODEL = "glm-5.3"
# Meta Superintelligence Labs fuehrt seine Modelle bei OpenRouter unter
# "meta/" als Muse-Reihe. Basis ist das offene, destillierte Glimmer 30B
# (0,30/1,20 $ je Mio. Token), Pro das multimodale Spark 1.3 (1,25/4,25 $).
# Der billigere "muse-spark-1.3-contributor"-Tarif ist bewusst NICHT
# aufgenommen: dort duerfen Prompts und Antworten in Metas Produkte
# einfliessen, was der ZDR-Zusage in Terms/Privacy widerspricht.
MUSE_BASE_MODEL = "muse-glimmer-30b"
MUSE_PRO_MODEL = "muse-spark-1.3"

# ---------------------------------------------------------------------------
# Provider-Registry: die eine Quelle fuer alles, was je Modellfamilie gilt.
# Eine neue Familie ist ein Eintrag hier plus ihre Modelle in der Admin-DB --
# nicht ein Dutzend paralleler Dicts. Die abgeleiteten Strukturen darunter
# (Defaults, Labels, OpenRouter-Praefixe, Judge-Basis, Consensus-Aliasse)
# behalten ihre bisherigen Namen, weil Module und Tests sie direkt importieren.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProviderConfig:
    key: str
    label: str
    openrouter_prefix: str
    # Guenstiges Standardmodell der Familie (Alias "<Label>") und Premium-
    # Modell (Alias "<Label>-Pro"). Der Basis-Default ist IMMER das guenstige
    # Modell; wo Pro gewollt ist, steht es ausdruecklich da.
    base_model: str
    pro_model: str
    # Erlaubte Modelle. Wird von der Admin-Konfiguration IN PLACE mutiert --
    # die ALLOWED_*_MODELS-Aliasse zeigen auf genau dieses Set.
    models: set[str] = field(default_factory=set)
    # Modelle, die auch nach einem Admin-Override erlaubt bleiben muessen,
    # weil Presets/Aliasse/Deep-Think auf sie zeigen (base/pro implizit).
    required_models: frozenset[str] = frozenset()
    # Praesentation. Die App zeigt eine Familie unter bis zu drei Namen; die
    # DOM-IDs sind Vertrag mit CSS, JS und den E2E-Tests und leiten sich aus
    # dom_key/short_label ab.
    dom_key: str = ""        # ID-Stamm im DOM ("anthropic" -> "claude")
    title: str = ""          # Ueberschrift der Antwortbox ("openai" -> "ChatGPT")
    short_label: str = ""    # data-short-label, Chips, Checkbox-ID
    icon: str = ""           # Datei unter static/icons/chat_icons/
    icon_class: str = ""     # zusaetzliche CSS-Klasse am <img>
    citation_label: str = "" # ausgeschriebener Name in Zitaten
    # Produktregel auf Modellebene: None = alle Modelle koennen Anhaenge
    # verarbeiten (nativ oder ueber Text-Fallback), leeres Set = keines. Das
    # ist bewusst feiner als Provider-Support: GLM 5.3 Flash ist multimodal,
    # GLM 5.3 dagegen text-only.
    attachment_models: frozenset[str] | None = None

    def model_accepts_attachments(self, model_id: str | None) -> bool:
        return (
            self.attachment_models is None
            or str(model_id or "") in self.attachment_models
        )

    @property
    def response_id(self) -> str:
        return f"{self.dom_key}Response"

    @property
    def select_id(self) -> str:
        return f"{self.dom_key}ModelSelect"

    @property
    def text_id(self) -> str:
        return f"{self.dom_key}ModelText"

    @property
    def checkbox_id(self) -> str:
        return f"select{self.short_label}"

    @property
    def ask_endpoint(self) -> str:
        return f"/ask_{self.dom_key}"


def _provider(key, label, prefix, base, pro, models, required=(),
              dom_key="", title="", short_label="", icon="", icon_class="",
              citation_label="", attachment_models=None):
    dom_key = dom_key or key
    title = title or label
    short_label = short_label or title
    return ProviderConfig(
        key=key,
        label=label,
        openrouter_prefix=prefix,
        base_model=base,
        pro_model=pro,
        models=set(models),
        required_models=frozenset({base, pro, *required}),
        dom_key=dom_key,
        title=title,
        short_label=short_label,
        icon=icon or f"{key}.png",
        icon_class=icon_class,
        citation_label=citation_label or label,
        attachment_models=(
            None if attachment_models is None else frozenset(attachment_models)
        ),
    )


PROVIDERS: dict[str, ProviderConfig] = {
    provider.key: provider
    for provider in (
        _provider(
            "openai", "OpenAI", "openai/", DEFAULT_OPENAI_MODEL, OPENAI_PRO_MODEL,
            {
                "gpt-5-nano", "gpt-5-mini", "gpt-4.1", "gpt-4o", "gpt-3.5-turbo",
                "gpt-5", "gpt-5.1", "gpt-5.2", "gpt-5.4",
                OPENAI_PRO_MODEL, DEFAULT_OPENAI_MODEL,
                OPENAI_LUNA_MODEL, OPENAI_SOL_MODEL,
            },
            required=(OPENAI_LUNA_MODEL, OPENAI_SOL_MODEL),
            title="ChatGPT", short_label="OpenAI",
            icon="chatgpt.png", icon_class="chatgpt-logo",
        ),
        _provider(
            "mistral", "Mistral", "mistralai/", DEFAULT_MISTRAL_MODEL, MISTRAL_PRO_MODEL,
            {DEFAULT_MISTRAL_MODEL, MISTRAL_PRO_MODEL},
        ),
        _provider(
            "anthropic", "Anthropic", "anthropic/", DEFAULT_ANTHROPIC_MODEL, ANTHROPIC_PRO_MODEL,
            {DEFAULT_ANTHROPIC_MODEL, ANTHROPIC_PRO_MODEL},
            dom_key="claude", title="Claude", icon="claude.png",
            citation_label="Anthropic Claude",
        ),
        _provider(
            "gemini", "Gemini", "google/", DEFAULT_GEMINI_MODEL, GEMINI_PRO_MODEL,
            {
                GEMINI_FLASH_MODEL, GEMINI_36_FLASH_MODEL, "gemini-3.1-flash-lite",
                "gemini-3.1-flash-lite-preview", "gemini-2.5-flash",
                GEMINI_35_FLASH_MODEL,
                GEMINI_PRO_MODEL, "gemini-2.5-pro",
            },
            required=(GEMINI_36_FLASH_MODEL, GEMINI_35_FLASH_MODEL),
            icon="gemini-icon.png", citation_label="Google Gemini",
        ),
        _provider(
            "deepseek", "DeepSeek", "deepseek/", DEFAULT_DEEPSEEK_MODEL, DEEPSEEK_PRO_MODEL,
            {DEEPSEEK_FLASH_MODEL, DEEPSEEK_PRO_MODEL},
            attachment_models=(),
        ),
        _provider(
            "grok", "Grok", "x-ai/", DEFAULT_GROK_MODEL, GROK_PRO_MODEL,
            {
                GROK_NO_REASONING_MODEL, "grok-4.20", DEFAULT_GROK_MODEL,
                GROK_PRO_MODEL,
            },
            icon_class="grok-logo",
        ),
        _provider(
            "kimi", "Kimi", "moonshotai/", KIMI_BASE_MODEL, KIMI_PRO_MODEL,
            {KIMI_BASE_MODEL, KIMI_PRO_MODEL},
            icon="kimi.svg", icon_class="mono-logo", citation_label="Moonshot AI Kimi",
        ),
        _provider(
            "glm", "GLM", "z-ai/", GLM_BASE_MODEL, GLM_PRO_MODEL,
            {GLM_BASE_MODEL, GLM_PRO_MODEL},
            icon="zai.svg", icon_class="mono-logo", citation_label="Z.ai GLM",
            attachment_models=(GLM_BASE_MODEL,),
        ),
        _provider(
            "meta", "Meta", "meta/", MUSE_BASE_MODEL, MUSE_PRO_MODEL,
            {MUSE_BASE_MODEL, MUSE_PRO_MODEL},
            dom_key="muse", title="Muse", short_label="Muse",
            icon="meta.svg", icon_class="mono-logo", citation_label="Meta Muse",
        ),
    )
}

def _all_allowed_models() -> set[str]:
    """Alle erlaubten Modelle ueber alle Familien."""
    return set().union(*(provider.models for provider in PROVIDERS.values()))


DEFAULT_MODEL_BY_PROVIDER = {
    provider.key: provider.base_model for provider in PROVIDERS.values()
}

# Defaults fuer Nutzer ohne Pro: durchweg die guenstigen Basis-Modelle.
FREE_DEFAULT_MODEL_BY_PROVIDER = dict(DEFAULT_MODEL_BY_PROVIDER)

# Unveraenderliche Basis fuer den Free-Default je Provider. Der Admin kann den
# Free-Default pro Provider in Firestore (Feld "defaults") ueberschreiben; ohne
# Override gilt diese Basis (siehe apply_default_models).
_BASE_FREE_DEFAULTS = dict(FREE_DEFAULT_MODEL_BY_PROVIDER)

# Antwortmodelle fuer geplante Consensus-Watches. Pro Tier kann je Provider
# genau ein Modell aktiv sein; fehlende Legacy-Konfiguration behaelt das
# bisherige Verhalten mit drei guenstigen Modellen bei.
_BASE_WATCH_MODELS_BY_TIER = {
    "free": {
        "openai": DEFAULT_OPENAI_MODEL,
        "mistral": DEFAULT_MISTRAL_MODEL,
        "gemini": DEFAULT_GEMINI_MODEL,
    },
    "pro": {
        "openai": DEFAULT_OPENAI_MODEL,
        "mistral": DEFAULT_MISTRAL_MODEL,
        "gemini": DEFAULT_GEMINI_MODEL,
    },
}
WATCH_MODELS_BY_TIER = {
    tier: dict(models) for tier, models in _BASE_WATCH_MODELS_BY_TIER.items()
}

PROVIDER_LABEL_BY_ID = {
    provider.key: provider.label for provider in PROVIDERS.values()
}


def provider_label(provider: str | None) -> str:
    value = str(provider or "").lower()
    return PROVIDER_LABEL_BY_ID.get(value, value or "Provider")

# Separate Synthese-Engine fuer Consensus-Watches. Sie ist bewusst nicht an
# eines der Antwortmodelle gebunden: der konfigurierte Provider kann den
# Consensus aus den unabhaengigen Antworten erzeugen, ohne selbst eine
# Antwortposition beizusteuern. Legacy-Dokumente behalten mit OpenAI die
# bisherige bevorzugte Engine, solange der Provider verfuegbar ist.
_BASE_WATCH_CONSENSUS_MODELS_BY_TIER = {
    "free": "OpenAI",
    "pro": "OpenAI",
}
WATCH_CONSENSUS_MODELS_BY_TIER = dict(_BASE_WATCH_CONSENSUS_MODELS_BY_TIER)

# Vom Admin gepflegte Anzeige-Reihenfolge der Modelle je Provider in den normalen
# Pickern. Leere Liste => deterministischer Auto-Sort (model_picker_sort_key).
MODEL_ORDER_BY_PROVIDER: dict[str, list[str]] = {
    provider: [] for provider in DEFAULT_MODEL_BY_PROVIDER
}

# Consensus-Engine, auf die Deep Think die Synthese festkoppelt. Vom Admin
# ueber Firestore (Feld "deep_think_model") umstellbar; ungueltige Werte
# fallen auf die Basis zurueck (siehe apply_deep_think_model).
_BASE_DEEP_THINK_CONSENSUS_MODEL = GEMINI_35_FLASH_MODEL
DEEP_THINK_CONSENSUS_MODEL = _BASE_DEEP_THINK_CONSENSUS_MODEL

# Standard-Judges der Differences-/Resolve-Engine je Provider (Pro-Judges
# loesen weiterhin ueber die "<Familie>-Pro"-Aliasse auf, siehe
# consensus_engine._judge_engine). Vom Admin ueber Firestore (Feld
# "judge_models") umstellbar; ungueltige Werte fallen je Provider auf die
# Basis zurueck (siehe apply_judge_models). WICHTIG: das dict wird in-place
# mutiert, damit Modul-Aliasse (consensus_engine, resolve_engine) live bleiben.
_BASE_DIFFERENCES_JUDGE_BY_PROVIDER = dict(DEFAULT_MODEL_BY_PROVIDER)
DIFFERENCES_JUDGE_MODEL_BY_PROVIDER = dict(_BASE_DIFFERENCES_JUDGE_BY_PROVIDER)

# Anzeige-Reihenfolge der Familien-Aliasse im Consensus-Picker. Familien ohne
# Eintrag haengen in Registry-Reihenfolge hinten an.
_CONSENSUS_ALIAS_ORDER = [
    "grok", "openai", "anthropic", "mistral", "gemini", "deepseek", "kimi", "glm",
    "meta",
]


def _consensus_alias_providers() -> list[ProviderConfig]:
    ordered = [PROVIDERS[key] for key in _CONSENSUS_ALIAS_ORDER if key in PROVIDERS]
    listed = {provider.key for provider in ordered}
    ordered += [
        provider for provider in PROVIDERS.values() if provider.key not in listed
    ]
    return ordered


DEFAULT_CONSENSUS_MODELS = [
    GEMINI_35_FLASH_MODEL,
    GEMINI_36_FLASH_MODEL,
    OPENAI_LUNA_MODEL,
    *(provider.label for provider in _consensus_alias_providers()),
    *(f"{provider.label}-Pro" for provider in _consensus_alias_providers()),
]

ALLOWED_CONSENSUS_MODELS = list(DEFAULT_CONSENSUS_MODELS)
# Produktdefinition der Model-Set-Presets. Die sechs Antwortmodelle plus
# Consensus-Engine koennen getrennt davon aus Firestore ueberschrieben werden;
# "Custom" oeffnet weiterhin die volle Engine-Liste.
# High Quality (interne ID: thorough) bleibt als Produktregel Pro-only; Daily
# (interne ID: fast) und Balanced werden bei der
# Admin-Normalisierung strikt auf Free-faehige Modelle begrenzt.
# Die IDs sind Vertrag (Firestore-Overrides, gespeicherte Nutzerauswahl) und
# bleiben deshalb unveraendert, auch wenn sich das Label aendert.
CONSENSUS_PRESET_DEFINITIONS = [
    {
        "id": "fast",
        "label": "Daily",
        "hint": "Quick synthesis for everyday questions",
        "pro_only": False,
    },
    {
        "id": "balanced",
        "label": "Balanced",
        "hint": "Reliable default for most questions",
        "pro_only": False,
    },
    {
        "id": "thorough",
        "label": "High Quality",
        "hint": "The expensive models from end to end, the priciest run there is",
        "pro_only": True,
    },
]

_BASE_CONSENSUS_PRESET_MODELS = {
    "fast": {
        "answers": {
            "openai": OPENAI_LUNA_MODEL,
            "mistral": DEFAULT_MISTRAL_MODEL,
            "anthropic": DEFAULT_ANTHROPIC_MODEL,
            "gemini": DEFAULT_GEMINI_MODEL,
            "deepseek": DEEPSEEK_FLASH_MODEL,
            "grok": GROK_FAST_MODEL,
        },
        "consensus": "Gemini",
    },
    "balanced": {
        "answers": {
            "openai": OPENAI_LUNA_MODEL,
            "mistral": DEFAULT_MISTRAL_MODEL,
            "anthropic": DEFAULT_ANTHROPIC_MODEL,
            "gemini": DEFAULT_GEMINI_MODEL,
            "deepseek": DEEPSEEK_FLASH_MODEL,
            "grok": DEFAULT_GROK_MODEL,
        },
        "consensus": OPENAI_LUNA_MODEL,
    },
    "thorough": {
        "answers": {
            "openai": OPENAI_SOL_MODEL,
            "mistral": MISTRAL_PRO_MODEL,
            "anthropic": ANTHROPIC_PRO_MODEL,
            "gemini": GEMINI_PRO_MODEL,
            "deepseek": DEEPSEEK_PRO_MODEL,
            "grok": "grok-4.3",
        },
        "consensus": "Gemini-Pro",
    },
}
# Alte gespeicherte Preset-Werte, die bei einem Deploy automatisch auf die
# aktuelle Basis migriert werden. Das Modell bleibt ausserhalb des Presets
# weiterhin erlaubt, solange andere Flows/Aliase es noch verwenden.
DEPRECATED_CONSENSUS_PRESET_MODELS = {
    "thorough": {"openai": {"gpt-5.5"}},
}
CONSENSUS_PRESET_MODELS = {
    preset_id: {
        "answers": dict(models["answers"]),
        "consensus": models["consensus"],
    }
    for preset_id, models in _BASE_CONSENSUS_PRESET_MODELS.items()
}
DEFAULT_CONSENSUS_PRESET = "balanced"

VALID_LEADERBOARD_MODELS = {
    label
    for provider in PROVIDERS.values()
    for label in (provider.label, f"{provider.label}-Pro")
}

CONSENSUS_ENGINE_ALIASES = {
    alias: (provider.key, model)
    for provider in PROVIDERS.values()
    for alias, model in (
        (provider.label, provider.base_model),
        (f"{provider.label}-Pro", provider.pro_model),
    )
}

# Pro-Judges der Differences-/Resolve-Engine je Provider. Basis sind die
# API-Modelle der "<Familie>-Pro"-Aliasse; vom Admin ueber Firestore (Feld
# "judge_models_pro") umstellbar. Wie DIFFERENCES_JUDGE_MODEL_BY_PROVIDER
# in-place mutiert (Modul-Aliasse bleiben live).
_BASE_PRO_JUDGE_BY_PROVIDER = {
    provider.key: provider.pro_model for provider in PROVIDERS.values()
}
PRO_JUDGE_MODEL_BY_PROVIDER = dict(_BASE_PRO_JUDGE_BY_PROVIDER)

# Modell, das die Chat-Memory laengerer Unterhaltungen fortschreibt
# (app/services/chat_context.py). Die FAMILIE bleibt die der Consensus-Engine
# des Turns; der eine OpenRouter-Key gilt fuer alle Familien. Das Modell
# INNERHALB der Familie ist vom Admin ueber Firestore (Feld
# "chat_memory_models") umstellbar. Basis sind bewusst die guenstigen
# Standardmodelle: die Aufgabe ist strukturierte Extraktion nach JSON-Schema,
# kein Denken. Wie die Judge-Dicts in-place mutiert.
_BASE_CHAT_MEMORY_MODEL_BY_PROVIDER = dict(_BASE_DIFFERENCES_JUDGE_BY_PROVIDER)
CHAT_MEMORY_MODEL_BY_PROVIDER = dict(_BASE_CHAT_MEMORY_MODEL_BY_PROVIDER)

# Familien-Prioritaet der Judge-Wahl: primaerer und Fallback-Judge nehmen die
# erste andere Familie; die gemeinsame OpenRouter-Verfügbarkeit wird davor geprüft.
# Gemini/OpenAI remain the preferred independent judges. Mistral is a working
# emergency fallback, but intentionally comes after every other family.
_JUDGE_FAMILY_PRIORITY_BASE = [
    "gemini", "openai", "deepseek", "grok", "anthropic", "mistral"
]
JUDGE_FAMILY_PRIORITY = [
    *_JUDGE_FAMILY_PRIORITY_BASE,
    *(provider for provider in PROVIDERS if provider not in _JUDGE_FAMILY_PRIORITY_BASE),
]

# Optionales Admin-Mapping Engine-Familie -> bevorzugte Judge-Familie
# (Firestore-Feld "judge_families"). Fehlt ein Eintrag, greift
# JUDGE_FAMILY_PRIORITY (Auto).
JUDGE_FAMILY_BY_ENGINE: dict[str, str] = {}
LEADERBOARD_MODEL_ALIASES = {
    "Claude": "Anthropic",
}

# Die erlaubten Modelle stehen in der Provider-Registry. Diese Namen bleiben
# als Aliasse auf DASSELBE Set-Objekt bestehen: Admin-Loads mutieren in place,
# und Module/Tests importieren sie direkt.
ALLOWED_OPENAI_MODELS = PROVIDERS["openai"].models
ALLOWED_MISTRAL_MODELS = PROVIDERS["mistral"].models
ALLOWED_ANTHROPIC_MODELS = PROVIDERS["anthropic"].models
ALLOWED_GEMINI_MODELS = PROVIDERS["gemini"].models
ALLOWED_DEEPSEEK_MODELS = PROVIDERS["deepseek"].models
ALLOWED_GROK_MODELS = PROVIDERS["grok"].models
ALLOWED_KIMI_MODELS = PROVIDERS["kimi"].models
ALLOWED_GLM_MODELS = PROVIDERS["glm"].models
ALLOWED_META_MODELS = PROVIDERS["meta"].models

MISTRAL_REASONING_MODELS = {
    DEFAULT_MISTRAL_MODEL,
    MISTRAL_PRO_MODEL,
    "mistralai/mistral-small-2603",
    "mistralai/mistral-medium-3-5",
}
DEPRECATED_MISTRAL_MODELS = {
    "devstral-small-2507", "devstral-small-latest", "devstral-medium-2507",
    "mistral-large-2411", "pixtral-large-2411", "pixtral-large-latest",
}

# Alte xAI-Aliasse, die seit Mai 2026 ohnehin auf Grok 4.3 umgeleitet werden.
# Beim Laden bestehender Admin-Daten werden sie auf unsere expliziten
# Reasoning-Varianten migriert, damit keine doppelten/irrefuehrenden Zeilen
# im Picker und in der Admin-DB verbleiben.
GROK_MODEL_MIGRATIONS = {
    "grok-4-1-fast-non-reasoning": GROK_NO_REASONING_MODEL,
    "grok-4-1-fast-non-reasoning-latest": GROK_NO_REASONING_MODEL,
    "grok-4-fast-non-reasoning": GROK_NO_REASONING_MODEL,
    "grok-4-fast-non-reasoning-latest": GROK_NO_REASONING_MODEL,
    "grok-3": GROK_NO_REASONING_MODEL,
    "grok-3-latest": GROK_NO_REASONING_MODEL,
    "grok-4-1-fast-reasoning": "grok-4.3",
    "grok-4-1-fast-reasoning-latest": "grok-4.3",
    "grok-4-fast-reasoning": "grok-4.3",
    "grok-4-fast-reasoning-latest": "grok-4.3",
    "grok-4-0709": "grok-4.3",
    "grok-4": "grok-4.3",
    "grok-4-latest": "grok-4.3",
}
DEPRECATED_GROK_MODELS = set(GROK_MODEL_MIGRATIONS)
REMOVED_MODEL_IDS = {
    # Nicht mehr im OpenRouter-Katalog vorhandene Legacy-IDs. Die Tombstones
    # verhindern, dass alte Firestore-Konfigurationen sie erneut aktivieren.
    "gpt-5-chat-latest",
    "gpt-5.3",
    "gpt-5.3-chat-latest",
    "mistral-large-latest",
    "mistral-medium-latest",
    "ministral-3b-latest",
    "ministral-8b-latest",
    "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-haiku-20241022",
    "claude-sonnet-4-5",
    "claude-opus-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "gemini-2.0-flash",
    "gpt-5.5-frontier-low",
    "claude-opus-4-8-frontier-low",
    "gemini-3.1-pro-preview-frontier-low",
    # Diese beiden IDs waren bereits vor der Early-Bereinigung als nicht
    # aufrufbar bekannt. Neue Gemini-Modelle werden direkt ueber die Admin-
    # Konfiguration zugelassen; bekannte tote Preview-IDs bleiben Tombstones.
    "gemini-3.1-flash-preview",
    "gemini-3-pro-preview",
    "grok-4.3-frontier-low",
    "grok-4.3-low-reasoning",
}


def canonical_model_id(model_id: str | None, provider: str | None = None) -> str:
    """Migriert eine gespeicherte Modell-ID auf ihre aktuelle Form.

    Ohne Provider werden die Migrationen aller Familien geprueft -- Aufrufer
    ohne Familienkontext (Consensus-Werte, Preset-Engines) kommen sonst mit
    einer toten ID durch."""
    value = str(model_id or "").strip()
    if provider is None:
        for migrations in PROVIDER_MODEL_MIGRATIONS.values():
            if value in migrations:
                return migrations[value]
        return value
    return PROVIDER_MODEL_MIGRATIONS.get(
        str(provider).lower(), {}
    ).get(value, value)


def canonical_model_ids(models, provider: str) -> list[str]:
    return list(dict.fromkeys(
        canonical_model_id(model, provider)
        for model in (models or [])
        if canonical_model_id(model, provider)
    ))

DEPRECATED_DEEPSEEK_MODELS = {"deepseek-chat", "deepseek-reasoner"}

# Familienspezifische Hygiene. Jede Familie ohne Eintrag braucht keine: der
# generische Lade-/Initialisierungspfad fragt hier nach, statt pro Familie
# einen eigenen Block zu fuehren.
PROVIDER_MODEL_MIGRATIONS: dict[str, dict[str, str]] = {
    "grok": GROK_MODEL_MIGRATIONS,
}
PROVIDER_DEPRECATED_MODELS: dict[str, set[str]] = {
    "mistral": DEPRECATED_MISTRAL_MODELS,
    "deepseek": DEPRECATED_DEEPSEEK_MODELS,
    "grok": DEPRECATED_GROK_MODELS,
}

def ensure_default_models_allowed():
    """Initialisiert ausschliesslich die Code-Fallback-Konfiguration.

    Firestore-Providerlisten sind nach einem erfolgreichen Load autoritativ;
    diese Funktion darf deshalb dort nicht erneut aufgerufen werden.
    """
    for provider in PROVIDERS.values():
        provider.models.difference_update(
            PROVIDER_DEPRECATED_MODELS.get(provider.key, set())
        )
        provider.models.update(provider.required_models)
        provider.models.difference_update(REMOVED_MODEL_IDS)

ensure_default_models_allowed()

PREMIUM_MODELS = {
    "gpt-5", "gpt-5.1", "gpt-5.2", "gpt-5.4",
    "gpt-5.5",
    OPENAI_SOL_MODEL,
    ANTHROPIC_PRO_MODEL,
    MISTRAL_PRO_MODEL,
    GEMINI_PRO_MODEL, GEMINI_35_FLASH_MODEL, "gemini-2.5-pro",
    DEEPSEEK_PRO_MODEL,
    "grok-4.20", GROK_PRO_MODEL,
}
# Die Pro-Modelle der Registry sind per Definition Premium.
PREMIUM_MODELS.update(provider.pro_model for provider in PROVIDERS.values())
PREMIUM_MODELS.difference_update(REMOVED_MODEL_IDS)
PREMIUM_MODELS.difference_update(DEPRECATED_MISTRAL_MODELS)
PREMIUM_MODELS.difference_update(DEPRECATED_DEEPSEEK_MODELS)
PREMIUM_MODELS.difference_update(DEPRECATED_GROK_MODELS)

ALL_ALLOWED_MODELS = _all_allowed_models()

MODEL_LABEL_OVERRIDES = {
    OPENAI_LUNA_MODEL: "GPT-5.6 Luna",
    OPENAI_SOL_MODEL: "GPT-5.6 Sol",
    "gpt-5.5": "GPT-5.5",
    DEFAULT_OPENAI_MODEL: "GPT-5.4 mini",
    DEFAULT_ANTHROPIC_MODEL: "Claude Haiku 4.5",
    DEFAULT_GEMINI_MODEL: "Gemini 3.5 Flash-Lite",
    # Grok-Labels bewusst nach einem Schema: "<Version> · <Reasoning-Zustand>".
    # Die frueheren Mischformen ("High reasoning" vs. "Reasoning") legten nahe,
    # es handle sich um verschiedene Modelle statt um Reasoning-Varianten
    # desselben Modells.
    DEFAULT_GROK_MODEL: "Grok 4.20 · No reasoning",
    GROK_NO_REASONING_MODEL: "Grok 4.3 · No reasoning",
    "grok-4.5": "Grok 4.5",
    "mistral-small-latest": "Mistral Small 4",
    MISTRAL_PRO_MODEL: "Mistral Medium 3.5",
    ANTHROPIC_PRO_MODEL: "Claude Opus 4.8",
    GEMINI_36_FLASH_MODEL: "Gemini 3.6 Flash",
    GEMINI_35_FLASH_MODEL: "Gemini 3.5 Flash",
    GEMINI_PRO_MODEL: "Gemini 3.1",
    "grok-4.20": "Grok 4.20 · Reasoning",
    "grok-4.3": "Grok 4.3 · Reasoning",
    DEEPSEEK_FLASH_MODEL: "DeepSeek V4 Flash",
    DEEPSEEK_PRO_MODEL: "DeepSeek V4 Pro",
    KIMI_BASE_MODEL: "Kimi K2.6",
    KIMI_PRO_MODEL: "Kimi K3",
    GLM_BASE_MODEL: "GLM 5.3 Flash",
    GLM_PRO_MODEL: "GLM 5.3",
    MUSE_BASE_MODEL: "Muse Glimmer 30B",
    MUSE_PRO_MODEL: "Muse Spark 1.3",
}

MODEL_CONFIGS: dict[str, ModelConfig] = {}

# Request-Zusaetze je Modell-ID. Reine Daten: eine Reasoning-Variante ist ein
# Eintrag hier, kein Sonderfall in rebuild_model_configs.
MODEL_REQUEST_CONFIG: dict[str, dict[str, Any]] = {
    GROK_NO_REASONING_MODEL: {"reasoning": {"effort": "none"}},
    GROK_PRO_MODEL: {"reasoning": {"effort": "high"}},
    KIMI_BASE_MODEL: {"reasoning": {"enabled": False}},
    KIMI_PRO_MODEL: {"reasoning": {"enabled": False}},
    GLM_BASE_MODEL: {"reasoning": {"effort": "low"}},
    GLM_PRO_MODEL: {"reasoning": {"effort": "low"}},
    # Muse denkt zwingend (OpenRouter meldet reasoning.mandatory=true); die
    # Modelle kennen kein Abschalten, nur einen Aufwand. "low" haelt die als
    # Output abgerechneten Reasoning-Tokens klein.
    MUSE_BASE_MODEL: {"reasoning": {"effort": "low"}},
    MUSE_PRO_MODEL: {"reasoning": {"effort": "low"}},
}

# Die Produktkonfiguration verwendet stabile, providerneutrale interne IDs.
# Provider-Requests laufen dagegen über OpenRouter und benötigen dessen
# kanonische Publisher-Präfixe. Diese Auflösung bleibt absichtlich an einer
# Stelle, damit neue erlaubte IDs nicht in einzelnen Flows Sonderbehandlung
# brauchen.
OPENROUTER_MODEL_PREFIXES = {
    provider.key: provider.openrouter_prefix for provider in PROVIDERS.values()
}
OPENROUTER_MODEL_ALIASES = {
    ("mistral", "mistral-small-latest"): "mistral-small-2603",
    ("anthropic", "claude-haiku-4-5"): "claude-haiku-4.5",
    ("anthropic", "claude-opus-4-8"): "claude-opus-4.8",
    ("grok", GROK_NO_REASONING_MODEL): "grok-4.3",
    ("grok", DEFAULT_GROK_MODEL): "grok-4.20",
}


def openrouter_model_id(model_id: str | None, provider: str | None) -> str:
    """Resolve an internal provider model ID to its OpenRouter model ID."""
    value = str(model_id or "").strip()
    prefix = OPENROUTER_MODEL_PREFIXES.get(str(provider or "").lower())
    if not value or not prefix or value.startswith(prefix):
        return value
    value = OPENROUTER_MODEL_ALIASES.get((str(provider or "").lower(), value), value)
    return f"{prefix}{value}"


def _fallback_label(model_id: str) -> str:
    override = MODEL_LABEL_OVERRIDES.get(model_id)
    if override:
        return override
    raw = str(model_id or "").strip()
    if not raw or " " in raw:
        return raw
    parts = [part for part in raw.lower().split("-") if part]
    if not parts:
        return raw
    family = parts.pop(0)
    family_labels = {
        "gpt": "GPT",
        "claude": "Claude",
        "mistral": "Mistral",
        "gemini": "Gemini",
        "deepseek": "DeepSeek",
        "grok": "Grok",
        "kimi": "Kimi",
        "glm": "GLM",
        "muse": "Muse",
    }
    if family not in family_labels:
        return raw
    ignored = {"latest", "preview"}
    words = []
    number_parts = []
    for part in parts:
        if part in ignored:
            continue
        if part.isdigit():
            number_parts.append(part)
            continue
        if number_parts:
            words.append(".".join(number_parts))
            number_parts = []
        words.append({
            "non": "No",
            "no": "No",
            "reasoning": "reasoning",
            "mini": "mini",
            "flash": "Flash",
            "lite": "Lite",
            "small": "Small",
            "medium": "Medium",
            "large": "Large",
            "haiku": "Haiku",
            "sonnet": "Sonnet",
            "opus": "Opus",
            "chat": "Chat",
            "pro": "Pro",
        }.get(part, part.capitalize()))
    if number_parts:
        words.append(".".join(number_parts))
    if family == "gpt" and words:
        label = "GPT-" + words[0]
        if len(words) > 1:
            label += " " + " ".join(words[1:])
    else:
        label = " ".join([family_labels[family], *words]).strip()
    label = label.replace("No reasoning", "· No reasoning")
    return label


def _provider_allowed_sets() -> dict[str, set]:
    return {provider.key: provider.models for provider in PROVIDERS.values()}


def rebuild_model_configs():
    MODEL_CONFIGS.clear()
    for provider, models in _provider_allowed_sets().items():
        for model_id in models:
            MODEL_CONFIGS[model_id] = ModelConfig(
                internal_id=model_id,
                provider=provider,
                api_model=openrouter_model_id(model_id, provider),
                label=_fallback_label(model_id),
                is_free=model_id not in PREMIUM_MODELS,
                is_pro=model_id in PREMIUM_MODELS,
                accepts_attachments=PROVIDERS[provider].model_accepts_attachments(model_id),
                request_config=dict(MODEL_REQUEST_CONFIG.get(model_id, {})),
            )


def virtual_model_ids() -> dict[str, str]:
    """Interne IDs -> API-Modell, wo beide voneinander abweichen.

    Reasoning-Varianten wie grok-4.3-no-reasoning sind reine Produkt-IDs; beim
    Provider existieren sie nicht. Jede Stelle, die eine konfigurierte Modell-ID
    in einen Request schreibt, muss sie vorher hierueber aufloesen.
    """
    return {
        model_id: config.api_model
        for model_id, config in MODEL_CONFIGS.items()
        if config.api_model and config.api_model != model_id
    }


def get_model_config(model_id: str | None, provider: str | None = None) -> ModelConfig | None:
    if not model_id:
        return None
    model_id = canonical_model_id(model_id, provider)
    config = MODEL_CONFIGS.get(model_id)
    if config:
        return config
    return ModelConfig(
        internal_id=model_id,
        provider=provider or "",
        api_model=openrouter_model_id(model_id, provider),
        label=_fallback_label(model_id),
        is_free=model_id not in PREMIUM_MODELS,
        is_pro=model_id in PREMIUM_MODELS,
        accepts_attachments=(
            PROVIDERS[provider].model_accepts_attachments(model_id)
            if provider in PROVIDERS else True
        ),
    )


def effective_model_reasoning(
    provider: str,
    model_id: str | None,
    *,
    deep_think: bool = False,
) -> tuple[dict[str, Any] | None, str]:
    """Resolve the reasoning payload and its source for an answer-model call.

    The precedence intentionally matches ``build_provider_payload``: an
    explicit model policy wins, Mistral reasoning models default to ``high``,
    and Deep Think only fills a still-unset policy with its global effort.
    Returning the source makes the exact runtime decision inspectable without
    duplicating the rules in the Admin UI.
    """
    provider_key = str(provider or "").lower()
    internal_model = canonical_model_id(model_id, provider_key)
    model_config = get_model_config(internal_model, provider_key)
    request_config = dict(model_config.request_config or {}) if model_config else {}
    explicit = request_config.get("reasoning")
    if isinstance(explicit, dict) and explicit:
        return dict(explicit), "MODEL_REQUEST_CONFIG"
    if provider_key == "mistral" and internal_model in MISTRAL_REASONING_MODELS:
        return {"effort": "high"}, "MISTRAL_REASONING_MODELS"
    if deep_think:
        return {"effort": REASONING_EFFORT_FOR_DEEP}, "REASONING_EFFORT_FOR_DEEP"
    return None, "provider default"


def judge_reasoning_effort(provider: str) -> str:
    """Provider-compatible effort requested by Differences/Coverage judges."""
    return REASONING_EFFORT_FOR_JUDGE_BY_PROVIDER.get(
        str(provider or "").lower(),
        REASONING_EFFORT_FOR_JUDGE,
    )


def resolve_api_model(model_id: str | None, default_model: str, provider: str) -> tuple[str, ModelConfig]:
    selected_model = model_id or default_model
    config = get_model_config(selected_model, provider) or get_model_config(default_model, provider)
    return config.api_model, config


def get_model_label(model_id: str) -> str:
    config = get_model_config(model_id)
    return config.label if config else _fallback_label(model_id)


def get_model_badge(model_id: str) -> str:
    config = get_model_config(model_id)
    return "Pro" if (config and config.is_pro) or model_id in PREMIUM_MODELS else ""


def get_consensus_model_config(model_id: str | None) -> ModelConfig | None:
    if not model_id:
        return None
    alias = CONSENSUS_ENGINE_ALIASES.get(model_id)
    if alias:
        provider, api_model = alias
        return ModelConfig(
            internal_id=model_id,
            provider=provider,
            api_model=openrouter_model_id(api_model, provider),
            label=_fallback_label(api_model),
            is_free=not str(model_id).endswith("-Pro"),
            is_pro=str(model_id).endswith("-Pro"),
        )
    return get_model_config(model_id)


def is_premium_consensus_model(model_id: str | None) -> bool:
    config = get_consensus_model_config(model_id)
    return bool(config and config.is_pro)


def get_consensus_model_label(model_id: str) -> str:
    config = get_consensus_model_config(model_id)
    return config.label if config else _fallback_label(model_id)


def get_consensus_model_badge(model_id: str) -> str:
    if str(model_id or "").endswith("-Pro"):
        return "Pro"
    config = get_consensus_model_config(model_id)
    return "Pro" if config and config.is_pro else ""


def get_consensus_presets() -> list[dict]:
    """Liefert die produktseitigen Preset-Metadaten plus das aktuell aus der
    Admin-Konfiguration aufgeloeste Antwort-/Consensus-Model-Set."""
    return [
        {
            **preset,
            "models": dict(CONSENSUS_PRESET_MODELS[preset["id"]]["answers"]),
            "consensus_model": CONSENSUS_PRESET_MODELS[preset["id"]]["consensus"],
        }
        for preset in CONSENSUS_PRESET_DEFINITIONS
    ]


def get_consensus_preset_models() -> dict[str, dict]:
    return {
        preset_id: {
            "answers": dict(models["answers"]),
            "consensus": models["consensus"],
        }
        for preset_id, models in CONSENSUS_PRESET_MODELS.items()
    }


def _preset_answer_mapping(value) -> dict:
    """Liest das neue ``answers``-Mapping und das alte Provider-Key-Schema.

    Altbestand hatte je Registry-Familie einen Top-Level-Key. Seit mehr als
    sechs Familien existieren, speichert ein Preset nur noch die hoechstens
    sechs tatsaechlich gewaehlten Familien unter ``answers``.
    """
    supplied = value if isinstance(value, dict) else {}
    nested = supplied.get("answers")
    if isinstance(nested, dict):
        return nested
    return {
        provider: supplied.get(provider)
        for provider in PROVIDERS
        if supplied.get(provider)
    }


def _normalize_preset_answers(
    preset_id: str,
    supplied,
    *,
    allowed_sets: dict[str, set] | None = None,
    premium_models: set[str] | None = None,
    defaults: dict[str, str] | None = None,
) -> dict[str, str]:
    allowed_sets = allowed_sets or _provider_allowed_sets()
    premium_models = PREMIUM_MODELS if premium_models is None else premium_models
    defaults = FREE_DEFAULT_MODEL_BY_PROVIDER if defaults is None else defaults
    definitions = {preset["id"]: preset for preset in CONSENSUS_PRESET_DEFINITIONS}
    pro_only = bool(definitions[preset_id]["pro_only"])
    base_answers = _BASE_CONSENSUS_PRESET_MODELS[preset_id]["answers"]
    supplied_answers = _preset_answer_mapping(supplied)
    provider_order = list(dict.fromkeys([
        *supplied_answers,
        *base_answers,
        *PROVIDERS,
    ]))
    target_count = min(MAX_RUN_FAMILIES, len(PROVIDERS))
    clean: dict[str, str] = {}
    for provider in provider_order:
        if provider not in PROVIDERS or len(clean) >= target_count:
            continue
        chosen = canonical_model_id(supplied_answers.get(provider), provider)
        deprecated = (
            DEPRECATED_CONSENSUS_PRESET_MODELS
            .get(preset_id, {})
            .get(provider, set())
        )
        if chosen in deprecated:
            chosen = ""
        allowed = set(allowed_sets.get(provider, set()))
        if chosen not in allowed or (not pro_only and chosen in premium_models):
            candidates = (
                canonical_model_id(base_answers.get(provider), provider),
                defaults.get(provider),
                DEFAULT_MODEL_BY_PROVIDER.get(provider),
                *get_ordered_models(provider),
                *sorted(allowed),
            )
            chosen = next((
                candidate for candidate in candidates
                if candidate in allowed
                and (pro_only or candidate not in premium_models)
            ), "")
        if chosen:
            clean[provider] = chosen
    return clean


def apply_consensus_preset_models(config: dict | None) -> None:
    """Validiert und aktiviert die Firestore-Model-Sets. Fast/Balanced duerfen
    keine Pro-Modelle enthalten; High Quality (ID: thorough) ist durch
    die Produktdefinition Pro-gated und darf die Premium-Modelle nutzen."""
    incoming = config if isinstance(config, dict) else {}
    definitions = {preset["id"]: preset for preset in CONSENSUS_PRESET_DEFINITIONS}

    for preset_id, base in _BASE_CONSENSUS_PRESET_MODELS.items():
        supplied = incoming.get(preset_id)
        supplied = supplied if isinstance(supplied, dict) else {}
        pro_only = bool(definitions[preset_id]["pro_only"])
        clean_answers = _normalize_preset_answers(preset_id, supplied)

        consensus = canonical_model_id(supplied.get("consensus"))
        consensus_config = get_consensus_model_config(consensus)
        if not consensus_config or not consensus_config.provider:
            consensus = canonical_model_id(base["consensus"])
        if not pro_only and is_premium_consensus_model(consensus):
            consensus = "Gemini"
        if not get_consensus_model_config(consensus):
            consensus = "Gemini"
        CONSENSUS_PRESET_MODELS[preset_id].clear()
        CONSENSUS_PRESET_MODELS[preset_id].update({
            "answers": clean_answers,
            "consensus": consensus,
        })


def normalize_consensus_models(models) -> list[str]:
    incoming = [canonical_model_id(model) for model in (models or []) if str(model or "").strip()]
    if not incoming:
        incoming = list(DEFAULT_CONSENSUS_MODELS)
    allowed = []
    for model in incoming:
        if model in allowed:
            continue
        config = get_consensus_model_config(model)
        if config and config.provider:
            allowed.append(model)
    # Deep Think koppelt die Synthese fest an das konfigurierte Deep-Think-
    # Modell (Basis: Gemini 3.5 Flash). Deshalb muss das Modell auch bei einer
    # Admin-/Firestore-Liste ohne diesen Eintrag als Consensus-Option
    # verfuegbar bleiben.
    if DEEP_THINK_CONSENSUS_MODEL not in allowed:
        allowed.append(DEEP_THINK_CONSENSUS_MODEL)
    # Auch die Admin-konfigurierten Preset-Engines muessen im nativen Select
    # vorhanden sein; die sichtbare Preset-Ebene setzt genau diese Werte.
    for preset in CONSENSUS_PRESET_MODELS.values():
        model = preset["consensus"]
        if model not in allowed:
            allowed.append(model)
    return allowed


def get_deep_think_consensus_model() -> str:
    return DEEP_THINK_CONSENSUS_MODEL


def is_valid_deep_think_model(model_id) -> bool:
    """Gueltig ist jeder Consensus-Wert (Alias oder direkte Modell-ID), der
    sich auf einen Provider aufloesen laesst."""
    chosen = str(model_id or "").strip()
    if not chosen:
        return False
    config = get_consensus_model_config(chosen)
    return bool(config and config.provider)


def apply_deep_think_model(model_id) -> None:
    """Setzt die Deep-Think-Consensus-Engine. Ungueltige/leere Werte fallen
    auf die Basis (Gemini 3.5 Flash) zurueck."""
    global DEEP_THINK_CONSENSUS_MODEL
    chosen = str(model_id or "").strip()
    if chosen and is_valid_deep_think_model(chosen):
        DEEP_THINK_CONSENSUS_MODEL = chosen
    else:
        DEEP_THINK_CONSENSUS_MODEL = _BASE_DEEP_THINK_CONSENSUS_MODEL


def is_valid_judge_model(provider: str, model_id) -> bool:
    """Gueltiger Standard-Judge: erlaubtes Modell des Providers."""
    chosen = str(model_id or "").strip()
    if not chosen:
        return False
    return chosen in _provider_allowed_sets().get(provider, set())


def apply_judge_models(overrides: dict | None) -> None:
    """Setzt den Standard-Differences-Judge je Provider. Ungueltige/fehlende
    Werte fallen je Provider auf die Basis zurueck. Mutiert das dict in-place
    (Modul-Aliasse in consensus_engine/resolve_engine bleiben live)."""
    data = overrides if isinstance(overrides, dict) else {}
    for provider, base in _BASE_DIFFERENCES_JUDGE_BY_PROVIDER.items():
        chosen = str(data.get(provider) or "").strip()
        if not (chosen and is_valid_judge_model(provider, chosen)):
            allowed = _provider_allowed_sets().get(provider, set())
            chosen = next(
                (
                    candidate for candidate in (
                        FREE_DEFAULT_MODEL_BY_PROVIDER.get(provider),
                        base,
                        *get_ordered_models(provider),
                    )
                    if candidate in allowed
                ),
                "",
            )
        DIFFERENCES_JUDGE_MODEL_BY_PROVIDER[provider] = chosen


def get_judge_models() -> dict:
    return dict(DIFFERENCES_JUDGE_MODEL_BY_PROVIDER)


def apply_pro_judge_models(overrides: dict | None) -> None:
    """Setzt den Pro-Differences-Judge je Provider. Ungueltige/fehlende Werte
    fallen je Provider auf die Basis (API-Modell des "<Familie>-Pro"-Alias)
    zurueck. Mutiert das dict in-place."""
    data = overrides if isinstance(overrides, dict) else {}
    for provider, base in _BASE_PRO_JUDGE_BY_PROVIDER.items():
        chosen = str(data.get(provider) or "").strip()
        if not (chosen and is_valid_judge_model(provider, chosen)):
            allowed = _provider_allowed_sets().get(provider, set())
            ordered = get_ordered_models(provider)
            chosen = next(
                (
                    candidate for candidate in (
                        base,
                        *(model for model in ordered if model in PREMIUM_MODELS),
                        FREE_DEFAULT_MODEL_BY_PROVIDER.get(provider),
                        *ordered,
                    )
                    if candidate in allowed
                ),
                "",
            )
        PRO_JUDGE_MODEL_BY_PROVIDER[provider] = chosen


def get_pro_judge_models() -> dict:
    return dict(PRO_JUDGE_MODEL_BY_PROVIDER)


def apply_chat_memory_models(overrides: dict | None) -> None:
    """Setzt das Chat-Memory-Modell je Provider. Ungueltige/fehlende Werte
    fallen je Provider auf die Basis zurueck. Mutiert das dict in-place."""
    data = overrides if isinstance(overrides, dict) else {}
    for provider, base in _BASE_CHAT_MEMORY_MODEL_BY_PROVIDER.items():
        chosen = str(data.get(provider) or "").strip()
        if not (chosen and is_valid_judge_model(provider, chosen)):
            allowed = _provider_allowed_sets().get(provider, set())
            chosen = next(
                (
                    candidate for candidate in (
                        FREE_DEFAULT_MODEL_BY_PROVIDER.get(provider),
                        base,
                        *get_ordered_models(provider),
                    )
                    if candidate in allowed
                ),
                "",
            )
        CHAT_MEMORY_MODEL_BY_PROVIDER[provider] = chosen


def get_chat_memory_models() -> dict:
    return dict(CHAT_MEMORY_MODEL_BY_PROVIDER)


def get_chat_memory_model(provider: str) -> str:
    """Modell, mit dem die Chat-Memory dieser Provider-Familie fortgeschrieben
    wird. Leer heisst: keine gueltige Wahl — der Aufrufer bleibt dann bei der
    Consensus-Engine des Turns."""
    return str(CHAT_MEMORY_MODEL_BY_PROVIDER.get(str(provider or "").strip().lower()) or "")


def apply_judge_families(overrides: dict | None) -> None:
    """Setzt das Mapping Engine-Familie -> bevorzugte Judge-Familie. Gueltig
    sind nur bekannte Provider, die sich von der Engine-Familie unterscheiden
    (Anti-Self-Judging); alles andere faellt auf Auto (Prioritaetsliste)
    zurueck. Mutiert das dict in-place."""
    data = overrides if isinstance(overrides, dict) else {}
    providers = set(_BASE_DIFFERENCES_JUDGE_BY_PROVIDER)
    JUDGE_FAMILY_BY_ENGINE.clear()
    for engine_provider in providers:
        chosen = str(data.get(engine_provider) or "").strip()
        if chosen in providers and chosen != engine_provider:
            JUDGE_FAMILY_BY_ENGINE[engine_provider] = chosen


def get_judge_families() -> dict:
    return dict(JUDGE_FAMILY_BY_ENGINE)


# Wie viele Antwortmodelle ein Lauf hoechstens umfassen darf. Mehr Familien
# als das duerfen konfiguriert sein -- ein Lauf bleibt trotzdem ein
# Sechs-Modell-Vergleich (Prompt-Laenge, Kosten, Lesbarkeit).
MAX_RUN_FAMILIES = 6


def get_model_families() -> list[dict]:
    """Familien-Metadaten fuer /app: Template und Frontend teilen sich genau
    diese Liste, damit Antwortbox, Picker und Sendepfad nie auseinanderlaufen."""
    return [
        {
            "provider": provider.key,
            "label": provider.label,
            "title": provider.title,
            "shortLabel": provider.short_label,
            "citationLabel": provider.citation_label,
            "icon": f"/static/icons/chat_icons/{provider.icon}",
            "iconClass": provider.icon_class,
            "domKey": provider.dom_key,
            "checkboxId": provider.checkbox_id,
            "selectId": provider.select_id,
            "responseId": provider.response_id,
            "textId": provider.text_id,
            "endpoint": provider.ask_endpoint,
            "deepThinkModel": provider.pro_model,
            "deepThinkLabel": get_model_label(provider.pro_model),
            # None = alle Modelle; eine Liste = nur diese Modelle. Das alte
            # Boolean bleibt additiv fuer noch gecachte Browser erhalten.
            "attachmentModels": (
                None
                if provider.attachment_models is None
                else sorted(provider.attachment_models)
            ),
            "handlesAttachments": provider.model_accepts_attachments(
                provider.base_model
            ),
        }
        for provider in PROVIDERS.values()
    ]


def get_model_picker_metadata() -> dict[str, dict[str, str]]:
    return {
        model_id: {
            "label": get_model_label(model_id),
            "badge": get_model_badge(model_id),
        }
        for model_id in ALL_ALLOWED_MODELS
    }


def model_picker_sort_key(model_id: str):
    config = get_model_config(model_id)
    label = config.label if config else model_id
    is_premium = model_id in PREMIUM_MODELS
    return (is_premium, label.lower(), model_id.lower())


def get_ordered_models(provider: str) -> list[str]:
    """Modelle eines Providers in Anzeige-Reihenfolge fuer die normalen Picker.
    Die vom Admin gepflegte Reihenfolge (MODEL_ORDER_BY_PROVIDER) gewinnt; alle
    erlaubten Modelle ohne explizite Position werden deterministisch angehaengt,
    damit neu hinzugefuegte Modelle nie verschwinden."""
    allowed = _provider_allowed_sets().get(provider, set())
    ordered = [model for model in MODEL_ORDER_BY_PROVIDER.get(provider, []) if model in allowed]
    seen = set(ordered)
    rest = sorted((model for model in allowed if model not in seen), key=model_picker_sort_key)
    return ordered + rest


def apply_model_order(order_by_provider: dict | None) -> None:
    """Uebernimmt die Admin-Reihenfolge je Provider (auf erlaubte Modelle gefiltert)."""
    data = order_by_provider or {}
    allowed_sets = _provider_allowed_sets()
    for provider in MODEL_ORDER_BY_PROVIDER:
        incoming = data.get(provider)
        allowed = allowed_sets.get(provider, set())
        if isinstance(incoming, list):
            seen = set()
            ordered = []
            for model in incoming:
                model = canonical_model_id(model, provider)
                if model in allowed and model not in seen:
                    seen.add(model)
                    ordered.append(model)
            MODEL_ORDER_BY_PROVIDER[provider] = ordered
        else:
            MODEL_ORDER_BY_PROVIDER[provider] = []


def apply_default_models(defaults: dict | None) -> None:
    """Setzt den Free-Default je Provider. Ein Override gilt nur, wenn das Modell
    erlaubt und nicht Premium ist. Sonst greift die Basis."""
    overrides = defaults or {}
    allowed_sets = _provider_allowed_sets()
    for provider, base in _BASE_FREE_DEFAULTS.items():
        chosen = canonical_model_id(overrides.get(provider), provider)
        allowed = allowed_sets.get(provider, set())
        if not (chosen and chosen in allowed and chosen not in PREMIUM_MODELS):
            chosen = next(
                (
                    candidate for candidate in (base, *get_ordered_models(provider))
                    if candidate in allowed and candidate not in PREMIUM_MODELS
                ),
                "",
            )
        FREE_DEFAULT_MODEL_BY_PROVIDER[provider] = chosen


def apply_watch_models(config: dict | None) -> None:
    """Apply validated per-tier Watch model mappings with legacy fallbacks."""
    incoming = config if isinstance(config, dict) else {}
    allowed_sets = _provider_allowed_sets()
    for tier in ("free", "pro"):
        tier_data = incoming.get(tier)
        tier_data = tier_data if isinstance(tier_data, dict) else {}
        clean = {}
        for provider in DEFAULT_MODEL_BY_PROVIDER:
            model = canonical_model_id(tier_data.get(provider), provider)
            if not model or model not in allowed_sets.get(provider, set()):
                continue
            if tier == "free" and model in PREMIUM_MODELS:
                continue
            clean[provider] = model
        if len(clean) < 2:
            clean = {}
            for provider in DEFAULT_MODEL_BY_PROVIDER:
                candidates = (
                    FREE_DEFAULT_MODEL_BY_PROVIDER.get(provider),
                    _BASE_WATCH_MODELS_BY_TIER[tier].get(provider),
                    *get_ordered_models(provider),
                )
                model = next(
                    (
                        candidate for candidate in candidates
                        if candidate in allowed_sets.get(provider, set())
                        and (tier == "pro" or candidate not in PREMIUM_MODELS)
                    ),
                    "",
                )
                if model:
                    clean[provider] = model
                if len(clean) >= 3:
                    break
        WATCH_MODELS_BY_TIER[tier].clear()
        WATCH_MODELS_BY_TIER[tier].update(clean)


def get_watch_models(tier) -> dict[str, str]:
    """Watch-Modelle der Stufe. Plus faehrt bewusst die Free-Modelle: es
    bekommt mehr Watches, aber keine teureren Laeufe."""
    key = TIER_PRO if normalize_tier(tier) == TIER_PRO else TIER_FREE
    return dict(WATCH_MODELS_BY_TIER[key])


def apply_watch_consensus_models(config: dict | None) -> None:
    """Apply one valid Consensus engine per Watch tier with safe fallbacks."""
    incoming = config if isinstance(config, dict) else {}
    for tier in ("free", "pro"):
        chosen = canonical_model_id(incoming.get(tier))
        resolved = get_consensus_model_config(chosen)
        if not resolved or not resolved.provider:
            chosen = _BASE_WATCH_CONSENSUS_MODELS_BY_TIER[tier]
        if tier == "free" and is_premium_consensus_model(chosen):
            chosen = _BASE_WATCH_CONSENSUS_MODELS_BY_TIER[tier]
        WATCH_CONSENSUS_MODELS_BY_TIER[tier] = chosen


def get_watch_consensus_model(tier) -> str:
    key = TIER_PRO if normalize_tier(tier) == TIER_PRO else TIER_FREE
    return WATCH_CONSENSUS_MODELS_BY_TIER[key]


rebuild_model_configs()


def _coerce_limit(value, fallback: int) -> int:
    if isinstance(value, bool):
        return fallback
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed >= 0 else fallback


def _sync_limit_constants():
    global MAX_WORDS, DEEP_SEARCH_MAX_WORDS, MAX_TOKENS, DEEP_SEARCH_MAX_TOKENS
    global CONSENSUS_MAX_TOKENS, DIFFERENCES_MAX_TOKENS, COVERAGE_MAX_TOKENS
    global GEMINI_MAX_TOKENS, GEMINI_DEEP_MAX_TOKENS

    MAX_WORDS = LIMITS["free_max_words"]
    DEEP_SEARCH_MAX_WORDS = LIMITS["pro_deep_search_max_words"]
    MAX_TOKENS = LIMITS["pro_max_tokens"]
    DEEP_SEARCH_MAX_TOKENS = LIMITS["pro_deep_search_max_tokens"]
    CONSENSUS_MAX_TOKENS = LIMITS["consensus_max_tokens"]
    DIFFERENCES_MAX_TOKENS = LIMITS["differences_max_tokens"]
    COVERAGE_MAX_TOKENS = LIMITS["coverage_max_tokens"]
    GEMINI_MAX_TOKENS = MAX_TOKENS
    GEMINI_DEEP_MAX_TOKENS = DEEP_SEARCH_MAX_TOKENS


def normalize_limits_config(limits_data=None) -> dict:
    incoming = limits_data if isinstance(limits_data, dict) else {}
    return {
        key: _coerce_limit(incoming.get(key, fallback), fallback)
        for key, fallback in DEFAULT_LIMITS.items()
    }


def apply_limits(limits_data=None):
    normalized = normalize_limits_config(limits_data)

    LIMITS.clear()
    LIMITS.update(normalized)
    _sync_limit_constants()


def get_limits_config() -> dict:
    return dict(LIMITS)


_MEMORY_EDIT_INTEGER_RANGES = {
    "memory_free_chars": (1_000, 12_000),
    "memory_plus_chars": (1_000, 24_000),
    "memory_pro_chars": (1_000, 24_000),
    "memory_free_ai_edits_daily": (0, 100),
    "memory_plus_ai_edits_daily": (0, 500),
    "memory_pro_ai_edits_daily": (0, 500),
    "memory_ai_edits_per_minute": (1, 20),
    "memory_edit_input_chars": (50, 2_000),
    "memory_edit_output_tokens": (32, 500),
    "memory_edit_timeout_seconds": (1, 30),
    "memory_global_calls_daily": (0, 100_000),
}


def normalize_memory_edit_config(value=None) -> dict:
    incoming = value if isinstance(value, dict) else {}
    result = dict(DEFAULT_MEMORY_EDIT_CONFIG)
    if isinstance(incoming.get("memory_edit_enabled"), bool):
        result["memory_edit_enabled"] = incoming["memory_edit_enabled"]
    model = incoming.get("memory_edit_model")
    if isinstance(model, str) and model.strip() in ALLOWED_OPENAI_MODELS:
        result["memory_edit_model"] = model.strip()
    for key, (minimum, maximum) in _MEMORY_EDIT_INTEGER_RANGES.items():
        raw = incoming.get(key)
        if isinstance(raw, bool):
            continue
        try:
            parsed = int(raw)
        except (TypeError, ValueError):
            continue
        if minimum <= parsed <= maximum:
            result[key] = parsed
    if result["memory_pro_chars"] < result["memory_free_chars"]:
        result["memory_pro_chars"] = DEFAULT_MEMORY_EDIT_CONFIG["memory_pro_chars"]
    # Plus liegt zwischen Free und Pro. Ein Admin-Wert, der aus dieser Spanne
    # faellt, wuerde die Stufenordnung umdrehen - dann gilt wieder der Default.
    if not (result["memory_free_chars"] <= result["memory_plus_chars"] <= result["memory_pro_chars"]):
        result["memory_plus_chars"] = min(
            max(DEFAULT_MEMORY_EDIT_CONFIG["memory_plus_chars"], result["memory_free_chars"]),
            result["memory_pro_chars"],
        )
    return result


def apply_memory_edit_config(value=None) -> None:
    MEMORY_EDIT_CONFIG.clear()
    MEMORY_EDIT_CONFIG.update(normalize_memory_edit_config(value))


def get_memory_edit_config() -> dict:
    return dict(MEMORY_EDIT_CONFIG)


# --- Tier-abhaengige Limits ------------------------------------------------
# Die Getter hiessen frueher alle ``is_pro: bool``. Sie heissen jetzt ``tier``
# und nehmen weiterhin einen Bool entgegen (True -> Pro, False -> Free), damit
# bestehende Aufrufer und Tests unveraendert funktionieren; normalize_tier
# uebersetzt beides auf "free"/"plus"/"pro".


def _tier_limit(tier, keys: dict[str, str]) -> int:
    """Limit fuer die Stufe.

    Fehlt der Schluessel der Stufe - im Mapping oder in LIMITS selbst -, gilt
    der Free-Wert. Nie der Pro-Wert: eine Luecke in der Konfiguration darf
    hoechstens Komfort kosten, nie Geld.
    """
    key = keys.get(normalize_tier(tier))
    if key is None or key not in LIMITS:
        key = keys[TIER_FREE]
    return LIMITS[key]


def get_memory_char_limit(tier) -> int:
    keys = {
        TIER_FREE: "memory_free_chars",
        TIER_PLUS: "memory_plus_chars",
        TIER_PRO: "memory_pro_chars",
    }
    resolved = normalize_tier(tier)
    return int(MEMORY_EDIT_CONFIG[keys[resolved]])


def get_memory_ai_edit_limit(tier) -> int:
    keys = {
        TIER_FREE: "memory_free_ai_edits_daily",
        TIER_PLUS: "memory_plus_ai_edits_daily",
        TIER_PRO: "memory_pro_ai_edits_daily",
    }
    resolved = normalize_tier(tier)
    return int(MEMORY_EDIT_CONFIG[keys[resolved]])


def get_consensus_run_limit(tier) -> int:
    return _tier_limit(tier, {
        TIER_FREE: "free_consensus_run_limit",
        TIER_PLUS: "plus_consensus_run_limit",
        TIER_PRO: "pro_consensus_run_limit",
    })


def get_deep_think_run_limit(tier) -> int:
    """Deep Think faehrt Frontier-Modelle und bleibt deshalb Pro vorbehalten.
    Plus bekommt hier bewusst kein eigenes Admin-Feld: ein Kontingent, das die
    Capability-Pruefung ohnehin blockiert, waere ein toter Schalter."""
    if normalize_tier(tier) == TIER_PRO:
        return LIMITS["pro_deep_think_run_limit"]
    return LIMITS["free_deep_think_run_limit"]


def get_word_limit(tier, deep_search: bool = False) -> int:
    if deep_search:
        # Deep Search gibt es nur mit Pro; alle anderen Stufen sehen das
        # Free-Limit (in der Regel 0).
        return _tier_limit(tier, {
            TIER_FREE: "free_deep_search_max_words",
            TIER_PRO: "pro_deep_search_max_words",
        })
    return _tier_limit(tier, {
        TIER_FREE: "free_max_words",
        TIER_PLUS: "plus_max_words",
        TIER_PRO: "pro_max_words",
    })


def get_consensus_answer_char_limit() -> int:
    return LIMITS["consensus_max_answer_chars"]


def get_consensus_question_char_limit() -> int:
    return LIMITS["consensus_max_question_chars"]


def get_followup_question_char_limit() -> int:
    return LIMITS["followup_max_question_chars"]


def get_followup_consensus_char_limit() -> int:
    return LIMITS["followup_max_consensus_chars"]


def get_watch_active_limit(tier) -> int:
    return max(0, _tier_limit(tier, {
        TIER_FREE: "watch_free_active_limit",
        TIER_PLUS: "watch_plus_active_limit",
        TIER_PRO: "watch_pro_active_limit",
    }))


def get_watch_max_runs_per_day() -> int:
    return max(0, int(LIMITS["watch_max_runs_per_day"]))


def watch_daily_requires_pro() -> bool:
    return int(LIMITS["watch_daily_interval_requires_pro"]) != 0


def watch_plus_daily_allowed() -> bool:
    return int(LIMITS["watch_plus_daily_interval_allowed"]) != 0


def is_watch_daily_allowed(tier) -> bool:
    resolved = normalize_tier(tier)
    if resolved == TIER_PRO:
        return True
    if not watch_daily_requires_pro():
        return True
    return resolved == TIER_PLUS and watch_plus_daily_allowed()


def get_output_token_limit(tier, deep_search: bool = False) -> int:
    if deep_search:
        return _tier_limit(tier, {
            TIER_FREE: "free_deep_search_max_tokens",
            TIER_PRO: "pro_deep_search_max_tokens",
        })
    return _tier_limit(tier, {
        TIER_FREE: "free_max_tokens",
        TIER_PLUS: "plus_max_tokens",
        TIER_PRO: "pro_max_tokens",
    })


def _capture_runtime_config() -> dict:
    return {
        "providers": {
            provider: set(models) for provider, models in _provider_allowed_sets().items()
        },
        "premium": set(PREMIUM_MODELS),
        "consensus": list(ALLOWED_CONSENSUS_MODELS),
        "presets": get_consensus_preset_models(),
        "deep_think": DEEP_THINK_CONSENSUS_MODEL,
        "judges": dict(DIFFERENCES_JUDGE_MODEL_BY_PROVIDER),
        "pro_judges": dict(PRO_JUDGE_MODEL_BY_PROVIDER),
        "memory": dict(CHAT_MEMORY_MODEL_BY_PROVIDER),
        "families": dict(JUDGE_FAMILY_BY_ENGINE),
        "order": {key: list(value) for key, value in MODEL_ORDER_BY_PROVIDER.items()},
        "defaults": dict(FREE_DEFAULT_MODEL_BY_PROVIDER),
        "watch": {key: dict(value) for key, value in WATCH_MODELS_BY_TIER.items()},
        "watch_consensus": dict(WATCH_CONSENSUS_MODELS_BY_TIER),
        "limits": dict(LIMITS),
        "memory_edit_config": dict(MEMORY_EDIT_CONFIG),
    }


def _restore_runtime_config(state: dict) -> None:
    global ALL_ALLOWED_MODELS, DEEP_THINK_CONSENSUS_MODEL
    for provider, target in _provider_allowed_sets().items():
        target.clear()
        target.update(state["providers"].get(provider, set()))
    PREMIUM_MODELS.clear()
    PREMIUM_MODELS.update(state["premium"])
    ALLOWED_CONSENSUS_MODELS.clear()
    ALLOWED_CONSENSUS_MODELS.extend(state["consensus"])
    CONSENSUS_PRESET_MODELS.clear()
    CONSENSUS_PRESET_MODELS.update(
        {
            key: {
                "answers": dict(value["answers"]),
                "consensus": value["consensus"],
            }
            for key, value in state["presets"].items()
        }
    )
    DEEP_THINK_CONSENSUS_MODEL = state["deep_think"]
    for target, key in (
        (DIFFERENCES_JUDGE_MODEL_BY_PROVIDER, "judges"),
        (PRO_JUDGE_MODEL_BY_PROVIDER, "pro_judges"),
        (CHAT_MEMORY_MODEL_BY_PROVIDER, "memory"),
        (JUDGE_FAMILY_BY_ENGINE, "families"),
        (FREE_DEFAULT_MODEL_BY_PROVIDER, "defaults"),
    ):
        target.clear()
        target.update(state[key])
    for provider in MODEL_ORDER_BY_PROVIDER:
        MODEL_ORDER_BY_PROVIDER[provider] = list(state["order"].get(provider, []))
    for tier in WATCH_MODELS_BY_TIER:
        WATCH_MODELS_BY_TIER[tier].clear()
        WATCH_MODELS_BY_TIER[tier].update(state["watch"].get(tier, {}))
    WATCH_CONSENSUS_MODELS_BY_TIER.clear()
    WATCH_CONSENSUS_MODELS_BY_TIER.update(state["watch_consensus"])
    LIMITS.clear()
    LIMITS.update(state["limits"])
    MEMORY_EDIT_CONFIG.clear()
    MEMORY_EDIT_CONFIG.update(state["memory_edit_config"])
    _sync_limit_constants()
    ALL_ALLOWED_MODELS = _all_allowed_models()
    rebuild_model_configs()


def load_models_from_db(*, strict: bool = False, persist_backfill: bool = True) -> bool:
    global ALL_ALLOWED_MODELS
    import logging
    from app.core.observability import safe_exception
    from app.core.security import db_firestore
    _RUNTIME_CONFIG_LOCK.acquire()
    previous_state = _capture_runtime_config()
    try:
        doc_ref = db_firestore.collection("app_config").document("models")
        # Model defaults are already present in code. On quota exhaustion we
        # must fail fast instead of letting Firestore's default retry policy
        # hold application startup for up to five minutes.
        doc = doc_ref.get(timeout=5.0, retry=None)
        if doc.exists:
            data = doc.to_dict()
            new_provider_keys = [
                provider for provider in PROVIDERS if provider not in data
            ]
            
            # Modellliste je Familie: Firestore ist autoritativ, danach
            # Migration alter IDs und Abzug der stillgelegten.
            for provider in PROVIDERS.values():
                incoming = data.get(provider.key)
                if incoming is None:
                    continue
                provider.models.clear()
                provider.models.update(canonical_model_ids(incoming, provider.key))
                provider.models.difference_update(
                    PROVIDER_DEPRECATED_MODELS.get(provider.key, set())
                )
                provider.models.difference_update(REMOVED_MODEL_IDS)

            # Update Premium
            if "premium" in data:
                PREMIUM_MODELS.clear()
                PREMIUM_MODELS.update(data["premium"])
                PREMIUM_MODELS.difference_update(REMOVED_MODEL_IDS)
                for deprecated in PROVIDER_DEPRECATED_MODELS.values():
                    PREMIUM_MODELS.difference_update(deprecated)
                PREMIUM_MODELS.intersection_update(_all_allowed_models())
                # Der <Familie>-Pro-Alias ist ein harter Produktvertrag. Bei
                # neuen Registry-Familien kennt ein altes Firestore-Dokument
                # deren Pro-Modell noch nicht; es darf dadurch nicht Free sein.
                PREMIUM_MODELS.update(
                    provider.pro_model for provider in PROVIDERS.values()
                    if provider.pro_model in provider.models
                )
            ALL_ALLOWED_MODELS = _all_allowed_models()
            rebuild_model_configs()

            # Preset-Model-Sets brauchen die finalen Provider-/Tier-Listen und
            # muessen vor der Consensus-Normalisierung aktiv sein, damit ihre
            # Consensus-Engines sicher im nativen Picker landen.
            apply_consensus_preset_models(data.get("preset_models"))

            # Deep-Think-Modell VOR der Consensus-Normalisierung anwenden,
            # damit normalize_consensus_models das konfigurierte Modell in der
            # Liste sicherstellt.
            apply_deep_think_model(data.get("deep_think_model"))

            # Judges (Differences/Resolve) je Provider; braucht die
            # aktualisierten Provider-Listen fuer die Validierung.
            apply_judge_models(data.get("judge_models"))
            apply_pro_judge_models(data.get("judge_models_pro"))
            apply_judge_families(data.get("judge_families"))
            apply_chat_memory_models(data.get("chat_memory_models"))

            if "consensus" in data:
                ALLOWED_CONSENSUS_MODELS.clear()
                ALLOWED_CONSENSUS_MODELS.extend(normalize_consensus_models(data["consensus"]))
            else:
                ALLOWED_CONSENSUS_MODELS.clear()
                ALLOWED_CONSENSUS_MODELS.extend(normalize_consensus_models(DEFAULT_CONSENSUS_MODELS))
            for provider_key in new_provider_keys:
                label = PROVIDERS[provider_key].label
                for alias in (label, f"{label}-Pro"):
                    if alias not in ALLOWED_CONSENSUS_MODELS:
                        ALLOWED_CONSENSUS_MODELS.append(alias)

            # Admin-gepflegte Picker-Reihenfolge (aus den geordneten Provider-Listen)
            # und Free-Default je Provider uebernehmen.
            apply_model_order({provider: data.get(provider) for provider in MODEL_ORDER_BY_PROVIDER})
            apply_default_models(data.get("defaults"))
            apply_watch_models(data.get("watch_models"))
            apply_watch_consensus_models(data.get("watch_consensus_models"))

            apply_limits(data.get("limits"))
            apply_memory_edit_config(data.get("memory_edit"))
            # Schema-Backfill: neue Limitfelder (z. B. die run-basierten UTC-
            # Tageslimits) nicht nur im Prozess defaulten, sondern im Admin-
            # Dokument persistieren. Bestehende gueltige Adminwerte bleiben
            # durch apply_limits/get_limits_config erhalten.
            normalized_limits = get_limits_config()
            if persist_backfill and data.get("limits") != normalized_limits:
                doc_ref.set(
                    {"limits": normalized_limits}, merge=True,
                    timeout=5.0, retry=None,
                )
            normalized_memory_edit = get_memory_edit_config()
            if persist_backfill and data.get("memory_edit") != normalized_memory_edit:
                doc_ref.set(
                    {"memory_edit": normalized_memory_edit}, merge=True,
                    timeout=5.0, retry=None,
                )
            normalized_watch_consensus = dict(WATCH_CONSENSUS_MODELS_BY_TIER)
            if (
                persist_backfill
                and data.get("watch_consensus_models") != normalized_watch_consensus
            ):
                doc_ref.set(
                    {"watch_consensus_models": normalized_watch_consensus}, merge=True,
                    timeout=5.0, retry=None,
                )
            schema_backfill = {
                provider.key: sorted(provider.models)
                for provider in PROVIDERS.values()
                if provider.key not in data
            }
            normalized_presets = get_consensus_preset_models()
            if data.get("preset_models") != normalized_presets:
                schema_backfill["preset_models"] = normalized_presets
            normalized_runtime_fields = {
                "premium": sorted(PREMIUM_MODELS),
                "consensus": list(ALLOWED_CONSENSUS_MODELS),
                "defaults": dict(FREE_DEFAULT_MODEL_BY_PROVIDER),
                "judge_models": get_judge_models(),
                "judge_models_pro": get_pro_judge_models(),
                "chat_memory_models": get_chat_memory_models(),
            }
            for field_name, normalized_value in normalized_runtime_fields.items():
                current_value = data.get(field_name)
                if field_name == "premium":
                    current_value = sorted(current_value or [])
                if current_value != normalized_value:
                    schema_backfill[field_name] = normalized_value
            if schema_backfill and persist_backfill:
                doc_ref.set(
                    schema_backfill, merge=True, timeout=5.0, retry=None,
                )
            # Update ALL_ALLOWED_MODELS
            ALL_ALLOWED_MODELS = _all_allowed_models()
            rebuild_model_configs()
            logging.info("Models configuration loaded from Firestore successfully.")
        elif persist_backfill:
            # If document doesn't exist, create it with default values
            doc_ref.set({
                **{
                    provider.key: sorted(provider.models)
                    for provider in PROVIDERS.values()
                },
                "premium": list(PREMIUM_MODELS),
                "consensus": list(ALLOWED_CONSENSUS_MODELS),
                "preset_models": get_consensus_preset_models(),
                "deep_think_model": DEEP_THINK_CONSENSUS_MODEL,
                "judge_models": get_judge_models(),
                "judge_models_pro": get_pro_judge_models(),
                "judge_families": get_judge_families(),
                "chat_memory_models": get_chat_memory_models(),
                "watch_models": {
                    tier: dict(models) for tier, models in WATCH_MODELS_BY_TIER.items()
                },
                "watch_consensus_models": dict(WATCH_CONSENSUS_MODELS_BY_TIER),
                "limits": get_limits_config(),
                "memory_edit": get_memory_edit_config(),
            }, timeout=5.0, retry=None)
            rebuild_model_configs()
            logging.info("Created default models configuration in Firestore.")
        else:
            logging.info(
                "Models configuration is missing; using code defaults until the "
                "supervised post-readiness backfill runs."
            )
        return True
    except Exception as exc:
        _restore_runtime_config(previous_state)
        logging.error(
            "Failed to load models from Firestore category=%s",
            safe_exception(exc),
        )
        if strict:
            raise
        return False
    finally:
        _RUNTIME_CONFIG_LOCK.release()

DEEP_THINK_PROMPT = "Deep Think: Focus as hard as you can! But only on the essentials."
