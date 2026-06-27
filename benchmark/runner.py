"""Orchestrierung: Zellen-Loop, JSONL-Writer, Resume, Budget, Dry-Run (Plan §5/§8).

Eine "Zelle" = ein API-Call. Rollen: ``model`` (6 Provider), ``consensus`` und
optional ``synth_alone``. ``query_differences`` wird in v1 nicht aufgerufen (E7).

Der Hauptpfad nutzt die produktive Consensus-Logik **unveraendert mit
Modellnamen** (E5). Transport und Consensus sind injizierbar, damit der Loop ohne
echte Calls testbar ist; in Phase 2 wird ausschliesslich der Dry-Run gefahren
(``--dry-run``) – **kein echter Call**.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import app.core.config as cfg
from app.services.llm.engines import build_provider_payload

from benchmark import audit, config, cost, transport
from benchmark.parse import extract_letter, grade, majority_vote
from benchmark.prompt import build_mc_question

logger = logging.getLogger(__name__)

CONSENSUS_MAX_TOKENS = int(cfg.CONSENSUS_MAX_TOKENS)


# --- Resume ----------------------------------------------------------------


def cell_key(question_id: int, role: str, provider: str) -> tuple:
    """Idempotenz-Key einer Zelle (Plan §8)."""
    return (int(question_id), str(role), str(provider))


def load_done_keys(jsonl_path: Path) -> set[tuple]:
    """Liest ``calls.jsonl`` und liefert die Keys erfolgreich erledigter Zellen
    (``error`` ist None/leer). Fehlerhafte Zellen gelten als nicht erledigt."""
    path = Path(jsonl_path)
    done: set[tuple] = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("error"):
            continue
        done.add(cell_key(record.get("question_id"), record.get("role"), record.get("provider")))
    return done


# --- Budget ----------------------------------------------------------------


def should_stop_for_budget(spent: float, next_estimate: float, cap: float | None) -> bool:
    """True, wenn die naechste Zelle das Budget-Cap sprengen wuerde (Plan §8)."""
    if cap is None:
        return False
    return (spent + next_estimate) > cap


# --- Kostenschaetzung pro Zelle (Dry-Run) ----------------------------------


def estimate_model_cell(api_model: str, prompt_text: str, output_cap: int) -> dict:
    """Schaetzt Input-Tokens (len/4) + Output-Cap einer Modell-Zelle und die
    daraus projizierten Maximalkosten."""
    prompt_tokens = cost.estimate_tokens(prompt_text)
    est = cost.est_cost_usd(api_model, prompt_tokens, output_cap)
    return {"prompt_tokens": prompt_tokens, "output_cap": output_cap, "est_cost_usd": est}


# --- JSONL-Writer ----------------------------------------------------------


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


# --- Runner ----------------------------------------------------------------


@dataclass
class DryRunReport:
    cells: int = 0
    model_cells: int = 0
    consensus_cells: int = 0
    synth_cells: int = 0
    projected_cost_usd: float = 0.0
    audited_payloads: int = 0
    missing_pricing: set = field(default_factory=set)


class BenchmarkRunner:
    def __init__(
        self,
        *,
        models=config.MODELS,
        consensus_model: str = config.CONSENSUS_MODEL,
        system_prompt: str = config.SYSTEM_PROMPT,
        output_tokens: int = config.OUTPUT_TOKEN_LIMIT,
        label_mode: str = config.DEFAULT_LABEL_MODE,
        include_synth_alone: bool = config.INCLUDE_SYNTH_ALONE,
    ):
        self.models = list(models)
        self.consensus_model = consensus_model
        self.system_prompt = system_prompt
        self.output_tokens = int(output_tokens)
        self.label_mode = label_mode
        self.include_synth_alone = include_synth_alone

    def build_model_request(self, model: config.BenchmarkModel, user_prompt: str) -> dict:
        """Baut den closed-book Payload eines Modells (benchmark_mode=True)."""
        return build_provider_payload(
            model.provider,
            question=user_prompt,
            system_prompt=self.system_prompt,
            model_override=model.internal_id,
            max_output_tokens=self.output_tokens,
            benchmark_mode=True,
        )

    def dry_run(self, records: list[dict]) -> DryRunReport:
        """Baut alle Modell-Payloads, auditiert jeden (assert_no_web_tools) und
        projiziert die Maximalkosten – **ohne HTTP** (Plan §8)."""
        report = DryRunReport()
        for record in records:
            user_prompt = build_mc_question(record["question"], record["options"])
            for model in self.models:
                request_data = self.build_model_request(model, user_prompt)
                audit.assert_no_web_tools(
                    request_data["payload"], context=f"{model.provider}:{record['question_id']}"
                )
                report.audited_payloads += 1
                if not cost.has_pricing(model.api_model):
                    report.missing_pricing.add(model.api_model)
                est = estimate_model_cell(model.api_model, user_prompt, self.output_tokens)
                report.projected_cost_usd += est["est_cost_usd"]
                report.model_cells += 1

            # Consensus-Zelle: konservative Obergrenze (6 Kandidatenantworten als
            # Input + Frage, Output = Consensus-Cap).
            consensus_api_model = _consensus_api_model(self.consensus_model)
            consensus_input_tokens = 6 * self.output_tokens + cost.estimate_tokens(user_prompt)
            report.projected_cost_usd += cost.est_cost_usd(
                consensus_api_model, consensus_input_tokens, CONSENSUS_MAX_TOKENS
            )
            if not cost.has_pricing(consensus_api_model):
                report.missing_pricing.add(consensus_api_model)
            report.consensus_cells += 1

            if self.include_synth_alone:
                report.projected_cost_usd += cost.est_cost_usd(
                    consensus_api_model, cost.estimate_tokens(user_prompt), CONSENSUS_MAX_TOKENS
                )
                report.synth_cells += 1

        report.cells = report.model_cells + report.consensus_cells + report.synth_cells
        return report


def _consensus_api_model(consensus_model: str) -> str:
    """Loest das Consensus-Modell auf seinen api_model auf (fuer Pricing)."""
    model_config = cfg.get_model_config(consensus_model)
    return model_config.api_model if model_config else consensus_model
