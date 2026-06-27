"""Orchestrierung: Zellen-Loop, JSONL-Writer, Resume, Budget, Dry-Run (Plan §5/§8).

Eine "Zelle" = ein API-Call. Rollen: ``model`` (6 Provider), ``consensus`` und
optional ``synth_alone``. ``query_differences`` wird in v1 nicht aufgerufen (E7).

Der Hauptpfad nutzt die produktive Consensus-Logik **unveraendert mit
Modellnamen** (E5). Transport und Consensus sind injizierbar, damit der gesamte
``run()``-Loop (JSONL-Append, Resume, Budget-Stopp) ohne HTTP und ohne echte
API-Keys end-to-end testbar ist. In Phase 2 wird kein echter Call gefahren –
weder ueber ``--dry-run`` noch ueber ``run()``; der reale Pilot ist Phase 3.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import app.core.config as cfg
from app.services.llm.engines import build_provider_payload

from benchmark import audit, config, cost, transport
from benchmark.parse import NO_MAJORITY, extract_letter, grade, majority_vote
from benchmark.prompt import build_mc_question

logger = logging.getLogger(__name__)

CONSENSUS_MAX_TOKENS = int(cfg.CONSENSUS_MAX_TOKENS)


# --- Resume ----------------------------------------------------------------


def cell_key(question_id: int, role: str, provider: str) -> tuple:
    """Idempotenz-Key einer Zelle (Plan §8)."""
    return (int(question_id), str(role), str(provider))


def load_existing_records(jsonl_path: Path) -> list[dict]:
    """Liest alle Zellen-Records aus ``calls.jsonl`` (append-only, inkl.
    Fehlerzeilen). Robust gegen leere/kaputte Zeilen."""
    path = Path(jsonl_path)
    if not path.exists():
        return []
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            continue
    return records


def index_existing(records: list[dict]) -> dict[tuple, dict]:
    """Aggregiert die Records je Zellen-Key zu ``{success, errors}``.

    ``success`` ist der letzte erfolgreiche Record (oder None), ``errors`` die
    Liste der Fehlversuche. Spaetere Erfolge ueberschreiben fruehere Fehler
    (append-only Retry-Semantik).
    """
    index: dict[tuple, dict] = {}
    for record in records:
        key = cell_key(record.get("question_id"), record.get("role"), record.get("provider"))
        slot = index.setdefault(key, {"success": None, "errors": []})
        if record.get("error"):
            slot["errors"].append(record)
        else:
            slot["success"] = record
    return index


def load_done_keys(jsonl_path: Path) -> set[tuple]:
    """Keys erfolgreich erledigter Zellen (``error`` leer). Fehlerhafte Zellen
    gelten als nicht erledigt."""
    index = index_existing(load_existing_records(jsonl_path))
    return {key for key, slot in index.items() if slot["success"]}


def compute_skip_keys(index: dict[tuple, dict], retry_failed: bool) -> set[tuple]:
    """Zellen-Keys, die beim Resume uebersprungen werden.

    Erfolgreiche Zellen werden **immer** uebersprungen. Reine Fehl-Zellen werden
    nur dann erneut versucht, wenn ``retry_failed=True`` – sonst ebenfalls
    uebersprungen (kontrollierter Resume, Plan §8).
    """
    skip: set[tuple] = set()
    for key, slot in index.items():
        if slot["success"]:
            skip.add(key)
        elif slot["errors"] and not retry_failed:
            skip.add(key)
    return skip


def spent_from_index(index: dict[tuple, dict]) -> float:
    """Bereits verbuchte Ist-Kosten aus den erfolgreichen Zellen (fuer Resume)."""
    total = 0.0
    for slot in index.values():
        if slot["success"]:
            total += float(slot["success"].get("est_cost_usd") or 0.0)
    return total


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


@dataclass
class RunResult:
    run_id: str = ""
    cells_written: int = 0
    cells_skipped: int = 0
    cells_failed: int = 0
    spent_usd: float = 0.0
    stopped: bool = False
    stop_reason: str = ""


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

    # --- Realer Lauf -------------------------------------------------------

    def run(
        self,
        records: list[dict],
        *,
        run_dir: Path,
        api_keys: dict | None = None,
        transport_execute=transport.execute,
        consensus_fn=None,
        budget: float | None = None,
        retry_failed: bool = False,
    ) -> RunResult:
        """Faehrt den Zellen-Loop: pro Frage 6 Modell-Calls, Consensus, optional
        Synthesizer-allein. Schreibt jede Zelle als JSONL-Zeile (append-only),
        ueberspringt beim Resume erfolgreiche (und – ausser ``retry_failed`` –
        fehlerhafte) Zellen und stoppt **vor** dem naechsten Call, wenn das
        Budget-Cap gesprengt wuerde.

        ``transport_execute`` und ``consensus_fn`` sind injizierbar, damit der
        gesamte Loop ohne HTTP/echte Keys getestet werden kann.
        """
        run_dir = Path(run_dir)
        calls_path = run_dir / "calls.jsonl"
        run_id = run_dir.name
        api_keys = api_keys or {}

        index = index_existing(load_existing_records(calls_path))
        skip = compute_skip_keys(index, retry_failed)
        spent = spent_from_index(index)

        consensus_api_model = _consensus_api_model(self.consensus_model)
        if consensus_fn is None:
            consensus_fn = _default_consensus_fn(api_keys, self.consensus_model)

        self.write_or_validate_manifest(run_dir)

        result = RunResult(run_id=run_id, spent_usd=spent)

        for record in records:
            qid = record["question_id"]
            user_prompt = build_mc_question(record["question"], record["options"])
            answers: dict[str, str] = {}

            # --- 6 Modell-Zellen ---
            stopped = False
            for model in self.models:
                key = cell_key(qid, "model", model.provider)
                if key in skip:
                    result.cells_skipped += 1
                    stored = index.get(key, {}).get("success")
                    if stored:
                        answers[model.provider] = stored.get("parsed_text") or ""
                    continue

                est = estimate_model_cell(model.api_model, user_prompt, self.output_tokens)["est_cost_usd"]
                if should_stop_for_budget(spent, est, budget):
                    self._mark_stopped(result, budget, f"{model.provider} q{qid}")
                    stopped = True
                    break

                request_data = self.build_model_request(model, user_prompt)
                audit.assert_no_web_tools(request_data["payload"], context=f"{model.provider}:{qid}")
                api_key = api_keys.get(config.PROVIDER_API_KEY_NAME[model.provider])
                outcome = transport_execute(request_data, api_key)

                cell = self._make_cell_record(
                    run_id=run_id,
                    qrecord=record,
                    role="model",
                    provider=model.provider,
                    internal_model=model.internal_id,
                    api_model=model.api_model,
                    user_prompt=user_prompt,
                    payload=request_data["payload"],
                    outcome=outcome,
                )
                append_jsonl(calls_path, cell)
                spent += cell["est_cost_usd"]
                result.spent_usd = spent
                if outcome.get("error"):
                    result.cells_failed += 1
                else:
                    result.cells_written += 1
                    answers[model.provider] = cell["parsed_text"]

            if stopped:
                return result

            # --- Consensus-Zelle (Produktionslogik, mit Modellnamen) ---
            ckey = cell_key(qid, "consensus", self.consensus_model)
            if ckey in skip:
                result.cells_skipped += 1
            else:
                est = self._consensus_estimate(consensus_api_model, user_prompt)
                if should_stop_for_budget(spent, est, budget):
                    self._mark_stopped(result, budget, f"consensus q{qid}")
                    return result
                outcome = self._call_consensus(consensus_fn, record, answers)
                cell = self._make_cell_record(
                    run_id=run_id,
                    qrecord=record,
                    role="consensus",
                    provider=self.consensus_model,
                    internal_model=self.consensus_model,
                    api_model=consensus_api_model,
                    user_prompt=user_prompt,
                    payload={"consensus_model": self.consensus_model, "answer_providers": sorted(answers)},
                    outcome=outcome,
                    est_cost=self._consensus_estimate(consensus_api_model, user_prompt, outcome.get("text")),
                )
                append_jsonl(calls_path, cell)
                spent += cell["est_cost_usd"]
                result.spent_usd = spent
                result.cells_failed += 1 if outcome.get("error") else 0
                result.cells_written += 0 if outcome.get("error") else 1

            # --- Synthesizer allein (optional) ---
            if self.include_synth_alone:
                skey = cell_key(qid, "synth_alone", self.consensus_model)
                if skey in skip:
                    result.cells_skipped += 1
                else:
                    est = estimate_model_cell(consensus_api_model, user_prompt, self.output_tokens)["est_cost_usd"]
                    if should_stop_for_budget(spent, est, budget):
                        self._mark_stopped(result, budget, f"synth_alone q{qid}")
                        return result
                    request_data = build_provider_payload(
                        cfg.get_model_config(self.consensus_model).provider,
                        question=user_prompt,
                        system_prompt=self.system_prompt,
                        model_override=self.consensus_model,
                        max_output_tokens=self.output_tokens,
                        benchmark_mode=True,
                    )
                    audit.assert_no_web_tools(request_data["payload"], context=f"synth:{qid}")
                    api_key = api_keys.get(config.PROVIDER_API_KEY_NAME[request_data["provider"]])
                    outcome = transport_execute(request_data, api_key)
                    cell = self._make_cell_record(
                        run_id=run_id,
                        qrecord=record,
                        role="synth_alone",
                        provider=self.consensus_model,
                        internal_model=self.consensus_model,
                        api_model=consensus_api_model,
                        user_prompt=user_prompt,
                        payload=request_data["payload"],
                        outcome=outcome,
                    )
                    append_jsonl(calls_path, cell)
                    spent += cell["est_cost_usd"]
                    result.spent_usd = spent
                    result.cells_failed += 1 if outcome.get("error") else 0
                    result.cells_written += 0 if outcome.get("error") else 1

        return result

    # --- run()-Helfer ------------------------------------------------------

    def _make_cell_record(
        self,
        *,
        run_id,
        qrecord,
        role,
        provider,
        internal_model,
        api_model,
        user_prompt,
        payload,
        outcome,
        est_cost=None,
    ) -> dict:
        text = outcome.get("text") or ""
        error = outcome.get("error")
        usage = outcome.get("usage") or {"prompt": 0, "completion": 0, "total": 0}
        ground_truth = qrecord["answer"]
        letter = None if error else extract_letter(text, options=qrecord["options"])
        if est_cost is None:
            est_cost = cost.est_cost_usd(api_model, usage.get("prompt", 0), usage.get("completion", 0))
        return {
            "run_id": run_id,
            "ts": _now_iso(),
            "question_id": qrecord["question_id"],
            "category": qrecord["category"],
            "role": role,
            "provider": provider,
            "internal_model": internal_model,
            "api_model": api_model,
            "benchmark_mode": True,
            "label_mode": self.label_mode,
            "system_prompt": self.system_prompt,
            "user_prompt": user_prompt,
            "request_payload": _redact_payload(payload),
            "parsed_text": text,
            "extracted_letter": letter,
            "ground_truth": ground_truth,
            "correct": bool(letter) and grade(letter, ground_truth),
            "abstain": (letter is None) and not error,
            "usage": {
                "prompt": int(usage.get("prompt", 0) or 0),
                "completion": int(usage.get("completion", 0) or 0),
                "total": int(usage.get("total", 0) or 0),
            },
            "est_cost_usd": round(float(est_cost), 8),
            "latency_ms": outcome.get("latency_ms"),
            "http_status": outcome.get("status"),
            "error": error,
            "error_code": outcome.get("error_code"),
        }

    def _call_consensus(self, consensus_fn, record, answers) -> dict:
        """Ruft die Consensus-Synthese und verpackt das Ergebnis wie ein
        Transport-Outcome. query_consensus liefert nur Text (keine Usage) und
        signalisiert Fehler als Text-Prefix – beides wird hier erkannt."""
        try:
            text = consensus_fn(question=record["question"], answers=answers, model_sources=None)
        except Exception as exc:  # noqa: BLE001
            return {"text": "", "usage": None, "latency_ms": None, "status": None,
                    "error": str(exc), "error_code": "consensus_failed"}
        text = text or ""
        if text.strip().startswith(("Consensus error", "Invalid consensus model")):
            return {"text": "", "usage": None, "latency_ms": None, "status": None,
                    "error": text.strip(), "error_code": "consensus_failed"}
        return {"text": text, "usage": None, "latency_ms": None, "status": None,
                "error": None, "error_code": None}

    def _consensus_estimate(self, consensus_api_model, user_prompt, completion_text=None) -> float:
        """Kostenschaetzung der Consensus-Zelle (query_consensus liefert keine
        Usage). Input = 6 Kandidaten-Caps + Frage; Output = Cap oder Ist-Laenge."""
        input_tokens = 6 * self.output_tokens + cost.estimate_tokens(user_prompt)
        completion_tokens = (
            cost.estimate_tokens(completion_text) if completion_text else CONSENSUS_MAX_TOKENS
        )
        return cost.est_cost_usd(consensus_api_model, input_tokens, completion_tokens)

    @staticmethod
    def _mark_stopped(result: RunResult, budget, where: str) -> None:
        result.stopped = True
        result.stop_reason = f"budget cap ${budget} would be exceeded before {where}"

    def build_manifest(self, run_id: str) -> dict:
        """Eingefrorene Run-Config (E6). Enthaelt **keine** Secrets."""
        return {
            "run_id": run_id,
            "created": _now_iso(),
            "label_mode": self.label_mode,
            "include_synth_alone": self.include_synth_alone,
            "consensus_model": self.consensus_model,
            "temperature": config.TEMPERATURE,
            "output_token_limit": self.output_tokens,
            "system_prompt": self.system_prompt,
            "models": config.resolve_pins(),
            "pricing_usd_per_1m": config.PRICING_USD_PER_1M,
        }

    # Felder, die beim Resume mit dem bestehenden Manifest uebereinstimmen
    # muessen (eingefrorene Config, E6).
    _MANIFEST_FROZEN_FIELDS = (
        "label_mode", "include_synth_alone", "consensus_model",
        "temperature", "output_token_limit", "system_prompt", "models",
    )

    def write_or_validate_manifest(self, run_dir: Path) -> dict:
        """Schreibt das Manifest beim ersten Lauf; beim Resume wird das
        bestehende Manifest gegen die aktuelle Config geprueft und bei Drift in
        eingefrorenen Feldern abgebrochen (E6)."""
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = run_dir / "manifest.json"
        current = self.build_manifest(run_dir.name)

        if not manifest_path.exists():
            manifest_path.write_text(
                json.dumps(current, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            return current

        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        drifted = [
            field
            for field in self._MANIFEST_FROZEN_FIELDS
            if existing.get(field) != current.get(field)
        ]
        if drifted:
            raise RuntimeError(
                f"Run config drifted from frozen manifest {manifest_path} in: {', '.join(drifted)}"
            )
        return existing


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


# Feld-Namen, deren Werte als Secret behandelt und redigiert werden.
_SECRET_MARKERS = (
    "authorization", "api_key", "apikey", "api-key", "x-api-key",
    "token", "bearer", "secret", "password", "credential",
)
# Keys, die komplett rausfliegen (Header-Bloecke landen ohnehin nicht im Body,
# werden aber defensiv entfernt).
_DROP_KEYS = ("headers", "header")
_REDACTED = "[REDACTED]"


def _is_secret_key(key: str) -> bool:
    lowered = str(key).lower()
    return any(marker in lowered for marker in _SECRET_MARKERS)


def _redact_payload(value):
    """Rekursive Redaction fuer die JSONL-Ablage: typische Secret-Felder
    (Authorization/api_key/x-api-key/token/bearer/...) und ganze Header-Bloecke
    werden durch ``[REDACTED]`` ersetzt. Die produktiven Payloads tragen zwar
    keine Secrets im Body (Keys stehen in Headern/Params), aber die Redaction
    stellt das auch bei kuenftigen Aenderungen sicher (Pflicht, kein No-op)."""
    if isinstance(value, dict):
        out = {}
        for key, sub in value.items():
            if str(key).lower() in _DROP_KEYS or _is_secret_key(key):
                out[key] = _REDACTED
            else:
                out[key] = _redact_payload(sub)
        return out
    if isinstance(value, (list, tuple)):
        return [_redact_payload(item) for item in value]
    return value


def _default_consensus_fn(api_keys: dict, consensus_model: str):
    """Standard-Consensus: produktive query_consensus-Logik (mit Modellnamen, E5)."""
    from app.services.llm.consensus_engine import query_consensus

    def _fn(question: str, answers: dict, model_sources=None) -> str:
        return query_consensus(
            question,
            answers.get("openai", ""),
            answers.get("mistral", ""),
            answers.get("anthropic", ""),
            answers.get("gemini", ""),
            answers.get("deepseek", ""),
            answers.get("grok", ""),
            excluded_models=[],
            consensus_model=consensus_model,
            api_keys=api_keys,
            model_sources=model_sources,
        )

    return _fn


def _consensus_api_model(consensus_model: str) -> str:
    """Loest das Consensus-Modell auf seinen api_model auf (fuer Pricing)."""
    model_config = cfg.get_model_config(consensus_model)
    return model_config.api_model if model_config else consensus_model
