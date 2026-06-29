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
import random
from dataclasses import dataclass, field
from pathlib import Path

import app.core.config as cfg
from app.services.llm.engines import build_provider_payload

from benchmark import audit, config, cost, transport
from benchmark.parse import extract_letter, grade
from benchmark.prompt import (
    LETTERS,
    build_anonymized_consensus_question,
    build_consensus_question,
    build_mc_question,
)

logger = logging.getLogger(__name__)

# Benchmark-eigenes Consensus-Output-Limit (NICHT der Produktions-Default 8192).
CONSENSUS_MAX_TOKENS = int(config.CONSENSUS_OUTPUT_TOKEN_LIMIT)


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
        sample_role: str = "benchmark",
        sample_manifest: str | None = None,
    ):
        self.models = list(models)
        self.consensus_model = consensus_model
        self.system_prompt = system_prompt
        self.output_tokens = int(output_tokens)
        self.label_mode = label_mode
        self.include_synth_alone = include_synth_alone
        self.sample_role = sample_role
        self.sample_manifest = sample_manifest

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
                    est = estimate_model_cell(consensus_api_model, user_prompt, CONSENSUS_MAX_TOKENS)["est_cost_usd"]
                    if should_stop_for_budget(spent, est, budget):
                        self._mark_stopped(result, budget, f"synth_alone q{qid}")
                        return result
                    request_data = self.build_synth_request(user_prompt)
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

    def build_synth_request(self, user_prompt: str) -> dict:
        """Baut den Synthesizer-alone-Payload mit demselben Modellpin und den
        effektiven Output-/Temperatur-Settings wie der Consensus-Pfad."""
        model_config = cfg.get_model_config(self.consensus_model)
        if not model_config or not model_config.provider:
            raise ValueError(f"Consensus model is not resolvable: {self.consensus_model}")
        request_data = build_provider_payload(
            model_config.provider,
            question=user_prompt,
            system_prompt=self.system_prompt,
            model_override=self.consensus_model,
            max_output_tokens=CONSENSUS_MAX_TOKENS,
            benchmark_mode=True,
        )
        # query_consensus nutzt Gemini Pro ohne explizite Temperatur; Synth-alone
        # muss fuer den Benchmark dieselben effektiven Settings dokumentieren/nutzen.
        if model_config.provider == "gemini":
            request_data["payload"].get("generationConfig", {}).pop("temperature", None)
        return request_data

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
            text = consensus_fn(
                question=build_consensus_question(record["question"], record["options"]),
                answers=answers,
                model_sources=None,
            )
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
            "sample_role": self.sample_role,
            "sample_manifest": self.sample_manifest,
            "label_mode": self.label_mode,
            "include_synth_alone": self.include_synth_alone,
            "consensus_model": self.consensus_model,
            "temperature": config.TEMPERATURE,
            "output_token_limit": self.output_tokens,
            "consensus_output_token_limit": CONSENSUS_MAX_TOKENS,
            "system_prompt": self.system_prompt,
            "consensus_prompt_template": _consensus_prompt_template(),
            "models": self._model_manifest_rows(),
            "consensus": self._consensus_manifest_row(),
            "synth_alone": self._synth_manifest_row(),
            "pricing_usd_per_1m": config.PRICING_USD_PER_1M,
        }

    # Felder, die beim Resume mit dem bestehenden Manifest uebereinstimmen
    # muessen (eingefrorene Config, E6).
    _MANIFEST_FROZEN_FIELDS = (
        "sample_role", "sample_manifest", "label_mode", "include_synth_alone",
        "consensus_model", "temperature", "output_token_limit",
        "consensus_output_token_limit", "system_prompt", "consensus_prompt_template",
        "models", "consensus", "synth_alone",
    )

    def _model_manifest_rows(self) -> list[dict]:
        rows = []
        for model in self.models:
            request_data = self.build_model_request(model, "manifest dry run")
            rows.append(_manifest_row_for_request(model, request_data, self.output_tokens))
        return rows

    def _consensus_manifest_row(self) -> dict:
        request_data = self.build_synth_request("manifest dry run")
        return _manifest_row_for_request(
            config.BenchmarkModel(
                request_data["provider"],
                self.consensus_model,
                request_data["api_model"],
                "provider_default",
            ),
            request_data,
            CONSENSUS_MAX_TOKENS,
        )

    def _synth_manifest_row(self) -> dict:
        row = self._consensus_manifest_row()
        row["matches_consensus"] = True
        return row

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

    # --- E4-Audits (im Pilot-Flow ausgefuehrt + gespeichert) ---------------

    def _stored_answers(self, index: dict, qid: int) -> dict[str, str]:
        """Bereits gespeicherte (erfolgreiche) Kandidatenantworten je Provider
        aus ``calls.jsonl`` – Basis fuer beide Audits, ohne erneute Calls."""
        answers: dict[str, str] = {}
        for model in self.models:
            stored = index.get(cell_key(qid, "model", model.provider), {}).get("success")
            if stored:
                answers[model.provider] = stored.get("parsed_text") or ""
        return answers

    def audit_option_permutation(
        self,
        records: list[dict],
        run_dir: Path,
        *,
        transport_execute=transport.execute,
        api_keys: dict | None = None,
        rng: random.Random | None = None,
        subset_size: int = 2,
    ) -> dict:
        """Positions-Bias-Audit (E4-2): mischt auf einem kleinen Subset die
        Optionen und prueft, dass die extrahierte Antwort auf **denselben
        Options-Inhalt** zeigt (nicht auf eine feste Buchstabenposition)."""
        run_dir = Path(run_dir)
        api_keys = api_keys or {}
        rng = rng or random.Random(config.PILOT_SEED)
        index = index_existing(load_existing_records(run_dir / "calls.jsonl"))
        subset = records[: max(0, subset_size)]

        checks: list[dict] = []
        for record in subset:
            qid = record["question_id"]
            new_options, _new_idx, _new_letter, order = audit.permute_options(
                record["options"], record["answer_index"], rng
            )
            user_prompt = build_mc_question(record["question"], new_options)
            for model in self.models:
                stored = index.get(cell_key(qid, "model", model.provider), {}).get("success")
                original_letter = stored.get("extracted_letter") if stored else None

                request_data = build_provider_payload(
                    model.provider,
                    question=user_prompt,
                    system_prompt=self.system_prompt,
                    model_override=model.internal_id,
                    max_output_tokens=self.output_tokens,
                    benchmark_mode=True,
                )
                audit.assert_no_web_tools(request_data["payload"], context=f"perm:{model.provider}:{qid}")
                api_key = api_keys.get(config.PROVIDER_API_KEY_NAME[model.provider])
                outcome = transport_execute(request_data, api_key)
                permuted_letter = (
                    None if outcome.get("error") else extract_letter(outcome.get("text"), options=new_options)
                )
                checks.append({
                    "question_id": qid,
                    "provider": model.provider,
                    "original_letter": original_letter,
                    "permuted_letter": permuted_letter,
                    "consistent": _permutation_consistent(original_letter, permuted_letter, order),
                })

        conclusive = [c for c in checks if c["consistent"] is not None]
        return {
            "subset_question_ids": [r["question_id"] for r in subset],
            "checks": checks,
            "consistent": sum(1 for c in conclusive if c["consistent"]),
            "conclusive": len(conclusive),
            "total": len(checks),
        }

    def audit_consensus_order(
        self,
        records: list[dict],
        run_dir: Path,
        *,
        consensus_fn=None,
        api_keys: dict | None = None,
        rng: random.Random | None = None,
    ) -> dict:
        """Reihenfolge-Invarianz der Consensus-Synthese (E4-3): rechnet den
        Consensus auf identischen, **bereits gespeicherten** Kandidatenantworten
        in drei Reihenfolgen neu (normal/umgekehrt/gemischt) – **keine** erneuten
        Kandidaten-Calls – und protokolliert die Stabilitaet des extrahierten
        Buchstabens.

        Hinweis: Der produktive Consensus-Prompt listet die Experten in fester
        Label-Reihenfolge; bei reiner Reihenfolge-Variation der Antworten misst
        dieser Audit daher zunaechst die Synthese-Stabilitaet/Nichtdeterminismus.
        Echte Label-Reihenfolge-Permutation haengt am aufgeschobenen
        geordneten/anonymisierten Prompt-Builder (E5).
        """
        run_dir = Path(run_dir)
        api_keys = api_keys or {}
        rng = rng or random.Random(config.FINAL_SEED)
        if consensus_fn is None:
            consensus_fn = _default_consensus_fn(api_keys, self.consensus_model)
        index = index_existing(load_existing_records(run_dir / "calls.jsonl"))

        questions: list[dict] = []
        for record in records:
            answers = self._stored_answers(index, record["question_id"])
            if len(answers) < 2:
                continue
            providers = [m.provider for m in self.models if m.provider in answers]

            def recompute(order, _answers=answers, _record=record):
                ordered = {provider: _answers[provider] for provider in order}
                text = consensus_fn(
                    question=build_consensus_question(_record["question"], _record["options"]),
                    answers=ordered,
                    model_sources=None,
                )
                return extract_letter(text, options=_record["options"])

            outcome = audit.run_consensus_order_audit(providers, recompute, rng)
            questions.append({
                "question_id": record["question_id"],
                "letters": outcome["letters"],
                "stable": outcome["stable"],
            })

        return {
            "questions": questions,
            "stable": sum(1 for q in questions if q["stable"]),
            "total": len(questions),
        }

    def audit_anonymized_consensus(
        self,
        records: list[dict],
        run_dir: Path,
        *,
        transport_execute=transport.execute,
        api_keys: dict | None = None,
        rng: random.Random | None = None,
    ) -> dict:
        """Modellnamen-Bias-Audit (E5): berechnet den Consensus auf denselben
        gespeicherten Kandidatenantworten erneut mit anonymen ``Response A-F``-
        Labels. Es gibt keine erneuten Kandidaten-Calls und ``audits.json``
        enthaelt keine Rohantworten."""
        run_dir = Path(run_dir)
        api_keys = api_keys or {}
        rng = rng or random.Random(config.FINAL_SEED)
        index = index_existing(load_existing_records(run_dir / "calls.jsonl"))
        consensus_api_model = _consensus_api_model(self.consensus_model)
        model_config = cfg.get_model_config(self.consensus_model)
        if not model_config or not model_config.provider:
            raise ValueError(f"Consensus model is not resolvable: {self.consensus_model}")

        questions: list[dict] = []
        for record in records:
            qid = record["question_id"]
            answers = self._stored_answers(index, qid)
            if len(answers) < 2:
                continue
            providers = [m.provider for m in self.models if m.provider in answers]
            rng.shuffle(providers)
            anon_map = {
                provider: f"Response {LETTERS[idx]}"
                for idx, provider in enumerate(providers)
            }
            anonymous_answers = [
                (anon_map[provider], answers[provider])
                for provider in providers
            ]
            user_prompt = build_anonymized_consensus_question(
                record["question"], record["options"], anonymous_answers
            )
            request_data = build_provider_payload(
                model_config.provider,
                question=user_prompt,
                system_prompt=self.system_prompt,
                model_override=self.consensus_model,
                max_output_tokens=CONSENSUS_MAX_TOKENS,
                benchmark_mode=True,
            )
            if model_config.provider == "gemini":
                request_data["payload"].get("generationConfig", {}).pop("temperature", None)
            audit.assert_no_web_tools(request_data["payload"], context=f"anon_consensus:{qid}")
            api_key = api_keys.get(config.PROVIDER_API_KEY_NAME[request_data["provider"]])
            outcome = transport_execute(request_data, api_key)
            anon_letter = (
                None if outcome.get("error") else extract_letter(outcome.get("text"), options=record["options"])
            )
            named = index.get(cell_key(qid, "consensus", self.consensus_model), {}).get("success")
            named_letter = named.get("extracted_letter") if named else None
            questions.append({
                "question_id": qid,
                "anon_map": anon_map,
                "named_letter": named_letter,
                "anonymous_letter": anon_letter,
                "stable": named_letter is not None and anon_letter == named_letter,
                "error": outcome.get("error"),
                "est_cost_usd": round(float(cost.est_cost_usd(
                    consensus_api_model,
                    (outcome.get("usage") or {}).get("prompt", 0),
                    (outcome.get("usage") or {}).get("completion", 0),
                )), 8),
            })

        comparable = [q for q in questions if q["named_letter"] is not None and not q["error"]]
        return {
            "label_mode": "anon",
            "questions": questions,
            "stable": sum(1 for q in comparable if q["stable"]),
            "comparable": len(comparable),
            "total": len(questions),
            "cost_usd": round(sum(float(q.get("est_cost_usd") or 0.0) for q in questions), 6),
        }

    # --- Pilot-Orchestrierung (run + Audits + Auswertung) ------------------

    def run_pilot(
        self,
        records: list[dict],
        *,
        run_dir: Path,
        api_keys: dict | None = None,
        transport_execute=transport.execute,
        consensus_fn=None,
        budget: float | None = None,
        retry_failed: bool = False,
        permutation_subset: int = 2,
        rng: random.Random | None = None,
    ):
        """Voller Pilot-Flow: Zellen-Loop -> E4-Audits -> Auswertung.
        Schreibt ``calls.jsonl``, ``audits.json`` und ``results.json``.
        Injizierbarer Transport/Consensus -> ohne HTTP/Keys testbar.
        """
        run_dir = Path(run_dir)
        api_keys = api_keys or {}
        rng = rng or random.Random(config.FINAL_SEED)

        result = self.run(
            records,
            run_dir=run_dir,
            api_keys=api_keys,
            transport_execute=transport_execute,
            consensus_fn=consensus_fn,
            budget=budget,
            retry_failed=retry_failed,
        )

        audits = None
        summary = None
        if not result.stopped:
            audits = {
                "option_permutation": self.audit_option_permutation(
                    records, run_dir, transport_execute=transport_execute,
                    api_keys=api_keys, rng=rng, subset_size=permutation_subset,
                ),
                "consensus_order": self.audit_consensus_order(
                    records, run_dir, consensus_fn=consensus_fn, api_keys=api_keys, rng=rng,
                ),
                "consensus_anonymized": self.audit_anonymized_consensus(
                    records, run_dir, transport_execute=transport_execute,
                    api_keys=api_keys, rng=rng,
                ),
            }
            (run_dir / "audits.json").write_text(
                json.dumps(audits, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            from benchmark import results as results_mod

            summary = results_mod.write_results(run_dir, consensus_model=self.consensus_model)

        return result, audits, summary

    def run_smoke(
        self,
        records: list[dict],
        *,
        run_dir: Path,
        api_keys: dict | None = None,
        transport_execute=transport.execute,
        consensus_fn=None,
        budget: float | None = None,
        retry_failed: bool = False,
    ):
        """Dedizierter 1-Frage-Smoke: 6 Modellzellen + Consensus +
        Synthesizer-alone, keine E4-Zusatzaudits. Schreibt bei erfolgreichem
        Abschluss ``calls.jsonl``, ``audits.json`` und ``results.json``."""
        self.validate_smoke_setup(records)
        run_dir = Path(run_dir)
        result = self.run(
            records,
            run_dir=run_dir,
            api_keys=api_keys or {},
            transport_execute=transport_execute,
            consensus_fn=consensus_fn,
            budget=budget,
            retry_failed=retry_failed,
        )
        audits = None
        summary = None
        if not result.stopped:
            audits = {
                "option_permutation": {
                    "enabled": False,
                    "reason": "disabled_for_smoke",
                },
                "consensus_order": {
                    "enabled": False,
                    "reason": "disabled_for_smoke",
                },
                "consensus_anonymized": {
                    "enabled": False,
                    "reason": "disabled_for_smoke",
                },
            }
            (run_dir / "audits.json").write_text(
                json.dumps(audits, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            from benchmark import results as results_mod

            summary = results_mod.write_results(run_dir, consensus_model=self.consensus_model)
        return result, audits, summary

    def validate_smoke_setup(self, records: list[dict]) -> None:
        """Harte Smoke-Invarianten, bevor ein echter Smoke-Lauf starten darf."""
        if len(records) != 1:
            raise ValueError(f"Smoke requires exactly 1 question, got {len(records)}")
        if len(self.models) != 6 or len({model.provider for model in self.models}) != 6:
            raise ValueError("Smoke requires exactly six distinct candidate models")
        if not self.consensus_model:
            raise ValueError("Smoke requires exactly one consensus model")
        if not self.include_synth_alone:
            raise ValueError("Smoke requires synth_alone to be enabled")
        synth = self._synth_manifest_row()
        consensus = self._consensus_manifest_row()
        comparable_keys = (
            "provider", "internal_id", "resolved_api_model", "reasoning_settings",
            "temperature", "temperature_source", "output_token_limit",
        )
        if any(synth.get(key) != consensus.get(key) for key in comparable_keys):
            raise ValueError("Synthesizer-alone settings must match consensus settings")


def _manifest_row_for_request(
    model: config.BenchmarkModel,
    request_data: dict,
    output_token_limit: int,
) -> dict:
    payload = request_data["payload"]
    resolved_api_model = request_data["api_model"]
    return {
        "provider": model.provider,
        "internal_id": model.internal_id,
        "frozen_api_model": model.api_model,
        "resolved_api_model": resolved_api_model,
        "reasoning": model.reasoning,
        "reasoning_settings": _reasoning_settings(request_data["provider"], payload),
        "temperature": _temperature(payload),
        "temperature_source": "payload" if _temperature(payload) is not None else "provider_default",
        "output_token_limit": _output_token_limit(payload, output_token_limit),
        "alias_status": {
            "internal_alias": model.internal_id != resolved_api_model,
            "latest_alias": _is_latest_alias(model.internal_id) or _is_latest_alias(resolved_api_model),
            "preview": "preview" in model.internal_id.lower() or "preview" in resolved_api_model.lower(),
        },
        "is_low_reasoning": bool(request_data.get("is_low_reasoning")),
        "matches": resolved_api_model == model.api_model,
    }


def _reasoning_settings(provider: str, payload: dict):
    if provider in ("openai", "grok"):
        return payload.get("reasoning")
    if provider == "mistral":
        return {"reasoning_effort": payload.get("completion_args", {}).get("reasoning_effort")}
    if provider == "anthropic":
        settings = {}
        if "thinking" in payload:
            settings["thinking"] = payload["thinking"]
        if "output_config" in payload:
            settings["output_config"] = payload["output_config"]
        return settings or None
    if provider == "gemini":
        thinking = payload.get("generationConfig", {}).get("thinkingConfig")
        return {"thinkingConfig": thinking} if thinking else None
    return None


def _temperature(payload: dict):
    if "temperature" in payload:
        return payload.get("temperature")
    if "completion_args" in payload:
        return payload.get("completion_args", {}).get("temperature")
    if "generationConfig" in payload:
        return payload.get("generationConfig", {}).get("temperature")
    return None


def _output_token_limit(payload: dict, fallback: int) -> int:
    if "max_output_tokens" in payload:
        return int(payload["max_output_tokens"])
    if "max_tokens" in payload:
        return int(payload["max_tokens"])
    if "completion_args" in payload and "max_tokens" in payload["completion_args"]:
        return int(payload["completion_args"]["max_tokens"])
    if "generationConfig" in payload and "maxOutputTokens" in payload["generationConfig"]:
        return int(payload["generationConfig"]["maxOutputTokens"])
    return int(fallback)


def _is_latest_alias(model_id: str) -> bool:
    return str(model_id or "").endswith("-latest")


def _permutation_consistent(original_letter, permuted_letter, order: list[int]):
    """True/False ob die permutierte Antwort auf dieselbe Original-Option zeigt;
    None, wenn nicht entscheidbar (eine Seite ohne Buchstabe)."""
    if not original_letter or not permuted_letter:
        return None
    if original_letter not in LETTERS or permuted_letter not in LETTERS:
        return None
    original_index = LETTERS.index(original_letter)
    permuted_position = LETTERS.index(permuted_letter)
    if permuted_position >= len(order):
        return None
    return order[permuted_position] == original_index


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
    """Standard-Consensus: produktive query_consensus-Logik (mit Modellnamen, E5).

    ``query_consensus`` liest sein Output-Limit aus dem Modul-Global
    ``cfg.CONSENSUS_MAX_TOKENS`` (Default 8192) und nimmt keinen Parameter. Da der
    Benchmark in einem **eigenen Prozess** laeuft, spiegeln wir hier das
    Benchmark-Limit in dieses Global – die Produktion (anderer Prozess) bleibt
    unberuehrt. So bricht die Synthese auf langen Aufgaben nicht vorzeitig ab.
    """
    from app.services.llm.consensus_engine import query_consensus

    cfg.CONSENSUS_MAX_TOKENS = int(config.CONSENSUS_OUTPUT_TOKEN_LIMIT)

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


# Platzhalter fuer die pro Frage eingesetzten, dynamischen Teile des
# Consensus-Prompts (Frage + sechs Kandidatenantworten). Das uebrige Geruest
# (Synthese-Anweisungen) ist ueber alle Fragen/Runs konstant = der V0-Prompt.
_CONSENSUS_TEMPLATE_PLACEHOLDERS = (
    "{QUESTION}", "{ANSWER_OPENAI}", "{ANSWER_MISTRAL}", "{ANSWER_ANTHROPIC}",
    "{ANSWER_GEMINI}", "{ANSWER_DEEPSEEK}", "{ANSWER_GROK}",
)


def _consensus_prompt_template() -> str:
    """Der produktive (V0-)Consensus-Synthese-Prompt als Template – die pro Frage
    variablen Teile (Frage, sechs Kandidatenantworten) sind durch Platzhalter
    ersetzt. So wird der exakt verwendete Synthese-Prompt im Manifest fuer
    Transparenz/Reproduzierbarkeit festgehalten, ohne Rohantworten zu speichern.
    """
    from app.services.llm.consensus_engine import _build_consensus_prompt

    return _build_consensus_prompt(*_CONSENSUS_TEMPLATE_PLACEHOLDERS, [], model_sources=None)
