"""Automatische Auswertung: ``calls.jsonl`` -> ``results.json`` (Plan §7).

Vergleichsgroessen: jedes Einzelmodell, **Majority Vote**, **Consensus** und –
falls vorhanden – **Synthesizer-allein**. Ausgewiesen werden Accuracy gesamt und
auf der **Uneinigkeits-Teilmenge**, dazu ``no_majority``, abstain/unparseable,
Fehler-/Parse-Quote sowie Kosten und Latenzen.

Hier werden ``majority_vote`` und ``NO_MAJORITY`` tatsaechlich genutzt.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from benchmark import config
from benchmark.parse import NO_MAJORITY, grade, majority_vote
from benchmark.runner import index_existing, load_existing_records

MODEL_PROVIDERS = [model.provider for model in config.MODELS]


def _ratio(numerator: int, denominator: int):
    return round(numerator / denominator, 4) if denominator else None


def _wilson_ci(correct: int, n: int, z: float = 1.96):
    """Wilson-Score-95%-Konfidenzintervall fuer eine Trefferquote (Plan §4:
    Accuracy mit Konfidenzintervallen berichten). Bei kleinem n (z. B. 98 Fragen
    oder die Uneinigkeits-Teilmenge) ist die normale Naeherung zu optimistisch –
    Wilson ist dort robust. Gibt ``[lo, hi]`` (gerundet) oder None bei n=0."""
    if not n:
        return None
    phat = correct / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / denom
    return [round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4)]


def _latency_stats(latencies: list) -> dict:
    values = [lat for lat in latencies if isinstance(lat, (int, float))]
    if not values:
        return {"avg": None, "p50": None, "max": None, "n": 0}
    return {
        "avg": round(sum(values) / len(values), 2),
        "p50": round(statistics.median(values), 2),
        "max": round(max(values), 2),
        "n": len(values),
    }


def _cell_system_stats(cells: dict, qids: list[int], dis_set: set[int]) -> dict:
    """Kennzahlen fuer ein Zellen-basiertes System (Einzelmodell, Consensus,
    Synth-allein). ``cells`` mappt question_id -> cell-record (oder None)."""
    correct = abstain = error = parsed = attempted = 0
    correct_dis = 0
    cost = 0.0
    latencies: list = []

    for qid in qids:
        cell = cells.get(qid)
        if not cell:
            continue
        cost += float(cell.get("est_cost_usd") or 0.0)
        if cell.get("error"):
            error += 1
            continue
        attempted += 1
        if cell.get("latency_ms") is not None:
            latencies.append(cell["latency_ms"])
        if cell.get("extracted_letter"):
            parsed += 1
        else:
            abstain += 1
        if cell.get("correct"):
            correct += 1
            if qid in dis_set:
                correct_dis += 1

    return {
        "accuracy_overall": _ratio(correct, len(qids)),
        "accuracy_overall_ci": _wilson_ci(correct, len(qids)),
        "accuracy_disagreement": _ratio(correct_dis, len(dis_set)),
        "accuracy_disagreement_ci": _wilson_ci(correct_dis, len(dis_set)),
        "correct": correct,
        "total": len(qids),
        "attempted": attempted,
        "parsed": parsed,
        "abstain": abstain,
        "error": error,
        "parse_rate": _ratio(parsed, attempted),
        "error_rate": _ratio(error, len(qids)),
        "cost_usd": round(cost, 6),
        "latency_ms": _latency_stats(latencies),
    }


def aggregate(records: list[dict], *, consensus_model: str = config.CONSENSUS_MODEL) -> dict:
    """Aggregiert deduplizierte Zellen-Records zu Vergleichsgroessen."""
    by_q: dict[int, dict] = {}
    for record in records:
        slot = by_q.setdefault(
            record["question_id"],
            {"category": record.get("category"), "ground_truth": record.get("ground_truth"),
             "model": {}, "consensus": None, "synth_alone": None},
        )
        role = record.get("role")
        if role == "model":
            slot["model"][record.get("provider")] = record
        elif role == "consensus":
            slot["consensus"] = record
        elif role == "synth_alone":
            slot["synth_alone"] = record

    qids = sorted(by_q)
    n_questions = len(qids)

    # Majority-Vote + Uneinigkeits-Teilmenge je Frage.
    majority_by_q: dict[int, str] = {}
    disagreement_qids: list[int] = []
    for qid in qids:
        letters = [(by_q[qid]["model"].get(p) or {}).get("extracted_letter") for p in MODEL_PROVIDERS]
        majority_by_q[qid] = majority_vote(letters)
        if len(set(letters)) > 1:  # nicht alle 6 Letters identisch (None zaehlt mit)
            disagreement_qids.append(qid)
    dis_set = set(disagreement_qids)
    n_dis = len(dis_set)

    systems: dict[str, dict] = {}

    # Einzelmodelle.
    for provider in MODEL_PROVIDERS:
        cells = {qid: by_q[qid]["model"].get(provider) for qid in qids}
        systems[f"model:{provider}"] = _cell_system_stats(cells, qids, dis_set)

    # Majority Vote (abgeleitet, keine eigene Zelle/Kosten).
    maj_correct = maj_correct_dis = no_majority = 0
    for qid in qids:
        maj = majority_by_q[qid]
        if maj == NO_MAJORITY:
            no_majority += 1
        if grade(maj, by_q[qid]["ground_truth"]):
            maj_correct += 1
            if qid in dis_set:
                maj_correct_dis += 1
    systems["majority_vote"] = {
        "accuracy_overall": _ratio(maj_correct, n_questions),
        "accuracy_overall_ci": _wilson_ci(maj_correct, n_questions),
        "accuracy_disagreement": _ratio(maj_correct_dis, n_dis),
        "accuracy_disagreement_ci": _wilson_ci(maj_correct_dis, n_dis),
        "correct": maj_correct,
        "total": n_questions,
        "no_majority": no_majority,
    }

    # Consensus.
    systems["consensus"] = _cell_system_stats(
        {qid: by_q[qid]["consensus"] for qid in qids}, qids, dis_set
    )

    # Synthesizer-allein (nur falls vorhanden).
    if any(by_q[qid]["synth_alone"] for qid in qids):
        systems["synth_alone"] = _cell_system_stats(
            {qid: by_q[qid]["synth_alone"] for qid in qids}, qids, dis_set
        )

    # Gesamtsummen + Rollen-Aufschluesselung.
    by_role: dict[str, dict] = {}
    total_cost = 0.0
    total_errors = 0
    for record in records:
        total_cost += float(record.get("est_cost_usd") or 0.0)
        if record.get("error"):
            total_errors += 1
        slot = by_role.setdefault(record.get("role"), {"cells": 0, "cost_usd": 0.0, "errors": 0})
        slot["cells"] += 1
        slot["cost_usd"] += float(record.get("est_cost_usd") or 0.0)
        if record.get("error"):
            slot["errors"] += 1
    for slot in by_role.values():
        slot["cost_usd"] = round(slot["cost_usd"], 6)

    return {
        "consensus_model": consensus_model,
        "n_questions": n_questions,
        "n_disagreement": n_dis,
        "disagreement_question_ids": disagreement_qids,
        "systems": systems,
        "totals": {
            "cells": len(records),
            "cost_usd": round(total_cost, 6),
            "errors": total_errors,
            "by_role": by_role,
        },
    }


def dedupe_records(run_dir: Path) -> list[dict]:
    """Liest ``calls.jsonl`` und liefert je Zellen-Key genau einen Record
    (erfolgreicher bevorzugt, sonst letzter Fehlversuch) – verhindert
    Doppelzaehlung bei Resume-Retries."""
    index = index_existing(load_existing_records(Path(run_dir) / "calls.jsonl"))
    deduped: list[dict] = []
    for slot in index.values():
        chosen = slot["success"] or (slot["errors"][-1] if slot["errors"] else None)
        if chosen:
            deduped.append(chosen)
    return deduped


def write_results(run_dir: Path, *, consensus_model: str = config.CONSENSUS_MODEL) -> dict:
    """Erzeugt ``results.json`` aus ``calls.jsonl`` und gibt das Summary zurueck."""
    run_dir = Path(run_dir)
    summary = aggregate(dedupe_records(run_dir), consensus_model=consensus_model)
    summary["run_id"] = run_dir.name
    (run_dir / "results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary
