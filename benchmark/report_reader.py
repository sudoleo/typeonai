"""Read-only Reporting-Schicht fuer die Benchmark-Runs (Admin-Visualisierung).

Bewusst **nur stdlib** (``json``/``pathlib``) – importiert **nicht** den Runner,
die Provider-Engines oder ``app.*``. Damit kann der Admin-Router die Run-Daten
lesen, ohne den schweren LLM-Importgraphen zu ziehen, und der Lesepfad bleibt
isoliert vom Ausfuehrungspfad.

Liest die bereits erzeugten Artefakte je Run aus ``data/benchmark/runs/<run_id>/``:
``manifest.json``, ``results.json``, ``audits.json`` und – fuer die Pro-Frage-
Matrix – ``calls.jsonl`` (dedupliziert: Erfolg gewinnt vor Fehlversuch).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "data" / "benchmark" / "runs"

NO_MAJORITY = "no_majority"


def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def _safe_run_dir(run_id: str) -> Path | None:
    """Validiert die Run-ID gegen Path-Traversal und liefert das Verzeichnis."""
    name = str(run_id or "").strip()
    if not name or name != Path(name).name or name in (".", ".."):
        return None
    run_dir = RUNS_DIR / name
    if not run_dir.is_dir():
        return None
    return run_dir


def list_runs() -> list[dict]:
    """Kompakte Uebersicht aller Runs (neueste zuerst nach ``created``)."""
    if not RUNS_DIR.is_dir():
        return []
    runs: list[dict] = []
    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        manifest = _read_json(run_dir / "manifest.json") or {}
        results = _read_json(run_dir / "results.json") or {}
        runs.append({
            "run_id": run_dir.name,
            "sample_role": manifest.get("sample_role"),
            "label_mode": manifest.get("label_mode"),
            "consensus_model": manifest.get("consensus_model"),
            "created": manifest.get("created"),
            "n_questions": results.get("n_questions"),
            "n_disagreement": results.get("n_disagreement"),
            "total_cost_usd": (results.get("totals") or {}).get("cost_usd"),
            "has_results": bool(results),
            "has_audits": (run_dir / "audits.json").exists(),
        })
    runs.sort(key=lambda r: (r.get("created") or ""), reverse=True)
    return runs


def _dedupe_calls(calls_path: Path) -> list[dict]:
    """Je Zellen-Key (question_id, role, provider) genau ein Record – Erfolg
    gewinnt vor Fehlversuch (gleiche Semantik wie ``results.dedupe_records``)."""
    if not calls_path.exists():
        return []
    index: dict[tuple, dict] = {}
    for line in calls_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        key = (record.get("question_id"), record.get("role"), record.get("provider"))
        slot = index.setdefault(key, {"success": None, "error": None})
        if record.get("error"):
            slot["error"] = record
        else:
            slot["success"] = record
    return [slot["success"] or slot["error"] for slot in index.values()
            if slot["success"] or slot["error"]]


def _majority(letters: list) -> str:
    valid = [l for l in letters if l]
    if not valid:
        return NO_MAJORITY
    ordered = Counter(valid).most_common()
    top = ordered[0][1]
    leaders = [l for l, c in ordered if c == top]
    return leaders[0] if len(leaders) == 1 else NO_MAJORITY


def build_question_matrix(run_dir: Path) -> list[dict]:
    """Pro-Frage-Matrix aus ``calls.jsonl``: je Frage die Buchstaben aller
    Modelle, Majority, Consensus und Synth – inkl. correct/abstain/error-Flags."""
    records = _dedupe_calls(run_dir / "calls.jsonl")
    by_q: dict = {}
    providers: list[str] = []
    for rec in records:
        qid = rec.get("question_id")
        slot = by_q.setdefault(qid, {
            "question_id": qid,
            "category": rec.get("category"),
            "ground_truth": rec.get("ground_truth"),
            "models": {}, "consensus": None, "synth_alone": None,
        })
        cell = {
            "letter": rec.get("extracted_letter"),
            "correct": bool(rec.get("correct")),
            "abstain": bool(rec.get("abstain")),
            "error": bool(rec.get("error")),
        }
        role = rec.get("role")
        if role == "model":
            provider = rec.get("provider")
            slot["models"][provider] = cell
            if provider not in providers:
                providers.append(provider)
        elif role == "consensus":
            slot["consensus"] = cell
        elif role == "synth_alone":
            slot["synth_alone"] = cell

    rows: list[dict] = []
    for qid in sorted(by_q):
        slot = by_q[qid]
        letters = [(slot["models"].get(p) or {}).get("letter") for p in providers]
        majority = _majority(letters)
        slot["majority"] = {
            "letter": majority,
            "correct": majority != NO_MAJORITY
            and majority == str(slot.get("ground_truth") or "").strip().upper(),
        }
        slot["disagreement"] = len(set(letters)) > 1
        rows.append(slot)
    return rows


def get_run(run_id: str) -> dict | None:
    """Vollstaendige Run-Detaildaten fuer die Visualisierung oder None."""
    run_dir = _safe_run_dir(run_id)
    if run_dir is None:
        return None
    return {
        "run_id": run_dir.name,
        "manifest": _read_json(run_dir / "manifest.json"),
        "results": _read_json(run_dir / "results.json"),
        "audits": _read_json(run_dir / "audits.json"),
        "questions": build_question_matrix(run_dir),
    }
