"""Firestore-backed Benchmark-Reports fuer das Admin-Dashboard.

Die Benchmark-Rohartefakte bleiben lokal unter ``data/benchmark/runs``. Online
persistiert wird nur der kompakte Dashboard-Snapshot: Manifest, Results, Audits
und die abgeleitete Fragenmatrix ohne Prompts, Rohantworten oder Payloads.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

from benchmark import report_reader

COLLECTION = "benchmark_runs"
STORAGE_VERSION = 1


def _collection():
    # Lazy import, damit reine Benchmark-Reader-Tests weiterhin ohne Firebase-
    # Admin-Importgraph laufen koennen.
    from app.core.security import db_firestore

    return db_firestore.collection(COLLECTION)


def _is_safe_run_id(run_id: str) -> bool:
    name = str(run_id or "").strip()
    return bool(name and name == Path(name).name and name not in (".", ".."))


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _summary_from_report(report: dict) -> dict:
    manifest = report.get("manifest") or {}
    results = report.get("results") or {}
    return {
        "run_id": report.get("run_id"),
        "sample_role": manifest.get("sample_role"),
        "label_mode": manifest.get("label_mode"),
        "consensus_model": manifest.get("consensus_model"),
        "created": manifest.get("created"),
        "n_questions": results.get("n_questions"),
        "n_disagreement": results.get("n_disagreement"),
        "total_cost_usd": (results.get("totals") or {}).get("cost_usd"),
        "has_results": bool(results),
        "has_audits": bool(report.get("audits")),
        "source": "firestore",
    }


def _doc_to_report(doc) -> dict | None:
    if not getattr(doc, "exists", False):
        return None
    data = doc.to_dict() or {}
    report = data.get("report")
    if not isinstance(report, dict):
        return None
    return report


def list_published_runs() -> list[dict]:
    """Listet in Firestore publizierte Benchmark-Reports."""
    runs: list[dict] = []
    for doc in _collection().stream():
        report = _doc_to_report(doc)
        if not report:
            continue
        summary = _summary_from_report(report)
        summary["published_at"] = (doc.to_dict() or {}).get("published_at")
        runs.append(summary)
    runs.sort(key=lambda r: (r.get("created") or "", r.get("published_at") or ""), reverse=True)
    return runs


def get_published_run(run_id: str) -> dict | None:
    """Liest einen publizierten Run-Snapshot aus Firestore."""
    if not _is_safe_run_id(run_id):
        return None
    return _doc_to_report(_collection().document(str(run_id)).get())


def publish_run_dir(run_dir: Path) -> dict:
    """Publiziert den kompakten Dashboard-Snapshot eines lokalen Run-Ordners."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if not _is_safe_run_id(run_dir.name):
        raise ValueError(f"Unsafe run_id: {run_dir.name}")
    report = report_reader.build_run_report(run_dir)
    if not report.get("manifest"):
        raise ValueError(f"Run has no manifest.json: {run_dir}")
    if not report.get("results"):
        raise ValueError(f"Run has no results.json: {run_dir}")

    doc = {
        "run_id": report["run_id"],
        "storage_version": STORAGE_VERSION,
        "created": (report.get("manifest") or {}).get("created"),
        "published_at": _utc_now_iso(),
        "report": report,
    }
    _collection().document(report["run_id"]).set(doc)
    return _summary_from_report(report)


def list_runs_with_disk_fallback() -> list[dict]:
    """Mergt Firestore-Reports mit lokalen Runs; Firestore gewinnt bei Duplikaten."""
    merged: dict[str, dict] = {}
    for run in report_reader.list_runs():
        item = dict(run)
        item["source"] = "disk"
        merged[item["run_id"]] = item
    for run in list_published_runs():
        merged[run["run_id"]] = run
    runs = list(merged.values())
    runs.sort(key=lambda r: (r.get("created") or ""), reverse=True)
    return runs


def get_run_with_disk_fallback(run_id: str) -> dict | None:
    """Liest zuerst Firestore, sonst lokale Disk."""
    published = get_published_run(run_id)
    if published is not None:
        return published
    return report_reader.get_run(run_id)
