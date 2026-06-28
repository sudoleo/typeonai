import json
from pathlib import Path

from app.services import benchmark_reports
from benchmark import report_reader


class FakeDocSnapshot:
    def __init__(self, data=None):
        self._data = data
        self.exists = data is not None

    def to_dict(self):
        return self._data


class FakeDocRef:
    def __init__(self, store, doc_id):
        self.store = store
        self.doc_id = doc_id

    def get(self):
        return FakeDocSnapshot(self.store.get(self.doc_id))

    def set(self, data):
        self.store[self.doc_id] = data


class FakeCollection:
    def __init__(self):
        self.store = {}

    def document(self, doc_id):
        return FakeDocRef(self.store, doc_id)

    def stream(self):
        return [FakeDocSnapshot(data) for data in self.store.values()]


def _write_finished_run(root: Path, run_id: str, *, raw_marker: str = "SECRET_RAW_ANSWER"):
    run = root / run_id
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({
        "sample_role": "pilot",
        "label_mode": "names",
        "consensus_model": "gemini-x",
        "created": "2026-06-28T10:00:00+00:00",
        "models": [],
    }), encoding="utf-8")
    (run / "results.json").write_text(json.dumps({
        "n_questions": 1,
        "n_disagreement": 0,
        "systems": {"consensus": {"accuracy_overall": 1.0}},
        "totals": {"cost_usd": 0.12, "cells": 8, "errors": 0},
    }), encoding="utf-8")
    (run / "audits.json").write_text(json.dumps({
        "option_permutation": {"enabled": False},
    }), encoding="utf-8")
    (run / "calls.jsonl").write_text(json.dumps({
        "question_id": 1,
        "category": "math",
        "ground_truth": "C",
        "role": "model",
        "provider": "openai",
        "extracted_letter": "C",
        "correct": True,
        "abstain": False,
        "error": None,
        "user_prompt": "SECRET_RAW_PROMPT",
        "parsed_text": raw_marker,
        "request_payload": {"secret": "SECRET_RAW_PAYLOAD"},
    }) + "\n", encoding="utf-8")
    return run


def test_publish_run_dir_stores_compact_report_only(monkeypatch, tmp_path):
    fake = FakeCollection()
    monkeypatch.setattr(benchmark_reports, "_collection", lambda: fake)
    run = _write_finished_run(tmp_path, "pilot_v1")

    summary = benchmark_reports.publish_run_dir(run)

    assert summary["run_id"] == "pilot_v1"
    stored = fake.store["pilot_v1"]
    stored_json = json.dumps(stored)
    assert stored["storage_version"] == benchmark_reports.STORAGE_VERSION
    assert stored["report"]["questions"][0]["models"]["openai"]["letter"] == "C"
    assert "SECRET_RAW_ANSWER" not in stored_json
    assert "SECRET_RAW_PROMPT" not in stored_json
    assert "SECRET_RAW_PAYLOAD" not in stored_json


def test_firestore_runs_win_over_disk_duplicates(monkeypatch, tmp_path):
    fake = FakeCollection()
    monkeypatch.setattr(benchmark_reports, "_collection", lambda: fake)
    old_runs_dir = report_reader.RUNS_DIR
    report_reader.RUNS_DIR = tmp_path
    try:
        _write_finished_run(tmp_path, "pilot_v1")
        benchmark_reports.publish_run_dir(tmp_path / "pilot_v1")

        runs = benchmark_reports.list_runs_with_disk_fallback()

        assert len(runs) == 1
        assert runs[0]["run_id"] == "pilot_v1"
        assert runs[0]["source"] == "firestore"
    finally:
        report_reader.RUNS_DIR = old_runs_dir
