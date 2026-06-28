"""CLI (Phase 2.5a): Argumente, Run-ID/Run-Verzeichnis, Manifest, Live-Preflight.
Kein HTTP, keine echten Keys; Transport-Aufruf wuerde den Test sofort fehlschlagen
lassen."""

import pytest

from app.services.llm import credentials
from app.services import benchmark_reports
from benchmark import __main__ as cli
from benchmark import config
from benchmark import report_reader
from benchmark import runner as runner_mod

RECORDS = [
    {"question_id": 754, "question": "Q?", "options": ["a", "b", "c", "d"],
     "answer": "B", "answer_index": 1, "category": "math"}
]


@pytest.fixture
def patched(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(cli, "get_records", lambda args: list(RECORDS))

    def boom(*args, **kwargs):
        raise AssertionError("transport HTTP call attempted in a no-HTTP test")

    monkeypatch.setattr(runner_mod.transport, "execute", boom)
    return tmp_path


def test_parse_args_supports_new_flags():
    args = cli._parse_args(["--pilot", "--run-id", "r1", "--budget", "5", "--live"])
    assert args.pilot and args.live and args.run_id == "r1" and args.budget == 5.0
    smoke = cli._parse_args(["--smoke", "--run-id", "s1", "--budget", "1", "--live"])
    assert smoke.smoke and smoke.live and smoke.run_id == "s1" and smoke.budget == 1.0


def test_sample_flags_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        cli._parse_args(["--smoke", "--pilot"])
    with pytest.raises(SystemExit):
        cli._parse_args(["--smoke", "--final"])


def test_resolve_run_id_precedence():
    assert cli.resolve_run_id(cli._parse_args(["--resume", "abc"])) == "abc"
    assert cli.resolve_run_id(cli._parse_args(["--run-id", "xyz"])) == "xyz"
    assert cli.resolve_run_id(cli._parse_args(["--smoke"])) == "smoke"
    assert cli.resolve_run_id(cli._parse_args(["--pilot"])) == "pilot"
    assert cli.resolve_run_id(cli._parse_args([])) == "final"


def test_dry_run_creates_run_dir_and_manifest_without_http(patched, capsys):
    rc = cli.main(["--pilot", "--dry-run"])
    assert rc == 0
    run_dir = config.RUNS_DIR / "pilot"
    assert (run_dir / "manifest.json").exists()
    assert not (run_dir / "calls.jsonl").exists()
    assert "Dry-Run" in capsys.readouterr().out


def test_run_id_flag_controls_directory(patched):
    cli.main(["--pilot", "--run-id", "custom_run", "--dry-run"])
    assert (config.RUNS_DIR / "custom_run" / "manifest.json").exists()


def test_publish_run_publishes_existing_directory_without_pin_check(patched, monkeypatch, capsys):
    run_dir = config.RUNS_DIR / "pilot_v1"
    run_dir.mkdir(parents=True)
    published = {}

    def fake_publish(path):
        published["path"] = path
        return {"run_id": path.name, "n_questions": 5}

    def boom():
        raise AssertionError("pin check should not run for publish-only mode")

    monkeypatch.setattr(config, "assert_pins_match_config", boom)
    monkeypatch.setattr(benchmark_reports, "publish_run_dir", fake_publish)

    rc = cli.main(["--publish-run", "pilot_v1"])

    assert rc == 0
    assert published["path"] == run_dir
    assert "Published pilot_v1" in capsys.readouterr().out


def test_publish_all_uses_finished_runs_only(patched, monkeypatch, capsys):
    old_runs_dir = report_reader.RUNS_DIR
    report_reader.RUNS_DIR = config.RUNS_DIR
    try:
        finished = config.RUNS_DIR / "finished"
        unfinished = config.RUNS_DIR / "unfinished"
        finished.mkdir(parents=True)
        unfinished.mkdir(parents=True)
        (finished / "manifest.json").write_text('{"created":"2026-06-28T10:00:00+00:00"}')
        (finished / "results.json").write_text('{"n_questions":1}')
        (unfinished / "manifest.json").write_text('{"created":"2026-06-28T11:00:00+00:00"}')
        seen = []

        def fake_publish(path):
            seen.append(path.name)
            return {"run_id": path.name, "n_questions": 1}

        monkeypatch.setattr(benchmark_reports, "publish_run_dir", fake_publish)

        rc = cli.main(["--publish-all"])

        assert rc == 0
        assert seen == ["finished"]
        assert "Published runs: 1" in capsys.readouterr().out
    finally:
        report_reader.RUNS_DIR = old_runs_dir


def test_smoke_dry_run_uses_own_directory_and_manifest(patched, capsys):
    rc = cli.main(["--smoke", "--dry-run"])
    assert rc == 0
    run_dir = config.RUNS_DIR / "smoke"
    assert (run_dir / "manifest.json").exists()
    assert not (run_dir / "calls.jsonl").exists()
    manifest = (run_dir / "manifest.json").read_text(encoding="utf-8")
    assert '"sample_role": "smoke"' in manifest
    assert '"sample_manifest": "mmlu_pro_smoke_v1.json"' in manifest
    assert "smoke (1 questions)" in capsys.readouterr().out


def test_smoke_live_requires_budget(patched, capsys):
    rc = cli.main(["--smoke", "--live"])
    assert rc == 2
    assert "requires an explicit --budget" in capsys.readouterr().err


def test_pilot_live_requires_budget(patched, capsys):
    rc = cli.main(["--pilot", "--live"])
    assert rc == 2
    assert "requires an explicit --budget" in capsys.readouterr().err


def test_smoke_rejects_limit(patched, capsys):
    rc = cli.main(["--smoke", "--limit", "1", "--dry-run"])
    assert rc == 2
    assert "--limit is not allowed" in capsys.readouterr().err


def test_live_aborts_on_missing_credentials(patched, monkeypatch, capsys):
    monkeypatch.setattr(credentials, "resolve_developer_api_keys",
                        lambda providers=None: {p: None for p in cli.REQUIRED_PROVIDERS})
    monkeypatch.setattr(credentials, "gemini_adc_available", lambda: False)
    rc = cli.main(["--pilot", "--live", "--budget", "5"])
    assert rc == 2
    assert "missing credentials" in capsys.readouterr().err
    assert not (config.RUNS_DIR / "pilot" / "calls.jsonl").exists()


def test_final_live_preflight_passes_but_is_gated_no_http(patched, monkeypatch, capsys):
    monkeypatch.setattr(credentials, "resolve_developer_api_keys",
                        lambda providers=None: {p: "k" for p in cli.REQUIRED_PROVIDERS})
    rc = cli.main(["--final", "--live", "--budget", "5"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Preflight" in out
    assert "GATED" in out
    run_dir = config.RUNS_DIR / "final"
    assert (run_dir / "manifest.json").exists()
    # Gate -> keine Ausfuehrung, kein calls.jsonl
    assert not (run_dir / "calls.jsonl").exists()


def test_pilot_live_executes_when_pilot_gate_enabled(patched, monkeypatch, capsys):
    monkeypatch.setattr(credentials, "resolve_developer_api_keys",
                        lambda providers=None: {p: "k" for p in cli.REQUIRED_PROVIDERS})
    called = {"pilot": False}

    def fake_run_pilot(self, records, **kwargs):
        called["pilot"] = True
        return (
            type("Result", (), {
                "cells_written": 40, "cells_skipped": 0, "cells_failed": 0,
                "spent_usd": 1.2345, "stopped": False,
            })(),
            {"option_permutation": {"enabled": True}},
            {"n_questions": 5, "n_disagreement": 2},
        )

    monkeypatch.setattr(runner_mod.BenchmarkRunner, "run_pilot", fake_run_pilot)
    rc = cli.main(["--pilot", "--live", "--budget", "5"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Preflight" in out
    assert "Run finished" in out
    assert called["pilot"]


def test_smoke_live_executes_when_smoke_gate_enabled(patched, monkeypatch, capsys):
    monkeypatch.setattr(credentials, "resolve_developer_api_keys",
                        lambda providers=None: {p: "k" for p in cli.REQUIRED_PROVIDERS})
    called = {"smoke": False}

    def fake_run_smoke(self, records, **kwargs):
        called["smoke"] = True
        return (
            type("Result", (), {
                "cells_written": 8, "cells_skipped": 0, "cells_failed": 0,
                "spent_usd": 0.1234, "stopped": False,
            })(),
            {"option_permutation": {"enabled": False}},
            {"n_questions": 1, "n_disagreement": 0},
        )

    monkeypatch.setattr(runner_mod.BenchmarkRunner, "run_smoke", fake_run_smoke)
    rc = cli.main(["--smoke", "--live", "--budget", "5"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Preflight" in out
    assert "Run finished" in out
    assert called["smoke"]
    run_dir = config.RUNS_DIR / "smoke"
    assert (run_dir / "manifest.json").exists()


def test_final_preview_requires_budget(patched, capsys):
    rc = cli.main(["--final", "--limit", "10", "--live"])
    assert rc == 2
    assert "requires an explicit --budget" in capsys.readouterr().err


def test_final_limit_over_preview_cap_is_rejected(patched, capsys):
    rc = cli.main(["--final", "--limit", "50", "--live", "--budget", "8"])
    assert rc == 2
    assert "capped at" in capsys.readouterr().err


def test_full_final_without_limit_stays_gated(patched, monkeypatch, capsys):
    # Der volle Final-Run (ohne --limit) bleibt durch LIVE_EXECUTION_ENABLED gesperrt.
    monkeypatch.setattr(credentials, "resolve_developer_api_keys",
                        lambda providers=None: {p: "k" for p in cli.REQUIRED_PROVIDERS})
    rc = cli.main(["--final", "--live", "--budget", "8"])
    assert rc == 0
    assert "GATED" in capsys.readouterr().out
    assert not (config.RUNS_DIR / "final" / "calls.jsonl").exists()


def test_final_preview_executes_when_preview_gate_enabled(patched, monkeypatch, capsys):
    monkeypatch.setattr(credentials, "resolve_developer_api_keys",
                        lambda providers=None: {p: "k" for p in cli.REQUIRED_PROVIDERS})
    called = {"run": False}

    def fake_run_pilot(self, records, **kwargs):
        called["run"] = True
        return (
            type("Result", (), {
                "cells_written": 80, "cells_skipped": 0, "cells_failed": 0,
                "spent_usd": 2.5, "stopped": False,
            })(),
            {"option_permutation": {"enabled": True}},
            {"n_questions": 10, "n_disagreement": 3},
        )

    monkeypatch.setattr(runner_mod.BenchmarkRunner, "run_pilot", fake_run_pilot)
    rc = cli.main(["--final", "--limit", "10", "--live", "--budget", "8", "--run-id", "preview_x"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Run finished" in out
    assert called["run"]


def test_live_execution_stays_gated_constant():
    # Sicherheitsnetz: voller Final-Run bleibt hart gesperrt; Smoke/Pilot/Preview separat.
    assert cli.LIVE_EXECUTION_ENABLED is False
    assert cli.SMOKE_LIVE_EXECUTION_ENABLED is True
    assert cli.PILOT_LIVE_EXECUTION_ENABLED is True
    assert cli.PREVIEW_LIVE_EXECUTION_ENABLED is True
    assert cli.PREVIEW_MAX_QUESTIONS == 20
