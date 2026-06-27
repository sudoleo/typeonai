"""CLI (Phase 2.5a): Argumente, Run-ID/Run-Verzeichnis, Manifest, Live-Preflight.
Kein HTTP, keine echten Keys; Transport-Aufruf wuerde den Test sofort fehlschlagen
lassen."""

import pytest

from app.services.llm import credentials
from benchmark import __main__ as cli
from benchmark import config
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


def test_smoke_rejects_limit(patched, capsys):
    rc = cli.main(["--smoke", "--limit", "1", "--dry-run"])
    assert rc == 2
    assert "--limit is not allowed" in capsys.readouterr().err


def test_live_aborts_on_missing_credentials(patched, monkeypatch, capsys):
    monkeypatch.setattr(credentials, "resolve_developer_api_keys",
                        lambda providers=None: {p: None for p in cli.REQUIRED_PROVIDERS})
    monkeypatch.setattr(credentials, "gemini_adc_available", lambda: False)
    rc = cli.main(["--pilot", "--live"])
    assert rc == 2
    assert "missing credentials" in capsys.readouterr().err
    assert not (config.RUNS_DIR / "pilot" / "calls.jsonl").exists()


def test_live_preflight_passes_but_is_gated_no_http(patched, monkeypatch, capsys):
    monkeypatch.setattr(credentials, "resolve_developer_api_keys",
                        lambda providers=None: {p: "k" for p in cli.REQUIRED_PROVIDERS})
    rc = cli.main(["--pilot", "--live", "--budget", "5"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Preflight" in out
    assert "GATED" in out
    run_dir = config.RUNS_DIR / "pilot"
    assert (run_dir / "manifest.json").exists()
    # Gate -> keine Ausfuehrung, kein calls.jsonl
    assert not (run_dir / "calls.jsonl").exists()


def test_smoke_live_preflight_passes_but_is_gated_no_http(patched, monkeypatch, capsys):
    monkeypatch.setattr(credentials, "resolve_developer_api_keys",
                        lambda providers=None: {p: "k" for p in cli.REQUIRED_PROVIDERS})
    rc = cli.main(["--smoke", "--live", "--budget", "5"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Preflight" in out
    assert "GATED" in out
    run_dir = config.RUNS_DIR / "smoke"
    assert (run_dir / "manifest.json").exists()
    assert not (run_dir / "calls.jsonl").exists()


def test_live_execution_stays_gated_constant():
    # Sicherheitsnetz: in Phase 2.5a bleibt der Live-Run hart gesperrt.
    assert cli.LIVE_EXECUTION_ENABLED is False
