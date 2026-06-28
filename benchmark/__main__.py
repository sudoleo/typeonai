"""CLI fuer den Benchmark-Runner.

Sicherheits-Modell (Phase 2.5a – Pilot-Startfaehigkeit **ohne** Live-Run):
- Ohne ``--live`` passiert **nie** ein Provider-HTTP-Call: es laeuft die
  Dry-Run-Vorschau (Payloads, Tool-Audit, Kostenprojektion).
- Mit ``--live`` werden Credentials aller sechs Provider **vor** dem ersten Call
  validiert (fehlende -> klarer Abbruch mit Provider-Liste), Run-Verzeichnis +
  Manifest angelegt/geprueft und ein Preflight (Audit + Projektion) gefahren.
  Die eigentliche Ausfuehrung ist in dieser Phase **bewusst gesperrt**
  Pilot/Final bleiben mit ``LIVE_EXECUTION_ENABLED = False`` gesperrt; Smoke
  ist ueber ``SMOKE_LIVE_EXECUTION_ENABLED = True`` separat freigegeben.

Aufrufe:
  python -m benchmark --dry-run --smoke
  python -m benchmark --smoke --live --budget 5
  python -m benchmark --dry-run --pilot
  python -m benchmark --pilot --run-id pilot_v1 --budget 5
  python -m benchmark --pilot --live --budget 5
  python -m benchmark --final --limit 10 --live --budget 8  # begrenzter Preview
  python -m benchmark --resume pilot_v1 --live --budget 5  # Resume-Preflight
  python -m benchmark --publish-run pilot_v1                 # Dashboard-Snapshot
  python -m benchmark --publish-all                          # alle fertigen Runs

Der volle ``--final``-Lauf (ohne --limit) bleibt durch ``LIVE_EXECUTION_ENABLED``
gesperrt; ein begrenzter Final-Sample-Preview laeuft ueber den separaten
``PREVIEW_LIVE_EXECUTION_ENABLED``-Gate (max. ``PREVIEW_MAX_QUESTIONS`` Fragen).
"""

from __future__ import annotations

import argparse
import sys

from app.services.llm import credentials

from benchmark import config, dataset
from benchmark.runner import BenchmarkRunner

# Alle sechs Provider muessen vor einem Live-Lauf Credentials haben (Consensus
# nutzt Gemini, das bereits in der Liste steht).
REQUIRED_PROVIDERS = ["OpenAI", "Mistral", "Anthropic", "Gemini", "DeepSeek", "Grok"]

# Globaler Gate: der volle 98-Fragen-Final-Run bleibt gesperrt, bis er bewusst
# freigegeben wird.
LIVE_EXECUTION_ENABLED = False
# Separater Gate fuer den kontrollierten 1-Frage-Smoke.
SMOKE_LIVE_EXECUTION_ENABLED = True
# Separater Gate fuer den kontrollierten 5-Fragen-Pilot.
PILOT_LIVE_EXECUTION_ENABLED = True
# Separater Gate fuer einen begrenzten Final-Sample-PREVIEW (``--final --limit N``)
# OHNE den vollen Lauf zu oeffnen. Der ungebremste ``--final`` (ohne --limit oder
# ueber PREVIEW_MAX_QUESTIONS hinaus) braucht weiterhin LIVE_EXECUTION_ENABLED.
PREVIEW_LIVE_EXECUTION_ENABLED = True
PREVIEW_MAX_QUESTIONS = 20


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m benchmark")
    parser.add_argument("--dry-run", action="store_true", help="Payloads bauen, kein HTTP")
    sample = parser.add_mutually_exclusive_group()
    sample.add_argument("--smoke", action="store_true", help="dediziertes 1-Frage-Smoke-Sample")
    sample.add_argument("--pilot", action="store_true", help="Pilot-Sample statt Final-Sample")
    sample.add_argument("--final", action="store_true", help="Final-Sample explizit waehlen")
    parser.add_argument("--limit", type=int, default=None, help="auf N Fragen begrenzen")
    parser.add_argument("--budget", type=float, default=None, help="Budget-Cap in USD")
    parser.add_argument("--run-id", dest="run_id", default=None, help="explizite Run-ID")
    parser.add_argument("--resume", metavar="RUN_ID", default=None, help="Run fortsetzen")
    publish = parser.add_mutually_exclusive_group()
    publish.add_argument(
        "--publish-run",
        metavar="RUN_ID",
        default=None,
        help="bestehenden Run als kompakten Firestore-Dashboard-Snapshot publizieren",
    )
    publish.add_argument(
        "--publish-all",
        action="store_true",
        help="alle fertigen lokalen Runs als kompakte Firestore-Dashboard-Snapshots publizieren",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Live-Preflight (Credential-Check + Manifest); Ausfuehrung bleibt gesperrt",
    )
    return parser.parse_args(argv)


def resolve_run_id(args: argparse.Namespace) -> str:
    """Deterministische Run-ID: --resume > --run-id > sample-basierter Default."""
    if args.resume:
        return args.resume
    if args.run_id:
        return args.run_id
    if args.smoke:
        return "smoke"
    return "pilot" if args.pilot else "final"


def _load_records():
    try:
        df = dataset.load_dataframe()
    except ImportError as exc:
        print(
            "ERROR: Parquet-Engine fehlt (pyarrow/fastparquet). MMLU-Pro liegt nur als "
            "Parquet vor; bitte einen Parquet-Reader installieren.\n"
            f"Detail: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return dataset.records_from_dataframe(df)


def sample_role(args: argparse.Namespace) -> str:
    if args.smoke:
        return "smoke"
    if args.pilot:
        return "pilot"
    return "final"


def _manifest_path(role: str):
    if role == "smoke":
        return config.SMOKE_MANIFEST
    if role == "pilot":
        return config.PILOT_MANIFEST
    return config.SAMPLE_MANIFEST


def _manifest_ids(role: str):
    path = _manifest_path(role)
    if not path.exists():
        print(f"ERROR: Sample-Manifest fehlt: {path}", file=sys.stderr)
        raise SystemExit(2)
    return dataset.load_sample_manifest(path)["question_ids"]


def get_records(args: argparse.Namespace):
    role = sample_role(args)
    ids = _manifest_ids(role)
    if args.limit is not None:
        if role == "smoke":
            print("ERROR: --smoke uses exactly one fixed question; --limit is not allowed.", file=sys.stderr)
            raise SystemExit(2)
        ids = ids[: args.limit]
    return dataset.records_for_ids(_load_records(), ids)


def _print_dry_run(runner: BenchmarkRunner, records, args, header: str) -> None:
    report = runner.dry_run(records)
    role = sample_role(args)
    print(f"=== {header} ===")
    print(f"Sample:            {role} ({len(records)} questions)")
    print(f"Model cells:       {report.model_cells}")
    print(f"Consensus cells:   {report.consensus_cells}")
    print(f"Synth-alone cells: {report.synth_cells}")
    print(f"Total cells:       {report.cells}")
    print(f"Audited payloads:  {report.audited_payloads} (no web tools)")
    print(f"Projected max cost: ${report.projected_cost_usd:.4f} (estimate)")
    if report.missing_pricing:
        print(f"WARNING: no pricing for: {', '.join(sorted(report.missing_pricing))}")
    if args.budget is not None and report.projected_cost_usd > args.budget:
        print(f"WARNING: projected cost exceeds budget ${args.budget:.2f}")


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.publish_run or args.publish_all:
        return _publish_existing_runs(args)

    config.assert_pins_match_config()
    try:
        _validate_args(args)
        if args.smoke:
            _validate_smoke_manifests()
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    role = sample_role(args)
    runner = BenchmarkRunner(
        sample_role=role,
        sample_manifest=_manifest_path(role).name,
    )
    run_id = resolve_run_id(args)
    run_dir = config.RUNS_DIR / run_id
    records = get_records(args)

    # Run-Verzeichnis + Manifest werden vor jedem Pfad angelegt/validiert (E6).
    runner.write_or_validate_manifest(run_dir)

    if not args.live:
        _print_dry_run(runner, records, args, "Benchmark Dry-Run (no --live)")
        print(f"Run dir:           {run_dir}")
        if not args.dry_run:
            print("Hinweis: ohne --live wird NICHT live ausgefuehrt (sichere Vorschau).")
        return 0

    # --- Live-Preflight (kein HTTP an LLM-Provider) ---
    api_keys = credentials.resolve_developer_api_keys(REQUIRED_PROVIDERS)
    missing = credentials.missing_credentials(api_keys, REQUIRED_PROVIDERS)
    if missing:
        print(
            f"ERROR: missing credentials for: {', '.join(missing)}. "
            "Set the corresponding DEVELOPER_*_API_KEY (or Gemini ADC) before a live run.",
            file=sys.stderr,
        )
        return 2

    _print_dry_run(runner, records, args, "Live Preflight")
    print(f"Run dir:           {run_dir}")
    print(f"Resume:            {args.resume or '(new run)'}")
    print(f"Budget cap:        {f'${args.budget:.2f}' if args.budget is not None else '(none)'}")
    print("Credentials:       OK for all 6 providers")

    if not _live_execution_allowed(args):
        print(
            "LIVE EXECUTION GATED: preflight passed, no HTTP call made. "
            "Enable the appropriate live gate for this sample mode."
        )
        return 0

    # Naechster Schritt (separat freizugeben): echter Lauf. Smoke laeuft ohne
    # E4-Zusatzaudits; Pilot laeuft mit E4-Audits.
    if args.smoke:
        result, _audits, summary = runner.run_smoke(
            records,
            run_dir=run_dir,
            api_keys=api_keys,
            budget=args.budget,
            retry_failed=bool(args.resume),
        )
    else:
        result, _audits, summary = runner.run_pilot(
            records,
            run_dir=run_dir,
            api_keys=api_keys,
            budget=args.budget,
            retry_failed=bool(args.resume),
        )
    print(
        f"Run finished: written={result.cells_written} skipped={result.cells_skipped} "
        f"failed={result.cells_failed} spent=${result.spent_usd:.4f} stopped={result.stopped}"
    )
    if summary:
        print(f"Results: {run_dir / 'results.json'} (n={summary['n_questions']}, "
              f"disagreement={summary['n_disagreement']})")
    return 0


def _publish_existing_runs(args: argparse.Namespace) -> int:
    """Publiziert fertige lokale Runs ohne Provider-Calls und ohne Rohartefakte."""
    from app.services import benchmark_reports
    from benchmark import report_reader

    if args.publish_run:
        run_ids = [args.publish_run]
    else:
        run_ids = [
            run["run_id"]
            for run in report_reader.list_runs()
            if run.get("has_results")
        ]
    if not run_ids:
        print("No finished benchmark runs found to publish.", file=sys.stderr)
        return 2

    published = 0
    for run_id in run_ids:
        run_dir = config.RUNS_DIR / run_id
        try:
            summary = benchmark_reports.publish_run_dir(run_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: failed to publish {run_id}: {exc}", file=sys.stderr)
            return 2
        published += 1
        print(
            f"Published {summary['run_id']}: {summary.get('n_questions')}Q, "
            f"source=Firestore compact report"
        )
    print(f"Published runs: {published}")
    return 0


def _validate_args(args: argparse.Namespace) -> None:
    if args.smoke and args.limit is not None:
        raise ValueError("--smoke uses exactly one fixed question; --limit is not allowed")
    if args.smoke and args.live and args.budget is None:
        raise ValueError("--smoke --live requires an explicit --budget value")
    if args.pilot and args.live and args.budget is None:
        raise ValueError("--pilot --live requires an explicit --budget value")
    if _is_final_preview(args) and args.live and args.budget is None:
        raise ValueError("--final --limit (preview) requires an explicit --budget value")
    if args.final and args.limit is not None and not _is_final_preview(args):
        raise ValueError(
            f"--final --limit preview is capped at {PREVIEW_MAX_QUESTIONS} questions; "
            "the full run requires LIVE_EXECUTION_ENABLED"
        )


def _is_final_preview(args: argparse.Namespace) -> bool:
    """Begrenzter Final-Sample-Preview: ``--final`` mit kleinem ``--limit``.
    Der volle Lauf (kein/zu grosses --limit) faellt NICHT hierunter."""
    return bool(
        args.final
        and args.limit is not None
        and 0 < args.limit <= PREVIEW_MAX_QUESTIONS
    )


def _live_execution_allowed(args: argparse.Namespace) -> bool:
    return bool(
        LIVE_EXECUTION_ENABLED
        or (args.smoke and SMOKE_LIVE_EXECUTION_ENABLED)
        or (args.pilot and PILOT_LIVE_EXECUTION_ENABLED)
        or (_is_final_preview(args) and PREVIEW_LIVE_EXECUTION_ENABLED)
    )


def _validate_smoke_manifests() -> None:
    smoke = dataset.load_sample_manifest(config.SMOKE_MANIFEST)
    pilot = dataset.load_sample_manifest(config.PILOT_MANIFEST)
    final = dataset.load_sample_manifest(config.SAMPLE_MANIFEST)
    smoke_ids = smoke.get("question_ids") or []
    if smoke.get("role") != "smoke":
        raise ValueError("Smoke manifest must have role='smoke'")
    if len(smoke_ids) != config.SMOKE_SIZE:
        raise ValueError(f"Smoke manifest must contain exactly {config.SMOKE_SIZE} question")
    smoke_id = smoke_ids[0]
    if smoke_id in set(pilot.get("question_ids") or []):
        raise ValueError(f"Smoke question {smoke_id} overlaps with pilot sample")
    if smoke_id in set(final.get("question_ids") or []):
        raise ValueError(f"Smoke question {smoke_id} overlaps with final sample")
    runner = BenchmarkRunner(sample_role="smoke", sample_manifest=config.SMOKE_MANIFEST.name)
    runner.validate_smoke_setup([{
        "question_id": smoke_id,
        "question": "validation only",
        "options": ["A", "B"],
        "answer": "A",
        "answer_index": 0,
        "category": "validation",
    }])


if __name__ == "__main__":
    raise SystemExit(main())
