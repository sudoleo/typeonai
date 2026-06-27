"""CLI fuer den Benchmark-Runner.

Sicherheits-Modell (Phase 2.5a – Pilot-Startfaehigkeit **ohne** Live-Run):
- Ohne ``--live`` passiert **nie** ein Provider-HTTP-Call: es laeuft die
  Dry-Run-Vorschau (Payloads, Tool-Audit, Kostenprojektion).
- Mit ``--live`` werden Credentials aller sechs Provider **vor** dem ersten Call
  validiert (fehlende -> klarer Abbruch mit Provider-Liste), Run-Verzeichnis +
  Manifest angelegt/geprueft und ein Preflight (Audit + Projektion) gefahren.
  Die eigentliche Ausfuehrung ist in dieser Phase **bewusst gesperrt**
  (``LIVE_EXECUTION_ENABLED = False``) und loest selbst **keinen** HTTP-Call aus.

Aufrufe:
  python -m benchmark --dry-run --pilot
  python -m benchmark --pilot --run-id pilot_v1 --budget 5
  python -m benchmark --pilot --live --budget 5            # Preflight (gated)
  python -m benchmark --resume pilot_v1 --live --budget 5  # Resume-Preflight
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

# Harter Gate: die echte Zellen-Ausfuehrung ist in Phase 2.5a gesperrt. Das
# Umlegen (+ Verdrahten von runner.run()) ist der naechste, separate Schritt
# (1-Frage-Live-Smoke-Test).
LIVE_EXECUTION_ENABLED = False


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m benchmark")
    parser.add_argument("--dry-run", action="store_true", help="Payloads bauen, kein HTTP")
    parser.add_argument("--pilot", action="store_true", help="Pilot-Sample statt Final-Sample")
    parser.add_argument("--limit", type=int, default=None, help="auf N Fragen begrenzen")
    parser.add_argument("--budget", type=float, default=None, help="Budget-Cap in USD")
    parser.add_argument("--run-id", dest="run_id", default=None, help="explizite Run-ID")
    parser.add_argument("--resume", metavar="RUN_ID", default=None, help="Run fortsetzen")
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


def _manifest_ids(pilot: bool):
    path = config.PILOT_MANIFEST if pilot else config.SAMPLE_MANIFEST
    if not path.exists():
        print(f"ERROR: Sample-Manifest fehlt: {path}", file=sys.stderr)
        raise SystemExit(2)
    return dataset.load_sample_manifest(path)["question_ids"]


def get_records(args: argparse.Namespace):
    ids = _manifest_ids(args.pilot)
    if args.limit is not None:
        ids = ids[: args.limit]
    return dataset.records_for_ids(_load_records(), ids)


def _print_dry_run(runner: BenchmarkRunner, records, args, header: str) -> None:
    report = runner.dry_run(records)
    print(f"=== {header} ===")
    print(f"Sample:            {'pilot' if args.pilot else 'final'} ({len(records)} questions)")
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
    config.assert_pins_match_config()

    runner = BenchmarkRunner()
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

    if not LIVE_EXECUTION_ENABLED:
        print(
            "LIVE EXECUTION GATED (Phase 2.5a): preflight passed, no HTTP call made. "
            "Enable runner.run() wiring + LIVE_EXECUTION_ENABLED for the smoke test."
        )
        return 0

    # Naechster Schritt (separat freizugeben): echter Lauf.
    result = runner.run(
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
