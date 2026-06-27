"""CLI: ``python -m benchmark --dry-run --pilot [--limit N] [--budget USD] [--resume <run_id>]``.

Phase 2 nutzt ausschliesslich ``--dry-run`` (Kostenprojektion + Tool-Audit) –
**kein echter API-Call**. Der reale Lauf (ohne ``--dry-run``) ist Phase 3/4.
"""

from __future__ import annotations

import argparse
import sys

from benchmark import config, dataset
from benchmark.runner import BenchmarkRunner


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m benchmark")
    parser.add_argument("--dry-run", action="store_true", help="Payloads bauen, kein HTTP")
    parser.add_argument("--pilot", action="store_true", help="Pilot-Sample statt Final-Sample")
    parser.add_argument("--limit", type=int, default=None, help="auf N Fragen begrenzen")
    parser.add_argument("--budget", type=float, default=None, help="Budget-Cap in USD")
    parser.add_argument("--resume", metavar="RUN_ID", default=None, help="Run fortsetzen")
    return parser.parse_args(argv)


def _load_records():
    try:
        df = dataset.load_dataframe()
    except ImportError as exc:
        print(
            "ERROR: Parquet-Engine fehlt (pyarrow/fastparquet). MMLU-Pro liegt nur als "
            "Parquet vor; bitte einen Parquet-Reader installieren, um den Datensatz zu laden.\n"
            f"Detail: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return dataset.records_from_dataframe(df)


def _manifest_ids(pilot: bool):
    path = config.PILOT_MANIFEST if pilot else config.SAMPLE_MANIFEST
    if not path.exists():
        print(
            f"ERROR: Sample-Manifest fehlt: {path}\n"
            "Bitte zuerst die eingefrorenen Samples bauen (dataset.build_samples + "
            "write_sample_manifest).",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return dataset.load_sample_manifest(path)["question_ids"]


def main(argv=None) -> int:
    args = _parse_args(argv)

    config.assert_pins_match_config()

    ids = _manifest_ids(args.pilot)
    if args.limit is not None:
        ids = ids[: args.limit]

    records = dataset.records_for_ids(_load_records(), ids)
    runner = BenchmarkRunner()

    if args.dry_run:
        report = runner.dry_run(records)
        print("=== Benchmark Dry-Run ===")
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
        return 0

    print(
        "Realer Lauf ist Phase 3/4 und in Phase 2 nicht vorgesehen. Bitte --dry-run nutzen.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
