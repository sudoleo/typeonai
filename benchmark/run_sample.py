"""Generischer Launcher fuer **repraesentative** Zufalls-Sample-Runs (V0).

Faehrt die bestehende ``BenchmarkRunner.run()``-Pipeline **unveraendert** (V0 =
Produktiv-Consensus, kein Prompt-Eingriff, **keine** E4-Audits) auf einem
beliebigen, committeten Sample-Manifest. Anders als ``run_experiment`` ist das
Sample hier **nicht** disagreement-angereichert, sondern flach zufaellig gezogen
(``dataset.sample_random``) – damit bleibt die Gesamt-Accuracy eine unverzerrte
MMLU-Pro-Aussage und die Uneinigkeits-Teilmenge faellt unverzerrt mit an. Mehrere
solcher Runs (je 100-200 Fragen, disjunkt) werden ueber die Zeit **gepoolt**.

Sicherheit (wie der regulaere CLI):
- ohne ``--live`` -> Dry-Run (Payloads + Kostenprojektion, **kein HTTP**).
- mit ``--live`` -> Credential-Check aller 6 Provider, dann echter Lauf.
  ``--budget`` ist dann verpflichtend. Resume-bar (gleiche run-id, ``--resume``).

Budget-Cap-Hinweis: der Cap wird gegen die getrackte ``est_cost`` geprueft; die
Consensus-Zelle ist darin cap-basiert **ueberschaetzt** (~$0.22/Frage Phantom).
Real ~$4-5 / 66 Fragen; tracked ~$18. Cap grosszuegig setzen (Phantom skaliert
mit der Fragenzahl): grob ``>= 0.3 * Fragenzahl`` USD.

Aufruf:
  venv/Scripts/python.exe -m benchmark.run_sample --manifest data/benchmark/mmlu_pro_random_v1.json --run-id random_v1 --dry-run
  venv/Scripts/python.exe -m benchmark.run_sample --manifest data/benchmark/mmlu_pro_random_v1.json --run-id random_v1 --live --budget 60
  venv/Scripts/python.exe -m benchmark.run_sample --manifest data/benchmark/mmlu_pro_random_v1.json --run-id random_v1 --live --budget 60 --resume
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.services.llm import credentials

from benchmark import config, dataset
from benchmark import results as results_mod
from benchmark.runner import BenchmarkRunner

REQUIRED_PROVIDERS = ["OpenAI", "Mistral", "Anthropic", "Gemini", "DeepSeek", "Grok"]


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m benchmark.run_sample")
    parser.add_argument("--manifest", required=True, help="Pfad zum Sample-Manifest (committet)")
    parser.add_argument("--run-id", required=True, help="Run-ID (Verzeichnis unter data/benchmark/runs/)")
    parser.add_argument("--dry-run", action="store_true", help="Payloads + Projektion, kein HTTP")
    parser.add_argument("--live", action="store_true", help="echten Lauf ausfuehren (Credential-Check)")
    parser.add_argument("--budget", type=float, default=None, help="Budget-Cap in USD (bei --live Pflicht)")
    parser.add_argument("--resume", action="store_true", help="Fehl-Zellen erneut versuchen")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    config.assert_pins_match_config()

    manifest_path = Path(args.manifest)
    ids = dataset.load_sample_manifest(manifest_path)["question_ids"]
    all_records = dataset.records_from_dataframe(dataset.load_dataframe())
    records = dataset.records_for_ids(all_records, ids)

    runner = BenchmarkRunner(sample_role="random", sample_manifest=manifest_path.name)
    run_dir = config.RUNS_DIR / args.run_id
    runner.write_or_validate_manifest(run_dir)

    report = runner.dry_run(records)
    print("=== Zufalls-Sample-Run (V0, Produktiv-Prompt, repraesentativ, keine Audits) ===")
    print(f"Manifest:           {manifest_path.name}")
    print(f"Sample:             random ({len(records)} Fragen)")
    print(f"Total cells:        {report.cells} "
          f"({report.model_cells} model + {report.consensus_cells} consensus + {report.synth_cells} synth)")
    print(f"Audited (no web):   {report.audited_payloads}")
    print(f"Projected MAX cost: ${report.projected_cost_usd:.2f} (Worst-Case; Consensus ueberschaetzt)")
    print(f"Run dir:            {run_dir}")

    if not args.live:
        rec = max(60, round(0.3 * len(records)))
        print(f"Dry-Run: kein HTTP. Fuer den echten Lauf: --live --budget {rec}")
        return 0

    if args.budget is None:
        print("ERROR: --live verlangt ein explizites --budget.", file=sys.stderr)
        return 2

    api_keys = credentials.resolve_developer_api_keys(REQUIRED_PROVIDERS)
    missing = credentials.missing_credentials(api_keys, REQUIRED_PROVIDERS)
    if missing:
        print(
            f"ERROR: fehlende Credentials fuer: {', '.join(missing)}. "
            "DEVELOPER_*_API_KEY (bzw. Gemini ADC) setzen.",
            file=sys.stderr,
        )
        return 2

    result = runner.run(
        records,
        run_dir=run_dir,
        api_keys=api_keys,
        budget=args.budget,
        retry_failed=args.resume,
    )
    print(
        f"Run finished: written={result.cells_written} skipped={result.cells_skipped} "
        f"failed={result.cells_failed} spent=${result.spent_usd:.4f} stopped={result.stopped}"
    )

    if result.stopped:
        print("STOPPED (Budget-Cap). Resume mit demselben Aufruf + --resume.", file=sys.stderr)
        return 0

    summary = results_mod.write_results(run_dir, consensus_model=config.CONSENSUS_MODEL)
    print(f"results.json: {run_dir / 'results.json'} "
          f"(n={summary['n_questions']}, disagreement={summary['n_disagreement']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
