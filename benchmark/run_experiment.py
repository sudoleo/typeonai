"""Launcher fuer das Disagreement-Charakterisierungs-Experiment (V0 only).

Faehrt die bestehende ``BenchmarkRunner.run()``-Pipeline **unveraendert** auf dem
disagreement-angereicherten Sample (``mmlu_pro_disagreement_v1.json``, 66 Fragen)
mit dem **produktiven** Consensus-Prompt (V0 – kein Prompt-Eingriff). **Keine
E4-Audits** (die skalieren nicht auf grosse Samples, siehe final_v1: ~404
ungezaehlte Calls ohne Checkpoint). Danach ``results.json`` direkt aus
``calls.jsonl``.

Ziel: charakterisieren, wie sich Consensus / Majority / bestes Einzelmodell /
Synth-allein auf einer **uneinigkeits-dichten** Stichprobe verhalten, BEVOR ueber
einen Prompt-Eingriff entschieden wird. Auswertung primaer ueber die
Disagreement-Teilmenge und ``Consensus − Synth-allein`` (kontrolliert fuers
Eigenwissen des Synthesizers).

Sicherheit (wie der regulaere CLI):
- ohne ``--live`` -> Dry-Run (Payloads + Kostenprojektion, **kein HTTP**).
- mit ``--live`` -> Credential-Check aller 6 Provider, dann echter Lauf.
  ``--budget`` ist dann verpflichtend. Resume-bar (gleiche run-id, ``--resume``).

WICHTIG zum Budget-Cap: der Cap wird gegen die getrackte ``est_cost`` geprueft;
die Consensus-Zelle ist darin cap-basiert **ueberschaetzt** (~$0.22/Frage Phantom,
real ~$1 gesamt). Realer Spend fuer 66 Fragen ~$4; tracked ~$14-18. Cap daher
NICHT zu niedrig setzen (>= $30), sonst stoppt das Phantom-Accounting vorzeitig.

Aufruf (neue Session):
  venv/Scripts/python.exe -m benchmark.run_experiment --dry-run
  venv/Scripts/python.exe -m benchmark.run_experiment --live --budget 30
  venv/Scripts/python.exe -m benchmark.run_experiment --live --budget 30 --resume
"""

from __future__ import annotations

import argparse
import sys

from app.services.llm import credentials

from benchmark import config, dataset
from benchmark import results as results_mod
from benchmark.runner import BenchmarkRunner

RUN_ID = "experiment_v1"
REQUIRED_PROVIDERS = ["OpenAI", "Mistral", "Anthropic", "Gemini", "DeepSeek", "Grok"]


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m benchmark.run_experiment")
    parser.add_argument("--dry-run", action="store_true", help="Payloads + Projektion, kein HTTP")
    parser.add_argument("--live", action="store_true", help="echten Lauf ausfuehren (Credential-Check)")
    parser.add_argument("--budget", type=float, default=None, help="Budget-Cap in USD (bei --live Pflicht)")
    parser.add_argument("--resume", action="store_true", help="Fehl-Zellen erneut versuchen")
    return parser.parse_args(argv)


def _load_records() -> list[dict]:
    ids = dataset.load_sample_manifest(config.EXPERIMENT_MANIFEST)["question_ids"]
    all_records = dataset.records_from_dataframe(dataset.load_dataframe())
    return dataset.records_for_ids(all_records, ids)


def main(argv=None) -> int:
    args = _parse_args(argv)
    config.assert_pins_match_config()

    runner = BenchmarkRunner(
        sample_role="experiment",
        sample_manifest=config.EXPERIMENT_MANIFEST.name,
    )
    run_dir = config.RUNS_DIR / RUN_ID
    records = _load_records()
    runner.write_or_validate_manifest(run_dir)

    report = runner.dry_run(records)
    print("=== Disagreement-Experiment (V0, Produktiv-Prompt, keine Audits) ===")
    print(f"Sample:             experiment ({len(records)} Fragen)")
    print(f"Total cells:        {report.cells} "
          f"({report.model_cells} model + {report.consensus_cells} consensus + {report.synth_cells} synth)")
    print(f"Audited (no web):   {report.audited_payloads}")
    print(f"Projected MAX cost: ${report.projected_cost_usd:.2f} (Worst-Case; Consensus ueberschaetzt)")
    print(f"Run dir:            {run_dir}")

    if not args.live:
        print("Dry-Run: kein HTTP. Fuer den echten Lauf: --live --budget 30")
        return 0

    if args.budget is None:
        print("ERROR: --live verlangt ein explizites --budget (empfohlen: 30).", file=sys.stderr)
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
