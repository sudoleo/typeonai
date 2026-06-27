"""MMLU-Pro laden & reproduzierbar samplen (Plan §6, E3/E3b).

- Quelle: ``hf_hub_download(TIGER-Lab/MMLU-Pro, data/test-*.parquet)`` ->
  ``pandas.read_parquet`` (benoetigt ein Parquet-Engine, z. B. pyarrow).
- Disjunktes, reproduzierbares Sampling: erst Pilot (fester ``pilot_seed``),
  dessen IDs ausschliessen, dann Final = je ``per_cat`` Fragen pro Kategorie
  (fester ``final_seed``).

Das Sampling arbeitet auf plain-Python-Records (nicht direkt auf dem DataFrame)
und nutzt ``random.Random`` ueber **sortierte** question_ids – damit ist es
unabhaengig von pandas-/numpy-Version exakt rekonstruierbar.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from benchmark import config


def download_test_parquets() -> list[Path]:
    """Laedt die test-*.parquet-Dateien von HuggingFace und gibt lokale Pfade."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    files = [
        name
        for name in api.list_repo_files(config.HF_REPO_ID, repo_type=config.HF_REPO_TYPE)
        if name.startswith("data/test-") and name.endswith(".parquet")
    ]
    if not files:
        raise RuntimeError("No MMLU-Pro test parquet files found on HuggingFace.")
    return [
        Path(hf_hub_download(config.HF_REPO_ID, name, repo_type=config.HF_REPO_TYPE))
        for name in sorted(files)
    ]


def load_dataframe(parquet_paths: list[Path] | None = None):
    """Liest die Parquet-Dateien zu einem DataFrame zusammen."""
    import pandas as pd

    if parquet_paths is None:
        parquet_paths = download_test_parquets()
    frames = [pd.read_parquet(path) for path in parquet_paths]
    return pd.concat(frames, ignore_index=True)


def records_from_dataframe(df) -> list[dict]:
    """Extrahiert die benoetigten Felder als plain-Python-Records."""
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "question_id": int(row["question_id"]),
                "question": str(row["question"]),
                "options": [str(opt) for opt in list(row["options"])],
                "answer": str(row["answer"]).strip().upper(),
                "answer_index": int(row["answer_index"]),
                "category": str(row["category"]),
            }
        )
    return records


def _index_by_id(records: list[dict]) -> dict[int, dict]:
    return {rec["question_id"]: rec for rec in records}


def sample_pilot(
    records: list[dict], seed: int = config.PILOT_SEED, size: int = config.PILOT_SIZE
) -> list[int]:
    """Zieht ``size`` Pilot-Fragen deterministisch aus dem gesamten Pool."""
    all_ids = sorted(rec["question_id"] for rec in records)
    if len(all_ids) < size:
        raise ValueError(f"Not enough questions for pilot: need {size}, have {len(all_ids)}")
    rng = random.Random(seed)
    return sorted(rng.sample(all_ids, size))


def sample_final(
    records: list[dict],
    exclude_ids: set[int],
    seed: int = config.FINAL_SEED,
    per_cat: int = config.QUESTIONS_PER_CATEGORY,
) -> list[int]:
    """Zieht ``per_cat`` Fragen pro Kategorie deterministisch aus dem um
    ``exclude_ids`` bereinigten Pool. Bricht ab, falls eine Kategorie < ``per_cat``
    verfuegbare Fragen hat (kein stilles Auffuellen, E3b)."""
    by_category: dict[str, list[int]] = {}
    for rec in records:
        if rec["question_id"] in exclude_ids:
            continue
        by_category.setdefault(rec["category"], []).append(rec["question_id"])

    rng = random.Random(seed)
    selected: list[int] = []
    for category in sorted(by_category):
        ids = sorted(by_category[category])
        if len(ids) < per_cat:
            raise ValueError(
                f"Category {category!r} has only {len(ids)} questions, need {per_cat}"
            )
        selected.extend(sorted(rng.sample(ids, per_cat)))
    return sorted(selected)


def build_samples(
    records: list[dict],
    pilot_seed: int = config.PILOT_SEED,
    final_seed: int = config.FINAL_SEED,
    pilot_size: int = config.PILOT_SIZE,
    per_cat: int = config.QUESTIONS_PER_CATEGORY,
) -> tuple[list[int], list[int]]:
    """Baut disjunkte Pilot- und Final-Samples (E3)."""
    pilot_ids = sample_pilot(records, seed=pilot_seed, size=pilot_size)
    final_ids = sample_final(
        records, exclude_ids=set(pilot_ids), seed=final_seed, per_cat=per_cat
    )
    overlap = set(pilot_ids) & set(final_ids)
    if overlap:
        raise AssertionError(f"Pilot and final samples overlap: {sorted(overlap)}")
    return pilot_ids, final_ids


def records_for_ids(records: list[dict], ids: list[int]) -> list[dict]:
    """Gibt die Records zu ``ids`` in der Reihenfolge von ``ids`` zurueck."""
    lookup = _index_by_id(records)
    missing = [qid for qid in ids if qid not in lookup]
    if missing:
        raise KeyError(f"Question ids not found in dataset: {missing}")
    return [lookup[qid] for qid in ids]


def write_sample_manifest(
    path: Path, *, version: str, seed: int, question_ids: list[int], records: list[dict], extra: dict | None = None
) -> dict:
    """Schreibt ein eingefrorenes Sample-Manifest (committet)."""
    selected = records_for_ids(records, question_ids)
    categories: dict[str, int] = {}
    for rec in selected:
        categories[rec["category"]] = categories.get(rec["category"], 0) + 1
    manifest = {
        "version": version,
        "source_repo": config.HF_REPO_ID,
        "seed": seed,
        "count": len(question_ids),
        "question_ids": list(question_ids),
        "categories": dict(sorted(categories.items())),
    }
    if extra:
        manifest.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def load_sample_manifest(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
