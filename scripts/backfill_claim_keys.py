#!/usr/bin/env python3
"""Give the stored Topic history the claim identities new runs write by default.

Runs published before the identity Judge existed carry no ``key`` on their
Position Map dimensions, so the Claim Ledger falls back to comparing wording
for them. That works, but it splits a claim whose phrasing was rewritten
heavily into several short-lived entries. This walks a Topic's runs oldest
first and asks the same Judge new runs use to align each run's claims with the
claims already tracked, then writes the resulting keys back.

Runs stay otherwise untouched: only ``opinion_map.dimensions[*].key`` is
added. Re-running the script is safe -- runs that already carry keys on every
dimension are skipped unless --force is given.

    python scripts/backfill_claim_keys.py --slug release-news-for-gpt-6 --dry-run
    python scripts/backfill_claim_keys.py --all
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import topic_runner, topics  # noqa: E402
from app.services.llm import provider_transport  # noqa: E402
from app.services.llm.consensus_engine import query_claim_identity  # noqa: E402


def _judge_engine(keys: dict) -> str:
    """First configured provider label, matching how a Topic run picks one."""
    for provider in provider_transport.PROVIDER_ORDER:
        if provider_transport.provider_available(provider, keys):
            return provider_transport.PROVIDER_LABELS[provider]
    raise SystemExit("No provider credentials are configured for the Judge.")


def backfill_topic(topic: dict, *, dry_run: bool, force: bool) -> int:
    db = topics.db_firestore
    runs = topics.list_runs(topic["id"], db=db)
    api_keys = provider_transport.developer_keys()
    engine = _judge_engine(api_keys)
    print(f"\n{topic['slug']}: {len(runs)} runs")
    written = 0
    for position, run in enumerate(runs):
        dimensions = (run.get("opinion_map") or {}).get("dimensions") or []
        if not dimensions:
            continue
        if not force and all(dimension.get("key") for dimension in dimensions):
            continue
        known = topic_runner.known_claims_from_runs(
            runs[max(0, position - topic_runner.KNOWN_CLAIM_RUNS):position]
        )
        labels = [str(dimension.get("label") or "") for dimension in dimensions]
        matches = query_claim_identity(known, labels, api_keys, engine) if known else {}
        for index, dimension in enumerate(dimensions):
            dimension["key"] = matches.get(index) or f"{run['id']}-{index}"
        assigned = ", ".join(
            f"{index}->{dimension['key']}" for index, dimension in enumerate(dimensions)
        )
        print(f"  v{run.get('version')} {str(run.get('observed_at'))[:10]}: {assigned}")
        if dry_run:
            continue
        (
            db.collection(topics.TOPICS_COLLECTION).document(topic["id"])
            .collection("runs").document(run["id"])
            .set({"opinion_map": run["opinion_map"]}, merge=True)
        )
        written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", help="Topic slug to backfill")
    parser.add_argument("--all", action="store_true", help="Backfill every Topic")
    parser.add_argument("--dry-run", action="store_true", help="Print, do not write")
    parser.add_argument(
        "--force", action="store_true", help="Re-key runs that already carry keys"
    )
    args = parser.parse_args()
    if not args.slug and not args.all:
        parser.error("pass --slug or --all")
    if os.getenv("MOCK_LLM") == "1":
        parser.error("MOCK_LLM=1 would write fixture identities into real runs")

    if args.all:
        selected = [
            topics.get_topic(item["id"])
            for item in topics.list_public_topics()
        ]
    else:
        topic = topics.get_topic_by_slug(args.slug)
        if not topic:
            raise SystemExit(f"Topic not found: {args.slug}")
        selected = [topic]

    written = sum(
        backfill_topic(topic, dry_run=args.dry_run, force=args.force)
        for topic in selected if topic
    )
    print(f"\n{'Would update' if args.dry_run else 'Updated'} {written} run(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
