"""Topic-specific projection on top of the neutral Consensus pipeline.

Topics own research/timeline semantics.  They intentionally do not call the
Watch executor or inherit Watch notification, baseline or quota behaviour.
"""

from __future__ import annotations

import logging

from app.core.observability import safe_exception
from app.services import opinion_map, share_snapshots
from app.services.consensus_pipeline import run_consensus_pipeline
from app.services.llm.consensus_engine import query_claim_identity, query_consensus_change
from app.services.llm.mock_llm import mock_llm_enabled
from app.services.llm import provider_transport


def _stamp_claim_keys(position_map, known_claims, keys, engine, prefix: str) -> None:
    """Give every claim of this run a stable identity, in place.

    A claim that continues one the Topic already tracks keeps that key; every
    other claim gets a fresh one derived from the run it entered in. When the
    identity Judge is unavailable the run still publishes -- the Claim Ledger
    then falls back to comparing wording, which is how the whole history
    before this step is read anyway.
    """
    dimensions = (position_map or {}).get("dimensions") or []
    if not dimensions:
        return
    labels = [str(dimension.get("label") or "") for dimension in dimensions]
    matches = {}
    if known_claims:
        try:
            matches = query_claim_identity(known_claims, labels, keys, engine)
        except Exception as exc:
            logging.warning(
                "Topic claim identity skipped category=%s", safe_exception(exc)
            )
    for index, dimension in enumerate(dimensions):
        dimension["key"] = matches.get(index) or f"{prefix or 'c'}-{index}"


def execute_topic(
    question: str,
    previous_consensus: str,
    condition: str = "",
    previous_opinion_map=None,
    is_pro: bool = False,
    baseline_consensus: str = "",
    model_overrides=None,
    known_claims=None,
    claim_key_prefix: str = "",
) -> dict:
    keys = provider_transport.developer_keys()
    configured = dict(model_overrides or {})
    provider_models = {
        provider: configured[provider]
        for provider in provider_transport.PROVIDER_ORDER
        if configured.get(provider)
        and provider_transport.provider_available(provider, keys)
    }
    if mock_llm_enabled():
        for provider in provider_models:
            keys[provider_transport.PROVIDER_LABELS[provider]] = "mock"
    if len(provider_models) < 2:
        raise RuntimeError("Fewer than two Topic provider transports are available")
    def first_successful_engine(answers) -> str:
        provider = next(name for name in provider_models if name in answers)
        return provider_transport.PROVIDER_LABELS[provider]

    pipeline = run_consensus_pipeline(
        question=question,
        provider_models=provider_models,
        consensus_model=first_successful_engine,
        keys=keys,
        is_pro=is_pro,
        provider_order=provider_transport.PROVIDER_ORDER,
        log_context="Consensus Topic",
    )
    included = {item["provider"] for item in pipeline["model_answers"]}
    engine = provider_transport.PROVIDER_LABELS[
        next(name for name in provider_models if provider_transport.PROVIDER_LABELS[name] in included)
    ]
    consensus = pipeline["consensus_response"]
    if str(previous_consensus or "").strip():
        change = query_consensus_change(
            previous_consensus, consensus, keys, engine, condition=condition
        )
    else:
        change = {
            "changed": False,
            "severity": "minor",
            "change_summary": "First consensus established.",
        }
    differences = pipeline["differences_data"]
    position_map = opinion_map.build_opinion_map(
        differences,
        previous_opinion_map,
        consensus_changed=bool(change.get("changed")),
    )
    _stamp_claim_keys(position_map, known_claims, keys, engine, claim_key_prefix)
    model_answers = pipeline["model_answers"]
    model_sources = {
        item["provider"]: item.get("sources") or []
        for item in model_answers if item.get("sources")
    }
    return {
        "consensus": consensus,
        "agreement_score": pipeline["agreement"]["score"],
        "verdict": pipeline["agreement"].get("level") or "",
        "opinion_map": position_map,
        "differences_data": differences,
        "differences_text": pipeline["differences"],
        "sources": share_snapshots.sanitize_sources(model_sources),
        "included_models": share_snapshots.build_included_models(
            [item["provider"] for item in model_answers],
            {item["provider"]: item["model"] for item in model_answers},
        ),
        "consensus_model": engine,
        **change,
    }
