"""Topic-specific projection on top of the neutral Consensus pipeline.

Topics own research/timeline semantics.  They intentionally do not call the
Watch executor or inherit Watch notification, baseline or quota behaviour.
"""

from __future__ import annotations

from app.services import opinion_map, share_snapshots
from app.services.consensus_pipeline import run_consensus_pipeline
from app.services.llm.consensus_engine import query_consensus_change
from app.services.llm.mock_llm import mock_llm_enabled
from app.services.llm import provider_transport


def execute_topic(
    question: str,
    previous_consensus: str,
    condition: str = "",
    previous_opinion_map=None,
    is_pro: bool = False,
    excluded_providers=None,
    baseline_consensus: str = "",
    model_overrides=None,
) -> dict:
    keys = provider_transport.developer_keys()
    excluded = {
        str(provider or "").strip().lower()
        for provider in (excluded_providers or ())
    }
    configured = dict(model_overrides or {})
    provider_models = {
        provider: configured[provider]
        for provider in provider_transport.PROVIDER_ORDER
        if provider not in excluded
        and configured.get(provider)
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
