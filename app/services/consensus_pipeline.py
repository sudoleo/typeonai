"""Neutral server-side Consensus orchestration.

The pipeline combines already normalized provider answers with synthesis,
Differences parsing and agreement scoring.  API, Watch, Topic and the browser
JSON endpoint use this module; product-specific persistence and notifications
remain in their owning services.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Mapping

import app.core.config as cfg
from app.services.llm.citations import to_plain
from app.services.llm.consensus_engine import (
    compute_agreement_score,
    is_consensus_error_text,
    query_consensus,
    query_differences,
)
from app.services.llm.provider_transport import (
    PROVIDER_LABELS,
    PROVIDER_ORDER,
    ProviderAnswer,
    fan_out_provider_answers,
    query_provider,
)


@dataclass(frozen=True)
class ConsensusAnalysis:
    consensus: str
    differences_text: str
    differences_data: dict | None
    agreement: dict | None


def _answer_slots(answers: Mapping[str, ProviderAnswer | str]) -> dict[str, str]:
    """Antworttexte je Familie in PROVIDER_ORDER-Reihenfolge (leer = fehlt).
    Genau diese Reihenfolge geht als Mapping in Synthese und Judge."""
    return {
        provider: (
            answer.response if isinstance(answer, ProviderAnswer) else str(answer or "")
        )
        for provider in PROVIDER_ORDER
        for answer in [answers.get(provider, "")]
    }


def analyze_provider_answers(
    *,
    question: str,
    answers: Mapping[str, ProviderAnswer | str],
    consensus_model: str,
    keys: dict,
    model_sources: dict | None = None,
    resolved_question: str = "",
    synthesize: Callable = query_consensus,
    judge: Callable = query_differences,
    consensus_error: Callable[[str], bool] = is_consensus_error_text,
    allow_consensus_error: bool = False,
    skipped_differences_text: str = "",
    require_differences_data: bool = True,
) -> ConsensusAnalysis:
    """Run domain synthesis/parsing/scoring for normalized provider answers."""
    if len([answer for answer in answers.values() if answer]) < 2:
        raise RuntimeError("Fewer than two provider answers completed")
    slots = _answer_slots(answers)
    excluded = [
        PROVIDER_LABELS[provider]
        for provider in PROVIDER_ORDER
        if not slots[provider]
    ]
    if model_sources is None:
        model_sources = {
            answer.provider: answer.sources
            for answer in answers.values()
            if isinstance(answer, ProviderAnswer) and answer.sources
        }
    # Nur weitergereicht, wenn eine aufgeloeste Lesart vorliegt (Folgefragen).
    # synthesize/judge sind injizierbar -- Benchmark und Topic-Runner geben
    # eigene Callables herein, die dieses Argument nicht kennen und es ohne
    # Folgefrage auch nie zu sehen bekommen.
    follow_up = (
        {"resolved_question": resolved_question} if resolved_question else {}
    )
    consensus = synthesize(
        question,
        slots,
        excluded,
        consensus_model,
        keys,
        model_sources=model_sources,
        **follow_up,
    )
    if consensus_error(consensus):
        if not allow_consensus_error:
            raise RuntimeError("Consensus synthesis failed")
        return ConsensusAnalysis(
            consensus=consensus,
            differences_text=skipped_differences_text,
            differences_data=None,
            agreement=None,
        )
    differences_text, differences_data = judge(
        slots,
        consensus,
        keys,
        differences_model=consensus_model,
        excluded_models=excluded,
        **follow_up,
    )
    if not isinstance(differences_data, dict):
        if require_differences_data:
            raise RuntimeError("Differences analysis failed")
        return ConsensusAnalysis(
            consensus=consensus,
            differences_text=differences_text,
            differences_data=None,
            agreement=None,
        )
    agreement = differences_data.get("agreement")
    if not isinstance(agreement, dict):
        agreement = compute_agreement_score(differences_data)
        differences_data["agreement"] = agreement
    return ConsensusAnalysis(
        consensus=consensus,
        differences_text=differences_text,
        differences_data=differences_data,
        agreement=agreement,
    )


def run_consensus_pipeline(
    *,
    question: str,
    provider_models: dict[str, str],
    consensus_model: str | Callable[[Mapping[str, ProviderAnswer]], str],
    keys: dict,
    is_pro: bool,
    deep_think: bool = False,
    provider_order: Iterable[str] = PROVIDER_ORDER,
    provider_call: Callable = query_provider,
    synthesize: Callable = query_consensus,
    judge: Callable = query_differences,
    answer_char_limit: int | None = None,
    log_context: str = "Consensus",
) -> dict:
    answers = fan_out_provider_answers(
        question=question,
        provider_models=provider_models,
        keys=keys,
        is_pro=is_pro,
        deep_think=deep_think,
        provider_order=provider_order,
        provider_call=provider_call,
        answer_char_limit=answer_char_limit or cfg.get_consensus_answer_char_limit(),
        log_context=log_context,
    )
    if len(answers) < 2:
        raise RuntimeError("Fewer than two provider answers completed")
    resolved_consensus_model = (
        consensus_model(answers) if callable(consensus_model) else consensus_model
    )
    analysis = analyze_provider_answers(
        question=question,
        answers=answers,
        consensus_model=resolved_consensus_model,
        keys=keys,
        synthesize=synthesize,
        judge=judge,
    )
    return to_plain({
        "consensus_response": analysis.consensus,
        "differences": analysis.differences_text,
        "differences_data": analysis.differences_data,
        "agreement": analysis.agreement,
        "model_answers": [
            asdict(answers[provider])
            for provider in provider_order
            if provider in answers
        ],
    })
