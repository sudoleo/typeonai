"""Deterministic SEO recommendations plus an explicitly invoked content judge."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Annotated, Literal

import openai
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.services import seo_data, seo_dossier
from app.services.seo_repository import FirestoreSeoRepository


RULE_VERSION = "seo-recommendation-v1"
RECOMMENDATIONS = (
    "wait",
    "monitor",
    "protect_winner",
    "refresh_title_and_intro",
    "refresh_content",
    "investigate_decline",
    "noindex_candidate",
)
NOINDEX_MIN_OBSERVED_DAYS = 60
NOINDEX_MAX_IMPRESSIONS = 3
REQUIRED_FINAL_DAYS = 28
MAX_PROMPT_CHARS = 8_000


class SeoRecommendationError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.safe_message = message


class ContentJudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recommendation: Literal[
        "wait", "monitor", "protect_winner", "refresh_title_and_intro",
        "refresh_content", "investigate_decline", "noindex_candidate",
    ]
    confidence: float = Field(ge=0, le=1)
    evidence: list[Annotated[str, Field(min_length=1, max_length=300)]] = Field(
        min_length=1, max_length=8
    )
    proposed_changes: list[Annotated[str, Field(min_length=1, max_length=400)]] = Field(
        min_length=0, max_length=8
    )
    review_after_days: int = Field(ge=1, le=365)
    requires_human_approval: Literal[True]


CONTENT_JUDGE_SCHEMA = ContentJudgeResult.model_json_schema()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value) -> datetime | None:
    if isinstance(value, datetime):
        return value.replace(tzinfo=value.tzinfo or timezone.utc).astimezone(timezone.utc)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.combine(date.fromisoformat(text), datetime.min.time())
        except ValueError:
            return None
    return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).astimezone(timezone.utc)


def _clip_list(values, *, count=8, chars=300) -> list[str]:
    result = []
    for value in values or []:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if text:
            result.append(text[:chars])
        if len(result) >= count:
            break
    return result


def _period_metrics(metrics: list[dict], start: date, end: date) -> dict:
    return seo_data.aggregate_metrics([
        row for row in metrics
        if start.isoformat() <= str(row.get("date") or "") <= end.isoformat()
    ])


def _has_positive_development(metrics: list[dict], final_date: date) -> bool:
    previous = _period_metrics(
        metrics, final_date - timedelta(days=27), final_date - timedelta(days=14)
    )
    recent = _period_metrics(
        metrics, final_date - timedelta(days=13), final_date
    )
    impressions_up = (
        recent["impressions"] >= previous["impressions"] + 2
        and recent["impressions"] >= previous["impressions"] * 1.25
    )
    clicks_up = recent["clicks"] > previous["clicks"]
    return bool(impressions_up or clicks_up)


def deterministic_recommendation(
    page: dict,
    metrics: list[dict],
    query_snapshot: dict | None,
    *,
    final_date: date,
    now: datetime,
) -> dict:
    """Pure, inspectable rules. It never writes or mutates page state."""
    summary = seo_data.aggregate_metrics(metrics)
    status_class = seo_data.classify_status(metrics, final_date)
    first_seen = _as_utc(page.get("first_seen_at"))
    observed_days = max(0, (now.date() - first_seen.date()).days) if first_seen else 0
    unique_days = len({str(row.get("date") or "") for row in metrics if row.get("date")})
    positive_development = _has_positive_development(metrics, final_date)
    dossier = page.get("dossier") if isinstance(page.get("dossier"), dict) else {}

    uncertainties = list(dossier.get("technical_uncertainties") or [])
    if not first_seen:
        uncertainties.append("missing_first_seen_at")
    if unique_days < REQUIRED_FINAL_DAYS:
        uncertainties.append("insufficient_final_daily_data")
    if not query_snapshot:
        uncertainties.append("missing_query_data")
    else:
        if str(query_snapshot.get("period_end") or "") != final_date.isoformat():
            uncertainties.append("stale_query_data")
        if query_snapshot.get("partial") or not query_snapshot.get("complete", False):
            uncertainties.append("partial_query_data")
    if not dossier.get("title"):
        uncertainties.append("missing_title")
    if not dossier.get("meta_description"):
        uncertainties.append("missing_meta_description")
    uncertainties = sorted(set(uncertainties))

    practically_invisible = (
        summary["clicks"] == 0
        and summary["impressions"] <= NOINDEX_MAX_IMPRESSIONS
    )
    safeguards = {
        "observed_at_least_60_days": observed_days >= NOINDEX_MIN_OBSERVED_DAYS,
        "has_28_finalized_days": unique_days >= REQUIRED_FINAL_DAYS,
        "practically_no_visibility": practically_invisible,
        "no_positive_development": not positive_development,
        "no_technical_uncertainties": not uncertainties,
    }
    noindex_eligible = all(safeguards.values())

    evidence = [
        f"Status class: {status_class}.",
        f"Observed for {observed_days} days; {unique_days} finalized daily rows are available.",
        f"28-day visibility: {summary['clicks']} clicks and {summary['impressions']} impressions.",
    ]
    if positive_development:
        evidence.append("The recent 14-day window shows positive development.")
    if uncertainties:
        evidence.append("Uncertainties: " + ", ".join(uncertainties) + ".")

    if noindex_eligible:
        recommendation, confidence, review_after = "noindex_candidate", 0.88, 14
        evidence.append("Every noindex safeguard passed; this remains a human-only recommendation.")
    elif status_class == "winner":
        recommendation, confidence, review_after = "protect_winner", 0.92, 28
    elif status_class == "opportunity":
        recommendation, confidence, review_after = "refresh_title_and_intro", 0.82, 21
    elif status_class == "declining":
        recommendation, confidence, review_after = "investigate_decline", 0.84, 14
    elif status_class == "insufficient_data" or observed_days < 14:
        recommendation, confidence, review_after = "wait", 0.9, 14
    elif status_class == "invisible":
        # An invisible status alone is never enough for noindex_candidate.
        recommendation, confidence, review_after = "monitor", 0.78, 21
    elif summary["clicks"] == 0 and summary["impressions"] >= 30:
        recommendation, confidence, review_after = "refresh_content", 0.72, 28
    else:
        recommendation, confidence, review_after = "monitor", 0.74, 28

    return {
        "rule_version": RULE_VERSION,
        "status_class": status_class,
        "recommendation": recommendation,
        "confidence": confidence,
        "evidence": _clip_list(evidence),
        "review_after_days": review_after,
        "requires_human_approval": True,
        "data_window": {
            "period_start": (final_date - timedelta(days=27)).isoformat(),
            "period_end": final_date.isoformat(),
            "finalized_days": unique_days,
            "query_data_complete": bool(
                query_snapshot and query_snapshot.get("complete") and not query_snapshot.get("partial")
            ),
        },
        "safeguards": safeguards,
    }


class SeoContentJudge:
    def __init__(self, *, api_key=None, model=None, caller=None):
        self.api_key = api_key if api_key is not None else os.environ.get("DEVELOPER_OPENAI_API_KEY")
        self.model = model if model is not None else os.environ.get("SEO_CONTENT_JUDGE_MODEL")
        self.caller = caller

    @property
    def configured(self) -> bool:
        return bool(str(self.api_key or "").strip() and str(self.model or "").strip())

    def status(self) -> dict:
        return {
            "configured": self.configured,
            "status": "configured" if self.configured else "not_configured",
            "message": "Optional content judge is configured."
            if self.configured else "Set SEO_CONTENT_JUDGE_MODEL and the server OpenAI key to enable it.",
        }

    @staticmethod
    def validate_response(raw, *, deterministic_recommendation_value: str) -> dict:
        try:
            parsed = ContentJudgeResult.model_validate_json(raw) if isinstance(raw, str) else ContentJudgeResult.model_validate(raw)
        except (ValidationError, ValueError, TypeError):
            raise SeoRecommendationError(
                "invalid_llm_response", "The content judge returned invalid structured JSON."
            ) from None
        result = parsed.model_dump()
        result["evidence"] = _clip_list(result["evidence"], count=8, chars=300)
        result["proposed_changes"] = _clip_list(
            result["proposed_changes"], count=8, chars=400
        )
        if (
            result["recommendation"] == "noindex_candidate"
            and deterministic_recommendation_value != "noindex_candidate"
        ):
            raise SeoRecommendationError(
                "unsafe_llm_response",
                "The content judge cannot introduce a noindex candidate that failed deterministic safeguards.",
            )
        return result

    def ask(self, context: dict, *, deterministic_recommendation_value: str) -> dict:
        if not self.configured:
            raise SeoRecommendationError(
                "llm_not_configured", "The optional SEO content judge is not configured."
            )
        instructions = (
            "Review this read-only SEO page dossier. Return only the requested JSON. "
            "Do not propose publishing, deleting, redirecting, reindexing, or changing robots directives. "
            "Any change requires human approval.\n\n"
        )
        # Keep the JSON intact while enforcing a hard prompt cap. The complete
        # page/share body is never sent; this further tightens the already
        # bounded dossier representation and query list.
        bounded = json.loads(json.dumps(context, ensure_ascii=False, default=str))
        bounded["top_queries"] = list(bounded.get("top_queries") or [])[:10]
        dossier = bounded.get("dossier") if isinstance(bounded.get("dossier"), dict) else {}
        dossier["content_representation"] = str(
            dossier.get("content_representation") or ""
        )[:1_800]
        dossier["content_summary"] = str(dossier.get("content_summary") or "")[:600]
        bounded["dossier"] = dossier
        serialized = json.dumps(
            bounded, ensure_ascii=False, default=str, separators=(",", ":")
        )
        if len(instructions) + len(serialized) > MAX_PROMPT_CHARS:
            bounded["top_queries"] = bounded["top_queries"][:5]
            dossier["content_representation"] = dossier["content_representation"][:800]
            serialized = json.dumps(
                bounded, ensure_ascii=False, default=str, separators=(",", ":")
            )
        if len(instructions) + len(serialized) > MAX_PROMPT_CHARS:
            # All remaining fields are summaries/metrics; removing the content
            # excerpt is preferable to slicing structured JSON.
            dossier["content_representation"] = ""
            serialized = json.dumps(
                bounded, ensure_ascii=False, default=str, separators=(",", ":")
            )
        prompt = instructions + serialized
        if len(prompt) > MAX_PROMPT_CHARS:
            raise SeoRecommendationError(
                "prompt_too_large", "The minimized SEO dossier is still too large to review safely."
            )
        if self.caller:
            raw = self.caller(prompt, CONTENT_JUDGE_SCHEMA)
        else:
            client = openai.OpenAI(api_key=self.api_key, timeout=45)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a conservative SEO content reviewer."},
                    {"role": "user", "content": prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "seo_content_judgement",
                        "strict": True,
                        "schema": CONTENT_JUDGE_SCHEMA,
                    },
                },
            )
            raw = response.choices[0].message.content or ""
        return self.validate_response(
            raw, deterministic_recommendation_value=deterministic_recommendation_value
        )


class SeoRecommendationService:
    def __init__(self, db=None, *, repository=None, content_judge=None, clock=None):
        self.repository = repository or FirestoreSeoRepository(db)
        self.content_judge = content_judge or SeoContentJudge()
        self.clock = clock or _utcnow

    def _context(
        self, page_id: str, *, include_inactive: bool = False
    ) -> tuple[dict, list[dict], dict | None, date]:
        if not re.fullmatch(r"[0-9a-f]{64}", str(page_id or "")):
            raise SeoRecommendationError("not_found", "SEO page not found.")
        page = self.repository.get_page(page_id)
        if not page or (not include_inactive and not page.get("active", False)):
            raise SeoRecommendationError("not_found", "SEO page not found.")
        _, final_date = seo_data.date_window(self.clock())
        start = final_date - timedelta(days=27)
        metrics = self.repository.list_metrics(page_id, start, final_date)
        query_snapshot = self.repository.latest_query_snapshot(page_id)
        return page, metrics, query_snapshot, final_date

    @staticmethod
    def _serialize_judgment(entry: dict, *, created=False) -> dict:
        result = dict(entry)
        value = result.get("created_at")
        if isinstance(value, datetime):
            result["created_at"] = value.isoformat()
        result["created"] = created
        return result

    def generate(self, page_id: str, *, include_inactive: bool = False) -> dict:
        page, metrics, query_snapshot, final_date = self._context(
            page_id, include_inactive=include_inactive
        )
        return self.generate_from_context(
            page_id,
            page=page,
            metrics=metrics,
            query_snapshot=query_snapshot,
            final_date=final_date,
        )

    def generate_from_context(
        self,
        page_id: str,
        *,
        page: dict,
        metrics: list[dict],
        query_snapshot: dict | None,
        final_date: date,
    ) -> dict:
        """Persist a recommendation from already loaded SEO data.

        Portfolio reviews already need the page, its 28-day metrics and its
        query snapshot for the overview. Reusing that exact snapshot avoids a
        second set of Firestore reads per page and keeps the recommendation
        fingerprint tied to the displayed portfolio data.
        """
        if not re.fullmatch(r"[0-9a-f]{64}", str(page_id or "")):
            raise SeoRecommendationError("not_found", "SEO page not found.")
        if not page:
            raise SeoRecommendationError("not_found", "SEO page not found.")
        now = self.clock()
        result = deterministic_recommendation(
            page, metrics, query_snapshot, final_date=final_date, now=now
        )
        dossier_summary = seo_dossier.journal_summary(page)
        fingerprint_payload = {
            "page_id": page_id,
            "rule_version": RULE_VERSION,
            "data_window": result["data_window"],
            "dossier_summary": dossier_summary,
            "metrics": [
                {
                    key: row.get(key)
                    for key in ("date", "clicks", "impressions", "ctr", "position")
                }
                for row in metrics
            ],
            "query_snapshot": {
                key: (query_snapshot or {}).get(key)
                for key in (
                    "period_start", "period_end", "coverage", "partial",
                    "complete", "partial_reasons", "top_queries",
                )
            },
            "recommendation": result["recommendation"],
            "evidence": result["evidence"],
        }
        fingerprint = hashlib.sha256(
            json.dumps(fingerprint_payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        ).hexdigest()
        judgment_id = "det-" + fingerprint
        document = {
            "schema_version": 1,
            "page_id": page_id,
            "created_at": now,
            "rule_version": RULE_VERSION,
            "data_window": result["data_window"],
            "dossier_summary": dossier_summary,
            "status_class": result["status_class"],
            "recommendation": result["recommendation"],
            "confidence": result["confidence"],
            "evidence": result["evidence"],
            "review_after_days": result["review_after_days"],
            "requires_human_approval": True,
            "safeguards": result["safeguards"],
            "llm_evaluation": None,
            "user_feedback": None,
            "analysis_fingerprint": fingerprint,
        }
        stored, created = self.repository.append_judgment(judgment_id, document)
        return self._serialize_judgment(stored, created=created)

    def ask_content_judge(self, page_id: str) -> dict:
        deterministic = self.generate(page_id)
        if (
            deterministic.get("status_class") not in {"opportunity", "declining"}
            and deterministic.get("recommendation") != "noindex_candidate"
        ):
            raise SeoRecommendationError(
                "content_judge_not_applicable",
                "The content judge is limited to opportunity, declining, and safeguarded noindex candidates.",
            )
        page, _, query_snapshot, _ = self._context(page_id)
        dossier = page.get("dossier") if isinstance(page.get("dossier"), dict) else {}
        context = {
            "deterministic_judgement": {
                key: deterministic.get(key)
                for key in ("status_class", "recommendation", "confidence", "evidence", "data_window")
            },
            "dossier": {
                **seo_dossier.journal_summary(page),
                "content_representation": str(dossier.get("content_representation") or "")[:3_200],
            },
            "top_queries": list((query_snapshot or {}).get("top_queries") or [])[:20],
            "query_data_partial": bool((query_snapshot or {}).get("partial", True)),
        }
        llm_result = self.content_judge.ask(
            context,
            deterministic_recommendation_value=str(deterministic.get("recommendation") or ""),
        )
        now = self.clock()
        judgment_id = "llm-" + uuid.uuid4().hex
        document = {
            "schema_version": 1,
            "page_id": page_id,
            "created_at": now,
            "rule_version": RULE_VERSION,
            "data_window": deterministic.get("data_window") or {},
            "dossier_summary": seo_dossier.journal_summary(page),
            "status_class": deterministic.get("status_class"),
            "recommendation": deterministic.get("recommendation"),
            "confidence": deterministic.get("confidence"),
            "evidence": deterministic.get("evidence") or [],
            "review_after_days": deterministic.get("review_after_days"),
            "requires_human_approval": True,
            "safeguards": deterministic.get("safeguards") or {},
            "llm_evaluation": llm_result,
            "user_feedback": None,
            "parent_judgment_id": deterministic.get("judgment_id"),
        }
        stored, created = self.repository.append_judgment(judgment_id, document)
        return self._serialize_judgment(stored, created=created)
