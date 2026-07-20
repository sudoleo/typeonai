"""Collection and read-only aggregation for the first SEO data foundation."""

from __future__ import annotations

import threading
import uuid
import re
from datetime import date, datetime, timedelta, timezone
from urllib.parse import urlsplit, urlunsplit

from app.services import google_search_console as gsc
from app.services import seo_dossier
from app.services import share_snapshots
from app.services.seo_repository import FirestoreSeoRepository


FINALIZATION_LAG_DAYS = 3
HISTORY_DAYS = 90
MAX_QUERY_RANGE_DAYS = 31
TOP_QUERIES_PER_PAGE = 20
MAX_QUERY_PAGES_PER_RUN = 100
DATA_SOURCE = "google_search_console"
DISCLAIMER = (
    "Search Console rows can be incomplete and arrive with a delay; only final data "
    "through the displayed final date is collected."
)


class CollectionAlreadyRunning(Exception):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def normalize_url(value: str) -> str:
    parts = urlsplit(str(value or "").strip())
    path = parts.path or "/"
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, "", ""))


def discover_indexable_pages() -> list[dict]:
    # Lazy import avoids making the heavy pages/router import part of every
    # service import while keeping SITEMAP_URLS the single static-page source.
    from app.api.routers.pages import SITE_URL, SITEMAP_URLS

    pages = [{
        "url": normalize_url(item["loc"]),
        "origin": "static_page",
        "share_id": None,
        "dossier": seo_dossier.build_static_dossier(item["loc"], item.get("lastmod")),
    } for item in SITEMAP_URLS]
    for item in share_snapshots.list_indexed_share_urls():
        path = str(item.get("path") or "")
        share_id = path.rsplit("-", 1)[-1] if path else None
        pages.append({
            "url": normalize_url(SITE_URL + path),
            "origin": "share",
            "share_id": share_id,
            "dossier": seo_dossier.build_share_dossier(share_id) if share_id else {},
        })
    deduplicated = {page["url"]: page for page in pages if page["url"]}
    return [deduplicated[url] for url in sorted(deduplicated)]


def date_window(now: datetime) -> tuple[date, date]:
    final_date = now.astimezone(timezone.utc).date() - timedelta(days=FINALIZATION_LAG_DAYS)
    return final_date - timedelta(days=HISTORY_DAYS - 1), final_date


def _dates_between(start_date: date, end_date: date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def _query_ranges(days: set[date]) -> list[tuple[date, date]]:
    if not days:
        return []
    ordered = sorted(days)
    ranges = []
    start = previous = ordered[0]
    for current in ordered[1:]:
        contiguous = current == previous + timedelta(days=1)
        within_cap = (current - start).days < MAX_QUERY_RANGE_DAYS
        if not contiguous or not within_cap:
            ranges.append((start, previous))
            start = current
        previous = current
    ranges.append((start, previous))
    return ranges


def _coerce_number(value, default=0.0) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return default


_EMAIL_RE = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def sanitize_query_text(value) -> str | None:
    """Drop query rows that look like personal/contact identifiers."""
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", str(value or ""))
    text = re.sub(r"\s+", " ", text).strip()
    if not text or len(text) > 160:
        return None
    if _EMAIL_RE.search(text) or _PHONE_RE.search(text) or _IP_RE.search(text):
        return None
    return text


def aggregate_metrics(metrics: list[dict]) -> dict:
    clicks = sum(_coerce_number(item.get("clicks")) for item in metrics)
    impressions = sum(_coerce_number(item.get("impressions")) for item in metrics)
    weighted_position = sum(
        _coerce_number(item.get("position")) * _coerce_number(item.get("impressions"))
        for item in metrics
        if item.get("position") is not None
    )
    positioned_impressions = sum(
        _coerce_number(item.get("impressions"))
        for item in metrics
        if item.get("position") is not None
    )
    return {
        "clicks": round(clicks, 3),
        "impressions": round(impressions, 3),
        "ctr": round(clicks / impressions, 6) if impressions else 0.0,
        "position": round(weighted_position / positioned_impressions, 3)
        if positioned_impressions else None,
        "days": len(metrics),
    }


STATUS_RULES = {
    "insufficient_data": "fewer than 7 finalized daily rows",
    "invisible": "at least 7 finalized rows and 0 impressions over 28 days",
    "winner": "at least 100 impressions, 10 clicks, 5% CTR and average position <= 10 over 28 days",
    "opportunity": "at least 100 impressions and position <= 20 with CTR < 3% over 28 days",
    "declining": "previous 7 days had at least 20 impressions and recent clicks or impressions fell by at least 40%",
    "emerging": "visible page that does not match winner, opportunity or declining",
}


def classify_status(metrics_28: list[dict], final_date: date) -> str:
    summary = aggregate_metrics(metrics_28)
    if summary["days"] < 7:
        return "insufficient_data"
    if summary["impressions"] == 0:
        return "invisible"

    recent_start = final_date - timedelta(days=6)
    previous_start = final_date - timedelta(days=13)
    recent = aggregate_metrics([
        row for row in metrics_28 if str(row.get("date") or "") >= recent_start.isoformat()
    ])
    previous = aggregate_metrics([
        row for row in metrics_28
        if previous_start.isoformat() <= str(row.get("date") or "") < recent_start.isoformat()
    ])
    if previous["impressions"] >= 20 and (
        recent["impressions"] <= previous["impressions"] * 0.6
        or (previous["clicks"] >= 2 and recent["clicks"] <= previous["clicks"] * 0.6)
    ):
        return "declining"
    if (
        summary["impressions"] >= 100
        and summary["clicks"] >= 10
        and summary["ctr"] >= 0.05
        and summary["position"] is not None
        and summary["position"] <= 10
    ):
        return "winner"
    if (
        summary["impressions"] >= 100
        and summary["position"] is not None
        and summary["position"] <= 20
        and summary["ctr"] < 0.03
    ):
        return "opportunity"
    return "emerging"


class SeoDataService:
    def __init__(
        self,
        db=None,
        *,
        repository=None,
        client_factory=None,
        page_discovery=None,
        clock=None,
    ):
        self.repository = repository or FirestoreSeoRepository(db)
        self.client_factory = client_factory or gsc.GoogleSearchConsoleClient.from_env
        self.page_discovery = page_discovery or discover_indexable_pages
        self.clock = clock or utcnow
        self._collection_lock = threading.Lock()

    def check_connection(self) -> dict:
        return gsc.connection_status()

    def collect(self) -> dict:
        if not self._collection_lock.acquire(blocking=False):
            raise CollectionAlreadyRunning()
        run_id = uuid.uuid4().hex
        started_at = self.clock()
        start_date, end_date = date_window(started_at)
        self.repository.create_run(run_id, {
            "schema_version": 1,
            "status": "running",
            "source": DATA_SOURCE,
            "started_at": started_at,
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
        })
        try:
            try:
                client = self.client_factory()
            except gsc.SearchConsoleError as exc:
                return self._finish_non_success(
                    run_id, started_at, exc.code, exc.safe_message, start_date, end_date
                )

            pages = self.repository.sync_pages(self.page_discovery(), started_at)
            all_days = set(_dates_between(start_date, end_date))
            missing_by_page: dict[str, set[date]] = {}
            for page in pages:
                existing = self.repository.existing_metric_dates(
                    page["page_id"], start_date, end_date
                )
                missing_by_page[page["page_id"]] = {
                    day for day in all_days if day.isoformat() not in existing
                }
            missing_days = set().union(*missing_by_page.values()) if missing_by_page else set()

            rows_by_key = {}
            requests_made = 0
            gsc_rows = 0
            truncated = False
            complete_days: set[date] = set()
            eligible_by_url = {normalize_url(page["url"]): page for page in pages}
            for query_start, query_end in _query_ranges(missing_days):
                result = client.query(query_start, query_end)
                requests_made += int(result.get("requests") or 0)
                range_truncated = bool(result.get("truncated"))
                truncated = truncated or range_truncated
                if not range_truncated:
                    complete_days.update(_dates_between(query_start, query_end))
                for row in result.get("rows") or []:
                    keys = row.get("keys") if isinstance(row.get("keys"), list) else []
                    if len(keys) < 2:
                        continue
                    url = normalize_url(keys[0])
                    day_value = str(keys[1])
                    page = eligible_by_url.get(url)
                    if not page:
                        continue
                    try:
                        day = date.fromisoformat(day_value)
                    except ValueError:
                        continue
                    if day not in missing_by_page[page["page_id"]]:
                        continue
                    rows_by_key[(page["page_id"], day)] = row
                    gsc_rows += 1

            collected_at = self.clock()
            metrics = []
            for page in pages:
                for day in sorted(missing_by_page[page["page_id"]]):
                    row = rows_by_key.get((page["page_id"], day))
                    # A capped response is incomplete. Persist rows that were
                    # actually returned, but never turn omitted rows into
                    # zeroes; those URL/days stay missing and are retried.
                    if row is None and day not in complete_days:
                        continue
                    row = row or {}
                    clicks = _coerce_number(row.get("clicks"))
                    impressions = _coerce_number(row.get("impressions"))
                    position = row.get("position")
                    position = _coerce_number(position) if position is not None else None
                    metrics.append({
                        "page_id": page["page_id"],
                        "schema_version": 1,
                        "url": page["url"],
                        "date": day.isoformat(),
                        "clicks": clicks,
                        "impressions": impressions,
                        "ctr": _coerce_number(row.get("ctr"), clicks / impressions if impressions else 0.0),
                        "position": position,
                        "collected_at": collected_at,
                        "source": DATA_SOURCE,
                        "origin": page["origin"],
                        "share_id": page.get("share_id"),
                    })
            written = self.repository.upsert_metrics(metrics)
            query_start = end_date - timedelta(days=27)
            query_pages_collected = 0
            query_pages_skipped = 0
            query_failures = 0
            query_partial_pages = 0
            query_supported = hasattr(client, "query_page_queries")
            for page in pages:
                if not query_supported:
                    break
                if self.repository.get_query_snapshot(page["page_id"], end_date):
                    continue
                if query_pages_collected >= MAX_QUERY_PAGES_PER_RUN:
                    query_pages_skipped += 1
                    continue
                try:
                    query_result = client.query_page_queries(
                        query_start, end_date, page["url"], limit=TOP_QUERIES_PER_PAGE
                    )
                    requests_made += int(query_result.get("requests") or 0)
                except gsc.SearchConsoleError:
                    query_failures += 1
                    continue

                top_queries = []
                redacted_queries = 0
                for row in query_result.get("rows") or []:
                    keys = row.get("keys") if isinstance(row.get("keys"), list) else []
                    query_text = sanitize_query_text(keys[0] if keys else "")
                    if not query_text:
                        redacted_queries += 1
                        continue
                    clicks = _coerce_number(row.get("clicks"))
                    impressions = _coerce_number(row.get("impressions"))
                    position = row.get("position")
                    top_queries.append({
                        "query": query_text,
                        "clicks": clicks,
                        "impressions": impressions,
                        "ctr": _coerce_number(
                            row.get("ctr"), clicks / impressions if impressions else 0.0
                        ),
                        "position": _coerce_number(position) if position is not None else None,
                    })
                partial_reasons = []
                if query_result.get("truncated"):
                    partial_reasons.append("row_cap")
                if redacted_queries:
                    partial_reasons.append("privacy_filter")
                is_partial = bool(partial_reasons)
                if is_partial:
                    query_partial_pages += 1
                self.repository.save_query_snapshot(page["page_id"], end_date, {
                    "schema_version": 1,
                    "period_start": query_start.isoformat(),
                    "period_end": end_date.isoformat(),
                    "data_state": "final",
                    "coverage": query_result.get("coverage") or "top_queries_only",
                    "row_limit": TOP_QUERIES_PER_PAGE,
                    "top_queries": top_queries[:TOP_QUERIES_PER_PAGE],
                    "partial": is_partial,
                    "complete": not is_partial,
                    "partial_reasons": partial_reasons,
                    "redacted_query_rows": redacted_queries,
                    "collected_at": collected_at,
                    "source": DATA_SOURCE,
                })
                query_pages_collected += 1

            collection_partial = bool(
                truncated or query_pages_skipped or query_failures or query_partial_pages
            )
            status = "partial" if collection_partial else "success"
            result = {
                "status": status,
                "run_id": run_id,
                "period_start": start_date.isoformat(),
                "period_end": end_date.isoformat(),
                "eligible_urls": len(pages),
                "days_requested": len(missing_days),
                "days_collected": len(complete_days),
                "metrics_written": written,
                "gsc_rows_matched": gsc_rows,
                "requests_made": requests_made,
                "truncated": truncated,
                "query_pages_collected": query_pages_collected,
                "query_pages_skipped": query_pages_skipped,
                "query_failures": query_failures,
                "query_partial_pages": query_partial_pages,
                "message": "Collection completed with explicitly marked partial data."
                if collection_partial else "Collection completed.",
            }
            self.repository.update_run(run_id, {**result, "finished_at": collected_at})
            return result
        except gsc.SearchConsoleError as exc:
            return self._finish_non_success(
                run_id, started_at, "error", exc.safe_message, start_date, end_date
            )
        except Exception:
            self._finish_non_success(
                run_id,
                started_at,
                "error",
                "SEO collection failed safely. Check server diagnostics.",
                start_date,
                end_date,
            )
            raise
        finally:
            self._collection_lock.release()

    def _finish_non_success(self, run_id, started_at, status, message, start_date, end_date):
        result = {
            "status": status,
            "run_id": run_id,
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
            "message": message,
        }
        self.repository.update_run(run_id, {**result, "finished_at": self.clock()})
        return result

    def overview(self) -> dict:
        now = self.clock()
        _, final_date = date_window(now)
        start_28 = final_date - timedelta(days=27)
        start_7 = final_date - timedelta(days=6)
        pages = self.repository.list_pages(active_only=True)
        histories = self.repository.list_judgments(
            [page["page_id"] for page in pages], max_per_page=3
        )
        rows = []
        captured = 0
        for page in pages:
            metrics = self.repository.list_metrics(page["page_id"], start_28, final_date)
            if metrics or self.repository.has_metrics(page["page_id"]):
                captured += 1
            metrics_7 = [
                item for item in metrics if str(item.get("date") or "") >= start_7.isoformat()
            ]
            rows.append({
                "page_id": page.get("page_id"),
                "url": page.get("url"),
                "origin": page.get("origin"),
                "share_id": page.get("share_id"),
                "dossier": seo_dossier.journal_summary(page),
                "query_data": self.repository.latest_query_snapshot(page["page_id"]),
                "metrics_7d": aggregate_metrics(metrics_7),
                "metrics_28d": aggregate_metrics(metrics),
                "status": classify_status(metrics, final_date),
                "recommendation_history": histories.get(page["page_id"], []),
            })
        rows.sort(key=lambda item: (-item["metrics_28d"]["clicks"], item["url"] or ""))
        return {
            "configuration": gsc.configuration_status(),
            "last_run": self.repository.last_run(),
            "captured_urls": captured,
            "eligible_urls": len(pages),
            "final_date": final_date.isoformat(),
            "rows": rows,
            "status_rules": STATUS_RULES,
            "disclaimer": DISCLAIMER,
        }
