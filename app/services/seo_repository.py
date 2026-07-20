"""Firestore persistence for SEO pages, metrics, queries, runs, and judgements."""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta, timezone
from typing import Iterable

from google.api_core.exceptions import AlreadyExists
from google.cloud.firestore_v1 import Query
from google.cloud.firestore_v1.base_query import FieldFilter


SEO_PAGES_COLLECTION = "seo_pages"
SEO_RUNS_COLLECTION = "seo_collection_runs"
DAILY_METRICS_SUBCOLLECTION = "daily_metrics"
QUERY_SNAPSHOTS_SUBCOLLECTION = "query_snapshots"
SEO_JUDGMENTS_COLLECTION = "seo_judgements"


def page_id_for_url(url: str) -> str:
    return hashlib.sha256(str(url).encode("utf-8")).hexdigest()


def _iso_datetime(value):
    return value.isoformat() if isinstance(value, datetime) else value or None


class FirestoreSeoRepository:
    def __init__(self, db):
        self.db = db

    def sync_pages(self, pages: list[dict], now: datetime) -> list[dict]:
        incoming_ids = {page_id_for_url(page["url"]) for page in pages}
        collection = self.db.collection(SEO_PAGES_COLLECTION)
        existing_by_id = {}
        writes = []
        for snapshot in collection.stream():
            data = snapshot.to_dict() or {}
            existing_by_id[snapshot.id] = data
            if snapshot.id not in incoming_ids and data.get("active", True):
                writes.append((
                    snapshot.reference,
                    {"active": False, "last_seen_at": now, "updated_at": now},
                ))

        normalized = []
        for page in pages:
            page_id = page_id_for_url(page["url"])
            ref = collection.document(page_id)
            existing_data = existing_by_id.get(page_id, {})
            record = {
                "schema_version": 2,
                "url": page["url"],
                "origin": page["origin"],
                "share_id": page.get("share_id"),
                "active": True,
                "indexable": True,
                "first_seen_at": existing_data.get("first_seen_at") or now,
                "last_seen_at": now,
                "updated_at": now,
            }
            for field in ("metrics_coverage_start", "metrics_coverage_end"):
                if existing_data.get(field):
                    record[field] = existing_data[field]
            dossier = page.get("dossier")
            if isinstance(dossier, dict):
                record["dossier"] = dossier
            writes.append((ref, record))
            normalized.append({**record, "page_id": page_id})
        if hasattr(self.db, "batch"):
            for offset in range(0, len(writes), 400):
                batch = self.db.batch()
                for ref, data in writes[offset:offset + 400]:
                    batch.set(ref, data, merge=True)
                batch.commit()
        else:
            for ref, data in writes:
                ref.set(data, merge=True)
        return normalized

    def set_metric_coverage(
        self, page_id: str, start_date: date, end_date: date, now: datetime
    ) -> None:
        """Persist a contiguous finalized-metrics coverage watermark."""
        self.db.collection(SEO_PAGES_COLLECTION).document(page_id).set(
            {
                "metrics_coverage_start": start_date.isoformat(),
                "metrics_coverage_end": end_date.isoformat(),
                "metrics_coverage_updated_at": now,
            },
            merge=True,
        )

    def set_metric_coverages(
        self, page_ids: Iterable[str], start_date: date, end_date: date, now: datetime
    ) -> None:
        """Persist the same contiguous coverage watermark with bounded batches."""
        page_ids = list(dict.fromkeys(page_ids))
        if not page_ids:
            return
        payload = {
            "metrics_coverage_start": start_date.isoformat(),
            "metrics_coverage_end": end_date.isoformat(),
            "metrics_coverage_updated_at": now,
        }
        if hasattr(self.db, "batch"):
            for offset in range(0, len(page_ids), 400):
                batch = self.db.batch()
                for page_id in page_ids[offset:offset + 400]:
                    ref = self.db.collection(SEO_PAGES_COLLECTION).document(page_id)
                    batch.set(ref, payload, merge=True)
                batch.commit()
            return
        for page_id in page_ids:
            self.set_metric_coverage(page_id, start_date, end_date, now)

    def list_pages(self, *, active_only: bool = False) -> list[dict]:
        pages = []
        for snapshot in self.db.collection(SEO_PAGES_COLLECTION).stream():
            data = snapshot.to_dict() or {}
            if active_only and not data.get("active", False):
                continue
            pages.append({**data, "page_id": snapshot.id})
        pages.sort(key=lambda item: item.get("url") or "")
        return pages

    def get_page(self, page_id: str) -> dict | None:
        snapshot = self.db.collection(SEO_PAGES_COLLECTION).document(page_id).get()
        if not snapshot.exists:
            return None
        return {**(snapshot.to_dict() or {}), "page_id": snapshot.id}

    def existing_metric_dates(
        self, page_id: str, start_date: date, end_date: date
    ) -> set[str]:
        metrics = (
            self.db.collection(SEO_PAGES_COLLECTION)
            .document(page_id)
            .collection(DAILY_METRICS_SUBCOLLECTION)
        )
        dates = set()
        for snapshot in self._metric_range_stream(metrics, start_date, end_date):
            data = snapshot.to_dict() or {}
            value = str(data.get("date") or snapshot.id)
            if start_date.isoformat() <= value <= end_date.isoformat():
                dates.add(value)
        return dates

    def existing_metric_dates_for_pages(
        self, page_ids: Iterable[str], start_date: date, end_date: date
    ) -> dict[str, set[str]]:
        """Batch-read exact daily documents for several pages when supported."""
        page_ids = list(dict.fromkeys(page_ids))
        result = {page_id: set() for page_id in page_ids}
        if not page_ids:
            return result
        if not hasattr(self.db, "get_all"):
            return {
                page_id: self.existing_metric_dates(page_id, start_date, end_date)
                for page_id in page_ids
            }
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current.isoformat())
            current += timedelta(days=1)
        refs = []
        page_by_ref_path = {}
        for page_id in page_ids:
            for day in dates:
                ref = (
                    self.db.collection(SEO_PAGES_COLLECTION)
                    .document(page_id)
                    .collection(DAILY_METRICS_SUBCOLLECTION)
                    .document(day)
                )
                refs.append(ref)
                page_by_ref_path[str(getattr(ref, "path", ""))] = page_id
        for offset in range(0, len(refs), 400):
            for snapshot in self.db.get_all(refs[offset:offset + 400]):
                if not snapshot.exists:
                    continue
                data = snapshot.to_dict() or {}
                page_id = str(data.get("page_id") or page_by_ref_path.get(
                    str(getattr(snapshot.reference, "path", "")), ""
                ))
                if not page_id:
                    try:
                        page_id = snapshot.reference.parent.parent.id
                    except AttributeError:
                        page_id = ""
                day = str(data.get("date") or snapshot.id)
                if page_id in result and start_date.isoformat() <= day <= end_date.isoformat():
                    result[page_id].add(day)
        return result

    def upsert_metrics(self, metrics: Iterable[dict]) -> int:
        prepared = list(metrics)
        count = len(prepared)
        if not prepared:
            return 0
        if hasattr(self.db, "batch"):
            for offset in range(0, count, 400):
                batch = self.db.batch()
                for metric in prepared[offset:offset + 400]:
                    ref = (
                        self.db.collection(SEO_PAGES_COLLECTION)
                        .document(metric["page_id"])
                        .collection(DAILY_METRICS_SUBCOLLECTION)
                        .document(metric["date"])
                    )
                    batch.set(
                        ref,
                        {k: v for k, v in metric.items() if k != "page_id"},
                        merge=True,
                    )
                batch.commit()
            return count

        # Lightweight Firestore mocks used by unit tests need no batch API.
        for metric in prepared:
            ref = (
                self.db.collection(SEO_PAGES_COLLECTION)
                .document(metric["page_id"])
                .collection(DAILY_METRICS_SUBCOLLECTION)
                .document(metric["date"])
            )
            ref.set({k: v for k, v in metric.items() if k != "page_id"}, merge=True)
        return count

    def list_metrics(self, page_id: str, start_date: date, end_date: date) -> list[dict]:
        metrics = []
        collection = (
            self.db.collection(SEO_PAGES_COLLECTION)
            .document(page_id)
            .collection(DAILY_METRICS_SUBCOLLECTION)
        )
        for snapshot in self._metric_range_stream(collection, start_date, end_date):
            data = snapshot.to_dict() or {}
            value = str(data.get("date") or snapshot.id)
            if start_date.isoformat() <= value <= end_date.isoformat():
                metrics.append(data)
        metrics.sort(key=lambda item: item.get("date") or "")
        return metrics

    def list_metrics_for_pages(
        self, page_ids: Iterable[str], start_date: date, end_date: date
    ) -> dict[str, list[dict]]:
        """Load exact 28-day documents in bounded BatchGet requests."""
        page_ids = list(dict.fromkeys(page_ids))
        result = {page_id: [] for page_id in page_ids}
        if not page_ids:
            return result
        if not hasattr(self.db, "get_all"):
            return {
                page_id: self.list_metrics(page_id, start_date, end_date)
                for page_id in page_ids
            }
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current.isoformat())
            current += timedelta(days=1)
        refs = []
        page_by_ref_path = {}
        for page_id in page_ids:
            for day in dates:
                ref = (
                    self.db.collection(SEO_PAGES_COLLECTION)
                    .document(page_id)
                    .collection(DAILY_METRICS_SUBCOLLECTION)
                    .document(day)
                )
                refs.append(ref)
                page_by_ref_path[str(getattr(ref, "path", ""))] = page_id
        for offset in range(0, len(refs), 400):
            for snapshot in self.db.get_all(refs[offset:offset + 400]):
                if not snapshot.exists:
                    continue
                data = snapshot.to_dict() or {}
                page_id = str(data.get("page_id") or page_by_ref_path.get(
                    str(getattr(snapshot.reference, "path", "")), ""
                ))
                if not page_id:
                    try:
                        page_id = snapshot.reference.parent.parent.id
                    except AttributeError:
                        page_id = ""
                if page_id in result:
                    result[page_id].append(data)
        for metrics in result.values():
            metrics.sort(key=lambda item: item.get("date") or "")
        return result

    def has_metrics(self, page_id: str) -> bool:
        collection = (
            self.db.collection(SEO_PAGES_COLLECTION)
            .document(page_id)
            .collection(DAILY_METRICS_SUBCOLLECTION)
        )
        if hasattr(collection, "limit"):
            return any(collection.limit(1).stream())
        return any(collection.stream())

    def get_query_snapshot(self, page_id: str, final_date: date | str) -> dict | None:
        snapshot = (
            self.db.collection(SEO_PAGES_COLLECTION)
            .document(page_id)
            .collection(QUERY_SNAPSHOTS_SUBCOLLECTION)
            .document(str(final_date))
            .get()
        )
        if not snapshot.exists:
            return None
        return {**(snapshot.to_dict() or {}), "snapshot_id": snapshot.id}

    def save_query_snapshot(self, page_id: str, final_date: date | str, data: dict) -> None:
        (
            self.db.collection(SEO_PAGES_COLLECTION)
            .document(page_id)
            .collection(QUERY_SNAPSHOTS_SUBCOLLECTION)
            .document(str(final_date))
            .set(data)
        )

    def latest_query_snapshot(self, page_id: str) -> dict | None:
        collection = (
            self.db.collection(SEO_PAGES_COLLECTION)
            .document(page_id)
            .collection(QUERY_SNAPSHOTS_SUBCOLLECTION)
        )
        if hasattr(collection, "order_by"):
            stream = (
                collection.order_by("period_end", direction=Query.DESCENDING)
                .limit(1)
                .stream()
            )
            snapshot = next(iter(stream), None)
            if snapshot is None:
                return None
            return {**(snapshot.to_dict() or {}), "snapshot_id": snapshot.id}

        # Lightweight unit-test doubles do not expose Firestore query methods.
        snapshots = []
        for snapshot in collection.stream():
            data = snapshot.to_dict() or {}
            snapshots.append({**data, "snapshot_id": snapshot.id})
        if not snapshots:
            return None
        snapshots.sort(
            key=lambda item: str(item.get("period_end") or item.get("snapshot_id") or ""),
            reverse=True,
        )
        return snapshots[0]

    @staticmethod
    def _metric_range_stream(collection, start_date: date, end_date: date):
        """Use a bounded Firestore query, with a small mock-compatible fallback."""
        if hasattr(collection, "where"):
            return (
                collection
                .where(filter=FieldFilter("date", ">=", start_date.isoformat()))
                .where(filter=FieldFilter("date", "<=", end_date.isoformat()))
                .stream()
            )
        return collection.stream()

    def create_run(self, run_id: str, data: dict) -> None:
        self.db.collection(SEO_RUNS_COLLECTION).document(run_id).set(data)

    def update_run(self, run_id: str, data: dict) -> None:
        self.db.collection(SEO_RUNS_COLLECTION).document(run_id).set(data, merge=True)

    def last_run(self) -> dict | None:
        collection = self.db.collection(SEO_RUNS_COLLECTION)
        if hasattr(collection, "order_by"):
            stream = (
                collection.order_by("started_at", direction=Query.DESCENDING)
                .limit(1)
                .stream()
            )
            snapshot = next(iter(stream), None)
            if snapshot is None:
                return None
            run = {**(snapshot.to_dict() or {}), "run_id": snapshot.id}
        else:
            # Lightweight unit-test doubles do not expose Firestore query methods.
            runs = []
            for snapshot in collection.stream():
                data = snapshot.to_dict() or {}
                runs.append({**data, "run_id": snapshot.id})
            if not runs:
                return None

            def sort_key(item):
                value = item.get("started_at")
                if not isinstance(value, datetime):
                    return float("-inf")
                if value.tzinfo is None:
                    return value.replace(tzinfo=timezone.utc).timestamp()
                return value.timestamp()

            runs.sort(key=sort_key, reverse=True)
            run = dict(runs[0])
        for field in ("started_at", "finished_at"):
            run[field] = _iso_datetime(run.get(field))
        return run

    def append_judgment(self, judgment_id: str, data: dict) -> tuple[dict, bool]:
        """Create one immutable journal entry and return an existing retry unchanged."""
        ref = self.db.collection(SEO_JUDGMENTS_COLLECTION).document(judgment_id)
        existing = ref.get()
        if existing.exists:
            return {**(existing.to_dict() or {}), "judgment_id": judgment_id}, False
        try:
            if hasattr(ref, "create"):
                ref.create(data)
            else:
                ref.set(data)
        except AlreadyExists:
            existing = ref.get()
            return {**(existing.to_dict() or {}), "judgment_id": judgment_id}, False
        return {**data, "judgment_id": judgment_id}, True

    def list_judgments(
        self, page_ids: Iterable[str], *, max_per_page: int = 3, max_scan: int = 1000
    ) -> dict[str, list[dict]]:
        wanted = set(page_ids)
        grouped = {page_id: [] for page_id in wanted}
        collection = self.db.collection(SEO_JUDGMENTS_COLLECTION)
        if hasattr(collection, "order_by"):
            stream = (
                collection.order_by("created_at", direction=Query.DESCENDING)
                .limit(max_scan)
                .stream()
            )
        else:
            stream = collection.stream()
        for snapshot in stream:
            data = snapshot.to_dict() or {}
            page_id = str(data.get("page_id") or "")
            if page_id in wanted:
                grouped[page_id].append({**data, "judgment_id": snapshot.id})

        def timestamp(item):
            value = item.get("created_at")
            if isinstance(value, datetime):
                if value.tzinfo is None:
                    value = value.replace(tzinfo=timezone.utc)
                return value.timestamp()
            return float("-inf")

        for page_id, entries in grouped.items():
            entries.sort(key=timestamp, reverse=True)
            del entries[max_per_page:]
            for entry in entries:
                entry["created_at"] = _iso_datetime(entry.get("created_at"))
        return grouped
