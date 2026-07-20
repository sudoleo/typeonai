"""Firestore persistence for SEO pages, daily metrics, and collection runs."""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timezone
from typing import Iterable

from google.cloud.firestore_v1.base_query import FieldFilter


SEO_PAGES_COLLECTION = "seo_pages"
SEO_RUNS_COLLECTION = "seo_collection_runs"
DAILY_METRICS_SUBCOLLECTION = "daily_metrics"


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
        for snapshot in collection.stream():
            data = snapshot.to_dict() or {}
            if snapshot.id not in incoming_ids and data.get("active", True):
                snapshot.reference.set(
                    {"active": False, "last_seen_at": now, "updated_at": now}, merge=True
                )

        normalized = []
        for page in pages:
            page_id = page_id_for_url(page["url"])
            ref = collection.document(page_id)
            existing = ref.get()
            existing_data = existing.to_dict() if existing.exists else {}
            record = {
                "schema_version": 1,
                "url": page["url"],
                "origin": page["origin"],
                "share_id": page.get("share_id"),
                "active": True,
                "indexable": True,
                "first_seen_at": existing_data.get("first_seen_at") or now,
                "last_seen_at": now,
                "updated_at": now,
            }
            ref.set(record, merge=True)
            normalized.append({**record, "page_id": page_id})
        return normalized

    def list_pages(self, *, active_only: bool = False) -> list[dict]:
        pages = []
        for snapshot in self.db.collection(SEO_PAGES_COLLECTION).stream():
            data = snapshot.to_dict() or {}
            if active_only and not data.get("active", False):
                continue
            pages.append({**data, "page_id": snapshot.id})
        pages.sort(key=lambda item: item.get("url") or "")
        return pages

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

    def has_metrics(self, page_id: str) -> bool:
        collection = (
            self.db.collection(SEO_PAGES_COLLECTION)
            .document(page_id)
            .collection(DAILY_METRICS_SUBCOLLECTION)
        )
        if hasattr(collection, "limit"):
            return any(collection.limit(1).stream())
        return any(collection.stream())

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
        runs = []
        for snapshot in self.db.collection(SEO_RUNS_COLLECTION).stream():
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
