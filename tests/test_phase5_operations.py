import asyncio
import json
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import anyio
import pytest
import requests
from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI
from fastapi.testclient import TestClient
from firebase_admin import firestore

from app.api.routers import pages as pages_router
from app.api.routers import topics as topics_router
from app.api.routers import chat as chat_router
from app.core.rate_limit import limiter
from app.core.observability import CorrelationMiddleware, metrics_snapshot, record_metric
from app.services import favicons, follow_challenges, persistence_guard, watch_brief
from app.services.llm import consensus_engine, engines, provider_transport
from app.services.llm.provider_runtime import managed_provider_resource


class Snapshot:
    def __init__(self, ref, data):
        self.reference = ref
        self.id = ref.id
        self._data = None if data is None else dict(data)
        self.exists = data is not None

    def to_dict(self):
        return None if self._data is None else dict(self._data)


class Document:
    def __init__(self, db, path):
        self.db = db
        self.path = tuple(path)
        self.id = self.path[-1]

    def collection(self, name):
        return Collection(self.db, self.path + (name,))

    def get(self, transaction=None):
        return Snapshot(self, self.db.data.get(self.path))

    def set(self, data, merge=False):
        current = dict(self.db.data.get(self.path) or {}) if merge else {}
        for key, value in dict(data).items():
            if value is firestore.SERVER_TIMESTAMP:
                current[key] = datetime.now(timezone.utc)
            elif type(value).__name__ == "Increment":
                amount = getattr(value, "value", 1)
                current[key] = int(current.get(key) or 0) + int(amount)
            elif merge and isinstance(value, dict) and isinstance(current.get(key), dict):
                current[key] = {**current[key], **value}
            else:
                current[key] = value
        self.db.data[self.path] = current

    def update(self, data):
        self.set(data, merge=True)

    def delete(self):
        self.db.data.pop(self.path, None)


class Query:
    def __init__(self, collection, filters=(), order=None, limit_value=None):
        self.collection = collection
        self.filters = list(filters)
        self.order = order
        self.limit_value = limit_value

    def where(self, *args, filter=None):
        if filter is not None:
            condition = (filter.field_path, filter.op_string, filter.value)
        else:
            condition = tuple(args)
        return Query(self.collection, self.filters + [condition], self.order, self.limit_value)

    def order_by(self, field, direction=None):
        return Query(self.collection, self.filters, (field, direction), self.limit_value)

    def limit(self, value):
        return Query(self.collection, self.filters, self.order, value)

    def stream(self):
        docs = self.collection.stream()
        for field, operator, expected in self.filters:
            def matches(snapshot):
                actual = (snapshot.to_dict() or {}).get(field)
                return actual == expected if operator == "==" else actual <= expected
            docs = [doc for doc in docs if matches(doc)]
        if self.order:
            docs.sort(key=lambda doc: (doc.to_dict() or {}).get(self.order[0]))
        return docs[: self.limit_value] if self.limit_value is not None else docs


class Collection:
    def __init__(self, db, path):
        self.db = db
        self.path = tuple(path)

    def document(self, doc_id=None):
        if doc_id is None:
            self.db.sequence += 1
            doc_id = f"auto-{self.db.sequence}"
        return Document(self.db, self.path + (str(doc_id),))

    def stream(self):
        return [
            Snapshot(Document(self.db, path), data)
            for path, data in self.db.data.items()
            if len(path) == len(self.path) + 1 and path[:-1] == self.path
        ]

    def where(self, *args, filter=None):
        return Query(self).where(*args, filter=filter)


class Transaction:
    def get(self, ref):
        # Match google-cloud-firestore 2.20.x: Transaction.get() is a
        # generator, not a DocumentSnapshot.  Production code must use
        # DocumentReference.get(transaction=transaction) for single-document
        # reads.
        yield ref.get()

    def set(self, ref, data, merge=False):
        ref.set(data, merge=merge)

    def update(self, ref, data):
        ref.update(data)

    def delete(self, ref):
        ref.delete()


class Database:
    def __init__(self):
        self.data = {}
        self.sequence = 0
        self.lock = threading.Lock()

    def collection(self, name):
        return Collection(self, (name,))

    def run_transaction(self, operation):
        with self.lock:
            return operation(Transaction())


def test_bookmark_quota_counts_merges_and_rejects_oversize(monkeypatch):
    db = Database()
    ref = db.collection("users").document("u1").collection("bookmarks").document("b1")
    first = persistence_guard.write_bookmark(
        uid="u1", doc_ref=ref,
        patch={
            "query": "Q",
            "responses": {"OpenAI": "A"},
            "timestamp": firestore.SERVER_TIMESTAMP,
        },
        db=db,
    )
    second = persistence_guard.write_bookmark(
        uid="u1", doc_ref=ref, patch={"responses": {"Gemini": "B"}}, db=db
    )
    assert first["query"] == "Q"
    assert isinstance(first["timestamp"], datetime)
    assert jsonable_encoder(first)["timestamp"] == first["timestamp"].isoformat()
    assert second["responses"] == {"OpenAI": "A", "Gemini": "B"}
    usage = next(
        data for path, data in db.data.items()
        if path[0] == persistence_guard.USAGE_COLLECTION
    )
    assert usage["bookmark_count"] == 1

    monkeypatch.setattr(persistence_guard, "MAX_BOOKMARK_DOCUMENT_BYTES", 20)
    with pytest.raises(persistence_guard.PersistenceLimitError, match="too large"):
        persistence_guard.write_bookmark(
            uid="u1", doc_ref=ref, patch={"responses": {"OpenAI": "x" * 100}}, db=db
        )


def test_bookmark_delete_updates_quota_with_firestore_transaction_contract():
    db = Database()
    ref = db.collection("users").document("u1").collection("bookmarks").document("b1")
    persistence_guard.write_bookmark(
        uid="u1", doc_ref=ref, patch={"query": "Q", "responses": {"OpenAI": "A"}}, db=db
    )

    deleted = persistence_guard.delete_bookmark(uid="u1", doc_ref=ref, db=db)

    usage = next(
        data for path, data in db.data.items()
        if path[0] == persistence_guard.USAGE_COLLECTION
    )
    assert deleted["query"] == "Q"
    assert ref.path not in db.data
    assert usage["bookmark_count"] == 0
    assert usage["bookmark_bytes"] == 0


def test_account_deletion_tombstone_fences_normal_bookmark_delete():
    db = Database()
    ref = db.collection("users").document("u1").collection("bookmarks").document("b1")
    persistence_guard.write_bookmark(
        uid="u1", doc_ref=ref, patch={"query": "private"}, db=db
    )
    db.data[(persistence_guard.ACCOUNT_DELETION_JOBS_COLLECTION, "u1")] = {
        "status": "pending"
    }
    before = {path: dict(data) for path, data in db.data.items()}

    with pytest.raises(persistence_guard.AccountDeletionInProgress):
        persistence_guard.delete_bookmark(uid="u1", doc_ref=ref, db=db)

    assert db.data == before


@pytest.mark.parametrize("operation", ["bookmark", "feedback", "vote"])
def test_account_deletion_tombstone_fences_owner_persistence(operation):
    db = Database()
    db.data[(persistence_guard.ACCOUNT_DELETION_JOBS_COLLECTION, "u1")] = {
        "status": "pending"
    }
    now = datetime(2026, 8, 11, tzinfo=timezone.utc)
    before = dict(db.data)

    with pytest.raises(persistence_guard.AccountDeletionInProgress):
        if operation == "bookmark":
            ref = (
                db.collection("users").document("u1").collection("bookmarks")
                .document("b1")
            )
            persistence_guard.write_bookmark(
                uid="u1", doc_ref=ref, patch={"query": "private"}, db=db
            )
        elif operation == "feedback":
            persistence_guard.create_feedback(
                uid="u1",
                feedback={"message": "private", "uid": "u1", "timestamp": now},
                db=db,
                now=now,
            )
        else:
            result_id = "R" * 16
            db.data[("pending_results", result_id)] = {
                "owner_uid": "u1",
                "expires_at": now + timedelta(hours=1),
                "differences_data": {"best_model": "OpenAI"},
            }
            before = dict(db.data)
            persistence_guard.record_model_vote(
                uid="u1",
                result_id=result_id,
                model="OpenAI",
                vote_type="BestModel",
                db=db,
                now=now,
            )

    assert db.data == before


def test_feedback_cooldown_and_daily_limit_are_persistent(monkeypatch):
    db = Database()
    now = datetime(2026, 8, 11, 10, tzinfo=timezone.utc)
    persistence_guard.create_feedback(
        uid="u1", feedback={"message": "one", "uid": "u1", "timestamp": now}, db=db, now=now
    )
    with pytest.raises(persistence_guard.PersistenceLimitError, match="wait"):
        persistence_guard.create_feedback(
            uid="u1", feedback={"message": "two", "uid": "u1", "timestamp": now}, db=db,
            now=now + timedelta(seconds=5),
        )
    monkeypatch.setattr(persistence_guard, "MAX_FEEDBACK_PER_UTC_DAY", 1)
    with pytest.raises(persistence_guard.PersistenceLimitError, match="Daily"):
        persistence_guard.create_feedback(
            uid="u1", feedback={"message": "two", "uid": "u1", "timestamp": now}, db=db,
            now=now + timedelta(minutes=1),
        )


def test_vote_is_run_bound_and_exactly_once():
    db = Database()
    now = datetime(2026, 8, 11, tzinfo=timezone.utc)
    result_id = "R" * 16
    db.data[("pending_results", result_id)] = {
        "owner_uid": "u1",
        "expires_at": now + timedelta(hours=1),
        "differences_data": {"best_model": "OpenAI"},
    }
    assert persistence_guard.record_model_vote(
        uid="u1", result_id=result_id, model="OpenAI", vote_type="BestModel", db=db, now=now
    ) is True
    assert persistence_guard.record_model_vote(
        uid="u1", result_id=result_id, model="OpenAI", vote_type="BestModel", db=db, now=now
    ) is False
    assert db.data[("leaderboard", "OpenAI")]["BestModel"] == 1
    with pytest.raises(persistence_guard.PersistenceLimitError):
        persistence_guard.record_model_vote(
            uid="u2", result_id=result_id, model="OpenAI", vote_type="BestModel", db=db, now=now
        )


@pytest.mark.parametrize(
    "expires_at",
    [
        None,
        datetime(2026, 8, 11, 1, 0),
        datetime(2026, 8, 10, 23, 59, tzinfo=timezone.utc),
    ],
    ids=["missing", "naive", "expired"],
)
def test_vote_expiry_is_fail_closed(expires_at):
    db = Database()
    now = datetime(2026, 8, 11, tzinfo=timezone.utc)
    result_id = "E" * 16
    pending = {
        "owner_uid": "u1",
        "differences_data": {"best_model": "OpenAI"},
    }
    if expires_at is not None:
        pending["expires_at"] = expires_at
    db.data[("pending_results", result_id)] = pending

    with pytest.raises(
        persistence_guard.PersistenceLimitError, match="Completed result not found"
    ):
        persistence_guard.record_model_vote(
            uid="u1",
            result_id=result_id,
            model="OpenAI",
            vote_type="BestModel",
            db=db,
            now=now,
        )

    assert not any(path[0] == persistence_guard.VOTES_COLLECTION for path in db.data)
    assert ("leaderboard", "OpenAI") not in db.data


def test_follow_confirmation_has_persistent_resend_and_recipient_budgets(monkeypatch):
    db = Database()
    now = datetime(2026, 8, 11, tzinfo=timezone.utc)
    assert follow_challenges.claim_confirmation_send(
        resource_type="watch", resource_id="w1", email="reader@example.com", db=db, now=now
    ) is True
    assert follow_challenges.claim_confirmation_send(
        resource_type="watch", resource_id="w1", email="reader@example.com", db=db,
        now=now + timedelta(minutes=1),
    ) is False
    monkeypatch.setattr(follow_challenges, "MAX_PER_RECIPIENT_DAY", 1)
    assert follow_challenges.claim_confirmation_send(
        resource_type="topic", resource_id="t1", email="reader@example.com", db=db,
        now=now + timedelta(minutes=16),
    ) is False
    assert all("reader@example.com" not in str(data) for data in db.data.values())


def test_favicon_proxy_single_flights_and_uses_lru(monkeypatch):
    favicons._CACHE.clear()
    favicons._INFLIGHT.clear()
    calls = 0
    lock = threading.Lock()

    def fetch(host):
        nonlocal calls
        with lock:
            calls += 1
        time.sleep(0.03)
        return b"icon", "image/png"

    monkeypatch.setattr(favicons, "_fetch", fetch)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(favicons.get_favicon, ["example.com"] * 8))
    assert calls == 1
    assert all(result[0] == b"icon" for result in results)

    monkeypatch.setattr(favicons, "_MAX_ENTRIES", 2)
    favicons.get_favicon("one.example")
    favicons.get_favicon("two.example")
    favicons.get_favicon("three.example")
    assert len(favicons._CACHE) == 2
    assert "three.example" in favicons._CACHE


def test_favicon_exception_releases_followers_and_allows_retry(monkeypatch):
    favicons._CACHE.clear()
    favicons._INFLIGHT.clear()
    entered = threading.Event()
    release = threading.Event()

    def fail(_host):
        entered.set()
        release.wait(timeout=1)
        raise RuntimeError("decoder failed")

    monkeypatch.setattr(favicons, "_fetch", fail)
    with ThreadPoolExecutor(max_workers=2) as pool:
        leader = pool.submit(favicons.get_favicon, "broken.example")
        assert entered.wait(timeout=1)
        follower = pool.submit(favicons.get_favicon, "broken.example")
        release.set()
        with pytest.raises(RuntimeError, match="decoder failed"):
            leader.result(timeout=1)
        assert follower.result(timeout=1) == (None, "")

    assert "broken.example" not in favicons._INFLIGHT
    monkeypatch.setattr(favicons, "_fetch", lambda _host: (b"retry", "image/png"))
    assert favicons.get_favicon("broken.example") == (b"retry", "image/png")


def test_topics_router_imports_after_asyncio_run_and_uses_the_active_loop():
    """A closed default loop must not poison the module-level favicon gate."""
    script = """
import asyncio

asyncio.run(asyncio.sleep(0))
from app.api.routers import topics

async def acquire_once():
    semaphore = topics._favicon_concurrency()
    await semaphore.acquire()
    semaphore.release()
    return id(semaphore)

first = asyncio.run(acquire_once())
second = asyncio.run(acquire_once())
assert first != second
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_favicon_capacity_fallback_is_never_cached(monkeypatch):
    app = FastAPI()
    app.state.limiter = limiter
    limiter.reset()
    app.include_router(topics_router.router)

    class BusyGate:
        async def acquire(self):
            await asyncio.sleep(60)

        def release(self):
            raise AssertionError("an unacquired gate must not be released")

    gate = BusyGate()
    monkeypatch.setattr(topics_router, "_favicon_concurrency", lambda: gate)
    monkeypatch.setattr(topics_router, "_FAVICON_ACQUIRE_TIMEOUT_SECONDS", 0.001)

    response = TestClient(app).get("/api/topics/favicon?d=example.com")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/svg+xml")
    assert response.headers["cache-control"] == "no-store"


def test_favicon_unexpected_failure_returns_private_fallback_without_raw_log(
    monkeypatch, caplog
):
    app = FastAPI()
    app.state.limiter = limiter
    limiter.reset()
    app.include_router(topics_router.router)
    secret = "owner@example.test|decoder-private"

    def fail(_value):
        raise RuntimeError(secret)

    monkeypatch.setattr(favicons, "get_favicon", fail)
    with caplog.at_level("WARNING"):
        response = TestClient(app).get("/api/topics/favicon?d=example.com")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert "RuntimeError" in caplog.text
    assert secret not in caplog.text


def test_invalid_vote_model_never_reaches_logs(caplog):
    app = FastAPI()
    app.state.limiter = limiter
    limiter.reset()
    app.include_router(pages_router.router)
    client_model = "owner@example.test|sk-secret-9f3c"
    client_token = "id-token-private-7a4d"

    with caplog.at_level("WARNING"):
        response = TestClient(app).post(
            "/vote",
            json={
                "id_token": client_token,
                "model": client_model,
                "vote_type": "BestModel",
                "result_id": "r" * 16,
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid model name."
    assert "Invalid vote model rejected" in caplog.text
    assert "owner@example.test" not in caplog.text
    assert "sk-secret-9f3c" not in caplog.text
    assert client_token not in caplog.text


def test_correlation_header_metrics_and_log_redaction(caplog, monkeypatch):
    app = FastAPI()
    app.add_middleware(CorrelationMiddleware)

    @app.get("/ok")
    def ok():
        record_metric("provider", "OpenAI", duration_ms=12)
        return {"ok": True}

    response = TestClient(app).get("/ok")
    assert response.headers["x-correlation-id"].startswith("req-")
    assert metrics_snapshot()["provider:OpenAI"]["count"] >= 1

    secret = "private-user-answer-9f3c"
    data = {
        "claims": [{"anchor": secret, "dissent": [{"model": "OpenAI", "quote": secret}]}],
        "differences": [],
    }
    with caplog.at_level("INFO"):
        consensus_engine._verify_differences_data(data, "different", {"OpenAI": "different"})
    assert secret not in caplog.text

    monkeypatch.setattr(
        consensus_engine, "_call_engine_text",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError(secret)),
    )
    monkeypatch.setattr(consensus_engine, "_fallback_judge_engine", lambda *args: None)
    with caplog.at_level("WARNING"):
        result = consensus_engine.query_consensus(
            "question", "answer one", "answer two", None, None, None, None,
            excluded_models=[], consensus_model="OpenAI", api_keys={"OpenAI": "key"},
        )
    assert secret not in caplog.text
    assert secret not in result


@pytest.mark.parametrize(
    ("raw_result", "expected_outcome", "has_answer"),
    [
        ({"text": "answer", "sources": []}, "success", True),
        ({"text": "", "error": "generic", "error_code": "provider_request_failed"},
         "failure", False),
        (None, "timeout", False),
    ],
)
def test_provider_pipeline_metrics_classify_normalized_results_without_secrets(
    raw_result, expected_outcome, has_answer, monkeypatch, caplog
):
    secret = "owner@example.test|provider-body-private"
    recorded = []
    if raw_result is None:
        raw_result = engines._error("OpenAI", requests.Timeout(secret))
        assert raw_result["error_code"] == "provider_timeout"

    monkeypatch.setattr(
        provider_transport,
        "record_metric",
        lambda family, name, **values: recorded.append((family, name, values)),
    )
    with caplog.at_level("WARNING"):
        answers = provider_transport.fan_out_provider_answers(
            question="private question",
            provider_models={"openai": "model"},
            keys={"OpenAI": "secret-key"},
            is_pro=False,
            deep_think=False,
            provider_call=lambda *args: raw_result,
            log_context="Consensus_Watch",
        )

    assert ("openai" in answers) is has_answer
    assert recorded[0][0:2] == ("provider", "openai:Consensus_Watch")
    assert recorded[0][2]["outcome"] == expected_outcome
    assert secret not in caplog.text


def test_upstream_http_timeout_status_is_normalized_without_response_body(caplog):
    secret = "private-provider-response-body"
    response = requests.Response()
    response.status_code = 504
    error = requests.HTTPError(secret, response=response)

    with caplog.at_level("ERROR"):
        result = engines._error("Mistral", error)

    assert result["error_code"] == "provider_timeout"
    assert "HTTPError:504" in caplog.text
    assert secret not in caplog.text


@pytest.mark.parametrize("stream_requested", [False, True])
def test_direct_ask_metrics_classify_normalized_provider_timeouts(
    stream_requested, monkeypatch
):
    timeout_result = {
        "text": "",
        "sources": [],
        "error": "Provider timed out.",
        "error_code": "provider_timeout",
    }
    recorded = []
    provider = chat_router.AskProvider(
        label="OpenAI",
        allowed_models_attr="ALLOWED_OPENAI_MODELS",
        query_fn=lambda *_args, **_kwargs: timeout_result,
        stream_fn=lambda *_args, **_kwargs: iter(
            ({"type": "final", "result": timeout_result},)
        ),
        developer_key_env="DEVELOPER_OPENAI_API_KEY",
    )
    monkeypatch.setattr(
        chat_router,
        "record_metric",
        lambda family, name, **values: recorded.append((family, name, values)),
    )

    response = chat_router._run_ask(
        provider,
        stream_requested=stream_requested,
        question="question",
        key="key",
        system_prompt="system",
        deep_search=False,
        model="model",
        max_tokens=100,
        attachments=[],
        extras={},
    )
    if stream_requested:
        async def consume():
            async for _chunk in response.body_iterator:
                pass

        asyncio.run(consume())

    assert recorded[0][0:2] == ("provider", "OpenAI")
    assert recorded[0][2]["outcome"] == "timeout"


def test_direct_ask_disconnect_records_cancellation_not_success(monkeypatch):
    entered = threading.Event()
    released = threading.Event()
    stopped = threading.Event()
    recorded = []

    class BlockingResource:
        def close(self):
            released.set()

    def source(*_args, **_kwargs):
        try:
            with managed_provider_resource(BlockingResource()):
                entered.set()
                released.wait(timeout=2)
                yield {"type": "delta", "text": "late"}
        finally:
            stopped.set()

    provider = chat_router.AskProvider(
        label="OpenAI",
        allowed_models_attr="ALLOWED_OPENAI_MODELS",
        query_fn=lambda *_args, **_kwargs: None,
        stream_fn=source,
        developer_key_env="DEVELOPER_OPENAI_API_KEY",
    )
    monkeypatch.setattr(
        chat_router,
        "record_metric",
        lambda family, name, **values: recorded.append((family, name, values)),
    )
    response = chat_router._run_ask(
        provider,
        stream_requested=True,
        question="question",
        key="key",
        system_prompt="system",
        deep_search=False,
        model="model",
        max_tokens=100,
        attachments=[],
        extras={},
    )

    async def disconnect():
        async def receive():
            await anyio.to_thread.run_sync(lambda: entered.wait(timeout=1))
            return {"type": "http.disconnect"}

        async def send(_message):
            return None

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "method": "GET",
            "path": "/stream",
            "headers": [],
            "query_string": b"",
            "http_version": "1.1",
            "scheme": "http",
            "server": ("test", 80),
            "client": ("test", 123),
        }
        await response(scope, receive, send)

    anyio.run(disconnect)

    assert stopped.wait(timeout=1)
    assert len(recorded) == 1
    assert recorded[0][0:2] == ("provider", "OpenAI")
    assert recorded[0][2]["outcome"] == "cancelled"


def test_cancelled_provider_metric_has_a_dedicated_counter():
    record_metric("provider", "CancellationContract", outcome="cancelled")
    item = metrics_snapshot()["provider:CancellationContract"]

    assert item["count"] >= 1
    assert item["cancellations"] >= 1


def test_due_brief_query_filters_and_limits_in_firestore():
    db = Database()
    now = datetime(2026, 8, 11, tzinfo=timezone.utc)
    db.data[("watch_briefs", "due")] = {"enabled": True, "next_send_at": now}
    db.data[("watch_briefs", "later")] = {
        "enabled": True, "next_send_at": now + timedelta(days=1)
    }
    db.data[("watch_briefs", "off")] = {"enabled": False, "next_send_at": now}
    assert watch_brief.list_due_brief_uids(now=now, db=db, max_items=1) == ["due"]


def test_phase5_frontend_vote_binding_and_deployment_contracts():
    root = Path(__file__).resolve().parents[1]
    firebase_js = (root / "static" / "firebase.js").read_text(encoding="utf-8")
    consensus_js = (root / "static" / "js" / "consensus-run.js").read_text(
        encoding="utf-8"
    )
    assert "result_id: resultId" in firebase_js
    assert "data.result_id" in consensus_js

    firebase_config = json.loads((root / "firebase.json").read_text(encoding="utf-8"))
    assert firebase_config["firestore"]["indexes"] == "firestore.indexes.json"
    indexes = json.loads((root / "firestore.indexes.json").read_text(encoding="utf-8"))
    collections = {entry["collectionGroup"] for entry in indexes["indexes"]}
    assert {"watches", "topics", "watch_briefs"} <= collections


def test_benchmark_only_dependencies_are_not_in_production_requirements():
    root = Path(__file__).resolve().parents[1]
    production = (root / "requirements.txt").read_text(encoding="utf-8").lower()
    benchmark = (root / "benchmark" / "requirements-benchmark.txt").read_text(
        encoding="utf-8"
    ).lower()
    for package in ("huggingface-hub", "pandas"):
        assert package not in production
        assert package in benchmark
