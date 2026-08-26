import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routers import pages as pages_router
from app.api.routers import topics as topics_router
from app.core.rate_limit import limiter
from app.services import (
    follow_challenges, mailer, topic_pipeline, topic_runner, topics,
)
from app.services.account_deletion import FirestoreAccountDeletion


NOW = datetime(2026, 7, 23, 12, 0, tzinfo=timezone.utc)
ROOT = Path(__file__).resolve().parents[1]


class FakeSnapshot:
    def __init__(self, reference, data):
        self.reference = reference
        self.id = reference.id
        self._data = dict(data) if data is not None else None
        self.exists = data is not None

    def to_dict(self):
        return dict(self._data or {})


class FakeDocument:
    def __init__(self, db, path):
        self.db = db
        self.path = tuple(path)
        self.id = self.path[-1]

    def get(self, transaction=None):
        if transaction is not None:
            return transaction.get(self)
        return FakeSnapshot(self, self.db.documents.get(self.path))

    def set(self, data, merge=False):
        if merge and self.path in self.db.documents:
            self.db.documents[self.path].update(dict(data))
        else:
            self.db.documents[self.path] = dict(data)

    def update(self, data):
        self.db.documents[self.path].update(dict(data))

    def delete(self):
        self.db.documents.pop(self.path, None)

    def collection(self, name):
        return FakeCollection(self.db, self.path + (name,))


class FakeQuery:
    def __init__(self, collection, filters):
        self.collection = collection
        self.filters = filters

    def where(self, field, op, value):
        return FakeQuery(self.collection, self.filters + [(field, op, value)])

    def stream(self):
        snapshots = self.collection.stream()
        for field, op, value in self.filters:
            assert op == "=="
            snapshots = [
                snap for snap in snapshots if (snap.to_dict() or {}).get(field) == value
            ]
        return snapshots


class FakeCollection:
    def __init__(self, db, path):
        self.db = db
        self.path = tuple(path)

    def document(self, doc_id):
        return FakeDocument(self.db, self.path + (doc_id,))

    def where(self, field, op, value):
        return FakeQuery(self, [(field, op, value)])

    def stream(self):
        return [
            FakeSnapshot(FakeDocument(self.db, path), data)
            for path, data in self.db.documents.items()
            if len(path) == len(self.path) + 1 and path[:-1] == self.path
        ]


class FakeFirestore:
    def __init__(self):
        self.documents = {}
        self.fail_transaction_after_staged_writes = None

    def collection(self, name):
        return FakeCollection(self, (name,))

    def run_transaction(self, operation):
        transaction = FakeTransaction(self)
        result = operation(transaction)
        transaction.commit()
        return result


class FakeTransaction:
    def __init__(self, db):
        self.db = db
        self.operations = []

    def get(self, ref):
        return FakeSnapshot(ref, self.db.documents.get(ref.path))

    def set(self, ref, data, merge=False):
        self.operations.append(("set", ref, dict(data), merge))
        fail_after = self.db.fail_transaction_after_staged_writes
        if fail_after is not None and len(self.operations) >= fail_after:
            raise RuntimeError("injected transaction failure")

    def update(self, ref, data):
        self.operations.append(("update", ref, dict(data), False))

    def delete(self, ref):
        self.operations.append(("delete", ref, None, False))

    def commit(self):
        for operation, ref, data, merge in self.operations:
            if operation == "set":
                ref.set(data, merge=merge)
            elif operation == "update":
                ref.update(data)
            else:
                ref.delete()


def topic_payload(**overrides):
    payload = {
        "title": "GPT-6",
        "slug": "gpt-6",
        "lead_question": "What is the current evidence about GPT-6?",
        "category": "Model releases",
        "summary": "A curated record of claims, releases, and model consensus.",
        "status": "active",
        "update_interval": "weekly",
        "models": ["OpenAI · GPT-5.6", "Anthropic · Claude Opus 4.6"],
        "source_rules": {
            "allowed_types": ["official", "x", "press"],
            "preferred_domains": ["openai.com"],
            "notes": "Prefer primary sources.",
        },
        "evidence": [{
            "type": "official",
            "title": "Official product update",
            "url": "https://openai.com/index/example",
            "publisher": "OpenAI",
            "published_at": "2026-07-22",
            "excerpt": "Primary evidence.",
        }],
        "seo": {
            "title": "GPT-6 Consensus Timeline",
            "description": "Track the evidence and model consensus around GPT-6.",
            "noindex": False,
        },
    }
    payload.update(overrides)
    return payload


def run_payload(**overrides):
    payload = {
        "observed_at": NOW.isoformat(),
        "consensus_md": "## Current consensus\n\nNo confirmed release date exists.",
        "agreement_score": 78,
        "change_type": "stable",
        "change_summary": "No material shift.",
        "opinion_changes": [],
    }
    payload.update(overrides)
    return payload


def test_topic_runs_are_immutable_and_keep_historical_editorial_state():
    db = FakeFirestore()
    topic = topics.create_topic(topic_payload(), actor_uid="admin", db=db, now=NOW)
    first = topics.create_run(
        topic["id"], run_payload(), actor_uid="admin", db=db, now=NOW
    )

    topics.update_topic(
        topic["id"],
        topic_payload(title="GPT-6 Release", slug="gpt-6-release"),
        actor_uid="admin",
        db=db,
        now=NOW,
    )
    second = topics.create_run(
        topic["id"],
        run_payload(
            agreement_score=61,
            change_type="major",
            change_summary="A product announcement changed the expected timeline.",
            consensus_md="The expected timeline moved.",
            opinion_changes=[{
                "model": "Anthropic · Claude Opus 4.6",
                "from": "No estimate",
                "to": "Release is more likely this year",
                "summary": "The official announcement changed its weighting.",
            }],
        ),
        actor_uid="admin",
        db=db,
        now=NOW,
    )

    stored_first = topics.get_run(topic["id"], first["id"], db=db)
    assert stored_first["consensus_md"].startswith("## Current consensus")
    assert stored_first["topic_state"]["title"] == "GPT-6"
    assert stored_first["evidence"][0]["url"] == "https://openai.com/index/example"
    assert second["version"] == 2

    current = topics.get_topic(topic["id"], db=db)
    assert current["latest_run_id"] == second["id"]
    assert current["latest_agreement_score"] == 61
    assert current["run_count"] == 2
    assert topics.list_public_topics(db=db)[0]["slug"] == "gpt-6-release"


def test_topic_run_and_latest_pointer_commit_or_fail_together():
    db = FakeFirestore()
    topic = topics.create_topic(topic_payload(), actor_uid="admin", db=db, now=NOW)
    before = dict(db.documents[("topics", topic["id"])])
    db.fail_transaction_after_staged_writes = 2

    with pytest.raises(RuntimeError, match="injected transaction failure"):
        topics.create_run(
            topic["id"], run_payload(), actor_uid="admin", db=db, now=NOW
        )

    assert db.documents[("topics", topic["id"])] == before
    assert not any(
        len(path) == 4 and path[:3] == ("topics", topic["id"], "runs")
        for path in db.documents
    )


def test_stale_topic_claim_cannot_publish_or_mark_newer_claim_failed():
    db = FakeFirestore()
    topic = topics.create_topic(topic_payload(), actor_uid="admin", db=db, now=NOW)
    topic_path = ("topics", topic["id"])
    db.documents[topic_path].update({
        "current_run_id": "new-claim",
        "last_run_status": "running",
    })

    with pytest.raises(topics.TopicError, match="lease is no longer current"):
        topics.create_run(
            topic["id"],
            run_payload(),
            actor_uid="admin",
            db=db,
            now=NOW,
            run_id="old-claim",
            expected_claim_id="old-claim",
        )
    assert topics.fail_topic_run(
        topic["id"],
        "old worker failed",
        db=db,
        now=NOW,
        expected_claim_id="old-claim",
    ) is False
    assert db.documents[topic_path]["current_run_id"] == "new-claim"
    assert db.documents[topic_path]["last_run_status"] == "running"


def test_archived_topics_leave_public_discovery_and_reject_new_runs():
    db = FakeFirestore()
    topic = topics.create_topic(topic_payload(), actor_uid="admin", db=db, now=NOW)
    topics.create_run(topic["id"], run_payload(), actor_uid="admin", db=db, now=NOW)
    assert len(topics.list_indexed_topic_urls(db=db)) == 1

    archived = topics.update_topic(
        topic["id"],
        topic_payload(status="archived"),
        actor_uid="admin",
        db=db,
        now=NOW,
    )

    assert archived["status"] == "archived"
    assert topics.list_public_topics(db=db) == []
    assert topics.list_indexed_topic_urls(db=db) == []
    with pytest.raises(topics.TopicError, match="Archived topics"):
        topics.create_run(
            topic["id"], run_payload(), actor_uid="admin", db=db, now=NOW
        )


def test_noindex_and_unpublished_topics_are_not_in_topic_sitemap():
    db = FakeFirestore()
    unpublished = topics.create_topic(
        topic_payload(slug="unpublished"), actor_uid="admin", db=db, now=NOW
    )
    noindex = topics.create_topic(
        topic_payload(
            title="Claude Pricing",
            slug="claude-pricing",
            seo={"title": "", "description": "", "noindex": True},
        ),
        actor_uid="admin",
        db=db,
        now=NOW,
    )
    topics.create_run(
        noindex["id"], run_payload(), actor_uid="admin", db=db, now=NOW
    )

    assert unpublished["latest_run_id"] == ""
    assert topics.list_indexed_topic_urls(db=db) == []
    assert len(topics.list_public_topics(db=db)) == 1


def test_slug_uniqueness_and_evidence_url_validation():
    db = FakeFirestore()
    topics.create_topic(topic_payload(), actor_uid="admin", db=db, now=NOW)
    with pytest.raises(topics.TopicError, match="slug is already"):
        topics.create_topic(
            topic_payload(title="Other", slug="gpt-6"),
            actor_uid="admin",
            db=db,
            now=NOW,
        )
    with pytest.raises(topics.TopicError, match="valid http"):
        topics.create_topic(
            topic_payload(
                title="Unsafe",
                slug="unsafe",
                evidence=[{
                    "type": "press",
                    "title": "Bad URL",
                    "url": "javascript:alert(1)",
                }],
            ),
            actor_uid="admin",
            db=db,
            now=NOW,
        )


def test_renamed_topic_keeps_its_runs_and_redirects_the_old_url(monkeypatch):
    db = FakeFirestore()
    monkeypatch.setattr(topics, "db_firestore", db)
    topic = topics.create_topic(topic_payload(), actor_uid="admin", db=db, now=NOW)
    topics.create_run(topic["id"], run_payload(), actor_uid="admin", db=db, now=NOW)
    topics.create_run(
        topic["id"], run_payload(agreement_score=61), actor_uid="admin", db=db, now=NOW
    )

    renamed = topics.update_topic(
        topic["id"],
        topic_payload(slug="when-will-gpt-6-be-released"),
        actor_uid="admin",
        db=db,
        now=NOW,
    )

    # The record lives under the topic id, so the rename cannot touch it.
    assert renamed["slug_history"] == ["gpt-6"]
    assert renamed["run_count"] == 2
    assert len(topics.list_runs(topic["id"], db=db)) == 2
    assert topics.list_indexed_topic_urls(db=db)[0]["path"] == (
        "/topics/when-will-gpt-6-be-released"
    )
    found, retired = topics.resolve_topic_by_slug("gpt-6", db=db)
    assert retired is True and found["id"] == topic["id"]
    assert topics.resolve_topic_by_slug("when-will-gpt-6-be-released", db=db)[1] is False

    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(topics_router.router)
    client = TestClient(app)

    moved = client.get("/topics/gpt-6", follow_redirects=False)
    assert moved.status_code == 301
    assert moved.headers["location"] == "/topics/when-will-gpt-6-be-released"
    assert client.get("/topics/when-will-gpt-6-be-released").status_code == 200
    assert client.get("/topics/never-existed", follow_redirects=False).status_code == 404


def test_a_retired_slug_cannot_be_claimed_and_can_be_taken_back(monkeypatch):
    db = FakeFirestore()
    monkeypatch.setattr(topics, "db_firestore", db)
    topic = topics.create_topic(topic_payload(), actor_uid="admin", db=db, now=NOW)
    topics.update_topic(
        topic["id"], topic_payload(slug="gpt-6-release-date"),
        actor_uid="admin", db=db, now=NOW,
    )

    # The old URL still belongs to this topic, so nobody else gets its traffic.
    with pytest.raises(topics.TopicError, match="slug is already"):
        topics.create_topic(
            topic_payload(title="Other", slug="gpt-6"),
            actor_uid="admin", db=db, now=NOW,
        )

    back = topics.update_topic(
        topic["id"], topic_payload(slug="gpt-6"), actor_uid="admin", db=db, now=NOW
    )
    assert back["slug"] == "gpt-6"
    assert back["slug_history"] == ["gpt-6-release-date"]
    # Canonical again: it must stop redirecting to itself.
    assert topics.resolve_topic_by_slug("gpt-6", db=db)[1] is False
    assert topics.resolve_topic_by_slug("gpt-6-release-date", db=db)[1] is True


@pytest.mark.parametrize(("url", "expected"), [
    ("https://openai.com/index/update", "primary"),
    ("https://arxiv.org/abs/2607.12345", "research"),
    ("https://github.com/example/project/releases/tag/v1", "documentation"),
    ("https://www.reuters.com/technology/example", "reporting"),
    ("https://www.youtube.com/watch?v=example", "community"),
    ("https://kalshi.com/markets/example", "rumor"),
])
def test_evidence_sources_receive_specific_public_roles(url, expected):
    assert topics.classify_evidence(url)["role"] == expected


def test_google_redirects_are_unwrapped_or_flagged_as_indirect():
    direct = "https://openai.com/index/update"
    wrapped = "https://www.google.com/url?q=https%3A%2F%2Fopenai.com%2Findex%2Fupdate"
    assert topics.canonical_evidence_url(wrapped) == direct
    indirect = topics.classify_evidence(
        "https://vertexaisearch.cloud.google.com/grounding-api-redirect/signed"
    )
    assert indirect["is_indirect"] is True
    assert indirect["quality"] == "low"


def test_automatic_evidence_orders_direct_sources_before_rumors():
    evidence = topic_runner.evidence_from_sources([
        {"title": "Prediction market", "url": "https://kalshi.com/markets/gpt6"},
        {"title": "Official update", "url": "https://openai.com/index/update"},
        {"title": "Paper", "url": "https://arxiv.org/abs/2607.12345"},
    ], topics.normalize_source_rules({}))
    assert [item["type"] for item in evidence] == ["primary", "research", "rumor"]


def test_topic_followers_use_separate_collection_and_double_opt_in(monkeypatch):
    monkeypatch.setenv("WATCH_UNSUBSCRIBE_SECRET", "topic-test-secret")
    db = FakeFirestore()
    topic = topics.create_topic(topic_payload(), actor_uid="admin", db=db, now=NOW)
    topics.create_run(topic["id"], run_payload(), actor_uid="admin", db=db, now=NOW)

    pending = topics.request_follow(topic["id"], "Reader@Example.com", db=db)
    assert pending["token"]
    assert not any(path[0] == topics.FOLLOWERS_COLLECTION for path in db.documents)

    confirmed = topics.confirm_follow(pending["token"], db=db, now=NOW)
    assert confirmed["email"] == "reader@example.com"
    follower_paths = [
        path for path in db.documents if path[0] == topics.FOLLOWERS_COLLECTION
    ]
    assert len(follower_paths) == 1
    assert not any(path[0] == "watch_followers" for path in db.documents)
    with pytest.raises(topics.TopicError, match="invalid or expired"):
        topics.confirm_follow(pending["token"], db=db, now=NOW)

    follower_doc_id = follower_paths[0][1]
    assert topics.claim_delivery(
        topic["id"], "run-before-unsubscribe", follower_doc_id, db=db
    )
    token = topics.make_unsubscribe_token(topic["id"], confirmed["email"], now=NOW)
    topics.unsubscribe_follow(token, db=db)
    assert not any(path[0] == topics.FOLLOWERS_COLLECTION for path in db.documents)
    assert not any(path[0] == topics.DELIVERIES_COLLECTION for path in db.documents)


def test_topic_account_cleanup_invalidates_outstanding_confirm_link(monkeypatch):
    monkeypatch.setenv("WATCH_UNSUBSCRIBE_SECRET", "topic-test-secret")
    db = FakeFirestore()
    topic = topics.create_topic(topic_payload(), actor_uid="admin", db=db, now=NOW)
    topics.create_run(topic["id"], run_payload(), actor_uid="admin", db=db, now=NOW)
    email = "cleanup@example.com"
    pending = topics.request_follow(topic["id"], email, db=db)

    assert follow_challenges.delete_for_email(email, db=db) >= 1
    with pytest.raises(topics.TopicError, match="invalid or expired"):
        topics.confirm_follow(pending["token"], db=db, now=NOW)
    assert not any(path[0] == topics.FOLLOWERS_COLLECTION for path in db.documents)


def test_topic_confirm_consumes_challenge_atomically_with_follower_write(monkeypatch):
    monkeypatch.setenv("WATCH_UNSUBSCRIBE_SECRET", "topic-test-secret")
    db = FakeFirestore()
    topic = topics.create_topic(topic_payload(), actor_uid="admin", db=db, now=NOW)
    topics.create_run(topic["id"], run_payload(), actor_uid="admin", db=db, now=NOW)
    pending = topics.request_follow(topic["id"], "atomic@example.com", db=db)
    challenge_paths = [
        path for path in db.documents
        if path[0] == follow_challenges.COLLECTION
        and str(path[1]).startswith("challenge-")
    ]
    assert len(challenge_paths) == 1

    # Challenge delete is staged first, follower create second. A transaction
    # failure must commit neither side of that pair.
    db.fail_transaction_after_staged_writes = 2
    with pytest.raises(RuntimeError, match="injected transaction failure"):
        topics.confirm_follow(pending["token"], db=db, now=NOW)

    assert challenge_paths[0] in db.documents
    assert not any(path[0] == topics.FOLLOWERS_COLLECTION for path in db.documents)

    db.fail_transaction_after_staged_writes = None
    assert topics.confirm_follow(pending["token"], db=db, now=NOW)["email"] == (
        "atomic@example.com"
    )
    assert challenge_paths[0] not in db.documents


def test_topic_notification_delivery_is_deduplicated_and_multipart():
    db = FakeFirestore()
    db.documents[(topics.FOLLOWERS_COLLECTION, "follower")] = {
        "topic_id": "topic",
        "email": "reader@example.com",
        "created_at": NOW,
    }
    assert topics.claim_delivery("topic", "run", "follower", db=db) is True
    assert topics.claim_delivery("topic", "run", "follower", db=db) is False
    topics.finish_delivery(
        "topic", "run", "follower", success=False, db=db
    )
    assert topics.claim_delivery("topic", "run", "follower", db=db) is True
    topics.finish_delivery(
        "topic", "run", "follower", success=True, db=db
    )
    assert topics.claim_delivery("topic", "run", "follower", db=db) is False

    message = mailer.build_topic_change_message(
        recipient="reader@example.com",
        title="GPT-6",
        question="What changed?",
        old_score=58,
        new_score=76,
        change_type="major",
        summary="Primary evidence changed the consensus.",
        topic_url="https://www.consens.io/topics/gpt-6",
        unsubscribe_url="https://www.consens.io/topic-follow/unsubscribe?token=x",
    )
    assert message.is_multipart()
    assert "Topic update: GPT-6" == message["Subject"]
    assert "Primary evidence changed the consensus." in message.get_body(
        preferencelist=("plain",)
    ).get_content()


def test_topic_delivery_claim_requires_a_live_topic_bound_follower():
    db = FakeFirestore()
    follower_path = (topics.FOLLOWERS_COLLECTION, "follower")
    db.documents[follower_path] = {
        "topic_id": "other-topic",
        "email": "reader@example.com",
        "created_at": NOW,
    }

    assert topics.claim_delivery("topic", "run", "follower", db=db) is False
    db.documents.pop(follower_path)
    assert topics.claim_delivery("topic", "run", "follower", db=db) is False
    assert not any(path[0] == topics.DELIVERIES_COLLECTION for path in db.documents)


def test_topic_delivery_finish_does_not_recreate_a_cleaned_claim():
    db = FakeFirestore()
    follower_path = (topics.FOLLOWERS_COLLECTION, "follower")
    db.documents[follower_path] = {
        "topic_id": "topic",
        "email": "reader@example.com",
        "created_at": NOW,
    }
    assert topics.claim_delivery("topic", "run", "follower", db=db) is True

    assert topics.delete_follower_and_deliveries("follower", db=db) == 1
    assert topics.finish_delivery(
        "topic", "run", "follower", success=True, db=db
    ) is False
    assert follower_path not in db.documents
    assert not any(path[0] == topics.DELIVERIES_COLLECTION for path in db.documents)
    assert topics.claim_delivery("topic", "run", "follower", db=db) is False


def test_account_deletion_fences_claims_before_topic_delivery_cleanup():
    db = FakeFirestore()
    follower_path = (topics.FOLLOWERS_COLLECTION, "follower")
    db.documents[follower_path] = {
        "topic_id": "topic",
        "email": "owner@example.com",
        "created_at": NOW,
    }
    assert topics.claim_delivery("topic", "run", "follower", db=db) is True

    FirestoreAccountDeletion(db)._delete_email_follows("owner@example.com")

    assert follower_path not in db.documents
    assert not any(path[0] == topics.DELIVERIES_COLLECTION for path in db.documents)
    assert topics.claim_delivery("topic", "run", "follower", db=db) is False


def test_topic_templates_expose_timeline_evidence_follow_and_admin_controls():
    detail = (ROOT / "templates" / "topic.html").read_text(encoding="utf-8")
    hub = (ROOT / "templates" / "topics.html").read_text(encoding="utf-8")
    admin = (ROOT / "templates" / "admin.html").read_text(encoding="utf-8")
    nav = (ROOT / "templates" / "partials" / "public_nav.html").read_text(encoding="utf-8")

    assert 'class="topic-timeline"' in detail
    assert 'id="evidence"' in detail
    assert 'id="record"' in detail
    assert 'class="claim-lifeline"' in detail
    # The finding leads, the statements it is made of follow, and the
    # apparatus that produced them sits behind a disclosure below both.
    assert detail.index("topic-finding-line") < detail.index("topic-facts")
    assert detail.index("topic-facts") < detail.index('class="topic-deep"')
    assert 'id="topicStrip"' in detail
    # The band is for readers who have been here before; it ships empty and
    # hidden, and CSS has to leave the hidden attribute alone.
    assert 'id="topicReturn" hidden' in detail
    # The Position Map grid takes exactly two children: the label block and
    # the clusters. A bare type + heading + clusters puts the clusters under
    # the label and stretches the type across the row.
    assert 'class="watch-position-question"' in detail
    assert 'id="topicFollowForm"' in detail
    assert 'class="topic-now-grid"' not in detail
    # The apparatus opens on demand; nothing above it is hidden.
    assert '<details class="legal-panel topic-deep-item" id="answer">' in detail
    assert 'topic-deep-item" id="answer" open' not in detail
    assert 'class="legal-panel topic-timeline-card" open' in detail
    assert "selected.evidence[:5]" in detail
    assert "selected.evidence[5:]" in detail
    assert 'item.role_label' in detail
    assert 'rel="canonical" href="https://www.consens.io/topics"' in hub
    assert 'id="tab-topics"' in admin
    assert 'id="runAdminTopicBtn"' in admin
    assert "Seed links and manual Consensus text are not required." in admin
    assert 'href="/topics"' in nav


def test_legacy_topic_admin_url_redirects_into_main_admin():
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(pages_router.router)
    client = TestClient(app)

    response = client.get("/admin/topics", follow_redirects=False)

    assert response.status_code == 308
    assert response.headers["location"] == "/admin#topics"


def test_mock_llm_instances_never_publish_topic_runs(monkeypatch):
    """MOCK_LLM servers share the production Firestore, so they must refuse a
    Topic run before claiming the slot instead of publishing fixture text."""
    db = FakeFirestore()
    topic = topics.create_topic(topic_payload(evidence=[]), actor_uid="admin", db=db, now=NOW)
    monkeypatch.setenv("MOCK_LLM", "1")

    with pytest.raises(topics.TopicError) as manual:
        topic_runner.run_topic_now(topic["id"], actor_uid="admin", db=db, now=NOW)
    with pytest.raises(topics.TopicError):
        topic_runner.execute_claimed_topic(
            {**topic, "current_run_id": "blocked"}, actor_uid="admin", db=db, now=NOW,
        )

    assert manual.value.code == "mock_mode"
    stored = topics.get_topic(topic["id"], db=db)
    # Neither the claim nor a failure marker reached the shared database.
    assert not stored.get("latest_run_id")
    assert stored.get("last_run_status") != "failed"
    assert asyncio.run(topic_runner.run_due_topic_tick()) == 0
    assert mailer._deliver(mailer._base_message("a@b.c", "s", "p", "<p>h</p>")) is False


def test_automatic_topic_run_researches_sources_and_builds_timeline_point():
    db = FakeFirestore()
    topic = topics.create_topic(topic_payload(evidence=[]), actor_uid="admin", db=db, now=NOW)
    claimed = {
        **topic,
        "current_run_id": "automatic-run",
        "claimed_until": NOW,
    }
    db.documents[("topics", topic["id"])].update({
        "current_run_id": "automatic-run",
        "claimed_until": NOW,
        "last_run_status": "running",
    })

    def execute(question, previous_consensus, **kwargs):
        assert "Research the current state" in question
        assert kwargs["model_overrides"] == topic["run_config"]["provider_models"]
        return {
            "consensus": "## Current consensus\n\nThe researched state is current.",
            "agreement_score": 82,
            "changed": False,
            "severity": "minor",
            "change_summary": "First consensus established.",
            "opinion_map": {},
            "differences_data": {"agreement": {"score": 82}},
            "sources": [{
                "id": "S1",
                "title": "Official update",
                "url": "https://openai.com/index/current-update",
                "provider": "OpenAI",
            }],
            "included_models": ["OpenAI: GPT-5.6", "Google Gemini: Gemini 3.5 Flash"],
        }

    run = topic_runner.execute_claimed_topic(
        claimed, actor_uid="admin", db=db, now=NOW, executor=execute
    )

    assert run["run_mode"] == "automatic"
    assert run["evidence"][0]["url"] == "https://openai.com/index/current-update"
    assert run["evidence"][0]["type"] == "primary"
    assert topics.get_topic(topic["id"], db=db)["last_run_status"] == "success"


def test_topic_run_carries_the_tracked_claims_into_the_identity_judge(monkeypatch):
    """A new run has to know which claims the Topic already tracks, otherwise
    every reworded claim starts a new life in the Claim Ledger."""
    db = FakeFirestore()
    topic = topics.create_topic(topic_payload(evidence=[]), actor_uid="admin", db=db, now=NOW)
    topics.create_run(
        topic["id"],
        run_payload(opinion_map={
            "schema_version": 1,
            "dimensions": [{
                "label": "No release date has been announced for GPT-6",
                "type": "claim",
                "key": "seed-0",
                "positions": [{"stance": "No date", "models": ["OpenAI", "Gemini"]}],
            }],
            "models": [],
            "shift_score": 0,
            "shift_label": "Stable",
            "center": [],
        }),
        actor_uid="admin", db=db, now=NOW,
    )
    db.documents[("topics", topic["id"])].update({
        "current_run_id": "second-run",
        "claimed_until": NOW,
        "last_run_status": "running",
    })
    stored = topics.get_topic(topic["id"], db=db)
    claimed = {**stored, "id": topic["id"], "current_run_id": "second-run"}
    seen = {}

    def execute(question, previous_consensus, **kwargs):
        seen.update(kwargs)
        return {
            "consensus": "## Current consensus\n\nStill no date.",
            "agreement_score": 82,
            "changed": False,
            "severity": "minor",
            "change_summary": "No material shift.",
            "opinion_map": {},
            "differences_data": {},
            "sources": [],
            "included_models": ["OpenAI: GPT-5.6", "Google Gemini: Gemini 3.5"],
        }

    topic_runner.execute_claimed_topic(
        claimed, actor_uid="admin", db=db, now=NOW, executor=execute
    )

    assert seen["known_claims"] == [
        {"key": "seed-0", "label": "No release date has been announced for GPT-6"}
    ]
    # The run id keys this run's new claims, so two runs can never collide.
    assert seen["claim_key_prefix"] == "second-run"


def test_claim_identity_falls_back_to_fresh_keys_when_the_judge_is_unavailable():
    """An unreachable identity Judge may cost the Ledger a link between two
    wordings. It may never cost the Topic its run."""
    position_map = {"dimensions": [
        {"label": "No release date has been announced", "positions": []},
        {"label": "GPT-5.6 is the frontier line", "positions": []},
    ]}

    def explode(*args, **kwargs):
        raise RuntimeError("judge offline")

    with mock.patch.object(topic_pipeline, "query_claim_identity", explode):
        topic_pipeline._stamp_claim_keys(
            position_map,
            [{"key": "k1", "label": "No date announced"}],
            {}, "OpenAI", "run-7",
        )

    assert [item["key"] for item in position_map["dimensions"]] == ["run-7-0", "run-7-1"]


def test_claim_identity_result_maps_claims_onto_the_keys_they_continue():
    position_map = {"dimensions": [
        {"label": "No launch window is on record", "positions": []},
        {"label": "A brand new claim about pricing", "positions": []},
    ]}

    with mock.patch.object(
        topic_pipeline, "query_claim_identity", lambda *args, **kwargs: {0: "k1"}
    ):
        topic_pipeline._stamp_claim_keys(
            position_map,
            [{"key": "k1", "label": "No release date has been announced"}],
            {}, "OpenAI", "run-8",
        )

    assert [item["key"] for item in position_map["dimensions"]] == ["k1", "run-8-1"]


def test_admin_topic_api_creates_updates_and_versions_without_share_data(
    monkeypatch,
):
    db = FakeFirestore()
    monkeypatch.setattr(topics, "db_firestore", db)
    monkeypatch.setattr(topics_router, "_require_admin", lambda request, data=None: "admin")
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(topics_router.router)
    client = TestClient(app)

    created = client.post("/api/admin/topics", json=topic_payload())
    assert created.status_code == 200
    topic_id = created.json()["topic"]["id"]
    published = client.post(
        f"/api/admin/topics/{topic_id}/runs", json=run_payload()
    )
    assert published.status_code == 200
    assert published.json()["run"]["version"] == 1
    detail = client.get(f"/api/admin/topics/{topic_id}")
    assert detail.status_code == 200
    assert detail.json()["runs"][0]["agreement_score"] == 78

    collection_names = {path[0] for path in db.documents}
    assert "topics" in collection_names
    assert "shares" not in collection_names
    assert "watches" not in collection_names


def opinion_map_payload(*, stance="A late-2026 launch", moved=False):
    return {
        "schema_version": 1,
        "dimensions": [{
            "label": "When does the model ship?",
            "type": "contradiction",
            "positions": [
                {"stance": stance, "models": ["OpenAI", "Gemini"]},
                {"stance": "No launch before 2027", "models": ["Anthropic"]},
            ],
        }],
        "models": [
            {"provider": "OpenAI", "movement_score": 100 if moved else 0, "moved": moved, "summary": ""},
            {"provider": "Anthropic", "movement_score": 0, "moved": False, "summary": ""},
        ],
        "shift_score": 50 if moved else 0,
        "shift_label": "Turning" if moved else "Stable",
        "center": [stance],
    }


def test_topic_page_shows_the_position_map_and_agreement_history(monkeypatch):
    db = FakeFirestore()
    monkeypatch.setattr(topics, "db_firestore", db)
    topic = topics.create_topic(topic_payload(), actor_uid="admin", db=db, now=NOW)
    first = topics.create_run(
        topic["id"],
        run_payload(opinion_map=opinion_map_payload()),
        actor_uid="admin", db=db, now=NOW,
    )
    topics.create_run(
        topic["id"],
        run_payload(
            agreement_score=52,
            change_type="major",
            change_summary="Anthropic moved to a 2027 window.",
            opinion_map=opinion_map_payload(stance="A 2027 launch", moved=True),
        ),
        actor_uid="admin", db=db, now=NOW,
    )
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(topics_router.router)
    client = TestClient(app)

    current = client.get("/topics/gpt-6")
    historical = client.get(f"/topics/gpt-6?version={first['id']}")

    assert current.status_code == 200
    assert "Where the models actually split" in current.text
    assert "No launch before 2027" in current.text
    assert "Direction Shift" in current.text
    assert "How this answer held up" in current.text
    assert "Anthropic moved to a 2027 window." in current.text
    assert "</b> AI models" in current.text
    assert "<b>1</b> contradiction" in current.text
    assert "<b>2</b> checks since" in current.text

    # A historical version shows the state of knowledge at that time, so the
    # later snapshot must not leak into its curve or its map. The sidebar
    # timeline stays complete, because it is navigation.
    assert historical.status_code == 200
    assert "A late-2026 launch" in historical.text
    assert "A 2027 launch" not in historical.text
    assert "How this answer held up" not in historical.text
    assert "Return to the current consensus" in historical.text


def test_topic_page_leads_with_the_finding_and_folds_unchanged_checks(monkeypatch):
    """The page states the finding first, then the statements it is made of,
    then the record that produced them.

    The record is what makes this page worth more than one model's answer, but
    it is the second question a reader has. A page that opens on the archive
    makes them work for the sentence they came for.
    """
    db = FakeFirestore()
    monkeypatch.setattr(topics, "db_firestore", db)
    topic = topics.create_topic(topic_payload(), actor_uid="admin", db=db, now=NOW)
    held = "OpenAI has not announced a release date for GPT-6"
    for index in range(5):
        observed = datetime(2026, 7, 23 + index, 12, 0, tzinfo=timezone.utc)
        material = index == 1
        topics.create_run(
            topic["id"],
            run_payload(
                observed_at=observed,
                change_type="major" if material else "stable",
                change_summary=(
                    "A rumoured window entered the answer." if material
                    else "Only the wording changed."
                ),
                evidence=[{
                    "type": "primary",
                    "title": "Release notes",
                    "url": f"https://openai.com/index/note-{min(index, 1)}",
                    "publisher": "OpenAI",
                }],
                opinion_map={
                    "schema_version": 1,
                    "dimensions": [
                        {
                            "label": held if index else held + " so far",
                            "type": "claim",
                            "positions": [{"stance": held, "models": ["OpenAI", "Gemini"]}],
                        },
                        {
                            "label": "The GPT-5.6 family is the current frontier line",
                            "type": "claim",
                            "positions": [{
                                "stance": "GPT-5.6 is current", "models": ["OpenAI", "Gemini"],
                            }],
                        },
                    ],
                    "models": [],
                    "shift_score": 0,
                    "shift_label": "Stable",
                    "center": [held],
                },
            ),
            actor_uid="admin", db=db, now=observed,
        )
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(topics_router.router)
    client = TestClient(app)

    page = client.get("/topics/gpt-6")

    assert page.status_code == 200
    # The finding is a sentence about the world, not a score, and it is the
    # first thing on the page after the question itself.
    assert '<h2 class="topic-finding-line' in page.text
    assert held + "." in page.text
    assert page.text.index("topic-finding-line") < page.text.index("What the answer is made of")
    assert page.text.index("What the answer is made of") < page.text.index("Read the full answer")
    # How long it has stood is stated next to the finding, not instead of it.
    assert "Unchanged through 3 checks" in page.text
    assert "Last material change on Jul 24, 2026" in page.text
    # One strip cell per check, oldest first, each one a link into that check.
    strip = page.text.split('id="topicStrip"')[1].split("</div>")[0]
    assert strip.count('<a class="topic-strip-cell') == 5
    assert strip.count('href="/topics/gpt-6?version=') == 4
    # The newest check is the page itself, so its cell carries no version.
    assert strip.count('href="/topics/gpt-6"') == 1
    # Each statement carries its own life, not one row per run.
    assert "Held 5 of 5 checks" in page.text
    assert "Restated check after check" in page.text
    # Unchanged checks are folded away instead of listed one by one.
    assert "2 checks, no material change" in page.text
    # A wording-only score step is not presented as a change event.
    assert "No check in this window was graded a material change" not in page.text
    assert "A rumoured window entered the answer." in page.text
    # Sources carry their place in the record.
    assert "Cited since Jul 24, 2026" in page.text
    assert "Sources that left the record" in page.text


def test_public_topic_history_is_ssr_and_historical_version_is_noindex(monkeypatch):
    db = FakeFirestore()
    monkeypatch.setattr(topics, "db_firestore", db)
    topic = topics.create_topic(topic_payload(), actor_uid="admin", db=db, now=NOW)
    first = topics.create_run(
        topic["id"], run_payload(), actor_uid="admin", db=db, now=NOW
    )
    topics.create_run(
        topic["id"],
        run_payload(
            agreement_score=64,
            change_type="minor",
            change_summary="The expected date moved.",
            consensus_md="The **current** consensus moved.",
        ),
        actor_uid="admin",
        db=db,
        now=NOW,
    )
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(topics_router.router)
    client = TestClient(app)

    current = client.get("/topics/gpt-6")
    historical = client.get(f"/topics/gpt-6?version={first['id']}")

    assert current.status_code == 200
    assert "The <strong>current</strong> consensus moved." in current.text
    assert current.headers["x-robots-tag"] == "index, follow"
    assert historical.status_code == 200
    assert "No confirmed release date exists." in historical.text
    assert historical.headers["x-robots-tag"] == "noindex, follow"
    assert "Return to the current consensus" in historical.text


def test_disagreement_leads_the_statement_list_and_is_labelled_as_its_own_kind(
    monkeypatch,
):
    """Where the models split is the one thing asking a single model cannot
    show. It opens the list, and every row says which kind of statement it is
    in a word, a colour and a shape -- not by its position alone."""
    db = FakeFirestore()
    monkeypatch.setattr(topics, "db_firestore", db)
    topic = topics.create_topic(topic_payload(), actor_uid="admin", db=db, now=NOW)
    for index in range(3):
        observed = datetime(2026, 7, 23 + index, 12, 0, tzinfo=timezone.utc)
        topics.create_run(
            topic["id"],
            run_payload(
                observed_at=observed,
                opinion_map={
                    "schema_version": 1,
                    "dimensions": [
                        {
                            "label": "When does the model ship?",
                            "type": "contradiction",
                            "positions": [
                                {"stance": "A late-2026 launch", "models": ["OpenAI"]},
                                {"stance": "Nothing before 2027", "models": ["Gemini"]},
                            ],
                        },
                        {
                            "label": "OpenAI has not announced a release date",
                            "type": "claim",
                            "positions": [{
                                "stance": "No date announced",
                                "models": ["OpenAI", "Gemini"],
                            }],
                        },
                    ],
                    "models": [],
                    "shift_score": 0,
                    "shift_label": "Stable",
                    "center": [],
                },
            ),
            actor_uid="admin", db=db, now=observed,
        )
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(topics_router.router)
    client = TestClient(app)

    page = client.get("/topics/gpt-6")

    assert page.status_code == 200
    assert "The models do not agree here" in page.text
    assert page.text.index("The models do not agree here") < page.text.index(
        "Restated check after check"
    )
    # Each row repeats its kind, so a linked row still says what it is.
    assert '<span class="fact-tag is-split">' in page.text
    assert '<span class="fact-tag is-settled">' in page.text
    # The finding names the disagreement without claiming a contested statement.
    assert "OpenAI has not announced a release date." in page.text
    assert "The models disagree" in page.text
    assert "1 open disagreement" in page.text
