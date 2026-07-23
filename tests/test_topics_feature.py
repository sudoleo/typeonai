from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routers import pages as pages_router
from app.api.routers import topics as topics_router
from app.core.rate_limit import limiter
from app.services import mailer, topic_runner, topics


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

    def get(self):
        return FakeSnapshot(self, self.db.documents.get(self.path))

    def set(self, data, merge=False):
        if merge and self.path in self.db.documents:
            self.db.documents[self.path].update(dict(data))
        else:
            self.db.documents[self.path] = dict(data)

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

    def collection(self, name):
        return FakeCollection(self, (name,))


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

    token = topics.make_unsubscribe_token(topic["id"], confirmed["email"], now=NOW)
    topics.unsubscribe_follow(token, db=db)
    assert not any(path[0] == topics.FOLLOWERS_COLLECTION for path in db.documents)


def test_topic_notification_delivery_is_deduplicated_and_multipart():
    db = FakeFirestore()
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


def test_topic_templates_expose_timeline_evidence_follow_and_admin_controls():
    detail = (ROOT / "templates" / "topic.html").read_text(encoding="utf-8")
    hub = (ROOT / "templates" / "topics.html").read_text(encoding="utf-8")
    admin = (ROOT / "templates" / "admin.html").read_text(encoding="utf-8")
    nav = (ROOT / "templates" / "partials" / "public_nav.html").read_text(encoding="utf-8")

    assert 'class="topic-timeline"' in detail
    assert 'id="evidence"' in detail
    assert 'id="topicFollowForm"' in detail
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


def test_automatic_topic_run_researches_sources_and_builds_timeline_point():
    db = FakeFirestore()
    topic = topics.create_topic(topic_payload(evidence=[]), actor_uid="admin", db=db, now=NOW)
    claimed = {
        **topic,
        "current_run_id": "automatic-run",
        "claimed_until": NOW,
    }

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
    assert run["evidence"][0]["type"] == "official"
    assert topics.get_topic(topic["id"], db=db)["last_run_status"] == "success"


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
