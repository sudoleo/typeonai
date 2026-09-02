from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routers import pages as pages_router
from app.core.rate_limit import limiter


class LeaderboardSnapshot:
    def __init__(self, document_id, selections):
        self.id = document_id
        self._selections = selections

    def to_dict(self):
        return {"BestModel": self._selections}


class LeaderboardCollection:
    def stream(self):
        return [
            LeaderboardSnapshot("Anthropic", 8),
            LeaderboardSnapshot("Claude", 5),
            LeaderboardSnapshot("OpenAI-Pro", 11),
            LeaderboardSnapshot("Mistral", 0),
        ]


class VoteSnapshot:
    def __init__(self, model, *, vote_type="BestModel"):
        self._data = {
            "model": model,
            "vote_type": vote_type,
            "created_at": datetime(2026, 8, 31, 12, tzinfo=timezone.utc),
        }

    def to_dict(self):
        return dict(self._data)


class VoteCollection:
    def __init__(self):
        self.filter = None

    def where(self, *, filter):
        self.filter = filter
        return self

    def stream(self):
        assert self.filter.field_path == "created_at"
        assert self.filter.op_string == ">="
        return [
            VoteSnapshot("OpenAI"),
            VoteSnapshot("Kimi-Pro"),
            VoteSnapshot("GLM"),
            VoteSnapshot("Grok", vote_type="WorstModel"),
        ]


class LeaderboardDb:
    def __init__(self):
        self.votes = VoteCollection()

    def collection(self, name):
        if name == "leaderboard":
            return LeaderboardCollection()
        assert name == "model_votes"
        return self.votes


def test_public_model_leaderboard_aggregates_aliases_and_sets_cache(monkeypatch):
    monkeypatch.setattr(pages_router, "db_firestore", LeaderboardDb())
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(pages_router.router)

    response = TestClient(app).get("/api/model-leaderboard")

    assert response.status_code == 200
    data = response.json()
    assert data["period"] == "all"
    assert data["period_start"] is None
    assert data["total_selections"] == 24
    assert [(row["family"], row["selections"]) for row in data["rows"]] == [
        ("Anthropic / Claude", 13),
        ("OpenAI / ChatGPT", 11),
        ("Mistral", 0),
        ("Google / Gemini", 0),
        ("DeepSeek", 0),
        ("xAI / Grok", 0),
        ("Moonshot AI / Kimi", 0),
        ("Z.ai / GLM", 0),
        ("Meta / Muse", 0),
    ]
    kimi = next(row for row in data["rows"] if row["family"] == "Moonshot AI / Kimi")
    glm = next(row for row in data["rows"] if row["family"] == "Z.ai / GLM")
    meta = next(row for row in data["rows"] if row["family"] == "Meta / Muse")
    assert kimi["icon"].endswith("/kimi.svg")
    assert glm["icon"].endswith("/zai.svg")
    assert meta["icon"].endswith("/meta.svg")
    assert kimi["available_since"] == glm["available_since"] == "2026-08-31"
    # Jede spaeter ergaenzte Familie traegt ihr eigenes Startdatum, sonst
    # liest sich ein niedriger Stand wie ein Ergebnis.
    assert meta["available_since"] == "2026-09-02"
    assert response.headers["cache-control"] == "public, max-age=60, stale-while-revalidate=300"


def test_public_model_leaderboard_supports_shared_window(monkeypatch):
    db = LeaderboardDb()
    monkeypatch.setattr(pages_router, "db_firestore", db)
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(pages_router.router)

    response = TestClient(app).get("/api/model-leaderboard?period=since-2026-08-31")

    assert response.status_code == 200
    data = response.json()
    assert data["period"] == "since-2026-08-31"
    assert data["period_start"] == "2026-08-31"
    assert data["total_selections"] == 3
    assert [(row["family"], row["selections"]) for row in data["rows"][:3]] == [
        ("OpenAI / ChatGPT", 1),
        ("Moonshot AI / Kimi", 1),
        ("Z.ai / GLM", 1),
    ]


def test_public_model_leaderboard_rejects_unknown_period(monkeypatch):
    monkeypatch.setattr(pages_router, "db_firestore", LeaderboardDb())
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(pages_router.router)

    response = TestClient(app).get("/api/model-leaderboard?period=yesterday")

    assert response.status_code == 400
