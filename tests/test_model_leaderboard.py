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


class LeaderboardDb:
    def collection(self, name):
        assert name == "leaderboard"
        return LeaderboardCollection()


def test_public_model_leaderboard_aggregates_aliases_and_sets_cache(monkeypatch):
    monkeypatch.setattr(pages_router, "db_firestore", LeaderboardDb())
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(pages_router.router)

    response = TestClient(app).get("/api/model-leaderboard")

    assert response.status_code == 200
    assert response.json() == {
        "rows": [
            {"family": "Anthropic / Claude", "selections": 13},
            {"family": "OpenAI / ChatGPT", "selections": 11},
        ],
        "total_selections": 24,
    }
    assert response.headers["cache-control"] == "public, max-age=60, stale-while-revalidate=300"
