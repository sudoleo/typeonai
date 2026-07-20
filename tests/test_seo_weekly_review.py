from datetime import datetime, timedelta, timezone

import pytest

from app.services import seo_weekly_review as weekly


NOW = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
RUN_ID = "a" * 32
PAGE_ID = "b" * 64


class Snap:
    def __init__(self, ref, data):
        self.reference = ref
        self.id = ref.id
        self.exists = data is not None
        self.data = dict(data or {})

    def to_dict(self):
        return dict(self.data)


class Doc:
    def __init__(self, db, path):
        self.db, self.path = db, tuple(path)
        self.id = self.path[-1]

    def get(self, **kwargs):
        return Snap(self, self.db.data.get(self.path))

    def set(self, data, merge=False):
        if merge:
            self.db.data.setdefault(self.path, {}).update(dict(data))
        else:
            self.db.data[self.path] = dict(data)

    def update(self, data):
        self.db.data[self.path].update(dict(data))


class Query:
    def __init__(self, collection, field=None, value=None):
        self.collection, self.field, self.value = collection, field, value

    def stream(self):
        items = self.collection.stream()
        if self.field is None:
            return items
        return [item for item in items if item.to_dict().get(self.field) == self.value]


class Collection:
    def __init__(self, db, path):
        self.db, self.path = db, tuple(path)

    def document(self, doc_id):
        return Doc(self.db, self.path + (doc_id,))

    def stream(self):
        return [
            Snap(Doc(self.db, path), data)
            for path, data in self.db.data.items()
            if len(path) == len(self.path) + 1 and path[:-1] == self.path
        ]

    def where(self, field, op, value):
        assert op == "=="
        return Query(self, field, value)


class Db:
    def __init__(self):
        self.data = {}

    def collection(self, name):
        return Collection(self, (name,))


def test_default_interval_is_seven_days_and_lease_is_persistent():
    db = Db()
    repo = weekly.WeeklyReviewRepository(db)

    config = repo.get_config(NOW)
    assert config["interval_days"] == 7
    assert repo.acquire(RUN_ID, NOW) is True
    assert repo.acquire("c" * 32, NOW + timedelta(minutes=1)) is False
    assert db.data[("app_config", "seo_weekly_review")]["lease_run_id"] == RUN_ID


def test_weekly_schedule_uses_configured_local_time():
    scheduled = weekly.next_scheduled_review(
        NOW,
        interval_days=7,
        run_time="09:00",
        timezone_name="Europe/Berlin",
        last_run_at=NOW,
    )
    assert scheduled == datetime(2026, 7, 27, 7, 0, tzinfo=timezone.utc)


class RunRepo:
    def __init__(self):
        self.config = {
            "enabled": True, "interval_days": 7, "last_run_at": None,
            "next_run_at": NOW, "lease_until": None, "lease_run_id": "",
            "run_time": "09:00", "timezone": "Europe/Berlin",
        }
        self.reviews = {}

    def get_config(self, now): return dict(self.config)
    def acquire(self, run_id, now):
        if self.config["lease_until"] and self.config["lease_until"] > now: return False
        self.config.update(lease_until=now + timedelta(minutes=45), lease_run_id=run_id)
        return True
    def create_review(self, run_id, data): self.reviews[run_id] = dict(data)
    def update_review(self, run_id, data): self.reviews[run_id].update(data)
    def finish_lease(self, run_id, finished_at, interval_days, run_time="09:00", timezone_name="Europe/Berlin"):
        self.config.update(lease_until=None, lease_run_id="", last_run_at=finished_at,
                           next_run_at=finished_at + timedelta(days=interval_days))
    def latest_review(self): return next(iter(self.reviews.values()), None)
    def get_review(self, run_id):
        return {**self.reviews[run_id], "run_id": run_id} if run_id in self.reviews else None
    def mark_page_reviewed(self, page_id, admin_uid, now): pass


class DataService:
    def __init__(self, status="success"):
        self.collection_status = status

    def collect(self): return {"status": self.collection_status, "period_end": "2026-07-17"}
    def overview(self, **kwargs): return {"final_date": "2026-07-17", "rows": []}


class RecommendationService:
    def generate(self, page_id, **kwargs):
        return {
            "recommendation": "monitor", "confidence": .8, "evidence": [],
            "safeguards": {}, "data_window": {"finalized_days": 28, "query_data_complete": True},
            "analysis_fingerprint": "fingerprint",
        }


class Judge:
    configured = True
    calls = 0

    def status(self): return {"configured": True, "model": "judge"}
    def ask(self, pages, current_topic_brief):
        self.calls += 1
        return {
            "summary": "Portfolio is stable.", "positive_patterns": [], "negative_patterns": [],
            "grouped_recommendations": [], "proposed_topic_brief": None,
            "topic_brief_reason": None, "topic_brief_evidence_page_ids": [],
        }


def build_run_service(monkeypatch, collection_status="success"):
    repo = RunRepo()
    judge = Judge()
    judge.calls = 0
    db = Db()
    monkeypatch.setattr(weekly.publisher_config, "get_config", lambda db=None: {
        **weekly.publisher_config.DEFAULT_CONFIG, "topic_brief": "Current brief"
    })
    monkeypatch.setattr(weekly.watch_service, "publisher_watch_counts", lambda db=None: {"active": 0, "paused": 0})
    service = weekly.SeoWeeklyReviewService(
        db, repository=repo, data_service=DataService(collection_status),
        recommendation_service=RecommendationService(), judge=judge, clock=lambda: NOW,
    )
    return service, repo, judge


def test_review_uses_at_most_one_portfolio_judge_call(monkeypatch):
    service, _repo, judge = build_run_service(monkeypatch)
    result = service.run(force=True)
    assert result["status"] == "completed"
    assert judge.calls == 1
    assert result["judge_called"] is True


def test_review_reuses_overview_context_without_second_metric_scan(monkeypatch):
    service, _repo, _judge = build_run_service(monkeypatch)
    page_id = "b" * 64
    context_page = {
        "page_id": page_id,
        "active": True,
        "origin": "static_page",
        "first_seen_at": NOW - timedelta(days=60),
        "dossier": {"title": "Page", "meta_description": "Description"},
    }
    service.data_service.overview = lambda **kwargs: {
        "final_date": "2026-07-17",
        "rows": [{
            "page_id": page_id,
            "url": "https://www.consens.io/page",
            "origin": "static_page",
            "metrics_7d": {},
            "metrics_28d": {},
            "status": "emerging",
            "query_data": None,
            "_analysis_context": {
                "page": context_page,
                "metrics": [],
                "query_snapshot": None,
                "final_date": NOW.date(),
            },
        }],
    }
    calls = []

    def from_context(*args, **kwargs):
        calls.append((args, kwargs))
        return {
            "recommendation": "monitor",
            "confidence": .8,
            "evidence": [],
            "safeguards": {},
            "data_window": {"finalized_days": 0, "query_data_complete": False},
            "analysis_fingerprint": "fingerprint",
        }

    service.recommendation_service.generate_from_context = from_context
    service.recommendation_service.generate = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("review must not reload page metrics")
    )

    result = service.run(force=True)

    assert result["status"] == "completed"
    assert len(calls) == 1
    assert result["pages"][0]["analysis_fingerprint"] == "fingerprint"


def test_failed_collection_never_calls_portfolio_judge(monkeypatch):
    service, _repo, judge = build_run_service(monkeypatch, "error")
    result = service.run(force=True)
    assert result["status"] == "collection_failed"
    assert result["judge_called"] is False
    assert judge.calls == 0


def test_mature_portfolio_can_persist_optional_topic_brief_suggestion(monkeypatch):
    service, _repo, judge = build_run_service(monkeypatch)
    service.data_service.overview = lambda **kwargs: {
        "final_date": "2026-07-17", "rows": [{"page_id": str(i)} for i in range(3)]
    }
    service._build_page = lambda row, now: {
        "page_id": row["page_id"], "recommendation": "protect_winner",
        "watch_status": "none", "publication_source": "", "age_days": 60,
        "data_window": {"finalized_days": 28, "query_data_complete": True},
    }
    def suggest(pages, current):
        judge.calls += 1
        return {
            "summary": "A clear pattern exists.", "positive_patterns": [], "negative_patterns": [],
            "grouped_recommendations": [], "proposed_topic_brief": "Focus on specific developer-tool launches.",
            "topic_brief_reason": "Three mature pages support this intent.",
            "topic_brief_evidence_page_ids": ["0", "1", "2"],
        }
    judge.ask = suggest
    result = service.run(force=True)
    assert result["proposed_topic_brief"] == "Focus on specific developer-tool launches."
    assert result["topic_brief_evidence_page_ids"] == ["0", "1", "2"]


def test_deterministic_recommendations_are_grouped_without_llm_override():
    pages = [
        {"page_id": "1", "recommendation": "protect_winner", "watch_status": "active", "publication_source": "", "age_days": 90},
        {"page_id": "2", "recommendation": "monitor", "watch_status": "active", "publication_source": "scheduled_publisher", "age_days": 90},
        {"page_id": "3", "recommendation": "refresh_content", "watch_status": "none", "publication_source": "", "age_days": 90},
    ]
    service = object.__new__(weekly.SeoWeeklyReviewService)
    groups = service._group_pages(pages)
    assert groups["keep_indexed"] == ["1"]
    assert groups["pause_watch_only"] == ["2"]
    assert groups["manual_improvement"] == ["3"]


class ActionRepo(RunRepo):
    def __init__(self, page):
        super().__init__()
        self.reviews[RUN_ID] = {"pages": [page], "groups": {name: [] for name in weekly.GROUPS}, "applied_actions": []}
        self.reviews[RUN_ID]["groups"][page["group"]] = [page["page_id"]]
        self.reviewed = []

    def mark_page_reviewed(self, page_id, admin_uid, now): self.reviewed.append(page_id)


def action_page(group, *, indexed=True, watch_status="active", source="scheduled_publisher"):
    return {
        "page_id": PAGE_ID, "page_type": "share", "share_id": "S" * 16,
        "share_status": "active", "publication_source": source, "indexed": indexed,
        "watch_id": "watch-1", "watch_status": watch_status, "group": group,
        "recommendation": "noindex_candidate", "analysis_fingerprint": "fingerprint",
        "data_window": {"query_data_complete": True}, "safeguards": {"safe": True},
    }


def build_action_service(monkeypatch, page):
    share = {"status": "active", "publication_source": page["publication_source"], "indexed": page["indexed"]}
    watch = {"id": "watch-1", "status": page["watch_status"]}
    deleted = []
    monkeypatch.setattr(weekly.share_snapshots, "get_share", lambda share_id, db=None: share)
    monkeypatch.setattr(weekly.watch_service, "find_watch_for_share", lambda share_id, db=None: watch)
    def set_status(watch_id, status, db=None):
        watch["status"] = status
        return dict(watch)
    monkeypatch.setattr(weekly.watch_service, "set_watch_status_admin", set_status)
    def moderate(share_id, indexed, **kwargs):
        share["indexed"] = indexed
        return dict(share)
    monkeypatch.setattr(weekly.share_snapshots, "moderate_share", moderate)
    monkeypatch.setattr(weekly.share_snapshots, "hard_delete_share", lambda share_id, db=None: deleted.append(share_id) or {"share_id": share_id})
    monkeypatch.setattr(weekly.publisher_config, "get_config", lambda db=None: {**weekly.publisher_config.DEFAULT_CONFIG, "max_active_publisher_watches": 12})
    monkeypatch.setattr(weekly.watch_service, "publisher_watch_counts", lambda db=None: {"active": 0, "paused": 1})
    repo = ActionRepo(page)
    class SafeRecommendationService:
        def generate(self, page_id, **kwargs):
            return {
                "recommendation": "noindex_candidate",
                "analysis_fingerprint": "fingerprint",
                "safeguards": {"safe": True},
                "data_window": {"query_data_complete": True},
            }
    service = weekly.SeoWeeklyReviewService(
        Db(), repository=repo, data_service=DataService(),
        recommendation_service=SafeRecommendationService(), judge=Judge(), clock=lambda: NOW,
    )
    return service, share, watch, deleted


def test_pause_and_resume_watch_never_change_indexing(monkeypatch):
    pause = action_page("pause_watch_only", indexed=True, watch_status="active")
    service, share, watch, _ = build_action_service(monkeypatch, pause)
    service.apply(RUN_ID, admin_uid="admin", group="pause_watch_only")
    assert watch["status"] == "paused"
    assert share["indexed"] is True

    resume = action_page("resume_watch", indexed=False, watch_status="paused")
    service, share, watch, _ = build_action_service(monkeypatch, resume)
    service.apply(RUN_ID, admin_uid="admin", group="resume_watch")
    assert watch["status"] == "active"
    assert share["indexed"] is False


def test_noindex_only_leaves_watch_and_combined_reports_both_steps(monkeypatch):
    page = action_page("noindex_only", indexed=True, watch_status="active")
    service, share, watch, _ = build_action_service(monkeypatch, page)
    result = service.apply(RUN_ID, admin_uid="admin", group="noindex_only")
    assert share["indexed"] is False
    assert watch["status"] == "active"
    assert [step["action"] for step in result["results"][0]["steps"]] == ["noindex", "mark_reviewed"]

    page = action_page("noindex_and_pause_watch", indexed=True, watch_status="active")
    service, share, watch, _ = build_action_service(monkeypatch, page)
    result = service.apply(RUN_ID, admin_uid="admin", group="noindex_and_pause_watch")
    assert share["indexed"] is False and watch["status"] == "paused"
    assert [step["action"] for step in result["results"][0]["steps"]][:2] == ["pause_watch", "noindex"]

    def fail_pause(*args, **kwargs):
        raise weekly.watch_service.WatchError("failed", "Pause failed")
    monkeypatch.setattr(weekly.watch_service, "set_watch_status_admin", fail_pause)
    page = action_page("noindex_and_pause_watch", indexed=True, watch_status="active")
    service, share, _watch, _ = build_action_service(monkeypatch, page)
    monkeypatch.setattr(weekly.watch_service, "set_watch_status_admin", fail_pause)
    result = service.apply(RUN_ID, admin_uid="admin", group="noindex_and_pause_watch")
    assert result["results"][0]["status"] == "partial"
    assert result["results"][0]["steps"][0]["status"] == "error"
    assert result["results"][0]["steps"][1]["status"] == "noindex"
    assert share["indexed"] is False


def test_apply_all_never_includes_delete_and_delete_requires_publisher_lineage(monkeypatch):
    page = action_page("delete_candidate")
    service, _share, _watch, deleted = build_action_service(monkeypatch, page)
    assert service.apply(RUN_ID, admin_uid="admin", apply_all=True)["results"] == []
    assert deleted == []

    user_page = action_page("delete_candidate", source="")
    service, _share, _watch, deleted = build_action_service(monkeypatch, user_page)
    result = service.apply(RUN_ID, admin_uid="admin", group="delete_candidate", confirm_delete=True)
    assert result["results"][0]["status"] == "error"
    assert deleted == []

    service, _share, _watch, deleted = build_action_service(monkeypatch, page)
    result = service.apply(RUN_ID, admin_uid="admin", group="delete_candidate", confirm_delete=True)
    assert result["results"][0]["status"] == "success"
    assert deleted == ["S" * 16]


def test_topic_brief_accept_preserves_other_config_and_detects_manual_change(monkeypatch):
    page = action_page("keep_indexed")
    repo = ActionRepo(page)
    repo.reviews[RUN_ID].update(current_topic_brief="Old", proposed_topic_brief="New")
    service = weekly.SeoWeeklyReviewService(Db(), repository=repo, clock=lambda: NOW)
    current = {**weekly.publisher_config.DEFAULT_CONFIG, "topic_brief": "Old", "auto_index": False}
    saved = []
    monkeypatch.setattr(weekly.publisher_config, "get_config", lambda db=None: dict(current))
    monkeypatch.setattr(weekly.publisher_config, "save_config", lambda data, updated_by, db=None: saved.append(dict(data)) or data)
    result = service.accept_topic_brief(RUN_ID, admin_uid="admin")
    assert result["config"]["topic_brief"] == "New"
    assert saved[0]["auto_index"] is False

    current["topic_brief"] = "Manual edit"
    with pytest.raises(weekly.ReviewError, match="changed"):
        service.accept_topic_brief(RUN_ID, admin_uid="admin")
