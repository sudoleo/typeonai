from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routers import admin as admin_router
from app.services import google_search_console as gsc
from app.services import seo_data
from app.services import seo_dossier
from app.services import seo_recommendation
from app.services.seo_repository import FirestoreSeoRepository, page_id_for_url


NOW = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
PAGE_URL = "https://www.consens.io/consensus-engine"


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

    def collection(self, name):
        return FakeCollection(self.db, self.path + (name,))


class FakeCollection:
    def __init__(self, db, path):
        self.db = db
        self.path = tuple(path)

    def document(self, doc_id):
        return FakeDocument(self.db, self.path + (doc_id,))

    def stream(self):
        snapshots = []
        for path, data in self.db.documents.items():
            if len(path) == len(self.path) + 1 and path[:-1] == self.path:
                snapshots.append(FakeSnapshot(FakeDocument(self.db, path), data))
        return snapshots


class FakeFirestore:
    def __init__(self):
        self.documents = {}
        self.get_all_calls = []

    def collection(self, name):
        return FakeCollection(self, (name,))

    def get_all(self, references):
        references = list(references)
        self.get_all_calls.append(len(references))
        return [reference.get() for reference in references]


class FakeSearchConsoleClient:
    def __init__(self, rows=None):
        self.rows = list(rows or [])
        self.calls = []

    def query(self, start_date, end_date):
        self.calls.append((start_date, end_date))
        rows = [
            row for row in self.rows
            if start_date.isoformat() <= row["keys"][1] <= end_date.isoformat()
        ]
        return {"rows": rows, "requests": 1, "truncated": False}


def build_service(db, client):
    return seo_data.SeoDataService(
        repository=FirestoreSeoRepository(db),
        client_factory=lambda: client,
        page_discovery=lambda: [
            {"url": PAGE_URL, "origin": "static_page", "share_id": None}
        ],
        clock=lambda: NOW,
    )


def daily_rows(final_date, *, impressions, clicks, position=5.0, days=28):
    rows = []
    for offset in range(days):
        day = final_date - timedelta(days=offset)
        rows.append({
            "date": day.isoformat(),
            "clicks": clicks,
            "impressions": impressions,
            "ctr": clicks / impressions if impressions else 0.0,
            "position": position if impressions else None,
        })
    return rows


def test_configuration_errors_are_safe(monkeypatch):
    monkeypatch.delenv("GSC_SITE_URL", raising=False)
    monkeypatch.delenv("GSC_SERVICE_ACCOUNT_JSON", raising=False)
    missing = gsc.configuration_status()
    assert missing["configured"] is False
    assert missing["status"] == "not_configured"

    secret_marker = "DO_NOT_ECHO_THIS_SECRET"
    monkeypatch.setenv("GSC_SITE_URL", "https://www.consens.io/")
    monkeypatch.setenv("GSC_SERVICE_ACCOUNT_JSON", "{" + secret_marker)
    invalid = gsc.configuration_status()
    assert invalid["configured"] is False
    assert invalid["status"] == "invalid_configuration"
    assert secret_marker not in invalid["message"]

    # Even valid-looking inline JSON is a path value and must not be parsed
    # directly from the environment variable.
    monkeypatch.setenv("GSC_SERVICE_ACCOUNT_JSON", '{"type":"service_account"}')
    inline_json = gsc.configuration_status()
    assert inline_json["status"] == "invalid_configuration"

    monkeypatch.setenv("GSC_SITE_URL", "sc-domain:")
    invalid_site = gsc.configuration_status()
    assert invalid_site["status"] == "invalid_configuration"


def test_service_account_variable_is_a_relative_repository_root_path(monkeypatch, tmp_path):
    credential_file = tmp_path / "local-gsc-service-account.json"
    credential_file.write_text('{"type":"service_account","project_id":"test"}', encoding="utf-8")
    monkeypatch.setattr(gsc, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setenv("GSC_SITE_URL", "sc-domain:consens.io")
    monkeypatch.setenv("GSC_SERVICE_ACCOUNT_JSON", credential_file.name)

    config = gsc.SearchConsoleConfig.from_env()

    assert config.site_url == "sc-domain:consens.io"
    assert config.service_account_info["project_id"] == "test"
    assert not hasattr(config, "credentials_path")


def test_client_builds_credentials_with_readonly_scope_only(monkeypatch):
    captured = {}

    class Credentials:
        @staticmethod
        def from_service_account_info(info, scopes):
            captured["info"] = info
            captured["scopes"] = scopes
            return object()

    monkeypatch.setattr(gsc.service_account, "Credentials", Credentials)
    monkeypatch.setattr(gsc, "AuthorizedSession", lambda credentials: object())

    gsc.GoogleSearchConsoleClient(
        gsc.SearchConsoleConfig("sc-domain:consens.io", {"type": "service_account"})
    )

    assert captured["scopes"] == [gsc.READONLY_SCOPE]


def test_successful_collection_persists_90_days_with_origin_and_source():
    db = FakeFirestore()
    final_date = date(2026, 7, 17)
    client = FakeSearchConsoleClient([{
        "keys": [PAGE_URL, final_date.isoformat()],
        "clicks": 4,
        "impressions": 100,
        "ctr": 0.04,
        "position": 8.5,
    }])

    result = build_service(db, client).collect()

    assert result["status"] == "success"
    assert result["metrics_written"] == 90
    assert result["days_collected"] == 90
    metric_path = (
        "seo_pages", page_id_for_url(PAGE_URL), "daily_metrics", final_date.isoformat()
    )
    stored = db.documents[metric_path]
    assert stored["url"] == PAGE_URL
    assert stored["clicks"] == 4
    assert stored["impressions"] == 100
    assert stored["origin"] == "static_page"
    assert stored["share_id"] is None
    assert stored["source"] == "google_search_console"
    assert stored["collected_at"] == NOW


def test_collection_is_idempotent_and_skips_already_finalized_days():
    db = FakeFirestore()
    client = FakeSearchConsoleClient()
    service = build_service(db, client)

    first = service.collect()
    calls_after_first = list(client.calls)
    batch_reads_after_first = list(db.get_all_calls)
    service.repository.existing_metric_dates = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("coverage watermark should avoid rescanning daily metrics")
    )
    second = service.collect()

    assert first["metrics_written"] == 90
    assert second["metrics_written"] == 0
    assert second["days_collected"] == 0
    assert client.calls == calls_after_first
    assert db.get_all_calls == batch_reads_after_first
    metric_paths = [path for path in db.documents if "daily_metrics" in path]
    assert len(metric_paths) == 90
    page = db.documents[("seo_pages", page_id_for_url(PAGE_URL))]
    assert page["metrics_coverage_start"] == "2026-04-19"
    assert page["metrics_coverage_end"] == "2026-07-17"


def test_truncated_ranges_do_not_persist_omitted_rows_as_zeroes():
    class TruncatedClient(FakeSearchConsoleClient):
        def query(self, start_date, end_date):
            self.calls.append((start_date, end_date))
            rows = []
            if start_date <= date(2026, 7, 17) <= end_date:
                rows = [{
                    "keys": [PAGE_URL, "2026-07-17"],
                    "clicks": 1,
                    "impressions": 5,
                    "ctr": 0.2,
                    "position": 4,
                }]
            return {"rows": rows, "requests": 10, "truncated": True}

    db = FakeFirestore()
    result = build_service(db, TruncatedClient()).collect()

    assert result["status"] == "partial"
    assert result["days_requested"] == 90
    assert result["days_collected"] == 0
    metric_paths = [path for path in db.documents if "daily_metrics" in path]
    assert len(metric_paths) == 1
    assert metric_paths[0][-1] == "2026-07-17"


def test_date_boundaries_exclude_the_three_day_delay_and_chunk_requests():
    db = FakeFirestore()
    client = FakeSearchConsoleClient()

    result = build_service(db, client).collect()

    assert result["period_start"] == "2026-04-19"
    assert result["period_end"] == "2026-07-17"
    assert client.calls == [
        (date(2026, 4, 19), date(2026, 5, 19)),
        (date(2026, 5, 20), date(2026, 6, 19)),
        (date(2026, 6, 20), date(2026, 7, 17)),
    ]


def test_not_configured_collection_records_safe_result():
    db = FakeFirestore()

    def fail_config():
        raise gsc.SearchConsoleError("not_configured", "GSC configuration is missing.")

    service = seo_data.SeoDataService(
        repository=FirestoreSeoRepository(db),
        client_factory=fail_config,
        page_discovery=lambda: [],
        clock=lambda: NOW,
    )
    result = service.collect()

    assert result["status"] == "not_configured"
    runs = db.collection("seo_collection_runs").stream()
    assert len(runs) == 1
    assert runs[0].to_dict()["message"] == "GSC configuration is missing."


def test_search_console_http_pagination_and_final_data_state():
    class Response:
        status_code = 200

        def __init__(self, rows):
            self.rows = rows

        def json(self):
            return {"rows": self.rows}

    class Session:
        def __init__(self):
            self.payloads = []

        def post(self, url, json, timeout):
            self.payloads.append(json)
            if json["startRow"] == 0:
                return Response([{"keys": [PAGE_URL, "2026-07-16"]}] * 2)
            return Response([{"keys": [PAGE_URL, "2026-07-17"]}])

    session = Session()
    config = gsc.SearchConsoleConfig("sc-domain:consens.io", {"type": "service_account"})
    client = gsc.GoogleSearchConsoleClient(config, session=session, row_limit=2, max_pages=3)

    result = client.query(date(2026, 7, 16), date(2026, 7, 17))

    assert len(result["rows"]) == 3
    assert result["requests"] == 2
    assert [payload["startRow"] for payload in session.payloads] == [0, 2]
    assert all(payload["dataState"] == "final" for payload in session.payloads)
    assert gsc.READONLY_SCOPE == "https://www.googleapis.com/auth/webmasters.readonly"


def test_connection_check_returns_only_sanitized_status():
    class Response:
        status_code = 200

        def json(self):
            return {"siteUrl": "sc-domain:consens.io", "permissionLevel": "siteFullUser"}

    class Session:
        def __init__(self):
            self.calls = []

        def get(self, url, timeout):
            self.calls.append((url, timeout))
            return Response()

    session = Session()
    client = gsc.GoogleSearchConsoleClient(
        gsc.SearchConsoleConfig("sc-domain:consens.io", {"type": "service_account"}),
        session=session,
    )

    result = client.check_connection()

    assert result == {
        "configured": True,
        "connected": True,
        "status": "connected",
        "message": "Search Console connection successful.",
    }
    assert "sc-domain%3Aconsens.io" in session.calls[0][0]
    assert "siteUrl" not in result


def test_connection_failure_never_exposes_google_body_or_credential_path():
    secret_marker = "PRIVATE_KEY_AND_PATH_MUST_NOT_ESCAPE"

    class Response:
        status_code = 403

        def json(self):
            return {"error": secret_marker}

    class Session:
        def get(self, url, timeout):
            return Response()

    client = gsc.GoogleSearchConsoleClient(
        gsc.SearchConsoleConfig("sc-domain:consens.io", {"type": "service_account"}),
        session=Session(),
    )

    try:
        client.check_connection()
        assert False, "expected SearchConsoleError"
    except gsc.SearchConsoleError as exc:
        assert exc.code == "permission_denied"
        assert secret_marker not in exc.safe_message


def test_status_classification_rules():
    final_date = date(2026, 7, 17)
    assert seo_data.classify_status([], final_date) == "insufficient_data"
    assert seo_data.classify_status(daily_rows(final_date, impressions=0, clicks=0), final_date) == "invisible"
    assert seo_data.classify_status(daily_rows(final_date, impressions=10, clicks=1, position=5), final_date) == "winner"
    assert seo_data.classify_status(daily_rows(final_date, impressions=10, clicks=0, position=12), final_date) == "opportunity"
    assert seo_data.classify_status(daily_rows(final_date, impressions=2, clicks=0, position=30), final_date) == "emerging"

    declining = daily_rows(final_date, impressions=10, clicks=2, position=8, days=14)
    for row in declining[:7]:
        row.update({"impressions": 2, "clicks": 0, "ctr": 0.0})
    assert seo_data.classify_status(declining, final_date) == "declining"


def test_admin_endpoints_require_admin(monkeypatch):
    class StubSeoService:
        def overview(self):
            return {"configuration": {"configured": False}, "rows": []}

        def collect(self):
            return {"status": "success"}

        def check_connection(self):
            return {"configured": True, "connected": True, "status": "connected"}

    class StubContentJudge:
        def status(self):
            return {"configured": False, "status": "not_configured"}

    class StubRecommendationService:
        content_judge = StubContentJudge()

        def generate(self, page_id):
            return {"page_id": page_id, "recommendation": "monitor"}

        def ask_content_judge(self, page_id):
            return {"page_id": page_id, "recommendation": "monitor", "llm_evaluation": {}}

    class StubWeeklyReviewService:
        def status(self): return {"config": {"enabled": True, "interval_days": 7}}
        def save_config(self, **kwargs): return self.status()
        def run(self, **kwargs): return {"status": "completed", "run_id": "a" * 32}
        def preview(self, run_id, **kwargs): return {"run_id": run_id, "pages": []}
        def apply(self, run_id, **kwargs): return {"run_id": run_id, "results": []}
        def accept_topic_brief(self, run_id, **kwargs): return {"status": "success"}
        def reject_topic_brief(self, run_id, **kwargs): return {"status": "success"}
        def record_editorial_decision(self, run_id, **kwargs): return {"status": "success"}

    monkeypatch.setattr(admin_router, "seo_data_service", StubSeoService())
    monkeypatch.setattr(admin_router, "seo_recommendation_service", StubRecommendationService())
    monkeypatch.setattr(admin_router, "seo_weekly_review_service", StubWeeklyReviewService())
    monkeypatch.setattr(admin_router, "verify_user_token", lambda token: "uid-1")
    monkeypatch.setattr(admin_router, "is_user_admin", lambda uid: False)
    app = FastAPI()
    app.include_router(admin_router.router)
    client = TestClient(app)

    assert client.get("/api/admin/seo").status_code == 401
    assert client.get("/api/admin/seo", headers={"Authorization": "Bearer token"}).status_code == 403
    assert client.post("/api/admin/seo/check", headers={"Authorization": "Bearer token"}).status_code == 403
    assert client.post("/api/admin/seo/collect", headers={"Authorization": "Bearer token"}).status_code == 403
    assert client.get("/api/admin/seo/review", headers={"Authorization": "Bearer token"}).status_code == 403
    assert client.put("/api/admin/seo/review/config", headers={"Authorization": "Bearer token"}, json={"enabled": True, "interval_days": 7}).status_code == 403
    assert client.post("/api/admin/seo/review/run", headers={"Authorization": "Bearer token"}).status_code == 403
    assert client.post(f"/api/admin/seo/reviews/{'a' * 32}/preview", headers={"Authorization": "Bearer token"}, json={}).status_code == 403
    assert client.post(f"/api/admin/seo/reviews/{'a' * 32}/apply", headers={"Authorization": "Bearer token"}, json={}).status_code == 403
    assert client.post(f"/api/admin/seo/reviews/{'a' * 32}/topic-brief/accept", headers={"Authorization": "Bearer token"}).status_code == 403
    assert client.post(f"/api/admin/seo/reviews/{'a' * 32}/topic-brief/reject", headers={"Authorization": "Bearer token"}).status_code == 403
    assert client.post(
        f"/api/admin/seo/reviews/{'a' * 32}/editorial-decision",
        headers={"Authorization": "Bearer token"},
        json={"page_id": "b" * 64, "decision": "keep_as_is"},
    ).status_code == 403
    page_id = "a" * 64
    assert client.post(
        f"/api/admin/seo/pages/{page_id}/recommendation",
        headers={"Authorization": "Bearer token"},
    ).status_code == 403
    assert client.post(
        f"/api/admin/seo/pages/{page_id}/content-judge",
        headers={"Authorization": "Bearer token"},
    ).status_code == 403

    monkeypatch.setattr(admin_router, "is_user_admin", lambda uid: True)
    assert client.get("/api/admin/seo", headers={"Authorization": "Bearer token"}).status_code == 200
    assert client.post("/api/admin/seo/check", headers={"Authorization": "Bearer token"}).status_code == 200
    assert client.post("/api/admin/seo/collect", headers={"Authorization": "Bearer token"}).status_code == 200
    assert client.get("/api/admin/seo/review", headers={"Authorization": "Bearer token"}).status_code == 200
    assert client.put("/api/admin/seo/review/config", headers={"Authorization": "Bearer token"}, json={"enabled": True, "interval_days": 7}).status_code == 200
    assert client.post("/api/admin/seo/review/run", headers={"Authorization": "Bearer token"}).status_code == 200
    assert client.post(f"/api/admin/seo/reviews/{'a' * 32}/preview", headers={"Authorization": "Bearer token"}, json={}).status_code == 200
    assert client.post(f"/api/admin/seo/reviews/{'a' * 32}/apply", headers={"Authorization": "Bearer token"}, json={}).status_code == 200
    assert client.post(f"/api/admin/seo/reviews/{'a' * 32}/topic-brief/accept", headers={"Authorization": "Bearer token"}).status_code == 200
    assert client.post(f"/api/admin/seo/reviews/{'a' * 32}/topic-brief/reject", headers={"Authorization": "Bearer token"}).status_code == 200
    assert client.post(
        f"/api/admin/seo/reviews/{'a' * 32}/editorial-decision",
        headers={"Authorization": "Bearer token"},
        json={"page_id": "b" * 64, "decision": "keep_as_is"},
    ).status_code == 200
    assert client.post(
        f"/api/admin/seo/pages/{page_id}/recommendation",
        headers={"Authorization": "Bearer token"},
    ).status_code == 200
    assert client.post(
        f"/api/admin/seo/pages/{page_id}/content-judge",
        headers={"Authorization": "Bearer token"},
    ).status_code == 200


def test_admin_seo_collect_action_is_hidden_until_admin_request_succeeds():
    root = Path(__file__).resolve().parents[1]
    template = (root / "templates" / "admin.html").read_text(encoding="utf-8")
    main_source = (root / "main.py").read_text(encoding="utf-8")

    assert 'data-tab="seo"' in template
    assert 'id="collectSeoBtn" type="button" class="admin-btn" hidden' in template
    assert 'id="checkSeoConnectionBtn" type="button" class="admin-btn secondary" hidden' in template
    assert "shareAdminRequest('GET', '/api/admin/seo')" in template
    assert "shareAdminRequest('POST', '/api/admin/seo/check', {})" in template
    assert "Generate recommendation" in template
    assert "Ask content judge" in template
    assert "/api/admin/seo/pages/${encodeURIComponent(pageId)}/${suffix}" in template
    assert 'id="runSeoReviewBtn"' in template
    assert "'/api/admin/seo/review/run'" in template
    assert "/api/admin/seo/reviews/${encodeURIComponent(runId)}/preview" in template
    assert "/api/admin/seo/reviews/${encodeURIComponent(runId)}/apply" in template
    assert "Apply all safe recommendations" in template
    assert "Accept suggested Topic Brief" in template
    assert "Reject and keep current" in template
    assert "Confirm decision" in template
    assert 'id="seoReviewTime" type="time"' in template
    assert 'id="seoReviewTimezone"' in template
    assert "Completed in this review" in template
    assert "This only records that you reviewed these pages" in template
    assert 'os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", "gen-lang-client-0234219247-53b2b1c0e355.json")' in main_source
    assert "GSC_SERVICE_ACCOUNT_JSON" not in main_source


def recommendation_page(*, observed_days=61):
    return {
        "page_id": page_id_for_url(PAGE_URL),
        "url": PAGE_URL,
        "origin": "static_page",
        "active": True,
        "first_seen_at": NOW - timedelta(days=observed_days),
        "dossier": {
            "title": "How the Consensus Engine Works | consens.io",
            "meta_description": "A technical explanation.",
            "content_summary": "How several model answers are compared.",
            "technical_uncertainties": [],
            "source_freshness": {"source_count": 0, "snapshot_at": None},
            "watch_freshness": {"last_checked_at": None, "last_material_change_at": None},
        },
    }


def test_static_dossier_contains_bounded_content_and_freshness_fields():
    dossier = seo_dossier.build_static_dossier(PAGE_URL, "2026-07-09")

    assert dossier["title"] == "How the Consensus Engine Works | consens.io"
    assert dossier["meta_description"].startswith("A clear, technical explanation")
    assert dossier["last_content_change_at"] == "2026-07-09"
    assert len(dossier["content_summary"]) <= seo_dossier.MAX_CONTENT_SUMMARY_CHARS
    assert dossier["source_freshness"] == {"source_count": 0, "snapshot_at": None}
    assert dossier["watch_freshness"]["last_checked_at"] is None
    assert dossier["technical_uncertainties"] == []


def test_share_dossier_keeps_only_a_bounded_representation(monkeypatch):
    created = NOW - timedelta(days=90)
    checked = NOW - timedelta(days=1)
    changed = NOW - timedelta(days=5)
    monkeypatch.setattr(seo_dossier.share_snapshots, "get_share", lambda share_id, db=None: {
        "question": "What changed?",
        "consensus_md": "A" * 10_000,
        "created_at": created,
        "answered_at": created.isoformat(),
        "last_watch_run_at": checked,
        "sources": [{"url": "https://example.com"}],
    })
    monkeypatch.setattr(
        seo_dossier.share_snapshots,
        "list_watch_history",
        lambda share_id, db=None, max_items=100: [{"ts": changed, "changed": True}],
    )

    dossier = seo_dossier.build_share_dossier("abc123")

    assert len(dossier["content_representation"]) <= seo_dossier.MAX_SHARE_CONTENT_REPRESENTATION_CHARS
    assert dossier["source_freshness"]["source_count"] == 1
    assert dossier["watch_freshness"]["last_checked_at"] == checked.isoformat()
    assert dossier["last_content_change_at"] == changed.isoformat()
    assert dossier["technical_uncertainties"] == []


def complete_query_snapshot(final_date=date(2026, 7, 17), *, partial=False):
    return {
        "period_start": (final_date - timedelta(days=27)).isoformat(),
        "period_end": final_date.isoformat(),
        "complete": not partial,
        "partial": partial,
        "coverage": "top_queries_only",
        "top_queries": [],
    }


def test_noindex_candidate_requires_every_safeguard_not_just_invisible_status():
    final_date = date(2026, 7, 17)
    metrics = daily_rows(final_date, impressions=0, clicks=0)

    eligible = seo_recommendation.deterministic_recommendation(
        recommendation_page(observed_days=61),
        metrics,
        complete_query_snapshot(),
        final_date=final_date,
        now=NOW,
    )
    too_young = seo_recommendation.deterministic_recommendation(
        recommendation_page(observed_days=59),
        metrics,
        complete_query_snapshot(),
        final_date=final_date,
        now=NOW,
    )
    missing_queries = seo_recommendation.deterministic_recommendation(
        recommendation_page(observed_days=61), metrics, None,
        final_date=final_date, now=NOW,
    )
    partial_queries = seo_recommendation.deterministic_recommendation(
        recommendation_page(observed_days=61), metrics, complete_query_snapshot(partial=True),
        final_date=final_date, now=NOW,
    )

    assert eligible["recommendation"] == "noindex_candidate"
    assert all(eligible["safeguards"].values())
    assert too_young["status_class"] == "invisible"
    assert too_young["recommendation"] == "monitor"
    assert missing_queries["recommendation"] == "monitor"
    assert partial_queries["recommendation"] == "monitor"
    assert partial_queries["safeguards"]["no_technical_uncertainties"] is False


def test_noindex_candidate_rejects_missing_final_days_and_positive_development():
    final_date = date(2026, 7, 17)
    short = daily_rows(final_date, impressions=0, clicks=0, days=27)
    short_result = seo_recommendation.deterministic_recommendation(
        recommendation_page(), short, complete_query_snapshot(),
        final_date=final_date, now=NOW,
    )

    growing = daily_rows(final_date, impressions=0, clicks=0)
    growing[0]["impressions"] = 2
    growing[0]["position"] = 30
    growing_result = seo_recommendation.deterministic_recommendation(
        recommendation_page(), growing, complete_query_snapshot(),
        final_date=final_date, now=NOW,
    )

    assert short_result["recommendation"] != "noindex_candidate"
    assert short_result["safeguards"]["has_28_finalized_days"] is False
    assert growing_result["recommendation"] != "noindex_candidate"
    assert growing_result["safeguards"]["no_positive_development"] is False


def test_deterministic_recommendation_maps_existing_status_classes():
    final_date = date(2026, 7, 17)
    cases = [
        (daily_rows(final_date, impressions=10, clicks=1, position=5), "protect_winner"),
        (daily_rows(final_date, impressions=10, clicks=0, position=12), "refresh_title_and_intro"),
    ]
    declining = daily_rows(final_date, impressions=10, clicks=2, position=8, days=14)
    for row in declining[:7]:
        row.update({"impressions": 2, "clicks": 0, "ctr": 0.0})
    cases.append((declining, "investigate_decline"))

    for metrics, expected in cases:
        result = seo_recommendation.deterministic_recommendation(
            recommendation_page(), metrics, complete_query_snapshot(),
            final_date=final_date, now=NOW,
        )
        assert result["recommendation"] == expected


def test_query_snapshot_marks_privacy_filtered_rows_as_partial():
    class QueryClient(FakeSearchConsoleClient):
        def query_page_queries(self, start_date, end_date, page_url, *, limit):
            return {
                "requests": 1,
                "truncated": False,
                "coverage": "top_queries_only",
                "rows": [
                    {"keys": ["consensus engine"], "clicks": 1, "impressions": 5, "ctr": .2, "position": 4},
                    {"keys": ["person@example.com"], "clicks": 0, "impressions": 1, "ctr": 0, "position": 8},
                ],
            }

    db = FakeFirestore()
    result = build_service(db, QueryClient()).collect()
    snapshot = db.documents[
        ("seo_pages", page_id_for_url(PAGE_URL), "query_snapshots", "2026-07-17")
    ]

    assert result["status"] == "partial"
    assert snapshot["partial"] is True
    assert snapshot["partial_reasons"] == ["privacy_filter"]
    assert snapshot["redacted_query_rows"] == 1
    assert [row["query"] for row in snapshot["top_queries"]] == ["consensus engine"]
    assert "person@example.com" not in str(db.documents)


def test_search_console_query_request_is_final_bounded_and_page_filtered():
    class Response:
        status_code = 200

        def json(self):
            return {"rows": [
                {"keys": ["one"]}, {"keys": ["two"]}, {"keys": ["sentinel"]}
            ]}

    class Session:
        def __init__(self):
            self.payload = None

        def post(self, url, json, timeout):
            self.payload = json
            return Response()

    session = Session()
    client = gsc.GoogleSearchConsoleClient(
        gsc.SearchConsoleConfig("sc-domain:consens.io", {"type": "service_account"}),
        session=session,
    )
    result = client.query_page_queries(
        date(2026, 6, 20), date(2026, 7, 17), PAGE_URL, limit=2
    )

    assert result["truncated"] is True
    assert len(result["rows"]) == 2
    assert session.payload["dimensions"] == ["query"]
    assert session.payload["dataState"] == "final"
    assert session.payload["rowLimit"] == 3
    assert session.payload["dimensionFilterGroups"][0]["filters"][0]["expression"] == PAGE_URL


def test_journal_generation_is_idempotent_and_append_only():
    db = FakeFirestore()
    page_id = page_id_for_url(PAGE_URL)
    db.documents[("seo_pages", page_id)] = recommendation_page()
    final_date = date(2026, 7, 17)
    for row in daily_rows(final_date, impressions=0, clicks=0):
        db.documents[("seo_pages", page_id, "daily_metrics", row["date"])] = row
    db.documents[("seo_pages", page_id, "query_snapshots", final_date.isoformat())] = complete_query_snapshot()
    service = seo_recommendation.SeoRecommendationService(
        repository=FirestoreSeoRepository(db), clock=lambda: NOW,
    )

    first = service.generate(page_id)
    second = service.generate(page_id)
    journal_paths = [path for path in db.documents if path[0] == "seo_judgements"]

    assert first["judgment_id"] == second["judgment_id"]
    assert first["created"] is True
    assert second["created"] is False
    assert len(journal_paths) == 1
    assert db.documents[journal_paths[0]]["user_feedback"] is None
    assert "content_representation" not in db.documents[journal_paths[0]]["dossier_summary"]


def test_weekly_review_can_generate_for_historically_captured_inactive_pages():
    db = FakeFirestore()
    page_id = page_id_for_url(PAGE_URL)
    page = recommendation_page()
    page["active"] = False
    db.documents[("seo_pages", page_id)] = page
    service = seo_recommendation.SeoRecommendationService(
        repository=FirestoreSeoRepository(db), clock=lambda: NOW,
    )

    try:
        service.generate(page_id)
        assert False, "normal per-page admin generation should remain active-only"
    except seo_recommendation.SeoRecommendationError as exc:
        assert exc.code == "not_found"

    result = service.generate(page_id, include_inactive=True)
    assert result["recommendation"] in seo_recommendation.RECOMMENDATIONS


def test_llm_json_validation_is_strict_and_cannot_bypass_noindex_safeguards():
    valid = {
        "recommendation": "refresh_content",
        "confidence": 0.7,
        "evidence": ["The introduction does not match the visible queries."],
        "proposed_changes": ["Rewrite the introduction."],
        "review_after_days": 21,
        "requires_human_approval": True,
    }
    parsed = seo_recommendation.SeoContentJudge.validate_response(
        valid, deterministic_recommendation_value="refresh_title_and_intro"
    )
    assert parsed["requires_human_approval"] is True

    invalid = dict(valid, unexpected="field")
    try:
        seo_recommendation.SeoContentJudge.validate_response(
            invalid, deterministic_recommendation_value="refresh_title_and_intro"
        )
        assert False, "expected invalid structured response"
    except seo_recommendation.SeoRecommendationError as exc:
        assert exc.code == "invalid_llm_response"

    unsafe = dict(valid, recommendation="noindex_candidate")
    try:
        seo_recommendation.SeoContentJudge.validate_response(
            unsafe, deterministic_recommendation_value="refresh_title_and_intro"
        )
        assert False, "expected noindex safeguard rejection"
    except seo_recommendation.SeoRecommendationError as exc:
        assert exc.code == "unsafe_llm_response"


def test_optional_content_judge_uses_bounded_prompt_and_structured_schema():
    captured = {}
    response = {
        "recommendation": "refresh_content",
        "confidence": 0.75,
        "evidence": ["The content does not cover the strongest query intent."],
        "proposed_changes": ["Add a concise section answering that intent."],
        "review_after_days": 21,
        "requires_human_approval": True,
    }

    def caller(prompt, schema):
        captured["prompt"] = prompt
        captured["schema"] = schema
        return response

    judge = seo_recommendation.SeoContentJudge(
        api_key="server-key", model="configured-model", caller=caller
    )
    context = {
        "dossier": {
            "content_representation": "x" * 20_000,
            "content_summary": "y" * 5_000,
        },
        "top_queries": [{"query": "q" * 160, "impressions": 1}] * 50,
    }
    result = judge.ask(
        context, deterministic_recommendation_value="refresh_title_and_intro"
    )

    assert result["recommendation"] == "refresh_content"
    assert len(captured["prompt"]) <= seo_recommendation.MAX_PROMPT_CHARS
    assert captured["schema"]["additionalProperties"] is False
    assert set(captured["schema"]["required"]) == {
        "recommendation", "confidence", "evidence", "proposed_changes",
        "review_after_days", "requires_human_approval",
    }


def test_content_judge_is_not_called_for_winner_status():
    db = FakeFirestore()
    page_id = page_id_for_url(PAGE_URL)
    db.documents[("seo_pages", page_id)] = recommendation_page()
    final_date = date(2026, 7, 17)
    for row in daily_rows(final_date, impressions=10, clicks=1, position=5):
        db.documents[("seo_pages", page_id, "daily_metrics", row["date"])] = row
    db.documents[("seo_pages", page_id, "query_snapshots", final_date.isoformat())] = complete_query_snapshot()
    calls = []
    judge = seo_recommendation.SeoContentJudge(
        api_key="server-key",
        model="configured-model",
        caller=lambda prompt, schema: calls.append(prompt),
    )
    service = seo_recommendation.SeoRecommendationService(
        repository=FirestoreSeoRepository(db), content_judge=judge, clock=lambda: NOW,
    )

    try:
        service.ask_content_judge(page_id)
        assert False, "expected applicability guard"
    except seo_recommendation.SeoRecommendationError as exc:
        assert exc.code == "content_judge_not_applicable"
    assert calls == []
