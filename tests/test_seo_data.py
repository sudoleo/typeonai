from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routers import admin as admin_router
from app.services import google_search_console as gsc
from app.services import seo_data
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

    def collection(self, name):
        return FakeCollection(self, (name,))


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
    second = service.collect()

    assert first["metrics_written"] == 90
    assert second["metrics_written"] == 0
    assert second["days_collected"] == 0
    assert client.calls == calls_after_first
    metric_paths = [path for path in db.documents if "daily_metrics" in path]
    assert len(metric_paths) == 90


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

    monkeypatch.setattr(admin_router, "seo_data_service", StubSeoService())
    monkeypatch.setattr(admin_router, "verify_user_token", lambda token: "uid-1")
    monkeypatch.setattr(admin_router, "is_user_admin", lambda uid: False)
    app = FastAPI()
    app.include_router(admin_router.router)
    client = TestClient(app)

    assert client.get("/api/admin/seo").status_code == 401
    assert client.get("/api/admin/seo", headers={"Authorization": "Bearer token"}).status_code == 403
    assert client.post("/api/admin/seo/check", headers={"Authorization": "Bearer token"}).status_code == 403
    assert client.post("/api/admin/seo/collect", headers={"Authorization": "Bearer token"}).status_code == 403

    monkeypatch.setattr(admin_router, "is_user_admin", lambda uid: True)
    assert client.get("/api/admin/seo", headers={"Authorization": "Bearer token"}).status_code == 200
    assert client.post("/api/admin/seo/check", headers={"Authorization": "Bearer token"}).status_code == 200
    assert client.post("/api/admin/seo/collect", headers={"Authorization": "Bearer token"}).status_code == 200


def test_admin_seo_collect_action_is_hidden_until_admin_request_succeeds():
    root = Path(__file__).resolve().parents[1]
    template = (root / "templates" / "admin.html").read_text(encoding="utf-8")
    main_source = (root / "main.py").read_text(encoding="utf-8")

    assert 'data-tab="seo"' in template
    assert 'id="collectSeoBtn" type="button" class="admin-btn" hidden' in template
    assert 'id="checkSeoConnectionBtn" type="button" class="admin-btn secondary" hidden' in template
    assert "shareAdminRequest('GET', '/api/admin/seo')" in template
    assert "shareAdminRequest('POST', '/api/admin/seo/check', {})" in template
    assert 'os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", "gen-lang-client-0234219247-53b2b1c0e355.json")' in main_source
    assert "GSC_SERVICE_ACCOUNT_JSON" not in main_source
