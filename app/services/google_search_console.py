"""Small Search Console Search Analytics client.

The integration deliberately uses ``google-auth`` plus HTTP instead of the
large discovery client.  Credentials are read only from the environment and
all public errors are stable, sanitized messages that never contain credential
fields or Google response bodies.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.parse import urlsplit

from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account


READONLY_SCOPE = "https://www.googleapis.com/auth/webmasters.readonly"
SEARCH_ANALYTICS_URL = (
    "https://searchconsole.googleapis.com/webmasters/v3/sites/"
    "{site_url}/searchAnalytics/query"
)
SITE_DETAILS_URL = "https://searchconsole.googleapis.com/webmasters/v3/sites/{site_url}"
DEFAULT_ROW_LIMIT = 25_000
DEFAULT_MAX_PAGES = 10
MAX_TOP_QUERY_ROWS = 100
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class SearchConsoleError(Exception):
    """A safe operational error suitable for an admin response."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.safe_message = message


@dataclass(frozen=True)
class SearchConsoleConfig:
    site_url: str
    service_account_info: dict[str, Any]

    @classmethod
    def from_env(cls) -> "SearchConsoleConfig":
        site_url = str(os.environ.get("GSC_SITE_URL") or "").strip()
        credentials_path_value = str(
            os.environ.get("GSC_SERVICE_ACCOUNT_JSON") or ""
        ).strip()
        if not site_url or not credentials_path_value:
            raise SearchConsoleError(
                "not_configured",
                "GSC_SITE_URL and GSC_SERVICE_ACCOUNT_JSON are not configured.",
            )
        domain_property = site_url.removeprefix("sc-domain:").strip()
        parsed_site = urlsplit(site_url)
        valid_property = (
            (site_url.startswith("sc-domain:") and bool(domain_property))
            or (parsed_site.scheme in {"http", "https"} and bool(parsed_site.netloc))
        )
        if not valid_property:
            raise SearchConsoleError(
                "invalid_configuration", "GSC_SITE_URL is not a valid Search Console property."
            )

        try:
            # This variable is intentionally a file path in every environment.
            # Relative paths resolve against the repository root; Render uses
            # an absolute /etc/secrets/... path. The resolved path is never
            # retained in public state, logged, or returned by an endpoint.
            credentials_path = Path(credentials_path_value)
            if not credentials_path.is_absolute():
                credentials_path = REPOSITORY_ROOT / credentials_path
            info = json.loads(credentials_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise SearchConsoleError(
                "invalid_configuration", "GSC_SERVICE_ACCOUNT_JSON is invalid."
            ) from None
        if not isinstance(info, dict) or info.get("type") != "service_account":
            raise SearchConsoleError(
                "invalid_configuration", "GSC_SERVICE_ACCOUNT_JSON is invalid."
            )
        return cls(site_url=site_url, service_account_info=info)


class GoogleSearchConsoleClient:
    def __init__(
        self,
        config: SearchConsoleConfig,
        *,
        session=None,
        row_limit: int = DEFAULT_ROW_LIMIT,
        max_pages: int = DEFAULT_MAX_PAGES,
    ):
        self.config = config
        self.row_limit = max(1, min(int(row_limit), DEFAULT_ROW_LIMIT))
        self.max_pages = max(1, int(max_pages))
        if session is None:
            try:
                credentials = service_account.Credentials.from_service_account_info(
                    config.service_account_info,
                    scopes=[READONLY_SCOPE],
                )
                session = AuthorizedSession(credentials)
            except Exception:
                raise SearchConsoleError(
                    "invalid_configuration", "Search Console credentials are invalid."
                ) from None
        self._session = session

    @classmethod
    def from_env(cls) -> "GoogleSearchConsoleClient":
        return cls(SearchConsoleConfig.from_env())

    def query(self, start_date: date, end_date: date) -> dict[str, Any]:
        """Return page/date rows, paginating with a hard request cap."""
        if start_date > end_date:
            return {"rows": [], "requests": 0, "truncated": False}

        endpoint = SEARCH_ANALYTICS_URL.format(
            site_url=quote(self.config.site_url, safe="")
        )
        rows: list[dict[str, Any]] = []
        requests_made = 0
        truncated = False
        for page_number in range(self.max_pages):
            start_row = page_number * self.row_limit
            payload = {
                "startDate": start_date.isoformat(),
                "endDate": end_date.isoformat(),
                "dimensions": ["page", "date"],
                "type": "web",
                "dataState": "final",
                "aggregationType": "byPage",
                "rowLimit": self.row_limit,
                "startRow": start_row,
            }
            try:
                response = self._session.post(endpoint, json=payload, timeout=30)
            except Exception:
                raise SearchConsoleError(
                    "request_failed", "The Search Console request failed."
                ) from None
            requests_made += 1
            if int(getattr(response, "status_code", 0)) != 200:
                status = int(getattr(response, "status_code", 0))
                raise SearchConsoleError(
                    "request_failed",
                    f"Search Console returned HTTP {status or 'error'}.",
                )
            try:
                body = response.json()
            except Exception:
                raise SearchConsoleError(
                    "invalid_response", "Search Console returned an invalid response."
                ) from None
            page_rows = body.get("rows") if isinstance(body, dict) else None
            page_rows = page_rows if isinstance(page_rows, list) else []
            rows.extend(row for row in page_rows if isinstance(row, dict))
            if len(page_rows) < self.row_limit:
                break
        else:
            truncated = True

        return {"rows": rows, "requests": requests_made, "truncated": truncated}

    def query_page_queries(
        self, start_date: date, end_date: date, page_url: str, *, limit: int = 20
    ) -> dict[str, Any]:
        """Return a bounded top-query sample for one exact page.

        Search Console can suppress anonymized queries. ``coverage`` therefore
        remains explicit even when the row cap was not reached.
        """
        if start_date > end_date:
            return {
                "rows": [], "requests": 0, "truncated": False,
                "coverage": "top_queries_only",
            }
        limit = max(1, min(int(limit), MAX_TOP_QUERY_ROWS))
        endpoint = SEARCH_ANALYTICS_URL.format(
            site_url=quote(self.config.site_url, safe="")
        )
        payload = {
            "startDate": start_date.isoformat(),
            "endDate": end_date.isoformat(),
            "dimensions": ["query"],
            "dimensionFilterGroups": [{
                "groupType": "and",
                "filters": [{
                    "dimension": "page",
                    "operator": "equals",
                    "expression": str(page_url),
                }],
            }],
            "type": "web",
            "dataState": "final",
            # Ask for one sentinel row so the stored snapshot can distinguish
            # a complete short list from a response cut at our own limit.
            "rowLimit": min(limit + 1, MAX_TOP_QUERY_ROWS + 1),
            "startRow": 0,
        }
        try:
            response = self._session.post(endpoint, json=payload, timeout=30)
        except Exception:
            raise SearchConsoleError(
                "request_failed", "The Search Console query request failed."
            ) from None
        status = int(getattr(response, "status_code", 0))
        if status != 200:
            raise SearchConsoleError(
                "request_failed",
                f"Search Console returned HTTP {status or 'error'} for query data.",
            )
        try:
            body = response.json()
        except Exception:
            raise SearchConsoleError(
                "invalid_response", "Search Console returned an invalid query response."
            ) from None
        rows = body.get("rows") if isinstance(body, dict) else None
        rows = [row for row in (rows or []) if isinstance(row, dict)]
        return {
            "rows": rows[:limit],
            "requests": 1,
            "truncated": len(rows) > limit,
            "coverage": "top_queries_only",
        }

    def check_connection(self) -> dict[str, Any]:
        """Verify credentials and property access without returning site data."""
        endpoint = SITE_DETAILS_URL.format(
            site_url=quote(self.config.site_url, safe="")
        )
        try:
            response = self._session.get(endpoint, timeout=20)
        except Exception:
            raise SearchConsoleError(
                "connection_failed", "The Search Console connection check failed."
            ) from None

        status = int(getattr(response, "status_code", 0))
        if status in {401, 403, 404}:
            raise SearchConsoleError(
                "permission_denied",
                "The credentials cannot access the configured Search Console property.",
            )
        if status != 200:
            raise SearchConsoleError(
                "connection_failed",
                f"Search Console connection check returned HTTP {status or 'error'}.",
            )
        try:
            body = response.json()
        except Exception:
            raise SearchConsoleError(
                "invalid_response", "Search Console returned an invalid response."
            ) from None
        permission = str(body.get("permissionLevel") or "") if isinstance(body, dict) else ""
        if not permission or permission == "siteUnverifiedUser":
            raise SearchConsoleError(
                "permission_denied",
                "The credentials cannot access the configured Search Console property.",
            )
        return {
            "configured": True,
            "connected": True,
            "status": "connected",
            "message": "Search Console connection successful.",
        }


def configuration_status() -> dict[str, Any]:
    """Validate local configuration without making a Google request."""
    try:
        config = SearchConsoleConfig.from_env()
        # Also validate the credential structure/private key through google-auth.
        service_account.Credentials.from_service_account_info(
            config.service_account_info, scopes=[READONLY_SCOPE]
        )
    except SearchConsoleError as exc:
        return {"configured": False, "status": exc.code, "message": exc.safe_message}
    except Exception:
        return {
            "configured": False,
            "status": "invalid_configuration",
            "message": "Search Console credentials are invalid.",
        }
    return {"configured": True, "status": "configured", "message": "Configured."}


def connection_status() -> dict[str, Any]:
    """Run a sanitized live property-access check for the admin dashboard."""
    try:
        return GoogleSearchConsoleClient.from_env().check_connection()
    except SearchConsoleError as exc:
        return {
            "configured": exc.code not in {"not_configured", "invalid_configuration"},
            "connected": False,
            "status": exc.code,
            "message": exc.safe_message,
        }
    except Exception:
        return {
            "configured": False,
            "connected": False,
            "status": "connection_failed",
            "message": "The Search Console connection check failed safely.",
        }
