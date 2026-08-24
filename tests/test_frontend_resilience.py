"""Regression checks for browser failures that must degrade gracefully."""

from datetime import datetime
from functools import lru_cache
from pathlib import Path
import re
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.source_contract


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_missing_markdown_dependencies_render_plaintext_instead_of_throwing():
    markdown = read("static/js/markdown-stream.js")

    assert 'typeof window.marked?.parse !== "function"' in markdown
    assert 'typeof window.DOMPurify?.sanitize !== "function"' in markdown
    assert 'message: "Markdown renderer unavailable; displaying unformatted text."' in markdown
    assert markdown.count("return escapeHtml(prepared);") >= 2


def test_usage_storage_busy_is_retried_and_stops_before_model_fanout():
    query = read("static/js/query-send.js")
    limits = read("static/js/usage-limit.js")

    assert '=== "usage_storage_busy"' in query
    assert "async function prepareWithRetry(payload, signal)" in query
    assert "for (let attempt = 1; attempt <= 3; attempt += 1)" in query
    assert "window.App.usageLimit?.showTemporaryStorageBusy?.();" in query
    assert "function showTemporaryStorageBusy()" in limits
    assert "No model was asked; please try this question again in a moment." in limits

    busy_branch = query.index("if (!prepared.response.ok && isUsageStorageBusy(prepared.data))")
    fanout = query.index("context.config.providers.map(provider => runProvider", busy_branch)
    assert busy_branch < fanout


LOCAL_ASSET_PATTERN = re.compile(
    r'''(?:src|href)=["'](?P<html>/static/[^"']+\.(?:js|css))'''
    r'''(?:\?v=(?P<html_version>[^"']+))?["']'''
    r'''|@import\s+(?:url\()?['"]?(?P<css>[^'"\)\s]+\.css)'''
    r'''(?:\?v=(?P<css_version>[^'"\)\s]+))?'''
)
VERSION_PATTERN = re.compile(r"^(?P<date>\d{8})-[a-z0-9][a-z0-9.-]*$")


def _active_local_asset_references():
    sources = sorted((ROOT / "templates").rglob("*.html"))
    sources += sorted((ROOT / "static").rglob("*.css"))
    for source in sources:
        text = source.read_text(encoding="utf-8")
        for match in LOCAL_ASSET_PATTERN.finditer(text):
            if match.group("html"):
                target = ROOT / match.group("html").lstrip("/")
                version = match.group("html_version")
            else:
                target = source.parent / match.group("css")
                version = match.group("css_version")
            yield source, target.resolve(), version


@lru_cache(maxsize=1)
def _asset_git_state():
    dirty_output = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", "static"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    dirty = {
        Path(line[3:].split(" -> ")[-1].strip().strip('"'))
        for line in dirty_output.splitlines()
        if line.strip()
    }
    history_output = subprocess.run(
        ["git", "log", "--format=@@%cs", "--name-only", "--", "static"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    latest_dates = {}
    commit_date = ""
    for line in history_output.splitlines():
        if line.startswith("@@"):
            commit_date = line[2:].replace("-", "")
        elif line.strip() and commit_date:
            latest_dates.setdefault(Path(line.strip()), commit_date)
    return dirty, latest_dates


def _required_asset_date(target: Path) -> str:
    relative = target.relative_to(ROOT)
    dirty, latest_dates = _asset_git_state()
    if relative in dirty:
        return datetime.fromtimestamp(target.stat().st_mtime).strftime("%Y%m%d")
    committed = latest_dates.get(relative)
    assert committed, f"tracked asset has no Git history: {relative}"
    return committed


def test_all_active_local_assets_have_current_consistent_cache_keys():
    """Validate the real asset graph instead of blessing remembered versions."""

    versions_by_target = {}
    references = list(_active_local_asset_references())
    assert references, "no local JS/CSS asset references were discovered"
    for source, target, version in references:
        relative_source = source.relative_to(ROOT)
        relative_target = target.relative_to(ROOT)
        assert target.is_file(), f"{relative_source} references missing {relative_target}"
        assert version, f"{relative_source} references {relative_target} without ?v="
        parsed = VERSION_PATTERN.fullmatch(version)
        assert parsed, f"invalid cache key for {relative_target}: {version}"
        assert parsed.group("date") >= _required_asset_date(target), (
            f"stale cache key for {relative_target}: {version} predates its latest change"
        )
        versions_by_target.setdefault(relative_target, set()).add(version)

    inconsistent = {
        str(target): sorted(versions)
        for target, versions in versions_by_target.items()
        if len(versions) != 1
    }
    assert not inconsistent, f"inconsistent active cache keys: {inconsistent}"


def test_mobile_enter_keeps_the_textarea_newline_behavior():
    app_init = read("static/js/app-init.js")
    keydown = app_init.split(
        'document.getElementById("questionInput").addEventListener("keydown"', 1
    )[1].split("// Es gibt genau EINEN sichtbaren Sidebar-Toggle", 1)[0]

    assert 'window.matchMedia("(max-width: 768px)").matches' in keydown
    assert "event.isComposing" in keydown
    assert keydown.index("matchMedia") < keydown.index("event.preventDefault()")
