"""Regression checks for browser failures that must degrade gracefully."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
    assert "async function prepareWithUsageRetry" in query
    assert "const maxAttempts = 3;" in query
    assert "window.App.usageLimit?.showTemporaryStorageBusy?.();" in query
    assert "function showTemporaryStorageBusy()" in limits
    assert "No model was asked; please try this question again in a moment." in limits

    busy_branch = query.index("if (isUsageStorageBusyError(prepareData))")
    fanout = query.index("const deepSearchActive", busy_branch)
    assert busy_branch < fanout


def test_changed_frontend_scripts_are_cache_busted():
    template = read("templates/index.html")

    assert "markdown-stream.js?v=20260802-markdownfallback1" in template
    assert "firebase.js?v=20260804-bookmarkturns1" in template
    assert "demo.js?v=20260803-demorunstages1" in template
    assert "usage-limit.js?v=20260802-storagebusy1" in template
    assert "query-send.js?v=20260804-bookmarkturns1" in template
