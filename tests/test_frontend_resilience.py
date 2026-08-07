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

    assert "markdown-stream.js?v=20260805-chatmulti1" in template
    assert "firebase.js?v=20260806-bookmarkcontinue1" in template
    assert "demo.js?v=20260803-demorunstages1" in template
    assert "usage-limit.js?v=20260802-storagebusy1" in template
    assert "consensus-insights.js?v=20260806-chatbookmark1" in template
    assert "chat-session.js?v=20260806-chatlimits1" in template
    assert "app-init.js?v=20260806-followupdefault1" in template
    for changed in (
        "app-core.js",
        "attachments.js",
        "consensus-progress.js",
        "consensus-run.js",
        "query-send.js",
    ):
        assert f"{changed}?v=20260807-threadmessages1" in template
    # Nachtrag: eingeklappter Composer + Cursor-Fix.
    for changed in ("style.css", "composer-collapse.js"):
        assert f"{changed}?v=20260807-threadmessages2" in template


def test_mobile_enter_keeps_the_textarea_newline_behavior():
    app_init = read("static/js/app-init.js")
    keydown = app_init.split(
        'document.getElementById("questionInput").addEventListener("keydown"', 1
    )[1].split("// Es gibt genau EINEN sichtbaren Sidebar-Toggle", 1)[0]

    assert 'window.matchMedia("(max-width: 768px)").matches' in keydown
    assert "event.isComposing" in keydown
    assert keydown.index("matchMedia") < keydown.index("event.preventDefault()")
