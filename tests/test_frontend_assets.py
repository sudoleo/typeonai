"""Asset URLs must be content-addressed, in both modes.

The 30 hand-maintained ``?v=`` marks these replace had one failure mode:
editing a file and forgetting the bump served a stale copy to returning users.
Nothing here may depend on a human remembering to change a string.
"""

from __future__ import annotations

import json
import re

import pytest

from app.core import assets


@pytest.fixture(autouse=True)
def clear_asset_cache():
    assets._cache.clear()
    yield
    assets._cache.clear()


@pytest.fixture
def source_mode(monkeypatch):
    monkeypatch.setenv("FRONTEND_DEV", "1")


def test_bundles_json_lists_only_files_that_exist():
    config = json.loads(assets.BUNDLES_FILE.read_text(encoding="utf-8"))

    for group in config["groups"]:
        if group.get("kind") == "module":
            assert (assets.ROOT / group["entry"]).is_file(), group["entry"]
            continue
        for entry in group["files"]:
            relative = entry if isinstance(entry, str) else entry["file"]
            assert (assets.ROOT / relative).is_file(), relative


def test_source_mode_serves_every_file_in_declared_order(source_mode):
    config = json.loads(assets.BUNDLES_FILE.read_text(encoding="utf-8"))
    expected = []
    for group in config["groups"]:
        if group.get("kind") == "module":
            expected.append("/" + group["entry"])
            continue
        for entry in group["files"]:
            expected.append("/" + (entry if isinstance(entry, str) else entry["file"]))

    served = [tag.src.split("?")[0] for tag in assets.frontend_assets().scripts]

    assert served == expected


def test_source_mode_marks_defer_and_module_per_group(source_mode):
    by_group: dict[str, list[assets.ScriptTag]] = {}
    for tag in assets.frontend_assets().scripts:
        by_group.setdefault(tag.group, []).append(tag)

    # The head group seeds state before the first paint and must stay blocking.
    assert all(not tag.defer and not tag.module for tag in by_group["head"])
    assert all(tag.defer and not tag.module for tag in by_group["app"])
    assert all(tag.module and not tag.defer for tag in by_group["firebase"])


def test_every_source_url_carries_a_content_hash(source_mode):
    for tag in assets.frontend_assets().scripts:
        assert re.search(r"\?v=[0-9a-f]{12}$", tag.src), tag.src


def test_editing_a_file_changes_its_url(source_mode, tmp_path, monkeypatch):
    target = assets.ROOT / "static" / "js" / "app-state.js"
    original = target.read_bytes()
    before = assets.asset_url("/static/js/app-state.js")
    try:
        target.write_bytes(original + b"\n// touched by the test\n")
        assets._cache.clear()
        after = assets.asset_url("/static/js/app-state.js")
    finally:
        target.write_bytes(original)

    assert before != after


def test_style_url_changes_when_an_imported_sheet_changes(source_mode):
    target = assets.ROOT / "static" / "css" / "shell.css"
    original = target.read_bytes()
    before = assets.frontend_assets().style
    try:
        # style.css itself is untouched -- only a file it @imports changes.
        target.write_bytes(original + b"\n/* touched by the test */\n")
        assets._cache.clear()
        after = assets.frontend_assets().style
    finally:
        target.write_bytes(original)

    assert before != after, "an @import edit must bust the aggregated sheet"


def test_built_mode_is_used_when_a_manifest_exists(monkeypatch):
    monkeypatch.delenv("FRONTEND_DEV", raising=False)
    if not assets.MANIFEST_FILE.exists():
        pytest.skip("no build output; run npm run build")

    built = assets.frontend_assets()

    assert built.built is True
    assert len(built.scripts) < 10, "the whole point is fewer requests"
    assert all(tag.src.startswith("/static/dist/") for tag in built.scripts)
    assert built.style.startswith("/static/dist/")


def test_built_mode_keeps_the_head_group_first_and_blocking(monkeypatch):
    monkeypatch.delenv("FRONTEND_DEV", raising=False)
    if not assets.MANIFEST_FILE.exists():
        pytest.skip("no build output; run npm run build")

    scripts = assets.frontend_assets().scripts

    assert scripts[0].group == "head"
    assert not scripts[0].defer
    assert [tag.group for tag in scripts] == [
        "head",
        "auth",
        "firebase",
        "demo",
        "app",
    ]


def test_dev_flag_overrides_an_existing_build(monkeypatch):
    monkeypatch.setenv("FRONTEND_DEV", "1")

    assert assets.frontend_assets().built is False


def test_asset_url_leaves_unknown_and_external_paths_alone():
    assert assets.asset_url("https://cdn.example/x.js") == "https://cdn.example/x.js"
    # A missing file must not take the page render down.
    assert assets.asset_url("/static/does-not-exist.png") == "/static/does-not-exist.png"


def test_no_manual_version_marks_are_left_in_the_app_template():
    html = (assets.ROOT / "templates" / "index.html").read_text(encoding="utf-8")

    hardcoded = re.findall(r'(?:src|href)="(/static[^"]*\?v=[^"]*)"', html)

    assert hardcoded == [], "index.html must not carry hand-maintained cache marks"


def test_no_manual_version_marks_hide_inside_app_javascript_imports():
    """A nested ESM import used to retain an old hand-written ``?v=`` even
    after index.html moved to content hashes. Such an import stays stale in
    source mode because changing the dependency does not change its URL."""

    config = json.loads(assets.BUNDLES_FILE.read_text(encoding="utf-8"))
    sources = []
    for group in config["groups"]:
        if group.get("kind") == "module":
            sources.append(group["entry"])
        else:
            sources.extend(
                entry if isinstance(entry, str) else entry["file"]
                for entry in group["files"]
            )

    offenders = []
    pattern = re.compile(r'''(?:from\s+|import\s*\()?["']/static/[^"']+\?v=''' )
    for relative in sources:
        if pattern.search((assets.ROOT / relative).read_text(encoding="utf-8")):
            offenders.append(relative)

    assert offenders == []
