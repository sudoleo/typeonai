"""The committed build output must match the sources it was built from.

Without this, the ``?v=`` trap comes back one level up: edit a JS or CSS file,
forget ``npm run build``, and production keeps serving the previous bundle.
The build records a fingerprint of its inputs; this recomputes it in Python so
the guard runs in the normal suite, with no node installed.
"""

from __future__ import annotations

import json

import pytest

from app.core import assets


@pytest.fixture(autouse=True)
def clear_asset_cache():
    assets._cache.clear()
    yield
    assets._cache.clear()


requires_build = pytest.mark.skipif(
    not assets.MANIFEST_FILE.exists(),
    reason="no build output; run npm run build",
)


@requires_build
def test_dist_is_not_stale():
    manifest = json.loads(assets.MANIFEST_FILE.read_text(encoding="utf-8"))

    assert manifest["sources"] == assets.source_fingerprint(), (
        "static/dist was built from different sources. Run: npm run build"
    )
    assert assets.build_is_stale() is False


@requires_build
def test_manifest_records_every_input_needed_for_python_only_staleness_checks():
    manifest = json.loads(assets.MANIFEST_FILE.read_text(encoding="utf-8"))
    inputs = set(manifest["inputs"])
    config = json.loads(assets.BUNDLES_FILE.read_text(encoding="utf-8"))

    declared = {"static/js/bundles.json", "scripts/build_frontend.mjs", "package-lock.json"}
    for group in config["groups"]:
        if group.get("kind") == "module":
            declared.add(group["entry"])
        else:
            declared.update(
                entry if isinstance(entry, str) else entry["file"]
                for entry in group["files"]
            )
    for style in config["styles"]:
        declared.add(style["entry"])

    assert declared <= inputs
    assert all((assets.ROOT / relative).is_file() for relative in inputs)


@requires_build
def test_every_file_the_manifest_points_at_exists():
    manifest = json.loads(assets.MANIFEST_FILE.read_text(encoding="utf-8"))
    urls = [item["src"] for item in manifest["scripts"]] + list(manifest["styles"].values())

    for url in urls:
        assert (assets.ROOT / url.lstrip("/")).is_file(), url


@requires_build
def test_bundling_actually_reduced_the_request_count_and_bytes():
    built = assets.frontend_assets()
    source_bytes = sum(
        (assets.ROOT / tag.src.split("?")[0].lstrip("/")).stat().st_size
        for tag in assets._source_assets().scripts
    )
    built_bytes = sum(
        (assets.ROOT / tag.src.lstrip("/")).stat().st_size for tag in built.scripts
    )

    assert len(built.scripts) <= 6 < len(assets._source_assets().scripts)
    assert built_bytes < source_bytes / 2, (
        f"minified {built_bytes} vs source {source_bytes}"
    )


@requires_build
def test_the_bundle_keeps_the_window_contracts_addressable():
    """Minification must not rename the top-level names the modules call each
    other by. esbuild only leaves them alone while the input is treated as a
    classic script -- a bundler flag that wrapped it in a module or IIFE would
    silently break every window.* handoff."""

    manifest = json.loads(assets.MANIFEST_FILE.read_text(encoding="utf-8"))
    app_bundle = next(item for item in manifest["scripts"] if item["name"] == "app")
    code = (assets.ROOT / app_bundle["src"].lstrip("/")).read_text(encoding="utf-8")

    for global_name in (
        "sendQuestion",
        "getConsensus",
        "escapeHtml",
        "injectMarkdown",
        "renderMarkdownHtml",
    ):
        assert global_name in code, global_name

    # A module/IIFE wrapper is exactly the failure this guards against.
    assert not code.startswith("(()=>{")
    assert not code.startswith("(function(){")


@requires_build
def test_css_bundle_keeps_relative_asset_paths_resolvable():
    """static/dist is a sibling of static/css, so ../fonts and ../icons still
    point at the same files. If the output ever moves, the fonts vanish."""

    manifest = json.loads(assets.MANIFEST_FILE.read_text(encoding="utf-8"))
    sheet = assets.ROOT / manifest["styles"]["app"].lstrip("/")
    css = sheet.read_text(encoding="utf-8")

    assert "@import" not in css, "imports must be inlined, not re-fetched"
    assert "../fonts/inter/InterVariable.woff2" in css
    for relative in ("../fonts/inter/InterVariable.woff2", "../icons/consensus.png"):
        assert (sheet.parent / relative).resolve().is_file(), relative
