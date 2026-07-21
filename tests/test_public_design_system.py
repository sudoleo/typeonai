from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_TEMPLATES = (
    "landing.html",
    "about.html",
    "ai-model-comparison.html",
    "consensus-engine.html",
    "benchmark.html",
    "privacy.html",
    "terms.html",
    "imprint.html",
    "share.html",
    "share_unavailable.html",
)


def read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_public_pages_share_navigation_and_footer_partials():
    for template_name in PUBLIC_TEMPLATES:
        template = read(f"templates/{template_name}")
        assert '{% include "partials/public_nav.html" %}' in template
        assert '{% include "partials/public_footer.html" %}' in template


def test_public_styles_share_the_app_aligned_token_layer():
    token_import = "@import url('./public-tokens.css?v=20260716-public-system1');"
    assert token_import in read("static/css/landing.css")
    assert token_import in read("static/css/public-pages.css")

    app_tokens = read("static/css/variables.css")
    public_tokens = read("static/css/public-tokens.css")
    for contract in (
        "--radius-lg: 16px;",
        "--bg-color: #f5f5f4;",
        "--accent-secondary: #4fc2a3;",
        "--glass-blur: 18px;",
    ):
        assert contract in app_tokens

    for contract in (
        "--radius-lg: 16px;",
        "--page-bg: var(--bg-grey);",
        "--accent-secondary: #4fc2a3;",
        "--glass-blur: 18px;",
    ):
        assert contract in public_tokens


def test_product_result_mockup_is_reused_and_public_copy_has_no_em_dash():
    # The landing hero is demo-first (input field CTA) since 2026-07-17 and no
    # longer embeds the result mockup; the consensus-engine page still does.
    include = '{% include "partials/product_result_mockup.html" %}'
    assert include in read("templates/consensus-engine.html")

    for template_name in PUBLIC_TEMPLATES:
        assert "—" not in read(f"templates/{template_name}")
    assert "—" not in read("templates/partials/product_result_mockup.html")


def test_share_page_loads_the_common_math_renderer():
    template = read("templates/share.html")
    assert "katex@0.17.0/dist/katex.min.js" in template
    assert "/static/js/math-render.js?v=20260720-math1" in template
    assert '<main class="page-shell" data-math-render>' in template


def test_landing_explains_consensus_watch_as_fourth_product_step():
    landing = read("templates/landing.html")
    navigation = read("templates/partials/public_nav.html")

    assert 'id="watch"' in landing
    assert "04 · Monitor" in landing
    assert "Know when the answer changes." in landing
    assert 'href="/app/watches"' in landing
    assert 'href="/#watch">Watches</a>' in navigation
