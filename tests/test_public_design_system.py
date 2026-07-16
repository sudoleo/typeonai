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
    include = '{% include "partials/product_result_mockup.html" %}'
    assert include in read("templates/landing.html")
    assert include in read("templates/consensus-engine.html")

    for template_name in PUBLIC_TEMPLATES:
        assert "—" not in read(f"templates/{template_name}")
    assert "—" not in read("templates/partials/product_result_mockup.html")
