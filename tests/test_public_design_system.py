import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_TEMPLATES = (
    "landing.html",
    "about.html",
    "ai-model-comparison.html",
    "consensus-engine.html",
    "benchmark.html",
    "model-pulse.html",
    "privacy.html",
    "terms.html",
    "imprint.html",
    "share.html",
    "share_unavailable.html",
    "topics.html",
    "topic.html",
)


def read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_public_pages_share_navigation_and_footer_partials():
    for template_name in PUBLIC_TEMPLATES:
        template = read(f"templates/{template_name}")
        assert '{% include "partials/public_nav.html" %}' in template
        assert '{% include "partials/public_footer.html" %}' in template


def test_public_styles_share_the_app_aligned_token_layer():
    # Version-agnostic: the cache buster is bumped on every CSS change, and
    # pinning it here only ever fails for the bump, never for a real drift.
    token_import = re.compile(r"@import url\('\./public-tokens\.css\?v=[\w.-]+'\);")
    assert token_import.search(read("static/css/landing.css"))
    assert token_import.search(read("static/css/public-pages.css"))

    app_tokens = read("static/css/variables.css")
    public_tokens = read("static/css/public-tokens.css")

    # The surface scale is the contract between /app and the public pages:
    # the marketing mockups are supposed to be made of the same material as
    # the product, so these five values have to be identical in both files —
    # in light AND dark. Everything else is derived from them.
    for contract in (
        "--radius-lg: 16px;",
        "--ground: #f2f1ef;",
        "--raise: #eae8e5;",
        "--well: #e6e4e0;",
        "--ink: #222428;",
        "--ground: #191a1c;",
        "--raise: #212325;",
        "--accent-secondary: var(--agree);",
        "--glass-blur: 0px;",
    ):
        assert contract in app_tokens, f"missing in variables.css: {contract}"
        assert contract in public_tokens, f"missing in public-tokens.css: {contract}"

    for contract in ("--page-bg: var(--ground);",):
        assert contract in public_tokens


def test_typography_is_self_hosted_and_shared_by_every_surface():
    typography = read("static/css/typography.css")
    app_tokens = read("static/css/variables.css")
    public_tokens = read("static/css/public-tokens.css")

    assert typography.count('@font-face {') == 2
    assert "../fonts/inter/InterVariable.woff2" in typography
    assert "../fonts/inter/InterVariable-Italic.woff2" in typography
    for contract in (
        '--font-sans: "Inter Variable"',
        "--font-weight-regular: 400;",
        "--font-weight-medium: 500;",
        "--font-weight-semibold: 600;",
        "--font-weight-bold: 700;",
        "--font-size-display-lg:",
    ):
        assert contract in typography

    typography_import = re.compile(r"@import url\('\./typography\.css\?v=[\w.-]+'\);")
    assert typography_import.search(app_tokens)
    assert typography_import.search(public_tokens)

    font_dir = ROOT / "static" / "fonts" / "inter"
    assert (font_dir / "InterVariable.woff2").stat().st_size > 300_000
    assert (font_dir / "InterVariable-Italic.woff2").stat().st_size > 300_000
    assert "SIL OPEN FONT LICENSE" in (font_dir / "OFL.txt").read_text(encoding="utf-8")

    for template_path in (ROOT / "templates").rglob("*.html"):
        assert "fonts.googleapis.com" not in template_path.read_text(encoding="utf-8")

    security = read("app/core/security.py")
    assert "fonts.googleapis.com" not in security
    assert "fonts.gstatic.com" not in security


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


def test_watch_header_keeps_intro_left_aligned_and_dates_visible():
    template = read("templates/share.html")
    public_css = read("static/css/public-pages.css")
    intro_rule = public_css.split(".watch-plain-intro {", 1)[1].split("}", 1)[0]

    assert "margin: 16px 0 0" in intro_rule
    assert "Schedule and check dates" not in template
    assert 'class="watch-meta-compact is-{{ watch_page.status }}"' in template


def test_landing_explains_consensus_watch_as_fourth_product_step():
    landing = read("templates/landing.html")
    navigation = read("templates/partials/public_nav.html")

    assert 'id="watch"' in landing
    assert "04 · Monitor" in landing
    assert "Know when the answer changes." in landing
    assert 'href="/app/watches"' in landing
    assert 'href="/#watch">Watches</a>' in navigation
