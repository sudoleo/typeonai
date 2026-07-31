from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "templates"
PARTIAL = "partials/analytics.html"

# Seiten mit Tracking. Alles andere unter templates/ darf das Script nicht
# selbst einbinden — sonst wandert die Website-ID wieder in die Kopien zurueck.
TRACKED = (
    "index.html",
    "landing.html",
    "about.html",
    "ai-model-comparison.html",
    "consensus-engine.html",
    "benchmark.html",
    "model-pulse.html",
    "privacy.html",
    "terms.html",
    "imprint.html",
    "questions.html",
    "share.html",
    "topics.html",
    "topic.html",
)


def test_tracked_pages_include_the_shared_partial():
    for name in TRACKED:
        template = (TEMPLATES / name).read_text(encoding="utf-8")
        assert '{%% include "%s" %%}' % PARTIAL in template, name


def test_website_id_lives_only_in_the_partial():
    for path in TEMPLATES.rglob("*.html"):
        if path == TEMPLATES / PARTIAL:
            continue
        assert "cloud.umami.is" not in path.read_text(encoding="utf-8"), path.name


def test_admin_pages_are_not_tracked():
    # Admin-Traffic ist zu 100 % Eigen-Traffic und verzerrt die Zahlen.
    for name in ("admin.html", "admin_benchmark.html"):
        template = (TEMPLATES / name).read_text(encoding="utf-8")
        assert PARTIAL not in template, name


def test_partial_ships_the_self_exclusion_switch():
    partial = (TEMPLATES / PARTIAL).read_text(encoding="utf-8")

    # Umami prueft localStorage['umami.disabled'] vor jedem Send; ?notrack=1
    # setzt den Schluessel auch auf Geraeten ohne Konsole, ?notrack=0 loescht ihn.
    assert 'localStorage.setItem("umami.disabled", "1")' in partial
    assert 'localStorage.removeItem("umami.disabled")' in partial
    assert partial.index("notrack") < partial.index("cloud.umami.is"), (
        "Das Opt-out muss vor dem Tracker laufen, sonst geht der erste Pageview raus."
    )
