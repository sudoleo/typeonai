"""Ein aufgebrauchtes Kontingent muss SICHTBAR sein.

Der Fehler, den diese Datei festhaelt, war kein fehlender Text, sondern ein
fehlender Ort: der gefuehrte Lauf startet schon bei /prepare, und der
Limit-Pfad meldete danach setAgentModeStatus("error") — was den Block per
dismiss() wieder wegnahm. Die eigentliche Meldung landete in den
Antwortboxen, die im Agent Mode (Default seit 2026-07-27) hinter "Compare
answers" liegen. Ergebnis: eine Sekunde Fortschritt, dann eine leere Seite.
"""

from pathlib import Path
import re

from app.core import assets as assets_module
from app.core.assets import frontend_assets
from tests.frontend_order import loads_before


ROOT = Path(__file__).resolve().parents[1]


def read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_blocked_card_exists_and_is_loaded_before_its_callers():
    template = read("templates/index.html")

    assert 'id="runBlocked"' in template
    for part in (
        'id="runBlockedTitle"',
        'id="runBlockedBody"',
        'id="runBlockedMeta"',
        'id="runBlockedActions"',
        'id="runBlockedClose"',
    ):
        assert template.count(part) == 1, f"expected exactly one {part}"

    # role="alert" statt einer stillen Statuszeile: eine Absage ist keine
    # Fortschrittsmeldung.
    blocked = template[template.index('id="runBlocked"'):]
    assert 'role="alert"' in blocked[:200]

    # Vor query-send.js und consensus-run.js, die es benutzen.
    assert loads_before("usage-limit.js", "query-send.js")
    assert loads_before("usage-limit.js", "consensus-run.js")


def test_blocked_card_sits_where_the_answer_would_be():
    """Die Karte steht im Thread, nicht am Seitenkopf. Die Container-Spalte
    ordnet per flex-order — ohne eigenen Wert waere die Karte vor die Frage
    gerutscht."""
    template = read("templates/index.html")
    shell = read("static/css/shell.css")

    assert template.index('id="threadAsk"') < template.index('id="runBlocked"')
    assert template.index('id="runBlocked"') < template.index('class="input-section"')
    assert 'body:not(.is-hero) #runBlocked { order: 2; }' in shell

    # Der globale Button-Reset darf die Karten-Buttons nicht ueberfahren.
    controls = read("static/css/components-input.css")
    assert ":not(.run-blocked-btn)" in controls
    assert ":not(.run-blocked-close)" in controls


def test_server_error_codes_are_actually_matched():
    """chat.py sendet "total_usage_limit_exceeded" bzw.
    "deep_think_usage_limit_exceeded". Ein Vergleich auf
    "usage_limit_exceeded" trifft deshalb NIE — genau dieser Vergleich stand
    bis 2026-08-01 in consensus-run.js und machte die Absage dort stumm."""
    chat = read("app/api/routers/chat.py")
    assert 'f"{exc.limiting_bucket}_usage_limit_exceeded"' in chat

    consensus = read("static/js/consensus-run.js")
    assert 'error_code === "usage_limit_exceeded"' not in consensus
    assert "usageLimit.isLimitError" in consensus
    assert "usageLimit.show" in consensus

    # Ein Detektor fuer die ganze App, nicht drei divergierende Kopien.
    usage = read("static/js/usage-limit.js")
    assert "deep_think" in usage and "total" in usage
    assert "window.App.usageLimit?.isLimitError" in read("static/js/query-send.js")


def test_run_is_blocked_before_it_appears_to_start():
    """Die Vorab-Pruefung liegt VOR onPrepare(). Sonst zeigt der gefuehrte
    Lauf einen Schritt, den es nicht gibt, und nimmt ihn gleich wieder weg."""
    query = read("static/js/query-send.js")
    view = read("static/js/run-view.js")

    preflight = query.index("usageLimit?.blockIfExhausted")
    admission = query.index("const context = beginContext", preflight)
    assert preflight < admission
    assert "pipeline.onPrepare?.(context.startedAt)" in view

    # Die Demo kostet nichts und darf nie am Kontingent haengen. Ein echtes
    # Follow-up mit dem Text "Demo" bleibt dagegen ein normaler, gezaehlter
    # Kontextlauf und darf den archivierten Turn nicht umgehen.
    assert "if (isDemoQuery(question) && !registry.getSelectedConversationBasis())" in query

    # Der Server bleibt die Autoritaet: der /prepare-Zweig zeigt dieselbe
    # Karte, wenn die Vorab-Pruefung nichts wusste (Gast, frischer Tab).
    assert "usageLimit?.show?.({" in query


def test_blocked_follow_up_gets_its_context_back():
    """Die Karte sagt "nichts wurde gesendet" — dann darf der Lauf auch nichts
    verbraucht haben. consume() gibt den Follow-up-Kontext beim Absenden aus
    und ruft dabei reset(); ohne die gemerkte Kopie waere der Chip nach einer
    Kontingent-Absage weg und die naechste Frage ginge stillschweigend ohne
    Kontext raus."""
    query = read("static/js/query-send.js")
    registry = read("static/js/run-registry.js")

    # Die Basis wird vor Admission nur gelesen und danach als frozen Context-
    # Feld behalten. Scheitert der Lauf vor Fan-out, stellt der sichtbare Run
    # genau diese Basis zusammen mit Draft/Zitat wieder her.
    assert "const basis = registry.getSelectedConversationBasis();" in query
    restore = query.split("function restoreComposerAfterUnsentRun(context)", 1)[1].split(
        "function finishFailed", 1
    )[0]
    assert "context.previousExchange && context.basis" in restore
    assert "registry.selectConversationBasis(context.basis);" in restore
    assert "selectedConversationBasis = null;" in registry
    assert "if (failedBeforeFanout) restoreComposerAfterUnsentRun(context);" in query


def test_preflight_stays_conservative():
    """Lieber ein Server-Nein als ein falsches Client-Nein: eigene API-Keys
    zahlen nicht aufs Kontingent ein, und ein unbekannter Stand (null)
    blockiert nichts."""
    usage = read("static/js/usage-limit.js")

    assert "if (opts.useOwnKeys) return null;" in usage
    assert "runs && !runs.unlimited && runs.value <= 0" in usage
    assert "deep && !deep.unlimited && deep.value <= 0" in usage


def test_quota_numbers_have_a_single_source():
    """Die Karte liest dieselbe Zeile wie Ring und Panel (#usageDisplay ueber
    sidebar-quota). Eine zweite Rechnung waere eine zweite Stelle, an der die
    Zahl falsch sein kann."""
    quota = read("static/js/sidebar-quota.js")
    usage = read("static/js/usage-limit.js")

    assert "function deep()" in quota
    assert "deep: deep" in quota
    assert "window.App.sidebarQuota" in usage
    assert "parseLine(el(\"deepUsageDisplay\"))" in quota


def test_card_names_the_reset_and_never_sells_anything():
    """consens.io ist waehrend des Tests gratis — es gibt keinen Kauf-Ausweg.
    Die Karte sagt deshalb, wann das Kontingent zurueckkommt, und rechnet
    dafuer auf UTC-Tagen wie usage_repository.py."""
    usage = read("static/js/usage-limit.js")

    assert "getUTCDate() + 1" in usage
    assert "resets at " in usage
    for sales_word in ("Upgrade", "upgrade", "Buy", "Subscribe", "Pricing"):
        assert sales_word not in usage, f"the card must not sell: {sales_word}"


def test_deep_think_exhaustion_offers_the_cheaper_run():
    """Deep Think hat ein eigenes, kleineres Kontingent. Ist nur das leer,
    ist der normale Lauf noch da — und das ist ein Umweg, kein Stopp."""
    usage = read("static/js/usage-limit.js")
    shell = read("static/css/shell.css")

    assert "Send without Deep Think" in usage
    assert 'el("deepSearchToggle")' in usage
    assert 'new Event("change", { bubbles: true })' in usage

    # Warnfarbe statt Absagefarbe: die Ampel sagt "geht anders", nicht "geht
    # nicht".
    assert '.run-blocked[data-bucket="deep_think"]' in shell
    assert "--blocked-tone: var(--partial)" in shell


def test_css_cache_busting_needs_no_manual_bump(monkeypatch):
    """Frueher trug jede Teildatei eine handgepflegte ?v= und style.css musste
    bei JEDER Aenderung neu versioniert werden -- vergass man es, lieferte der
    Server altes CSS aus und die Karte erschien ungestylt. Die Marke kommt
    jetzt aus dem Inhalt (app/core/assets.py), also kann sie nicht mehr
    vergessen werden."""
    template = read("templates/index.html")
    shell = ROOT / "static" / "css" / "shell.css"

    assert not re.search(r'href="/static/style\.css\?v=', template)

    # Source mode: in built mode the hash comes from the build output, and
    # test_frontend_build.py is what keeps that output in sync with the sources.
    monkeypatch.setenv("FRONTEND_DEV", "1")
    assets_module._cache.clear()

    original = shell.read_bytes()
    before = frontend_assets().style
    try:
        shell.write_bytes(original + b"\n/* touched by the test */\n")
        assets_module._cache.clear()
        after = frontend_assets().style
    finally:
        shell.write_bytes(original)
        assets_module._cache.clear()

    assert before != after
