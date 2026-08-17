from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_consensus_result_precedes_model_answers_and_run_block_is_loaded():
    template = read("templates/index.html")

    assert template.index('class="consensus-section"') < template.index(
        'class="response-section"'
    )
    assert 'id="consensusRun"' in template
    assert 'id="runStatus"' in template
    assert 'src="/static/js/consensus-progress.js?' in template
    assert template.index("/static/js/agent-mode.js?") < template.index(
        "/static/js/consensus-progress.js?"
    ) < template.index("/static/js/consensus-lifecycle.js?")


def test_run_block_shows_one_step_at_a_time():
    """The guided run's whole point is that exactly one step is active. The
    markup therefore has ONE label, ONE count, ONE bar and ONE next line —
    if a second set ever appears here, the reduction has been undone."""
    template = read("templates/index.html")

    for single in ('id="runLabel"', 'id="runCount"', 'id="runBar"', 'id="runNext"'):
        assert template.count(single) == 1, f"expected exactly one {single}"

    # Finished steps collapse into the past line; per-model rows live in the
    # detail container and are only shown while the models answer.
    assert 'id="runPast"' in template
    assert 'id="runDetail"' in template


def test_run_covers_every_phase_and_terminal_state():
    progress = read("static/js/consensus-progress.js")
    lifecycle = read("static/js/consensus-lifecycle.js")
    query = read("static/js/query-send.js")
    demo = read("static/demo.js")

    # Four phases, in order, each with its own step.
    for stage in ("prepare", "answers", "consensus", "differences"):
        assert f'{stage}: {{' in progress or f'"{stage}"' in progress, stage

    assert 'responseState === "complete"' in progress
    assert "onConsensusStart" in lifecycle
    assert "onConsensusEnd" in lifecycle
    assert 'setAgentModeStatus("canceled")' in query
    assert "consensusPipeline?.onConsensusEnd" in demo

    # The run starts at /prepare, not at the fan-out: the gap between click
    # and first answer is exactly where a user needs to see something happen.
    assert "consensusPipeline?.onPrepare" in query
    # The differences judge gets its own step, signalled from the stream.
    assert "onDifferencesStart" in read("static/js/consensus-run.js")


def test_completed_consensus_survives_differences_or_transport_failure():
    run = read("static/js/consensus-run.js")
    chat = read("app/api/routers/chat.py")

    assert 'sse_pack("consensus.final"' in chat
    assert '"consensus.final": consensusFinalPhaseRenderer' in run
    assert "const preservedConsensus = completedConsensusText || streamedConsensusText" in run
    assert "differences analysis could not be completed" in run


def test_run_is_compact_and_unknowable_phases_stay_indeterminate():
    css = read("static/css/shell.css")

    assert ".run.is-visible" in css
    # Collapses away instead of vanishing, so the eye follows the answer up.
    assert ".run.is-gone" in css
    assert "height: 2px" in css
    # A phase that cannot report a share of itself sweeps, it does not fake
    # a percentage.
    assert ".run-track.is-indeterminate" in css
    assert "runSweep" in css
    assert "@media (prefers-reduced-motion: reduce)" in css


def test_run_hands_over_to_a_provenance_line():
    template = read("templates/index.html")
    progress = read("static/js/consensus-progress.js")
    css = read("static/css/shell.css")

    assert 'id="runProvenance"' in template
    assert 'id="runProvenanceFacts"' in template
    assert 'id="runReplayButton"' in template
    assert ".run-provenance" in css
    assert "renderProvenance" in progress


def test_result_footer_has_one_boundary_before_the_composer():
    """Closed drawers and the legacy divider must not stack hairlines."""
    css = read("static/css/shell.css")

    assert ".consensus-differences-panel[open]" in css
    assert ".consensus-divider {\n  display: none;" in css
    assert "body:not(.is-hero) .input-section::before" in css


def test_the_composer_carries_no_followup_affordance_at_all():
    """Ein Gespraech laeuft ueber das Eingabefeld weiter — ohne Kontext-Chip,
    ohne Angebots-Leiste und ohne ein zweites "New comparison" am Composer.
    Der Ausstieg steht in der Sidebar."""
    template = read("templates/index.html")
    run = read("static/js/consensus-run.js")

    assert 'id="followupChipBar"' not in template
    assert 'id="followupBar"' not in template
    assert "followupChipBar" not in run
    assert "followup-newrun" not in run
    assert 'id="newRunButton"' in template

    # Der Kontext geht trotzdem immer mit: ein fortsetzbarer Turn ist armed.
    assert "isArmed() {\n      return !!this.lastExchange;\n    }" in run


def test_followup_archives_the_previous_turn_before_rendering_the_next_one():
    template = read("templates/index.html")
    run = read("static/js/consensus-run.js")
    query = read("static/js/query-send.js")
    app_init = read("static/js/app-init.js")
    css = read("static/css/shell.css")

    # Der statische Verlauf steht im DOM vor dem aktiven Turn. Der Live-
    # Renderbaum mit seinen einmaligen IDs bleibt dadurch unveraendert.
    assert template.index('id="threadHistory"') < template.index('id="threadAsk"')
    assert "archiveCurrentExchange()" in run
    assert 'clone.removeAttribute("id")' in run
    assert "renderStoredTurn(turnData)" in run
    assert "renderStoredTurns(turns)" in run
    assert 'node.querySelector?.(".thread-history-question-text")' in run
    assert "buildStoredAgreement(differencesData)" in run
    assert "const insightRoot = window.App.consensusBodyEl?.() || document" in read(
        "static/js/consensus-insights.js"
    )
    # Erst archivieren, dann uebernehmen: promote() macht die abgeschickte
    # Nachricht zum Kopf des neuen Turns (siehe sentMessage in query-send.js).
    archive_at = query.index("window.App.followup?.archiveCurrentExchange?.();")
    assert archive_at < query.index("sentMessage.promote()", archive_at)
    assert "window.App.setThreadQuestion?.(this.question)" in query
    assert "clearHistory?.()" in app_init

    # User-Turns stehen rechts; Consensus-Turns bleiben als Lesetext links.
    assert "align-items: flex-end" in css
    assert ".thread-history-question" in css
    assert ".thread-history-answer-body" in css
    assert ".thread-history-verdict" in css
    assert "flex: 1 1 0" in css
    assert ".thread-history-verdict .verdict-judge" in css
    insights = read("static/js/consensus-insights.js")
    assert "TOPIC_MAX_SHOWN = 1" in insights
    assert "TOPIC_MAX_WORDS" not in insights


def test_the_sent_message_leaves_the_field_before_prepare_runs():
    """Zwischen Klick und dem Moment, in dem der Lauf den neuen Turn aufmacht,
    liegen /prepare und — im laufenden Gespraech — das Binden des Chat-
    Kontexts. Ab der dritten Frage dauerte das so lange, dass es aussah, als
    sei der Klick ins Leere gegangen: die Frage stand unveraendert im Feld
    (User-Befund 2026-08-15). Sie geht deshalb SOFORT raus und steht als
    eigene Blase im Thread — der Vorgaenger bleibt bis zum Archivieren
    unangetastet."""
    template = read("templates/index.html")
    query = read("static/js/query-send.js")
    core = read("static/js/app-core.js")
    css = read("static/css/shell.css")

    # Eigener Block, damit der aktive Kopf (und mit ihm die alte Antwort
    # darunter) stehen bleiben kann, solange der Lauf noch scheitern darf.
    assert 'id="threadPendingAsk"' in template
    assert 'id="threadPendingAskText"' in template
    assert 'id="threadPendingAskAttachments"' in template
    assert template.index('id="threadAsk"') < template.index('id="threadPendingAsk"')
    assert template.index('id="threadPendingAsk"') < template.index('id="consensusRun"')

    # Das Feld ist leer, bevor die erste Anfrage rausgeht.
    hold_at = query.index("sentMessage.hold(question,")
    assert hold_at < query.index("prepareWithUsageRetry(", hold_at)
    assert hold_at < query.index("chatSession?.beginRun?.({", hold_at)
    hold_block = query.split("hold(question,", 1)[1].split("promote()", 1)[0]
    assert 'input.value = "";' in hold_block
    assert "window.App.setPendingThreadQuestion?.(this.question" in hold_block

    # Die letzte Antwort bleibt im Gespraech stehen, bis sie archiviert ist.
    assert "if (!followupRequested) window.hideConsensusOutput?.();" in query

    # Sie steht dort, wo sie gleich als Kopf weiterlebt: unter der bisherigen
    # Antwort, mit dem gefuehrten Lauf darunter.
    assert ".thread-ask-pending { order: 4; }" in css
    assert "body.thread-message-pending:not(.is-hero) #consensusRun { order: 4; }" in css
    assert "body.thread-message-pending:not(.is-hero) .response-section { order: 3; }" in css
    assert css.index("body:not(.is-hero) .consensus-section { order: 3; }") < css.index(
        ".thread-ask-pending { order: 4; }"
    )

    # Jeder Kopf im Thread wird vom selben Renderer gefuellt (Clamp, Aufklapp-
    # Link, ResizeObserver) — sonst verhielte sich die Blase anders als der
    # Kopf, zu dem sie wird.
    assert "function renderThreadQuestion(wrap, text, question)" in core
    assert "const threadAskResizeObservers = new WeakMap()" in core
    # Wer den Kopf setzt, hat die schwebende Nachricht uebernommen — das gilt
    # auch fuer Bookmark-Restore, "New comparison" und den Direktvergleich.
    set_head = core.split("function setThreadQuestion(", 1)[1].split("}", 1)[0]
    assert "clearPendingThreadQuestion()" in set_head


def test_a_run_that_never_happens_gives_the_message_back():
    """Kontingent leer, fehlender Key, abgebrochen: dann ist nichts
    rausgegangen. Die Nachricht gehoert unveraendert und abschickbar zurueck
    ins Feld, und der bisherige Turn bleibt, wo er ist."""
    query = read("static/js/query-send.js")

    restore_block = query.split("restore() {", 1)[1].split("\n      }", 1)[0]
    assert "window.App.clearPendingThreadQuestion?.();" in restore_block
    # Zurueck kommt der Composer-Stand, aus dem die Nachricht entstanden ist:
    # das Getippte im Feld, das Zitat wieder darueber.
    assert "input.value = this.draft;" in restore_block
    assert "window.App.quote?.set?.(this.quote)" in restore_block
    # Ein inzwischen getippter Entwurf wird nie ueberschrieben.
    assert "!input.value.trim()" in restore_block

    # Jeder Pfad, der den Follow-up-Kontext zurueckgibt, gibt auch die
    # Nachricht zurueck: beides gehoert zu demselben Lauf, der nicht
    # stattgefunden hat.
    lines = query.splitlines()
    blocked = [i for i, line in enumerate(lines) if "restoreAfterBlockedRun?.()" in line]
    assert blocked
    for index in blocked:
        # Grosszuegiges Fenster: im Abbruch-Pfad liegt die Ruecknahme der
        # Antwortboxen dazwischen.
        window = "\n".join(lines[max(0, index - 20):index])
        assert "sentMessage.restore();" in window, (
            f"line {index + 1} restores the context but not the message"
        )


def test_the_new_message_is_scrolled_to_once_and_never_fights_the_reader():
    """Die Bewegung beim Absenden ist die einzige, die dieser Modul macht —
    und sie ist die, die der Nutzer selbst ausgeloest hat. Sie geht nie nach
    oben, unterbleibt bei kurzen Wegen und bricht bei der ersten eigenen
    Geste ab."""
    core = read("static/js/app-core.js")

    reveal = core.split("function startSentMessageReveal(", 1)[1].split(
        "\n  }", 1
    )[0]
    # Nie nach oben und nie fuer ein paar Pixel.
    assert "Math.max(0, Math.min(wanted, maxTop)) - from" in reveal
    assert "if (distance < REVEAL_MIN_DISTANCE) return;" in reveal
    # Nie ueber den Boden des Dokuments hinaus.
    assert "document.documentElement.scrollHeight - window.innerHeight" in reveal
    # Die erste eigene Geste gewinnt.
    assert 'REVEAL_INTERRUPTS = ["wheel", "touchstart", "keydown", "pointerdown"]' in core
    assert "stopSentMessageReveal()" in reveal
    assert 'window.matchMedia("(prefers-reduced-motion: reduce)")' in core
    # Genau ein Aufrufer: das Absenden. Alles andere waere ein Thread, der
    # beim Lesen unter den Fingern wegwandert.
    assert read("static/js/query-send.js").count("revealSentMessage") == 1


def test_archived_questions_clamp_like_the_active_one():
    """Eine lange Frage bleibt auch im Verlauf auf drei Zeilen eingeklappt.
    Ohne Clamp hat ab dem zweiten Turn jede lange Frage den Thread wieder
    aufgerissen — es sah aus, als schalte sich das Einklappen im Lauf eines
    Chats ab (User-Befund 2026-08-14)."""
    css = read("static/css/shell.css")
    run = read("static/js/consensus-run.js")
    core = read("static/js/app-core.js")

    clamp_block = css.split(".thread-history-question-text {", 1)[1].split("}", 1)[0]
    assert "-webkit-line-clamp: 3" in clamp_block
    assert ".thread-history-question.is-open .thread-history-question-text" in css
    assert ".thread-history-question.is-long .thread-ask-more" in css

    history_block = run.split("appendHistoryTurn(", 1)[1].split(
        "archiveCurrentExchange()", 1
    )[0]
    assert 'questionMore.className = "thread-ask-more"' in history_block
    assert 'question.classList.toggle(\n          "is-long",' in history_block

    # Ein Klick auf den Link gehoert zu der Frage, unter der er steht — nicht
    # fest zum aktiven Kopf.
    assert 'event.target.closest(".thread-ask-more")' in core
    assert 'more.closest(".thread-ask, .thread-history-question")' in core


def test_archived_turns_use_the_same_drawer_row_as_the_live_answer():
    """Differences, Sources und Model answers liegen im Verlauf NEBEN-, nicht
    untereinander: dieselben .consensus-tab-Chips wie im Fuss der aktiven
    Antwort. Drei gestapelte <details> haben jeden alten Turn im Thread um
    drei Zeilen verlaengert, obwohl alles zugeklappt war."""
    run = read("static/js/consensus-run.js")
    css = read("static/css/shell.css")

    history_block = run.split("appendHistoryTurn(", 1)[1].split(
        "archiveCurrentExchange()", 1
    )[0]
    assert "thread-history-details" not in history_block
    assert '"consensus-footer-tabs thread-history-tabs"' in history_block
    assert 'tab.className = "consensus-tab"' in history_block
    assert '"Review differences",\n          "Differences",' in history_block
    assert 'addDrawer("Verify sources", "Sources", turnSources.length' in history_block
    assert 'addDrawer("Compare answers", "Answers", usableAnswers.length' in history_block
    # Der Verlauf teilt sich den DOM mit dem Live-Renderbaum: keine ID darf
    # dort ein zweites Mal auftauchen.
    assert "`threadHistoryPanel-${++panelSequence}`" in history_block
    assert 'tab.setAttribute("aria-controls", panelId)' in history_block

    assert ".thread-history-footer" in css
    assert ".thread-history-panel[hidden]" in css


def test_composer_row_is_reduced_to_attach_run_switch_and_send():
    """Agent Mode, the mode explainer and the clear button left the composer.
    Their functions live in Settings, the (+) menu and 'New comparison'."""
    template = read("templates/index.html")

    assert 'id="attachTrigger"' in template
    assert 'id="consensusModelDropdown"' in template
    assert 'id="sendButton"' in template

    assert 'id="toggleAllButton"' not in template
    assert 'id="modeExplainerTrigger"' not in template
    assert 'id="clearButton"' not in template

    # ...but the functions are still reachable.
    assert 'id="agentModeSwitch"' in template
    assert 'id="autoConsensusToggle"' in template
    assert 'id="deepSearchToggle"' in template
    assert 'id="newRunButton"' in template


def test_sidebar_header_groups_brand_and_toggle_before_new_comparison():
    template = read("templates/index.html")

    brand = template.index('class="sidebar-brand-row"')
    toggle = template.index('id="sidebarToggleInner"')
    new_run = template.index('id="newRunButton"')
    search = template.index('id="chatSearch"')

    assert brand < toggle < new_run < search
    assert 'class="sidebar-top"' not in template
    # Exactly one sidebar toggle in the markup besides the floating one.
    assert template.count('id="sidebarToggleInner"') == 1
    assert template.count('id="toggleSidebarButton"') == 1


def test_consensus_loader_matches_the_run_visual_language():
    """Die synthetisierende Box bleibt still, der Loader sind die Punkte.

    Frueher lief hier eine wandernde Linie (.is-synthesizing::after mit
    @keyframes consensusLoadingLine) und davor ein Sweep-Verlauf. Beides wurde
    bewusst entfernt: der gefuehrte Lauf (#consensusRun) ist die EINZIGE
    Fortschrittsanzeige, die Antwortflaeche selbst zappelt nicht mehr. Der Test
    haelt genau diese Entscheidung fest, statt die geloeschte Animation zu
    fordern.
    """
    consensus_css = read("static/css/components-consensus.css")
    feedback_css = read("static/css/components-feedback.css")

    # Die Box waehrend der Synthese: rahmenlos und ohne eigene Bewegung.
    assert ".consensus-box.is-synthesizing" in consensus_css
    assert "background: transparent" in consensus_css
    # Keine wandernde Linie und kein Sweep mehr - weder als Regel noch als
    # Keyframe, sonst kaeme die alte Unruhe durch die Hintertuer zurueck.
    assert ".consensus-box.is-synthesizing::after" not in consensus_css
    assert "consensusLoadingLine" not in consensus_css
    assert "consensusLoadingLine" not in feedback_css
    assert "consensusSynthesisSweep" not in feedback_css
    # Der verbleibende Loader ist die ruhige Punktreihe (4px, kein Schatten).
    assert "animation: consensusLoaderDots" in feedback_css
    assert "width: 4px" in feedback_css
    assert "box-shadow: none" in feedback_css
