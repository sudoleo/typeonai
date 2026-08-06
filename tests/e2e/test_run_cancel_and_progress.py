"""Regressionen rund um den laufenden Consensus-Durchlauf:

1. Der Send-Button bleibt Cancel-Button, bis auch Consensus/Differences
   fertig sind (vorher sprang er zurueck, sobald die Modelle geantwortet
   hatten - der Lauf war dann nicht mehr abbrechbar).
2. Die Chip-Ladebalken starten beim ZWEITEN Lauf wieder bei 0 (vorher las
   der erste Tick noch das "complete" des Vorlaufs; der monotone Balken
   rastete sofort auf 100 % ein und haengte dort fest).

Die frueher hier gepruefte Fortschrittsleiste des Differences-Laufs ist
entfallen: der gefuehrte Lauf ueber dem Thread zeigt dieselbe Phase schon
mit einem Balken, zwei Balken fuer einen Vorgang waren zu viel.
"""

from playwright.sync_api import expect

from test_smoke import _send_question, _wait_for_all_final_answers


def _wait_for_consensus_idle(page, timeout=60000):
    page.wait_for_function(
        "() => window.App.consensusLifecycle.isRunning() === false",
        timeout=timeout,
    )


def _wait_for_consensus_start(page, timeout=30000):
    page.wait_for_function(
        "() => window.App.consensusLifecycle.isRunning() === true",
        timeout=timeout,
    )


def _wait_for_run_start(page, timeout=20000):
    """Beim zweiten Lauf stehen die alten Antworten noch in den Boxen (der
    Send-Pfad wartet erst auf Token/Usage-Release). Ohne diesen Sync wuerde
    ein Wait auf die Antworttexte sofort mit dem Vorlauf durchlaufen."""
    page.wait_for_function(
        """() => document.getElementById("openaiResponse").dataset.responseState === "pending" """,
        timeout=timeout,
    )


def test_send_button_stays_cancelable_until_consensus_is_done(app_page):
    # Zustand des Send-Buttons exakt beim Consensus-Start festhalten, damit
    # der Test nicht gegen die Stream-Dauer rennt.
    app_page.evaluate(
        """() => {
          const lifecycle = window.App.consensusLifecycle;
          const originalStartRun = lifecycle.startRun;
          window.__sendButtonWasCancelAtConsensusStart = null;
          lifecycle.startRun = function () {
            const run = originalStartRun.apply(this, arguments);
            window.__sendButtonWasCancelAtConsensusStart = document
              .getElementById("sendButton")
              .classList.contains("is-cancel-action");
            return run;
          };
        }"""
    )

    _send_question(app_page)
    _wait_for_all_final_answers(app_page)

    app_page.wait_for_function(
        "() => window.__sendButtonWasCancelAtConsensusStart !== null",
        timeout=30000,
    )
    assert app_page.evaluate("() => window.__sendButtonWasCancelAtConsensusStart") is True

    # Erst wenn der Consensus durch ist, wird wieder gesendet statt abgebrochen.
    _wait_for_consensus_idle(app_page)
    expect(app_page.locator("#sendButton")).not_to_have_class("is-cancel-action")


def test_send_button_cancels_a_running_consensus(app_page):
    _send_question(app_page)
    _wait_for_all_final_answers(app_page)
    app_page.wait_for_function(
        "() => window.App.consensusLifecycle.isRunning() === true",
        timeout=30000,
    )

    app_page.evaluate("() => window.sendQuestion()")
    assert app_page.evaluate("() => window.App.consensusLifecycle.isRunning()") is False
    expect(app_page.locator("#sendButton")).not_to_have_class("is-cancel-action")


def test_chip_progress_bars_restart_from_zero_on_a_second_run(app_page):
    app_page.set_viewport_size({"width": 390, "height": 844})
    app_page.evaluate("() => window.setAgentMode(true, { persist: true })")

    _send_question(app_page)
    _wait_for_all_final_answers(app_page)
    # Erst wenn der Consensus wirklich durch ist, ist der Send-Button wieder
    # ein Send-Button - vorher wuerde der naechste Klick den Lauf abbrechen.
    _wait_for_consensus_start(app_page)
    _wait_for_consensus_idle(app_page)

    # Nach Lauf 1 stehen alle Balken auf 100 %.
    assert app_page.evaluate(
        """() => [...document.querySelectorAll(".agent-mode-chip")]
          .every((chip) => parseFloat(chip.style.getPropertyValue("--stream-progress") || "0") > 0.99)"""
    ) is True

    # Lauf 2 engmaschig mitschneiden: der Balken MUSS zwischendurch wieder
    # klein werden, sonst haengt er auf dem Ergebnis des Vorlaufs fest.
    app_page.evaluate(
        """() => {
          window.__progressSamples = [];
          window.__progressSampler = setInterval(() => {
            const values = [...document.querySelectorAll(".agent-mode-chip")]
              .map((chip) => parseFloat(chip.style.getPropertyValue("--stream-progress") || "0"));
            if (values.length) window.__progressSamples.push(Math.max(...values));
          }, 30);
        }"""
    )

    # Nach einer beantworteten Frage steht der Follow-up-Kontext am Composer.
    # Dieser Test misst einen frischen Lauf, deshalb hier bewusst der Ausstieg
    # ueber "New comparison" — genau wie in der App.
    app_page.locator("#followupChipBar .followup-newrun").click()
    expect(app_page.locator("#questionInput")).to_be_visible()
    expect(app_page.locator("#questionInput")).to_be_enabled()

    _send_question(app_page)
    _wait_for_run_start(app_page)
    _wait_for_all_final_answers(app_page)
    app_page.evaluate("() => clearInterval(window.__progressSampler)")

    samples = app_page.evaluate("() => window.__progressSamples")
    assert samples, "Keine Chips gefunden - Agent Mode nicht aktiv?"
    assert min(samples) < 0.3, (
        "Ladebalken wurde beim zweiten Lauf nie zurueckgesetzt "
        f"(kleinster Wert: {min(samples)})"
    )
    assert max(samples) > 0.99, "Ladebalken lief im zweiten Lauf nicht voll"
