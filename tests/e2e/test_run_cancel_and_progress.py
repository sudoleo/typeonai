"""Regressionen rund um den laufenden Consensus-Durchlauf:

1. Der Send-Button bleibt Cancel-Button, bis auch Consensus/Differences
   fertig sind (vorher sprang er zurueck, sobald die Modelle geantwortet
   hatten - der Lauf war dann nicht mehr abbrechbar).
2. Die Chip-Ladebalken starten beim ZWEITEN Lauf wieder bei 0 (vorher las
   der erste Tick noch das "complete" des Vorlaufs; der monotone Balken
   rastete sofort auf 100 % ein und haengte dort fest).
3. Die Fortschrittsleiste des Differences-Laufs faellt waehrend der
   Konsens-Synthese noch nicht an und laeuft danach aus dem echten
   Judge-Stream hoch (der Judge liefert JSON, das nicht gerendert wird -
   ohne Leiste steht dort nur ein stummer Spinner).
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


def test_differences_progress_bar_fills_from_the_judge_stream(app_page):
    # Ab dem Consensus-Start engmaschig mitschneiden: die Leiste muss bei 0
    # anfangen (Synthese laeuft noch) und mit dem Judge-Stream hochlaufen.
    app_page.evaluate(
        """() => {
          window.__diffSamples = [];
          window.__diffSampler = setInterval(() => {
            const bar = document.querySelector(".differences-progress");
            if (bar) {
              window.__diffSamples.push(
                parseFloat(bar.style.getPropertyValue("--diff-progress") || "0")
              );
            }
          }, 30);
        }"""
    )

    _send_question(app_page)
    _wait_for_all_final_answers(app_page)
    _wait_for_consensus_start(app_page)
    _wait_for_consensus_idle(app_page)
    app_page.evaluate("() => clearInterval(window.__diffSampler)")

    samples = app_page.evaluate("() => window.__diffSamples")
    assert samples, "Differences-Spinner ohne Fortschrittsleiste gerendert"
    assert samples[0] == 0, (
        "Leiste startet nicht bei 0 - waehrend der Synthese darf sie noch "
        f"nicht laufen (erster Wert: {samples[0]})"
    )
    assert max(samples) > 0, "Leiste blieb waehrend des Judge-Streams auf 0"
    # Monoton: ein Ruecksprung waere fuer den Nutzer ein Fehlsignal.
    assert all(b >= a for a, b in zip(samples, samples[1:])), (
        f"Leiste sprang zurueck: {samples}"
    )
