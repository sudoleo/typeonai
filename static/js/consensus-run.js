// =====================================================================
// consensus-run.js
// Consensus request only: window.getConsensus builds the /consensus
// payload, drives the SSE stream, and renders the result (main answer,
// structured differences, citation/share metadata).
//
// Lifecycle (visibility, gate, run-state, abort) lives in
// consensus-lifecycle.js and is used here via window.App.consensusLifecycle.
// Query-send (the per-model /ask_* requests that produce the answers this
// consumes) stays in templates/index.html until its own extraction.
//
// Shared deps via existing window contracts:
//   - window.App.consensusLifecycle, window.App.trackAppEvent
//   - window.createStreamRenderer / streamSSERequest / injectMarkdown
//   - window.lastQuestion (also written by query-send in index.html)
//   - window.currentMaxLimit / window.currentDeepLimit (usage mirrors)
//   - window.saveBookmarkConsensus / window.recordModelVote (firebase.js)
//   - window.auth (firebase.js)
// =====================================================================

(function () {
  window.App = window.App || {};

  const consensusLifecycle = window.App.consensusLifecycle;
  const trackAppEvent = window.App.trackAppEvent;
  const createStreamRenderer = window.createStreamRenderer;
  const streamSSERequest = window.streamSSERequest;
  const injectMarkdown = window.injectMarkdown;

  function isActiveConsensusRun(runId) {
    return consensusLifecycle.isActiveRun(runId);
  }

  function finishConsensusRun(runId) {
    consensusLifecycle.finishRun(runId);
  }

  function setConsensusSynthesizing(isSynthesizing) {
    consensusLifecycle.setSynthesizing(isSynthesizing);
  }

  function isAbortError(error) {
    return error && error.name === "AbortError";
  }

  function parseBestModel(differencesText) {
    if (typeof differencesText !== "string") return null;
    const regex = /BestModel:\s*(.*)/i;
    const match = differencesText.match(regex);
    return match ? match[1].trim() : null;
  }

  // --------- Follow-up-Fragen: Kontext-State + Input-Affordance ---------
  // Genau eine Kontext-Ebene: das Frage/Konsens-Paar des letzten erfolgreichen
  // Laufs. offer() merkt sich das Paar und zeigt den "Ask a follow-up"-Button
  // im Input-Bereich, arm() aktiviert den Kontext-Chip, consume() liefert den
  // context-Payload für query-send.js und räumt auf.
  // Kein Tier-Gate: ein Follow-up ist ein vollwertiger Lauf und zaehlt wie
  // jede andere Frage gegen das Tagesbudget — mehr kostet es niemanden.
  const FOLLOWUP_ICON =
    '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" ' +
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
    '<polyline points="9 14 4 9 9 4"></polyline>' +
    '<path d="M20 20v-7a4 4 0 0 0-4-4H4"></path></svg>';

  const DEFAULT_INPUT_PLACEHOLDER = "Enter your question";
  const FOLLOWUP_INPUT_PLACEHOLDER = "Ask a follow-up question";

  function truncateLabel(text, max) {
    const t = (text || "").trim().replace(/\s+/g, " ");
    return t.length > max ? t.slice(0, max - 1).trimEnd() + "…" : t;
  }

  const followup = {
    lastExchange: null, // {question, consensus} des letzten Konsens-Laufs
    armed: false,
    // True, solange der gerade laufende Query selbst eine Follow-up-Frage ist.
    // Follow-ups duerfen sich nicht verketten (Kostenkontrolle): der Konsens
    // einer Follow-up-Frage bietet keine weitere Follow-up-Affordance an.
    followupInFlight: false,
    // Was consume() zuletzt ausgegeben hat. Nur dafuer da, den Kontext
    // zurueckzuholen, wenn der Lauf gar nicht stattgefunden hat.
    spentExchange: null,

    offer(question, consensusText) {
      // Der aktuelle Konsens ist selbst die Antwort auf eine Follow-up-Frage:
      // keine weitere Ebene anbieten. Erst eine frische Frage schaltet die
      // Affordance wieder frei.
      if (this.followupInFlight) {
        this.followupInFlight = false;
        this.lastExchange = null;
        this.armed = false;
        this.render();
        return;
      }
      if (!question || !consensusText) return;
      this.lastExchange = { question: question, consensus: consensusText };
      this.armed = false;
      // Ein Lauf ist durchgelaufen: nichts mehr zurueckzuholen.
      this.spentExchange = null;
      this.render();
    },

    arm() {
      if (!this.lastExchange) return;
      this.armed = true;
      trackAppEvent("app_followup_armed");
      this.render();
      document.getElementById("questionInput")?.focus();
    },

    discard() {
      this.armed = false;
      this.render();
    },

    // Ein Lauf, der gar nicht stattgefunden hat (Kontingent leer), darf den
    // Follow-up-Chip nicht gefressen haben: consume() ist beim Absenden
    // passiert, gesendet wurde aber nichts. Sonst waere der Kontext weg,
    // waehrend die Absage-Karte sagt "nichts wurde gesendet" — und die
    // naechste Frage ginge stillschweigend ohne Kontext raus.
    restoreAfterBlockedRun() {
      if (!this.spentExchange) return;
      this.lastExchange = this.spentExchange;
      this.spentExchange = null;
      this.followupInFlight = false;
      this.armed = true;
      this.render();
    },

    // Neuer Lauf ohne Kontext bzw. Clear: Affordance und Chip verschwinden.
    // Loescht auch das In-Flight-Flag (frische Frage darf wieder anbieten).
    reset() {
      this.lastExchange = null;
      this.armed = false;
      this.followupInFlight = false;
      this.spentExchange = null;
      this.render();
    },

    // context-Payload für /prepare + /ask_*; danach ist der Chip weg und der
    // laufende Query als Follow-up markiert, damit sein Konsens keine weitere
    // Follow-up-Ebene anbietet (nur einmalig, Kostenkontrolle).
    consume() {
      if (!this.armed || !this.lastExchange) return null;
      const ctx = {
        previous_question: this.lastExchange.question,
        previous_consensus: this.lastExchange.consensus
      };
      const spent = this.lastExchange;
      this.reset();
      this.followupInFlight = true;
      // Nach reset(), sonst raeumt es sich selbst wieder weg.
      this.spentExchange = spent;
      return ctx;
    },

    // Zwei Orte am Composer, ein Zustand:
    //   #composerGate     — der Zustand NACH einer beantworteten Frage: der
    //                       Lauf ist zu Ende, und hier steht, wie es
    //                       weitergeht (neue Frage frei, Follow-up Pro).
    //                       Das Angebot sass frueher in der Provenance-Zeile
    //                       an der Antwort; es gehoert aber an das Feld, in
    //                       das die naechste Frage getippt wird.
    //   #followupChipBar  — der aktivierte KONTEXT-CHIP, am Eingabefeld,
    //                       weil er beschreibt, was gleich rausgeht
    render() {
      const gate = document.getElementById("composerGate");
      const chipBar = document.getElementById("followupChipBar");
      // Die alte Angebots-Leiste in der Provenance-Zeile bleibt im DOM (andere
      // Module fragen sie ab), traegt aber nichts mehr.
      const offerBar = document.getElementById("followupBar");
      if (offerBar) {
        offerBar.innerHTML = "";
        offerBar.hidden = true;
      }
      if (chipBar) {
        chipBar.innerHTML = "";
        chipBar.hidden = true;
      }
      if (gate) {
        gate.innerHTML = "";
        gate.hidden = true;
      }

      const armed = !!(this.armed && this.lastExchange);
      const input = document.getElementById("questionInput");
      if (input) {
        input.placeholder = armed ? FOLLOWUP_INPUT_PLACEHOLDER : DEFAULT_INPUT_PLACEHOLDER;
      }

      if (armed && chipBar) {
        const chip = document.createElement("div");
        chip.className = "followup-chip";
        chip.title = "Your next question is sent with the previous question and its consensus answer as context.";

        const icon = document.createElement("span");
        icon.className = "followup-chip-icon";
        icon.innerHTML = FOLLOWUP_ICON;

        const text = document.createElement("span");
        text.className = "followup-chip-text";
        text.textContent = "Follow-up to: “" + truncateLabel(this.lastExchange.question, 90) + "”";

        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "followup-chip-remove";
        remove.title = "Discard follow-up context";
        remove.setAttribute("aria-label", "Discard follow-up context");
        remove.textContent = "✕";
        remove.addEventListener("click", () => this.discard());

        chip.append(icon, text, remove);
        chipBar.appendChild(chip);
        chipBar.hidden = false;
      } else if (this.lastExchange && gate) {
        gate.appendChild(this.buildGate());
        gate.hidden = false;
      }

      this.syncInputLock();
    },

    // Der Lauf ist beantwortet — und damit zu Ende. Das Eingabefeld klappt
    // jetzt komplett zu; diese kompakte Zeile bietet nur die beiden ehrlichen
    // Wege weiter an: frischer Vergleich oder Follow-up mit Kontext.
    buildGate() {
      const wrap = document.createElement("div");
      wrap.className = "composer-gate-inner";

      const copy = document.createElement("span");
      copy.className = "composer-gate-copy";
      copy.textContent = "What would you like to do next?";
      wrap.appendChild(copy);

      const actions = document.createElement("span");
      actions.className = "composer-gate-actions";

      const newBtn = document.createElement("button");
      newBtn.type = "button";
      newBtn.className = "composer-gate-new";
      newBtn.textContent = "New comparison";
      newBtn.title = "Clear this run and ask a fresh question";
      newBtn.addEventListener("click", () => {
        const trigger = document.getElementById("newRunButton");
        if (trigger) trigger.click();
        else window.clearResponseBoxes?.();
        document.getElementById("questionInput")?.focus();
      });
      actions.appendChild(newBtn);

      const followupBtn = document.createElement("button");
      followupBtn.type = "button";
      followupBtn.className = "followup-offer-btn composer-gate-followup";
      followupBtn.innerHTML = FOLLOWUP_ICON + '<span class="followup-offer-label">Ask a follow-up</span>';
      followupBtn.title = "Ask a follow-up question. The previous question and its consensus answer go along as context. Counts as one run.";
      followupBtn.addEventListener("click", () => this.arm());
      actions.appendChild(followupBtn);

      wrap.appendChild(actions);
      return wrap;
    },

    // Nach einer fertigen Antwort ist der Composer kompakt: der naechste
    // Schritt wird bewusst gewaehlt ("Follow-up" nimmt den Kontext mit,
    // "New comparison" setzt das Feld leer zurueck).
    syncInputLock() {
      const locked = !!this.lastExchange && !this.armed;
      // Die Login-Schranke (updateQuestionInputAccess) bleibt die staerkere
      // Bedingung: dieses Gate darf ein gesperrtes Feld nie oeffnen. Tippen
      // und Absenden sind dabei zwei Rechte: wer auf die E-Mail-Bestaetigung
      // wartet, darf seine Frage schon schreiben.
      const canAsk = typeof window.userCanAskQuestions === "function"
        ? window.userCanAskQuestions()
        : true;
      const canType = typeof window.userCanTypeQuestions === "function"
        ? window.userCanTypeQuestions()
        : canAsk;
      const disabled = locked || !canAsk;
      const inputDisabled = locked || !canType;
      const input = document.getElementById("questionInput");
      const send = document.getElementById("sendButton");
      document.body.classList.toggle("composer-locked", locked);
      if (input) {
        input.disabled = inputDisabled;
        input.setAttribute("aria-disabled", inputDisabled ? "true" : "false");
        if (locked) input.placeholder = "Ask a follow-up — or start a new comparison";
      }
      // Waehrend eines Laufs ist derselbe Knopf der Abbrechen-Knopf; der darf
      // nicht gesperrt werden, sonst haengt der Nutzer im Lauf fest.
      if (send && !send.classList.contains("is-cancel-action")) {
        send.disabled = disabled;
        send.setAttribute("aria-disabled", disabled ? "true" : "false");
      }
    }
  };
  window.App.followup = followup;

  window.getConsensus = async function (trigger = "manual") {
    if (consensusLifecycle.isRunning()) {
      window.cancelCurrentConsensus();
      return;
    }

    if (
      typeof window.isAgentModeEnabled === "function"
      && window.isAgentModeEnabled()
      && window.isAgentModeRunning()
    ) {
      if (typeof window.updateConsensusButtonAvailability === "function") {
        window.updateConsensusButtonAvailability();
      }
      return;
    }

    const consensusRun = consensusLifecycle.startRun();
    const consensusRunId = consensusRun.runId;
    const consensusSignal = consensusRun.signal;

    // Der Composer wird beim Senden geleert (die Frage steht im Thread-Kopf),
    // deshalb ist window.lastQuestion die Quelle — das Eingabefeld enthaelt
    // hoechstens schon die NAECHSTE, noch nicht gesendete Frage.
    const question = (window.lastQuestion ?? "").trim()
      || (document.getElementById("questionInput")?.value ?? "").trim();
    // Status, ob eigene API Keys genutzt werden sollen
    const useOwnKeys = document.getElementById("useOwnKeysSwitch").checked;
    const deepThink = window.App.usageRun?.current?.deepThink
      ?? (document.getElementById("deepSearchToggle")?.checked === true);
    const usageRun = window.App.usageRun?.ensure?.(deepThink, useOwnKeys);

    let id_token = null;
    if (window.auth && window.auth.currentUser) {
      try {
        id_token = await window.auth.currentUser.getIdToken();
      } catch (e) {
        console.error("Token refresh error in consensus:", e);
      }
    }
    if (!id_token) {
      window.App.showPopup(useOwnKeys
        ? "Please log in before using your own API keys."
        : "Please log in before generating a consensus.");
      finishConsensusRun(consensusRunId);
      return;
    }

    // Wenn die Frage neu oder geändert ist, werden Firebase-Votes aktualisiert.
    if (!isActiveConsensusRun(consensusRunId)) {
      return;
    }

    if (question !== window.lastQuestion) {
      // Für jedes Modell prüfen, ob es als "best" markiert ist.

      // Ebenso für "excluded" (sofern du das separat erfassen möchtest).

      // Aktualisiere die letzte verarbeitete Frage.
      window.lastQuestion = question;
    }

    // Setze den Konsens-Bereich (Spinner etc.) und rufe anschließend deinen Konsens-Endpunkt auf.
    const consensusDiv = document.getElementById("consensusResponse");

    const consensus_model = document.getElementById("consensusModelDropdown").value;

    // Hole die Antwort-Boxen
    const openaiBox = document.getElementById("openaiResponse");
    const mistralBox = document.getElementById("mistralResponse");
    const claudeBox = document.getElementById("claudeResponse");
    const geminiBox = document.getElementById("geminiResponse");
    const deepseekBox = document.getElementById("deepseekResponse");
    const grokBox = document.getElementById("grokResponse");

    // Lies die Antworten (trim für überflüssige Leerzeichen)
    function isIncludedBox(box) {
      return box && !box.classList.contains("excluded");
    }

    function getIncludedAnswer(box) {
      if (!isIncludedBox(box)) return "";
      if (box.dataset.responseError === "true") return "";
      return (box.dataset.consensusAnswer || box.querySelector(".collapsible-content")?.innerText || "").trim();
    }

    function getIncludedSources(box) {
      if (!isIncludedBox(box) || !box.dataset.consensusSources) return [];
      try {
        const parsed = JSON.parse(box.dataset.consensusSources);
        return Array.isArray(parsed) ? parsed : [];
      } catch (e) {
        return [];
      }
    }

    // Abgewählte Modelle werden bewusst als leer gesendet.
    const answer_openai = getIncludedAnswer(openaiBox);
    const answer_mistral = getIncludedAnswer(mistralBox);
    const answer_claude = getIncludedAnswer(claudeBox);
    const answer_gemini = getIncludedAnswer(geminiBox);
    const answer_deepseek = getIncludedAnswer(deepseekBox);
    const answer_grok = getIncludedAnswer(grokBox);
    const model_sources = {
      OpenAI: getIncludedSources(openaiBox),
      Mistral: getIncludedSources(mistralBox),
      Anthropic: getIncludedSources(claudeBox),
      Gemini: getIncludedSources(geminiBox),
      DeepSeek: getIncludedSources(deepseekBox),
      Grok: getIncludedSources(grokBox)
    };

    // Überprüfe nur die Modelle, die nicht als "ausgeschlossen" markiert sind
    // UND nicht mit einem Fehler zurückkamen. Ein einzelner ausgefallener
    // Provider (z. B. Gemini 503) darf den Konsens nicht blockieren, solange
    // genug andere Antworten vorliegen – er wird einfach ausgelassen.
    function isAnswerableBox(box) {
      return isIncludedBox(box) && box.dataset.responseError !== "true";
    }
    const needOpenAI = isAnswerableBox(openaiBox);
    const needGemini = isAnswerableBox(geminiBox);
    const needMistral = isAnswerableBox(mistralBox);
    const needClaude = isAnswerableBox(claudeBox);
    const needDeepseek = isAnswerableBox(deepseekBox);
    const needGrok = isAnswerableBox(grokBox);
    const includedAnswerCount = [answer_openai, answer_mistral, answer_claude, answer_gemini, answer_deepseek, answer_grok]
      .filter(Boolean).length;
    const excludedModelCount = [openaiBox, mistralBox, claudeBox, geminiBox, deepseekBox, grokBox]
      .filter(box => box?.classList.contains("excluded")).length;

    if (
      !question ||
      !consensus_model ||
      includedAnswerCount < 2 ||
      (needOpenAI && !answer_openai) ||
      (needGemini && !answer_gemini) ||
      (needMistral && !answer_mistral) ||
      (needClaude && !answer_claude) ||
      (needDeepseek && !answer_deepseek) ||
      (needGrok && !answer_grok)
    ) {
      alert("Please provide at least two completed model answers before generating a consensus.");
      if (window.resetCredibilityFrame) {
        window.resetCredibilityFrame(consensusDiv.querySelector(".consensus-differences"));
      }
      const emptyBodyEl = window.App.consensusBodyEl(consensusDiv);
      if (emptyBodyEl) emptyBodyEl.innerText = "";
      consensusDiv.querySelector(".consensus-differences p").innerText = "";
      finishConsensusRun(consensusRunId);
      return;
    }

    trackAppEvent("app_consensus_started", {
      trigger,
      included_models: includedAnswerCount,
      excluded_models: excludedModelCount,
      custom_credentials: useOwnKeys,
      logged_in: !!window.auth?.currentUser
    });

    if (window.resetCredibilityFrame) {
      window.resetCredibilityFrame(consensusDiv.querySelector(".consensus-differences"));
    }
    // Rahmenlosen Konsens-Bereich sanft einblenden (Fade-In + Slide-Up).
    window.revealConsensusOutput?.();
    setConsensusSynthesizing(true);
    const spinnerBodyEl = window.App.consensusBodyEl(consensusDiv);
    if (spinnerBodyEl) spinnerBodyEl.innerHTML = window.consensusSpinnerHTML || window.spinnerHTML;
    // Bewusst OHNE Fallback auf den Typing-Spinner: der gefuehrte Lauf ist die
    // einzige Fortschrittsanzeige, das Differences-Panel bleibt bis zum
    // Ergebnis leer statt eine zweite Ladeanzeige zu zeigen.
    const differencesSpinnerEl = consensusDiv.querySelector(".consensus-differences p");
    if (differencesSpinnerEl) {
      differencesSpinnerEl.innerHTML = window.consensusDifferencesSpinnerHTML || "";
    }

    // Die übrigen Parameter wie "excluded_models" werden wie bisher ermittelt
    const excludedModels = [];
    if (openaiBox.classList.contains("excluded")) {
      excludedModels.push(openaiBox.getAttribute("data-model"));
    }
    if (mistralBox.classList.contains("excluded")) {
      excludedModels.push(mistralBox.getAttribute("data-model"));
    }
    if (claudeBox.classList.contains("excluded")) {
      excludedModels.push(claudeBox.getAttribute("data-model"));
    }
    if (geminiBox.classList.contains("excluded")) {
      excludedModels.push(geminiBox.getAttribute("data-model"));
    }
    if (deepseekBox.classList.contains("excluded")) {
      excludedModels.push(deepseekBox.getAttribute("data-model"));
    }
    if (grokBox.classList.contains("excluded")) {
      excludedModels.push(grokBox.getAttribute("data-model"));
    }

    // Hole API Keys aus localStorage
    const openaiKey = localStorage.getItem("openaiKey") || "";
    const mistralKey = localStorage.getItem("mistralKey") || "";
    const anthropicKey = localStorage.getItem("anthropicKey") || "";
    const geminiKey = localStorage.getItem("geminiKey") || "";
    const deepseekKey = localStorage.getItem("deepseekKey") || "";
    const grokKey = localStorage.getItem("grokKey") || "";

    // --------- NEU: genaue Konsensus-Metadaten für Zitation speichern ---------
    const includedModelsDetailed = [];

    function addModelForCitation(boxId, selectId, label) {
      const box = document.getElementById(boxId);
      if (!box) return;
      if (box.classList.contains("excluded")) return;

      const contentEl = box.querySelector(".collapsible-content");
      // textContent: die Boxen koennen hinter "Compare answers" liegen,
      // und innerText ist fuer display:none leer.
      const text = contentEl ? contentEl.textContent.trim() : "";
      if (!text) return; // nur Modelle mit Antwort

      const select = document.getElementById(selectId);
      let modelName = "";
      if (select) {
        const opt = select.options[select.selectedIndex];
        // Reines Modell-Label ohne Badge-Suffix ("· New") für die Zitation.
        modelName = (opt?.dataset.modelLabel || opt?.text || select.value || "").trim();
      }

      includedModelsDetailed.push(modelName ? `${label}: ${modelName}` : label);
    }

    addModelForCitation("openaiResponse", "openaiModelSelect", "OpenAI");
    addModelForCitation("mistralResponse", "mistralModelSelect", "Mistral");
    addModelForCitation("claudeResponse", "claudeModelSelect", "Anthropic Claude");
    addModelForCitation("geminiResponse", "geminiModelSelect", "Google Gemini");
    addModelForCitation("deepseekResponse", "deepseekModelSelect", "DeepSeek");
    addModelForCitation("grokResponse", "grokModelSelect", "Grok");

    const consensusSelect = document.getElementById("consensusModelDropdown");
    const consensusModelValue = consensusSelect ? consensusSelect.value : "";
    const consensusModelLabel = consensusSelect
      ? (consensusSelect.options[consensusSelect.selectedIndex]?.text || consensusModelValue)
      : consensusModelValue;

    // URL für die Zitation „aufgeräumt“
    let cleanUrl = window.location.href;
    try {
      const urlObj = new URL(window.location.href);
      cleanUrl = urlObj.origin + urlObj.pathname;
    } catch (e) {
      // falls URL-Parsing scheitert, nimm einfach href
    }

    window.consensusCitationMeta = {
      question,
      includedModels: includedModelsDetailed,
      consensusModel: consensusModelLabel || consensusModelValue,
      dateISO: new Date().toISOString(),
      url: cleanUrl
    };

    // Share-Feature: result_id des letzten Laufs zurücksetzen; Modell-
    // Labels (Option-Text) für die serverseitige Snapshot-Zitation.
    window.clearPreparedBookmarkShareResult?.();
    window.lastShareResultId = null;
    // Resolve-Persistenz: Payload des letzten erfolgreichen Laufs invalidieren,
    // damit eine Resolve-Runde nie in ein fremdes Bookmark schreibt.
    window.lastConsensusBookmarkPayload = null;
    const shareModelLabels = {};
    let streamedConsensusText = "";
    let completedConsensusText = "";
    [["OpenAI", "openaiModelSelect"], ["Mistral", "mistralModelSelect"],
     ["Anthropic", "claudeModelSelect"], ["Gemini", "geminiModelSelect"],
     ["DeepSeek", "deepseekModelSelect"], ["Grok", "grokModelSelect"]
    ].forEach(([provider, selectId]) => {
      const select = document.getElementById(selectId);
      if (!select) return;
      const opt = select.options[select.selectedIndex];
      // Reines Modell-Label (data-model-label) ohne Badge-Suffix wie "· New".
      const label = (opt?.dataset.modelLabel || opt?.text || select.value || "").trim();
      if (label) shareModelLabels[provider] = label;
    });

    try {
      // Reasoning-Marker ({reasoning:true} auf consensus.delta/differences.delta)
      // flippen das Spinner-Label, solange noch kein Text streamt: sichtbar
      // machen, dass die Engine bzw. der Differences-Judge gerade denkt.
      function flipThinkingLabel(el, text) {
        const label = el && el.querySelector(".thinking.consensus-thinking");
        if (!label || label.textContent === text) return;
        label.textContent = text;
      }
      const consensusMainEl = window.App.consensusBodyEl(consensusDiv);
      const consensusMainRenderer = createStreamRenderer(
        consensusMainEl,
        () => isActiveConsensusRun(consensusRunId)
      );
      const appendConsensusDelta = consensusMainRenderer.append.bind(consensusMainRenderer);
      consensusMainRenderer.append = (chunk) => {
        const text = typeof chunk === "string" ? chunk : "";
        if (text) streamedConsensusText += text;
        appendConsensusDelta(chunk);
      };
      // Der Konsens-Spinner nutzt .consensus-thinking statt .typing-indicator,
      // daher das generische markReasoning des Renderers ersetzen.
      consensusMainRenderer.markReasoning = () => {
        if (!isActiveConsensusRun(consensusRunId)) return;
        if (consensusMainEl.classList.contains("is-streaming")) return;
        flipThinkingLabel(consensusMainEl, "Reasoning");
      };
      // Differences-Deltas werden nicht mehr live gerendert: die Engine
      // liefert JSON, das erst mit dem final-Event als strukturierte UI
      // (Verdict, Badges, Karten) dargestellt wird. Der Spinner bleibt
      // bis dahin stehen; Reasoning-Marker flippen nur sein Label.
      const differencesEl = consensusDiv.querySelector(".consensus-differences p");
      const differencesPhaseRenderer = {
        append() {
          if (!isActiveConsensusRun(consensusRunId)) return;
          // Erstes Differences-Byte = der Konsens steht, der Judge laeuft.
          // Der gefuehrte Lauf schaltet hier auf seinen letzten Schritt.
          window.App?.consensusPipeline?.onDifferencesStart?.();
        },
        markReasoning() {
          if (!isActiveConsensusRun(consensusRunId)) return;
          window.App?.consensusPipeline?.onDifferencesStart?.();
          flipThinkingLabel(differencesEl, "Reasoning");
        },
        stop() {}
      };
      const consensusFinalPhaseRenderer = {
        append(text) {
          if (!isActiveConsensusRun(consensusRunId) || typeof text !== "string" || !text.trim()) return;
          completedConsensusText = text;
          // Replace the throttled delta render with the authoritative complete
          // answer before Differences starts. From here on, Judge/transport
          // failures must never blank the finished synthesis.
          injectMarkdown(consensusMainEl, completedConsensusText);
          window.App?.consensusPipeline?.onDifferencesStart?.();
        },
        markReasoning() {},
        stop() {}
      };
      const consensusRequestResult = await streamSSERequest("/consensus", {
          id_token: id_token,
          useOwnKeys: useOwnKeys,
          usage_run_key: usageRun?.key || null,
          deep_search: deepThink,
          question: question,
          answer_openai: answer_openai,
          answer_mistral: answer_mistral,
          answer_claude: answer_claude,
          answer_gemini: answer_gemini,
          answer_deepseek: answer_deepseek,
          answer_grok: answer_grok,
          model_sources: model_sources,
          model_labels: shareModelLabels,
          consensus_model: consensus_model,
          excluded_models: excludedModels,
          openai_key: openaiKey,
          mistral_key: mistralKey,
          anthropic_key: anthropicKey,
          gemini_key: geminiKey,
          deepseek_key: deepseekKey,
          grok_key: grokKey,
          keepalive: true
        }, consensusSignal, {
          "consensus.delta": consensusMainRenderer,
          "consensus.final": consensusFinalPhaseRenderer,
          "differences.delta": differencesPhaseRenderer
        });
      const data = consensusRequestResult.data || {};
      if (!data.consensus_response && completedConsensusText) {
        data.consensus_response = completedConsensusText;
      }
      if (data?.usage_run_status) {
        window.App.usageRun?.mark?.(data.usage_run_status);
      }
      const consensusErrorDetail =
        data?.detail && typeof data.detail === "object"
          ? data.detail
          : null;
      const consensusErrorMessage =
        consensusErrorDetail?.error ||
        consensusErrorDetail?.message ||
        data.error ||
        data.detail ||
        `Consensus HTTP ${consensusRequestResult.status}`;

      if (!isActiveConsensusRun(consensusRunId)) {
        return;
      }

      // Aktualisiere Free Requests, falls vorhanden (Deep Think nicht benötigt)
      const freeUsageRemaining =
        data.free_usage_remaining !== undefined
          ? data.free_usage_remaining
          : consensusErrorDetail?.free_usage_remaining;
      const deepRemaining =
        data.deep_remaining !== undefined
          ? data.deep_remaining
          : consensusErrorDetail?.deep_remaining;

      window.App.renderUsageDisplay({
        remaining: freeUsageRemaining,
        deepRemaining,
        totalLimit: data?.limit ?? consensusErrorDetail?.limit ?? window.currentMaxLimit,
        deepLimit: data?.deep_limit ?? consensusErrorDetail?.deep_limit ?? window.currentDeepLimit
      });

      if (consensusRequestResult.ok && data.consensus_response) {
        // Share-Feature: nur mit result_id aus dem Final-Event ist
        // Teilen möglich (serverseitiger Snapshot vorhanden).
        window.lastShareResultId = data.result_id || null;

        const mainEl = window.App.consensusBodyEl(consensusDiv);
        const diffEl = consensusDiv.querySelector(".consensus-differences p");

        if (mainEl) {
          // Konsens-Text inkl. [S1]-Links, Copy-Buttons usw.
          injectMarkdown(mainEl, data.consensus_response);
        }

        if (diffEl) {
          // Strukturierte Auswertung (Verdict-Header, Badges, Karten),
          // fällt bei fehlenden/ungültigen Daten auf den Freitext zurück.
          let structuredRendered = false;
          try {
            structuredRendered = window.renderConsensusInsights
              ? window.renderConsensusInsights(data.differences_data, includedAnswerCount)
              : false;
          } catch (renderError) {
            // A malformed/legacy Judge payload may degrade the comparison UI,
            // but it must never replace a valid Consensus answer.
            console.error("Error rendering consensus differences:", renderError);
            window.resetConsensusInsights?.();
          }

          if (!structuredRendered) {
            // Ohne strukturierte Daten ist der Freitext die einzige Analyse:
            // Panel sichtbar aufklappen statt sie zuzuklappen.
            window.App.differencesPanel?.expandForFallback?.();
            const diffsMD = data.differences || (data.error
              ? "The consensus answer is complete, but the differences analysis could not be completed."
              : "No differences found.");
            if (window.applyCredibilityFrame) {
              window.applyCredibilityFrame(diffEl, diffsMD);
            }
            const cleaned = window.colorizeCredibility
              ? window.colorizeCredibility(diffsMD)
              : diffsMD;

            // Differences auch über injectMarkdown → [S1]-Links inkl.
            injectMarkdown(diffEl, cleaned);
          }
        }

        // Follow-up-Affordance im Input-Bereich anbieten (Pro-Feature, Free
        // sieht den Teaser) — nicht bei Fehlertexten aus dem Consensus-Stream.
        if (data.consensus_response
            && !/^(Consensus error:|Invalid consensus model selected:)/i.test(data.consensus_response.trim())) {
          followup.offer(question, data.consensus_response);
        }

        // Payload merken: eine spätere Resolve-Runde hängt ihr Ergebnis an
        // differences_data und speichert das Bookmark damit erneut.
        window.lastConsensusBookmarkPayload = {
          question: question,
          consensusText: data.consensus_response,
          differencesText: data.differences,
          differencesData: data.differences_data || null
        };
        if (window.auth?.currentUser) {
          window.saveBookmarkConsensus(
            question, data.consensus_response, data.differences, data.differences_data,
            data.result_id, consensus_model, shareModelLabels
          );
        }
        trackAppEvent("app_consensus_completed", {
          status: data.error ? "partial" : "success",
          trigger,
          included_models: includedAnswerCount
        });
        window.App.watch?.showFeatureNudge?.();

        const bestModelFromConsensus =
          (data.differences_data && data.differences_data.best_model) ||
          parseBestModel(data.differences);
        if (bestModelFromConsensus) {
          window.recordModelVote(bestModelFromConsensus, "BestModel");
        }
      } else {
        if (window.resetCredibilityFrame) {
          window.resetCredibilityFrame(consensusDiv.querySelector(".consensus-differences"));
        }
        // Der Server sendet "total_usage_limit_exceeded" bzw.
        // "deep_think_usage_limit_exceeded" (chat.py). Der frühere Vergleich
        // auf "usage_limit_exceeded" traf deshalb nie — die Absage blieb
        // stumm. Erkennung liegt jetzt zentral in usage-limit.js.
        const consensusHitUsageLimit = window.App.usageLimit
          ? window.App.usageLimit.isLimitError(consensusErrorDetail || data, consensusErrorMessage)
          : false;
        if (consensusHitUsageLimit) {
          if (typeof window.setAgentModeStatus === "function" && window.isAgentModeEnabled?.()) {
            window.setAgentModeStatus("error", consensusErrorMessage);
          }
          // Die Antworten der Modelle stehen schon da; was fehlt, ist die
          // Synthese. Die Karte sagt genau das, statt den Nutzer vor einer
          // Fehlerzeile im Konsens-Feld raten zu lassen.
          window.App.usageLimit.show({
            data: consensusErrorDetail || data,
            source: "consensus",
            phase: "consensus"
          });
        }
        const errorBodyEl = window.App.consensusBodyEl(consensusDiv);
        if (errorBodyEl) errorBodyEl.innerText = "Error: " + consensusErrorMessage;
        consensusDiv.querySelector(".consensus-differences p").innerText = "";
        trackAppEvent("app_consensus_completed", {
          status: "error",
          trigger,
          included_models: includedAnswerCount
        });
        if (consensusRequestResult.status >= 500 || (consensusRequestResult.ok && data.error)) {
          window.App.reportCriticalError?.({
            type: "consensus_failed",
            phase: "consensus",
            message: String(consensusErrorMessage || "Consensus completed without a usable result."),
            details: `HTTP ${consensusRequestResult.status}; included models: ${includedAnswerCount}`
          });
        }
      }

    } catch (error) {
      if (isAbortError(error) || !isActiveConsensusRun(consensusRunId)) {
        return;
      }
      console.error("Error fetching consensus:", error);
      const failBodyEl = window.App.consensusBodyEl(consensusDiv);
      const preservedConsensus = completedConsensusText || streamedConsensusText;
      if (preservedConsensus) {
        if (failBodyEl) injectMarkdown(failBodyEl, preservedConsensus);
        window.resetConsensusInsights?.();
        window.App.differencesPanel?.expandForFallback?.();
        const diffEl = consensusDiv.querySelector(".consensus-differences p");
        if (diffEl) {
          injectMarkdown(
            diffEl,
            completedConsensusText
              ? "The consensus answer is complete, but the differences analysis could not be completed."
              : "The connection ended before the consensus response could be confirmed as complete."
          );
        }
      } else {
        if (window.resetCredibilityFrame) {
          window.resetCredibilityFrame(consensusDiv.querySelector(".consensus-differences"));
        }
        if (failBodyEl) failBodyEl.innerText = "Error in the consensus calculation.";
        consensusDiv.querySelector(".consensus-differences p").innerText = "";
        window.App.reportCriticalError?.({
          type: "consensus_failed",
          phase: "consensus_connection",
          message: error?.message || "The consensus request was interrupted without a result.",
          error
        });
      }
      trackAppEvent("app_consensus_completed", {
        status: preservedConsensus ? "partial" : "error",
        trigger,
        included_models: includedAnswerCount
      });
    } finally {
      finishConsensusRun(consensusRunId);
    }
  };
})();
