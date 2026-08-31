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

  // Namen aller konfigurierten Familien (Serververtrag), aus der einen
  // Frontend-Quelle window.App.modelPrefs.
  function familyKeys() {
    return (window.App.modelPrefs || []).map(pref => pref.key);
  }

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

  // "BestModel: Anthropic" ist Buchhaltung fuer die Modell-Wertung, kein Satz
  // fuer Leser. Im Freitext-Fallback stand sie bisher sichtbar unter der
  // Analyse.
  function stripBestModelLine(differencesText) {
    return String(differencesText || "")
      .replace(/^[ \t]*\**BestModel:\**[^\n]*$/gim, "")
      .trim();
  }

  // Die Karten eines archivierten Turns sollen die Modelle nennen, die DAMALS
  // geantwortet haben. modelDisplayName in consensus-insights.js liest die
  // Live-Antwortboxen — die tragen laengst die Auswahl des neuesten Laufs.
  function storedModelLabeller(modelAnswers) {
    const labels = {};
    Object.entries(modelAnswers && typeof modelAnswers === "object" ? modelAnswers : {})
      .forEach(([provider, item]) => {
        const key = String(item?.provider || provider || "").toLowerCase();
        const label = String(item?.model_label || "").trim();
        if (key && label) labels[key] = label;
      });
    return function (model) {
      return labels[String(model || "").toLowerCase()] || model;
    };
  }

  // --------- Follow-up-Fragen: Kontext-State ---------
  // Genau eine Kontext-Ebene: das Frage/Konsens-Paar des letzten erfolgreichen
  // Laufs. offer() merkt sich das Paar, consume() liefert den context-Payload
  // für query-send.js und räumt auf.
  //
  // Es gibt KEINE Follow-up-Entscheidung mehr: ein sichtbares Gespraech laeuft
  // ueber das Eingabefeld einfach weiter (wie in jedem Chat). Der frueher hier
  // gerenderte Kontext-Chip samt "New comparison" ist ersatzlos entfallen —
  // der Ausstieg steht in der Sidebar, wo ein neues Gespraech beginnt.
  // Kein Tier-Gate: ein Follow-up ist ein vollwertiger Lauf und zaehlt wie
  // jede andere Frage gegen das Tagesbudget — mehr kostet es niemanden.
  const DEFAULT_INPUT_PLACEHOLDER = "Enter your question";
  const FOLLOWUP_INPUT_PLACEHOLDER = "Ask a follow-up question";
  const UNAVAILABLE_INPUT_PLACEHOLDER = "Start a new comparison — saved context unavailable";

  // Jede archivierte Schublade braucht eine eigene ID fuer aria-controls.
  // Der Verlauf lebt im selben Dokument wie der Live-Renderbaum, dessen IDs
  // ein JS-Vertrag sind — hier darf nie eine davon ein zweites Mal auftauchen.
  let panelSequence = 0;

  const followup = {
    lastExchange: null, // {question, consensus, turn} des letzten Konsens-Laufs
    // True while the current query continues the preceding exchange. The flag
    // is reset by offer(); completed turns may continue indefinitely.
    followupInFlight: false,
    continuationUnavailable: false,
    // Was consume() zuletzt ausgegeben hat. Nur dafuer da, den Kontext
    // zurueckzuholen, wenn der Lauf gar nicht stattgefunden hat.
    spentExchange: null,

    // Ein sichtbares Gespraech laeuft immer weiter: sobald ein fortsetzbarer
    // Turn da ist, geht die naechste Frage mit seinem Kontext raus.
    isArmed() {
      return !!this.lastExchange;
    },

    hasContinuableExchange() {
      return !!this.lastExchange;
    },

    previousQuestionForBookmark() {
      return String(this.spentExchange?.question || "").trim();
    },

    previousTurnForBookmark() {
      const turn = this.spentExchange?.turn;
      return turn && typeof turn === "object" ? turn : null;
    },

    staticizeHistoryNode(node) {
      if (!node) return null;
      const clone = node.cloneNode(true);
      clone.removeAttribute("id");
      clone.hidden = false;
      clone.querySelectorAll("[id]").forEach(child => child.removeAttribute("id"));
      clone.querySelectorAll(
        ".consensus-copy-icon-btn, .copy-btn, .response-code-copy"
      ).forEach(child => child.remove());
      clone.querySelectorAll("button").forEach(button => {
        const replacement = document.createElement("span");
        replacement.className = button.className;
        replacement.innerHTML = button.innerHTML;
        replacement.setAttribute("aria-hidden", "true");
        button.replaceWith(replacement);
      });
      clone.querySelectorAll("[aria-controls]").forEach(child => {
        child.removeAttribute("aria-controls");
        child.removeAttribute("aria-expanded");
      });
      return clone;
    },

    buildStoredAgreement(differencesData) {
      const agreement = differencesData?.agreement;
      if (!agreement || typeof agreement.score !== "number") return null;
      const score = Math.max(0, Math.min(100, Math.round(agreement.score)));
      const verdict = document.createElement("div");
      verdict.className = "consensus-verdict thread-history-verdict "
        + (score >= 65 ? "is-calm" : score >= 40 ? "is-warn" : "is-alert");

      const gauge = document.createElement("span");
      gauge.className = "verdict-gauge";
      gauge.style.setProperty("--val", String(score));
      const scoreLine = document.createElement("span");
      scoreLine.className = "verdict-score";
      const num = document.createElement("span");
      num.className = "verdict-score-num";
      num.textContent = String(score);
      const unit = document.createElement("span");
      unit.className = "verdict-score-unit";
      unit.textContent = "/100";
      scoreLine.append(num, unit);
      const meter = document.createElement("span");
      meter.className = "verdict-meter";
      const fill = document.createElement("span");
      fill.className = "verdict-meter-fill";
      meter.appendChild(fill);
      gauge.append(scoreLine, meter);

      const main = document.createElement("span");
      main.className = "verdict-main";
      const headline = document.createElement("span");
      headline.className = "verdict-headline";
      headline.textContent = score >= 85 ? "High agreement"
        : score >= 65 ? "Strong agreement"
          : score >= 40 ? "Partial agreement"
            : score >= 20 ? "Low agreement" : "Very low agreement";
      const detail = document.createElement("span");
      detail.className = "verdict-detail";
      const modelCount = Number(agreement.model_count) || 0;
      detail.textContent = modelCount
        ? `${modelCount} model${modelCount === 1 ? "" : "s"} compared`
        : "Saved agreement score";
      main.append(headline, detail);
      verdict.append(gauge, main);
      return verdict;
    },

    appendHistoryTurn(turnData, liveBody = null, liveVerdict = null) {
      const history = document.getElementById("threadHistory");
      if (!history || !turnData?.question || !turnData?.consensus) return false;
      const turnId = String(turnData.turn_id || "").trim();
      const normalizedQuestion = String(turnData.question).replace(/\s+/g, " ").trim();
      // A replay of the same completed turn is idempotent. A colliding ID with
      // different content must never make the visible exchange disappear:
      // append it and let the owner-bound transcript remain the authority.
      if (turnId && Array.from(history.children).some(node => (
        node.dataset?.turnId === turnId
        && node.querySelector?.(".thread-history-question-text")?.textContent === normalizedQuestion
      ))) {
        return false;
      }

      const turn = document.createElement("article");
      turn.className = "thread-history-turn";
      if (turnId) turn.dataset.turnId = turnId;

      const question = document.createElement("div");
      question.className = "thread-history-question";
      const questionLabel = document.createElement("div");
      questionLabel.className = "thread-ask-label";
      questionLabel.textContent = "Question";
      const questionText = document.createElement("div");
      questionText.className = "thread-history-question-text";
      questionText.textContent = normalizedQuestion;
      question.append(questionLabel, questionText);

      // Anhaenge bleiben an ihrer Nachricht, auch wenn der Turn in den
      // Verlauf rutscht.
      const attachmentsMeta = Array.isArray(turnData.attachments) ? turnData.attachments : [];
      if (attachmentsMeta.length) {
        const attachmentRow = document.createElement("div");
        attachmentRow.className = "attachment-bar message-attachments";
        const rendered = window.App.attachments?.renderMessageAttachments?.(
          attachmentRow,
          attachmentsMeta
        );
        if (rendered) question.appendChild(attachmentRow);
      }

      // Eine archivierte Frage klappt wie die aktive auf drei Zeilen ein
      // (#threadAskMore in app-core.js schaltet beide). Ohne das hat ab dem
      // zweiten Turn jede lange Frage den Thread wieder aufgerissen.
      const questionMore = document.createElement("button");
      questionMore.type = "button";
      questionMore.className = "thread-ask-more";
      questionMore.textContent = "Show full question";
      question.appendChild(questionMore);

      const answer = document.createElement("div");
      answer.className = "thread-history-answer";
      const answerLabel = document.createElement("div");
      answerLabel.className = "thread-history-answer-label";
      answerLabel.textContent = "Consensus Answer";
      const turnSources = Array.isArray(turnData.sources) ? turnData.sources : [];
      const answerBody = document.createElement("div");
      answerBody.className = "consensus-answer-body";
      if (typeof window.injectMarkdown === "function") {
        window.injectMarkdown(answerBody, turnData.consensus, turnSources);
      } else {
        answerBody.textContent = turnData.consensus;
      }
      answerBody.classList.add("thread-history-answer-body");
      const claimsFallback = document.createElement("div");
      claimsFallback.className = "consensus-claims-fallback thread-history-claims-fallback";
      claimsFallback.hidden = true;
      window.renderStoredConsensusClaims?.(
        answerBody,
        turnData.differences_data,
        claimsFallback,
        turnSources
      );
      const verdict = this.staticizeHistoryNode(liveVerdict)
        || this.buildStoredAgreement(turnData.differences_data);
      if (verdict) verdict.classList.add("thread-history-verdict");
      answer.append(answerLabel, answerBody, claimsFallback);

      // Der Fuss eines archivierten Turns spricht dieselbe Sprache wie der
      // Fuss der aktiven Antwort: EINE Zeile leiser Schubladen nebeneinander
      // (#runProvenance / .consensus-footer-tabs), nicht drei untereinander
      // gestapelte <details>. Gestapelt hat jeder alte Turn den Thread um
      // drei weitere Zeilen verlaengert, obwohl alles zugeklappt war.
      const footer = document.createElement("div");
      footer.className = "thread-history-footer";
      if (verdict) footer.appendChild(verdict);
      const tabs = document.createElement("div");
      tabs.className = "consensus-footer-tabs thread-history-tabs";
      const panels = document.createElement("div");
      panels.className = "thread-history-panels";
      footer.append(tabs, panels);
      answer.appendChild(footer);

      // Chip + Schublade als Paar: gleiche Klassen wie im Live-Fuss, damit
      // shell.css beide gleich behandelt. Die IDs sind je Turn eindeutig —
      // der Verlauf teilt sich den DOM mit dem Live-Renderbaum.
      function addDrawer(label, shortLabel, count, fill) {
        const panelId = `threadHistoryPanel-${++panelSequence}`;
        const tab = document.createElement("button");
        tab.type = "button";
        tab.className = "consensus-tab";
        tab.setAttribute("aria-expanded", "false");
        tab.setAttribute("aria-controls", panelId);
        const chevron = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        chevron.setAttribute("class", "consensus-tab-chevron");
        chevron.setAttribute("viewBox", "0 0 12 12");
        chevron.setAttribute("aria-hidden", "true");
        chevron.setAttribute("fill", "none");
        chevron.setAttribute("stroke", "currentColor");
        chevron.setAttribute("stroke-width", "1.6");
        chevron.setAttribute("stroke-linecap", "round");
        chevron.setAttribute("stroke-linejoin", "round");
        const chevronPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
        chevronPath.setAttribute("d", "M2.5 4.5 6 8l3.5-3.5");
        chevron.appendChild(chevronPath);
        const tabLabel = document.createElement("span");
        tabLabel.className = "consensus-tab-label";
        tabLabel.dataset.short = shortLabel;
        tabLabel.textContent = label;
        const tabCount = document.createElement("span");
        tabCount.className = "consensus-tab-count";
        tabCount.textContent = count > 0 ? String(count) : "";
        tab.append(chevron, tabLabel, tabCount);

        const panel = document.createElement("div");
        panel.className = "thread-history-panel";
        panel.id = panelId;
        panel.hidden = true;
        fill(panel);

        tab.addEventListener("click", () => {
          const open = tab.getAttribute("aria-expanded") === "true";
          tab.setAttribute("aria-expanded", String(!open));
          panel.hidden = open;
        });

        tabs.appendChild(tab);
        panels.appendChild(panel);
      }

      const differencesText = stripBestModelLine(turnData.differences);
      const storedDifferences = turnData.differences_data?.differences;
      const hasStructuredDifferences = Array.isArray(storedDifferences);
      if (hasStructuredDifferences || differencesText) {
        addDrawer(
          "Review differences",
          "Differences",
          hasStructuredDifferences ? storedDifferences.length : 0,
          panel => {
            // Der Turn traegt dieselben strukturierten Daten wie der Live-Lauf.
            // Sie hier NICHT zu benutzen hiess: derselbe Befund las sich im
            // Verlauf als roher Judge-Text, sobald die Antwort nach oben rutschte.
            const cards = document.createElement("div");
            cards.className = "differences-cards thread-history-differences";
            const rendered = window.renderStoredDifferenceCards?.(
              cards,
              turnData.differences_data,
              { modelLabel: storedModelLabeller(turnData.model_answers) }
            );
            if (rendered) {
              panel.appendChild(cards);
              return;
            }
            const body = document.createElement("div");
            body.className = "thread-history-detail-body";
            const markup = window.colorizeCredibility
              ? window.colorizeCredibility(differencesText)
              : differencesText;
            if (typeof window.injectMarkdown === "function") {
              window.injectMarkdown(body, markup, turnSources);
            } else {
              body.textContent = differencesText;
            }
            panel.appendChild(body);
          }
        );
      }

      if (turnSources.length) {
        const list = document.createElement("ol");
        list.className = "thread-history-sources";
        turnSources.forEach((source, index) => {
          const item = document.createElement("li");
          const rawUrl = String(source?.url || "");
          let safeUrl = "";
          try {
            const parsed = new URL(rawUrl);
            if (["http:", "https:"].includes(parsed.protocol)) safeUrl = parsed.href;
          } catch (_) {}
          const title = String(source?.title || rawUrl || `Source ${index + 1}`);
          const label = safeUrl ? document.createElement("a") : document.createElement("span");
          label.textContent = title;
          if (safeUrl) {
            label.href = safeUrl;
            label.target = "_blank";
            label.rel = "noopener noreferrer";
          }
          item.appendChild(label);
          list.appendChild(item);
        });
        addDrawer("Verify sources", "Sources", turnSources.length, panel => {
          panel.appendChild(list);
        });
      }

      const storedAnswers = turnData.model_answers && typeof turnData.model_answers === "object"
        ? Object.entries(turnData.model_answers).map(([provider, item]) => ({ provider, item }))
        : [];
      const usableAnswers = storedAnswers.filter(({ item }) => String(item?.answer || "").trim());
      if (usableAnswers.length) {
        addDrawer("Compare answers", "Answers", usableAnswers.length, panel => {
          const models = document.createElement("div");
          models.className = "thread-history-models";
          usableAnswers.forEach(({ provider, item }) => {
            const section = document.createElement("section");
            section.dataset.provider = String(item.provider || provider);
            const heading = document.createElement("h4");
            heading.textContent = String(item.model_label || item.provider || "Model");
            const body = document.createElement("div");
            body.className = "thread-history-detail-body";
            const sources = Array.isArray(item.sources) ? item.sources : turnSources;
            if (typeof window.injectMarkdown === "function") {
              window.injectMarkdown(body, item.answer, sources);
            } else {
              body.textContent = item.answer;
            }
            section.append(heading, body);
            models.appendChild(section);
          });
          panel.appendChild(models);
        });
      }
      turn.append(question, answer);
      history.appendChild(turn);
      history.hidden = false;
      // Erst im DOM laesst sich messen, ob der Clamp ueberhaupt greift; nur
      // dann bekommt der Turn seinen Aufklapp-Link.
      requestAnimationFrame(() => {
        question.classList.toggle(
          "is-long",
          questionText.scrollHeight > questionText.clientHeight + 2
        );
      });
      return true;
    },

    // Der Live-Consensus wird fuer den naechsten Lauf wiederverwendet. Bevor
    // das passiert, frieren wir die sichtbare Frage und Antwort als statischen
    // Turn ein. Interaktive Marker werden dabei zu reinen Anzeigeelementen;
    // doppelte IDs oder tote Buttons duerfen nicht in den Live-DOM gelangen.
    archiveCurrentExchange() {
      const liveBody = window.App.consensusBodyEl?.();
      const exchange = this.spentExchange || this.lastExchange;
      if (!liveBody || !exchange?.question || !liveBody.textContent?.trim()) {
        return false;
      }
      const turnData = Object.assign(
        {},
        exchange.turn || { question: exchange.question, consensus: exchange.consensus }
      );
      // Die sichtbaren Anhaenge gehoeren zu genau diesem Turn und wandern mit
      // ihm in den Verlauf. Der gespeicherte Turn kennt sie inzwischen selbst;
      // der sichtbare Stand bleibt die Rueckfalllinie, wenn die Turn-Anlage
      // fuer diesen Lauf nicht zustande gekommen ist.
      if (!Array.isArray(turnData.attachments) || !turnData.attachments.length) {
        turnData.attachments = window.App.getThreadAttachments?.() || [];
      }
      return this.appendHistoryTurn(
        turnData,
        liveBody,
        document.getElementById("consensusVerdict")
      );
    },

    renderStoredTurn(turnData) {
      this.clearHistory();
      return this.appendHistoryTurn(turnData);
    },

    renderStoredTurns(turns) {
      this.clearHistory();
      let rendered = 0;
      (Array.isArray(turns) ? turns : []).forEach(turn => {
        if (this.appendHistoryTurn(turn)) rendered += 1;
      });
      return rendered;
    },

    clearHistory() {
      const history = document.getElementById("threadHistory");
      if (!history) return;
      history.replaceChildren();
      history.hidden = true;
    },

    offer(question, consensusText, turn = null) {
      if (!question || !consensusText) return;
      this.continuationUnavailable = false;
      this.followupInFlight = false;
      this.lastExchange = {
        question: question,
        consensus: consensusText,
        turn: turn && typeof turn === "object" ? turn : null
      };
      // Ein Lauf ist durchgelaufen: nichts mehr zurueckzuholen.
      this.spentExchange = null;
      this.render();
    },

    // Ein Lauf, der gar nicht stattgefunden hat (Kontingent leer), darf den
    // Gespraechsfaden nicht gefressen haben: consume() ist beim Absenden
    // passiert, gesendet wurde aber nichts. Sonst waere der Kontext weg,
    // waehrend die Absage-Karte sagt "nichts wurde gesendet" — und die
    // naechste Frage ginge stillschweigend ohne Kontext raus.
    restoreAfterBlockedRun() {
      if (!this.spentExchange) return;
      this.lastExchange = this.spentExchange;
      this.spentExchange = null;
      this.followupInFlight = false;
      this.render();
    },

    // Neuer Lauf ohne Kontext bzw. Clear: der Faden ist abgeschnitten.
    // Loescht auch das In-Flight-Flag (frische Frage darf wieder anbieten).
    reset() {
      this.lastExchange = null;
      this.followupInFlight = false;
      this.spentExchange = null;
      this.continuationUnavailable = false;
      this.render();
    },

    markContinuationUnavailable() {
      this.lastExchange = null;
      this.followupInFlight = false;
      this.spentExchange = null;
      this.continuationUnavailable = true;
      this.render();
    },

    // The returned one-hop payload remains for legacy bookmark restores. An
    // owned active chat uses only its server-issued context-version binding.
    consume() {
      if (!this.lastExchange) return null;
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

    // Der Kontext-Zustand hat KEINE eigene Flaeche mehr am Composer. Sichtbar
    // ist er dort, wo er hingehoert: im Thread, der ueber dem Eingabefeld
    // steht. Uebrig bleibt das Nachziehen der Login-Schranke und des
    // Platzhalters.
    render() {
      this.syncInputLock();
    },

    // Der Composer bleibt nach einer Antwort offen; hier wird nur noch die
    // Login-Schranke (updateQuestionInputAccess) nachgezogen und das
    // Platzhalter-Wording an den Kontext-Zustand angepasst — updateQuestion-
    // InputAccess laeuft nach jedem Auth-Update und wuerde den Follow-up-
    // Platzhalter sonst wieder ueberschreiben.
    syncInputLock() {
      // Tippen und Absenden sind zwei Rechte: wer auf die E-Mail-Bestaetigung
      // wartet, darf seine Frage schon schreiben.
      const canAsk = typeof window.userCanAskQuestions === "function"
        ? window.userCanAskQuestions()
        : true;
      const canType = typeof window.userCanTypeQuestions === "function"
        ? window.userCanTypeQuestions()
        : canAsk;
      const input = document.getElementById("questionInput");
      if (!input) return;
      input.disabled = !canType;
      input.setAttribute("aria-disabled", !canType ? "true" : "false");
      if (!canAsk) return;
      input.placeholder = this.isArmed()
        ? FOLLOWUP_INPUT_PLACEHOLDER
        : (this.continuationUnavailable
            ? UNAVAILABLE_INPUT_PLACEHOLDER
            : DEFAULT_INPUT_PLACEHOLDER);
    }
  };
  window.App.followup = followup;

  // ------------------------------------------------------------------
  // RunContext consensus path. Unlike the legacy compatibility function
  // below, this path never reads response boxes, current bookmark globals or
  // the visible chat session. Query-send passes the exact owning context.
  function contextConsensusRenderer(context, target) {
    const RENDER_INTERVAL = 120;
    let timer = null;
    let lastRender = 0;
    function render() {
      timer = null;
      lastRender = Date.now();
      if (window.App.runRegistry.isVisible(context.runId)) {
        window.App.runRegistry.renderVisible();
      }
    }
    function schedule() {
      if (!window.App.runRegistry.isVisible(context.runId)) return;
      const elapsed = Date.now() - lastRender;
      if (elapsed >= RENDER_INTERVAL) render();
      else if (!timer) timer = window.setTimeout(render, RENDER_INTERVAL - elapsed);
    }
    return {
      append(chunk) {
        if (!window.App.runRegistry.isExecuting(context.runId)) return;
        const text = String(chunk || "");
        if (!text) return;
        if (target === "consensus") {
          context.consensus.status = "streaming";
          context.consensus.streamText += text;
        } else if (target === "consensus-final") {
          context.consensus.text = text;
          context.consensus.streamText = text;
          context.consensus.status = "differences";
          context.phase = "differences";
        } else {
          context.consensus.status = "differences";
          context.phase = "differences";
        }
        schedule();
      },
      markReasoning() {
        if (!window.App.runRegistry.isExecuting(context.runId)) return;
        if (target === "differences") {
          context.consensus.status = "differences";
          context.phase = "differences";
        }
        schedule();
      },
      stop() {
        if (timer) window.clearTimeout(timer);
        timer = null;
      }
    };
  }

  function runAnswer(context, provider) {
    const result = context.modelResults?.[provider];
    return result?.status === "complete" ? String(result.text || "").trim() : "";
  }

  function runSources(context, provider) {
    const sources = context.modelResults?.[provider]?.sources;
    return Array.isArray(sources) ? sources.map(source => ({ ...source })) : [];
  }

  function runModelAnswers(context) {
    return Object.fromEntries(Object.entries(context.modelResults || {})
      .filter(([, result]) => result?.status === "complete" && String(result.text || "").trim())
      .map(([provider, result]) => [provider, {
        provider,
        model_label: result.modelLabel || provider,
        answer: result.text,
        sources: runSources(context, provider)
      }]));
  }

  function updateContextUsage(context, data) {
    const detail = data?.detail && typeof data.detail === "object" ? data.detail : {};
    const status = data?.usage_run_status || detail.usage_run_status;
    if (status && context.usage) context.usage.status = status;
    if (!window.App.runRegistry.isAuthCurrent(context)) return;
    const usageView = window.App.runRegistry.reconcileUsageSnapshot?.(context, {
      remaining: data?.free_usage_remaining ?? detail.free_usage_remaining,
      deepRemaining: data?.deep_remaining ?? detail.deep_remaining,
      totalLimit: data?.limit ?? detail.limit ?? window.currentMaxLimit,
      deepLimit: data?.deep_limit ?? detail.deep_limit ?? window.currentDeepLimit
    }) || {
      remaining: data?.free_usage_remaining ?? detail.free_usage_remaining,
      deepRemaining: data?.deep_remaining ?? detail.deep_remaining,
      totalLimit: data?.limit ?? detail.limit ?? window.currentMaxLimit,
      deepLimit: data?.deep_limit ?? detail.deep_limit ?? window.currentDeepLimit
    };
    window.App.renderUsageDisplay?.(usageView);
  }

  function contextCitationMeta(context) {
    let url = window.location.href;
    try {
      const parsed = new URL(window.location.href);
      url = parsed.origin + parsed.pathname;
    } catch (_) {}
    return {
      question: context.question,
      includedModels: (context.config.providers || [])
        .filter(provider => runAnswer(context, provider.provider))
        .map(provider => `${provider.provider}: ${provider.modelLabel || provider.modelId}`),
      consensusModel: context.config.consensusModelLabel || context.config.consensusModel,
      dateISO: new Date().toISOString(),
      url
    };
  }

  function consensusErrorMessage(result, data) {
    const detail = data?.detail;
    if (detail && typeof detail === "object") {
      return String(detail.error || detail.message || `Consensus HTTP ${result.status}`);
    }
    return String(data?.error || detail || `Consensus HTTP ${result.status}`);
  }

  window.App.executeConsensusRun = async function (context, options = {}) {
    const registry = window.App.runRegistry;
    if (!context || !registry.isExecuting(context.runId)) return null;
    if (["pending", "streaming", "differences"].includes(context.consensus.status)) return null;

    const trigger = String(options.trigger || "auto");
    let dispositionOnly = options.dispositionOnly === true;
    const controller = new AbortController();
    context.controllers.consensus = controller;
    context.phase = "consensus";
    context.consensus.status = "pending";
    context.consensus.error = null;
    registry.update(context.runId, () => {});

    const successfulAnswers = Object.values(context.modelResults || {})
      .filter(result => result?.status === "complete" && String(result.text || "").trim()).length;
    if (!dispositionOnly && successfulAnswers < 2 && context.chatSession?.pendingTurnId) {
      dispositionOnly = true;
    }
    if (!dispositionOnly && successfulAnswers < 2) {
      context.consensus.status = "error";
      context.consensus.error = { message: "At least two completed model answers are required." };
      context.credentials = null;
      context.attachments = [];
      registry.setStatus(context.runId, "failed", context.consensus.error);
      return null;
    }

    trackAppEvent("app_consensus_started", {
      trigger,
      included_models: successfulAnswers,
      excluded_models: Math.max(0, 6 - (context.config.providers || []).length),
      custom_credentials: context.config.useOwnKeys,
      logged_in: true
    });

    try {
      let idToken = null;
      try { idToken = await context.auth.user?.getIdToken?.(); } catch (_) {}
      if (!idToken || !registry.isAuthCurrent(context) || !registry.isExecuting(context.runId)) {
        throw new Error("Authentication changed while generating the consensus.");
      }

      const chatTurnIds = await context.chatSession?.ensurePendingTurn?.({
        idToken,
        question: context.question,
        consensusModel: context.config.consensusModel,
        signal: controller.signal
      }) || null;
      if (!registry.isExecuting(context.runId)) return null;

      const modelLabels = Object.fromEntries((context.config.providers || []).map(provider => [
        provider.provider,
        provider.modelLabel || provider.modelId || provider.provider
      ]));
      const modelSources = Object.fromEntries((context.config.providers || []).map(provider => [
        provider.provider,
        runSources(context, provider.provider)
      ]));
      const payload = {
        id_token: idToken,
        useOwnKeys: context.config.useOwnKeys,
        usage_run_key: context.usage?.key || null,
        deep_search: context.config.deepSearch,
        question: context.question,
        answers: Object.fromEntries(
          familyKeys().map(provider => [provider, runAnswer(context, provider)])
        ),
        model_sources: modelSources,
        model_labels: modelLabels,
        consensus_model: context.config.consensusModel,
        bookmarkId: context.bookmark.id,
        previousQuestion: context.previousExchange?.question || "",
        previousTurn: context.previousExchange?.turn || null,
        excluded_models: familyKeys()
          .filter(provider => !context.config.providers.some(item => item.provider === provider)),
        openrouter_key: context.credentials?.openrouterKey || "",
        keepalive: true
      };
      if (chatTurnIds) {
        payload.chat_id = chatTurnIds.chatId;
        payload.turn_id = chatTurnIds.turnId;
        if (chatTurnIds.contextVersionId) payload.context_version_id = chatTurnIds.contextVersionId;
        payload.turn_sources = context.evidenceSources.map(source => ({ ...source }));
      }

      const requestResult = await streamSSERequest("/consensus", payload, controller.signal, {
        "consensus.delta": contextConsensusRenderer(context, "consensus"),
        "consensus.final": contextConsensusRenderer(context, "consensus-final"),
        "differences.delta": contextConsensusRenderer(context, "differences")
      });
      const data = requestResult.data || {};
      if (!data.consensus_response && context.consensus.text) data.consensus_response = context.consensus.text;
      updateContextUsage(context, data);
      if (!registry.isExecuting(context.runId)) return data;

      const disposition = data?.chat_turn_state
        ? data
        : (data?.detail && typeof data.detail === "object" && data.detail.chat_turn_state ? data.detail : null);
      if (disposition) {
        context.chatSession?.handleConsensusResult?.({
          chatId: disposition.chat_id,
          turnId: disposition.turn_id,
          chatPersisted: disposition.chat_persisted === true,
          chatTurnState: disposition.chat_turn_state
        });
        context.chatTurnState = disposition.chat_turn_state;
        context.keepConversationLock = disposition.chat_turn_state === "pending";
      } else if (chatTurnIds) {
        context.chatSession?.markPendingUncertain?.();
        context.keepConversationLock = true;
      }

      const failedChatTurn = disposition?.chat_turn_state === "failed";
      const pendingChatTurn = disposition?.chat_turn_state === "pending";
      if (!requestResult.ok || !data.consensus_response || failedChatTurn || pendingChatTurn) {
        const message = failedChatTurn || pendingChatTurn
          ? String(
              disposition?.error
              || disposition?.message
              || (pendingChatTurn
                ? "The consensus was generated, but its conversation turn did not receive a final server status. Reload before continuing."
                : data.consensus_response)
              || "The consensus could not be completed."
            )
          : consensusErrorMessage(requestResult, data);
        context.consensus.status = "error";
        context.consensus.error = { message };
        context.consensus.text = context.consensus.text || context.consensus.streamText;
        context.phase = "failed";
        context.bookmark.status = "failed";
        context.credentials = null;
        context.attachments = [];
        registry.setStatus(context.runId, "failed", context.consensus.error);
        trackAppEvent("app_consensus_completed", { status: "error", trigger, included_models: successfulAnswers });
        return data;
      }

      if (Array.isArray(data.sources)) context.evidenceSources = data.sources.map(source => ({ ...source }));
      context.consensus.status = "complete";
      context.consensus.text = String(data.consensus_response || context.consensus.text || "");
      context.consensus.streamText = context.consensus.text;
      context.consensus.differences = String(data.differences || "");
      context.consensus.differencesData = data.differences_data || null;
      context.consensus.sources = context.evidenceSources.map(source => ({ ...source }));
      context.consensus.resultId = data.result_id || null;
      context.consensus.modelLabels = modelLabels;
      context.consensus.citationMeta = contextCitationMeta(context);

      const modelAnswers = data.model_answers && typeof data.model_answers === "object"
        && Object.keys(data.model_answers).length
        ? data.model_answers
        : runModelAnswers(context);
      const completedTurn = {
        turn_id: data.turn_id || chatTurnIds?.turnId || "",
        question: context.question,
        consensus: context.consensus.text,
        differences: context.consensus.differences,
        differences_data: context.consensus.differencesData,
        sources: context.evidenceSources.map(source => ({ ...source })),
        model_answers: modelAnswers,
        attachments: context.attachmentMeta.map(item => ({ ...item }))
      };
      context.consensus.completedTurn = completedTurn;

      const conversation = {
        runId: context.runId,
        auth: context.auth,
        bookmarkId: context.bookmark.id,
        chatId: data.chat_persisted === true && data.chat_turn_state === "completed"
          ? data.chat_id : null,
        turnId: data.chat_persisted === true && data.chat_turn_state === "completed"
          ? data.turn_id : null,
        sources: context.evidenceSources.map(source => ({ ...source })),
        modelResponses: Object.fromEntries(Object.entries(modelAnswers).map(([provider, item]) => [
          provider,
          typeof item === "string" ? item : String(item?.answer || "")
        ]))
      };
      context.consensus.bookmarkPayload = {
        question: context.question,
        resultId: context.consensus.resultId,
        previousQuestion: context.previousExchange?.question || "",
        previousTurn: context.previousExchange?.turn || null,
        consensusText: context.consensus.text,
        differencesText: context.consensus.differences,
        differencesData: context.consensus.differencesData,
        conversation
      };

      const completedBasis = {
        bookmarkId: context.bookmark.id,
        chatId: conversation.chatId || context.chatSession?.activeChatId || "",
        turnId: conversation.turnId || context.chatSession?.activeTurnId || completedTurn.turn_id,
        question: context.question,
        consensus: context.consensus.text,
        currentTurn: completedTurn,
        historyTurns: context.historyTurns,
        title: context.bookmark.title || context.question
      };
      context.completedBasis = completedBasis;
      context.phase = "done";
      context.bookmark.status = "succeeded";

      let savePromise = null;
      if (registry.isAuthCurrent(context) && data.chat_replayed !== true) {
        if (data.bookmark_persisted === true && data.bookmark_meta) {
          // The successful /consensus final event is now emitted only after
          // the primary server-side bookmark write. Apply its compact metadata
          // locally; no second network roundtrip is needed for the normal path.
          window.acceptPersistedConsensusBookmark?.(data.bookmark_meta, conversation);
          context.persistence.consensusWrite = true;
          savePromise = Promise.resolve(data.bookmark_meta);
        } else {
          // Cached servers and a failed primary write retain the idempotent
          // compatibility endpoint as a bounded, keepalive-enabled retry path.
          savePromise = window.saveBookmarkConsensus?.(
            context.question,
            context.consensus.text,
            context.consensus.differences,
            context.consensus.differencesData,
            context.consensus.resultId,
            context.config.consensusModel,
            modelLabels,
            context.previousExchange?.question || "",
            context.previousExchange?.turn || null,
            conversation
          );
        }
        context.persistence.consensusPromise = savePromise || null;
        savePromise?.catch?.(() => undefined);
      } else if (data.chat_replayed === true) {
        context.persistence.consensusWrite = true;
      }

      context.credentials = null;
      context.attachments = [];
      registry.setStatus(context.runId, "succeeded");
      registry.setCompletedBasis(context.runId, completedBasis);

      if (registry.isAuthCurrent(context) && data.chat_replayed !== true) {
        const best = context.consensus.differencesData?.best_model || parseBestModel(context.consensus.differences);
        if (best) window.recordModelVote?.(best, "BestModel", context.consensus.resultId);
      }
      if (registry.isVisible(context.runId) && data.chat_replayed !== true) {
        window.App.watch?.showFeatureNudge?.();
      }
      trackAppEvent("app_consensus_completed", {
        status: data.error ? "partial" : "success",
        trigger,
        included_models: successfulAnswers
      });
      return data;
    } catch (error) {
      if (isAbortError(error) || !registry.isExecuting(context.runId)) return null;
      context.chatSession?.markPendingUncertain?.();
      if (context.chatSession?.pendingTurnId) context.keepConversationLock = true;
      context.consensus.status = "error";
      context.consensus.error = { message: error?.message || "The consensus request failed." };
      context.consensus.text = context.consensus.text || context.consensus.streamText;
      context.phase = "failed";
      context.bookmark.status = "failed";
      context.credentials = null;
      context.attachments = [];
      registry.setStatus(context.runId, "failed", context.consensus.error);
      window.App.reportCriticalError?.({
        type: "consensus_failed",
        phase: "consensus_connection",
        message: "The consensus request ended without a confirmed result.",
        details: `run ${context.requestIdentity}`
      });
      trackAppEvent("app_consensus_completed", {
        status: context.consensus.text ? "partial" : "error",
        trigger,
        included_models: successfulAnswers
      });
      return null;
    } finally {
      context.controllers.consensus = null;
      if (registry.isVisible(context.runId)) registry.renderVisible();
      window.App.syncSendButtonRunning?.();
    }
  };

  window.getConsensus = async function (trigger = "manual") {
    const boundContext = window.App.runRegistry?.visible?.();
    if (boundContext && window.App.runRegistry.isExecuting(boundContext.runId)) {
      return await window.App.executeConsensusRun(boundContext, {
        trigger,
        dispositionOnly: trigger === "disposition"
      });
    }
    // Production execution is RunContext-only. The implementation below is
    // retained as a compatibility reference for old snapshots, but no user
    // action may start its DOM/singleton lifecycle: it would bypass the
    // registry's max-two admission, targeted logout cleanup and persistence
    // ownership.
    if (window.App.runRegistry) {
      if (trigger === "manual") {
        window.App.showPopup?.("Open a comparison that has answers ready before generating a consensus.");
      }
      return null;
    }
    const replayPendingTurn = trigger === "replay";
    const dispositionOnly = trigger === "disposition";
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
    const replayRun = replayPendingTurn ? window.App.chatSession?.logicalRun : null;
    // A completed reconciliation replays the original logical run even when
    // the user changed controls while the transport disposition was unknown.
    const useOwnKeys = replayRun
      ? replayRun.useOwnKeys === true
      : document.getElementById("useOwnKeysSwitch").checked;
    const deepThink = replayRun
      ? replayRun.deepSearch === true
      : (window.App.usageRun?.current?.deepThink
        ?? (document.getElementById("deepSearchToggle")?.checked === true));
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
      window.App.state.set("lastQuestion", question, "run");
    }

    // Setze den Konsens-Bereich (Spinner etc.) und rufe anschließend deinen Konsens-Endpunkt auf.
    const consensusDiv = document.getElementById("consensusResponse");

    const consensus_model = replayRun?.consensusModel
      || document.getElementById("consensusModelDropdown").value;

    // Die Familien dieses Laufs mit ihrer Antwortbox. Eine Quelle
    // (window.App.modelPrefs): Antworten, Quellen, Ausschluesse und Zitate
    // werden daraus abgeleitet statt sechsmal einzeln aufgezaehlt.
    const families = (window.App.modelPrefs || [])
      .map(pref => ({ ...pref, box: document.getElementById(pref.responseId) }))
      .filter(family => family.box);

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

    const providerBoxes = Object.fromEntries(
      families.map(family => [family.key, family.box])
    );

    // Replay of a stored turn: the providers were never called, so every box
    // still shows the previous run. Repaint the ones the stored turn actually
    // used and blank the rest, so no box can attribute an old answer to this
    // question. Returns how many answers were restored.
    function restoreStoredModelAnswers(storedAnswers) {
      const answers = storedAnswers && typeof storedAnswers === "object"
        ? storedAnswers
        : {};
      let restored = 0;
      Object.entries(providerBoxes).forEach(([provider, box]) => {
        if (!box) return;
        const outputEl = box.querySelector(".collapsible-content");
        if (!outputEl) return;
        const stored = answers[provider];
        const answer = typeof stored === "string"
          ? stored
          : String(stored?.answer || "");
        delete box.dataset.consensusAnswer;
        delete box.dataset.consensusSources;
        delete box.dataset.responseError;
        delete box.dataset.responseSkipped;
        if (!answer.trim()) {
          box.dataset.responseState = "idle";
          outputEl.innerHTML = "";
          return;
        }
        box.dataset.responseState = "complete";
        const sources = Array.isArray(stored?.sources) ? stored.sources : [];
        if (typeof window.renderModelResponseWithSources === "function") {
          window.renderModelResponseWithSources(outputEl, answer, sources);
        } else {
          window.injectMarkdown?.(outputEl, answer, sources);
        }
        restored += 1;
      });
      window.updateAgentModeUI?.();
      return restored;
    }

    // Abgewählte Modelle werden bewusst als leer gesendet.
    const answers = Object.fromEntries(
      families.map(family => [family.key, getIncludedAnswer(family.box)])
    );
    const model_sources = Object.fromEntries(
      families.map(family => [family.key, getIncludedSources(family.box)])
    );

    // Überprüfe nur die Modelle, die nicht als "ausgeschlossen" markiert sind
    // UND nicht mit einem Fehler zurückkamen. Ein einzelner ausgefallener
    // Provider (z. B. Gemini 503) darf den Konsens nicht blockieren, solange
    // genug andere Antworten vorliegen – er wird einfach ausgelassen.
    function isAnswerableBox(box) {
      return isIncludedBox(box) && box.dataset.responseError !== "true";
    }
    // Eine erwartete, aber noch fehlende Antwort blockiert den Lauf; eine
    // fehlgeschlagene nicht (isAnswerableBox schliesst Fehler aus).
    const awaitedFamilies = families.filter(family => isAnswerableBox(family.box));
    const missingAnswer = awaitedFamilies.some(family => !answers[family.key]);
    const includedAnswerCount = Object.values(answers).filter(Boolean).length;
    const excludedModelCount = families
      .filter(family => family.box.classList.contains("excluded")).length;

    if (
      !question ||
      !consensus_model ||
      (!(replayPendingTurn || dispositionOnly) && (
        includedAnswerCount < 2 || missingAnswer
      ))
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
    const excludedModels = families
      .filter(family => family.box.classList.contains("excluded"))
      .map(family => family.box.getAttribute("data-model"));

    // Hole API Keys aus localStorage
    const openrouterKey = localStorage.getItem("openrouterKey") || "";

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

    families.forEach(family => {
      addModelForCitation(family.responseId, family.selectId, family.citationLabel);
    });

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

    window.App.state.set("consensusCitationMeta", {
      question,
      includedModels: includedModelsDetailed,
      consensusModel: consensusModelLabel || consensusModelValue,
      dateISO: new Date().toISOString(),
      url: cleanUrl
    }, "consensus");

    // Share-Feature: result_id des letzten Laufs zurücksetzen; Modell-
    // Labels (Option-Text) für die serverseitige Snapshot-Zitation.
    window.clearPreparedBookmarkShareResult?.();
    window.App.state.set("lastShareResultId", null, "share");
    // Resolve-Persistenz: Payload des letzten erfolgreichen Laufs invalidieren,
    // damit eine Resolve-Runde nie in ein fremdes Bookmark schreibt.
    window.lastConsensusBookmarkPayload = null;
    const shareModelLabels = {};
    let streamedConsensusText = "";
    let completedConsensusText = "";
    families.forEach(({ key: provider, selectId }) => {
      const select = document.getElementById(selectId);
      if (!select) return;
      const opt = select.options[select.selectedIndex];
      // Reines Modell-Label (data-model-label) ohne Badge-Suffix wie "· New".
      const label = (opt?.dataset.modelLabel || opt?.text || select.value || "").trim();
      if (label) shareModelLabels[provider] = label;
    });

    try {
      const chatTurnIds = await window.App.chatSession?.ensurePendingTurn?.({
        idToken: id_token,
        question,
        consensusModel: consensus_model,
        signal: consensusSignal
      }) || null;
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
      const bookmarkPreviousQuestion = followup.previousQuestionForBookmark();
      const bookmarkPreviousTurn = followup.previousTurnForBookmark();
      const consensusPayload = {
          id_token: id_token,
          useOwnKeys: useOwnKeys,
          usage_run_key: usageRun?.key || null,
          deep_search: deepThink,
          question: question,
          answers: answers,
          model_sources: model_sources,
          model_labels: shareModelLabels,
          consensus_model: consensus_model,
          bookmarkId: window.App.bookmarkSession?.currentId?.() || "",
          previousQuestion: bookmarkPreviousQuestion,
          previousTurn: bookmarkPreviousTurn,
          excluded_models: excludedModels,
          openrouter_key: openrouterKey,
          keepalive: true
        };
      if (chatTurnIds) {
        consensusPayload.chat_id = chatTurnIds.chatId;
        consensusPayload.turn_id = chatTurnIds.turnId;
        if (chatTurnIds.contextVersionId) {
          consensusPayload.context_version_id = chatTurnIds.contextVersionId;
        }
        consensusPayload.turn_sources = Array.isArray(window.currentEvidenceSources)
          ? window.currentEvidenceSources
          : [];
      }
      const consensusRequestResult = await streamSSERequest("/consensus", consensusPayload, consensusSignal, {
          "consensus.delta": consensusMainRenderer,
          "consensus.final": consensusFinalPhaseRenderer,
          "differences.delta": differencesPhaseRenderer
        });
      const data = consensusRequestResult.data || {};
      const completedReplay = data.chat_replayed === true;
      if (!data.consensus_response && completedConsensusText) {
        data.consensus_response = completedConsensusText;
      }
      if (Array.isArray(data.sources)) {
        window.App.state.set("currentEvidenceSources", data.sources, "evidence");
        window.renderEvidenceSources?.(data.sources);
      }
      // A replay never ran the providers, so the boxes still hold whatever the
      // PREVIOUS run left there. Restore them from the stored turn instead —
      // otherwise the visible model answers belong to a different question.
      const replayedAnswerCount = completedReplay
        ? restoreStoredModelAnswers(data.model_answers)
        : 0;
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

      const chatDisposition = data?.chat_turn_state
        ? data
        : (consensusErrorDetail?.chat_turn_state ? consensusErrorDetail : null);
      if (chatDisposition) {
        window.App.chatSession?.handleConsensusResult?.({
          chatId: chatDisposition.chat_id,
          turnId: chatDisposition.turn_id,
          chatPersisted: chatDisposition.chat_persisted === true,
          chatTurnState: chatDisposition.chat_turn_state
        });
        if (chatDisposition.chat_turn_state === "failed") {
          followup.restoreAfterBlockedRun();
        }
      } else if (chatTurnIds) {
        window.App.chatSession?.markPendingUncertain?.();
      }

      if (consensusRequestResult.ok && data.consensus_response) {
        // Share-Feature: nur mit result_id aus dem Final-Event ist
        // Teilen möglich (serverseitiger Snapshot vorhanden).
        window.App.state.set("lastShareResultId", data.result_id || null, "share");

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
            // On a replay the DOM count describes the previous run; the stored
            // turn is the authority for how many models this answer rests on.
            structuredRendered = window.renderConsensusInsights
              ? window.renderConsensusInsights(
                  data.differences_data,
                  completedReplay ? replayedAnswerCount : includedAnswerCount
                )
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

        const completedTurn = {
          turn_id: data.turn_id || chatTurnIds?.turnId || "",
          question,
          consensus: data.consensus_response,
          differences: data.differences || "",
          differences_data: data.differences_data || null,
          sources: Array.isArray(window.currentEvidenceSources)
            ? window.currentEvidenceSources
            : [],
          model_answers: data.model_answers && typeof data.model_answers === "object"
            && Object.keys(data.model_answers).length
            ? data.model_answers
            : Object.fromEntries(Object.entries(answers)
            .filter(([, answer]) => String(answer || "").trim()).map(([provider, answer]) => [
            provider,
            {
              provider,
              model_label: shareModelLabels[provider] || provider,
              answer,
              sources: Array.isArray(model_sources[provider]) ? model_sources[provider] : []
            }
          ]))
        };
        const bookmarkConversation = data.chat_persisted === true
          && data.chat_turn_state === "completed"
          && data.chat_id && data.turn_id
          ? {
              bookmarkId: window.App.bookmarkSession?.currentId?.() || "",
              chatId: data.chat_id,
              turnId: data.turn_id,
              modelResponses: Object.fromEntries(
                Object.entries(completedTurn.model_answers || {}).map(([provider, item]) => [
                  provider,
                  typeof item === "string" ? item : String(item?.answer || "")
                ])
              )
            }
          : {
              bookmarkId: window.App.bookmarkSession?.currentId?.() || "",
              modelResponses: Object.fromEntries(
                Object.entries(completedTurn.model_answers || {}).map(([provider, item]) => [
                  provider,
                  typeof item === "string" ? item : String(item?.answer || "")
                ])
              )
            };

        // Follow-up-Affordance im Input-Bereich anbieten — nicht bei
        // Fehlertexten aus dem Consensus-Stream.
        if (data.consensus_response
            && !/^(Consensus error:|Invalid consensus model selected:)/i.test(data.consensus_response.trim())) {
          followup.offer(question, data.consensus_response, completedTurn);
        }

        // Payload merken: eine spätere Resolve-Runde hängt ihr Ergebnis an
        // differences_data und speichert das Bookmark damit erneut.
        window.lastConsensusBookmarkPayload = {
          question: question,
          resultId: data.result_id || null,
          previousQuestion: bookmarkPreviousQuestion,
          previousTurn: bookmarkPreviousTurn,
          consensusText: data.consensus_response,
          differencesText: data.differences,
          differencesData: data.differences_data || null,
          conversation: bookmarkConversation
        };
        if (!completedReplay && window.auth?.currentUser) {
          if (data.bookmark_persisted === true && data.bookmark_meta) {
            window.acceptPersistedConsensusBookmark?.(data.bookmark_meta, bookmarkConversation);
          } else {
            // Auch ein Follow-up schickt die result_id seines eigenen Laufs mit.
            // Sonst war der Chat-Turn der EINZIGE Beleg fuer "dieser Lauf gehoert
            // dir" -- und jeder Lauf ohne persistierten Turn endete mit sichtbarer
            // Antwort und der Meldung, das Bookmark liesse sich nicht speichern.
            // Der Server setzt share_result_id bei einem Follow-up ohnehin leer.
            window.saveBookmarkConsensus(
              question, data.consensus_response, data.differences, data.differences_data,
              data.result_id || null,
              consensus_model, shareModelLabels, bookmarkPreviousQuestion,
              bookmarkPreviousTurn, bookmarkConversation
            );
          }
        }
        if (!completedReplay) {
          trackAppEvent("app_consensus_completed", {
            status: data.error ? "partial" : "success",
            trigger,
            included_models: includedAnswerCount
          });
          window.App.watch?.showFeatureNudge?.();
        }

        const bestModelFromConsensus =
          (data.differences_data && data.differences_data.best_model) ||
          parseBestModel(data.differences);
        if (!completedReplay && bestModelFromConsensus) {
          window.recordModelVote(bestModelFromConsensus, "BestModel", data.result_id);
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
        if (isAbortError(error)) {
          window.App.chatSession?.handleConsensusCancelled?.();
        }
        return;
      }
      window.App.chatSession?.markPendingUncertain?.();
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
