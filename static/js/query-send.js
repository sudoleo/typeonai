// =====================================================================
// query-send.js
// Query send only: window.sendQuestion fans the question out to the
// selected providers (/prepare + /ask_*), streams each answer and updates
// usage/tier UI. Only Agent Mode triggers auto-consensus; with it disabled,
// the six streamed answers are the complete result. Plus the query run-state
// (controller/runId/running), cancel, and the small query helpers
// (mode detection, demo/search-intent, spinner/cancel bookkeeping).
//
// Consensus request lives in consensus-run.js; this only kicks it off via
// window.getConsensus("auto"). Lifecycle gate via window.App.consensusLifecycle.
//
// Shared deps via existing window contracts:
//   - window.App.consensusLifecycle / trackAppEvent / getSelectedModelCount
//   - window.isAgentModeEnabled / setAgentModeStatus
//   - window.createStreamRenderer / streamSSERequest
//   - window.validateInputText (kept inline, shared with the send listeners)
//   - window.getConsensus (consensus-run.js), window.runDemoFlow/setSpinnerEl (demo.js)
//   - window.saveBookmark (firebase.js), window.auth (firebase.js)
//   - window.lastQuestion, window.currentMaxLimit / window.currentDeepLimit
//
// index.html keeps the send-button/Enter wiring and asks
// window.isQueryRequestRunning() instead of reading the private flag.
// =====================================================================

(function () {
  window.App = window.App || {};

  const consensusLifecycle = window.App.consensusLifecycle;
  const trackAppEvent = window.App.trackAppEvent;
  const getSelectedModelCount = window.App.getSelectedModelCount;
  const isAgentModeEnabled = window.isAgentModeEnabled;
  const setAgentModeStatus = window.setAgentModeStatus;
  const createStreamRenderer = window.createStreamRenderer;
  const streamSSERequest = window.streamSSERequest;

  // Query-only state. consensusGenerated is written but never read (kept for
  // parity); totalRequiredResponses was an undeclared implicit global that
  // nothing reads back, declared here to keep it module-scoped.
  let consensusGenerated = false;
  let totalRequiredResponses = 0;


    function getActiveMode() {
      const deepSearchActive = document.getElementById("deepSearchToggle").checked;

      // Beispielhafte Logik: Priorisiere Deep Think, falls aktiviert:
      if (deepSearchActive) {
        return "Deep Think";
      }
      return "Standard"; // Default-Wert, falls keine Checkbox aktiviert ist
    }

    function isDemoQuery(q) {
      return (q || "").trim().toLowerCase() === "demo";
    }

    function predictSearchIntent(question) {
      if (!question) return false;
      const q = question.toLowerCase().trim();

      // Helper: Prüft gegen eine Liste von Regex-Patterns
      const check = (patterns) => patterns.some(pattern => pattern.test(q));

      // 1. Wetter / Weather
      // Matcht: wetter, weather, vorhersage, forecast, temp, rain, regnet, sonne, sun, grad, degrees
      const weatherPatterns = [
        /wetter/, /weather/,
        /vorhersage/, /forecast/,
        /temp(eratur|erature)?/, // matcht temp, temperatur, temperature
        /\b(regen|rain|regnet|raining|sonne|sun|wolken|clouds)\b/,
        /\b(grad|degrees?|celsius|fahrenheit)\b/
      ];
      if (check(weatherPatterns)) return true;

      // 2. Finanzen / Finance (Crypto & Stock)
      // Matcht: aktie(n), stock(s), kurs, price, bitcoin, btc, eur, usd, cap, etf
      const financePatterns = [
        /akti[en]/, /stock[s]?/, /share[s]?/, // Aktien
        /\b(kurs|price|wert|value)\b/,         // Preis/Wert (mit Boundary, damit "wert" nicht in "bewerten" matcht)
        /\b(market\s?cap|chart|invest|kaufen|buy|sell|verkaufen)\b/,
        // Crypto specific
        /bitcoin|btc|eth|ethereum|solana|xrp/,
        /krypto|crypto|coin[s]?|token/,
        // Währungen
        /\b(dollar|euro|eur|usd|chf)\b/
      ];
      if (check(financePatterns)) return true;

      // 3. News & Zeitgeschehen / News & Factual
      // Hier nutzen wir flexiblere Matches für Endungen
      const newsPatterns = [
        // Signalwörter für "Neuigkeiten"
        /news/, /nachricht(en)?/, /neuigkeit(en)?/,
        /update/, /ticker/, /schlagzeile[n]?/, /headline[s]?/,

        // Zeitbezug (Wichtig für Search Intent)
        /aktuell[a-z]*/, // Matcht: aktuell, aktuelle, aktuelles, etc.
        /latest/, /recent/, /current/,
        /\b(heute|today|gestern|yesterday|morgen|tomorrow)\b/,
        /\b(jetzt|now|live)\b/,

        // Fakten-Abfragen (Entities)
        /wer (ist|war)/, /who (is|was)/,
        /wie (viele|hoch|alt)/, /how (many|much|old|tall)/,
        /wann/, /when/,

        // Spezifische Domänen-Keywords
        /einwohner|population/,
        /präsident|president/, /kanzler|chancellor/, /ceo/,
        /ergebnis|result/, /spielstand|score/, /tabelle|standing/,
        /gewinner|winner/, /statistik|statistic/
      ];
      if (check(newsPatterns)) return true;

      // 4. Dynamische Jahreszahlen (Vergangenheit & Zukunft)
      // Wir prüfen auf letztes, aktuelles und nächstes Jahr.
      // Regex \b stellt sicher, dass "2024" nicht in "020240" gefunden wird.
      const currentYear = new Date().getFullYear();
      const yearPattern = new RegExp(`\\b(${currentYear - 1}|${currentYear}|${currentYear + 1})\\b`);

      if (yearPattern.test(q)) return true;

      return false;
    }

    let effectiveSystemPrompt = "";
    let currentQueryController = null;
    let currentQueryRunId = 0;
    let queryRequestRunning = false;

    // ---- Die abgeschickte Nachricht ----------------------------------------
    // Zwischen dem Klick auf Senden und dem Moment, in dem der Lauf den neuen
    // Turn aufmacht, liegen /prepare und — im laufenden Gespraech — das Binden
    // des Chat-Kontexts. Das sind Sekunden, und frueher passierte in ihnen
    // nichts Sichtbares: die Frage stand unveraendert im Feld, der Thread
    // zeigte weiter die vorige. Gleichzeitig darf der fertige Vorgaenger noch
    // nicht abgeraeumt sein, solange der Lauf am Kontingent scheitern kann.
    // Deshalb drei Schritte statt einem:
    //   hold()    - das Feld ist sofort leer, die Nachricht steht als eigene
    //               Blase unter der bisherigen Antwort,
    //   promote() - der Lauf findet statt: sie wird Kopf des neuen Turns,
    //   restore() - der Lauf findet NICHT statt: sie geht unveraendert und
    //               abschickbar ins Feld zurueck.
    const sentMessage = {
      question: "",
      draft: "",
      quote: "",
      attachmentsMeta: [],
      pending: false,
      ownsHead: false,

      // ownsHead: ob der Fragen-Kopf noch uebernommen werden muss. Ein frisches
      // Gespraech setzt ihn direkt (dort steht nichts, das erst archiviert
      // werden muesste); ein Follow-up erst nach dem Archivieren.
      // draft/quote: dieselbe Nachricht in ihren zwei Teilen, so wie der
      // Composer sie zeigt — nur so kann restore() sie wieder auseinanderlegen.
      hold(question, { ownsHead = false, draft = null, quote = "" } = {}) {
        this.question = String(question || "");
        this.draft = draft === null ? this.question : String(draft || "");
        this.quote = String(quote || "");
        this.attachmentsMeta = window.App.attachments?.messageMeta?.() || [];
        this.pending = true;
        this.ownsHead = ownsHead === true;
        if (this.ownsHead) {
          window.App.setPendingThreadQuestion?.(this.question, this.attachmentsMeta);
        }

        const input = document.getElementById("questionInput");
        if (input) {
          input.value = "";
          input.dispatchEvent(new Event("input", { bubbles: true }));
          window.syncDemoChipState?.();
        }
        // Das Zitat ist mit der Nachricht rausgegangen und steht ab jetzt in
        // ihr — ueber dem leeren Feld saehe es aus wie Kontext der naechsten
        // Frage (dieselbe Ueberlegung wie bei den Anhaengen).
        window.App.quote?.clear?.();
        // force: die Frage ist raus, also faellt der Composer auf eine Zeile
        // zusammen — auch dann, wenn der Fokus (und damit die Tastatur) noch im
        // gerade geleerten Feld steht. Tippt man dort weiter, geht er wieder auf.
        window.App.composer?.collapse?.({ force: true });
        window.App.revealSentMessage?.();
      },

      promote() {
        if (!this.pending) return;
        this.pending = false;
        if (this.ownsHead) {
          this.ownsHead = false;
          window.App.setThreadQuestion?.(this.question);
        }
        // setThreadQuestion raeumt die Blase mit ab; ohne eigenen Kopf steht
        // sie gar nicht erst — der Aufruf ist die Zusicherung, dass danach
        // keine schwebende Nachricht mehr im Thread haengt.
        window.App.clearPendingThreadQuestion?.();
        window.App.setThreadQuestionAttachments?.(this.attachmentsMeta);
      },

      restore() {
        if (!this.pending) return;
        this.pending = false;
        this.ownsHead = false;
        window.App.clearPendingThreadQuestion?.();
        const input = document.getElementById("questionInput");
        // Wer in der Zwischenzeit schon die naechste Frage getippt hat, behaelt
        // sie: der zurueckgegebene Text ueberschreibt nie einen Entwurf.
        if (input && !input.value.trim()) {
          input.value = this.draft;
          input.dispatchEvent(new Event("input", { bubbles: true }));
          window.syncDemoChipState?.();
          // Zitat und Entwurf gehoeren zusammen: entweder kommt die Nachricht
          // ganz zurueck oder gar nicht.
          if (this.quote && !window.App.quote?.has?.()) window.App.quote?.set?.(this.quote);
        }
      }
    };

    async function releaseReservedUsageRun() {
      const run = window.App.usageRun?.current;
      if (!run?.key || !["new", "reserved"].includes(run.status)) return;
      // Lokal zuerst entkoppeln: ein spaeter Release-Response darf einen neuen
      // Lauf nicht ueberschreiben. Der Server lehnt consumed terminal mit 409 ab.
      window.App.usageRun.clear();
      try {
        const token = await window.auth?.currentUser?.getIdToken?.();
        if (!token) return;
        await fetch("/usage/run/release", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ id_token: token, usage_run_key: run.key }),
          keepalive: true
        });
      } catch (error) {
        console.warn("Could not release unused run reservation:", error);
      }
    }

    function setSendButtonRunning(isRunning) {
      const sendButton = document.getElementById("sendButton");
      if (!sendButton) return;

      sendButton.disabled = false;
      sendButton.classList.toggle("is-cancel-action", isRunning);
      sendButton.title = isRunning ? "Cancel request" : "Send question";
      sendButton.setAttribute("aria-label", isRunning ? "Cancel request" : "Send question");
      sendButton.innerHTML = isRunning
        ? `<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
          <rect x="8" y="8" width="8" height="8" rx="1.3"></rect>
        </svg>`
        : `<svg viewBox="0 0 24 24" aria-hidden="true" style="width: 13px; height: 13px;">
          <path d="M3.4 3.6 21 12 3.4 20.4 7 12z" fill="currentColor"></path>
        </svg>`;

      if (!isRunning && typeof window.updateQuestionInputAccess === "function") {
        window.updateQuestionInputAccess();
      }
    }

    // Ein Lauf ist erst zu Ende, wenn auch Consensus/Differences fertig sind.
    // Solange eine der beiden Phasen laeuft, bleibt der Send-Button ein
    // Cancel-Button (Consensus startet direkt nach dem letzten Modell).
    function isConsensusRunActive() {
      return consensusLifecycle?.isRunning?.() === true;
    }

    function isRunActive() {
      return queryRequestRunning || isConsensusRunActive();
    }

    function syncSendButtonRunning() {
      setSendButtonRunning(isRunActive());
    }

    function isActiveQueryRun(runId) {
      return queryRequestRunning
        && runId === currentQueryRunId
        && currentQueryController
        && !currentQueryController.signal.aborted;
    }

    function isAbortError(error) {
      return error && error.name === "AbortError";
    }

    function finishQueryRun(runId) {
      if (runId !== currentQueryRunId) return;
      queryRequestRunning = false;
      currentQueryController = null;
      syncSendButtonRunning();
    }

    function markPendingQueryResponsesCanceled() {
      const boxIds = [
        "openaiResponse",
        "mistralResponse",
        "claudeResponse",
        "geminiResponse",
        "deepseekResponse",
        "grokResponse"
      ];

      boxIds.forEach(id => {
        const contentEl = document.getElementById(id)?.querySelector(".collapsible-content");
        if (contentEl && (contentEl.querySelector(".thinking-wrap") || contentEl.classList.contains("is-streaming"))) {
          contentEl.classList.remove("is-streaming");
          contentEl.innerText = "Request canceled.";
        }
      });

      const consensusBox = document.getElementById("consensusResponse");
      const consensusMain = window.App.consensusBodyEl(consensusBox);
      const consensusDiff = consensusBox?.querySelector(".consensus-differences p");
      if (consensusMain && (consensusMain.querySelector(".thinking-wrap") || consensusMain.classList.contains("is-streaming"))) {
        consensusMain.classList.remove("is-streaming");
        consensusMain.innerText = "Request canceled.";
      }
      if (consensusDiff && (consensusDiff.querySelector(".thinking-wrap") || consensusDiff.classList.contains("is-streaming"))) {
        consensusDiff.classList.remove("is-streaming");
        consensusDiff.innerText = "";
      }
      if (window.resetCredibilityFrame && consensusBox) {
        window.resetCredibilityFrame(consensusBox.querySelector(".consensus-differences"));
      }
      // Der Status-Hub steuert neben dem Agent-Panel auch die kompakte
      // Consensus-Pipeline. Deshalb immer melden, auch ohne Agent Mode.
      setAgentModeStatus("canceled");
    }

    window.cancelCurrentQuery = function () {
      if (!queryRequestRunning || !currentQueryController) return;
      const runId = currentQueryRunId;
      currentQueryController.abort();
      markPendingQueryResponsesCanceled();
      finishQueryRun(runId);
      // Solange die Nachricht noch schwebt, gehoert die sichtbare Antwort dem
      // VORIGEN Turn — der bleibt stehen, denn abgebrochen wurde der neue.
      if (!sentMessage.pending) window.hideConsensusOutput?.();
      sentMessage.restore();
      const chatSession = window.App.chatSession;
      if (chatSession?.pendingClientRequestId) {
        chatSession.markPendingUncertain?.();
        const input = document.getElementById("questionInput");
        const pendingQuestion = String(chatSession.logicalRun?.question || "");
        if (input && pendingQuestion) {
          input.value = pendingQuestion;
          input.dispatchEvent(new Event("input", { bubbles: true }));
        }
      } else {
        releaseReservedUsageRun();
      }
      window.App.followup?.restoreAfterBlockedRun?.();
      trackAppEvent("app_query_canceled");
    };

    // Senden der Frage an die aktiven Modelle
    window.sendQuestion = async function () {
      if (queryRequestRunning) {
        window.cancelCurrentQuery();
        return;
      }

      // Der Lauf haengt jetzt in der Consensus-/Differences-Phase: derselbe
      // Button bricht sie ab, statt eine neue Frage zu starten.
      if (isConsensusRunActive()) {
        window.cancelCurrentConsensus?.();
        return;
      }

      const selectedModelCount = getSelectedModelCount();
      if (selectedModelCount < 2) {
        window.updateQuestionInputAccess?.();
        trackAppEvent("app_query_blocked", {
          reason: "minimum_models",
          selected_models: selectedModelCount
        });
        return;
      }

      // Überprüfe zuerst das Wortlimit. Falls überschritten, wird die Funktion beendet.
      if (typeof window.updateQuestionInputAccess === "function" && !window.updateQuestionInputAccess()) {
        return;
      }

      const mode = getActiveMode();
      if (!window.validateInputText()) {
        return;
      }

      // Das Zitat aus "Ask about this" ist ab hier Teil der Frage: ein Text
      // fuer Thread-Kopf, Bookmark, Chat-Kontext und die sechs Modelle. Der
      // Entwurf ohne Zitat bleibt daneben stehen, damit ein geplatzter Lauf
      // Feld UND Zitatflaeche wieder so hinterlaesst, wie sie waren.
      const draftQuestion = document.getElementById("questionInput").value;
      const quotedContext = window.App.quote?.text?.() || "";
      const question = window.App.quote?.compose?.(draftQuestion) ?? draftQuestion;
      const agentModeAtStart = isAgentModeEnabled?.() === true;
      // Follow-ups are consensus conversations. A direct comparison is a
      // fresh one-question fan-out and must not create a pending chat turn
      // that can only be finalized by the disabled consensus endpoint.
      const followupRequested = agentModeAtStart
        && window.App.followup?.isArmed?.() === true;
      window.App.state.set("lastQuestion", question, "run");

      if (!question.trim()) {
        alert("Please enter a question.");
        return;
      }

      // Die Frage geht raus — der Entwurf aus der Bestaetigungsphase hat
      // seinen Zweck erfuellt und darf beim naechsten Laden nicht wieder
      // im Feld stehen.
      window.App.clearQuestionDraft?.();

      // Kontingent VOR dem ersten sichtbaren Schritt pruefen — aber nach dem
      // Demo-Check unten wuerde es zu spaet sein, also hier mit derselben
      // Ausnahme: die Demo kostet nichts. Ohne diese Schranke sah ein leeres
      // Kontingent so aus, als liefe der Vergleich los: onPrepare() zeigt den
      // gefuehrten Lauf, und der 403 aus /prepare nahm ihn per dismiss()
      // gleich wieder weg. Der Server bleibt die Autoritaet (siehe den
      // /prepare-Zweig weiter unten); das hier ist nur die Zusage, dass
      // nichts zu laufen scheint, was gar nicht laeuft.
      if (!isDemoQuery(question) || followupRequested) {
        if (window.App.usageLimit?.blockIfExhausted?.({
          useOwnKeys: document.getElementById("useOwnKeysSwitch")?.checked === true,
          deepThink: document.getElementById("deepSearchToggle")?.checked === true,
          source: "send"
        })) {
          trackAppEvent("app_query_blocked", { reason: "usage_limit" });
          return;
        }
        // Ein neuer Versuch raeumt die alte Absage weg: sie gehoert zu dem
        // Lauf, der nicht stattgefunden hat, nicht zu diesem hier.
        window.App.usageLimit?.hide?.();
      }

      // A restored bookmark temporarily shows the immutable model labels from
      // that historical run. A fresh run must display the current picker/Deep
      // Think labels again without the bookmark ever mutating those controls.
      window.App.updateDeepThinkText?.();
      window.App.bookmarkSession?.begin?.(question, { followup: followupRequested });

      if (agentModeAtStart) {
        // Agent Mode wird zum gefuehrten Thread: Frage oben, Composer unten.
        window.exitHeroMode?.();
        // Bei einem Follow-up bleibt der bisherige Turn waehrend /prepare noch
        // unangetastet. Erst wenn der Lauf wirklich fortgesetzt wird,
        // archivieren wir ihn und setzen die neue Frage darunter — bis dahin
        // steht sie als eigene Blase darunter (sentMessage.hold).
        if (!followupRequested) {
          window.App.setThreadQuestion?.(question);
        }
        // Der gefuehrte Lauf beginnt vor /prepare, damit die Pipeline bereits
        // zwischen Klick und erstem Modell-Token sichtbar ist.
        window.App?.consensusPipeline?.onPrepare?.();
      } else {
        // Direktvergleich: dieselbe ruhige Zwei-Spalten-Oberflaeche wie im
        // Ausgangszustand bleibt stehen. Kein Thread-Kopf, kein Pipeline-Widget.
        window.App.followup?.reset?.();
        window.App.chatSession?.reset?.();
        window.enterDirectComparisonView?.();
        window.App?.consensusPipeline?.dismiss?.();
      }

      // === DEMO: Früh raus, wenn "Demo" ===
      if (isDemoQuery(question) && !followupRequested) {
        trackAppEvent("app_demo_started", {
          selected_models: getSelectedModelCount(),
          agent_mode: typeof window.isAgentModeEnabled === "function" && window.isAgentModeEnabled()
        });
        // optional: Free/Deep Counter im Sidebar auf einen sicheren Dummy setzen
        window.App.renderUsageDisplay({
          remaining: 3,
          deepRemaining: 0,
          totalLimit: 3,
          deepLimit: 0
        });

        // Spinners zeigen und Demo durchspielen
        await window.runDemoFlow(question);
        return; // WICHTIG: keine echten API-Calls ausführen
      }
      // === DEMO: Früh raus, wenn "Demo" ===

      // Die Nachricht verlaesst das Eingabefeld SOFORT (die Demo tippt in
      // dasselbe Feld weiter und ist deshalb oben schon raus). Sonst sieht der
      // Weg bis zum ersten sichtbaren Schritt aus, als sei der Klick ins Leere
      // gegangen — im laufenden Gespraech dauert er am laengsten, weil dort
      // zusaetzlich der Chat-Kontext gebunden wird.
      if (agentModeAtStart) {
        sentMessage.hold(question, {
          ownsHead: followupRequested,
          draft: draftQuestion,
          quote: quotedContext
        });
      }

      // clearResponseBoxes();
      consensusGenerated = false;
      window.App.setAppTitle(question);

      trackAppEvent("app_query_started", {
        mode,
        selected_models: getSelectedModelCount(),
        custom_credentials: document.getElementById("useOwnKeysSwitch")?.checked === true,
        logged_in: !!window.auth?.currentUser,
        agent_mode: typeof window.isAgentModeEnabled === "function" && window.isAgentModeEnabled(),
        auto_consensus: agentModeAtStart
          && document.getElementById("autoConsensusToggle")?.checked === true
      });

      currentQueryRunId++;
      const queryRunId = currentQueryRunId;
      currentQueryController = new AbortController();
      const querySignal = currentQueryController.signal;
      queryRequestRunning = true;
      const queryAuthUser = window.auth?.currentUser || null;
      const queryAuthUid = queryAuthUser?.uid || null;
      const queryAuthGeneration = window.__consensioAuthState?.generation;
      const queryAuthIsCurrent = () => window.auth?.currentUser === queryAuthUser
        && (window.auth?.currentUser?.uid || null) === queryAuthUid
        && window.__consensioAuthState?.generation === queryAuthGeneration;
      setSendButtonRunning(true);
      // Keep consensus unavailable until the current model run produces enough complete answers.
      // Bei jeder neuen Frage den Konsens-Bereich wieder ausblenden — im
      // laufenden Gespraech aber erst, wenn der Lauf wirklich stattfindet.
      // Bis dahin ist die letzte Antwort das, was auf dem Schirm steht: sie
      // gehoert zu der Frage, die oben im Thread noch als Kopf sitzt, und ein
      // Lauf, der am Kontingent scheitert, darf sie nicht abgeraeumt haben.
      // Der Follow-up-Zweig weiter unten holt das nach — dort, wo sie in den
      // Verlauf uebergeht.
      if (!followupRequested) window.hideConsensusOutput?.();

      // 1. Definiere useOwnKeys frühzeitig
      const useOwnKeys = document.getElementById("useOwnKeysSwitch").checked;

      // --- NEU: Frisches Token holen ---
      let validIdToken = null;

      if (queryAuthUser) {
        try {
          // true erzwingt Refresh, false (Standard) nimmt Cache wenn gültig.
          // false reicht meistens, aber bei Fehlern ist das SDK smart genug.
          validIdToken = await queryAuthUser.getIdToken();
          // Token resolution itself is not abortable. Fence the continuation
          // before it may reserve usage or consume follow-up context.
          if (!isActiveQueryRun(queryRunId)) return;
          if (!queryAuthIsCurrent()) {
            currentQueryController?.abort();
            finishQueryRun(queryRunId);
            return;
          }

          // Optional: LocalStorage updaten, damit er nicht komplett asynchron läuft
          localStorage.setItem("id_token", validIdToken);
        } catch (e) {
          console.error("Fehler beim Abrufen des frischen Tokens:", e);
          // Fallback: Versuche es trotzdem mit dem alten Token aus dem Storage, falls vorhanden
          validIdToken = localStorage.getItem("id_token");
        }
      }

      if (!isActiveQueryRun(queryRunId)) return;
      if (!queryAuthIsCurrent()) {
        currentQueryController?.abort();
        finishQueryRun(queryRunId);
        return;
      }

      if (!validIdToken) {
        alert(useOwnKeys
          ? "Please log in before using your own API keys."
          : "Please log in before sending a question.");
        finishQueryRun(queryRunId);
        sentMessage.restore();
        return;
      }

      try {
        const reconciledTurn = await window.App.chatSession?.inspectPendingTurn?.({
          idToken: validIdToken,
          signal: querySignal
        });
        if (reconciledTurn?.status === "completed") {
          finishQueryRun(queryRunId);
          // Der Turn ist serverseitig fertig: dieselbe Frage wird nicht noch
          // einmal gestellt, sondern nachgezeichnet. Sie gehoert damit an den
          // Kopf des Threads — der Vorgaenger davor in den Verlauf.
          window.App.followup?.archiveCurrentExchange?.();
          sentMessage.promote();
          await window.getConsensus?.("replay");
          return;
        }
      } catch (error) {
        if (isAbortError(error) || !isActiveQueryRun(queryRunId)) return;
        finishQueryRun(queryRunId);
        sentMessage.restore();
        window.App.followup?.restoreAfterBlockedRun?.();
        window.App?.showPopup?.(
          error?.message || "The pending conversation turn could not be checked. Please retry."
        );
        return;
      }

      // 0. Zuerst den gespeicherten Prompt aus dem Speicher holen
      const savedPrompt = localStorage.getItem("systemPrompt");

      // 1. Dann basePrompt definieren
      const basePrompt =
        savedPrompt ||
        "Please answer thoroughly and precisely, explaining your reasoning and covering the relevant details. Do not oversimplify. Do not ask any follow-up or clarifying questions; answer directly with the information available.";

      // 2. Dann die Datums-Berechnung
      const dateObj = new Date();
      const weekday = dateObj.toLocaleDateString('en-US', { weekday: 'long' });
      const yyyy = dateObj.getFullYear();
      const mm = String(dateObj.getMonth() + 1).padStart(2, '0');
      const dd = String(dateObj.getDate()).padStart(2, '0');

      const todayStr = `${weekday}, ${yyyy}-${mm}-${dd}`;

      // 3. Jetzt storedSystemPrompt zusammenbauen
      const storedSystemPrompt = `Today is ${todayStr}. ${basePrompt}`;

      effectiveSystemPrompt = storedSystemPrompt;

      const deepSearchFlag = document.getElementById("deepSearchToggle").checked;
      const attachmentsPayload = (typeof window.getAttachmentsPayload === "function")
        ? window.getAttachmentsPayload()
        : [];
      if (attachmentsPayload.length && !window.isUserPro) {
        if (!window.App?.showProFeatureModal?.("File uploads")) {
          window.App?.showPopup?.("File uploads are off here. Remove the attachments to continue.");
        }
        finishQueryRun(queryRunId);
        sentMessage.restore();
        return;
      }

      // Metadaten der mitgeschickten Anhänge merken (für Bookmarks, ohne Dateidaten)
      window.lastQuestionAttachmentsMeta = attachmentsPayload.map(function (att) {
        return { name: att.name, mime: att.mime, size: att.size || 0 };
      });

      // DeepSeek's chat API cannot consume the supported attachment inputs.
      // Keep this as a hard send-time invariant, not merely UI state: a stale
      // checkbox or another renderer must never put DeepSeek back into the
      // request fan-out while files are attached.
      const deepSeekBlockedByAttachments = attachmentsPayload.length > 0;
      // Der Block gehoert ab hier dem Lauf und ueberlebt jeden Tier-Refresh
      // (jede /ask-Antwort ruft restoreModelSelections). Ohne Anhaenge faellt
      // er hier — vor der Fan-out-Auswahl, damit die gespeicherte Wahl wieder
      // gilt und kein Modell erst mitten im Lauf dazukommt.
      window.App.setRunModelBlock?.("deepseekResponse", deepSeekBlockedByAttachments);
      const consensusModelForRun = document.getElementById("consensusModelDropdown")?.value || "";
      const selectedProviderConfigsForRun = [
        ["OpenAI", "selectOpenAI", "openaiModelSelect"],
        ["Mistral", "selectMistral", "mistralModelSelect"],
        ["Anthropic", "selectClaude", "claudeModelSelect"],
        ["Gemini", "selectGemini", "geminiModelSelect"],
        ["DeepSeek", "selectDeepSeek", "deepseekModelSelect"],
        ["Grok", "selectGrok", "grokModelSelect"]
      ].filter(([provider, checkboxId]) => (
        document.getElementById(checkboxId)?.checked
        && !(provider === "DeepSeek" && deepSeekBlockedByAttachments)
      ));
      const selectedModelConfigsForRun = selectedProviderConfigsForRun
        .map(([, , selectId]) => document.getElementById(selectId)?.value)
        .filter(Boolean);
      // Attachment/tier/run filters are authoritative. Re-check the minimum
      // here, before a usage run or /prepare can be created; the initial
      // checkbox count may include a stale DeepSeek selection that files make
      // ineligible for this run.
      if (selectedProviderConfigsForRun.length < 2) {
        finishQueryRun(queryRunId);
        setAgentModeStatus("error", "Choose at least two compatible models for this run.");
        sentMessage.restore();
        window.App.followup?.restoreAfterBlockedRun?.();
        window.App?.showPopup?.("Choose at least two compatible models. Remove the attachment or select another model.");
        trackAppEvent("app_query_blocked", {
          reason: "minimum_models_after_filters",
          selected_models: selectedProviderConfigsForRun.length
        });
        return;
      }
      const selectedConsensusOptionForRun = document.getElementById("consensusModelDropdown")
        ?.selectedOptions?.[0];
      const memoryProviderForRun = String(
        selectedConsensusOptionForRun?.dataset?.engineProvider || ""
      ).trim().toLowerCase();
      const ownKeyStorageByProvider = {
        openai: "openaiKey",
        mistral: "mistralKey",
        anthropic: "anthropicKey",
        gemini: "geminiKey",
        deepseek: "deepseekKey",
        grok: "grokKey"
      };
      if (useOwnKeys) {
        const requiredOwnKeyProviders = new Set(
          selectedProviderConfigsForRun.map(([provider]) => provider.toLowerCase())
        );
        if (memoryProviderForRun) requiredOwnKeyProviders.add(memoryProviderForRun);
        const missingOwnKeyProviders = Array.from(requiredOwnKeyProviders).filter(provider => {
          const storageKey = ownKeyStorageByProvider[provider];
          return !storageKey || !String(localStorage.getItem(storageKey) || "").trim();
        });
        if (missingOwnKeyProviders.length) {
          finishQueryRun(queryRunId);
          sentMessage.restore();
          window.App?.showPopup?.(
            `Add API keys for the selected providers before sending: ${missingOwnKeyProviders.join(", ")}.`
          );
          return;
        }
      }
      const reuseUsageRun = window.App.chatSession?.canReuseUsageRun?.({
        question,
        mode,
        deepSearch: deepSearchFlag,
        selectedModels: selectedModelConfigsForRun,
        consensusModel: consensusModelForRun,
        isFollowup: followupRequested,
        useOwnKeys
      }) === true;
      if (!reuseUsageRun) await releaseReservedUsageRun();
      const usageRun = reuseUsageRun
        ? window.App.usageRun.ensure(deepSearchFlag, useOwnKeys)
        : window.App.usageRun.start(deepSearchFlag, useOwnKeys);

      function enforceDeepSeekAttachmentBlock() {
        if (!deepSeekBlockedByAttachments) return;
        window.App.setRunModelBlock?.("deepseekResponse", true);
      }
      enforceDeepSeekAttachmentBlock();

      const baseSpinnerHTML = window.spinnerHTML;

      // Nur Boxen der aktuell ausgewählten Modelle
      const modelBoxes = [];

      if (document.getElementById("selectOpenAI")?.checked) {
        const box = document.getElementById("openaiResponse");
        if (box) modelBoxes.push(box);
      }
      if (document.getElementById("selectMistral")?.checked) {
        const box = document.getElementById("mistralResponse");
        if (box) modelBoxes.push(box);
      }
      if (document.getElementById("selectClaude")?.checked) {
        const box = document.getElementById("claudeResponse");
        if (box) modelBoxes.push(box);
      }
      if (document.getElementById("selectGemini")?.checked) {
        const box = document.getElementById("geminiResponse");
        if (box) modelBoxes.push(box);
      }
      if (!deepSeekBlockedByAttachments && document.getElementById("selectDeepSeek")?.checked) {
        const box = document.getElementById("deepseekResponse");
        if (box) modelBoxes.push(box);
      }
      if (document.getElementById("selectGrok")?.checked) {
        const box = document.getElementById("grokResponse");
        if (box) modelBoxes.push(box);
      }
      if (modelBoxes.length < 2) {
        await releaseReservedUsageRun();
        finishQueryRun(queryRunId);
        setAgentModeStatus("error", "Choose at least two compatible models for this run.");
        sentMessage.restore();
        window.App.followup?.restoreAfterBlockedRun?.();
        return;
      }

      let queryHadBlockingError = false;
      let queryBlockingErrorMessage = "";
      let successfulResponses = 0;

      // --- Einzelne Modelle abbrechbar machen ---------------------------
      // Ein Lauf ist so langsam wie sein langsamstes Modell. Damit ein
      // haengendes Modell nicht den ganzen Lauf blockiert, bekommt jedes
      // seinen eigenen Controller; der Lauf-Controller kaskadiert darauf.
      // Ein uebersprungenes Modell zaehlt wie ein fehlgeschlagenes: der
      // Konsens laeuft ohne es weiter.
      const modelControllers = new Map();
      const skippedBoxIds = new Set();

      function signalFor(boxId) {
        let controller = modelControllers.get(boxId);
        if (!controller) {
          controller = new AbortController();
          modelControllers.set(boxId, controller);
        }
        return controller.signal;
      }

      querySignal.addEventListener("abort", () => {
        modelControllers.forEach(controller => controller.abort());
      });

      window.App.skipModel = function (boxId) {
        if (!isActiveQueryRun(queryRunId)) return false;
        if (skippedBoxIds.has(boxId)) return false;
        const box = document.getElementById(boxId);
        const outputEl = box?.querySelector(".collapsible-content");
        if (!box || !outputEl) return false;
        if (box.dataset.responseState !== "pending") return false;

        skippedBoxIds.add(boxId);
        modelControllers.get(boxId)?.abort();
        outputEl.classList.remove("is-streaming");
        box.dataset.responseError = "true";
        box.dataset.responseState = "error";
        box.dataset.responseSkipped = "true";
        outputEl.innerText = "Skipped — this model took too long, so the run went on without it.";
        if (isAgentModeEnabled()) window.updateAgentModeUI?.();
        trackAppEvent("app_model_skipped", { model: box.dataset.model || boxId });
        checkAllResponses();
        return true;
      };

      // Ein uebersprungenes Modell hat seinen checkAllResponses()-Aufruf schon
      // verbraucht: die spaete Antwort darf den Zaehler nicht ein zweites Mal
      // hochzaehlen und den fertigen Text nicht ueberschreiben.
      function isSkipped(boxId) {
        return skippedBoxIds.has(boxId);
      }

      function unwrapApiError(data) {
        const detail = data?.detail;
        if (detail && typeof detail === "object") {
          return {
            ...detail,
            error: detail.error || detail.message || "Request failed."
          };
        }
        return data || {};
      }

      function getApiErrorMessage(data, fallback = "Request failed.") {
        const normalized = unwrapApiError(data);
        const candidate = normalized.error || normalized.detail || normalized.message || fallback;
        if (typeof candidate === "string") return candidate;
        if (Array.isArray(candidate)) {
          const messages = candidate
            .map(item => item && (item.msg || item.message || item.error))
            .filter(value => typeof value === "string");
          if (messages.length) return messages.join(" ");
        }
        if (candidate && typeof candidate === "object") {
          const nested = candidate.error || candidate.message || candidate.detail;
          if (typeof nested === "string") return nested;
        }
        return fallback;
      }

      // Ein Detektor fuer die ganze App (usage-limit.js). Der lokale Fallback
      // bleibt, damit ein fehlgeschlagenes Modul-Laden den Limit-Pfad nicht
      // stillschweigend in "unbekannter Fehler" kippen laesst.
      function isUsageLimitError(data, message = "") {
        if (window.App.usageLimit?.isLimitError) {
          return window.App.usageLimit.isLimitError(data, message);
        }
        const normalized = unwrapApiError(data);
        const code = String(normalized.error_code || normalized.code || "").toLowerCase();
        const text = String(message || normalized.error || normalized.detail || "").toLowerCase();
        return code.includes("limit")
          || text.includes("usage limit")
          || text.includes("quota")
          || text.includes("used up")
          || text.includes("exhausted");
      }

      function isUsageStorageBusyError(data) {
        const normalized = unwrapApiError(data);
        return String(normalized.error_code || normalized.code || "").toLowerCase()
          === "usage_storage_busy";
      }

      async function prepareWithUsageRetry(payload, signal) {
        const maxAttempts = 3;
        let result = null;
        for (let attempt = 1; attempt <= maxAttempts; attempt++) {
          const response = await fetch("/prepare", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
            signal
          });
          let data = {};
          try {
            data = await response.json();
          } catch (_) {
            data = { error: `Request failed with HTTP ${response.status}.` };
          }
          result = { response, data };
          if (response.ok || !isUsageStorageBusyError(data) || attempt === maxAttempts) {
            return result;
          }
          await new Promise(resolve => window.setTimeout(resolve, attempt * 350));
        }
        return result;
      }

      function updateUsageDisplayFromData(data) {
        const normalized = unwrapApiError(data);
        if (normalized.usage_run_status) {
          window.App.usageRun?.mark?.(normalized.usage_run_status);
        }
        window.App.renderUsageDisplay({
          remaining: normalized.free_usage_remaining,
          deepRemaining: normalized.deep_remaining,
          totalLimit: normalized.limit ?? window.currentMaxLimit,
          deepLimit: normalized.deep_limit ?? window.currentDeepLimit
        });
        if (normalized.is_pro_user !== undefined) {
          window.updateUserTierUI(normalized.is_pro_user === true, !!window.auth.currentUser);
        }
      }

      function markQueryBlockingError(message, data = {}) {
        // Sechs parallele Modell-Calls koennen dieselbe Absage sechsmal
        // melden. Die Karte wird deshalb nur beim ersten Mal aufgebaut,
        // sonst springt der Scroll bei jeder Antwort erneut.
        const first = !queryHadBlockingError;
        queryHadBlockingError = true;
        queryBlockingErrorMessage = message || queryBlockingErrorMessage || "The request could not be completed.";
        updateUsageDisplayFromData(data);
        if (isAgentModeEnabled()) {
          setAgentModeStatus("error", queryBlockingErrorMessage);
        }
        if (first && isUsageLimitError(data, message)) {
          window.App.usageLimit?.show?.({
            data,
            source: "ask",
            phase: "answers"
          });
        }
      }

      function markModelSuccess(outputEl) {
        const box = outputEl?.closest?.(".response-box");
        if (box) {
          delete box.dataset.responseError;
          box.dataset.responseState = "complete";
          successfulResponses++;
        }
        if (isAgentModeEnabled()) {
          window.updateAgentModeUI?.();
        }
      }

      function markModelError(outputEl, message, data = {}) {
        message = getApiErrorMessage({ error: message }, "The model request failed.");
        const box = outputEl?.closest?.(".response-box");
        if (box) {
          box.dataset.responseError = "true";
          box.dataset.responseState = "error";
        }
        outputEl.innerText = message;
        if (isAgentModeEnabled()) {
          window.updateAgentModeUI?.();
        }
        if (isUsageLimitError(data, message)) {
          markQueryBlockingError(message, data);
        }
      }

      // consume() advances the visible thread. Owned active chats replace its
      // legacy one-hop payload with an authoritative context version below;
      // bookmark restores without an active chat keep the compatibility path.
      const authoritativeContinuation = Boolean(
        followupRequested && window.App.chatSession?.hasActiveChat?.()
      );
      const followupContext = window.App.followup?.consume?.() || null;
      const bookmarkPreviousQuestion = String(
        followupContext?.previous_question || ""
      ).trim();
      if (followupContext) {
        // consume() hat den Lauf schon als Follow-up markiert (kein reset,
        // sonst ginge das In-Flight-Flag verloren).
        trackAppEvent("app_followup_sent");
      } else {
        // Frische Frage ersetzt den alten Konsens: Affordance/Flag verwerfen.
        // Ohne Kontext gehoeren auch die archivierten Turns nicht mehr darueber:
        // die Modelle sehen sie nicht, also darf der Thread sie nicht behaupten.
        window.App.followup?.clearHistory?.();
        window.App.followup?.reset?.();
        if (followupRequested) {
          // Kein Kontext, also auch kein Vorgaenger, der stehen bleiben
          // duerfte: die schwebende Nachricht wird der Kopf, die alte Antwort
          // geht mit dem Rest des Fadens.
          sentMessage.promote();
          window.hideConsensusOutput?.();
        }
      }

      // Wir rufen /prepare IMMER auf, damit Wetter-Infos etc. injiziert werden können.
      let prepareSucceeded = false;
      try {
        // 🔹 Firebase-ID-Token holen (falls eingeloggt)
        let idToken = null;
        if (window.auth && window.auth.currentUser) {
          try {
            idToken = await window.auth.currentUser.getIdToken();
          } catch (e) {
            console.error("Failed to get id_token for /prepare:", e);
          }
        }

        const preparePayload = {
          question: question,
          system_prompt: storedSystemPrompt,
          deep_search: deepSearchFlag,
          mode: mode,
          useOwnKeys: useOwnKeys
        };
        if (usageRun.key) {
          preparePayload.usage_run_key = usageRun.key;
        }

        // Nur anhängen, wenn wirklich vorhanden
        if (idToken) {
          preparePayload.id_token = idToken;
        }
        if (followupContext && !authoritativeContinuation) {
          preparePayload.context = followupContext;
        }

        const { response: prepareResp, data: prepareData } = await prepareWithUsageRetry(
          preparePayload, querySignal
        );
        prepareSucceeded = prepareResp.ok;

        if (prepareResp.ok && prepareData.system_prompt) {
          updateUsageDisplayFromData(prepareData);
          // ÄNDERUNG: Wir prüfen, ob der neue Prompt ANDERS ist als der alte.
          if (prepareData.system_prompt !== storedSystemPrompt) {
            effectiveSystemPrompt = prepareData.system_prompt;
          }
          // Wenn sie gleich sind: Nichts tun, nichts loggen (effectiveSystemPrompt ist ja schon gesetzt).
        } else {
          if (isUsageStorageBusyError(prepareData)) {
            const message = getApiErrorMessage(
              prepareData,
              "The usage service is busy right now. Please try again in a moment."
            );
            if (!followupContext) {
              modelBoxes.forEach(box => {
                const outputEl = box.querySelector(".collapsible-content");
                if (outputEl) markModelError(outputEl, message, prepareData);
              });
            }
            finishQueryRun(queryRunId);
            setAgentModeStatus("error", message);
            window.App.usageLimit?.showTemporaryStorageBusy?.();
            sentMessage.restore();
            window.App.followup?.restoreAfterBlockedRun?.();
            return;
          }
          if (isUsageLimitError(prepareData)) {
            const message = getApiErrorMessage(prepareData, "Usage limit reached.");
            // Die Zahlen aus der Absage sind die frischesten, die es gibt:
            // erst den Ring korrigieren, dann die Karte bauen, die daraus
            // liest.
            updateUsageDisplayFromData(prepareData);
            if (!followupContext) {
              modelBoxes.forEach(box => {
                const outputEl = box.querySelector(".collapsible-content");
                if (outputEl) markModelError(outputEl, message, prepareData);
              });
            }
            finishQueryRun(queryRunId);
            // Raeumt den gefuehrten Lauf ab (dismiss); die bleibende Meldung
            // steht danach in #runBlocked — sonst faellt die Seite in genau
            // das leere Nichts zurueck, das dieser Pfad frueher hinterliess.
            setAgentModeStatus("error", message);
            window.App.usageRun?.clear?.();
            window.App.usageLimit?.show?.({
              data: prepareData,
              source: "prepare",
              phase: "prepare"
            });
            // Nichts ist rausgegangen, also wird auch nichts uebernommen: die
            // Nachricht geht unveraendert und abschickbar ins Feld zurueck,
            // der bisherige Turn bleibt stehen, wo er steht. Bei einem
            // Follow-up hat consume() den Kontext ausserdem schon verbraucht —
            // ohne diese Ruecknahme ginge die Wiederholung stillschweigend
            // ohne den Kontext raus, den die Absage-Karte als ungesendet
            // ausweist.
            sentMessage.restore();
            window.App.followup?.restoreAfterBlockedRun?.();
            return;
          }
          console.warn("No valid system_prompt from /prepare, keeping base.");
        }

        // Quellen aktualisieren (nur relevant, wenn Search an war, sonst ist die Liste leer)

      } catch (err) {
        if (isAbortError(err) || !isActiveQueryRun(queryRunId)) {
          return;
        }
        console.error("Error during /prepare:", err);
        // Fallback: effectiveSystemPrompt bleibt der gespeicherte Prompt
      }

      if (!isActiveQueryRun(queryRunId)) {
        return;
      }

      const chatSession = window.App.chatSession;
      chatSession?.beginRun?.({
        question,
        mode,
        deepSearch: deepSearchFlag,
        selectedModels: selectedModelConfigsForRun,
        consensusModel: consensusModelForRun,
        isFollowup: !!followupContext,
        prepareSucceeded,
        useOwnKeys,
        usageRunKey: usageRun.key
      });
      let authoritativeContextBinding = null;
      try {
        if (authoritativeContinuation) {
          const pendingTurn = await chatSession?.ensurePendingTurn?.({
            idToken: validIdToken,
            question,
            consensusModel: consensusModelForRun,
            signal: querySignal
          });
          if (!pendingTurn) {
            // Ein erreichtes Limit oder ein Tier-Gate ist dauerhaft - dann darf
            // hier nicht "Please retry" stehen. chatSession haelt den Grund
            // fest, sofern der Server einen genannt hat.
            throw new Error(
              chatSession?.lastPersistenceError
              || "The conversation turn could not be prepared. Please retry."
            );
          }
          const memoryApiKey = useOwnKeys && ownKeyStorageByProvider[memoryProviderForRun]
            ? (localStorage.getItem(ownKeyStorageByProvider[memoryProviderForRun]) || "")
            : "";
          if (useOwnKeys && !memoryApiKey) {
            throw new Error(`Missing API key for the conversation memory provider: ${memoryProviderForRun || "selected consensus engine"}.`);
          }
          await chatSession.ensureContext({
            idToken: validIdToken,
            useOwnKeys,
            usageRunKey: usageRun.key,
            memoryApiKey,
            signal: querySignal
          });
          authoritativeContextBinding = chatSession.contextBinding();
          if (!authoritativeContextBinding) {
            throw new Error("The conversation context could not be bound. Please retry.");
          }
        }
      } catch (error) {
        if (isAbortError(error) || !isActiveQueryRun(queryRunId)) {
          chatSession?.markPendingUncertain?.();
          return;
        }
        const message = error?.message || "The conversation context could not be prepared.";
        if (!followupContext) {
          modelBoxes.forEach(box => {
            const outputEl = box.querySelector(".collapsible-content");
            if (outputEl) markModelError(outputEl, message);
          });
        }
        finishQueryRun(queryRunId);
        setAgentModeStatus("error", message);
        sentMessage.restore();
        window.App.followup?.restoreAfterBlockedRun?.();
        window.App?.showPopup?.(message);
        return;
      }

      function attachConversationContext(payload) {
        if (authoritativeContextBinding) {
          Object.assign(payload, authoritativeContextBinding);
        } else if (followupContext && !authoritativeContinuation) {
          payload.context = followupContext;
        }
        return payload;
      }

      if (followupContext) {
        // Der Lauf findet statt: erst wandert der bisherige Turn in den
        // Verlauf, dann wird die schwebende Nachricht sein Nachfolger.
        window.App.followup?.archiveCurrentExchange?.();
        sentMessage.promote();
        // Der archivierte Turn ist jetzt die sichtbare alte Antwort. Das Live-
        // Renderziel wird fuer den neuen Consensus frei und darf nicht unter
        // der neuen Frage noch einmal den alten Text zeigen.
        window.hideConsensusOutput?.();
        const liveConsensusBody = window.App.consensusBodyEl?.();
        if (liveConsensusBody) liveConsensusBody.innerHTML = "";
        window.resetConsensusInsights?.();
      }

      // Der Direktvergleich kennt keinen Thread-Kopf und gibt die Frage
      // deshalb erst hier ab; im Thread ist das Feld seit sentMessage.hold()
      // leer und dieser Aufruf nur noch die Zusicherung, dass es das bleibt.
      const questionInputEl = document.getElementById("questionInput");
      if (questionInputEl && questionInputEl.value === draftQuestion) {
        questionInputEl.value = "";
        questionInputEl.dispatchEvent(new Event("input", { bubbles: true }));
        window.syncDemoChipState?.();
      }
      // Im Thread hat sentMessage.hold() das Zitat schon abgegeben; im
      // Direktvergleich ist hier die Stelle, an der die Frage rausgeht.
      window.App.quote?.clear?.();
      // Ein Lauf, der bis hierher gekommen ist, findet statt: was noch
      // schwebt, gehoert jetzt dem aktiven Turn (im Thread laengst erledigt,
      // hier nur die letzte Schranke).
      sentMessage.promote();
      // Die Anhaenge sind mit DIESER Frage rausgegangen: der Composer gibt sie
      // ab, die Chips stehen ab jetzt an der Nachricht im Thread. Vorher hingen
      // sie ueber dem leeren Feld und sahen aus wie ein Anhang der naechsten
      // Frage. Muss vor enforceDeepSeekAttachmentBlock() passieren: das
      // Abgeben gibt DeepSeek wieder frei, und fuer DIESEN Lauf bleibt es
      // ausgeschlossen.
      const sentAttachments = window.App.attachments?.detachForMessage?.() || [];
      window.App.setThreadQuestionAttachments?.(sentAttachments);
      // force: die Frage ist raus, also faellt der Composer auf eine Zeile
      // zusammen — auch dann, wenn der Fokus (und damit die Tastatur) noch im
      // gerade geleerten Feld steht. Tippt man dort weiter, geht er wieder auf.
      window.App.composer?.collapse?.({ force: true });

      // /prepare refreshes the authoritative user tier. That refresh restores
      // persisted model selections, so reassert the attachment invariant
      // before answer counting and request fan-out.
      enforceDeepSeekAttachmentBlock();

      // Only now may the live response tree be reused. Until /prepare and the
      // optional context binding have succeeded, it remains the readable
      // completed predecessor and must survive credential/202/network errors.
      window.App.state.set("spinnerHTML", baseSpinnerHTML, "runUi");
      modelBoxes.forEach(box => {
        delete box.dataset.consensusAnswer;
        delete box.dataset.consensusSources;
        delete box.dataset.responseError;
        delete box.dataset.responseSkipped;
        box.dataset.responseState = "pending";
        window.setSpinnerEl(box);
      });
      window.App.state.set("currentEvidenceSources", [], "evidence");
      window.renderEvidenceSources?.([]);


      const deepSearchActive = document.getElementById("deepSearchToggle").checked;

      // Konsens unterbinden, solange noch Antworten fehlen
      totalRequiredResponses = 0;

      const openaiBox = document.getElementById("openaiResponse");
      const mistralBox = document.getElementById("mistralResponse");
      const claudeBox = document.getElementById("claudeResponse");
      const geminiBox = document.getElementById("geminiResponse");
      const deepseekBox = document.getElementById("deepseekResponse");
      const grokBox = document.getElementById("grokResponse");

      // Zähle nur die Modelle, die nicht als "ausgeschlossen" markiert sind
      if (!openaiBox.classList.contains("excluded")) totalRequiredResponses++;
      if (!mistralBox.classList.contains("excluded")) totalRequiredResponses++;
      if (!claudeBox.classList.contains("excluded")) totalRequiredResponses++;
      if (!geminiBox.classList.contains("excluded")) totalRequiredResponses++;
      if (!deepseekBox.classList.contains("excluded")) totalRequiredResponses++;
      if (!grokBox.classList.contains("excluded")) totalRequiredResponses++;

      let activeModels = [];
      if (document.getElementById("selectOpenAI").checked) activeModels.push("OpenAI");
      if (document.getElementById("selectMistral").checked) activeModels.push("Mistral");
      if (document.getElementById("selectClaude").checked) activeModels.push("Anthropic");
      if (document.getElementById("selectGemini").checked) activeModels.push("Gemini");
      if (!deepSeekBlockedByAttachments && document.getElementById("selectDeepSeek").checked) {
        activeModels.push("DeepSeek");
      }
      if (document.getElementById("selectGrok").checked) activeModels.push("Grok");
      setAgentModeStatus(activeModels.length > 0 ? "running" : "idle");

      // Spinner in den jeweiligen Response-Boxen setzen
      if (activeModels.includes("OpenAI")) {
        document.getElementById("openaiResponse").querySelector(".collapsible-content").innerHTML = window.spinnerHTML;
      }
      if (activeModels.includes("Mistral")) {
        document.getElementById("mistralResponse").querySelector(".collapsible-content").innerHTML = window.spinnerHTML;
      }
      if (activeModels.includes("Anthropic")) {
        document.getElementById("claudeResponse").querySelector(".collapsible-content").innerHTML = window.spinnerHTML;
      }
      if (activeModels.includes("Gemini")) {
        document.getElementById("geminiResponse").querySelector(".collapsible-content").innerHTML = window.spinnerHTML;
      }
      if (activeModels.includes("DeepSeek")) {
        document.getElementById("deepseekResponse").querySelector(".collapsible-content").innerHTML = window.spinnerHTML;
      }
      if (activeModels.includes("Grok")) {
        document.getElementById("grokResponse").querySelector(".collapsible-content").innerHTML = window.spinnerHTML;
      }
      const consensusAnswerBody = document.getElementById("consensusAnswerBody");
      if (consensusAnswerBody) consensusAnswerBody.innerHTML = "";
      // Veraltete Auswertung (Verdict, Badges, Karten) der vorherigen Frage entfernen
      window.resetConsensusInsights?.();

      // API Keys aus localStorage abrufen
      const openaiKey = localStorage.getItem("openaiKey") || "";
      const mistralKey = localStorage.getItem("mistralKey") || "";
      const anthropicKey = localStorage.getItem("anthropicKey") || "";
      const geminiKey = localStorage.getItem("geminiKey") || "";
      const deepseekKey = localStorage.getItem("deepseekKey") || "";
      const grokKey = localStorage.getItem("grokKey") || "";

      let responsesReceived = 0;
      const totalActive = activeModels.length;

      function checkAllResponses() {
        if (!isActiveQueryRun(queryRunId)) return;
        responsesReceived++;
        if (responsesReceived === totalActive) {
          // Sende-Button immer wieder freischalten
          finishQueryRun(queryRunId);
          if (successfulResponses === 0 && !queryHadBlockingError) {
            const failedModels = modelBoxes
              .filter(box => box.dataset.responseState === "error" && box.dataset.responseSkipped !== "true")
              .map(box => {
                const label = box.dataset.model || box.id || "model";
                const reason = box.querySelector(".collapsible-content")?.textContent?.trim() || "failed";
                return `${label}: ${reason.slice(0, 240)}`;
              });
            if (failedModels.length) {
              window.App.reportCriticalError?.({
                type: "run_failed",
                phase: "model_fanout",
                message: `All ${totalActive} selected model requests failed.`,
                details: failedModels.join(" | ")
              });
              markQueryBlockingError("All selected model requests failed.");
            }
          }
          if (queryHadBlockingError) {
            setAgentModeStatus("error", queryBlockingErrorMessage);
            trackAppEvent("app_query_completed", {
              status: "error",
              selected_models: totalActive
            });
            return;
          }

          setAgentModeStatus("complete");
          trackAppEvent("app_query_completed", {
            status: "success",
            selected_models: totalActive
          });

          // Consensus gehoert ausschliesslich zum Agent Mode. Im
          // Direktvergleich sind die sechs Antworten bereits das Ergebnis.
          const autoConsensusOn = agentModeAtStart
            && isAgentModeEnabled?.() === true
            && document.getElementById("autoConsensusToggle")?.checked !== false;
          if (!autoConsensusOn) return;

          const canGenerate = typeof window.canGenerateConsensus === "function"
            ? window.canGenerateConsensus()
            : true;

          if (!canGenerate && window.App.chatSession?.pendingTurnId) {
            // The early turn exists only for an authoritative continuation.
            // Let /consensus record insufficient_answers before any engine or
            // credential work, instead of leaving a permanent pending orphan.
            window.getConsensus("disposition").catch((error) => {
              console.error("Could not finalize the incomplete chat turn:", error);
            });
            return;
          }

          if (autoConsensusOn && canGenerate) {
            window.getConsensus("auto").catch((error) => {
              console.error("Fehler bei der Konsensgenerierung:", error);
            });
          }
        }
      }

      // Hilfsfunktion, um einen leeren API Key zu prüfen
      if (totalActive === 0) {
        setAgentModeStatus("idle");
        finishQueryRun(queryRunId);
        return;
      }

      function validateUserKey(keyName) {
        const key = localStorage.getItem(keyName);
        return key && key.trim() !== "";
      }

      // Copy-Button in Codeblöcken: dezenter Icon-Button oben rechts im <pre>,
      // passend zu den übrigen Icon-Buttons der App (statt Emoji + globalem
      // Button-Gradient). Icons als SVG, damit der Button nichts in den
      // kopierten Text einschleppt.
      var CODE_COPY_ICON =
        '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" ' +
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
        '<rect x="9" y="9" width="11" height="11" rx="2.5"></rect>' +
        '<path d="M5 15H4.5A2.5 2.5 0 0 1 2 12.5v-8A2.5 2.5 0 0 1 4.5 2h8A2.5 2.5 0 0 1 15 4.5V5"></path></svg>';
      var CODE_COPIED_ICON =
        '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" ' +
        'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
        '<path d="M4.5 12.5 10 18 19.5 7"></path></svg>';

      function addCopyButtons(container) {
        container.querySelectorAll('pre').forEach(function (pre) {
          // Falls bereits ein Copy-Button existiert, überspringen
          if (pre.querySelector('.response-code-copy, .copy-btn')) return;
          var btn = document.createElement('button');
          btn.type = 'button';
          btn.className = 'response-code-copy';
          btn.title = 'Copy code';
          btn.setAttribute('aria-label', 'Copy code');
          btn.innerHTML = CODE_COPY_ICON;
          pre.appendChild(btn);

          btn.addEventListener('click', function () {
            // Falls ein <code> innerhalb des <pre> existiert, kopiere dessen innerText
            var codeElement = pre.querySelector('code');
            var codeText = codeElement ? codeElement.innerText : pre.innerText;
            navigator.clipboard.writeText(codeText).then(function () {
              btn.innerHTML = CODE_COPIED_ICON;
              btn.classList.add('is-copied');
              setTimeout(function () {
                btn.innerHTML = CODE_COPY_ICON;
                btn.classList.remove('is-copied');
              }, 2000);
            });
          });
        });
      }

      // Mache addCopyButtons global verfügbar:
      window.addCopyButtons = addCopyButtons;

      function addNewTabToLinks(container) {
        container.querySelectorAll('a').forEach(function (link) {
          // Falls noch kein target-Attribut gesetzt ist, füge es hinzu
          if (!link.hasAttribute('target')) {
            link.setAttribute('target', '_blank');
            link.setAttribute('rel', 'noopener noreferrer');
          }
        });
      }

      // Mache die Funktion global verfügbar:
      window.addNewTabToLinks = addNewTabToLinks;

      // OpenAI
      if (activeModels.includes("OpenAI")) {
        const useOwnKeys = document.getElementById("useOwnKeysSwitch").checked;
        const payload = {
          question: question,
          deep_search: deepSearchFlag,      // oben definiert
          system_prompt: effectiveSystemPrompt,
          mode: mode,
          model: document.getElementById("openaiModelSelect").value,
          id_token: validIdToken,
          useOwnKeys: useOwnKeys,
          usage_run_key: usageRun.key
        };
        if (attachmentsPayload.length) payload.attachments = attachmentsPayload;
        attachConversationContext(payload);


        if (!useOwnKeys) {
          if (validIdToken) {
            payload.id_token = validIdToken; // Nimm das frische Token Variable
          } else {
            // Fallback, falls kein User eingeloggt ist aber auch keine Keys da sind
            payload.api_key = localStorage.getItem("openaiKey") || "";
          }
        } else {
          if (!validateUserKey("openaiKey")) {
            const outputEl = document.getElementById("openaiResponse").querySelector(".collapsible-content");
            outputEl.innerHTML = "Please log in and enter your own API key.";
            alert("Please enter a valid OpenAI API key.");
            currentQueryController?.abort();
            markPendingQueryResponsesCanceled();
            finishQueryRun(queryRunId);
            return;
          }
          payload.api_key = localStorage.getItem("openaiKey");
        }

        const openaiStreamRenderer = createStreamRenderer(
          document.getElementById("openaiResponse").querySelector(".collapsible-content"),
          () => isActiveQueryRun(queryRunId)
        );
        streamSSERequest('/ask_openai', payload, signalFor("openaiResponse"), { "delta": openaiStreamRenderer })
          .then(({ ok, status, data }) => {
            if (!ok) {
              const msg = getApiErrorMessage(data, `OpenAI HTTP ${status}`);
              throw new Error(msg);
            }

            return data;
          })
          .then((data) => {
            if (!isActiveQueryRun(queryRunId) || isSkipped("openaiResponse")) return;
            updateUsageDisplayFromData(data);
            const outputEl = document
              .getElementById("openaiResponse")
              .querySelector(".collapsible-content");

            if (data.response) {
              markModelSuccess(outputEl);
              const responseWithSources = window.renderModelResponseWithSources
                ? window.renderModelResponseWithSources(outputEl, data.response, data.sources || [])
                : data.response;
              if (window.auth.currentUser) {
                window.saveBookmark(question, responseWithSources, "OpenAI", mode, bookmarkPreviousQuestion);
              }
            } else if (data.error) {
              markModelError(outputEl, data.error, data);
            } else if (data.detail) {
              // Falls dein Backend irgendwo noch "detail" liefert
              markModelError(outputEl, getApiErrorMessage(data), data);
            } else {
              markModelError(outputEl, "Unexpected empty response from server.", data);
            }

            checkAllResponses();
          })
          .catch((error) => {
            if (isAbortError(error) || !isActiveQueryRun(queryRunId) || isSkipped("openaiResponse")) return;
            const outputEl = document
              .getElementById("openaiResponse")
              .querySelector(".collapsible-content");
            // Zeig den echten Grund statt des generischen Login-Texts
            markModelError(outputEl, `OpenAI error: ${error.message}`, { error: error.message });
            console.error("Error at OpenAI:", error);
            checkAllResponses();
          });
      }


      // Mistral
      if (activeModels.includes("Mistral")) {
        const useOwnKeys = document.getElementById("useOwnKeysSwitch").checked;
        const payload = {
          question: question,
          deep_search: deepSearchFlag,
          system_prompt: effectiveSystemPrompt,
          mode: mode,
          model: document.getElementById("mistralModelSelect").value,
          id_token: validIdToken,
          useOwnKeys: useOwnKeys,
          usage_run_key: usageRun.key
        };
        if (attachmentsPayload.length) payload.attachments = attachmentsPayload;
        attachConversationContext(payload);

        if (!useOwnKeys) {
          if (validIdToken) {
            payload.id_token = validIdToken; // Nimm das frische Token Variable
          } else {
            // Fallback, falls kein User eingeloggt ist aber auch keine Keys da sind
            payload.api_key = localStorage.getItem("mistralKey") || "";
          }
        } else {
          if (!validateUserKey("mistralKey")) {
            const outputEl = document.getElementById("mistralResponse").querySelector(".collapsible-content");
            outputEl.innerHTML = "Please log in and enter your own API key.";
            alert("Please enter a valid Mistral API key.");
            currentQueryController?.abort();
            markPendingQueryResponsesCanceled();
            finishQueryRun(queryRunId);
            return;
          }
          payload.api_key = localStorage.getItem("mistralKey");
        }

        const mistralStreamRenderer = createStreamRenderer(
          document.getElementById("mistralResponse").querySelector(".collapsible-content"),
          () => isActiveQueryRun(queryRunId)
        );
        streamSSERequest('/ask_mistral', payload, signalFor("mistralResponse"), { "delta": mistralStreamRenderer })
          .then(({ data }) => data)
          .then(data => {
            if (!isActiveQueryRun(queryRunId) || isSkipped("mistralResponse")) return;
            updateUsageDisplayFromData(data);
            let outputEl = document.getElementById("mistralResponse").querySelector(".collapsible-content");
            if (data.response) {
              markModelSuccess(outputEl);
              const responseWithSources = window.renderModelResponseWithSources
                ? window.renderModelResponseWithSources(outputEl, data.response, data.sources || [])
                : data.response;

              if (window.auth.currentUser) {
                window.saveBookmark(question, responseWithSources, "Mistral", mode, bookmarkPreviousQuestion);
              }
            } else if (data.error) {
              markModelError(outputEl, data.error, data);
            } else if (data.detail) {
              markModelError(outputEl, getApiErrorMessage(data), data);
            } else {
              markModelError(outputEl, "Mistral returned an empty response.", data);
            }
            checkAllResponses();
          })
          .catch(error => {
            if (isAbortError(error) || !isActiveQueryRun(queryRunId) || isSkipped("mistralResponse")) return;
            const outputEl = document.getElementById("mistralResponse").querySelector(".collapsible-content");
            markModelError(outputEl, `Mistral error: ${error.message}`, { error: error.message });
            console.error("Error with Mistral:", error);
            checkAllResponses();
          });
      }

      // Anthropic Claude
      if (activeModels.includes("Anthropic")) {
        const useOwnKeys = document.getElementById("useOwnKeysSwitch").checked;
        const payload = {
          question: question,
          deep_search: deepSearchFlag,
          system_prompt: effectiveSystemPrompt,
          mode: mode,
          model: document.getElementById("claudeModelSelect").value,
          id_token: validIdToken,
          useOwnKeys: useOwnKeys,
          usage_run_key: usageRun.key
        };
        if (attachmentsPayload.length) payload.attachments = attachmentsPayload;
        attachConversationContext(payload);

        if (!useOwnKeys) {
          if (validIdToken) {
            payload.id_token = validIdToken; // Nimm das frische Token Variable
          } else {
            // Fallback, falls kein User eingeloggt ist aber auch keine Keys da sind
            payload.api_key = localStorage.getItem("anthropicKey") || "";
          }
        } else {
          if (!validateUserKey("anthropicKey")) {
            const outputEl = document.getElementById("claudeResponse").querySelector(".collapsible-content");
            outputEl.innerHTML = "Please log in and enter your own API key.";
            alert("Please enter a valid Anthropic API Key.");
            currentQueryController?.abort();
            markPendingQueryResponsesCanceled();
            finishQueryRun(queryRunId);
            return;
          }
          payload.api_key = localStorage.getItem("anthropicKey");
        }

        const claudeStreamRenderer = createStreamRenderer(
          document.getElementById("claudeResponse").querySelector(".collapsible-content"),
          () => isActiveQueryRun(queryRunId)
        );
        streamSSERequest('/ask_claude', payload, signalFor("claudeResponse"), { "delta": claudeStreamRenderer })
          .then(({ data }) => data)
          .then(data => {
            if (!isActiveQueryRun(queryRunId) || isSkipped("claudeResponse")) return;
            updateUsageDisplayFromData(data);
            const outputEl = document.getElementById("claudeResponse").querySelector(".collapsible-content");
            if (data.response) {
              markModelSuccess(outputEl);
              const responseWithSources = window.renderModelResponseWithSources
                ? window.renderModelResponseWithSources(outputEl, data.response, data.sources || [])
                : data.response;
              if (window.auth.currentUser) {
                window.saveBookmark(question, responseWithSources, "Anthropic", mode, bookmarkPreviousQuestion);
              }
            } else if (data.error) {
              markModelError(outputEl, data.error, data);
            } else if (data.detail) {
              markModelError(outputEl, getApiErrorMessage(data), data);
            } else {
              markModelError(outputEl, "Anthropic returned an empty response.", data);
            }
            checkAllResponses();
          })
          .catch(error => {
            if (isAbortError(error) || !isActiveQueryRun(queryRunId) || isSkipped("claudeResponse")) return;
            const outputEl = document.getElementById("claudeResponse").querySelector(".collapsible-content");
            markModelError(outputEl, `Anthropic error: ${error.message}`, { error: error.message });
            console.error("Error with Anthropic:", error);
            checkAllResponses();
          });
      }

      // Gemini
      if (activeModels.includes("Gemini")) {
        const useOwnKeys = document.getElementById("useOwnKeysSwitch").checked;
        const payload = {
          question: question,
          deep_search: deepSearchFlag,
          system_prompt: effectiveSystemPrompt,
          mode: mode,
          model: document.getElementById("geminiModelSelect").value,
          id_token: validIdToken,
          useOwnKeys: useOwnKeys,
          usage_run_key: usageRun.key
        };
        if (attachmentsPayload.length) payload.attachments = attachmentsPayload;
        attachConversationContext(payload);

        if (!useOwnKeys) {
          if (validIdToken) {
            payload.id_token = validIdToken; // Nimm das frische Token Variable
          } else {
            // Fallback, falls kein User eingeloggt ist aber auch keine Keys da sind
            payload.api_key = localStorage.getItem("geminiKey") || "";
          }
        } else {
          if (!validateUserKey("geminiKey")) {
            const outputEl = document.getElementById("geminiResponse").querySelector(".collapsible-content");
            outputEl.innerHTML = "Please log in and enter your own API key.";
            alert("Please enter a valid Gemini API Key.");
            currentQueryController?.abort();
            markPendingQueryResponsesCanceled();
            finishQueryRun(queryRunId);
            return;
          }
          payload.api_key = localStorage.getItem("geminiKey");
        }

        const geminiStreamRenderer = createStreamRenderer(
          document.getElementById("geminiResponse").querySelector(".collapsible-content"),
          () => isActiveQueryRun(queryRunId)
        );
        streamSSERequest('/ask_gemini', payload, signalFor("geminiResponse"), { "delta": geminiStreamRenderer })
          .then(({ data }) => data)
          .then(data => {
            if (!isActiveQueryRun(queryRunId) || isSkipped("geminiResponse")) return;
            updateUsageDisplayFromData(data);
            const outputEl = document.getElementById("geminiResponse").querySelector(".collapsible-content");
            if (data.response) {
              markModelSuccess(outputEl);
              const responseWithSources = window.renderModelResponseWithSources
                ? window.renderModelResponseWithSources(outputEl, data.response, data.sources || [])
                : data.response;
              if (window.auth.currentUser) {
                window.saveBookmark(question, responseWithSources, "Gemini", mode, bookmarkPreviousQuestion);
              }
            } else if (data.error) {
              markModelError(outputEl, data.error, data);
            } else if (data.detail) {
              markModelError(outputEl, getApiErrorMessage(data), data);
            } else {
              markModelError(outputEl, "Gemini returned an empty response.", data);
            }
            checkAllResponses();
          })
          .catch(error => {
            if (isAbortError(error) || !isActiveQueryRun(queryRunId) || isSkipped("geminiResponse")) return;
            const outputEl = document.getElementById("geminiResponse").querySelector(".collapsible-content");
            markModelError(outputEl, `Gemini error: ${error.message}`, { error: error.message });
            console.error("Error with Gemini:", error);
            checkAllResponses();
          });
      }

      // DeepSeek
      if (activeModels.includes("DeepSeek")) {
        const useOwnKeys = document.getElementById("useOwnKeysSwitch").checked;
        const payload = {
          question: question,
          deep_search: deepSearchFlag,
          system_prompt: effectiveSystemPrompt,
          mode: mode,
          model: document.getElementById("deepseekModelSelect").value,
          id_token: validIdToken,
          useOwnKeys: useOwnKeys,
          usage_run_key: usageRun.key
        };
        if (attachmentsPayload.length) payload.attachments = attachmentsPayload;
        attachConversationContext(payload);

        if (!useOwnKeys) {
          if (validIdToken) {
            payload.id_token = validIdToken; // Nimm das frische Token Variable
          } else {
            // Fallback, falls kein User eingeloggt ist aber auch keine Keys da sind
            payload.api_key = localStorage.getItem("deepseekKey") || "";
          }
        } else {
          if (!validateUserKey("deepseekKey")) {
            const outputEl = document.getElementById("deepseekResponse").querySelector(".collapsible-content");
            outputEl.innerHTML = "Please log in and enter your own API key.";
            alert("Please enter a valid DeepSeek API key.");
            currentQueryController?.abort();
            markPendingQueryResponsesCanceled();
            finishQueryRun(queryRunId);
            return;
          }
          payload.api_key = localStorage.getItem("deepseekKey");
        }

        const deepseekStreamRenderer = createStreamRenderer(
          document.getElementById("deepseekResponse").querySelector(".collapsible-content"),
          () => isActiveQueryRun(queryRunId)
        );
        streamSSERequest('/ask_deepseek', payload, signalFor("deepseekResponse"), { "delta": deepseekStreamRenderer })
          .then(({ data }) => data)
          .then(data => {
            if (!isActiveQueryRun(queryRunId) || isSkipped("deepseekResponse")) return;
            updateUsageDisplayFromData(data);
            const outputEl = document.getElementById("deepseekResponse").querySelector(".collapsible-content");
            if (data.response) {
              markModelSuccess(outputEl);
              const responseWithSources = window.renderModelResponseWithSources
                ? window.renderModelResponseWithSources(outputEl, data.response, data.sources || [])
                : data.response;
              if (window.auth.currentUser) {
                window.saveBookmark(question, responseWithSources, "DeepSeek", mode, bookmarkPreviousQuestion);
              }
            } else if (data.error) {
              markModelError(outputEl, data.error, data);
            } else if (data.detail) {
              markModelError(outputEl, getApiErrorMessage(data), data);
            } else {
              markModelError(outputEl, "DeepSeek returned an empty response.", data);
            }
            checkAllResponses();
          })
          .catch(error => {
            if (isAbortError(error) || !isActiveQueryRun(queryRunId) || isSkipped("deepseekResponse")) return;
            const outputEl = document.getElementById("deepseekResponse").querySelector(".collapsible-content");
            markModelError(outputEl, `DeepSeek error: ${error.message}`, { error: error.message });
            console.error("Fehler bei DeepSeek:", error);
            checkAllResponses();
          });
      }

      // Grok
      if (activeModels.includes("Grok")) {
        const useOwnKeys = document.getElementById("useOwnKeysSwitch").checked;
        const payload = {
          question: question,
          deep_search: deepSearchFlag,
          system_prompt: effectiveSystemPrompt,
          mode: mode,
          model: document.getElementById("grokModelSelect").value,
          id_token: validIdToken,
          useOwnKeys: useOwnKeys,
          usage_run_key: usageRun.key
        };
        if (attachmentsPayload.length) payload.attachments = attachmentsPayload;
        attachConversationContext(payload);

        if (!useOwnKeys) {
          if (validIdToken) {
            payload.id_token = validIdToken; // Nimm das frische Token Variable
          } else {
            // Fallback, falls kein User eingeloggt ist aber auch keine Keys da sind
            payload.api_key = localStorage.getItem("grokKey") || "";
          }
        } else {
          if (!validateUserKey("grokKey")) {
            const outputEl = document.getElementById("grokResponse").querySelector(".collapsible-content");
            outputEl.innerHTML = "Please log in and enter your own API key.";
            alert("Please enter a valid Grok API key.");
            currentQueryController?.abort();
            markPendingQueryResponsesCanceled();
            finishQueryRun(queryRunId);
            return;
          }
          payload.api_key = localStorage.getItem("grokKey");
        }
        const grokStreamRenderer = createStreamRenderer(
          document.getElementById("grokResponse").querySelector(".collapsible-content"),
          () => isActiveQueryRun(queryRunId)
        );
        streamSSERequest('/ask_grok', payload, signalFor("grokResponse"), { "delta": grokStreamRenderer })
          .then(({ data }) => data)
          .then(data => {
            if (!isActiveQueryRun(queryRunId) || isSkipped("grokResponse")) return;
            updateUsageDisplayFromData(data);
            const outputEl = document.getElementById("grokResponse").querySelector(".collapsible-content");
            if (data.response) {
              markModelSuccess(outputEl);
              const responseWithSources = window.renderModelResponseWithSources
                ? window.renderModelResponseWithSources(outputEl, data.response, data.sources || [])
                : data.response;
              if (window.auth.currentUser) {
                window.saveBookmark(question, responseWithSources, "Grok", mode, bookmarkPreviousQuestion);
              }
            } else if (data.error) {
              markModelError(outputEl, data.error, data);
            } else if (data.detail) {
              markModelError(outputEl, getApiErrorMessage(data), data);
            } else {
              markModelError(outputEl, "Grok returned an empty response.", data);
            }
            checkAllResponses();
          })
          .catch(error => {
            if (isAbortError(error) || !isActiveQueryRun(queryRunId) || isSkipped("grokResponse")) return;
            const outputEl = document.getElementById("grokResponse").querySelector(".collapsible-content");
            markModelError(outputEl, `Grok error: ${error.message}`, { error: error.message });
            console.error("Fehler bei Grok:", error);
            checkAllResponses();
          });
      }

    };

  window.isQueryRequestRunning = function () {
    return queryRequestRunning;
  };

  // Modell-Lauf ODER Consensus-Lauf; die Send-Wiring in app-init.js nutzt das,
  // damit der Cancel-Klick nicht durch die Eingabe-Validierung laeuft.
  window.isRunActive = isRunActive;
  // consensus-lifecycle.js meldet Start/Ende der Consensus-Phase hierher.
  window.App.syncSendButtonRunning = syncSendButtonRunning;
})();
