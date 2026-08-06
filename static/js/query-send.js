// =====================================================================
// query-send.js
// Query send only: window.sendQuestion fans the question out to the
// selected providers (/prepare + /ask_*), streams each answer, updates
// usage/tier UI, and triggers auto-consensus. Plus the query run-state
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

  // Local gate wrapper; index.html keeps its own for init/clearResponseBoxes.
  function setConsensusGate(disabled) {
    consensusLifecycle.setGate(disabled);
  }

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
      setConsensusGate(true);
      window.hideConsensusOutput?.();
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

      const question = document.getElementById("questionInput").value;
      const followupRequested = window.App.followup?.isArmed?.() === true;
      window.lastQuestion = question;  // Speichern in einer globalen Variable (auch von consensus-run.js gelesen)

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

      // Ab dem ersten echten Lauf wird die Seite zum Thread: Frage oben,
      // Composer unten. Der Demo-Pfad nutzt denselben Übergang.
      window.exitHeroMode?.();
      // Bei einem Follow-up bleibt der bisherige Turn waehrend /prepare noch
      // unangetastet. Erst wenn der Lauf wirklich fortgesetzt wird, archivieren
      // wir ihn und setzen die neue Frage darunter.
      if (!followupRequested) {
        window.App.setThreadQuestion?.(question);
      }

      // Der gefuehrte Lauf beginnt HIER, nicht erst beim Fan-out: zwischen
      // Klick und erster Modellantwort liegen Validierung und /prepare, und
      // genau in dieser Luecke soll der Nutzer schon sehen, dass etwas laeuft.
      window.App?.consensusPipeline?.onPrepare?.();

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

      // clearResponseBoxes();
      consensusGenerated = false;
      window.App.setAppTitle(question);

      trackAppEvent("app_query_started", {
        mode,
        selected_models: getSelectedModelCount(),
        custom_credentials: document.getElementById("useOwnKeysSwitch")?.checked === true,
        logged_in: !!window.auth?.currentUser,
        agent_mode: typeof window.isAgentModeEnabled === "function" && window.isAgentModeEnabled(),
        auto_consensus: document.getElementById("autoConsensusToggle")?.checked === true
      });

      currentQueryRunId++;
      const queryRunId = currentQueryRunId;
      currentQueryController = new AbortController();
      const querySignal = currentQueryController.signal;
      queryRequestRunning = true;
      setSendButtonRunning(true);
      // Keep consensus unavailable until the current model run produces enough complete answers.
      setConsensusGate(true);
      // Bei jeder neuen Frage den Konsens-Bereich wieder ausblenden.
      window.hideConsensusOutput?.();

      // 1. Definiere useOwnKeys frühzeitig
      const useOwnKeys = document.getElementById("useOwnKeysSwitch").checked;

      // --- NEU: Frisches Token holen ---
      let validIdToken = null;

      if (window.auth && window.auth.currentUser) {
        try {
          // true erzwingt Refresh, false (Standard) nimmt Cache wenn gültig.
          // false reicht meistens, aber bei Fehlern ist das SDK smart genug.
          validIdToken = await window.auth.currentUser.getIdToken();

          // Optional: LocalStorage updaten, damit er nicht komplett asynchron läuft
          localStorage.setItem("id_token", validIdToken);
        } catch (e) {
          console.error("Fehler beim Abrufen des frischen Tokens:", e);
          // Fallback: Versuche es trotzdem mit dem alten Token aus dem Storage, falls vorhanden
          validIdToken = localStorage.getItem("id_token");
        }
      }

      if (!validIdToken) {
        alert(useOwnKeys
          ? "Please log in before using your own API keys."
          : "Please log in before sending a question.");
        finishQueryRun(queryRunId);
        setConsensusGate(true);
        return;
      }

      try {
        const reconciledTurn = await window.App.chatSession?.inspectPendingTurn?.({
          idToken: validIdToken,
          signal: querySignal
        });
        if (reconciledTurn?.status === "completed") {
          finishQueryRun(queryRunId);
          await window.getConsensus?.("replay");
          return;
        }
      } catch (error) {
        if (isAbortError(error) || !isActiveQueryRun(queryRunId)) return;
        finishQueryRun(queryRunId);
        setConsensusGate(true);
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
          setConsensusGate(true);
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
        window.App.setModelSelectionState?.("deepseekResponse", false, {
          persist: false,
          syncCheckbox: true,
          animate: false
        });
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
      if (modelBoxes.length === 0) {
        await releaseReservedUsageRun();
        finishQueryRun(queryRunId);
        setConsensusGate(true);
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
        window.App.followup?.reset?.();
        if (followupRequested) window.App.setThreadQuestion?.(question);
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
            setConsensusGate(true);
            window.App.usageLimit?.showTemporaryStorageBusy?.();
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
            setConsensusGate(true);
            window.App.usageRun?.clear?.();
            window.App.usageLimit?.show?.({
              data: prepareData,
              source: "prepare",
              phase: "prepare"
            });
            // Der Composer bleibt bedienbar: die Frage steht noch im Feld
            // (geleert wird erst nach diesem Zweig), und der Nutzer soll sie
            // nach dem Reset unveraendert abschicken koennen. Bei einem
            // Follow-up hat consume() den Chip aber schon verbraucht — ohne
            // diese Ruecknahme sperrt syncInputLock das Feld zu, in dem die
            // Frage laut Karte noch stehen soll.
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
            throw new Error("The conversation turn could not be prepared. Please retry.");
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
        setConsensusGate(true);
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
        window.App.followup?.archiveCurrentExchange?.();
        window.App.setThreadQuestion?.(question);
        // Der archivierte Turn ist jetzt die sichtbare alte Antwort. Das Live-
        // Renderziel wird fuer den neuen Consensus frei und darf nicht unter
        // der neuen Frage noch einmal den alten Text zeigen.
        window.hideConsensusOutput?.();
        const liveConsensusBody = window.App.consensusBodyEl?.();
        if (liveConsensusBody) liveConsensusBody.innerHTML = "";
        window.resetConsensusInsights?.();
      }

      // Die Frage steht jetzt im Thread-Kopf; der Composer wird frei für die
      // nächste. Erst hier leeren — die frühen Abbruch-Pfade (Login, Limit)
      // sollen den getippten Text nicht verlieren.
      const questionInputEl = document.getElementById("questionInput");
      if (questionInputEl) {
        questionInputEl.value = "";
        questionInputEl.dispatchEvent(new Event("input", { bubbles: true }));
        window.syncDemoChipState?.();
      }

      // /prepare refreshes the authoritative user tier. That refresh restores
      // persisted model selections, so reassert the attachment invariant
      // before answer counting and request fan-out.
      enforceDeepSeekAttachmentBlock();

      // Only now may the live response tree be reused. Until /prepare and the
      // optional context binding have succeeded, it remains the readable
      // completed predecessor and must survive credential/202/network errors.
      window.spinnerHTML = baseSpinnerHTML;
      modelBoxes.forEach(box => {
        delete box.dataset.consensusAnswer;
        delete box.dataset.consensusSources;
        delete box.dataset.responseError;
        delete box.dataset.responseSkipped;
        box.dataset.responseState = "pending";
        window.setSpinnerEl(box);
      });
      window.currentEvidenceSources = [];
      window.renderEvidenceSources?.([]);


      const deepSearchActive = document.getElementById("deepSearchToggle").checked;

      // Konsens unterbinden, solange noch Antworten fehlen
      setConsensusGate(true);
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
      document.getElementById("consensusResponse").querySelector("p").innerHTML = "";
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
            }
          }
          if (queryHadBlockingError) {
            setAgentModeStatus("error", queryBlockingErrorMessage);
            trackAppEvent("app_query_completed", {
              status: "error",
              selected_models: totalActive
            });
            setConsensusGate(true);
            return;
          }

          setAgentModeStatus("complete");
          trackAppEvent("app_query_completed", {
            status: "success",
            selected_models: totalActive
          });

          // Konsens läuft jetzt immer automatisch – außer er ist in den
          // Einstellungen deaktiviert. Erst wenn ALLE Antworten fertig sind
          // (inkl. Agent Mode) und genug Antworten vorliegen, blenden wir den
          // rahmenlosen Konsens-Bereich sanft ein und starten die Synthese.
          const autoConsensusOn = document.getElementById("autoConsensusToggle")?.checked !== false;
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
        setConsensusGate(true);
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

      // Echte Anhänge bleiben nach dem Senden sichtbar (z. B. für Folgefragen zum
      // selben Dokument). Nur Vorschau-Chips aus früher geladenen Bookmarks
      // gehören nicht zur neuen Frage und werden entfernt.
      const hadPreviewChips = (window.pendingAttachments || []).some(att => att.previewOnly);
      if (hadPreviewChips) {
        window.pendingAttachments = (window.pendingAttachments || []).filter(att => !att.previewOnly);
        if (typeof window.renderAttachmentChips === "function") {
          window.renderAttachmentChips();
        }
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
