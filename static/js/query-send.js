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
        : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" style="width: 16px; height: 16px;">
          <line x1="22" y1="2" x2="11" y2="13"></line>
          <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
        </svg>`;

      if (!isRunning && typeof window.updateQuestionInputAccess === "function") {
        window.updateQuestionInputAccess();
      }
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
      setSendButtonRunning(false);
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
      const consensusMain = consensusBox?.querySelector(".consensus-main p");
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
      trackAppEvent("app_query_canceled");
    };

    // Senden der Frage an die aktiven Modelle
    window.sendQuestion = async function () {
      if (queryRequestRunning) {
        window.cancelCurrentQuery();
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
      window.lastQuestion = question;  // Speichern in einer globalen Variable (auch von consensus-run.js gelesen)

      if (!question.trim()) {
        alert("Please enter a question.");
        return;
      }

      // Ab dem ersten echten Lauf dockt das Eingabefeld dauerhaft oben an.
      // Der Demo-Pfad nutzt denselben Übergang.
      window.exitHeroMode?.();

      // === DEMO: Früh raus, wenn "Demo" ===
      if (isDemoQuery(question)) {
        trackAppEvent("app_demo_started", {
          selected_models: getSelectedModelCount(),
          agent_mode: typeof window.isAgentModeEnabled === "function" && window.isAgentModeEnabled()
        });
        // optional: Free/Deep Counter im Sidebar auf einen sicheren Dummy setzen
        document.getElementById("freeUsageDisplay").innerHTML = "Free requests: 25 / 25";
        document.getElementById("deepUsageDisplay").innerHTML = "Deep Think: 12 / 12";

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
        alert("File uploads are a Pro feature. Please remove the attachments or upgrade.");
        finishQueryRun(queryRunId);
        return;
      }

      // Metadaten der mitgeschickten Anhänge merken (für Bookmarks, ohne Dateidaten)
      window.lastQuestionAttachmentsMeta = attachmentsPayload.map(function (att) {
        return { name: att.name, mime: att.mime, size: att.size || 0 };
      });

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
      if (document.getElementById("selectDeepSeek")?.checked) {
        const box = document.getElementById("deepseekResponse");
        if (box) modelBoxes.push(box);
      }
      if (document.getElementById("selectGrok")?.checked) {
        const box = document.getElementById("grokResponse");
        if (box) modelBoxes.push(box);
      }

      // 🔸 PHASE 1: UI Feedback setzen
      setAgentModeStatus(modelBoxes.length > 0 ? "running" : "idle");
      window.spinnerHTML = baseSpinnerHTML;
      modelBoxes.forEach(box => {
        delete box.dataset.consensusAnswer;
        delete box.dataset.consensusSources;
        delete box.dataset.responseError;
        box.dataset.responseState = "pending";
        window.setSpinnerEl(box);
      });
      window.currentEvidenceSources = [];
      if (window.renderEvidenceSources) window.renderEvidenceSources([]);

      let queryHadBlockingError = false;
      let queryBlockingErrorMessage = "";

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
        return normalized.error || normalized.detail || normalized.message || fallback;
      }

      function isUsageLimitError(data, message = "") {
        const normalized = unwrapApiError(data);
        const code = String(normalized.error_code || normalized.code || "").toLowerCase();
        const text = String(message || normalized.error || normalized.detail || "").toLowerCase();
        return code.includes("limit")
          || text.includes("usage limit")
          || text.includes("quota")
          || text.includes("used up")
          || text.includes("exhausted");
      }

      function updateUsageDisplayFromData(data) {
        const normalized = unwrapApiError(data);
        if (normalized.free_usage_remaining !== undefined) {
          document.getElementById("freeUsageDisplay").innerHTML =
            "Requests: " + normalized.free_usage_remaining + " / " + window.currentMaxLimit;
        }
        if (normalized.deep_remaining !== undefined) {
          document.getElementById("deepUsageDisplay").innerHTML =
            "Deep Think: " + normalized.deep_remaining + " / " + window.currentDeepLimit;
        }
        if (normalized.is_pro_user !== undefined) {
          window.updateUserTierUI(normalized.is_pro_user === true, !!window.auth.currentUser);
        }
      }

      function markQueryBlockingError(message, data = {}) {
        queryHadBlockingError = true;
        queryBlockingErrorMessage = message || queryBlockingErrorMessage || "The request could not be completed.";
        updateUsageDisplayFromData(data);
        if (isAgentModeEnabled()) {
          setAgentModeStatus("error", queryBlockingErrorMessage);
        }
      }

      function markModelSuccess(outputEl) {
        const box = outputEl?.closest?.(".response-box");
        if (box) {
          delete box.dataset.responseError;
          box.dataset.responseState = "complete";
        }
        if (isAgentModeEnabled()) {
          window.updateAgentModeUI?.();
        }
      }

      function markModelError(outputEl, message, data = {}) {
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

      // Follow-up-Kontext (Pro): genau eine Ebene — vorherige Frage + Konsens-
      // Text. consume() liefert den Payload nur bei aktiviertem Chip und räumt
      // den State auf; reset() lässt auch die Affordance verschwinden, weil
      // dieser Lauf den alten Konsens ersetzt. Erst hier (nach den frühen
      // Abbruch-Pfaden), damit ein abgebrochener Send den Chip nicht frisst.
      const followupContext = window.App.followup?.consume?.() || null;
      if (followupContext) {
        // consume() hat den Lauf schon als Follow-up markiert (kein reset,
        // sonst ginge das In-Flight-Flag verloren).
        trackAppEvent("app_followup_sent");
      } else {
        // Frische Frage ersetzt den alten Konsens: Affordance/Flag verwerfen.
        window.App.followup?.reset?.();
      }

      // Wir rufen /prepare IMMER auf, damit Wetter-Infos etc. injiziert werden können.
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

        // Nur anhängen, wenn wirklich vorhanden
        if (idToken) {
          preparePayload.id_token = idToken;
        }
        if (followupContext) {
          preparePayload.context = followupContext;
        }

        const prepareResp = await fetch("/prepare", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(preparePayload),
          signal: querySignal
        });

        const prepareData = await prepareResp.json();

        if (prepareResp.ok && prepareData.system_prompt) {
          // ÄNDERUNG: Wir prüfen, ob der neue Prompt ANDERS ist als der alte.
          if (prepareData.system_prompt !== storedSystemPrompt) {
            effectiveSystemPrompt = prepareData.system_prompt;
          }
          // Wenn sie gleich sind: Nichts tun, nichts loggen (effectiveSystemPrompt ist ja schon gesetzt).
        } else {
          if (isUsageLimitError(prepareData)) {
            const message = getApiErrorMessage(prepareData, "Usage limit reached.");
            modelBoxes.forEach(box => {
              const outputEl = box.querySelector(".collapsible-content");
              if (outputEl) markModelError(outputEl, message, prepareData);
            });
            finishQueryRun(queryRunId);
            setAgentModeStatus("error", message);
            setConsensusGate(true);
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

      window.spinnerHTML = baseSpinnerHTML;
      modelBoxes.forEach(box => window.setSpinnerEl(box));


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
      if (document.getElementById("selectDeepSeek").checked) activeModels.push("DeepSeek");
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
          active_count: 1,
          deep_search: deepSearchFlag,      // oben definiert
          system_prompt: effectiveSystemPrompt,
          mode: mode,
          model: document.getElementById("openaiModelSelect").value,
          id_token: validIdToken,
          useOwnKeys: useOwnKeys
        };
        if (attachmentsPayload.length) payload.attachments = attachmentsPayload;
        if (followupContext) payload.context = followupContext;


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
        streamSSERequest('/ask_openai', payload, querySignal, { "delta": openaiStreamRenderer })
          .then(({ ok, status, data }) => {
            if (!ok) {
              // FastAPI gibt "detail" zurück; wir normalisieren auf eine Error-Message
              const msg =
                (data && (data.error || data.detail)) ||
                `OpenAI HTTP ${status}`;
              throw new Error(msg);
            }

            return data;
          })
          .then((data) => {
            if (!isActiveQueryRun(queryRunId)) return;
            const outputEl = document
              .getElementById("openaiResponse")
              .querySelector(".collapsible-content");

            if (data.response) {
              markModelSuccess(outputEl);
              const responseWithSources = window.renderModelResponseWithSources
                ? window.renderModelResponseWithSources(outputEl, data.response, data.sources || [])
                : data.response;
              if (window.auth.currentUser) {
                window.saveBookmark(question, responseWithSources, "OpenAI", mode);
              }
            } else if (data.error) {
              markModelError(outputEl, data.error, data);
            } else if (data.detail) {
              // Falls dein Backend irgendwo noch "detail" liefert
              markModelError(outputEl, getApiErrorMessage(data), data);
            } else {
              markModelError(outputEl, "Unexpected empty response from server.", data);
            }

            const freeRemaining = (data.free_usage_remaining !== undefined) ? data.free_usage_remaining : 0;
            const deepRemaining = (data.deep_remaining !== undefined) ? data.deep_remaining : 0;

            const isProConfirmed = (data.is_pro_user === true);

            // 2. Auth Status prüfen (WICHTIG, damit der 2. Parameter stimmt)
            const isLoggedIn = !!window.auth.currentUser;

            // 3. UI updaten
            window.updateUserTierUI(isProConfirmed, isLoggedIn);

            // Text setzen mit den neuen Variablen window.currentMaxLimit und window.currentDeepLimit
            document.getElementById("freeUsageDisplay").innerHTML =
              "Requests: " + freeRemaining + " / " + window.currentMaxLimit;

            document.getElementById("deepUsageDisplay").innerHTML =
              "Deep Think: " + deepRemaining + " / " + window.currentDeepLimit;

            checkAllResponses();
          })
          .catch((error) => {
            if (isAbortError(error) || !isActiveQueryRun(queryRunId)) return;
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
          active_count: 1,
          system_prompt: effectiveSystemPrompt,
          mode: mode,
          model: document.getElementById("mistralModelSelect").value,
          id_token: validIdToken,
          useOwnKeys: useOwnKeys
        };
        if (attachmentsPayload.length) payload.attachments = attachmentsPayload;
        if (followupContext) payload.context = followupContext;

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
        streamSSERequest('/ask_mistral', payload, querySignal, { "delta": mistralStreamRenderer })
          .then(({ data }) => data)
          .then(data => {
            if (!isActiveQueryRun(queryRunId)) return;
            let outputEl = document.getElementById("mistralResponse").querySelector(".collapsible-content");
            if (data.response) {
              markModelSuccess(outputEl);
              const responseWithSources = window.renderModelResponseWithSources
                ? window.renderModelResponseWithSources(outputEl, data.response, data.sources || [])
                : data.response;

              if (window.auth.currentUser) {
                window.saveBookmark(question, responseWithSources, "Mistral", mode);
              }
            } else if (data.error) {
              markModelError(outputEl, data.error, data);
            } else if (data.detail) {
              markModelError(outputEl, getApiErrorMessage(data), data);
            } else {
              markModelError(outputEl, "No response received. The model may have timed out — please try again.", data);
            }
            // Nutze Fallback-Werte, falls data.free_usage_remaining oder data.deep_remaining undefined sind
            const freeRemaining = (data.free_usage_remaining !== undefined) ? data.free_usage_remaining : 0;
            const deepRemaining = (data.deep_remaining !== undefined) ? data.deep_remaining : 0;

            const isProConfirmed = (data.is_pro_user === true);

            // 2. Auth Status prüfen (WICHTIG, damit der 2. Parameter stimmt)
            const isLoggedIn = !!window.auth.currentUser;

            // 3. UI updaten
            window.updateUserTierUI(isProConfirmed, isLoggedIn);

            // Text setzen mit den neuen Variablen window.currentMaxLimit und window.currentDeepLimit
            document.getElementById("freeUsageDisplay").innerHTML =
              "Requests: " + freeRemaining + " / " + window.currentMaxLimit;

            document.getElementById("deepUsageDisplay").innerHTML =
              "Deep Think: " + deepRemaining + " / " + window.currentDeepLimit;

            checkAllResponses();
          })
          .catch(error => {
            if (isAbortError(error) || !isActiveQueryRun(queryRunId)) return;
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
          active_count: 1,
          system_prompt: effectiveSystemPrompt,
          mode: mode,
          model: document.getElementById("claudeModelSelect").value,
          id_token: validIdToken,
          useOwnKeys: useOwnKeys
        };
        if (attachmentsPayload.length) payload.attachments = attachmentsPayload;
        if (followupContext) payload.context = followupContext;

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
        streamSSERequest('/ask_claude', payload, querySignal, { "delta": claudeStreamRenderer })
          .then(({ data }) => data)
          .then(data => {
            if (!isActiveQueryRun(queryRunId)) return;
            const outputEl = document.getElementById("claudeResponse").querySelector(".collapsible-content");
            if (data.response) {
              markModelSuccess(outputEl);
              const responseWithSources = window.renderModelResponseWithSources
                ? window.renderModelResponseWithSources(outputEl, data.response, data.sources || [])
                : data.response;
              if (window.auth.currentUser) {
                window.saveBookmark(question, responseWithSources, "Anthropic", mode);
              }
            } else if (data.error) {
              markModelError(outputEl, data.error, data);
            } else if (data.detail) {
              markModelError(outputEl, getApiErrorMessage(data), data);
            } else {
              markModelError(outputEl, "No response received. The model may have timed out — please try again.", data);
            }
            // Nutze Fallback-Werte, falls data.free_usage_remaining oder data.deep_remaining undefined sind
            const freeRemaining = (data.free_usage_remaining !== undefined) ? data.free_usage_remaining : 0;
            const deepRemaining = (data.deep_remaining !== undefined) ? data.deep_remaining : 0;

            const isProConfirmed = (data.is_pro_user === true);

            // 2. Auth Status prüfen (WICHTIG, damit der 2. Parameter stimmt)
            const isLoggedIn = !!window.auth.currentUser;

            // 3. UI updaten
            window.updateUserTierUI(isProConfirmed, isLoggedIn);

            // Text setzen mit den neuen Variablen window.currentMaxLimit und window.currentDeepLimit
            document.getElementById("freeUsageDisplay").innerHTML =
              "Requests: " + freeRemaining + " / " + window.currentMaxLimit;

            document.getElementById("deepUsageDisplay").innerHTML =
              "Deep Think: " + deepRemaining + " / " + window.currentDeepLimit;

            checkAllResponses();
          })
          .catch(error => {
            if (isAbortError(error) || !isActiveQueryRun(queryRunId)) return;
            console.error("Error with Anthropic:", error);
            checkAllResponses();
          });
      }

      // Gemini
      if (activeModels.includes("Gemini")) {
        const useOwnKeys = document.getElementById("useOwnKeysSwitch").checked;
        const payload = {
          question: question,
          active_count: 1,
          deep_search: deepSearchFlag,
          system_prompt: effectiveSystemPrompt,
          mode: mode,
          model: document.getElementById("geminiModelSelect").value,
          id_token: validIdToken,
          useOwnKeys: useOwnKeys
        };
        if (attachmentsPayload.length) payload.attachments = attachmentsPayload;
        if (followupContext) payload.context = followupContext;

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
        streamSSERequest('/ask_gemini', payload, querySignal, { "delta": geminiStreamRenderer })
          .then(({ data }) => data)
          .then(data => {
            if (!isActiveQueryRun(queryRunId)) return;
            const outputEl = document.getElementById("geminiResponse").querySelector(".collapsible-content");
            if (data.response) {
              markModelSuccess(outputEl);
              const responseWithSources = window.renderModelResponseWithSources
                ? window.renderModelResponseWithSources(outputEl, data.response, data.sources || [])
                : data.response;
              if (window.auth.currentUser) {
                window.saveBookmark(question, responseWithSources, "Gemini", mode);
              }
            } else if (data.error) {
              markModelError(outputEl, data.error, data);
            } else if (data.detail) {
              markModelError(outputEl, getApiErrorMessage(data), data);
            } else {
              markModelError(outputEl, "No response received. The model may have timed out — please try again.", data);
            }
            // Nutze Fallback-Werte, falls data.free_usage_remaining oder data.deep_remaining undefined sind
            const freeRemaining = (data.free_usage_remaining !== undefined) ? data.free_usage_remaining : 0;
            const deepRemaining = (data.deep_remaining !== undefined) ? data.deep_remaining : 0;

            const isProConfirmed = (data.is_pro_user === true);

            // 2. Auth Status prüfen (WICHTIG, damit der 2. Parameter stimmt)
            const isLoggedIn = !!window.auth.currentUser;

            // 3. UI updaten
            window.updateUserTierUI(isProConfirmed, isLoggedIn);

            // Text setzen mit den neuen Variablen window.currentMaxLimit und window.currentDeepLimit
            document.getElementById("freeUsageDisplay").innerHTML =
              "Requests: " + freeRemaining + " / " + window.currentMaxLimit;

            document.getElementById("deepUsageDisplay").innerHTML =
              "Deep Think: " + deepRemaining + " / " + window.currentDeepLimit;

            checkAllResponses();
          })
          .catch(error => {
            if (isAbortError(error) || !isActiveQueryRun(queryRunId)) return;
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
          active_count: 1,
          system_prompt: effectiveSystemPrompt,
          mode: mode,
          model: document.getElementById("deepseekModelSelect").value,
          id_token: validIdToken,
          useOwnKeys: useOwnKeys
        };
        if (attachmentsPayload.length) payload.attachments = attachmentsPayload;
        if (followupContext) payload.context = followupContext;

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
        streamSSERequest('/ask_deepseek', payload, querySignal, { "delta": deepseekStreamRenderer })
          .then(({ data }) => data)
          .then(data => {
            if (!isActiveQueryRun(queryRunId)) return;
            const outputEl = document.getElementById("deepseekResponse").querySelector(".collapsible-content");
            if (data.response) {
              markModelSuccess(outputEl);
              const responseWithSources = window.renderModelResponseWithSources
                ? window.renderModelResponseWithSources(outputEl, data.response, data.sources || [])
                : data.response;
              if (window.auth.currentUser) {
                window.saveBookmark(question, responseWithSources, "DeepSeek", mode);
              }
            } else if (data.error) {
              markModelError(outputEl, data.error, data);
            } else if (data.detail) {
              markModelError(outputEl, getApiErrorMessage(data), data);
            } else {
              markModelError(outputEl, "No response received. The model may have timed out — please try again.", data);
            }
            // Nutze Fallback-Werte, falls data.free_usage_remaining oder data.deep_remaining undefined sind
            const freeRemaining = (data.free_usage_remaining !== undefined) ? data.free_usage_remaining : 0;
            const deepRemaining = (data.deep_remaining !== undefined) ? data.deep_remaining : 0;

            const isProConfirmed = (data.is_pro_user === true);

            // 2. Auth Status prüfen (WICHTIG, damit der 2. Parameter stimmt)
            const isLoggedIn = !!window.auth.currentUser;

            // 3. UI updaten
            window.updateUserTierUI(isProConfirmed, isLoggedIn);

            // Text setzen mit den neuen Variablen window.currentMaxLimit und window.currentDeepLimit
            document.getElementById("freeUsageDisplay").innerHTML =
              "Requests: " + freeRemaining + " / " + window.currentMaxLimit;

            document.getElementById("deepUsageDisplay").innerHTML =
              "Deep Think: " + deepRemaining + " / " + window.currentDeepLimit;

            checkAllResponses();
          })
          .catch(error => {
            if (isAbortError(error) || !isActiveQueryRun(queryRunId)) return;
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
          active_count: 1,
          system_prompt: effectiveSystemPrompt,
          mode: mode,
          model: document.getElementById("grokModelSelect").value,
          id_token: validIdToken,
          useOwnKeys: useOwnKeys
        };
        if (attachmentsPayload.length) payload.attachments = attachmentsPayload;
        if (followupContext) payload.context = followupContext;

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
        streamSSERequest('/ask_grok', payload, querySignal, { "delta": grokStreamRenderer })
          .then(({ data }) => data)
          .then(data => {
            if (!isActiveQueryRun(queryRunId)) return;
            const outputEl = document.getElementById("grokResponse").querySelector(".collapsible-content");
            if (data.response) {
              markModelSuccess(outputEl);
              const responseWithSources = window.renderModelResponseWithSources
                ? window.renderModelResponseWithSources(outputEl, data.response, data.sources || [])
                : data.response;
              if (window.auth.currentUser) {
                window.saveBookmark(question, responseWithSources, "Grok", mode);
              }
            } else if (data.error) {
              markModelError(outputEl, data.error, data);
            } else if (data.detail) {
              markModelError(outputEl, getApiErrorMessage(data), data);
            } else {
              markModelError(outputEl, "No response received. The model may have timed out — please try again.", data);
            }
            // Nutze Fallback-Werte, falls data.free_usage_remaining oder data.deep_remaining undefined sind
            const freeRemaining = (data.free_usage_remaining !== undefined) ? data.free_usage_remaining : 0;
            const deepRemaining = (data.deep_remaining !== undefined) ? data.deep_remaining : 0;

            const isProConfirmed = (data.is_pro_user === true);

            // 2. Auth Status prüfen (WICHTIG, damit der 2. Parameter stimmt)
            const isLoggedIn = !!window.auth.currentUser;

            // 3. UI updaten
            window.updateUserTierUI(isProConfirmed, isLoggedIn);

            // Text setzen mit den neuen Variablen window.currentMaxLimit und window.currentDeepLimit
            document.getElementById("freeUsageDisplay").innerHTML =
              "Requests: " + freeRemaining + " / " + window.currentMaxLimit;

            document.getElementById("deepUsageDisplay").innerHTML =
              "Deep Think: " + deepRemaining + " / " + window.currentDeepLimit;

            checkAllResponses();
          })
          .catch(error => {
            if (isAbortError(error) || !isActiveQueryRun(queryRunId)) return;
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
})();
