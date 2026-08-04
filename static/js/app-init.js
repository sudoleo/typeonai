// =====================================================================
// app-init.js
// App initialisation / orchestration. This is the remaining initApp()
// bootstrap after the feature clusters were extracted into their own
// modules: theme, usage/limits + user status, response-box UI toggles,
// sidebar/layout, modals, tooltips, evidence rendering, API-key test and
// all the DOM event wiring that glues the window.* contracts together.
//
// MUST load last (after all other static/js modules, firebase.js and
// demo.js): it aliases window.App.* and many window.* helpers at
// initApp() time and wires the document's event listeners. Loaded via a
// deferred <script> at the end of <body> in templates/index.html, so it
// runs after every earlier module in document order.
//
// Server config that used to live inline as a Jinja value is now bridged
// through window.FREE_LIMIT (run limit, set in the Jinja <head> config block).
// =====================================================================

      (function () {
        function initApp() {
        // --------------------------
        // Dark/Light Mode Initialisierung
        // --------------------------
        // Zugriff auf die Checkboxen:

        // Global verfügbar für alle Funktionen
        window.spinnerHTML = `
          <span class="thinking-wrap" role="status" aria-live="polite" aria-busy="true">
            <span class="thinking typing-indicator" data-text="Typing" aria-label="Typing">Typing<span class="typing-dots" aria-hidden="true"><span>.</span><span>.</span><span>.</span></span></span>
          </span>
        `;
        window.consensusSpinnerHTML = `
          <span class="thinking-wrap consensus-thinking-wrap" role="status" aria-live="polite" aria-busy="true">
            <span class="consensus-loader" aria-hidden="true"><span></span><span></span><span></span><span></span></span>
            <span class="thinking consensus-thinking">Synthesizing consensus</span>
          </span>
        `;
        // Der Differences-Judge liefert JSON, das erst am Ende sichtbar wird.
        // Frueher stand hier erst eine eigene Fortschrittsleiste, dann noch das
        // Wort "Comparing responses". Beides sagte, was der gefuehrte Lauf
        // ueber dem Thread ohnehin schon ansagt ("Checking for contradictions").
        // Es gibt genau EINE Fortschrittsanzeige, also bleibt hier nichts.
        window.consensusDifferencesSpinnerHTML = "";

        // Geteilte Config + Helfer kommen aus static/js/app-core.js (window.App).
        // Lokale Aliase, damit die bestehenden Aufrufstellen in initApp
        // unverändert bleiben (Übergangsbus, siehe app-core.js).
        const {
          modelPrefs,
          deepThinkModelLabels,
          getModelOptionLabel,
          getSelectedModelCount,
          trackAppEvent,
          showPopup
        } = window.App;

        // Agent Mode ist nach static/js/agent-mode.js ausgelagert; lokale Aliase
        // für die bestehenden Aufrufstellen in initApp (Übergangsbus).
        const isAgentModeEnabled = window.isAgentModeEnabled;
        const setAgentMode = window.setAgentMode;
        const setAgentModeStatus = window.setAgentModeStatus;
        const updateAgentModeUI = window.updateAgentModeUI;

        // Model-Picker + Modell-Auswahl sind nach static/js/model-picker.js
        // ausgelagert; lokale Aliase für das in initApp verbliebene Wiring
        // (Event-Listener + Init-Aufrufe). restoreModelSelections und
        // syncCustomModelPickers liegen direkt auf window.
        const setModelSelectionState = window.App.setModelSelectionState;
        const openModelPicker = window.App.openModelPicker;
        const collapseExpandedModelPicker = window.App.collapseExpandedModelPicker;
        const initCustomModelPicker = window.App.initCustomModelPicker;

        // Tier-/Pro-UI ist nach static/js/user-tier.js ausgelagert; lokale Aliase
        // für die Aufrufstellen in initApp.
        const updateUserTierUI = window.updateUserTierUI;
        const updatePremiumModelsState = window.updatePremiumModelsState;
        // Brücke: user-tier.js nutzt updateDeepThinkText (gehoistete Fn-Decl in initApp).
        window.App.updateDeepThinkText = updateDeepThinkText;

        const deepSearchToggle = document.getElementById("deepSearchToggle");

        // Auslesen des aktuellen Zustands (true, wenn aktiviert, sonst false):
        const deepSearchActive = deepSearchToggle.checked;
        const modeToggles = Array.from(document.querySelectorAll(".theme-toggle"));

        function applyTheme(theme) {
          const isDark = theme === "dark";
          if (theme === "dark") {
            document.body.classList.add("dark-mode");
          } else {
            document.body.classList.remove("dark-mode");
          }

          modeToggles.forEach(toggle => {
            toggle.classList.toggle("is-dark", isDark);
            toggle.setAttribute("aria-pressed", String(isDark));
            toggle.setAttribute("aria-label", isDark ? "Switch to light mode" : "Switch to dark mode");
            toggle.title = isDark ? "Switch to light mode" : "Switch to dark mode";
          });
        }

        function setSpinner(boxId) {
          const box = document.getElementById(boxId);
          if (!box || box.classList.contains("excluded") || box.style.display === "none") return;
          const p = box.querySelector(".collapsible-content");
          if (p) p.innerHTML = window.spinnerHTML;
        }

        function setSpinnersForActive() {
          ["openaiResponse", "mistralResponse", "claudeResponse", "geminiResponse", "deepseekResponse", "grokResponse"]
            .forEach(setSpinner);
        }

        // Markdown-Rendering + SSE-Streaming-Helfer sind nach
        // static/js/markdown-stream.js ausgelagert (window.injectMarkdown,
        // window.createStreamRenderer, window.streamSSERequest). Lokale Aliase,
        // damit die bestehenden Aufrufstellen in initApp unverändert bleiben.
        const injectMarkdown = window.injectMarkdown;
        const createStreamRenderer = window.createStreamRenderer;
        const streamSSERequest = window.streamSSERequest;

        // Quellen-/Evidence-Handling ist nach static/js/sources.js ausgelagert
        // (window.linkifySourceTags, window.mergeEvidenceSources, window.rewriteSourceTags,
        // window.registerResponseSources, window.prepareResponseSources,
        // window.renderModelResponseWithSources). Alle Aufrufer nutzen bereits window.*,
        // daher keine lokalen Aliase noetig.

        // Lese gespeicherten Wert aus; wenn keiner vorhanden, verwende die Systempräferenz:
        let storedTheme = localStorage.getItem("theme");
        if (!storedTheme) {
          storedTheme = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
          localStorage.setItem("theme", storedTheme);
        }
        applyTheme(storedTheme);

        // Schalte den Dark Mode um und speichere die Auswahl:
        modeToggles.forEach(toggle => {
          toggle.addEventListener("click", function () {
            const newTheme = document.body.classList.contains("dark-mode") ? "light" : "dark";
            localStorage.setItem("theme", newTheme);
            applyTheme(newTheme);
            trackAppEvent("app_theme_changed", { theme: newTheme });
          });
        });

        window.App.consensusLifecycle.initAutoConsensusToggle();

        const questionInput = document.getElementById("questionInput");
        const defaultQuestionPlaceholder = "Enter your question";
        const lockedQuestionPlaceholder = "Sign in to start asking questions for free.";

        // Der Composer beginnt kompakt, waechst mit jeder Textzeile und wird ab
        // der CSS-Maximalhoehe zum intern scrollenden Feld. Die Grenze bleibt im
        // CSS, damit Desktop und Mobile sie unabhaengig setzen koennen.
        function resizeQuestionInput() {
          if (!questionInput) return;

          questionInput.style.height = "0px";
          questionInput.style.overflowY = "hidden";
          const styles = window.getComputedStyle(questionInput);
          const minHeight = Number.parseFloat(styles.minHeight) || 52;
          const maxHeight = Number.parseFloat(styles.maxHeight) || 220;
          const contentHeight = questionInput.scrollHeight;
          const nextHeight = Math.max(minHeight, Math.min(contentHeight, maxHeight));

          questionInput.style.height = `${Math.ceil(nextHeight)}px`;
          questionInput.style.overflowY = contentHeight > maxHeight + 1 ? "auto" : "hidden";
        }

        window.App.resizeQuestionInput = resizeQuestionInput;
        questionInput?.addEventListener("input", resizeQuestionInput);
        window.addEventListener("resize", resizeQuestionInput, { passive: true });
        requestAnimationFrame(resizeQuestionInput);

        function hasVerifiedSession() {
          return Boolean(window.auth?.currentUser?.emailVerified);
        }

        // Angemeldet, aber die E-Mail ist noch nicht bestaetigt. Frueher wurde
        // dieser Zustand sofort ausgeloggt; jetzt bleibt der Nutzer in der App
        // und sieht den Bestaetigungs-Streifen (#verifyBanner). Tippen darf er
        // dabei — nur der Lauf selbst wartet auf die Bestaetigung, damit die
        // getippte Frage den Klick auf den Verify-Link ueberlebt.
        function hasUnverifiedSession() {
          const user = window.auth?.currentUser;
          return Boolean(user && !user.emailVerified);
        }
        window.hasUnverifiedSession = hasUnverifiedSession;

        window.userCanAskQuestions = function () {
          return hasVerifiedSession();
        };

        // Wer tippen darf, ist nicht dasselbe wie wer starten darf.
        window.userCanTypeQuestions = function () {
          return hasVerifiedSession() || hasUnverifiedSession();
        };

        const unverifiedQuestionPlaceholder =
          "Type your question — confirm your e-mail to run it";

        window.updateQuestionInputAccess = function () {
          const canAsk = window.userCanAskQuestions();
          const canType = window.userCanTypeQuestions();
          const selectedModelCount = window.App.getSelectedModelCount?.() || 0;
          const hasMinimumModels = selectedModelCount >= 2;
          const canStartRun = canAsk && hasMinimumModels;
          const sendButton = document.getElementById("sendButton");
          const postDemoLoginPrompt = document.getElementById("postDemoLoginPrompt");

          if (questionInput) {
            questionInput.disabled = !canType;
            questionInput.placeholder = canAsk
              ? defaultQuestionPlaceholder
              : canType
                ? unverifiedQuestionPlaceholder
                : lockedQuestionPlaceholder;
            questionInput.setAttribute("aria-disabled", String(!canType));
          }

          if (sendButton && !sendButton.classList.contains("is-cancel-action")) {
            sendButton.disabled = !canStartRun;
            sendButton.title = !canAsk
              ? canType
                ? "Confirm your e-mail address to start this run"
                : "Sign in to ask questions or use your own API keys"
              : hasMinimumModels
                ? "Send question"
                : "Select at least two models to run consensus";
            sendButton.setAttribute("aria-label", sendButton.title);
            sendButton.setAttribute("aria-disabled", String(!canStartRun));
          }

          if (canAsk && postDemoLoginPrompt) {
            postDemoLoginPrompt.hidden = true;
            postDemoLoginPrompt.classList.remove("is-visible");
          }

          // Ein beantworteter Lauf klappt den Composer fuer alle Nutzer zur
          // naechsten Entscheidung zu (consens.io ist kein Chat). Diese
          // Funktion laeuft nach jedem Auth-Update und wuerde das Feld sonst
          // wieder aufmachen.
          window.App?.followup?.syncInputLock?.();

          return canStartRun;
        };

        window.updateQuestionInputAccess();

        // --- Frage-Entwurf ueber den Bestaetigungs-Umweg retten ------------
        // Wer auf den Link im Postfach klickt, verlaesst diese Seite. Kaeme er
        // mit leerem Feld zurueck, waere die Bestaetigung eine Strafe fuer den
        // Versuch, etwas zu fragen. Gespeichert wird nur, solange der Lauf
        // nicht starten kann — ein verifizierter Nutzer braucht keinen
        // Zwischenspeicher, und alte Fragen sollen nicht wieder auftauchen.
        const QUESTION_DRAFT_KEY = "consensio.questionDraft.v1";

        function storeQuestionDraft(value) {
          try {
            if (value) localStorage.setItem(QUESTION_DRAFT_KEY, value);
            else localStorage.removeItem(QUESTION_DRAFT_KEY);
          } catch (_) {
            // Private/gehaertete Browser: dann eben ohne Rettungsnetz.
          }
        }

        window.App.clearQuestionDraft = function () {
          storeQuestionDraft("");
        };

        // Nur ECHTE Tastenanschlaege sind ein Entwurf. Programmatische
        // input-Events (clearResponseBoxes beim Auth-Wechsel, Demo-Tippen)
        // haetten den Entwurf sonst geloescht, bevor er gebraucht wurde.
        questionInput?.addEventListener("input", event => {
          if (!event.isTrusted) return;
          if (window.userCanAskQuestions()) return;
          storeQuestionDraft(questionInput.value.trim().slice(0, 4000));
        });

        // Idempotent: fuellt nur ein leeres Feld. Wird beim Start UND nach
        // jedem Auth-Wechsel aufgerufen (firebase.js), weil der Auth-Callback
        // den Composer ueber resetLoadedRunAfterLogout() leert — und genau
        // dieser Callback laeuft, wenn der Nutzer nach der Bestaetigung
        // zurueckkommt.
        window.App.restoreQuestionDraft = function () {
          if (!questionInput || questionInput.value.trim()) return;
          let draft = "";
          try {
            draft = localStorage.getItem(QUESTION_DRAFT_KEY) || "";
          } catch (_) {}
          if (!draft) return;
          questionInput.value = draft;
          requestAnimationFrame(resizeQuestionInput);
          window.syncDemoChipState?.();
        };

        window.App.restoreQuestionDraft();

        let mobileInfoPopupTimer = null;

        function isCompactControlLayout() {
          return window.matchMedia("(max-width: 900px)").matches;
        }

        function showMobileInfoPopup(message) {
          if (!message || !isCompactControlLayout()) return;

          const popup = document.getElementById("disclaimerPopup");
          const popupText = popup ? popup.querySelector("p") : null;
          if (!popup || !popupText) return;

          popupText.textContent = message;
          popup.classList.add("show");

          if (mobileInfoPopupTimer) {
            clearTimeout(mobileInfoPopupTimer);
          }
          mobileInfoPopupTimer = setTimeout(() => {
            popup.classList.remove("show");
          }, 7000);
        }
        window.showMobileInfoPopup = showMobileInfoPopup;

        // Der Modus-Erklaerer ist mit seinem (i)-Trigger entfallen: die
        // Modi werden jetzt dort erklaert, wo man sie schaltet (Settings,
        // (+)-Menue) und waehrend sie laufen (gefuehrter Lauf).

        // deepThinkModelLabels stammt aus app-core.js (window.App), siehe Alias oben.
        // Admin-konfigurierbar via /admin (Firestore-Feld deep_think_model),
        // vom Server ueber window.DEEP_THINK_CONSENSUS_MODEL injiziert.
        const DEEP_THINK_CONSENSUS_MODEL = window.DEEP_THINK_CONSENSUS_MODEL || "gemini-3.5-flash";
        let consensusModelBeforeDeepThink = null;

        function syncDeepThinkConsensusModel(isActive) {
          const select = document.getElementById("consensusModelDropdown");
          if (!select) return;

          const target = Array.from(select.options).find(option =>
            option.value === DEEP_THINK_CONSENSUS_MODEL
          );

          if (isActive) {
            if (!target || target.disabled || select.value === target.value) return;
            if (consensusModelBeforeDeepThink === null) {
              consensusModelBeforeDeepThink = select.value;
            }
            select.value = target.value;
          } else {
            const previousValue = consensusModelBeforeDeepThink;
            consensusModelBeforeDeepThink = null;
            if (!previousValue || !target || select.value !== target.value) return;
            const previousOption = Array.from(select.options).find(option =>
              option.value === previousValue && !option.disabled
            );
            if (!previousOption) return;
            select.value = previousOption.value;
          }

          // Kein change-Event: die Deep-Think-Auswahl ist temporaer und darf
          // pref_select_consensus nicht ueberschreiben. Der Custom-Picker muss
          // den nativen Select-Wert trotzdem sofort spiegeln.
          if (typeof window.syncCustomModelPickers === "function") {
            window.syncCustomModelPickers();
          }
        }

        function updateDeepThinkText() {
          const deepSearchToggle = document.getElementById("deepSearchToggle");

          const deepSearchActive = !!deepSearchToggle && deepSearchToggle.checked;
          syncDeepThinkConsensusModel(deepSearchActive);

          const deepthinkDisclaimer = document.getElementById("deepthinkDisclaimer");
          const inputIndicator = document.getElementById("deepThinkInputIndicator");

          // Keep the active mode visible in the lower action row after the (+)
          // menu closes. The checkbox remains the single source of truth,
          // including programmatic tier resets.
          if (inputIndicator) {
            inputIndicator.hidden = !deepSearchActive;
          }

          // -------------------------
          // Suppress the inline Deep Think explainer next to the controls.
          // -------------------------
          const deepText = "";

          // -------------------------
          // Mobile / kleine Screens: Popup
          // -------------------------
          if (isCompactControlLayout()) {
            // Priorität Deep Think: wenn beides an ist, nur Deep Think Text zeigen
            if (deepthinkDisclaimer) deepthinkDisclaimer.style.display = "none";
          } else {
            // -------------------------
            // Desktop: Text RECHTS vom Deep-Think-Toggle
            // -------------------------
            if (deepthinkDisclaimer) {
              if (deepSearchActive) {
                // Priorität:
                // 1) Deep Think aktiv -> Deep Text
                // 2) Nur Web Search aktiv -> Web-Search-Text
                deepthinkDisclaimer.textContent = deepText;
                deepthinkDisclaimer.style.display = "none";
              } else {
                deepthinkDisclaimer.style.display = "none";
              }
            }

            // Den ursprünglichen Web-Search-Disclaimer-Span ausblenden
          }

          // Model-Picker zeigen/verstecken
          const showPickers = !deepSearchActive;
          document.querySelectorAll(".model-picker-wrapper").forEach(el => {
            el.style.display = showPickers ? "inline-flex" : "none";
          });

          const selectedModelLabel = (selectId) => {
            const select = document.getElementById(selectId);
            return getModelOptionLabel(select?.options[select.selectedIndex]) || select?.value || "";
          };
          const openaiModelText = deepSearchActive ? deepThinkModelLabels.OpenAI : selectedModelLabel("openaiModelSelect");
          const mistralModelText = deepSearchActive ? deepThinkModelLabels.Mistral : selectedModelLabel("mistralModelSelect");
          const geminiModelText = deepSearchActive ? deepThinkModelLabels.Gemini : selectedModelLabel("geminiModelSelect");
          const claudeModelText = deepSearchActive ? deepThinkModelLabels.Anthropic : selectedModelLabel("claudeModelSelect");
          const deepseekModelText = deepSearchActive ? deepThinkModelLabels.DeepSeek : selectedModelLabel("deepseekModelSelect");
          const grokModelText = deepSearchActive ? deepThinkModelLabels.Grok : selectedModelLabel("grokModelSelect");

          const setModelText = (id, txt) => {
            const el = document.getElementById(id);
            if (el) {
              el.textContent = txt;
              el.title = `Choose model: ${txt}`;
            }
          };
          setModelText("openaiModelText", openaiModelText);
          setModelText("mistralModelText", mistralModelText);
          setModelText("geminiModelText", geminiModelText);
          setModelText("claudeModelText", claudeModelText);
          setModelText("deepseekModelText", deepseekModelText);
          setModelText("grokModelText", grokModelText);

          if (typeof window.updateAgentModeUI === "function") {
            window.updateAgentModeUI();
          }
        }

        // DEEP THINK & SMOKE TEST LOGIK
        // Deep Search Toggle Text Update (bestehender Code)
        document.getElementById("deepSearchToggle").addEventListener("change", function () {
          updateDeepThinkText(true);
          trackAppEvent("app_deep_think_changed", { enabled: this.checked });
        });

        // --- Pro Modal Referenzen ---
        // Das Modal verkauft nichts: es erklaert nur, warum ein Feature aus ist.
        const proModal = document.getElementById("proFeatureModal");
        const closeProBtn = document.getElementById("closeProModal");
        const keepFreeBtn = document.getElementById("keepFreeBtn");

        // Funktion zum Schließen des Modals
        function closeProModal() {
          proModal.style.display = "none";
        }

        // Event Listener für Schließen-Buttons
        if (closeProBtn) closeProBtn.addEventListener("click", closeProModal);
        if (keepFreeBtn) keepFreeBtn.addEventListener("click", closeProModal);

        // Pro-Modal mit Feature-Name öffnen. Gibt zurück, ob das Modal gezeigt
        // werden konnte, damit Aufrufer sonst auf ein Popup ausweichen können.
        // Der Untertitel nennt beim geklickten Feature den echten Grund: was
        // dieser Lauf kostet (Fallback: generischer Text).
        const PRO_FEATURE_DESCRIPTIONS = {
          "Deep Think": "Deep Think puts the reasoning models on your question. One run costs several times a normal one, so it stays off unless I switch it on for an account.",
          "High Quality mode": "High Quality mode uses the expensive model set for all six answers and for the synthesis. It is the priciest run consens.io can do.",
          "Resolve": "A Resolve round sends the disagreeing models back at each other, which is a second full round of calls on top of the run you already made.",
          "More frequent Consensus Watch checks": "A Watch re-runs your question on a schedule. More Watches and shorter intervals mean more paid runs every single day.",
          "File uploads": "An attached file is read and sent along to every model, which makes all six calls a lot longer, and longer prompts cost more per run.",
        };
        const PRO_FEATURE_DESCRIPTION_FALLBACK = "This one costs a multiple of a normal run, so it stays off by default.";
        window.App.showProFeatureModal = function (featureName) {
          if (window.isUserPro) return false;
          const nameEl = document.getElementById("proModalFeatureName");
          if (nameEl && featureName) nameEl.textContent = featureName;
          const descEl = document.getElementById("proModalDescription");
          if (descEl) {
            descEl.textContent = PRO_FEATURE_DESCRIPTIONS[featureName] || PRO_FEATURE_DESCRIPTION_FALLBACK;
          }
          if (!proModal) return false;
          proModal.style.display = "block";
          trackAppEvent("app_pro_beta_opened", { feature: featureName || "general" });
          return true;
        };

        // Klick außerhalb schließt Modal
        window.addEventListener("click", (event) => {
          if (event.target === proModal) {
            closeProModal();
          }
        });

        // Kein Zugangs-Request mehr: das Modal erklaert nur noch die Kosten.
        // Der Server-Endpunkt /track-interest bleibt bestehen, wird aber von
        // der App nicht mehr aufgerufen.

        // --- DEEP THINK TOGGLE SPERRE ---
        document.getElementById("deepSearchToggle").addEventListener("click", function (event) {
          // Wir prüfen die globale Variable window.isUserPro
          if (!window.isUserPro) {
            event.preventDefault(); // Verhindert das Umschalten des Toggles
            trackAppEvent("app_deep_think_locked_click");

            // Modal anzeigen (mit passendem Feature-Namen im Header)
            if (!window.App.showProFeatureModal("Deep Think")) {
              window.App?.showPopup?.("Deep Think is off here. It costs a multiple of a normal run.");
            }
          }
        });

        updateDeepThinkText();

        // Datei-Anhaenge (Pro) sind nach static/js/attachments.js ausgelagert
        // (window.pendingAttachments, window.renderAttachmentChips,
        // window.clearPendingAttachments, window.getAttachmentsPayload,
        // window.showBookmarkAttachments). Alle Aufrufer nutzen window.*.

        function getConfiguredLimit(key, fallback) {
          const raw = (window.APP_LIMITS || {})[key];
          const value = Number(raw);
          return Number.isFinite(value) ? value : fallback;
        }

        const LIMITS = {
          FREE: {
            NORMAL: getConfiguredLimit("free_consensus_run_limit", 0),
            DEEP: getConfiguredLimit("free_deep_think_run_limit", 0)
          },
          PRO: {
            NORMAL: getConfiguredLimit("pro_consensus_run_limit", 0),
            DEEP: getConfiguredLimit("pro_deep_think_run_limit", 0)
          }
        };
        let currentMaxLimit = LIMITS.FREE.NORMAL;
        let currentDeepLimit = LIMITS.FREE.DEEP;

        function setCurrentUsageLimits(isPro, serverLimits = {}) {
          const normalLimit = Number(serverLimits.limit ?? serverLimits.total_limit);
          const deepLimit = Number(serverLimits.deep_limit ?? serverLimits.deep_total_limit);

          currentMaxLimit = Number.isFinite(normalLimit)
            ? normalLimit
            : (isPro ? LIMITS.PRO.NORMAL : LIMITS.FREE.NORMAL);
          currentDeepLimit = Number.isFinite(deepLimit)
            ? deepLimit
            : (isPro ? LIMITS.PRO.DEEP : LIMITS.FREE.DEEP);

          window.currentMaxLimit = currentMaxLimit;
          window.currentDeepLimit = currentDeepLimit;
        }

        setCurrentUsageLimits(false);
        window.setCurrentUsageLimits = setCurrentUsageLimits;

        // Diese Funktion prüft den Status sofort beim Laden
        async function checkUserStatusOnLoad(user) {
          if (!user) return;

          try {
            const token = await user.getIdToken();

            // Aufruf an den neuen Backend-Endpoint
            const response = await fetch("/user_status", {
              method: "GET",
              headers: {
                "Authorization": "Bearer " + token,
                "Content-Type": "application/json"
              }
            });

            if (response.ok) {
              const data = await response.json();

              // 1. UI sofort umschalten (Badge an, Modelle frei)
              updateUserTierUI(data.is_pro, true);

              // 2. Limits sofort aktualisieren (verhindert den 500/25 Fehler)
              setCurrentUsageLimits(data.is_pro, data);

              // 3. Sidebar Text initial befüllen (damit dort nicht 25 steht bis zum ersten Klick)
              // Wir rufen hier kurz den Usage-Endpoint auf, um die aktuellen Zahlen zu haben
              refreshUsageDisplay(token);
            }
          } catch (error) {
            console.error("Fehler beim Laden des User-Status:", error);
          }
        }

        // Hilfsfunktion um Sidebar zu aktualisieren (Refactoring)
        async function refreshUsageDisplay(token) {
          try {
            const resp = await fetch("/usage", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ id_token: token })
            });
            const data = await resp.json();
            setCurrentUsageLimits(data.is_pro === true, data);

            window.App.renderUsageDisplay({
              remaining: data.remaining,
              deepRemaining: data.deep_remaining,
              totalLimit: currentMaxLimit,
              deepLimit: currentDeepLimit
            });

          } catch (e) { console.error(e); }
        }

        // Muss zum Push/Overlay-Umschaltpunkt in layout.css (1099px) passen.
        const SIDEBAR_OVERLAY_BREAKPOINT = 1099;
        const SIDEBAR_COLLAPSED_STORAGE_KEY = "sidebar_collapsed";

        function usesOverlaySidebar() {
          return window.matchMedia(`(max-width: ${SIDEBAR_OVERLAY_BREAKPOINT}px)`).matches;
        }

        const threadComposer = document.querySelector(".input-section");
        function syncThreadComposerReserve() {
          if (!threadComposer || !usesOverlaySidebar()) {
            document.documentElement.style.removeProperty("--thread-composer-height");
            return;
          }

          const composerHeight = Math.ceil(threadComposer.getBoundingClientRect().height);
          if (composerHeight > 0) {
            document.documentElement.style.setProperty(
              "--thread-composer-height",
              `${composerHeight}px`
            );
          }
        }

        if (threadComposer && typeof ResizeObserver === "function") {
          const threadComposerObserver = new ResizeObserver(syncThreadComposerReserve);
          threadComposerObserver.observe(threadComposer);
        }
        requestAnimationFrame(syncThreadComposerReserve);

        function closeOverlaySidebar() {
          const sidebar = document.querySelector(".sidebar");
          if (!sidebar || !usesOverlaySidebar()) return;
          sidebar.classList.remove("active");
          sidebar.classList.add("collapsed");
          updateToggleButton();
        }

        document.addEventListener("click", function (event) {
          const sidebar = document.querySelector(".sidebar");
          if (!sidebar || !usesOverlaySidebar() || !sidebar.classList.contains("active")) return;

          if (!sidebar.contains(event.target) && !event.target.closest(".sidebar-toggle")) {
            closeOverlaySidebar();
          }
        });

        // Elemente für den Search Mode:
        const deepthinkDisclaimer = document.getElementById("deepthinkDisclaimer");
        const consensusDropdown = document.getElementById("consensusModelDropdown");

        // Checkboxen für die Modelle:
        const openaiCheckbox = document.getElementById("selectOpenAI");
        const mistralCheckbox = document.getElementById("selectMistral");
        const claudeCheckbox = document.getElementById("selectClaude");
        const geminiCheckbox = document.getElementById("selectGemini");
        const deepseekCheckbox = document.getElementById("selectDeepSeek");
        const grokCheckbox = document.getElementById("selectGrok");

        // Mapping: Response-Box → zugehörige Sidebar-Checkbox
        function getCheckboxForResponse(responseId) {
          switch (responseId) {
            case "openaiResponse": return openaiCheckbox;
            case "mistralResponse": return mistralCheckbox;
            case "claudeResponse": return claudeCheckbox;
            case "geminiResponse": return geminiCheckbox;
            case "deepseekResponse": return deepseekCheckbox;
            case "grokResponse": return grokCheckbox;
            default: return null;
          }
        }

        // Label-Container der Checkboxen:
        const openaiLabel = document.querySelector("label[for='selectOpenAI']");
        const mistralLabel = document.querySelector("label[for='selectMistral']");
        const claudeLabel = document.querySelector("label[for='selectClaude']");
        const deepseekLabel = document.querySelector("label[for='selectDeepSeek']");
        const grokLabel = document.querySelector("label[for='selectGrok']");
        const geminiLabel = document.querySelector("label[for='selectGemini']");

        // Funktion, um Response-Boxen komplett auszublenden oder einzublenden:
        function setResponseBoxDisplay(id, displayValue) {
          const el = document.getElementById(id);
          if (el) {
            el.style.display = displayValue;
          }
        }

        // Funktion, um Buttons in den Response-Boxen zu deaktivieren/aktivieren:
        function updateButtons(selector, disable) {
          const btns = document.querySelectorAll(selector);
          btns.forEach(btn => {
            if (disable) {
              btn.style.pointerEvents = "none";
              btn.style.opacity = "0.5";
            } else {
              btn.style.pointerEvents = "";
              btn.style.opacity = "";
            }
          });
        }

        // Beispiel für das Setzen des systemPrompt-Wertes, falls noch nicht gesetzt:
        const defaultPrompt = "Please answer thoroughly and precisely, explaining your reasoning and covering the relevant details. Do not oversimplify. Do not ask any follow-up or clarifying questions; answer directly with the information available.";
        if (!localStorage.getItem("systemPrompt")) {
          localStorage.setItem("systemPrompt", defaultPrompt);
        }

        // Öffnen des Modals beim Klick auf das Zahnrad
        document.getElementById("editSystemPromptBtn").addEventListener("click", function () {
          const modal = document.getElementById("systemPromptModal");
          const textarea = document.getElementById("systemPromptInput");
          textarea.value = localStorage.getItem("systemPrompt");
          modal.style.display = "block";
          trackAppEvent("app_settings_open");
        });

        // Schließen des Modals
        document.getElementById("closeSystemPromptModal").addEventListener("click", function () {
          document.getElementById("systemPromptModal").style.display = "none";
        });

        // Speichern des neuen Prompts
        document.getElementById("saveSystemPromptBtn").addEventListener("click", function () {
          const newPrompt = document.getElementById("systemPromptInput").value.trim();
          localStorage.setItem("systemPrompt", newPrompt); // Speichert auch leere Strings!
          document.getElementById("systemPromptModal").style.display = "none";
          trackAppEvent("app_settings_saved");
        });

        // Öffnen des Hilfemodals beim Klick auf den Hilfebutton
        document.getElementById("helpButton").addEventListener("click", function () {
          document.getElementById("helpModal").style.display = "block";
          trackAppEvent("app_help_open");
        });

        // Schließen des Modals beim Klick auf das Schließen-Symbol
        document.getElementById("closeHelpModal").addEventListener("click", function () {
          document.getElementById("helpModal").style.display = "none";
        });

        // Optional: Modal schließen, wenn außerhalb geklickt wird
        window.addEventListener("click", function (event) {
          if (event.target === document.getElementById("helpModal")) {
            document.getElementById("helpModal").style.display = "none";
          }
        });

        // Öffnen des Feedback-Modals beim Klick auf den Feedback-Button
        document.getElementById("feedbackButton").addEventListener("click", function () {
          document.getElementById("feedbackModal").style.display = "block";
          trackAppEvent("app_feedback_open");
        });

        // Schließen des Feedback-Modals beim Klick auf das Schließen-Symbol
        document.getElementById("closeFeedbackModal").addEventListener("click", function () {
          document.getElementById("feedbackModal").style.display = "none";
        });

        // Optional: Modal schließen, wenn außerhalb des Modal-Inhalts geklickt wird
        window.addEventListener("click", function (event) {
          if (event.target === document.getElementById("feedbackModal")) {
            document.getElementById("feedbackModal").style.display = "none";
          }
        });

        // Toggle FAQ items with icons. Most answers are a single paragraph;
        // richer answers (such as model insights) use the same direct-child
        // contract via .faq-answer.
        document.querySelectorAll('.faq-item h3').forEach((question) => {
          question.setAttribute('aria-expanded', 'false');
          question.addEventListener('click', () => {
            const answer = question.nextElementSibling;
            const icon = question.querySelector('.faq-toggle-icon');
            if (!answer || !icon) return;
            const shouldOpen = !answer.style.display || answer.style.display === 'none';
            answer.style.display = shouldOpen ? 'block' : 'none';
            icon.textContent = shouldOpen ? '−' : '+';
            question.setAttribute('aria-expanded', String(shouldOpen));
          });
        });

        document.querySelectorAll('.faq-item > p, .faq-item > .faq-answer').forEach((answer) => {
          answer.style.display = 'none';
        });

        const APP_LIMITS = window.APP_LIMITS || {};
        function validateInputText() {
          const text = document.getElementById("questionInput").value.trim();
          const wordCount = text.split(/\s+/).filter(word => word.length > 0).length;
          // Prüfe, ob der Deep Think Toggle aktiv ist:
          const deepSearchActive = document.getElementById("deepSearchToggle").checked;
          // Setze das Wortlimit abhängig vom Deep Think Status
          const isPro = Boolean(window.isUserPro);
          const maxWordsRaw = deepSearchActive
            ? (isPro ? APP_LIMITS.pro_deep_search_max_words : APP_LIMITS.free_deep_search_max_words)
            : (isPro ? APP_LIMITS.pro_max_words : APP_LIMITS.free_max_words);
          const maxWords = Number(maxWordsRaw || 0);

          if (wordCount > maxWords) {
            alert(`The query is above the limit of ${maxWords} words (you entered ${wordCount}).`);
            return false;
          }
          return true;
        }
        // Von query-send.js (window.sendQuestion) mitbenutzt.
        window.validateInputText = validateInputText;

        document.getElementById("sendButton").addEventListener("click", function (e) {
          // Laeuft noch etwas (Modelle ODER Consensus), ist der Klick ein
          // Abbruch — dann nicht gegen die Eingabe validieren.
          if (window.isRunActive && window.isRunActive()) return;
          if (window.isQueryRequestRunning && window.isQueryRequestRunning()) return;
          if (!validateInputText()) {
            e.preventDefault();
          }
        });

        // Event-Listener für Eingabefelder und Buttons
        // Frage per Enter (ohne Zeilenumbruch) absenden
        document.getElementById("questionInput").addEventListener("keydown", function (event) {
          if (event.key === "Enter" && !event.shiftKey) {
            if (window.isRunActive && window.isRunActive()) {
              event.preventDefault();
              return;
            }
            if (window.isQueryRequestRunning && window.isQueryRequestRunning()) {
              event.preventDefault();
              return;
            }
            // Wenn der Button deaktiviert ist, breche die Ausführung ab
            if (document.getElementById("sendButton").disabled) {
              event.preventDefault();
              return;
            }
            event.preventDefault();
            window.sendQuestion();
          }
        });

        // Es gibt genau EINEN sichtbaren Sidebar-Toggle: der schwebende
        // (.app-nav-float) bei geschlossener Sidebar, der im Sidebar-Kopf bei
        // offener. Beide teilen sich diesen Handler.
        document.querySelectorAll(".sidebar-toggle").forEach(function (button) {
          button.addEventListener("click", handleSidebarToggle);
        });

        function handleSidebarToggle() {
          const sidebar = document.querySelector(".sidebar");
          if (!sidebar) return;

          if (usesOverlaySidebar()) {
            const shouldOpen = !sidebar.classList.contains("active");
            sidebar.classList.toggle("active", shouldOpen);
            sidebar.classList.toggle("collapsed", !shouldOpen);
          } else {
            sidebar.classList.toggle("collapsed");
            sidebar.classList.remove("active");
            localStorage.setItem(
              SIDEBAR_COLLAPSED_STORAGE_KEY,
              String(sidebar.classList.contains("collapsed"))
            );
          }
          updateToggleButton();
          trackAppEvent("app_sidebar_toggle", { open: !sidebar.classList.contains("collapsed") });
        }

        // "New comparison": derselbe saubere Ausgangszustand, den auch der
        // Logout herstellt (firebase.js::resetLoadedRunAfterLogout) — laufende
        // Streams abbrechen, Lauf + Share-/Bookmark-Kontext leeren, zurück in
        // den Hero-Zustand. Auf der Overlay-Sidebar schließt sie sich danach,
        // sonst verdeckt sie das Eingabefeld, in dem man tippen soll.
        document.getElementById("newRunButton")?.addEventListener("click", function () {
          window.cancelCurrentQuery?.();
          window.cancelCurrentConsensus?.();
          window.clearResponseBoxes?.({ silent: true });
          window.clearPreparedBookmarkShareResult?.();
          window.currentEvidenceSources = [];
          window.consensusCitationMeta = null;

          const input = document.getElementById("questionInput");
          if (input) {
            input.value = "";
            input.disabled = false;
            input.dispatchEvent(new Event("input", { bubbles: true }));
          }
          document.body.classList.add("is-hero");
          window.syncHeroResponseAccess?.();
          window.App?.setAppTitle?.();
          window.App?.setThreadQuestion?.("");

          if (usesOverlaySidebar()) closeOverlaySidebar();
          input?.focus();
          trackAppEvent("app_new_comparison");
        });

        // Fenstergröße prüfen – wenn <1024px, Sidebar einklappen
        function checkWindowSize() {
          const sidebar = document.querySelector(".sidebar");
          syncThreadComposerReserve();
          if (!sidebar) return;

          if (usesOverlaySidebar()) {
            sidebar.classList.add("collapsed");
            sidebar.classList.remove("active");
          } else {
            const prefersCollapsed = localStorage.getItem(SIDEBAR_COLLAPSED_STORAGE_KEY) === "true";
            sidebar.classList.toggle("collapsed", prefersCollapsed);
            sidebar.classList.remove("active");
          }
          updateToggleButton();
        }
        window.addEventListener("resize", checkWindowSize);
        checkWindowSize(); // Initial

        // Fixed navigation should make room while a result is being read. The
        // main app scrolls the window, whereas the Watch dashboard owns an
        // independent scroll container, so both sources feed the same small
        // direction-aware controller. A little accumulated travel prevents
        // touchpad jitter from flickering the controls on and off.
        function initReadingChrome() {
          const body = document.body;
          const consensusOutput = document.getElementById("consensusOutput");
          const watchPage = document.getElementById("watchDashboard");
          const topRevealDistance = 64;
          const directionTravel = 14;
          const mainState = { lastY: window.scrollY, direction: 0, travel: 0, frame: 0 };
          const watchState = { lastY: watchPage?.scrollTop || 0, direction: 0, travel: 0, frame: 0 };

          function watchIsOpen() {
            return Boolean(watchPage && !watchPage.hidden);
          }

          function consensusIsReadable() {
            return Boolean(
              !watchIsOpen()
              && consensusOutput
              && !consensusOutput.classList.contains("is-hidden")
              && document.getElementById("consensusAnswerBody")?.textContent?.trim()
            );
          }

          function canAutoHide(source) {
            return source === watchPage ? watchIsOpen() : consensusIsReadable();
          }

          function setChromeHidden(hidden) {
            body.classList.toggle("is-reading-chrome-hidden", Boolean(hidden));
          }

          function resetState(state, y) {
            state.lastY = y;
            state.direction = 0;
            state.travel = 0;
          }

          function readScrollY(source) {
            return source === watchPage ? source.scrollTop : window.scrollY;
          }

          function handleScroll(source, state) {
            if (state.frame) return;
            state.frame = requestAnimationFrame(() => {
              state.frame = 0;
              const y = readScrollY(source);
              const delta = y - state.lastY;
              state.lastY = y;

              if (!canAutoHide(source) || y <= topRevealDistance) {
                state.direction = 0;
                state.travel = 0;
                setChromeHidden(false);
                return;
              }
              if (Math.abs(delta) < 0.5) return;

              const nextDirection = delta > 0 ? 1 : -1;
              if (nextDirection !== state.direction) {
                state.direction = nextDirection;
                state.travel = 0;
              }
              state.travel += Math.abs(delta);
              if (state.travel < directionTravel) return;

              setChromeHidden(nextDirection > 0);
              state.travel = 0;
            });
          }

          function revealAndResync() {
            setChromeHidden(false);
            resetState(mainState, window.scrollY);
            resetState(watchState, watchPage?.scrollTop || 0);
          }

          window.addEventListener("scroll", () => handleScroll(window, mainState), { passive: true });
          watchPage?.addEventListener("scroll", () => handleScroll(watchPage, watchState), { passive: true });

          // Hidden controls never trap keyboard users: starting keyboard
          // navigation reveals them before focus advances.
          document.addEventListener("keydown", event => {
            if (!body.classList.contains("is-reading-chrome-hidden")) return;
            if (["Tab", "Home", "PageUp", "ArrowUp"].includes(event.key)) {
              revealAndResync();
            }
          }, true);
          document.querySelector(".app-nav-float")?.addEventListener("focusin", revealAndResync);
          document.getElementById("viewSwitch")?.addEventListener("focusin", revealAndResync);

          // Opening/closing Watches, clearing a result, or returning to the
          // hero can happen without a scroll event. Keep the chrome state in
          // sync with those view transitions as well.
          const syncEligibility = () => {
            if (!watchIsOpen() && !consensusIsReadable()) revealAndResync();
            else {
              resetState(mainState, window.scrollY);
              resetState(watchState, watchPage?.scrollTop || 0);
            }
          };
          const observer = new MutationObserver(syncEligibility);
          observer.observe(body, { attributes: true, attributeFilter: ["class"] });
          if (consensusOutput) {
            observer.observe(consensusOutput, { attributes: true, attributeFilter: ["class"] });
          }
          if (watchPage) {
            observer.observe(watchPage, { attributes: true, attributeFilter: ["hidden"] });
          }
        }

        initReadingChrome();

        // Aktualisiert den Pfeil des Sidebar-Toggle-Buttons
        function updateToggleButton() {
          const sidebar = document.querySelector(".sidebar");
          if (!sidebar) return;
          const isOpen = usesOverlaySidebar()
            ? sidebar.classList.contains("active")
            : !sidebar.classList.contains("collapsed");
          if (!isOpen && sidebar.contains(document.activeElement)) {
            document.getElementById("toggleSidebarButton")?.focus();
          }
          sidebar.inert = !isOpen;
          if (isOpen) {
            sidebar.removeAttribute("aria-hidden");
          } else {
            sidebar.setAttribute("aria-hidden", "true");
          }
          document.querySelectorAll(".sidebar-toggle").forEach(function (toggleButton) {
            toggleButton.setAttribute("aria-expanded", String(isOpen));
            toggleButton.setAttribute("aria-label", isOpen ? "Collapse sidebar" : "Open sidebar");
            toggleButton.title = isOpen ? "Collapse sidebar" : "Open sidebar";
          });
          const newText = sidebar.classList.contains("collapsed") ? "►" : "◄";
          const arrow = document.querySelector(".sidebar-toggle .arrow");
          if (arrow) {
            arrow.textContent = newText;
          }
        }

        // Extrahiert aus dem Differences-Text den BestModel-Wert
        // parseBestModel ist nach static/js/consensus-run.js gewandert (einziger
        // Aufrufer war window.getConsensus).

        // Funktion, um das Popup anzuzeigen
        // showPopup stammt aus app-core.js (window.App), siehe Alias oben.

        // Quellenverzeichnis unter der Antwort. Die hochgestellten Zahlen im
        // Konsenstext zeigen hierher, deshalb ist die Reihenfolge dieser Liste
        // der Vertrag: Position n == [Sn]. Das Panel selbst bleibt zu, bis der
        // Quellen-Chip in der Fusszeile es oeffnet.
        function renderEvidenceSources(sources) {
          const panel = document.getElementById("consensusSourcesPanel");
          const listEl = document.getElementById("consensusSourcesList");
          if (!panel || !listEl) return;

          listEl.innerHTML = "";

          if (!sources || !sources.length) {
            panel.hidden = true;
            window.App?.consensusPipeline?.renderProvenance?.();
            return;
          }

          sources.forEach((src, idx) => {
            const number = idx + 1;
            const li = document.createElement("li");
            li.className = "consensus-source-item";
            li.value = number;

            const index = document.createElement("span");
            index.className = "consensus-source-index";
            index.textContent = String(number);
            li.appendChild(index);

            const body = document.createElement("div");
            body.className = "consensus-source-body";

            const url = String(src.url || "");
            let safeHref = "";
            let host = "";
            try {
              const parsed = new URL(url);
              if (["http:", "https:"].includes(parsed.protocol)) {
                safeHref = parsed.href;
                host = parsed.hostname.replace(/^www\./, "");
              }
            } catch (e) {
              // Ohne gueltige URL bleibt der Titel als reiner Text stehen.
            }

            const title = safeHref ? document.createElement("a") : document.createElement("span");
            title.className = "consensus-source-title";
            title.textContent = src.title || url || "Source " + number;
            if (safeHref) {
              title.href = safeHref;
              title.target = "_blank";
              title.rel = "noopener noreferrer";
            }
            body.appendChild(title);

            if (host) {
              const meta = document.createElement("span");
              meta.className = "consensus-source-host";
              meta.textContent = host;
              body.appendChild(meta);
            }

            if (src.snippet || src.text) {
              const snippet = document.createElement("div");
              snippet.className = "consensus-source-snippet";
              snippet.textContent = src.snippet || src.text;
              body.appendChild(snippet);
            }

            li.appendChild(body);
            listEl.appendChild(li);
          });

          // Sichtbar wird das Panel erst ueber den Chip; hier zaehlt nur, dass
          // es Inhalt hat, damit der Chip erscheinen kann.
          window.App?.consensusPipeline?.renderProvenance?.();
        }

        // global machen, falls du es anderswo brauchst
        window.renderEvidenceSources = renderEvidenceSources;

        // API Testbereich umschalten (für den Pfeil in der API Keys Section)
        window.toggleApiTest = function () {
          const area = document.getElementById("apiTestArea");
          const button = document.getElementById("toggleApiTest");
          const arrow = button.querySelector(".arrow");
          if (area.style.display === "none" || area.style.display === "") {
            area.style.display = "block";
            arrow.classList.add("rotated");
          } else {
            area.style.display = "none";
            arrow.classList.remove("rotated");
          }
        };

        // Models remains one compact sidebar row. Its detailed controls open
        // on the composer's existing run picker instead of expanding the
        // navigation into a six-row settings panel.
        document.getElementById("sidebarModelPicker")?.addEventListener("click", function (event) {
          event.preventDefault();
          event.stopPropagation();
          const consensusSelect = document.getElementById("consensusModelDropdown");
          if (!consensusSelect) return;

          const openPicker = () => {
            window.App.openModelPicker(consensusSelect);
            consensusSelect._customModelPicker?.displayButton?.focus({ preventScroll: true });
          };

          if (usesOverlaySidebar()) {
            closeOverlaySidebar();
            window.setTimeout(openPicker, 180);
          } else {
            openPicker();
          }
          trackAppEvent("app_model_picker_opened", { source: "sidebar" });
        });

        window.toggleAllResponses = function () {
          setAgentMode(!isAgentModeEnabled(), { persist: true });
        };

        // Collapse/Expand einer Antwort-Box
        window.toggleCollapse = function (responseId) {
          const responseBox = document.getElementById(responseId);
          const content = responseBox.querySelector(".collapsible-content");
          const arrow = responseBox.querySelector(".collapse-btn .arrow");
          if (content) content.classList.toggle("collapsed");
          if (arrow) arrow.classList.toggle("rotated");
        };

        // Exclude/Include einer Antwort-Box
        // ➜ steuert jetzt auch die Sidebar-Checkboxen / modelSelectionArea mit
        window.toggleExclude = function (responseId) {
          const box = document.getElementById(responseId);
          if (!box) return;

          const checkbox = getCheckboxForResponse(responseId);

          // Fallback: falls aus irgendeinem Grund keine Checkbox gefunden wird,
          // verhalte dich wie früher (nur .excluded toggeln).
          if (!checkbox) {
            if (!box.classList.contains("excluded")) {
              showPopup("You have excluded this answer. It is minimized and will not be included in the consensus.");
            }
            box.classList.toggle("excluded");
            return;
          }

          const currentlyChecked = checkbox.checked;
          const willBeChecked = !currentlyChecked;

          if (currentlyChecked && !box.classList.contains("excluded")) {
            showPopup("You have excluded this answer. It is minimized and will not be included in the consensus.");
          }

          checkbox.checked = willBeChecked;

          window.toggleModel(responseId, willBeChecked);
        };

        // --- MODEL PREFERENCES WIEDERHERSTELLUNG ---
        // setPickerToValue, applyTierDefaultModels, getModelPrefByResponseId,
        // animateResponseReorder, setModelSelectionState und restoreModelSelections
        // sind nach static/js/model-picker.js ausgelagert. Aliase siehe oben.

        // Agent Mode (Status, Timer, gruppierter Lauf) ist nach
        // static/js/agent-mode.js ausgelagert. Exporte: window.setAgentModeStatus,
        // window.updateAgentModeUI, window.isAgentModeEnabled, window.setAgentMode,
        // window.isAgentModeRunning. Lokale Aliase + Picker-Bruecke siehe oben.

        // 1. Initialer Aufruf beim Laden der Seite
        window.restoreModelSelections();
        updateAgentModeUI();

        window.addEventListener("pageshow", function () {
          window.restoreModelSelections();
          updateAgentModeUI();
        });

        // 2. Event Listener zum Speichern hinzufügen (bleibt gleich)
        modelPrefs.forEach(pref => {
          const checkbox = document.getElementById(pref.checkId);
          const select = document.getElementById(pref.selectId);
          const labelText = document.getElementById(pref.textId);

          if (checkbox) {
            checkbox.addEventListener("change", function () {
              setModelSelectionState(pref, this.checked, { persist: true, syncCheckbox: false });
            });
          }

          if (select) {
            select.addEventListener("change", function () {
              localStorage.setItem("pref_select_" + pref.key, this.value);
              window.App.markConsensusPresetCustom?.();
              const selectedLabel = getModelOptionLabel(this.options[this.selectedIndex]) || this.value;
              if (labelText) {
                labelText.textContent = selectedLabel;
                labelText.title = `Choose model: ${selectedLabel}`;
              }
              trackAppEvent("app_model_picker_changed", {
                provider: pref.key,
                model: selectedLabel
              });
              updateAgentModeUI();
            });
          }
        });

        const agentModeSwitch = document.getElementById("agentModeSwitch");
        if (agentModeSwitch) {
          agentModeSwitch.addEventListener("change", function () {
            setAgentMode(this.checked, { persist: true });
          });
        }

        const inlineAgentModeSwitch = document.getElementById("toggleAllButton");
        if (inlineAgentModeSwitch) {
          inlineAgentModeSwitch.addEventListener("change", function () {
            setAgentMode(this.checked, { persist: true });
          });
        }

        // --- NEU: Event Listener für Consensus Dropdown ---
        const consensusSelect = document.getElementById("consensusModelDropdown");
        if (consensusSelect) {
          consensusSelect.addEventListener("change", function () {
            // Speichere die Auswahl im LocalStorage, sobald der User sie ändert
            localStorage.setItem("pref_select_consensus", this.value);
            // Eine explizite Modellwahl (change-Event) verlaesst die
            // Preset-Ebene; Preset-Klicks und die temporaere Deep-Think-
            // Auswahl feuern bewusst KEIN change (siehe model-picker.js).
            window.App.markConsensusPresetCustom?.();
            const selectedLabel = getModelOptionLabel(this.options[this.selectedIndex]) || this.value;
            trackAppEvent("app_consensus_model_changed", { model: selectedLabel });

            // Optional: Fokus entfernen, wie bei den anderen Pickern
            this.blur();
          });
        }

        window.filterBookmarks = function (query) {
          const q = String(query || "").trim().toLowerCase();
          document.querySelectorAll("#bookmarksContainer .bookmark").forEach(el => {
            const text = el.querySelector("p")?.textContent?.toLowerCase() || "";
            el.style.display = q && !text.includes(q) ? "none" : "";
          });
        };

        const bookmarkSearchHead = document.querySelector(".sidebar-bookmarks-head");
        const bookmarkSearchTrigger = document.getElementById("bookmarkSearchTrigger");
        const bookmarkSearchInput = document.getElementById("chatSearch");

        function setBookmarkSearchOpen(isOpen, { focus = false, clear = false } = {}) {
          if (!bookmarkSearchHead || !bookmarkSearchTrigger || !bookmarkSearchInput) return;
          if (clear) {
            bookmarkSearchInput.value = "";
            window.filterBookmarks("");
          }
          bookmarkSearchHead.classList.toggle("is-searching", isOpen);
          bookmarkSearchTrigger.setAttribute("aria-expanded", String(isOpen));
          if (isOpen && focus) requestAnimationFrame(() => bookmarkSearchInput.focus());
        }

        bookmarkSearchTrigger?.addEventListener("click", function (event) {
          event.stopPropagation();
          setBookmarkSearchOpen(true, { focus: true });
        });

        // Search operates on compact metadata only. When needed, remaining
        // metadata pages are fetched without loading full bookmark answers.
        bookmarkSearchInput?.addEventListener("input", async function () {
          const query = this.value;
          if (query.trim()) await window.loadAllBookmarkMetadata?.();
          window.filterBookmarks(query);
        });

        bookmarkSearchInput?.addEventListener("keydown", function (event) {
          if (event.key !== "Escape") return;
          event.preventDefault();
          setBookmarkSearchOpen(false, { clear: true });
          document.getElementById("bookmarksToggle")?.focus();
        });

        bookmarkSearchHead?.addEventListener("focusout", function () {
          setTimeout(() => {
            if (!bookmarkSearchHead.contains(document.activeElement) && !bookmarkSearchInput?.value.trim()) {
              setBookmarkSearchOpen(false);
            }
          }, 0);
        });

        window.toggleBookmarks = function () {
          const section = document.querySelector(".bookmarks-section");
          const container = document.getElementById("bookmarksContainer");
          const toggle = document.getElementById("bookmarksToggle");
          if (!section || !container || !toggle) return;

          const isCollapsed = section.classList.toggle("is-collapsed");
          container.classList.toggle("hidden", isCollapsed);
          toggle.setAttribute("aria-expanded", String(!isCollapsed));
          if (isCollapsed) setBookmarkSearchOpen(false, { clear: true });
          localStorage.setItem("bookmarks_collapsed", String(isCollapsed));
          trackAppEvent("app_sidebar_section_toggled", { section: "bookmarks", open: !isCollapsed });
        };

        function restoreBookmarksState() {
          const section = document.querySelector(".bookmarks-section");
          const container = document.getElementById("bookmarksContainer");
          const toggle = document.getElementById("bookmarksToggle");
          if (!section || !container || !toggle) return;

          const isCollapsed = localStorage.getItem("bookmarks_collapsed") === "true";
          section.classList.toggle("is-collapsed", isCollapsed);
          container.classList.toggle("hidden", isCollapsed);
          toggle.setAttribute("aria-expanded", String(!isCollapsed));
        }

        restoreBookmarksState();

        // Custom-Model-Picker (eigene Listbox ueber den nativen <select>) ist
        // nach static/js/model-picker.js ausgelagert: getModelPickerState,
        // syncCustomModelPicker, renderCustomModelPicker, openModelPicker,
        // collapseExpandedModelPicker, initCustomModelPicker (+ modul-privater
        // expandedModelPicker-State). Aliase siehe oben.

        modelPrefs.forEach(pref => {
          const labelText = document.getElementById(pref.textId);
          const select = document.getElementById(pref.selectId);
          if (!labelText || !select) return;

          labelText.classList.add("model-name-trigger");
          labelText.setAttribute("role", "button");
          labelText.setAttribute("tabindex", "0");
          labelText.setAttribute("aria-haspopup", "listbox");
          labelText.setAttribute("aria-controls", pref.selectId);
          labelText.title = `Choose model: ${labelText.textContent}`;

          labelText.addEventListener("click", function (event) {
            event.stopPropagation();
            openModelPicker(select);
          });

          labelText.addEventListener("keydown", function (event) {
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault();
              openModelPicker(select);
            }
          });

          select.addEventListener("change", function () {
            collapseExpandedModelPicker(select);
          });

          select.addEventListener("keydown", function (event) {
            if (event.key === "Escape") {
              collapseExpandedModelPicker(select);
              labelText.focus({ preventScroll: true });
            }
          });

          initCustomModelPicker(select, { externalTrigger: labelText });
        });

        initCustomModelPicker(document.getElementById("consensusModelDropdown"), { presets: true });



        // consensusGenerated (query-state) lebt jetzt in static/js/query-send.js.

        const consensusLifecycle = window.App.consensusLifecycle;

        // Der frühere "Generate Consensus"-Button ist entfernt - Konsens läuft
        // automatisch. Lokaler Alias hält die bestehenden Aufrufstellen stabil.
        function setConsensusGate(disabled) {
          consensusLifecycle.setGate(disabled);
        }

        setConsensusGate(true);

        // Query-Send (window.sendQuestion, Cancel, Query-Run-State und die
        // Query-Helfer isDemoQuery/predictSearchIntent/getActiveMode) ist nach
        // static/js/query-send.js ausgelagert. Die Send-Listener unten fragen
        // window.isQueryRequestRunning() statt das private Flag zu lesen.

        // Modelle (Checkboxen) ein-/ausschalten
        window.toggleModel = function (responseId, isChecked) {
          setModelSelectionState(responseId, isChecked, { persist: true, syncCheckbox: true });
        };
        document.getElementById("selectOpenAI").addEventListener("change", function () {
          toggleModel("openaiResponse", this.checked);
        });
        document.getElementById("selectMistral").addEventListener("change", function () {
          toggleModel("mistralResponse", this.checked);
        });
        document.getElementById("selectClaude").addEventListener("change", function () {
          toggleModel("claudeResponse", this.checked);
        });
        document.getElementById("selectGemini").addEventListener("change", function () {
          toggleModel("geminiResponse", this.checked);
        });
        document.getElementById("selectDeepSeek").addEventListener("change", function () {
          toggleModel("deepseekResponse", this.checked);
        });
        document.getElementById("selectGrok").addEventListener("change", function () {
          toggleModel("grokResponse", this.checked);
        });

        // Modell-Dropdowns aktualisieren die angezeigten Namen und Tooltips
        function syncVisibleModelName(select, textId) {
          const el = document.getElementById(textId);
          if (!el || !select) return;
          const label = getModelOptionLabel(select.options[select.selectedIndex]) || select.value;
          el.textContent = label;
          el.title = `Choose model: ${label}`;
        }

        document.getElementById("openaiModelSelect").addEventListener("change", function () {
          syncVisibleModelName(this, "openaiModelText");
        });
        document.getElementById("mistralModelSelect").addEventListener("change", function () {
          syncVisibleModelName(this, "mistralModelText");
        });
        document.getElementById("claudeModelSelect").addEventListener("change", function () {
          syncVisibleModelName(this, "claudeModelText");
        });
        document.getElementById("geminiModelSelect").addEventListener("change", function () {
          syncVisibleModelName(this, "geminiModelText");
        });
        document.getElementById("deepseekModelSelect").addEventListener("change", function () {
          syncVisibleModelName(this, "deepseekModelText");
        });
        document.getElementById("grokModelSelect").addEventListener("change", function () {
          syncVisibleModelName(this, "grokModelText");
        });

        // collapse model dropdown after selection to avoid lingering focus
        document.querySelectorAll('.model-picker').forEach(function (sel) {
          sel.addEventListener('change', function () {
            this.blur();
          });
        });

        // Erneut API Keys in Felder schreiben (falls benötigt)
        ["openaiKey", "mistralKey", "anthropicKey", "geminiKey", "deepseekKey", "grokKey"].forEach(function (key) {
          const stored = localStorage.getItem(key);
          if (stored) {
            document.getElementById(key).value = stored;
          }
        });

        function showDisclaimerPopup() {
          const popup = document.getElementById('disclaimerPopup');
          popup.classList.add('show');
          // Popup nach 3 Sekunden wieder ausblenden
          setTimeout(() => {
            popup.classList.remove('show');
          }, 5000);
        }

        // Globale Variable, um die letzte verarbeitete Frage zu speichern.
        // Auf window gehoben: consensus-run.js (window.getConsensus) liest sie,
        // query-send.js (window.sendQuestion) schreibt sie.
        window.lastQuestion = "";

        // Consensus-Run (Request/Payload/Rendering) ist nach
        // static/js/consensus-run.js ausgelagert: window.getConsensus baut das
        // /consensus-Payload, faehrt den SSE-Stream und rendert das Ergebnis.
        // Run-State/Gate/Abort liegen in consensus-lifecycle.js
        // (window.App.consensusLifecycle), das die Bruecke bereitstellt.
        window.updateConsensusButtonAvailability();

        // Inline-Status statt Browser-alert() fuer den API-Key-Test.
        function setApiKeysStatus(message, tone) {
          const el = document.getElementById("apiKeysStatus");
          if (!el) return;
          if (!message) {
            el.hidden = true;
            el.textContent = "";
            el.classList.remove("is-error", "is-success");
            return;
          }
          el.hidden = false;
          el.textContent = message;
          el.classList.toggle("is-error", tone === "error");
          el.classList.toggle("is-success", tone === "success");
        }

        // Testet die API Keys und aktualisiert das Feedback
        window.testAllKeys = async function () {
          trackAppEvent("app_api_keys_test_started");
          setApiKeysStatus("");
          const currentUser = window.auth?.currentUser;
          if (!currentUser || !currentUser.emailVerified) {
            setApiKeysStatus("Please log in with a verified account before saving or testing your own API keys.", "error");
            trackAppEvent("app_api_keys_test_result", { status: "auth_required" });
            return;
          }

          const openaiKey = document.getElementById("openaiKey").value;
          const mistralKey = document.getElementById("mistralKey").value;
          const anthropicKey = document.getElementById("anthropicKey").value;
          const geminiKey = document.getElementById("geminiKey").value;
          const deepseekKey = document.getElementById("deepseekKey").value;
          const grokKey = document.getElementById("grokKey").value;
          const enteredKeys = [openaiKey, mistralKey, anthropicKey, geminiKey, deepseekKey, grokKey]
            .filter(key => (key || "").trim() !== "");
          if (!enteredKeys.length) {
            setApiKeysStatus("Enter at least one API key to test.", "error");
            trackAppEvent("app_api_keys_test_result", { status: "no_keys" });
            return;
          }

          let idToken = "";
          try {
            idToken = await currentUser.getIdToken();
          } catch (error) {
            setApiKeysStatus("Your login session could not be verified. Please log in again.", "error");
            trackAppEvent("app_api_keys_test_result", { status: "auth_error" });
            return;
          }

          localStorage.setItem("openaiKey", openaiKey);
          localStorage.setItem("mistralKey", mistralKey);
          localStorage.setItem("anthropicKey", anthropicKey);
          localStorage.setItem("geminiKey", geminiKey);
          localStorage.setItem("deepseekKey", deepseekKey);
          localStorage.setItem("grokKey", grokKey);
          if (typeof window.updateQuestionInputAccess === "function") {
            window.updateQuestionInputAccess();
          }
          const spinner = document.getElementById("apiSpinner");
          spinner.style.display = "inline-block";
          try {
            const response = await fetch("/check_keys", {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + idToken
              },
              body: JSON.stringify({
                id_token: idToken,
                openai_key: openaiKey,
                mistral_key: mistralKey,
                anthropic_key: anthropicKey,
                gemini_key: geminiKey,
                deepseek_key: deepseekKey,
                grok_key: grokKey
              })
            });
            if (!response.ok) {
              let errorMessage = "API key check failed.";
              try {
                const errorData = await response.json();
                errorMessage = errorData.detail || errorData.error || errorMessage;
              } catch (_) {}
              throw new Error(errorMessage);
            }
            const data = await response.json();
            if (!data || !data.results) {
              throw new Error("The response does not contain a 'results' object. Response: " + JSON.stringify(data));
            }
            const openaiResult = data.results["OpenAI"];
            const mistralResult = data.results["Mistral"];
            const anthropicResult = data.results["Anthropic"];
            const geminiResult = data.results["Gemini"];
            const deepseekResult = data.results["DeepSeek"];
            const grokResult = data.results["Grok"];
            const validCount = [openaiResult, mistralResult, anthropicResult, geminiResult, deepseekResult, grokResult]
              .filter(result => result === "valid").length;
            const openaiFeedback = document.getElementById("openaiFeedback");
            const mistralFeedback = document.getElementById("mistralFeedback");
            const anthropicFeedback = document.getElementById("anthropicFeedback");
            const geminiFeedback = document.getElementById("geminiFeedback");
            const deepseekFeedback = document.getElementById("deepseekFeedback");
            const grokFeedback = document.getElementById("grokFeedback");
            openaiFeedback.innerHTML = openaiResult === "valid" ? "&#9734;" : "&#10007;";
            openaiFeedback.style.color = openaiResult === "valid" ? "green" : "red";
            mistralFeedback.innerHTML = mistralResult === "valid" ? "&#9734;" : "&#10007;";
            mistralFeedback.style.color = mistralResult === "valid" ? "green" : "red";
            anthropicFeedback.innerHTML = anthropicResult === "valid" ? "&#9734;" : "&#10007;";
            anthropicFeedback.style.color = anthropicResult === "valid" ? "green" : "red";
            geminiFeedback.innerHTML = geminiResult === "valid" ? "&#9734;" : "&#10007;";
            geminiFeedback.style.color = geminiResult === "valid" ? "green" : "red";
            deepseekFeedback.innerHTML = deepseekResult === "valid" ? "&#9734;" : "&#10007;";
            deepseekFeedback.style.color = deepseekResult === "valid" ? "green" : "red";
            grokFeedback.innerHTML = grokResult === "valid" ? "&#9734;" : "&#10007;";
            grokFeedback.style.color = grokResult === "valid" ? "green" : "red";
            trackAppEvent("app_api_keys_test_result", { status: "success", valid_count: validCount });
            setApiKeysStatus(
              validCount > 0
                ? "Keys saved. " + validCount + " of 6 verified successfully."
                : "Keys saved, but none could be verified. Please check them.",
              validCount > 0 ? "success" : "error"
            );
          } catch (error) {
            console.error("Error while testing API keys:", error);
            trackAppEvent("app_api_keys_test_result", { status: "error" });
            setApiKeysStatus("Could not test the API keys: " + error.message, "error");
          } finally {
            spinner.style.display = "none";
          }
        };

        const feedbackForm = document.getElementById("feedbackForm");
        if (feedbackForm) {
          feedbackForm.addEventListener("submit", function (e) {
            e.preventDefault();
            const message = this.elements["message"].value;
            const email = this.elements["email"].value;
            trackAppEvent("app_feedback_submit", { logged_in: !!window.auth?.currentUser });
            // Hier rufen wir die neue sendFeedback-Funktion auf, die den Backend-Endpoint nutzt.
            window.sendFeedback(message, email)
              .then(data => {
                if (data.status === "success") {
                  this.reset();
                  trackAppEvent("app_feedback_result", { status: "success" });
                  alert("Feedback sent!");
                } else {
                  trackAppEvent("app_feedback_result", { status: "error" });
                  alert("Error: " + data.detail);
                }
              })
              .catch(error => {
                console.error("Error while saving feedback:", error);
                trackAppEvent("app_feedback_result", { status: "error" });
                alert("Could not save your feedback: " + error.message);
              });
          });
        }

        window.clearResponseBoxes = function (options = {}) {
          if (!options.silent) trackAppEvent("app_responses_cleared");
          const boxIds = [
            "openaiResponse",
            "mistralResponse",
            "claudeResponse",
            "geminiResponse",
            "deepseekResponse",
            "grokResponse"
          ];

          // Konsens unterbinden und den rahmenlosen Bereich wieder ausblenden.
          setConsensusGate(true);
          window.hideConsensusOutput?.();
          // Follow-up-Affordance/Chip gehören zum gelöschten Konsens.
          window.App.followup?.reset?.();

          // Lösche den Inhalt aller Modell-Antwortboxen.
          boxIds.forEach(id => {
            const box = document.getElementById(id);
            if (box) {
              delete box.dataset.consensusAnswer;
              delete box.dataset.consensusSources;
              const contentEl = box.querySelector(".collapsible-content");
              if (contentEl) {
                contentEl.innerHTML = "";
              }
            }
          });

          // Leere den Inhalt der Consensus-Antwortbox.
          const consensusBox = document.getElementById("consensusResponse");
          if (consensusBox) {
            const mainElement = window.App.consensusBodyEl(consensusBox);
            if (mainElement) {
              mainElement.innerHTML = "";
            }

            const diffElement = consensusBox.querySelector(".consensus-differences p");
            if (diffElement) {
              diffElement.innerHTML = "";
            }
            if (window.resetCredibilityFrame) {
              window.resetCredibilityFrame(consensusBox.querySelector(".consensus-differences"));
            }
          }

          // Leere den Inhalt der Input-Box mit der ID "questionInput".
          const inputBox = document.getElementById("questionInput");
          if (inputBox) {
            inputBox.value = "";
            inputBox.dispatchEvent(new Event("input", { bubbles: true }));
            window.syncDemoChipState?.();
          }
          window.App.setAppTitle();
          window.App.setThreadQuestion?.("");
          window.lastQuestion = "";
          window.currentEvidenceSources = [];
          window.consensusCitationMeta = null;
          window.clearPreparedBookmarkShareResult?.();

          setAgentModeStatus("idle");
        }

        // getActiveMode lebt jetzt in static/js/query-send.js (einziger Aufrufer war sendQuestion).

        function updateCountdown() {
          // Aktuelles Datum und Uhrzeit
          const now = new Date();

          // Erstelle ein Datum für heute um 00:15
          let resetTime = new Date(now.getFullYear(), now.getMonth(), now.getDate(), 0, 15, 0, 0);

          // Falls aktuelle Zeit bereits nach 00:15 liegt, setze resetTime auf morgen um 00:15
          if (now >= resetTime) {
            resetTime.setDate(resetTime.getDate() + 1);
          }

          // Differenz in Millisekunden bis zur festgelegten Reset-Zeit (00:15)
          const diff = resetTime - now;

          // Berechne Stunden, Minuten und Sekunden
          const hours = Math.floor(diff / (1000 * 60 * 60));
          const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
          const seconds = Math.floor((diff % (1000 * 60)) / 1000);

          // Formatieren mit führenden Nullen
          const formattedHours = hours.toString().padStart(2, "0");
          const formattedMinutes = minutes.toString().padStart(2, "0");
          const formattedSeconds = seconds.toString().padStart(2, "0");

          // Ausgabe im HTML-Element aktualisieren
          document.getElementById('countdownDisplay').innerHTML =
            "Resets in: " + formattedHours + ":" + formattedMinutes + ":" + formattedSeconds;

          // Wenn der Countdown abgelaufen ist, wird der Server (oder die Seite) neu gestartet
          if (diff <= 0) {
            location.reload(); // Hier kann auch ein anderer Reset-Mechanismus aufgerufen werden
          }
        }

        // Countdown sofort starten und jede Sekunde aktualisieren
        updateCountdown();
        setInterval(updateCountdown, 1000);

        document.getElementById("logoLink").addEventListener("click", () => {
          trackAppEvent("app_logo_home_click");
          // wenn ich wirklich zurück will, entferne den visitedLanding-Flag
          localStorage.removeItem("visitedLanding");
          // und damit gilt beim nächsten Laden der Landing-Seite wieder first-view
        });

        // Demo-Chip automatisch beim ersten echten Besuch (wenn noch nie benutzt)
        if (!localStorage.getItem("marketingPopupShown") || !localStorage.getItem("demoChipDismissed")) {
          createStartDemoChip();
        }

        // Consensus-Actions (Copy/Citation/Share controls) sind nach
        // static/js/consensus-actions.js ausgelagert.

        // === Share-Dialog (öffentliche Links) ===
        // Views: "confirm" (Opt-in vor dem Teilen), "success" (Link erstellt),
        // "list" (eigene Links verwalten/widerrufen).

        // Share-Dialog (oeffentliche Links) ist nach static/js/share-dialog.js
        // ausgelagert. Export: window.openShareDialog.

        // Initialer Aufruf: Alles sperren (Standard)
        updatePremiumModelsState(false);

        window.isUserPro = false;

        // Tier-/Pro-UI (updateUserTierUI, updatePremiumModelsState) ist nach
        // static/js/user-tier.js ausgelagert. Exporte gleichen Namens auf window.

        // Event Listener für den "Why limits?"-Link in der Sidebar
        const headerUpgradeLink = document.getElementById("upgradeLink");

        if (headerUpgradeLink) {
          headerUpgradeLink.addEventListener("click", function (e) {
            e.preventDefault();
            window.App.showProFeatureModal?.("The expensive extras");
          });
        }

        // 1. Tooltip Element einmalig erstellen
        const tooltip = document.createElement('div');
        tooltip.className = 'global-tooltip';
        document.body.appendChild(tooltip);

        let tooltipTimer = null;
        let activeTooltipTarget = null;

        const clearTooltipTimer = () => {
          if (tooltipTimer) {
            clearTimeout(tooltipTimer);
            tooltipTimer = null;
          }
        };

        // Funktion zum Anzeigen
        const showTooltip = (target) => {
          if (!target) return;

          const text = target.getAttribute('data-tooltip');
          if (!text) return;

          // Text setzen
          tooltip.textContent = text;
          tooltip.classList.add('visible');

          // Position berechnen
          const rect = target.getBoundingClientRect();

          // Standard: Links am Text ausgerichtet, unterhalb des Textes
          let top = rect.bottom + 5;
          let left = rect.left;

          // Sicherheitscheck: Falls Tooltip rechts aus dem Bild ragt (Mobile)
          // Wir setzen ihn temporär, um die Breite zu messen
          tooltip.style.left = left + 'px';
          tooltip.style.top = top + 'px';

          const tooltipRect = tooltip.getBoundingClientRect();
          if (tooltipRect.right > window.innerWidth) {
            // Nach links schieben, damit er im Bild bleibt
            left = window.innerWidth - tooltipRect.width - 10;
          }

          tooltip.style.left = left + 'px';
          tooltip.style.top = top + 'px';
        };

        // Funktion zum Verstecken
        const hideTooltip = () => {
          clearTooltipTimer();
          activeTooltipTarget = null;
          tooltip.classList.remove('visible');
        };

        // Event Listener für alle Elemente mit data-tooltip
        document.body.addEventListener('mouseover', (e) => {
          const target = e.target.closest('[data-tooltip]');
          if (!target || target.contains(e.relatedTarget)) return;

          clearTooltipTimer();
          activeTooltipTarget = target;
          tooltipTimer = setTimeout(() => {
            if (activeTooltipTarget === target) {
              showTooltip(target);
            }
          }, 2500);
        });

        document.body.addEventListener('mouseout', (e) => {
          const target = e.target.closest('[data-tooltip]');
          if (!target || target.contains(e.relatedTarget)) return;

          hideTooltip();
        });

        // Optional: Beim Scrollen verstecken, damit er nicht "floated"
        window.addEventListener('scroll', hideTooltip, true);

        }

        if (document.readyState === "loading") {
          document.addEventListener("DOMContentLoaded", initApp);
        } else {
          initApp();
        }
      })();
