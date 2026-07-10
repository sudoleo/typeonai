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
// through window.FREE_LIMIT (set in the Jinja <head> config block).
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
        window.consensusDifferencesSpinnerHTML = `
          <span class="thinking-wrap consensus-thinking-wrap" role="status" aria-live="polite" aria-busy="true">
            <span class="consensus-loader" aria-hidden="true"><span></span><span></span><span></span></span>
            <span class="thinking consensus-thinking">Comparing responses</span>
          </span>
        `;

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
        function hasVerifiedSession() {
          return Boolean(window.auth?.currentUser?.emailVerified);
        }

        window.userCanAskQuestions = function () {
          return hasVerifiedSession();
        };

        window.updateQuestionInputAccess = function () {
          const canAsk = window.userCanAskQuestions();
          const sendButton = document.getElementById("sendButton");

          if (questionInput) {
            questionInput.disabled = !canAsk;
            questionInput.placeholder = canAsk ? defaultQuestionPlaceholder : lockedQuestionPlaceholder;
            questionInput.setAttribute("aria-disabled", String(!canAsk));
          }

          if (sendButton && !sendButton.classList.contains("is-cancel-action")) {
            sendButton.disabled = !canAsk;
            sendButton.title = canAsk ? "Send question" : "Sign in to ask questions or use your own API keys";
          }

          return canAsk;
        };

        window.updateQuestionInputAccess();

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

        const MODE_EXPLAINER_STORAGE_KEY = "modeExplainerConfirmed";
        const MODE_EXPLAINER_TRIGGER_HIDDEN_KEY = "modeExplainerTriggerHidden";

        function initModeExplainer() {
          const explainer = document.getElementById("modeExplainer");
          const trigger = document.getElementById("modeExplainerTrigger");
          const dismiss = document.getElementById("modeExplainerDismiss");
          const hideTriggerCheckbox = document.getElementById("modeExplainerHideTrigger");
          if (!explainer || !trigger || !dismiss) return;

          const setExplainerVisible = (visible) => {
            explainer.hidden = !visible;
            explainer.classList.toggle("is-visible", visible);
            trigger.setAttribute("aria-expanded", String(visible));
            trigger.classList.toggle("is-active", visible);
          };

          const applyTriggerHidden = (hidden) => {
            trigger.classList.toggle("is-hidden", hidden);
          };

          // Standardmäßig eingeklappt – auch für neue Nutzer. Der Bereich wird nur
          // bewusst über den (i)-Trigger geöffnet.
          setExplainerVisible(false);

          const triggerHidden = localStorage.getItem(MODE_EXPLAINER_TRIGGER_HIDDEN_KEY) === "true";
          if (hideTriggerCheckbox) hideTriggerCheckbox.checked = triggerHidden;
          applyTriggerHidden(triggerHidden);

          trigger.addEventListener("click", () => {
            const nextVisible = explainer.hidden;
            setExplainerVisible(nextVisible);
            trackAppEvent("app_mode_help_toggled", { open: nextVisible });
          });

          dismiss.addEventListener("click", () => {
            localStorage.setItem(MODE_EXPLAINER_STORAGE_KEY, "true");
            setExplainerVisible(false);
            trackAppEvent("app_mode_help_confirmed");
          });

          hideTriggerCheckbox?.addEventListener("change", () => {
            const hidden = hideTriggerCheckbox.checked;
            localStorage.setItem(MODE_EXPLAINER_TRIGGER_HIDDEN_KEY, String(hidden));
            applyTriggerHidden(hidden);
            trackAppEvent("app_mode_help_trigger_hidden", { hidden });
          });
        }

        initModeExplainer();

        // deepThinkModelLabels stammt aus app-core.js (window.App), siehe Alias oben.

        function updateDeepThinkText() {
          const deepSearchToggle = document.getElementById("deepSearchToggle");

          const deepSearchActive = !!deepSearchToggle && deepSearchToggle.checked;

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

        // --- NEU: Pro Modal Referenzen ---
        const proModal = document.getElementById("proFeatureModal");
        const closeProBtn = document.getElementById("closeProModal");
        const keepFreeBtn = document.getElementById("keepFreeBtn");
        const upgradeBtn = document.getElementById("smokeTestUpgradeBtn");

        // Funktion zum Schließen des Modals
        function closeProModal() {
          proModal.style.display = "none";
        }

        // Event Listener für Schließen-Buttons
        if (closeProBtn) closeProBtn.addEventListener("click", closeProModal);
        if (keepFreeBtn) keepFreeBtn.addEventListener("click", closeProModal);

        // Pro-Modal mit Feature-Name öffnen ("Unlock Deep Think" / "Unlock
        // Resolve"). Gibt zurück, ob das Modal gezeigt werden konnte, damit
        // Aufrufer sonst auf ein Popup ausweichen können. Der Untertitel
        // passt sich dem geklickten Feature an (Fallback: generischer Text).
        const PRO_FEATURE_DESCRIPTIONS = {
          "Deep Think": "Complex reasoning requires advanced compute power. Upgrade to access the smartest AI models.",
          "Resolve": "Let the disagreeing models confront each other's position and see whether they revise or hold their ground.",
          "Follow-up questions": "Keep the conversation going — your previous question and its consensus answer travel along as context.",
        };
        const PRO_FEATURE_DESCRIPTION_FALLBACK = "Upgrade to unlock the full consens.io toolkit, including the smartest AI models.";
        window.App.showProFeatureModal = function (featureName) {
          const nameEl = document.getElementById("proModalFeatureName");
          if (nameEl && featureName) nameEl.textContent = featureName;
          const descEl = document.getElementById("proModalDescription");
          if (descEl) {
            descEl.textContent = PRO_FEATURE_DESCRIPTIONS[featureName] || PRO_FEATURE_DESCRIPTION_FALLBACK;
          }
          if (!proModal) return false;
          proModal.style.display = "block";
          return true;
        };

        // Klick außerhalb schließt Modal
        window.addEventListener("click", (event) => {
          if (event.target === proModal) {
            closeProModal();
          }
        });

        // smoke test
        if (upgradeBtn) {
          upgradeBtn.addEventListener("click", async (event) => {
            // 1. Verhindert, dass die Seite neu lädt
            event.preventDefault();


            // Zugriff über window.auth ist sicherer
            const currentUser = window.auth ? window.auth.currentUser : null;

            // --- ÄNDERUNG: Prüfung ZUERST durchführen ---
            if (currentUser) {
              // A) USER IST EINGELOGGT
              trackAppEvent("app_pro_interest_click", { logged_in: true });

              // UI Feedback (Erfolg)
              alert("Thanks for your support! We are still early in development and hard at work building this feature. We've noted your interest to help us prioritize it!");
              closeProModal();

              try {
                const id_token = await currentUser.getIdToken(false);

                await fetch("/track-interest", {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({
                    id_token: id_token,
                    source: "smoke_test_modal"
                  })
                });

                localStorage.setItem("proInterestShown", "true");

              } catch (err) {
                console.error("Tracking failed:", err);
              }
            } else {
              trackAppEvent("app_pro_interest_click", { logged_in: false });
              alert("Please log in to register your interest. We can only track your request if you are signed into your account.");
            }
          });
        }

        // --- DEEP THINK TOGGLE SPERRE ---
        document.getElementById("deepSearchToggle").addEventListener("click", function (event) {
          // Wir prüfen die globale Variable window.isUserPro
          if (!window.isUserPro) {
            event.preventDefault(); // Verhindert das Umschalten des Toggles
            trackAppEvent("app_deep_think_locked_click");

            // Modal anzeigen (mit passendem Feature-Namen im Header)
            if (!window.App.showProFeatureModal("Deep Think")) {
              alert("Deep Think is a Pro feature.");
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
            NORMAL: getConfiguredLimit("free_usage_limit", 25),
            DEEP: getConfiguredLimit("free_deep_search_limit", 12)
          },
          PRO: {
            NORMAL: getConfiguredLimit("pro_usage_limit", 500),
            DEEP: getConfiguredLimit("pro_deep_search_limit", 50)
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

            // Hier nutzen wir die GLOBALE Variable, die durch checkUserStatusOnLoad korrekt gesetzt wurde
            document.getElementById("freeUsageDisplay").innerHTML =
              "Requests: " + data.remaining + " / " + currentMaxLimit;

            document.getElementById("deepUsageDisplay").innerHTML =
              "Deep Think: " + data.deep_remaining + " / " + currentDeepLimit;

          } catch (e) { console.error(e); }
        }

        const SIDEBAR_OVERLAY_BREAKPOINT = 1549;

        function usesOverlaySidebar() {
          return window.matchMedia(`(max-width: ${SIDEBAR_OVERLAY_BREAKPOINT}px)`).matches;
        }

        function closeOverlaySidebar() {
          const sidebar = document.querySelector(".sidebar");
          if (!sidebar || !usesOverlaySidebar()) return;
          sidebar.classList.remove("active");
          sidebar.classList.add("collapsed");
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

        // Toggle FAQ items with icons
        document.querySelectorAll('.faq-item h3').forEach((question) => {
          question.addEventListener('click', () => {
            const answer = question.nextElementSibling;
            const icon = question.querySelector('.faq-toggle-icon');
            if (!answer.style.display || answer.style.display === 'none') {
              answer.style.display = 'block';
              icon.textContent = '－';
            } else {
              answer.style.display = 'none';
              icon.textContent = '＋';
            }
          });
        });

        // Standardmäßig Antworten ausgeblendet (optional)
        document.querySelectorAll('.faq-item h3').forEach((question) => {
          question.addEventListener('click', () => {
            const answer = question.nextElementSibling;
            const icon = question.querySelector('.faq-toggle-icon');
            if (answer && icon) {
              icon.textContent = answer.style.display === 'block' ? '-' : '+';
            }
          });
        });

        document.querySelectorAll('.faq-item p').forEach((answer) => {
          answer.style.display = 'none';
        });

        const APP_LIMITS = window.APP_LIMITS || {};
        const FREE_USAGE_LIMIT = APP_LIMITS.free_usage_limit || Number(window.FREE_LIMIT);

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
          if (window.isQueryRequestRunning && window.isQueryRequestRunning()) return;
          if (!validateInputText()) {
            e.preventDefault();
          }
        });

        // Event-Listener für Eingabefelder und Buttons
        // Frage per Enter (ohne Zeilenumbruch) absenden
        document.getElementById("questionInput").addEventListener("keydown", function (event) {
          if (event.key === "Enter" && !event.shiftKey) {
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

        document.getElementById("toggleSidebarButton").addEventListener("click", function () {
          const sidebar = document.querySelector(".sidebar");
          if (!sidebar) return;

          if (usesOverlaySidebar()) {
            const shouldOpen = !sidebar.classList.contains("active");
            sidebar.classList.toggle("active", shouldOpen);
            sidebar.classList.toggle("collapsed", !shouldOpen);
          } else {
            sidebar.classList.toggle("collapsed");
            sidebar.classList.remove("active");
          }
          updateToggleButton();
          trackAppEvent("app_sidebar_toggle", { open: !sidebar.classList.contains("collapsed") });
        });

        // Fenstergröße prüfen – wenn <1024px, Sidebar einklappen
        function checkWindowSize() {
          const sidebar = document.querySelector(".sidebar");
          if (!sidebar) return;

          if (usesOverlaySidebar()) {
            sidebar.classList.add("collapsed");
            sidebar.classList.remove("active");
          } else {
            sidebar.classList.remove("collapsed");
            sidebar.classList.remove("active");
          }
          updateToggleButton();
        }
        window.addEventListener("resize", checkWindowSize);
        checkWindowSize(); // Initial

        // Aktualisiert den Pfeil des Sidebar-Toggle-Buttons
        function updateToggleButton() {
          const sidebar = document.querySelector(".sidebar");
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

        function renderEvidenceSources(sources) {
          const container = document.getElementById("evidenceContainer");
          const listEl = document.getElementById("evidenceList");

          if (!container || !listEl) {
            // Wenn du das UI noch nicht eingebaut hast, einfach leise aussteigen
            return;
          }

          listEl.innerHTML = "";

          if (!sources || !sources.length) {
            container.style.display = "none";
            return;
          }

          sources.forEach((src, idx) => {
            const li = document.createElement("li");
            li.className = "evidence-item";

            const titleRow = document.createElement("div");
            titleRow.className = "evidence-title-row";

            const indexSpan = document.createElement("span");
            indexSpan.className = "evidence-index";
            indexSpan.textContent = (idx + 1) + ". ";

            const link = document.createElement("a");
            link.href = src.url || "#";
            link.target = "_blank";
            link.rel = "noopener noreferrer";
            link.textContent = src.title || src.url || "Source " + (idx + 1);

            titleRow.appendChild(indexSpan);
            titleRow.appendChild(link);
            li.appendChild(titleRow);

            if (src.snippet || src.text) {
              const snippetDiv = document.createElement("div");
              snippetDiv.className = "evidence-snippet";
              snippetDiv.textContent = src.snippet || src.text;
              li.appendChild(snippetDiv);
            }

            listEl.appendChild(li);
          });

          container.style.display = "block";
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

        // Modelle-Auswahl umschalten (für den Pfeil in der Modelle Section)
        window.toggleModelSelection = function () {
          const section = document.querySelector(".models-section");
          const area = document.getElementById("modelSelectionArea");
          const toggle = document.getElementById("toggleModelSelection");
          if (!section || !area || !toggle) return;

          const isCollapsed = section.classList.toggle("is-collapsed");
          area.classList.toggle("hidden", isCollapsed);
          toggle.setAttribute("aria-expanded", String(!isCollapsed));
          trackAppEvent("app_sidebar_section_toggled", { section: "models", open: !isCollapsed });
        };

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
            const selectedLabel = getModelOptionLabel(this.options[this.selectedIndex]) || this.value;
            trackAppEvent("app_consensus_model_changed", { model: selectedLabel });

            // Optional: Fokus entfernen, wie bei den anderen Pickern
            this.blur();
          });
        }

        let leaderboardInterval; // Variable zum Speichern des Interval-IDs

        window.toggleLeaderboard = function () {
          const section = document.querySelector(".leaderboard-section");
          const container = document.getElementById("leaderboardContentContainer");
          const toggle = document.getElementById("toggleLeaderboard");
          if (!section || !container || !toggle) return;

          const isCollapsed = section.classList.toggle("is-collapsed");
          container.classList.toggle("hidden", isCollapsed);
          toggle.setAttribute("aria-expanded", String(!isCollapsed));
          trackAppEvent("app_sidebar_section_toggled", { section: "leaderboard", open: !isCollapsed });
        };

        // Chat search filtering
        document.getElementById("chatSearch")?.addEventListener("input", function () {
          const q = this.value.trim().toLowerCase();
          document.querySelectorAll("#bookmarksContainer .bookmark").forEach(el => {
            const text = el.querySelector("p")?.textContent?.toLowerCase() || "";
            el.style.display = q && !text.includes(q) ? "none" : "";
          });
        });

        window.toggleBookmarks = function () {
          const section = document.querySelector(".bookmarks-section");
          const container = document.getElementById("bookmarksContainer");
          const toggle = document.getElementById("bookmarksToggle");
          if (!section || !container || !toggle) return;

          const isCollapsed = section.classList.toggle("is-collapsed");
          container.classList.toggle("hidden", isCollapsed);
          toggle.setAttribute("aria-expanded", String(!isCollapsed));
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

        initCustomModelPicker(document.getElementById("consensusModelDropdown"));



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

        window.clearResponseBoxes = function () {
          trackAppEvent("app_responses_cleared");
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
            const mainElement = consensusBox.querySelector(".consensus-main p");
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
        updatePremiumModelsState(false, false);

        window.isUserPro = false;
        window.isUserEarly = false;

        // Tier-/Pro-UI (updateUserTierUI, updatePremiumModelsState) ist nach
        // static/js/user-tier.js ausgelagert. Exporte gleichen Namens auf window.

        // Event Listener für den neuen Upgrade-Link im Header
        const headerUpgradeLink = document.getElementById("upgradeLink");

        if (headerUpgradeLink) {
          headerUpgradeLink.addEventListener("click", function (e) {
            e.preventDefault(); // Verhindert Springen nach oben (#)
            const modal = document.getElementById("proFeatureModal");
            if (modal) {
              modal.style.display = "block";
            }
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
