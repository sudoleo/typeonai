// =====================================================================
// app-core.js
// Geteilte Basis (Uebergangsbus window.App) fuer die ausgelagerten
// Feature-Module und die verbleibende initApp-Closure.
// Haelt zentrale Config (modelPrefs, deepThinkModelLabels) und
// cross-cutting Helfer (getModelOptionLabel, getSelectedModelCount,
// trackAppEvent). MUSS vor den Feature-Modulen geladen werden.
//
// Hinweis: window.App ist bewusst ein TEMPORAERER Bus, um die Cluster
// schrittweise aus index.html zu loesen. Der echte State-Refactor folgt
// spaeter (DOM-als-State aufloesen).
// =====================================================================

(function () {
  window.App = window.App || {};

  // Telemetrie-Wrapper (Guard um window.trackUmamiEvent).
  function trackAppEvent(eventName, eventData = {}) {
    if (typeof window.trackUmamiEvent === "function") {
      window.trackUmamiEvent(eventName, eventData);
    }
  }

  function getSelectedModelCount() {
    return modelPrefs.filter(pref => document.getElementById(pref.checkId)?.checked).length;
  }

  const DEFAULT_APP_TITLE = "Compare AI Answers | consens.io";

  function setAppTitle(question = "") {
    const normalized = String(question || "").replace(/\s+/g, " ").trim();
    if (!normalized) {
      document.title = DEFAULT_APP_TITLE;
      return;
    }

    const maxQuestionLength = 64;
    const shortened = normalized.length > maxQuestionLength
      ? `${normalized.slice(0, maxQuestionLength - 1).trimEnd()}…`
      : normalized;
    document.title = `${shortened} | consens.io`;
  }

  // Anhaenge der sichtbaren Frage. Sie gehoeren zu der Nachricht, mit der sie
  // rausgegangen sind — eine neue Frage erbt sie nicht. Gerendert werden sie
  // von attachments.js, das die Chip-Optik besitzt.
  let threadAttachments = [];

  function getThreadAttachments() {
    return threadAttachments.slice();
  }

  function setThreadQuestionAttachments(attachmentsMeta) {
    threadAttachments = (Array.isArray(attachmentsMeta) ? attachmentsMeta : [])
      .filter(item => item && item.name)
      .map(item => ({
        name: String(item.name),
        mime: String(item.mime || ""),
        size: Number(item.size) || 0
      }));
    const row = document.getElementById("threadAskAttachments");
    if (!row) return;
    if (typeof window.App.attachments?.renderMessageAttachments === "function") {
      window.App.attachments.renderMessageAttachments(row, threadAttachments);
      return;
    }
    row.innerHTML = "";
    row.hidden = true;
  }

  // Beide Fragen-Koepfe im Thread teilen sich Optik und Aufklapp-Logik: der
  // aktive (#threadAsk) und der der gerade abgeschickten, noch nicht
  // uebernommenen Nachricht (#threadPendingAsk). Leerer Text versteckt den
  // Block wieder. Lange Fragen clampen per CSS auf drei Zeilen; is-long
  // schaltet den Aufklapp-Link frei, is-open hebt den Clamp auf.
  function renderThreadQuestion(wrap, text, question) {
    if (!wrap || !text) return "";

    const normalized = String(question || "").replace(/\s+/g, " ").trim();
    const unchanged = text.textContent === normalized;
    // Die Multi-Run-Projektion schreibt den sichtbaren Context waehrend des
    // Streamings regelmaessig neu ins DOM. Eine identische Frage ist dabei
    // kein neuer Turn: ihren lokalen Disclosure-State zurueckzusetzen liess
    // "Show full question" unter dem Mauszeiger flackern und klappte einen
    // erfolgreichen Klick beim naechsten Stream-Update sofort wieder zu.
    // Nur neuer Inhalt initialisiert Clamp und Link deshalb von vorn.
    if (unchanged) {
      wrap.hidden = !normalized;
      if (normalized) observeThreadAskWidth(wrap, text);
      return normalized;
    }

    text.textContent = normalized;
    wrap.hidden = !normalized;
    wrap.classList.remove("is-open", "is-long");
    const more = wrap.querySelector(".thread-ask-more");
    if (more) {
      more.textContent = "Show full question";
      more.setAttribute("aria-expanded", "false");
    }
    if (!normalized) return "";

    requestAnimationFrame(() => syncThreadAskClamp(wrap, text));
    observeThreadAskWidth(wrap, text);
    return normalized;
  }

  // Kopf des Threads (#threadAsk): zeigt die gestellte Frage über dem Lauf.
  // Leerer Text versteckt den Block wieder (New comparison, Clear).
  function setThreadQuestion(question = "") {
    // Wer den Kopf setzt, hat die schwebende Nachricht uebernommen (oder den
    // Thread ganz geraeumt) — in beiden Faellen ist die Blase erledigt. Das
    // gilt auch fuer Aufrufer ausserhalb des Sendepfads (Bookmark-Restore,
    // "New comparison", Direktvergleich), damit sie nie stehen bleibt.
    clearPendingThreadQuestion();
    const wrap = document.getElementById("threadAsk");
    const text = document.getElementById("threadAskText");
    if (!wrap || !text) return;
    // Eine neue Frage beginnt ohne Anhaenge; wer welche mitschickt, meldet sie
    // direkt nach dem Senden ueber setThreadQuestionAttachments an.
    setThreadQuestionAttachments([]);
    renderThreadQuestion(wrap, text, question);
  }

  // ---- Die gerade abgeschickte Nachricht -----------------------------------
  // Sie steht sofort im Thread, nicht erst wenn der Lauf sie zum Kopf des
  // neuen Turns macht: dazwischen liegen /prepare und, im laufenden Gespraech,
  // das Binden des Chat-Kontexts. Das sind Sekunden, in denen frueher nichts
  // passierte — die Frage stand unveraendert im Feld, der Thread zeigte
  // weiter die vorige. Der Vorgaenger bleibt dabei unangetastet: erst wenn der
  // Lauf wirklich stattfindet, wandert er in den Verlauf.
  const PENDING_MESSAGE_CLASS = "thread-message-pending";

  function setPendingThreadQuestion(question = "", attachmentsMeta = []) {
    const wrap = document.getElementById("threadPendingAsk");
    const text = document.getElementById("threadPendingAskText");
    const normalized = renderThreadQuestion(wrap, text, question);
    const row = document.getElementById("threadPendingAskAttachments");
    if (row) {
      const renderer = window.App.attachments?.renderMessageAttachments;
      if (normalized && typeof renderer === "function") {
        renderer(row, attachmentsMeta);
      } else {
        row.replaceChildren();
        row.hidden = true;
      }
    }
    document.body.classList.toggle(PENDING_MESSAGE_CLASS, !!normalized);
    return !!normalized;
  }

  function clearPendingThreadQuestion() {
    document.body.classList.remove(PENDING_MESSAGE_CLASS);
    const wrap = document.getElementById("threadPendingAsk");
    const text = document.getElementById("threadPendingAskText");
    const row = document.getElementById("threadPendingAskAttachments");
    if (text) text.textContent = "";
    if (wrap) {
      wrap.hidden = true;
      wrap.classList.remove("is-open", "is-long");
    }
    if (row) {
      row.replaceChildren();
      row.hidden = true;
    }
  }

  // ---- Die abgeschickte Nachricht ins Bild holen ---------------------------
  // Genau EINE Bewegung, und zwar die, die der Nutzer selbst ausgeloest hat:
  // der Klick auf Senden. Danach scrollt hier nichts mehr von allein — ein
  // Thread, der beim Lesen unter den Fingern wegwandert, ist schlimmer als
  // eine Antwort, die man selbst nach unten holt. Deshalb drei Schranken:
  // die Bewegung geht NIE nach oben, sie unterbleibt, wenn das Ziel ohnehin
  // fast im Bild steht, und sie bricht bei der ersten eigenen Geste (Rad,
  // Finger, Taste) sofort ab, statt dagegen zu ziehen.
  const REVEAL_TOP_GAP = 76;      // Luft fuer die schwebende Navigation
  const REVEAL_MIN_DISTANCE = 24; // darunter waere es Zappeln, keine Bewegung
  const REVEAL_DURATION = 420;
  const REVEAL_INTERRUPTS = ["wheel", "touchstart", "keydown", "pointerdown"];
  const reducedMotionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");

  let revealFrame = 0;
  let revealRelease = null;

  function stopSentMessageReveal() {
    if (revealFrame) window.cancelAnimationFrame(revealFrame);
    revealFrame = 0;
    const release = revealRelease;
    revealRelease = null;
    if (release) release();
  }

  // Gemessen wird erst nach dem Layout, und zwar nach dem zweiten Frame: im
  // ersten wird die Blase sichtbar, im zweiten steht fest, ob die Frage
  // geklammert ist (das schaltet den "Show full question"-Link zu und aendert
  // damit die Hoehe). Vorher gemessen, zielte die Bewegung daneben.
  function revealSentMessage(element) {
    stopSentMessageReveal();
    revealFrame = window.requestAnimationFrame(() => {
      revealFrame = window.requestAnimationFrame(() => {
        revealFrame = 0;
        startSentMessageReveal(element);
      });
    });
  }

  function startSentMessageReveal(element) {
    const el = element
      || document.getElementById("threadPendingAsk")
      || document.getElementById("threadAsk");
    if (!el || el.hidden) return;

    // Der Boden des Dokuments ist die Grenze: weiter als bis dorthin laesst
    // sich nicht scrollen, und ein Ziel dahinter wuerde die Bewegung im
    // Nichts enden lassen.
    const maxTop = Math.max(
      0,
      document.documentElement.scrollHeight - window.innerHeight
    );
    const wanted = el.getBoundingClientRect().top + window.scrollY - REVEAL_TOP_GAP;
    const from = window.scrollY;
    const distance = Math.max(0, Math.min(wanted, maxTop)) - from;
    if (distance < REVEAL_MIN_DISTANCE) return;

    if (reducedMotionQuery.matches) {
      window.scrollTo(0, from + distance);
      return;
    }

    const startedAt = (window.performance?.now?.() ?? Date.now());
    const interrupt = () => stopSentMessageReveal();
    REVEAL_INTERRUPTS.forEach(name => {
      window.addEventListener(name, interrupt, { passive: true, capture: true });
    });
    revealRelease = () => REVEAL_INTERRUPTS.forEach(name => {
      window.removeEventListener(name, interrupt, { capture: true });
    });

    const step = (now) => {
      revealFrame = 0;
      const elapsed = (now || (window.performance?.now?.() ?? Date.now())) - startedAt;
      const progress = Math.min(1, Math.max(0, elapsed / REVEAL_DURATION));
      const eased = 1 - Math.pow(1 - progress, 3);
      window.scrollTo(0, Math.round(from + distance * eased));
      if (progress < 1) {
        revealFrame = window.requestAnimationFrame(step);
        return;
      }
      stopSentMessageReveal();
    };
    revealFrame = window.requestAnimationFrame(step);
  }

  // Ob eine Frage laenger als drei Zeilen ist, haengt an der Breite des
  // Blocks - und die steht im ersten Frame noch nicht fest: Der Ausstieg aus
  // dem Hero animiert den Container, und die Sidebar aendert ihn spaeter noch
  // einmal. Wurde nur einmal gemessen, blieb "Show full question" bei einer
  // langen Frage aus und die vierte Zeile verschwand lautlos - beim
  // gefuehrten Lauf ausgerechnet das Ende der Frage. Deshalb misst ein
  // ResizeObserver nach jeder Groessenaenderung nach — einer je Fragen-Kopf
  // (der aktive und der der gerade abgeschickten Nachricht), sonst zoege der
  // zweite Kopf am Beobachter des ersten vorbei.
  const threadAskResizeObservers = new WeakMap();

  function syncThreadAskClamp(wrap, text) {
    // Aufgeklappt gibt es nichts zu messen: dort ist scrollHeight gleich
    // clientHeight, und die Marke wuerde sich selbst zuruecknehmen.
    if (wrap.hidden || wrap.classList.contains("is-open")) return;
    const clamped = text.clientHeight;
    // scrollHeight allein reicht nicht: An einem geklammerten Block meldet er
    // je nach Zeitpunkt die geklammerte statt der vollen Hoehe - mal 4 Zeilen,
    // mal 3. Deshalb wird der Clamp fuer die Messung kurz aufgehoben. Das
    // passiert innerhalb eines Frames, es wird also nichts davon gezeichnet.
    const previous = text.style.webkitLineClamp;
    text.style.webkitLineClamp = "unset";
    const full = text.scrollHeight;
    text.style.webkitLineClamp = previous;
    wrap.classList.toggle("is-long", full > clamped + 2);
  }

  function observeThreadAskWidth(wrap, text) {
    if (typeof ResizeObserver !== "function" || threadAskResizeObservers.has(text)) return;
    const observer = new ResizeObserver(() => syncThreadAskClamp(wrap, text));
    observer.observe(text);
    threadAskResizeObservers.set(text, observer);
  }

  // Dieselbe Geste fuer die aktive Frage (#threadAskMore) und fuer jede
  // archivierte im Verlauf: der Link gehoert immer zu der Frage, unter der er
  // steht, deshalb wird der Umschalter aus dem geklickten Knopf abgeleitet.
  document.addEventListener("click", (event) => {
    const more = event.target.closest(".thread-ask-more");
    if (!more) return;
    const wrap = more.closest(".thread-ask, .thread-history-question");
    if (!wrap) return;
    const open = wrap.classList.toggle("is-open");
    more.textContent = open ? "Collapse question" : "Show full question";
    more.setAttribute("aria-expanded", String(open));
  });

  // Definition der Modelle und IDs (zentral, von mehreren Clustern genutzt).
  // EINE Quelle: der Server liefert die Familien aus cfg.PROVIDERS in
  // window.MODEL_FAMILIES; hier werden sie nur auf die im Frontend
  // etablierten Feldnamen gebracht. Eine neue Familie erscheint damit
  // ueberall, ohne dass eine dieser Listen nachgezogen werden muss.
  const modelFamilies = Array.isArray(window.MODEL_FAMILIES) ? window.MODEL_FAMILIES : [];
  const modelPrefs = modelFamilies.map(family => ({
    key: family.label,
    provider: family.provider,
    label: family.title,
    shortLabel: family.shortLabel,
    citationLabel: family.citationLabel || family.label,
    checkId: family.checkboxId,
    selectId: family.selectId,
    responseId: family.responseId,
    textId: family.textId,
    endpoint: family.endpoint,
    handlesAttachments: family.handlesAttachments !== false
  }));

  // Hoechstzahl gleichzeitig laufender Familien (Serverregel, siehe
  // cfg.MAX_RUN_FAMILIES): mehr Familien duerfen konfiguriert sein, ein Lauf
  // bleibt trotzdem ein Sechs-Modell-Vergleich.
  const maxRunFamilies = Number(window.MAX_RUN_FAMILIES) > 0
    ? Number(window.MAX_RUN_FAMILIES)
    : 6;

  const deepThinkModelLabels = Object.fromEntries(
    modelFamilies
      .filter(family => family.deepThinkLabel)
      .map(family => [family.label, family.deepThinkLabel])
  );

  function getModelOptionLabel(option) {
    const explicitLabel = option?.dataset?.modelLabel;
    if (explicitLabel) return explicitLabel;
    return (option?.textContent || "").replace(/(?:\s*(?:Â·|·)\s*Pro)+$/i, "").trim();
  }

  // Einziges Renderziel des Konsenstextes. Frueher war das ein einzelnes <p>
  // unter .consensus-main, adressiert per ".consensus-main p" an einem guten
  // Dutzend Stellen. Das Inline-Marker-Rendering braucht einen stabilen
  // Blockcontainer, deshalb laeuft jeder Zugriff jetzt ueber diesen Helfer.
  // Der scope-Parameter erlaubt es, gezielt in einer bestimmten Konsens-Box zu
  // suchen (z. B. der von getConsensus gehaltenen Referenz).
  function consensusBodyEl(scope) {
    const root = scope || document;
    return (
      root.querySelector?.("#consensusAnswerBody")
      || root.querySelector?.(".consensus-main .consensus-answer-body")
      || null
    );
  }

  // Kurzlebiges Hinweis-Popup (cross-cutting UI-Helfer, von vielen Clustern genutzt).
  function showPopup(message) {
    const popup = document.createElement('div');
    popup.className = 'explanation-popup';
    popup.innerText = message;
    document.body.appendChild(popup);

    setTimeout(() => {
      popup.style.opacity = '1';
    }, 100);

    setTimeout(() => {
      popup.style.opacity = '0';
      setTimeout(() => {
        popup.remove();
      }, 300);
    }, 3000);
  }

  // Desktop-Schwelle des Hero-CSS (components-input.css): ab hier sind die
  // Response-Boxen ohne Agent Mode schon vor der ersten Frage sichtbar.
  const heroDesktopQuery = window.matchMedia("(min-width: 1100px)");

  // Haelt inert/aria-hidden der .response-section synchron zur CSS-Sichtbarkeit
  // im Hero: verborgen nur, wenn der Hero zentriert ist (Agent Mode aktiv oder
  // kein Desktop). Wird auch von agent-mode.js (updateAgentModeUI) gerufen.
  function syncHeroResponseAccess() {
    const responses = document.querySelector(".response-section");
    if (!responses) return;
    const directComparisonActive = document.body.classList.contains("direct-comparison-active");
    const hiddenInHero =
      document.body.classList.contains("is-hero") &&
      !directComparisonActive &&
      (document.body.classList.contains("agent-mode-enabled") || !heroDesktopQuery.matches);
    responses.inert = hiddenInHero;
    if (hiddenInHero) {
      responses.setAttribute("aria-hidden", "true");
    } else {
      responses.removeAttribute("aria-hidden");
    }
  }

  if (typeof heroDesktopQuery.addEventListener === "function") {
    heroDesktopQuery.addEventListener("change", syncHeroResponseAccess);
  }
  syncHeroResponseAccess();

  // Der Thread ist das Gegenteil der Vergleichsflaeche: wer den Hero verlaesst,
  // verlaesst auch den Direktvergleich. Die Marke haengen zu lassen waere eine
  // Mine — sie steuert Sichtbarkeit und inert der .response-section.
  function exitHeroMode() {
    document.body.classList.remove("is-hero", "direct-comparison-active");
    syncHeroResponseAccess();
  }

  // Der Direktvergleich (Agent Mode aus) ist keine Zwischenstufe des Threads,
  // sondern eine eigene Ansicht: Composer oben, sechs Antworten darunter, kein
  // Thread-Kopf und kein Consensus. Ein frisch gesendeter Vergleich und ein aus
  // einem Bookmark wiederhergestellter muessen dieselbe Ansicht ergeben —
  // deshalb steht sie hier einmal statt zweimal (query-send.js, firebase.js).
  function enterDirectComparisonView() {
    document.body.classList.add("is-hero", "direct-comparison-active");
    setThreadQuestion("");
    syncHeroResponseAccess();
  }

  window.exitHeroMode = exitHeroMode;
  window.enterDirectComparisonView = enterDirectComparisonView;
  window.syncHeroResponseAccess = syncHeroResponseAccess;

  // Einziger Renderer fuer die Usage-Zeilen. API-Antworten ohne vollstaendige
  // Usage-Felder duerfen die zuletzt bekannte Anzeige nicht mit Fallback-Nullen
  // ueberschreiben. Das <strong>-Element ist zugleich Teil des Layout-Vertrags:
  // Label links, tabellarischer Wert rechts.
  function renderUsageDisplay({
    remaining,
    deepRemaining,
    totalLimit = window.currentMaxLimit,
    deepLimit = window.currentDeepLimit
  } = {}) {
    function renderLine(elementId, label, value, limit) {
      if (value === undefined || value === null) return;
      if (limit === undefined || limit === null) return;

      const element = document.getElementById(elementId);
      if (!element) return;

      const strong = document.createElement("strong");
      strong.textContent = value === "Unlimited" ? "Unlimited" : `${value} / ${limit}`;
      element.replaceChildren(document.createTextNode(`${label}: `), strong);
    }

    renderLine("freeUsageDisplay", "Runs", remaining, totalLimit);
    renderLine("deepUsageDisplay", "Deep Think", deepRemaining, deepLimit);
  }

  // Ein logischer UI-Lauf teilt genau einen serverseitigen Idempotency-Key
  // zwischen /prepare, allen parallelen /ask_* und /consensus. Kosten oder
  // Modellanzahl kommen bewusst nicht aus dem Client.
  const usageRun = {
    current: null,
    start(deepThink, useOwnKeys) {
      let key = null;
      if (!useOwnKeys) {
        key = globalThis.crypto?.randomUUID?.();
        if (!key) {
          key = `${Date.now()}-${Math.random().toString(16).slice(2)}-${Math.random().toString(16).slice(2)}`;
        }
      }
      this.current = {
        key,
        deepThink: deepThink === true,
        useOwnKeys: useOwnKeys === true,
        status: useOwnKeys ? "own_keys" : "new"
      };
      return this.current;
    },
    ensure(deepThink, useOwnKeys) {
      if (
        !this.current
        || this.current.deepThink !== (deepThink === true)
        || this.current.useOwnKeys !== (useOwnKeys === true)
      ) {
        return this.start(deepThink, useOwnKeys);
      }
      return this.current;
    },
    mark(status) {
      if (this.current && status) this.current.status = status;
    },
    clear() {
      this.current = null;
    }
  };

  Object.assign(window.App, {
    modelPrefs,
    maxRunFamilies,
    deepThinkModelLabels,
    getModelOptionLabel,
    getSelectedModelCount,
    setAppTitle,
    setThreadQuestion,
    setThreadQuestionAttachments,
    setPendingThreadQuestion,
    clearPendingThreadQuestion,
    revealSentMessage,
    getThreadAttachments,
    consensusBodyEl,
    trackAppEvent,
    showPopup,
    exitHeroMode,
    enterDirectComparisonView,
    syncHeroResponseAccess,
    renderUsageDisplay,
    usageRun
  });
})();
