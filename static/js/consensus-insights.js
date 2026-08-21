// =====================================================================
// consensus-insights.js
// Extrahiert aus templates/index.html (zweiter Inline-Script-Block).
// Credibility-Frames, Consensus-Insights-Popover, Spalten-Balancer.
// Kommuniziert ausschliesslich ueber window.* (siehe Exporte am Ende).
// Abhaengigkeiten (call-time): window.isAgentModeEnabled, window.setAgentMode,
// window.trackUmamiEvent.
// =====================================================================
        (function () {
          const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent)
            || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1); // iPadOS
          if (isIOS) document.documentElement.classList.add('ios');
        })();

        const credibilityFrameClasses = ["cred-very", "cred-largely", "cred-partially", "cred-hardly", "cred-not"];

        function getCredibilityClass(md) {
          if (!md) return null;
          const plain = String(md)
            .replace(/<[^>]*>/g, " ")
            .replace(/\*\*/g, "")
            .replace(/\s+/g, " ")
            .trim()
            .toLowerCase();
          const mapping = [
            { pattern: /the consensus answer is very credible/, cls: "cred-very" },
            { pattern: /the consensus answer is largely credible/, cls: "cred-largely" },
            { pattern: /the consensus answer is partially credible/, cls: "cred-partially" },
            { pattern: /the consensus answer is hardly credible/, cls: "cred-hardly" },
            { pattern: /the consensus answer is not credible/, cls: "cred-not" }
          ];

          for (const m of mapping) {
            if (m.pattern.test(plain)) return m.cls;
          }
          return null;
        }

        function resetCredibilityFrame(target) {
          // Strukturierte Auswertung (Verdict, Badges, Karten) immer mit zurücksetzen —
          // alle Aufrufer (Cancel, Fehler, neuer Lauf, Bookmarks, Demo) wollen beides.
          window.resetConsensusInsights?.();
          const frame = target?.classList?.contains("consensus-differences")
            ? target
            : target?.closest?.(".consensus-differences");
          if (!frame) return;
          frame.classList.remove("credibility-framed", ...credibilityFrameClasses);
        }

        function applyCredibilityFrame(target, md) {
          const frame = target?.classList?.contains("consensus-differences")
            ? target
            : target?.closest?.(".consensus-differences");
          if (!frame) return;
          resetCredibilityFrame(frame);
          const cls = getCredibilityClass(md);
          if (cls) {
            frame.classList.add("credibility-framed", cls);
          }
        }

        function colorizeCredibility(md) {
          if (!md) return md;
          const lines = md.trim().split("\n");
          if (!lines.length) return md;

          const first = lines[0].toLowerCase();
          const mapping = [
            { key: " very ", cls: "cred-very" },
            { key: " largely ", cls: "cred-largely" },
            { key: " partially ", cls: "cred-partially" },
            { key: " hardly ", cls: "cred-hardly" },
            { key: " not ", cls: "cred-not" }
          ];

          // Robust: prüft auf vollständige Satzfragmente
          let cls = null;
          if (first.includes("the consensus answer is")) {
            for (const m of mapping) {
              if (first.includes(m.key.trim())) { cls = m.cls; break; }
            }
          }

          if (cls) {
            // Markdown-Bold **…** in der Badge vermeiden, damit die Farbe klar lesbar bleibt
            lines[0] = lines[0].replace(/\*\*/g, "");
            lines[0] = `<span class="cred-badge ${cls}">${lines[0]}</span>`;
            return lines.join("\n");
          }
          return md;
        }

        window.getCredibilityClass = getCredibilityClass;
        window.applyCredibilityFrame = applyCredibilityFrame;
        window.resetCredibilityFrame = resetCredibilityFrame;
        window.colorizeCredibility = colorizeCredibility;

        // =====================================================================
        // Consensus Insights: Verdict-Header, Agreement-Badges, Differences-
        // Karten. Gespeist aus differences_data (strukturierte Ausgabe des
        // Differences-Calls). Fällt ohne Daten auf den Freitext-Pfad zurück.
        // =====================================================================
        (function () {
          const MODEL_BOX_IDS = {
            OpenAI: "openaiResponse",
            Mistral: "mistralResponse",
            Anthropic: "claudeResponse",
            Gemini: "geminiResponse",
            DeepSeek: "deepseekResponse",
            Grok: "grokResponse"
          };

          function $(id) { return document.getElementById(id); }

          function modelDisplayName(model) {
            const box = $(MODEL_BOX_IDS[model] || "");
            return (box && box.dataset.shortLabel) || model;
          }

          const LIVE_ANSWER_NAVIGATION = {
            canOpen: function (model) {
              return !!$(MODEL_BOX_IDS[model] || "");
            },
            open: function (model, quote) {
              jumpToModelAnswer(model, quote);
            }
          };

          function isMobileViewport() {
            return window.matchMedia("(max-width: 768px)").matches;
          }

          // --- Markdown-Auszeichnung entfernen -------------------------------
          // Anker und Zitate sind woertliche Kopien aus dem MARKDOWN-QUELLTEXT
          // der Antworten ("1. **Weltklasse:** ca. 1.300 Watt"). Gesucht und
          // angezeigt wird aber der GERENDERTE Text, in dem die Sternchen
          // laengst ein <strong> sind. Ein solcher Anker fand deshalb nie seine
          // Stelle im Konsens - und landete mitsamt sichtbarer Sternchen in der
          // Fallback-Liste "Key claims". Hier faellt dieselbe Auszeichnung weg,
          // die auch der Renderer schluckt.
          function inlineMarkdownSource(value) {
            return String(value || "")
              // Ein Anker aus einem Listenelement enthaelt haeufig dessen
              // Markdown-Zaehler. Im gerenderten <li> ist er kein Textknoten.
              .replace(/(^|\n)[ \t]*(?:>[ \t]*)*(?:[-*+][ \t]+|\d+[.)][ \t]+|#{1,6}[ \t]+)?/g, "$1")
              .trim();
          }

          function stripMarkdown(value) {
            const source = inlineMarkdownSource(value);
            if (!source) return "";

            // Dieselbe Markdown-Engine wie fuer den Konsens liefert die
            // tatsaechlich sichtbare Textform. So bleiben literale Sternchen
            // erhalten, waehrend **fett** und Links korrekt reduziert werden.
            if (window.marked?.parseInline && window.DOMPurify?.sanitize) {
              const template = document.createElement("template");
              template.innerHTML = window.DOMPurify.sanitize(
                window.marked.parseInline(source),
                { ALLOWED_TAGS: ["strong", "em", "del", "code", "br"], ALLOWED_ATTR: [] }
              );
              return (template.content.textContent || "").replace(/[ \t]{2,}/g, " ").trim();
            }

            // Defensive Degradation, falls eine CDN-Library nicht geladen ist.
            return source
              .replace(/!?\[([^\]]*)\]\([^)]*\)/g, "$1")
              .replace(/\*\*\*|\*\*|___|__|~~|`/g, "")
              .replace(/\*/g, "")
              .replace(/(^|[\s(["'])_([^_\n]+)_(?=$|[\s).,;:!?\]"'])/g, "$1$2")
              .replace(/[ \t]{2,}/g, " ")
              .trim();
          }

          function renderInlineMarkdown(element, value, prefix, suffix) {
            if (!element) return;
            // Anker aus aelteren Laeufen koennen eine abgesetzte Formel ohne
            // ihre "$$"-Zeilen enthalten. Ohne Trennzeichen bleibt sie
            // Quelltext; die Erkennung greift nur bei reinem LaTeX.
            const raw = inlineMarkdownSource(value);
            const source = window.ConsensusMath?.wrapBareLatex?.(raw) || raw;
            const before = prefix || "";
            const after = suffix || "";
            element.textContent = before;

            if (source && window.marked?.parseInline && window.DOMPurify?.sanitize) {
              const content = document.createElement("span");
              const hasEscapedLiteralStar = /\\\*/.test(source);
              const prepared = window.ConsensusMath?.prepareMarkdown
                ? window.ConsensusMath.prepareMarkdown(source)
                : source;
              content.innerHTML = window.DOMPurify.sanitize(
                window.marked.parseInline(prepared),
                { ALLOWED_TAGS: ["strong", "em", "del", "code", "br"], ALLOWED_ATTR: [] }
              );

              // Bereits gespeicherte Anker koennen aus einem Satz stammen, der
              // an einer Abkuerzung mitten in **fett** getrennt wurde. Marked
              // laesst die dadurch verwaisten Delimiter absichtlich als Text
              // stehen. In der reinen Anzeige sind sie aber nur kaputtes
              // Markdown; vollstaendige Sternchen-Paare wurden zu diesem
              // Zeitpunkt bereits in <strong>/<em> umgewandelt. Unterstriche
              // bleiben bewusst unangetastet (z. B. ein literales __init__).
              const textWalker = document.createTreeWalker(content, NodeFilter.SHOW_TEXT);
              let textNode;
              while ((textNode = textWalker.nextNode())) {
                if (textNode.parentElement?.closest("code")) continue;
                // Ein explizit escaptes \* ist sichtbarer Inhalt, kein kaputter
                // Markdown-Delimiter. In diesem seltenen Fall greift die
                // Bereinigung fuer den ganzen kurzen Inline-Wert nicht.
                if (!hasEscapedLiteralStar) {
                  textNode.nodeValue = textNode.nodeValue.replace(/\*{2,3}/g, "");
                }
              }

              element.appendChild(content);
            } else {
              element.appendChild(document.createTextNode(stripMarkdown(source)));
            }
            if (after) element.appendChild(document.createTextNode(after));
            // Math ist nicht von marked/DOMPurify abhaengig. Auch wenn eine
            // dieser CDN-Libraries fehlt und oben der Plaintext-Pfad greift,
            // bleiben \(...\)/\[...\] echte KaTeX-Ausdruecke.
            window.ConsensusMath?.render?.(element);
          }

          // --- Textsuche: Whitespace kollabieren, Anführungszeichen vereinheitlichen
          const {
            normalizeForSearch,
            findRangeInTextNode,
            searchVariants,
            sentenceBounds,
            locateAnchor
          } = window.App.consensusAnchor.create(stripMarkdown);
          function wrapFlatRange(slices, start, end, className) {
            const spans = [];
            slices.forEach(function (slice) {
              if (slice.end <= start || slice.start >= end) return;
              let node = slice.node;
              const from = Math.max(start, slice.start) - slice.start;
              const to = Math.min(end, slice.end) - slice.start;
              if (to < node.nodeValue.length) node.splitText(to);
              if (from > 0) node = node.splitText(from);
              const span = document.createElement("span");
              span.className = className;
              node.parentNode.insertBefore(span, node);
              span.appendChild(node);
              spans.push(span);
            });
            return spans;
          }

          // --- Popover / Bottom Sheet -------------------------------------
          let claimPopoverTrigger = null;
          let claimModalBackgroundState = [];

          function claimPopoverFocusables(pop) {
            return Array.from(pop?.querySelectorAll(
              'button:not([disabled]), a[href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
            ) || []).filter(element => !element.hidden && element.getClientRects().length > 0);
          }

          function setClaimModalBackgroundInert(active) {
            if (active) {
              claimModalBackgroundState = Array.from(document.body.children)
                .filter(element => element.id !== "claimPopover" && element.id !== "claimSheetBackdrop")
                .map(element => ({ element: element, inert: element.inert }));
              claimModalBackgroundState.forEach(item => { item.element.inert = true; });
              return;
            }
            claimModalBackgroundState.forEach(item => {
              if (item.element.isConnected) item.element.inert = item.inert;
            });
            claimModalBackgroundState = [];
          }

          function onDocClick(event) {
            const pop = $("claimPopover");
            if (!pop || pop.hidden) return;
            if (pop.contains(event.target)) return;
            if (event.target.closest?.(".claim-badge")) return;
            closeClaimPopover();
          }

          function onKeyDown(event) {
            const pop = $("claimPopover");
            if (!pop || pop.hidden) return;
            if (event.key === "Escape") {
              event.preventDefault();
              closeClaimPopover();
              return;
            }
            if (event.key !== "Tab" || !pop.classList.contains("is-modal")) return;
            const focusables = claimPopoverFocusables(pop);
            if (!focusables.length) {
              event.preventDefault();
              pop.focus();
              return;
            }
            const first = focusables[0];
            const last = focusables[focusables.length - 1];
            if (event.shiftKey && document.activeElement === first) {
              event.preventDefault();
              last.focus();
            } else if (!event.shiftKey && document.activeElement === last) {
              event.preventDefault();
              first.focus();
            }
          }

          // Popover/Backdrop nach <body> verschieben: Vorfahren mit
          // backdrop-filter/transform erzeugen sonst einen eigenen Containing
          // Block und verschieben absolute/fixed-Koordinaten.
          function ensureOverlayOnBody(el) {
            if (el && el.parentElement !== document.body) {
              document.body.appendChild(el);
            }
            return el;
          }

          function closeClaimPopover(options) {
            const restoreFocus = options?.restoreFocus !== false;
            const pop = $("claimPopover");
            const backdrop = $("claimSheetBackdrop");
            if (pop) {
              pop.hidden = true;
              pop.classList.remove("is-modal");
              pop.removeAttribute("aria-modal");
              pop.removeAttribute("aria-labelledby");
              pop.removeAttribute("tabindex");
              pop.innerHTML = "";
              pop.style.left = pop.style.top = pop.style.width = "";
            }
            if (backdrop) backdrop.hidden = true;
            setClaimModalBackgroundInert(false);
            document.removeEventListener("click", onDocClick, true);
            document.removeEventListener("keydown", onKeyDown, true);
            if (restoreFocus && claimPopoverTrigger?.isConnected) {
              claimPopoverTrigger.focus();
            }
            claimPopoverTrigger = null;
          }

          function buildModelRow(model, quote, status, answerNavigation) {
            const row = document.createElement("div");
            row.className = "claim-model-row is-" + status;

            const head = document.createElement("div");
            head.className = "claim-model-head";
            const name = document.createElement("span");
            name.className = "claim-model-name";
            name.textContent = modelDisplayName(model);
            head.appendChild(name);
            const navigation = answerNavigation || LIVE_ANSWER_NAVIGATION;
            if (navigation.canOpen?.(model)) {
              const jump = document.createElement("button");
              jump.type = "button";
              jump.className = "claim-jump-link";
              jump.textContent = "View answer";
              jump.addEventListener("click", function () {
                closeClaimPopover({ restoreFocus: false });
                navigation.open?.(model, quote);
              });
              head.appendChild(jump);
            }
            row.appendChild(head);

            if (quote) {
              const q = document.createElement("blockquote");
              q.className = "claim-model-quote";
              renderInlineMarkdown(q, quote);
              row.appendChild(q);
            }
            return row;
          }

          function modelsNotAddressingClaim(claim, modelsCompared) {
            if (!Array.isArray(modelsCompared)) return [];
            const addressed = new Set(claim.agree.concat(
              claim.dissent.map(function (item) { return item.model; })
            ));
            return modelsCompared.filter(function (model) { return !addressed.has(model); });
          }

          function openClaimPopover(claim, anchorEl, modelsCompared, answerNavigation) {
            const pop = ensureOverlayOnBody($("claimPopover"));
            const backdrop = ensureOverlayOnBody($("claimSheetBackdrop"));
            if (!pop) return;
            closeClaimPopover({ restoreFocus: false });
            claimPopoverTrigger = anchorEl || document.activeElement;

            const agreeCount = claim.agree.length;
            const total = agreeCount + claim.dissent.length;

            const header = document.createElement("div");
            header.className = "claim-popover-header";
            const title = document.createElement("span");
            title.className = "claim-popover-title";
            title.id = "claimPopoverTitle";
            title.textContent = claim.dissent.length
              ? `${agreeCount} of ${total} models agree`
              : `All ${total} models agree`;
            const close = document.createElement("button");
            close.type = "button";
            close.className = "claim-popover-close";
            close.setAttribute("aria-label", "Close");
            close.innerHTML = "&times;";
            close.addEventListener("click", closeClaimPopover);
            header.append(title, close);
            pop.appendChild(header);

            const claimText = document.createElement("div");
            claimText.className = "claim-popover-claim";
            renderInlineMarkdown(claimText, claim.anchor, "“", "”");
            pop.appendChild(claimText);

            if (claim.agree.length) {
              const section = document.createElement("div");
              section.className = "claim-popover-section";
              const label = document.createElement("div");
              label.className = "claim-section-label is-agree";
              label.textContent = "Agree";
              section.appendChild(label);
              claim.agree.forEach(model => section.appendChild(
                buildModelRow(model, "", "agree", answerNavigation)
              ));
              pop.appendChild(section);
            }
            if (claim.dissent.length) {
              const section = document.createElement("div");
              section.className = "claim-popover-section";
              const label = document.createElement("div");
              label.className = "claim-section-label is-dissent";
              label.textContent = "Deviate";
              section.appendChild(label);
              claim.dissent.forEach(item => section.appendChild(
                buildModelRow(item.model, item.quote, "dissent", answerNavigation)
              ));
              pop.appendChild(section);
            }
            const notAddressed = modelsNotAddressingClaim(claim, modelsCompared);
            if (notAddressed.length) {
              const section = document.createElement("div");
              section.className = "claim-popover-section";
              const label = document.createElement("div");
              label.className = "claim-section-label";
              label.textContent = "Not addressed";
              section.appendChild(label);
              notAddressed.forEach(model => section.appendChild(
                buildModelRow(model, "", "neutral", answerNavigation)
              ));
              pop.appendChild(section);
            }

            const asModal = isMobileViewport();
            pop.classList.toggle("is-modal", asModal);
            pop.setAttribute("aria-labelledby", title.id);
            pop.hidden = false;
            if (asModal) {
              pop.setAttribute("aria-modal", "true");
              pop.tabIndex = -1;
              setClaimModalBackgroundInert(true);
              if (backdrop) {
                backdrop.hidden = false;
                backdrop.addEventListener("click", closeClaimPopover, { once: true });
              }
              requestAnimationFrame(function () { close.focus(); });
            } else if (anchorEl) {
              // Direkt unter dem Badge, horizontal am Badge zentriert
              const rect = anchorEl.getBoundingClientRect();
              const width = Math.min(340, window.innerWidth - 24);
              pop.style.width = width + "px";
              const minLeft = window.scrollX + 12;
              const maxLeft = window.scrollX + window.innerWidth - width - 12;
              const centered = rect.left + rect.width / 2 - width / 2 + window.scrollX;
              pop.style.left = Math.max(minLeft, Math.min(centered, maxLeft)) + "px";
              pop.style.top = (rect.bottom + window.scrollY + 8) + "px";
            }
            setTimeout(function () {
              document.addEventListener("click", onDocClick, true);
              document.addEventListener("keydown", onKeyDown, true);
            }, 0);
          }

          // --- Sprung zur Originalantwort mit Zitat-Highlight ---------------
          // Markiert das Zitat und gibt das markierte Element zurück (oder null);
          // das Scrollen übernimmt der Aufrufer.
          function flashQuote(container, quote) {
            for (const needle of searchVariants(quote)) {
              const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
              let node;
              while ((node = walker.nextNode())) {
                if (!node.nodeValue || node.parentElement?.closest("code, pre, mark")) continue;
                const found = findRangeInTextNode(node, needle);
                if (!found) continue;
                const range = document.createRange();
                range.setStart(node, found.start);
                range.setEnd(node, found.end);
                const mark = document.createElement("mark");
                mark.className = "quote-flash";
                try {
                  range.surroundContents(mark);
                } catch (e) {
                  break;
                }
                setTimeout(function () {
                  const parent = mark.parentNode;
                  if (!parent) return;
                  while (mark.firstChild) parent.insertBefore(mark.firstChild, mark);
                  parent.removeChild(mark);
                  parent.normalize();
                }, 2400);
                return mark;
              }
              const blocks = container.querySelectorAll("p, li, h1, h2, h3, h4, td, blockquote");
              for (const block of blocks) {
                if (normalizeForSearch(block.textContent).includes(needle)) {
                  block.classList.add("quote-flash-block");
                  setTimeout(function () { block.classList.remove("quote-flash-block"); }, 2400);
                  return block;
                }
              }
            }
            return null;
          }

          function jumpToModelAnswer(model, quote) {
            const box = $(MODEL_BOX_IDS[model] || "");
            if (!box) return;

            // Einzelantworten liegen inzwischen in jedem Modus hinter derselben
            // Disclosure. Den Zielbereich aufdecken, den Modus aber beibehalten.
            window.App?.agentMode?.showModelAnswers?.();

            // Der Klassenwechsel ist synchron; die folgenden Geometrie-Abfragen
            // erzwingen das Layout selbst. So hängt der Sprung auch in
            // gedrosselten Hintergrund-Tabs weder an rAF noch an Timern.
            const content = box.querySelector(".collapsible-content");
            const highlight = (content && quote) ? flashQuote(content, quote) : null;
            // Offset, damit der Box-Header ("Response from …") sichtbar bleibt
            const headerY = box.getBoundingClientRect().top + window.scrollY - 84;

            if (highlight) {
              const quoteY = highlight.getBoundingClientRect().top + window.scrollY;
              if (quoteY - headerY < window.innerHeight * 0.7) {
                // Zitat liegt nah am Boxanfang: Header und Zitat zusammen zeigen
                window.scrollTo({ top: Math.max(0, headerY), behavior: "smooth" });
              } else {
                // Zitat liegt tief in der Antwort: Zitat mittig anfahren
                window.scrollTo({ top: Math.max(0, quoteY - window.innerHeight / 2), behavior: "smooth" });
              }
            } else {
              window.scrollTo({ top: Math.max(0, headerY), behavior: "smooth" });
              box.classList.add("jump-flash");
              setTimeout(function () { box.classList.remove("jump-flash"); }, 2000);
            }
            window.trackUmamiEvent?.("app_consensus_jump_to_answer", { model: model, found_quote: !!highlight });
          }

          function storedAnswerNavigation(body) {
            function targetFor(model) {
              const turn = body.closest?.(".thread-history-turn");
              if (!turn) return null;
              return Array.from(turn.querySelectorAll(
                ".thread-history-models section[data-provider]"
              )).find(function (section) {
                return section.dataset.provider === model;
              }) || null;
            }

            return {
              canOpen: function (model) {
                return !!targetFor(model);
              },
              open: function (model, quote) {
                const section = targetFor(model);
                if (!section) return;
                const panel = section.closest(".thread-history-panel");
                const turn = section.closest(".thread-history-turn");
                const tab = panel && turn
                  ? Array.from(turn.querySelectorAll(".consensus-tab")).find(function (candidate) {
                      return candidate.getAttribute("aria-controls") === panel.id;
                    })
                  : null;
                if (panel) panel.hidden = false;
                if (tab) tab.setAttribute("aria-expanded", "true");

                const content = section.querySelector(".thread-history-detail-body");
                const highlight = (content && quote) ? flashQuote(content, quote) : null;
                const headerY = section.getBoundingClientRect().top + window.scrollY - 84;
                if (highlight) {
                  const quoteY = highlight.getBoundingClientRect().top + window.scrollY;
                  const targetY = quoteY - headerY < window.innerHeight * 0.7
                    ? headerY
                    : quoteY - window.innerHeight / 2;
                  window.scrollTo({ top: Math.max(0, targetY), behavior: "smooth" });
                } else {
                  window.scrollTo({ top: Math.max(0, headerY), behavior: "smooth" });
                  section.classList.add("jump-flash");
                  setTimeout(function () { section.classList.remove("jump-flash"); }, 2000);
                }
                window.trackUmamiEvent?.("app_consensus_jump_to_answer", {
                  model: model,
                  found_quote: !!highlight,
                  stored_turn: true
                });
              }
            };
          }

          // Dasselbe fuer die Widerspruchs-Karten: ein archivierter Turn traegt
          // seine Karten in der eigenen Schublade (.thread-history-differences),
          // nicht im Live-Fuss. Die Karten entstehen erst NACH dieser Zeile
          // (der Turn wird von oben nach unten gebaut) — deshalb wird der
          // Container erst beim Klick gesucht, nicht beim Verankern.
          function storedDifferenceFocus(body) {
            return function (index) {
              const turn = body.closest?.(".thread-history-turn");
              const card = turn?.querySelectorAll(
                ".thread-history-differences .diff-card"
              )[index];
              if (!card) return;
              const panel = card.closest(".thread-history-panel");
              const tab = panel && turn
                ? Array.from(turn.querySelectorAll(".consensus-tab")).find(function (candidate) {
                    return candidate.getAttribute("aria-controls") === panel.id;
                  })
                : null;
              if (panel) panel.hidden = false;
              if (tab) tab.setAttribute("aria-expanded", "true");
              card.open = true;
              card.scrollIntoView({ behavior: "smooth", block: "center" });
              card.classList.add("is-focused");
              setTimeout(function () { card.classList.remove("is-focused"); }, 2000);
              window.trackUmamiEvent?.("app_consensus_marker_opened", {
                kind: "difference",
                stored_turn: true
              });
            };
          }

          // --- Verdict: worüber, nicht wie viele -------------------------------
          // Aus differences[].claim wird eine kurze Themenangabe abgeleitet -
          // rein deterministisch, ohne zusätzlichen LLM-Call. "The models
          // contradict each other on 1 point" sagt dem Leser nichts darüber,
          // ob ihn der strittige Punkt betrifft; das Thema schon.
          const TOPIC_PREFIX =
            /^(?:the\s+models?\s+)?(?:dis)?agree(?:ment)?\s+(?:on|about)\s+|^(?:do|does|did|is|are|was|were|should|shall|can|could|will|would|has|have|whether|how|what|which|when|why)\b\s*/i;
          // Genau EIN Thema. Mehrere aneinandergereihte Fragmente plus
          // "+2 more" waren ueberladen; das eine wichtigste Thema darf die
          // vorhandene Breite dagegen voll nutzen und natuerlich umbrechen.
          // Eine feste Wortgrenze schnitt es auch auf breiten Screens mitten
          // im Satz ab.
          const TOPIC_MAX_SHOWN = 1;

          function toTopic(claim) {
            let text = String(claim || "").trim().replace(/[.?!]+$/, "");
            text = text.replace(TOPIC_PREFIX, "").trim();
            if (!text) return "";
            // Groß-/Kleinschreibung und der vollstaendige Wortlaut bleiben
            // unangetastet; CSS uebernimmt den Umbruch an der echten Kante.
            return text;
          }

          function topicList(entries) {
            // Schwerwiegendes zuerst: bleibt nur ein Thema sichtbar, muss es
            // das sein, das den Leser am ehesten betrifft.
            const sorted = entries.slice().sort(
              (a, b) => (b.severity === "major") - (a.severity === "major")
            );
            const topics = sorted.map(d => toTopic(d.claim)).filter(Boolean);
            if (!topics.length) return "";
            // Kein "+N more": WIE VIELE sagt die Detailzeile, WORUEBER sagt
            // dieses eine Thema. Beides in dieselbe Zeile zu quetschen war
            // genau die Ueberfrachtung, die den Fuss unlesbar gemacht hat.
            return topics.slice(0, TOPIC_MAX_SHOWN).join(" · ");
          }

          // --- Verdict-Balken --------------------------------------------------
          // Neutraler Glas-Balken: Score-Ring links (Farbe = Semantik),
          // Headline + Detailzeile daneben, Judge-Attribution rechts.
          function renderVerdictHeader(differences, modelCount, agreement, judge) {
            const verdict = $("consensusVerdict");
            if (!verdict) return;
            const contradictions = differences.filter(d => d.type === "contradiction").length;
            const critical = differences.filter(d => d.type === "contradiction" && d.severity === "major").length;
            // Alte Bookmarks/Snapshots kennen keine Severity: dann keine
            // "critical"-Aussage machen statt fälschlich "none critical".
            const hasSeverity = differences.some(d => d.severity === "major" || d.severity === "minor");
            const emphases = differences.length - contradictions;
            const hasScore = agreement && typeof agreement.score === "number";

            const boundedScore = hasScore
              ? Math.max(0, Math.min(100, agreement.score))
              : null;
            // Farbe und Text muessen dieselbe 0-100-Aussage tragen. Der
            // Widerspruchsstatus bleibt als getrennte Detailzeile sichtbar.
            // Nur alte Snapshots ohne Score fallen auf die Difference-Ampel
            // zurueck.
            const cls = hasScore
              ? (boundedScore >= 65 ? "is-calm" : (boundedScore >= 40 ? "is-warn" : "is-alert"))
              : (contradictions === 0 ? "is-calm" : (critical > 0 ? "is-alert" : "is-warn"));
            verdict.classList.remove("is-calm", "is-warn", "is-alert");
            verdict.classList.add(cls);
            verdict.innerHTML = "";

            // Score-Anzeige; alte Bookmarks ohne Score behalten den kleinen Punkt.
            // Seit 2026-08-04 gesetzte Zahl plus feiner Messbalken statt
            // Conic-Ring auf getoenter Platte: die Zahl traegt die Aussage,
            // der Balken macht sie auf einen Blick vergleichbar, und die
            // Ampelfarbe bleibt die einzige Farbe im Fuss.
            if (hasScore) {
              const gauge = document.createElement("span");
              gauge.className = "verdict-gauge";
              gauge.style.setProperty("--val", String(boundedScore));
              gauge.title = "Agreement score " + agreement.score + "/100 across "
                + modelCount + " model" + (modelCount === 1 ? "" : "s");
              const score = document.createElement("span");
              score.className = "verdict-score";
              const num = document.createElement("span");
              num.className = "verdict-score-num";
              num.textContent = String(agreement.score);
              // Screenreader (und E2E-Check) lesen den vollen Score.
              const srUnit = document.createElement("span");
              srUnit.className = "visually-hidden";
              srUnit.textContent = "/100 agreement score";
              num.appendChild(srUnit);
              // Sichtbarer Nenner: eine nackte 28 sagt nicht, wovon sie 28 ist.
              // Das Wort "agreement" steht bereits in jeder Headline daneben,
              // eine zusaetzliche Bildunterschrift waere die dritte Wiederholung.
              const unit = document.createElement("span");
              unit.className = "verdict-score-unit";
              unit.setAttribute("aria-hidden", "true");
              unit.textContent = "/100";
              score.append(num, unit);
              const meter = document.createElement("span");
              meter.className = "verdict-meter";
              meter.setAttribute("aria-hidden", "true");
              const fill = document.createElement("span");
              fill.className = "verdict-meter-fill";
              meter.appendChild(fill);
              gauge.append(score, meter);
              verdict.appendChild(gauge);
            } else {
              const icon = document.createElement("span");
              icon.className = "verdict-icon";
              icon.setAttribute("aria-hidden", "true");
              verdict.appendChild(icon);
            }

            const contradictionTopics = topicList(
              differences.filter(d => d.type === "contradiction")
            );
            const emphasisTopics = topicList(
              differences.filter(d => d.type !== "contradiction")
            );

            const main = document.createElement("span");
            main.className = "verdict-main";
            const headline = document.createElement("span");
            headline.className = "verdict-headline";
            if (hasScore && boundedScore >= 85) {
              headline.textContent = "High agreement";
            } else if (hasScore && boundedScore >= 65) {
              headline.textContent = "Strong agreement";
            } else if (hasScore && boundedScore >= 40) {
              headline.textContent = "Partial agreement";
            } else if (hasScore && boundedScore >= 20) {
              headline.textContent = "Low agreement";
            } else if (hasScore) {
              headline.textContent = "Very low agreement";
            } else if (contradictions === 0) {
              headline.textContent = "No direct contradictions";
            } else if (contradictionTopics) {
              headline.textContent = "Disputed: " + contradictionTopics;
            } else {
              headline.textContent = "The models contradict each other";
            }
            // Die Zeile wird auf eine Zeile gekuerzt (CSS); der volle Satz
            // bleibt im Tooltip erreichbar.
            headline.title = headline.textContent;
            main.appendChild(headline);

            // Detailzeile: erst wie schwer, dann wer geurteilt hat. Die
            // Modellzahl steht seit 2026-07-28 nur noch in den Lauf-Fakten
            // eine Zeile tiefer — sie zweimal in zwei grauen Zeilen
            // untereinander zu wiederholen war der halbe Ballast im Fuss.
            const detail = document.createElement("span");
            detail.className = "verdict-detail";
            if (contradictions === 0) {
              if (emphases > 0) {
                detail.textContent = emphasisTopics
                  ? "different emphasis on " + emphasisTopics + ", no contradictions"
                  : emphases + " difference" + (emphases === 1 ? "" : "s") + " in emphasis, no contradictions";
              } else {
                detail.textContent = "no contradictions found";
              }
            } else if (hasSeverity && critical > 0) {
              const crit = document.createElement("span");
              crit.className = "verdict-detail-crit";
              crit.textContent = critical + " critical";
              detail.appendChild(crit);
              const minor = contradictions - critical;
              if (minor > 0) {
                detail.appendChild(document.createTextNode(
                  " · " + minor + " minor detail" + (minor === 1 ? "" : "s")));
              }
            } else if (hasSeverity) {
              detail.textContent = contradictions + " minor detail"
                + (contradictions === 1 ? "" : "s");
            } else {
              detail.textContent = contradictions + " disputed point"
                + (contradictions === 1 ? "" : "s");
            }
            if (contradictions > 0 && contradictionTopics) {
              detail.appendChild(document.createTextNode(
                " · disputed: " + contradictionTopics));
            }

            // Transparenz: welche (unabhängige) Modellfamilie die Analyse
            // geliefert hat. Als Nachsatz derselben Zeile statt als rechts
            // ausgerichteter Zweizeiler — eine Fußnote, die einen eigenen
            // Block bekommt, liest sich wie eine zweite Überschrift.
            if (judge && judge.provider) {
              const note = document.createElement("span");
              note.className = "verdict-judge";
              note.title = "The differences analysis runs on a different model"
                + " family than the consensus engine.";
              const provider = document.createElement("span");
              provider.textContent = "analysis by " + judge.provider
                + (judge.tier === "pro" ? " (Pro)" : "");
              const sub = document.createElement("span");
              sub.className = "verdict-judge-sub";
              sub.textContent = "independent of the consensus engine";
              note.append(provider, sub);
              if (detail.childNodes.length) {
                detail.appendChild(document.createTextNode(" · "));
              }
              detail.appendChild(note);
            }
            main.appendChild(detail);
            verdict.appendChild(main);
            verdict.hidden = false;
          }

          // --- Agreement-Badges in der Konsens-Antwort -----------------------
          function claimBadgeLabel(claim, modelsCompared) {
            const agreeCount = claim.agree.length;
            const total = agreeCount + claim.dissent.length;
            // Die Formulierung bleibt fuer Screenreader und den Detaildialog
            // erhalten. Die sichtbare Erklaerung liefert allein die formatierte
            // Hover-Vorschau, nicht zusaetzlich ein nativer Browser-Tooltip.
            const notAddressed = modelsNotAddressingClaim(claim, modelsCompared).length;
            return (claim.dissent.length
              ? agreeCount + " of " + total + " models support this"
              : "All " + total + " models that address this agree")
              + (notAddressed ? "; " + notAddressed + " did not address it" : "")
              + " — open details";
          }

          // Seit 2026-07-27 nur noch ein Punkt, kein "4/6" mehr. Neben den
          // hochgestellten Quellenzahlen standen im selben Satz zwei
          // konkurrierende Zahlensysteme — der Leser musste erst sortieren,
          // welche Zahl worauf zeigt. Die Quote steht jetzt dort, wo sie
          // ohnehin schon stand: in der Hover-Vorschau (buildClaimPreview),
          // im Tooltip und in der Karte beim Klick. Der Punkt bleibt als
          // fokussierbares, tippbares Steuerelement mit 44px-Trefferflaeche —
          // dieselbe Sprache wie der Widerspruchs-Marker daneben.
          function makeBadge(claim, modelsCompared, answerNavigation) {
            const badge = document.createElement("button");
            badge.type = "button";
            badge.className = "claim-badge" + (claim.dissent.length ? " has-dissent" : "");
            const agreeCount = claim.agree.length;
            const total = agreeCount + claim.dissent.length;
            const ratio = document.createElement("span");
            ratio.className = "claim-ratio";
            ratio.textContent = agreeCount + "/" + total;
            ratio.setAttribute("aria-hidden", "true");
            badge.appendChild(ratio);
            badge.setAttribute("aria-haspopup", "dialog");
            // Tastatur/Screenreader: sprechendes Label statt nacktem "4/6".
            badge.setAttribute("aria-label", claimBadgeLabel(claim, modelsCompared));
            badge.addEventListener("click", function (event) {
              event.stopPropagation();
              openClaimPopover(claim, badge, modelsCompared, answerNavigation);
            });
            return badge;
          }

          // --- Inline-Marker: Widersprüche und Claims im Antworttext ---------
          // Seit 2026-08-15 (User-Vorgabe) trägt die Textstelle einen farbigen
          // Textmarker statt einer Unterstreichung, und der Punkt daneben ist
          // ersatzlos weg: die Farbe sagt bereits alles, was der Punkt sagen
          // konnte. Bei einer Differenz ist die angestrichene Passage deshalb
          // selbst das Steuerelement (role="button"), bei einem Claim bleibt
          // die Quote daneben das sichtbare Ziel.

          // Ein Satz wird höchstens einmal dekoriert. Widersprüche laufen
          // deshalb zuerst (stärkere Stufe), Claim-Badges hängen sich danach an
          // denselben Satz, ohne ihn ein zweites Mal zu unterstreichen.
          function findOverlap(marked, start, end) {
            return marked.find(function (r) { return start < r.end && end > r.start; }) || null;
          }

          function diffMarkerLabel(diff) {
            const isMajor = diff.type === "contradiction" && diff.severity === "major";
            return (diff.type === "contradiction"
              ? (isMajor ? "The models contradict each other here" : "The models differ on a detail here")
              : "The models set a different focus here") + " — open details";
          }

          // Öffnet die zugehörige Karte im Differences-Überblick und hebt sie
          // kurz hervor. Das <details> darüber (Phase 4) wird mit aufgeklappt.
          function focusDifferenceCard(index) {
            const cards = $("differencesCards");
            const card = cards?.querySelectorAll(".diff-card")[index];
            if (!card) return;
            const panel = card.closest("details.consensus-differences-panel");
            if (panel) panel.open = true;
            card.open = true;
            card.scrollIntoView({ behavior: "smooth", block: "center" });
            card.classList.add("is-focused");
            setTimeout(function () { card.classList.remove("is-focused"); }, 2000);
            window.trackUmamiEvent?.("app_consensus_marker_opened", { kind: "difference" });
          }

          // Stufen der Satzmarkierung, schwach nach stark. Treffen zwei Marken
          // denselben Satz, gewinnt die staerkere die Linie - sonst stand
          // neben einem gelben 2/4-Badge eine graue Linie.
          const MARK_LEVELS = {
            "is-unanimous": 0,
            "is-minor": 1,
            "is-split": 2,
            "is-major": 3
          };

          // Markiert eine Textstelle satzweise und liefert den letzten Span
          // zurück (dahinter wird der Marker/das Badge eingehängt).
          function markSentence(container, anchor, severityClass, marked, occurrence) {
            const hit = locateAnchor(container, anchor, occurrence);
            if (!hit) return null;

            const bounds = sentenceBounds(hit.flat, hit.start, hit.end);
            if (bounds.end <= bounds.start) return null;

            // Derselbe Satz wird nicht zweimal dekoriert (ein Widerspruch und
            // ein Claim können auf dieselbe Stelle zeigen). Das zweite Element
            // hängt sich an die bereits erzeugten Spans an - hebt die Linie
            // aber auf seine Stufe an, wenn sie schwaecher ist als seine Marke.
            const existing = findOverlap(marked, bounds.start, bounds.end);
            if (existing) {
              if (MARK_LEVELS[severityClass] > MARK_LEVELS[existing.severity]) {
                existing.spans.forEach(function (span) {
                  span.classList.remove(existing.severity);
                  span.classList.add(severityClass);
                });
                existing.severity = severityClass;
              }
              return { spans: existing.spans, block: hit.block };
            }

            const spans = wrapFlatRange(
              hit.slices, bounds.start, bounds.end, "cx-claim " + severityClass
            );
            marked.push({
              start: bounds.start, end: bounds.end, spans: spans, severity: severityClass
            });
            return { spans, block: hit.block };
          }

          function insertAfterMark(result, el) {
            const last = result.spans[result.spans.length - 1];
            if (last && last.parentNode) {
              last.parentNode.insertBefore(el, last.nextSibling);
            } else {
              result.block.appendChild(el);
            }
          }

          // --- Passage und Marker aneinander koppeln --------------------------
          // Die markierte Passage ist ein ganzer Satz und damit das weitaus
          // groesste Ziel: Vorschau, Zeigefinger, Klick haengen an ihr. Der
          // Hover wirkt in beide Richtungen, damit sichtbar wird,
          // welches Badge zu welchem Satz gehoert (ein Absatz kann mehrere
          // tragen). Wo ein Badge steht, bleibt es das fokussierbare
          // Steuerelement und die Passage ist nur ein zusaetzlicher Mausweg;
          // an einer Differenz gibt es daneben nichts mehr, deshalb wird dort
          // die Passage selbst fokussierbar (attachPassageControl).
          function applyPassageHover(group) {
            group.spans.forEach(function (span) {
              span.classList.toggle("is-hovered", group.hover);
            });
            group.controls.forEach(function (ctrl) {
              ctrl.el.classList.toggle("is-linked-hover", group.hover);
            });
          }

          function setPassageHover(group, on) {
            group.hover = on;
            if (on) {
              applyPassageHover(group);
              scheduleHoverPreview(group);
              return;
            }
            hideHoverPreview();
            // Der Weg von einem Span zum naechsten (ein Satz kann in mehrere
            // Spans zerfallen) oder zum Marker feuert leave/enter nacheinander.
            // Ein Frame Verzoegerung verhindert das Flackern dazwischen.
            requestAnimationFrame(function () {
              if (!group.hover) applyPassageHover(group);
            });
          }

          // Ueberlappende Markierungen haben mehrere Controls. Sichtbar ist
          // immer nur eines: das zugunsten des Claims zurueckgetretene
          // Difference-Control (suppressed) darf weder den Klick abfangen noch
          // die Hover-Vorschau stellen.
          function activeControl(group) {
            return group.controls.find(function (item) {
              return !item.suppressed && !item.el.hidden
                && item.el.getAttribute("aria-hidden") !== "true";
            }) || group.controls[0] || null;
          }

          function passageGroup(spans) {
            if (spans[0].cxGroup) return spans[0].cxGroup;
            const group = { spans: spans, controls: [], hover: false };
            const supportsHover = canHoverPassages();
            spans.forEach(function (span) {
              span.cxGroup = group;
              span.classList.add("is-interactive");
              if (supportsHover) {
                span.addEventListener("mouseenter", function () { setPassageHover(group, true); });
                span.addEventListener("mouseleave", function () { setPassageHover(group, false); });
              }
              span.addEventListener("click", function (event) {
                // Quellenchips und [S1]-Links im Satz behalten Vorrang, und wer
                // Text markiert, will ihn kopieren und nichts oeffnen.
                if (event.target.closest("a, button")) return;
                const selection = window.getSelection?.();
                if (selection && !selection.isCollapsed) return;
                const target = activeControl(group);
                if (target) target.activate(event);
              });
            });
            return group;
          }

          // Hover nur auf Geraeten mit echtem Zeiger. Der Klick auf die Passage
          // bleibt dagegen auch auf Touch aktiv.
          function canHoverPassages() {
            return !!window.matchMedia?.("(hover: hover) and (pointer: fine)").matches;
          }

          // --- Vorschau beim Hovern -------------------------------------------
          // Die Unterstreichung sagt "hier ist etwas", aber nicht was. Wer mit
          // der Maus darauf verweilt, bekommt deshalb sofort die Kurzfassung —
          // wer klickt, weiterhin die ganze Karte. Reine Lesehilfe: die Karte
          // faengt keine Maus (pointer-events: none), damit sie den Klick auf
          // die Passage nicht abfaengt.
          const HOVER_DELAY_MS = 130;
          let hoverCard = null;
          let hoverTimer = null;

          function ensureHoverCard() {
            if (hoverCard && hoverCard.isConnected) return hoverCard;
            hoverCard = document.createElement("div");
            hoverCard.className = "insight-preview";
            hoverCard.setAttribute("aria-hidden", "true");
            hoverCard.hidden = true;
            document.body.appendChild(hoverCard);
            return hoverCard;
          }

          function hideHoverPreview() {
            window.clearTimeout(hoverTimer);
            hoverTimer = null;
            hoverGroup = null;
            if (hoverCard) {
              hoverCard.hidden = true;
              hoverCard.innerHTML = "";
            }
          }

          function placeHoverPreview(card, spans) {
            const first = spans[0].getBoundingClientRect();
            const last = spans[spans.length - 1].getBoundingClientRect();
            const width = Math.min(360, window.innerWidth - 24);
            card.style.width = width + "px";

            const centered = first.left + (last.right - first.left) / 2 - width / 2;
            const left = Math.max(12, Math.min(centered, window.innerWidth - width - 12));
            card.style.left = (left + window.scrollX) + "px";

            // Unter die Passage, solange darunter Platz ist — sonst darueber.
            const height = card.offsetHeight;
            const below = last.bottom + 8;
            const fitsBelow = below + height <= window.innerHeight - 12;
            const top = fitsBelow ? below : Math.max(12, first.top - height - 8);
            card.style.top = (top + window.scrollY) + "px";
          }

          function showHoverPreview(group) {
            const active = activeControl(group);
            const build = (active && active.preview) ? active
              : group.controls.find(function (c) { return c.preview && !c.suppressed; });
            if (!build || !build.preview) return;
            const card = ensureHoverCard();
            card.innerHTML = "";
            card.appendChild(build.preview());
            card.hidden = false;
            placeHoverPreview(card, group.spans);
          }

          function scheduleHoverPreview(group) {
            window.clearTimeout(hoverTimer);
            hoverTimer = window.setTimeout(function () {
              if (group.hover) showHoverPreview(group);
            }, HOVER_DELAY_MS);
          }

          // Die Karte klebt an einer Bildschirmposition, nicht am Dokument:
          // sobald sich darunter etwas bewegt oder der Nutzer klickt (und
          // damit die grosse Ansicht will), verschwindet sie.
          window.addEventListener("scroll", hideHoverPreview, true);
          window.addEventListener("resize", hideHoverPreview);
          document.addEventListener("click", hideHoverPreview, true);

          function previewHead(text, sevClass) {
            const head = document.createElement("div");
            head.className = "insight-preview-head";
            if (sevClass) {
              const dot = document.createElement("span");
              dot.className = "sev-dot " + sevClass;
              dot.setAttribute("aria-hidden", "true");
              head.appendChild(dot);
            }
            const label = document.createElement("span");
            label.textContent = text;
            head.appendChild(label);
            return head;
          }

          function previewRow(label, text) {
            const row = document.createElement("div");
            row.className = "insight-preview-row";
            const key = document.createElement("span");
            key.className = "insight-preview-key";
            key.textContent = label;
            const value = document.createElement("span");
            value.className = "insight-preview-value";
            value.textContent = text;
            row.append(key, value);
            return row;
          }

          function buildClaimPreview(claim, modelsCompared) {
            const frag = document.createDocumentFragment();
            const agreeCount = claim.agree.length;
            const total = agreeCount + claim.dissent.length;
            frag.appendChild(previewHead(
              claim.dissent.length
                ? agreeCount + " of " + total + " models support this"
                : "All " + total + " models agree",
              claim.dissent.length ? "is-warn" : null
            ));
            if (claim.agree.length) {
              frag.appendChild(previewRow(
                "Agree", claim.agree.map(modelDisplayName).join(", ")));
            }
            if (claim.dissent.length) {
              frag.appendChild(previewRow(
                "Deviate", claim.dissent.map(function (d) { return modelDisplayName(d.model); }).join(", ")));
              const quote = claim.dissent.find(function (d) { return d.quote; });
              if (quote) {
                const q = document.createElement("blockquote");
                q.className = "insight-preview-quote";
                renderInlineMarkdown(q, quote.quote);
                frag.appendChild(q);
              }
            }
            const notAddressed = modelsNotAddressingClaim(claim, modelsCompared);
            if (notAddressed.length) {
              frag.appendChild(previewRow(
                "Not addressed", notAddressed.map(modelDisplayName).join(", ")));
            }
            const foot = document.createElement("div");
            foot.className = "insight-preview-foot";
            foot.textContent = "Click for the full breakdown";
            frag.appendChild(foot);
            return frag;
          }

          function buildDiffPreview(diff) {
            const frag = document.createDocumentFragment();
            const isMajor = diff.type === "contradiction" && diff.severity === "major";
            let label = "Different emphasis";
            if (diff.type === "contradiction") {
              label = isMajor ? "Contradiction · critical" : "Contradiction · minor detail";
            }
            frag.appendChild(previewHead(label, isMajor ? "is-crit" : "is-warn"));

            const claimEl = document.createElement("div");
            claimEl.className = "insight-preview-claim";
            claimEl.textContent = diff.claim;
            frag.appendChild(claimEl);

            (diff.positions || []).slice(0, 2).forEach(function (pos) {
              const row = previewRow(
                pos.models.map(modelDisplayName).join(", "),
                "");
              renderInlineMarkdown(
                row.querySelector(".insight-preview-value"),
                pos.stance || pos.quote || "");
              frag.appendChild(row);
            });

            const foot = document.createElement("div");
            foot.className = "insight-preview-foot";
            foot.textContent = "Click to open the difference";
            frag.appendChild(foot);
            return frag;
          }

          // Haengt das Steuerelement hinter die Passage und verbindet beide.
          function attachControl(result, control, activate, preview) {
            insertAfterMark(result, control);
            const spans = result.spans;
            if (!spans || !spans.length) return null;
            const group = passageGroup(spans);
            const entry = { el: control, activate: activate, preview: preview };
            group.controls.push(entry);
            if (canHoverPassages()) {
              control.addEventListener("mouseenter", function () { setPassageHover(group, true); });
              control.addEventListener("mouseleave", function () { setPassageHover(group, false); });
            }
            return entry;
          }

          // Dieselbe Kopplung ohne eigenes Element: die angestrichene Passage
          // IST das Steuerelement. Seit die Punkte weg sind (2026-08-15) haengt
          // hier die gesamte Bedienbarkeit einer Differenz - ohne role/tabindex
          // waere sie fuer Tastatur und Screenreader unerreichbar. Der erste
          // Span traegt die Semantik; ein Satz, der ueber einen Zeilenumbruch
          // in mehrere Spans zerfaellt, bekommt trotzdem nur EINEN Tabstopp.
          function attachPassageControl(result, label, activate, preview) {
            const spans = result.spans;
            if (!spans || !spans.length) return null;
            const group = passageGroup(spans);
            const host = spans[0];
            host.setAttribute("role", "button");
            host.setAttribute("tabindex", "0");
            host.setAttribute("aria-label", label);
            if (!host.dataset.cxKeyboard) {
              host.dataset.cxKeyboard = "1";
              host.addEventListener("keydown", function (event) {
                if (event.key !== "Enter" && event.key !== " " && event.key !== "Spacebar") return;
                event.preventDefault();
                const target = activeControl(group);
                if (target) target.activate(event);
              });
              host.addEventListener("focus", function () { setPassageHover(group, true); });
              host.addEventListener("blur", function () { setPassageHover(group, false); });
            }
            const entry = { el: host, activate: activate, preview: preview, isPassage: true };
            group.controls.push(entry);
            return entry;
          }

          // Das unterlegene Control tritt zurueck, bleibt aber fuer Vorschau
          // und Zaehlung im Group-Objekt. Bei einer Passage heisst
          // "zuruecktreten" auch: den Tabstopp abgeben, sonst haette der Satz
          // zwei Fokusziele (Passage und Badge) fuer dieselbe Aussage.
          function suppressControl(entry) {
            if (!entry) return;
            entry.suppressed = true;
            if (entry.isPassage) {
              entry.el.removeAttribute("role");
              entry.el.removeAttribute("tabindex");
              entry.el.removeAttribute("aria-label");
              return;
            }
            entry.el.hidden = true;
            entry.el.setAttribute("aria-hidden", "true");
            entry.el.tabIndex = -1;
          }

          function renderInlineMarkers(claims, differences, modelsCompared, options) {
            options = options || {};
            const body = options.body || window.App.consensusBodyEl();
            const fallbackBox = options.fallbackBox || $("consensusClaimsFallback");
            const answerNavigation = options.answerNavigation || LIVE_ANSWER_NAVIGATION;
            // Ein archivierter Turn hat seine eigenen Karten; der Live-Fuss
            // gehoert bereits dem naechsten Lauf.
            const focusDifference = options.focusDifference || focusDifferenceCard;
            if (!body || !fallbackBox) return;

            // Pro Blockelement eine eigene Liste bereits markierter Bereiche:
            // die flachen Offsets gelten nur innerhalb ihres Blocks.
            const markedByBlock = new Map();
            function marksFor(block) {
              if (!markedByBlock.has(block)) markedByBlock.set(block, []);
              return markedByBlock.get(block);
            }
            // markSentence braucht die Liste, bevor der Block bekannt ist -
            // deshalb eine gemeinsame Liste je Aufruf und Zuordnung danach.
            function mark(anchor, severityClass, occurrence) {
              const probe = locateAnchor(body, anchor, occurrence);
              if (!probe) return null;
              return markSentence(
                body, anchor, severityClass, marksFor(probe.block), occurrence
              );
            }

            // Wenn Claim und Difference denselben Satz meinen, steht dort nur
            // EIN sichtbares Signal — welches, entscheidet die Schwere:
            //
            // - WIDERSPRUCH: die Passage gewinnt, das Claim-Badge entfaellt.
            //   Seit die Claims jeden pruefbaren Satz abdecken (Satz-Index,
            //   2026-08-07), traegt ein strittiger Satz fast IMMER auch ein
            //   Claim-Badge. Mit der alten Regel (Badge gewinnt) verschwand
            //   damit praktisch jeder Widerspruch aus dem Text: der
            //   strittige Satz sah aus wie jeder andere, und der Klick darauf
            //   oeffnete "4 of 6 models agree" statt der Widerspruchs-Karte.
            //   Genau das ist der Kern des Produkts — es darf nicht als
            //   Zustimmungsquote getarnt werden.
            // - EMPHASIS: das Badge gewinnt wie bisher; eine andere Gewichtung
            //   ist die schwaechere Aussage als die Stuetzungsquote.
            //
            // Das unterlegene Control bleibt als suppressed im Group-Objekt,
            // damit Hover-Vorschau und Zaehlung es weiterhin kennen.
            const differenceControls = [];

            // Trefferquote der Ankersuche: nur so laesst sich belegen, ob eine
            // Aenderung an Ankern oder Suche die Abdeckung wirklich hebt -
            // statt sie zu schaetzen. Fliesst in app_consensus_insights_rendered.
            let diffAnchorMisses = 0;

            // 1. Widersprüche zuerst: sie tragen die stärkere Markierung.
            differences.forEach(function (diff, index) {
              if (!diff.consensus_anchor) return;
              const isMajor = diff.type === "contradiction" && diff.severity === "major";
              const result = mark(
                diff.consensus_anchor,
                isMajor ? "is-major" : "is-minor",
                diff.anchor_occurrence
              );
              if (!result) {
                diffAnchorMisses += 1;
                return;
              }
              const control = attachPassageControl(
                result,
                diffMarkerLabel(diff) + ": " + diff.claim,
                function () { focusDifference(index); },
                function () { return buildDiffPreview(diff); }
              );
              differenceControls.push({
                spans: result.spans,
                control: control,
                isContradiction: diff.type === "contradiction"
              });
            });

            // 2. Claims: is-unanimous ist bewusst dekorationslos (nur Badge),
            //    dient hier aber als präziser Einhängepunkt für das Badge.
            //    Ein Claim mit Dissens traegt dieselbe Bernstein-Note wie sein
            //    Badge (is-split), nur eine Stufe leiser als der Widerspruch.
            const unanchored = [];
            const claimControls = [];
            claims.forEach(function (claim) {
              const result = mark(
                claim.anchor,
                claim.dissent.length ? "is-split" : "is-unanimous",
                claim.anchor_occurrence
              );
              if (!result) {
                unanchored.push(claim);
                return;
              }
              const overlappingDifference = differenceControls.find(function (entry) {
                return entry.spans === result.spans;
              });
              // Auf einem strittigen Satz behaelt der Widerspruch das Wort.
              // Der Satz ist bereits markiert (markSentence hat die staerkere
              // is-major/is-minor-Linie gesetzt), er bekommt hier nur kein
              // zweites, schwaecher klingendes Steuerelement daneben.
              if (overlappingDifference && overlappingDifference.isContradiction) return;

              const badge = makeBadge(claim, modelsCompared, answerNavigation);
              const total = claim.agree.length + claim.dissent.length;
              const support = total ? claim.agree.length / total : 1;
              const existingClaim = claimControls.find(function (entry) {
                return entry.spans === result.spans;
              });

              // Mehrere gepruefte Claims koennen im selben Satz liegen. Zwei
              // Quoten hinter demselben Satz waeren wieder genau die
              // Ueberladung, die diese Mikro-Marke vermeiden soll. Sichtbar
              // bleibt deshalb die konservative Satzquote: der am wenigsten
              // gestuetzte Claim.
              if (existingClaim && support >= existingClaim.support) return;
              if (existingClaim) {
                existingClaim.badge.remove();
                const group = result.spans[0]?.cxGroup;
                if (group) {
                  group.controls = group.controls.filter(function (control) {
                    return control.el !== existingClaim.badge;
                  });
                }
                existingClaim.badge = badge;
                existingClaim.support = support;
              } else {
                claimControls.push({
                  spans: result.spans,
                  badge: badge,
                  support: support
                });
              }
              // Nur noch Emphasis-Marken treten hinter das Badge zurueck.
              if (overlappingDifference) suppressControl(overlappingDifference.control);
              attachControl(result, badge, function () {
                openClaimPopover(claim, badge, modelsCompared, answerNavigation);
              }, function () { return buildClaimPreview(claim, modelsCompared); });
            });

            if (unanchored.length) {
              // Der Anker ist eine woertliche Kopie des Konsenstextes und
              // traegt damit dessen Quellentags. Im Fliesstext macht
              // linkifySourceTags daraus die hochgestellte Fussnote - hier
              // stand bis dahin die rohe Klammer "[S4]" im Satz.
              const sources = Array.isArray(options.sources)
                ? options.sources
                : (Array.isArray(window.currentEvidenceSources) ? window.currentEvidenceSources : []);
              fallbackBox.innerHTML = "";
              const title = document.createElement("div");
              title.className = "claims-fallback-title";
              title.textContent = "Key claims";
              fallbackBox.appendChild(title);
              unanchored.forEach(function (claim) {
                const row = document.createElement("div");
                row.className = "claims-fallback-row";
                const text = document.createElement("span");
                text.className = "claims-fallback-text";
                renderInlineMarkdown(text, claim.anchor);
                row.append(text, makeBadge(claim, modelsCompared, answerNavigation));
                fallbackBox.appendChild(row);
                // Erst nach dem Einhaengen: die Nummernform erkennt
                // linkifySourceTags an den Vorfahren der Zeile.
                if (sources.length) window.linkifySourceTags?.(text, sources);
              });
              fallbackBox.hidden = false;
            }

            // Legende nur, wenn wirklich etwas markiert wurde. Sie verhindert,
            // dass unmarkierter Text als geprueft-und-bestaetigt gelesen wird.
            if (!options.stored) {
              const legend = $("consensusMarkerLegend");
              if (legend) {
                legend.hidden = !body.querySelector(".cx-claim, .claim-badge");
              }

              // Die Provenance-Zeile zaehlt die strittigen Stellen aus genau
              // diesen Markern. Sie wird beim Laufende zuerst ohne sie gerendert
              // (die Marker entstehen erst hier) und holt die Zahl jetzt nach.
              window.App?.consensusPipeline?.renderProvenance?.();
            }

            return {
              claims_anchored: claims.length - unanchored.length,
              claims_unanchored: unanchored.length,
              diffs_unanchored: diffAnchorMisses
            };
          }

          // Gespeicherte Chat-Turns besitzen eigene Antwortcontainer und keine
          // globalen Live-IDs. Claims werden deshalb containerlokal erneut
          // verankert; Hover und Detaildialog bleiben genauso erreichbar wie
          // beim aktuell sichtbaren Consensus.
          function renderStoredConsensusClaims(body, data, fallbackBox, sources) {
            if (!body || !data || typeof data !== "object") return false;
            const claims = (Array.isArray(data.claims) ? data.claims : [])
              .filter(c => c && c.anchor && Array.isArray(c.agree) && Array.isArray(c.dissent));
            const modelsCompared = Array.isArray(data.models_compared)
              ? data.models_compared : [];
            // Die Widersprueche gehoeren MIT in den Verlauf: ohne sie fiel ein
            // strittiger Satz beim naechsten Turn auf sein Claim-Badge zurueck
            // und las sich als "1 of 6 models support this" (bernstein) statt
            // als roter Widerspruch — dieselbe Antwort, entschaerft.
            const differences = (Array.isArray(data.differences) ? data.differences : [])
              .filter(d => d && d.claim && Array.isArray(d.positions) && d.positions.length);
            renderInlineMarkers(claims, differences, modelsCompared, {
              body: body,
              fallbackBox: fallbackBox,
              stored: true,
              focusDifference: storedDifferenceFocus(body),
              // Ein archivierter Turn hat seine eigenen Quellen; die globale
              // Liste gehoert bereits dem naechsten Lauf.
              sources: Array.isArray(sources) ? sources : [],
              answerNavigation: storedAnswerNavigation(body)
            });
            return claims.length > 0;
          }

          // --- Resolve-Runde ---------------------------------------------------
          // Konfrontiert die dissentierenden Modelle eines Widerspruchs gezielt
          // mit der Gegenposition (POST /resolve). Pro-Feature: Free-Nutzer
          // sehen den Button als Teaser (öffnet das Pro-Modal). Ergebnis wird
          // in der Karte gerendert, am diff-Objekt gemerkt und über das
          // Consensus-Bookmark persistiert.
          const RESOLVE_STATUS = {
            resolved: { cls: "is-resolved", label: "Resolved" },
            standoff: { cls: "is-standoff", label: "Dissent confirmed" },
            mutual_revision: { cls: "is-mixed", label: "Still unclear" }
          };

          function resolveOutcomeSummary(outcome, results) {
            const revised = results.filter(r => r.decision === "revise").map(r => modelDisplayName(r.model));
            const maintained = results.filter(r => r.decision === "maintain").map(r => modelDisplayName(r.model));
            switch (outcome) {
              case "resolved":
                return {
                  cls: "is-resolved",
                  text: "Resolved: " + revised.join(", ") + " revised after seeing the counter-position; "
                    + maintained.join(", ") + " confirmed."
                };
              case "standoff":
                return { cls: "is-standoff", text: "Confirmed dissent: every model maintains its position after re-examination." };
              case "mutual_revision":
                return { cls: "is-mixed", text: "All models revised their position. The point stays unclear; verify independently." };
              default:
                return { cls: "is-error", text: "The resolve round did not return a usable result. Please try again." };
            }
          }

          // Down-Chevron fuer den Aufklapp-Pfeil (rotiert per CSS bei [open]).
          const RESOLVE_CHEVRON =
            '<svg viewBox="0 0 12 12" width="11" height="11" aria-hidden="true" fill="none" '
            + 'stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">'
            + '<path d="M2.5 4.5 6 8l3.5-3.5"/></svg>';

          function decisionLabel(decision) {
            return decision === "maintain" ? "maintains"
              : (decision === "revise" ? "revised" : "no result");
          }

          function renderResolveResult(resultBox, data) {
            resultBox.innerHTML = "";
            const summary = resolveOutcomeSummary(data.outcome, Array.isArray(data.results) ? data.results : []);
            const badge = document.createElement("div");
            badge.className = "resolve-outcome " + summary.cls;
            badge.textContent = summary.text;
            resultBox.appendChild(badge);

            (Array.isArray(data.results) ? data.results : []).forEach(function (r) {
              // Die im Resolve neu gegebene Antwort (position + reason) steckt
              // platzsparend hinter einem Aufklapp-Pfeil. Ohne Antwort (Fehler/
              // kein Ergebnis) bleibt die Zeile eine einfache, nicht klappbare Box.
              // Bewusst kein <details>/<summary>: das native Collapse ist auf
              // dieser Seite global ausgehebelt; wir klappen per Klasse selbst.
              const hasDetail = !!(r.position || r.reason);
              const row = document.createElement("div");
              row.className = "resolve-model-row" + (hasDetail ? " is-collapsible" : "");

              const head = document.createElement(hasDetail ? "button" : "div");
              head.className = "resolve-model-head" + (hasDetail ? " resolve-model-toggle" : "");
              if (hasDetail) {
                head.type = "button";
                head.setAttribute("aria-expanded", "false");
                head.title = "Show this model's revised answer";
              }

              const name = document.createElement("span");
              name.className = "resolve-model-name";
              name.textContent = modelDisplayName(r.model);
              const decision = document.createElement("span");
              decision.className = "resolve-decision is-" + (r.decision || "error");
              decision.textContent = decisionLabel(r.decision);
              head.append(name, decision);

              if (!hasDetail) {
                row.appendChild(head);
                resultBox.appendChild(row);
                return;
              }

              const disclosure = document.createElement("span");
              disclosure.className = "resolve-disclosure";
              disclosure.setAttribute("aria-hidden", "true");
              disclosure.innerHTML = RESOLVE_CHEVRON;
              head.appendChild(disclosure);

              const detail = document.createElement("div");
              detail.className = "resolve-model-detail";
              if (r.position) {
                const pos = document.createElement("div");
                pos.className = "resolve-position";
                pos.textContent = r.position;
                detail.appendChild(pos);
              }
              if (r.reason) {
                const reason = document.createElement("div");
                reason.className = "resolve-reason";
                reason.textContent = r.reason;
                detail.appendChild(reason);
              }

              head.addEventListener("click", function () {
                const open = row.classList.toggle("is-open");
                head.setAttribute("aria-expanded", open ? "true" : "false");
              });

              row.append(head, detail);
              resultBox.appendChild(row);
            });

            // Transparenz: den tatsaechlich gestellten Prompt hinter einem
            // Toggle zeigen. Nur in der Live-Session verfuegbar (das Feld wird
            // nicht ins Bookmark/Share persistiert, s. runResolveRound).
            const withPrompt = (Array.isArray(data.results) ? data.results : []).filter(function (r) { return r.prompt; });
            if (withPrompt.length) {
              const promptWrap = document.createElement("div");
              promptWrap.className = "resolve-prompt";
              const toggle = document.createElement("button");
              toggle.type = "button";
              toggle.className = "resolve-prompt-toggle";
              toggle.setAttribute("aria-expanded", "false");
              toggle.innerHTML = RESOLVE_CHEVRON + "<span>What the models were asked</span>";
              const body = document.createElement("div");
              body.className = "resolve-prompt-body";
              body.hidden = true;
              withPrompt.forEach(function (r) {
                const item = document.createElement("div");
                item.className = "resolve-prompt-item";
                const modelHead = document.createElement("div");
                modelHead.className = "resolve-prompt-model";
                modelHead.textContent = modelDisplayName(r.model);
                const text = document.createElement("pre");
                text.className = "resolve-prompt-text";
                text.textContent = r.prompt;
                item.append(modelHead, text);
                body.appendChild(item);
              });
              toggle.addEventListener("click", function () {
                const open = body.hidden;
                body.hidden = !open;
                promptWrap.classList.toggle("is-open", open);
                toggle.setAttribute("aria-expanded", open ? "true" : "false");
              });
              promptWrap.append(toggle, body);
              resultBox.appendChild(promptWrap);
            }
            resultBox.hidden = false;
          }

          // Karte sichtbar als "gelöst/bestätigt/unklar" kennzeichnen: Status-
          // Chip neben dem Typ-Tag plus Karten-Klasse für den Farbakzent.
          function markCardResolved(card, outcome) {
            const status = RESOLVE_STATUS[outcome];
            if (!card || !status) return;
            card.classList.add("has-resolution", "resolution-" + status.cls.slice(3));
            const tagRow = card.querySelector(".diff-card-tags");
            if (!tagRow || tagRow.querySelector(".diff-resolved-tag")) return;
            const chip = document.createElement("span");
            chip.className = "diff-resolved-tag " + status.cls;
            chip.textContent = status.label;
            tagRow.appendChild(chip);
          }

          // Nach einer Resolve-Runde das aktualisierte differences_data erneut
          // ins Consensus-Bookmark schreiben, damit der gelöste Zustand beim
          // Wiederöffnen erhalten bleibt.
          function persistResolutionToBookmark() {
            const payload = window.lastConsensusBookmarkPayload;
            if (!payload || !payload.question || !window.auth?.currentUser) return;
            if (typeof window.saveBookmarkConsensus !== "function") return;
            window.saveBookmarkConsensus(
              payload.question,
              payload.consensusText,
              payload.differencesText,
              payload.differencesData,
              payload.resultId || null,
              "",
              null,
              payload.previousQuestion || "",
              payload.previousTurn || null,
              payload.conversation || null
            );
          }

          function showResolveProTeaser() {
            window.trackUmamiEvent?.("app_resolve_pro_teaser_click");
            const shown = window.App?.showProFeatureModal?.("Resolve");
            if (!shown) {
              window.App?.showPopup?.("Resolve is off here. It is a second full round of model calls.");
            }
          }

          async function runResolveRound(diff, button, resultBox) {
            if (!window.auth?.currentUser) {
              window.App?.showPopup?.("Please log in to resolve contradictions.");
              return;
            }
            let idToken = null;
            try {
              idToken = await window.auth.currentUser.getIdToken();
            } catch (e) {
              console.error("Token refresh error in resolve:", e);
            }
            if (!idToken) {
              window.App?.showPopup?.("Please log in to resolve contradictions.");
              return;
            }

            const question = (window.lastQuestion || $("questionInput")?.value || "").trim();
            const useOwnKeys = !!$("useOwnKeysSwitch")?.checked;
            const resolveUsageRunKey = useOwnKeys
              ? null
              : (globalThis.crypto?.randomUUID?.()
                || `${Date.now()}-${Math.random().toString(16).slice(2)}`);
            // Nur den Label-Text austauschen; der Pro-Chip bleibt dabei an
            // seinem Platz und der CSS-Spinner kann den Ladezustand anzeigen.
            const labelEl = button.querySelector(".diff-resolve-btn-label");
            const setLabel = function (text) {
              if (labelEl) labelEl.textContent = text;
              else button.textContent = text;
            };
            const originalLabel = labelEl ? labelEl.textContent : button.textContent;
            button.disabled = true;
            button.classList.add("is-loading");
            setLabel("Asking the models…");
            window.trackUmamiEvent?.("app_resolve_started", { positions: diff.positions.length });

            try {
              const response = await fetch("/resolve", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  id_token: idToken,
                  useOwnKeys: useOwnKeys,
                  usage_run_key: resolveUsageRunKey,
                  question: question,
                  claim: diff.claim,
                  positions: diff.positions,
                  openai_key: localStorage.getItem("openaiKey") || "",
                  mistral_key: localStorage.getItem("mistralKey") || "",
                  anthropic_key: localStorage.getItem("anthropicKey") || "",
                  gemini_key: localStorage.getItem("geminiKey") || "",
                  deepseek_key: localStorage.getItem("deepseekKey") || "",
                  grok_key: localStorage.getItem("grokKey") || ""
                })
              });
              const data = await response.json().catch(() => ({}));
              if (!response.ok) {
                const detail = data?.detail && typeof data.detail === "object" ? data.detail : null;
                if (detail?.error_code === "pro_required") {
                  // Tier-Status war veraltet: Button in den Teaser-Zustand
                  // zurücksetzen und das Pro-Modal zeigen.
                  button.disabled = false;
                  button.classList.remove("is-loading");
                  setLabel(originalLabel);
                  showResolveProTeaser();
                  return;
                }
                const message = detail?.error || data?.error
                  || (typeof data?.detail === "string" ? data.detail : "")
                  || ("Resolve HTTP " + response.status);
                throw new Error(message);
              }

              window.App.renderUsageDisplay({
                remaining: data.free_usage_remaining,
                deepRemaining: data.deep_remaining,
                totalLimit: data.limit ?? window.currentMaxLimit,
                deepLimit: data.deep_limit ?? window.currentDeepLimit
              });

              renderResolveResult(resultBox, data);
              // Ladezustand beenden und Button entfernen (das [hidden] greift
              // erst durch die zugehoerige CSS-Regel, siehe Stylesheet).
              button.classList.remove("is-loading");
              button.hidden = true;
              // Ergebnis am Widerspruch merken und Karte kennzeichnen; über
              // das Bookmark persistieren, damit es beim Wiederöffnen bleibt.
              // Prompt-Feld vor der Persistenz strippen: Bookmarks/Shares
              // bleiben schlank, die Prompt-Ansicht gibt es nur live.
              diff.resolution = {
                outcome: data.outcome,
                results: (Array.isArray(data.results) ? data.results : []).map(function (r) {
                  const copy = Object.assign({}, r);
                  delete copy.prompt;
                  return copy;
                })
              };
              markCardResolved(resultBox.closest(".diff-card"), data.outcome);
              persistResolutionToBookmark();
              window.trackUmamiEvent?.("app_resolve_completed", { outcome: data.outcome });
            } catch (error) {
              console.error("Resolve round failed:", error);
              resultBox.innerHTML = "";
              const note = document.createElement("div");
              note.className = "resolve-outcome is-error";
              note.textContent = error?.message || "The resolve round failed. Please try again.";
              resultBox.appendChild(note);
              resultBox.hidden = false;
              button.disabled = false;
              button.classList.remove("is-loading");
              setLabel(originalLabel);
              window.trackUmamiEvent?.("app_resolve_completed", { outcome: "request_error" });
            }
          }

          const RESOLVE_BTN_ICON =
            '<svg class="diff-resolve-icon" viewBox="0 0 16 16" width="13" height="13" aria-hidden="true" fill="none" '
            + 'stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">'
            + '<path d="M2 5h9M11 5 8.8 2.8M11 5 8.8 7.2"/>'
            + '<path d="M14 11H5M5 11l2.2-2.2M5 11l2.2 2.2"/></svg>';

          function buildResolveSection(diff) {
            if (diff.type !== "contradiction") return null;

            const wrap = document.createElement("div");
            wrap.className = "diff-resolve";
            const resultBox = document.createElement("div");
            resultBox.className = "diff-resolve-result";
            resultBox.hidden = true;

            // Bereits gelöster Widerspruch (persistiert im Bookmark): Ergebnis
            // direkt zeigen, kein Button. Die Karten-Kennzeichnung übernimmt
            // renderDifferenceCards nach dem Einhängen der Karte.
            if (diff.resolution && RESOLVE_STATUS[diff.resolution.outcome]) {
              renderResolveResult(resultBox, diff.resolution);
              wrap.appendChild(resultBox);
              return wrap;
            }

            const involved = new Set();
            diff.positions.forEach(function (pos) {
              (pos.models || []).forEach(function (m) { if (MODEL_BOX_IDS[m]) involved.add(m); });
            });
            if (involved.size < 2 || diff.positions.length < 2) return null;

            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "diff-resolve-btn";
            btn.insertAdjacentHTML("afterbegin", RESOLVE_BTN_ICON);
            const label = document.createElement("span");
            label.className = "diff-resolve-btn-label";
            label.textContent = "Resolve with the models";
            btn.appendChild(label);
            btn.title = "Ask the disagreeing models to re-examine this point against each other's position (uses 1 request)";
            // Pro-Chip immer zeigen: Free-Nutzer sehen den Hinweis (Klick öffnet
            // die Kosten-Erklaerung), Pro-Nutzer die gleiche klare Kennzeichnung.
            const chip = document.createElement("span");
            chip.className = "pro-badge diff-resolve-pro-chip";
            chip.textContent = "Pro";
            btn.appendChild(chip);
            if (!window.isUserPro) {
              btn.classList.add("is-pro-locked");
              btn.title = "Off by default: a Resolve round is a second full round of model calls";
            }

            btn.addEventListener("click", function () {
              if (!window.isUserPro) {
                showResolveProTeaser();
                return;
              }
              runResolveRound(diff, btn, resultBox);
            });
            wrap.append(btn, resultBox);
            return wrap;
          }

          // --- Differences-Panel ----------------------------------------------
          // Die Unterschiede sind nicht mehr die zweite Spalte neben der
          // Antwort, sondern ein zugeklappter Ueberblick darunter. Waehrend der
          // Synthese ist er offen (Spinner), danach entscheidet das Ergebnis.
          const DIFF_SUMMARY_RUNNING = "Differences";
          const DIFF_SUMMARY_DONE = "See all differences";

          function differencesPanel() {
            return $("consensusDifferencesPanel");
          }

          function setPanelState(open, label) {
            const panel = differencesPanel();
            if (!panel) return;
            panel.open = !!open;
            const labelEl = panel.querySelector(".consensus-differences-summary-label");
            if (labelEl && label) labelEl.textContent = label;
          }

          window.App.differencesPanel = {
            // Zu, nicht offen: der Spinner in diesem Panel ist entfallen, weil
            // der gefuehrte Lauf dieselbe Phase schon ansagt. Ein aufgeklapptes
            // leeres Panel waere die dritte Anzeige desselben Vorgangs.
            setSynthesizing: function () { setPanelState(false, DIFF_SUMMARY_RUNNING); },
            // Freitext-Fallback: der alte Block muss sichtbar und aufgeklappt
            // erscheinen, sonst verschwindet die Analyse stillschweigend.
            expandForFallback: function () { setPanelState(true, DIFF_SUMMARY_DONE); }
          };

          // --- Differences-Karten --------------------------------------------
          // Baut die Karten in einen beliebigen Container, damit der Live-Fuss
          // und ein archivierter Turn dieselbe Darstellung teilen. Ein
          // archivierter Turn bekommt die statische Fassung: seine Sprunglinks
          // zeigten sonst auf die Antwortboxen des NEUESTEN Laufs, und eine
          // Resolve-Runde gehoert immer zum aktiven Lauf.
          function buildDifferenceCards(cards, differences, modelCount, options) {
            const opts = options || {};
            const isStatic = !!opts.static;
            const labelFor = typeof opts.modelLabel === "function"
              ? opts.modelLabel
              : modelDisplayName;
            cards.innerHTML = "";

            if (!differences.length) {
              // Flacher Empty-State: grüner Punkt + zwei Textzeilen statt Box.
              const empty = document.createElement("div");
              empty.className = "diff-empty-state";
              const dot = document.createElement("span");
              dot.className = "sev-dot is-ok";
              dot.setAttribute("aria-hidden", "true");
              const textWrap = document.createElement("div");
              textWrap.className = "diff-empty-text";
              const headline = document.createElement("div");
              headline.className = "diff-empty-headline";
              headline.textContent = "No substantive contradictions found across the "
                + modelCount + " answers.";
              const note = document.createElement("div");
              note.className = "diff-empty-note";
              note.textContent = "Agreement is a good signal, but not a guarantee of correctness.";
              textWrap.append(headline, note);
              empty.append(dot, textWrap);
              cards.appendChild(empty);
            } else {
              differences.forEach(function (diff) {
                const card = document.createElement("details");
                let cardClass = "diff-card " + (diff.type === "contradiction" ? "is-contradiction" : "is-emphasis");
                if (diff.type === "contradiction" && diff.severity === "major") cardClass += " is-major";
                card.className = cardClass;
                card.open = true;

                const summary = document.createElement("summary");
                summary.className = "diff-card-summary";
                // Severity-Punkt: trägt die Farbe, das Label bleibt dezenter Text.
                const sevDot = document.createElement("span");
                let dotCls = "is-info";
                if (diff.type === "contradiction") {
                  dotCls = diff.severity === "major" ? "is-crit" : "is-warn";
                }
                sevDot.className = "sev-dot " + dotCls;
                sevDot.setAttribute("aria-hidden", "true");
                const typeTag = document.createElement("span");
                typeTag.className = "diff-type-tag";
                // Ohne Severity (alte Bookmarks/Snapshots) bleibt das neutrale Label.
                let tagLabel = "Different emphasis";
                if (diff.type === "contradiction") {
                  tagLabel = "Contradiction";
                  if (diff.severity === "major") tagLabel = "Contradiction · critical";
                  else if (diff.severity === "minor") tagLabel = "Contradiction · minor detail";
                }
                typeTag.textContent = tagLabel;
                // Kopfzeile: Punkt + Label; nimmt nach dem Resolve auch den
                // rechtsbündigen Status-Text auf.
                const tagRow = document.createElement("span");
                tagRow.className = "diff-card-tags";
                tagRow.append(sevDot, typeTag);
                const claimEl = document.createElement("span");
                claimEl.className = "diff-card-claim";
                claimEl.textContent = diff.claim;
                summary.append(tagRow, claimEl);
                card.appendChild(summary);

                const body = document.createElement("div");
                body.className = "diff-card-body";
                diff.positions.forEach(function (pos) {
                  const posEl = document.createElement("div");
                  posEl.className = "diff-position";

                  // Modellnamen als kompakte Kopfzeile der Position statt
                  // "Position A (2 models: …)".
                  const label = document.createElement("div");
                  label.className = "diff-position-label";
                  label.textContent = pos.models.map(labelFor).join(", ");
                  posEl.appendChild(label);

                  if (pos.stance) {
                    const stance = document.createElement("div");
                    stance.className = "diff-position-stance";
                    renderInlineMarkdown(stance, pos.stance);
                    posEl.appendChild(stance);
                  }
                  if (pos.quote) {
                    const quote = document.createElement("blockquote");
                    quote.className = "diff-position-quote";
                    renderInlineMarkdown(quote, pos.quote);
                    posEl.appendChild(quote);
                  }

                  // Schlichte Textlinks (Modellname) statt Pill-Buttons.
                  if (!isStatic) {
                    const links = document.createElement("div");
                    links.className = "diff-position-links";
                    pos.models.forEach(function (model) {
                      if (!MODEL_BOX_IDS[model]) return;
                      const jump = document.createElement("button");
                      jump.type = "button";
                      jump.className = "diff-jump-link";
                      jump.textContent = labelFor(model);
                      jump.title = "Jump to the full answer from " + labelFor(model);
                      jump.addEventListener("click", function () { jumpToModelAnswer(model, pos.quote); });
                      links.appendChild(jump);
                    });
                    if (links.childNodes.length) posEl.appendChild(links);
                  }
                  body.appendChild(posEl);
                });

                if (diff.verify) {
                  const verify = document.createElement("div");
                  verify.className = "diff-verify";
                  const lead = document.createElement("b");
                  lead.textContent = "Worth verifying: ";
                  verify.append(lead, document.createTextNode(diff.verify));
                  body.appendChild(verify);
                }
                // Im Archiv nur das PERSISTIERTE Ergebnis, nie der Auslöser:
                // eine Resolve-Runde laeuft gegen die Modelle des aktiven Laufs.
                // Deshalb faellt jeder Knopf raus, auch bei einer Resolution
                // mit unbekanntem Ausgang — die baut sonst wieder den Starter.
                const resolveSection = (isStatic && !diff.resolution)
                  ? null
                  : buildResolveSection(diff);
                if (resolveSection && isStatic) {
                  resolveSection.querySelectorAll("button").forEach(b => b.remove());
                }
                if (resolveSection && resolveSection.childNodes.length) {
                  body.appendChild(resolveSection);
                }
                card.appendChild(body);
                cards.appendChild(card);
                // Persistierte Resolve-Runde (z. B. aus einem Bookmark): Karte
                // direkt als gelöst/bestätigt kennzeichnen.
                if (diff.resolution) markCardResolved(card, diff.resolution.outcome);
              });
            }
            return cards;
          }

          function renderDifferenceCards(differences, modelCount) {
            const cards = $("differencesCards");
            const diffP = document.querySelector("#consensusResponse .consensus-differences p");
            if (!cards) return;
            buildDifferenceCards(cards, differences, modelCount);

            cards.hidden = false;
            if (diffP) {
              diffP.innerHTML = "";
              diffP.hidden = true;
            }
            // Strukturierte Karten liegen ab hier zugeklappt unter der Antwort.
            setPanelState(false, DIFF_SUMMARY_DONE);
          }

          // Ein archivierter Turn hat dieselben strukturierten Daten wie der
          // Live-Lauf — bis 2026-08-17 fiel er trotzdem auf den Freitext des
          // Judges zurueck und zeigte dabei sogar dessen "BestModel:"-Zeile.
          // Gleiche Daten, gleiche Karten; interaktiv ist nur der aktive Lauf.
          function renderStoredDifferenceCards(container, differencesData, options) {
            if (!container || !differencesData || typeof differencesData !== "object") {
              return false;
            }
            if (!Array.isArray(differencesData.differences)) return false;
            const differences = differencesData.differences.filter(
              d => d && d.claim && Array.isArray(d.positions) && d.positions.length
            );
            const modelCount = (Array.isArray(differencesData.models_compared)
              && differencesData.models_compared.length)
              || Number(differencesData.agreement?.model_count)
              || 0;
            buildDifferenceCards(container, differences, modelCount, {
              static: true,
              modelLabel: options && options.modelLabel
            });
            return true;
          }

          // --- Reset & Haupteinstieg -----------------------------------------
          function resetConsensusInsights() {
            closeClaimPopover();
            const verdict = $("consensusVerdict");
            if (verdict) {
              verdict.hidden = true;
              verdict.innerHTML = "";
              verdict.classList.remove("is-calm", "is-warn");
            }
            const cards = $("differencesCards");
            if (cards) {
              cards.hidden = true;
              cards.innerHTML = "";
            }
            const fallbackBox = $("consensusClaimsFallback");
            if (fallbackBox) {
              fallbackBox.hidden = true;
              fallbackBox.innerHTML = "";
            }
            const legend = $("consensusMarkerLegend");
            if (legend) legend.hidden = true;
            // Nur der aktive Consensus wird fuer den naechsten Lauf
            // zurueckgesetzt. Statische Follow-up-Turns in #threadHistory
            // behalten ihre sichtbaren Quoten und Marken.
            const insightRoot = window.App.consensusBodyEl?.() || document;
            insightRoot.querySelectorAll(".claim-badge").forEach(function (el) { el.remove(); });
            // Inline-Markierungen auflösen: Span entfernen, Text an Ort und
            // Stelle lassen. normalize() führt die Textknoten wieder zusammen,
            // damit eine erneute Ankersuche nicht an Fragmenten scheitert.
            insightRoot.querySelectorAll(".cx-claim").forEach(function (span) {
              const parent = span.parentNode;
              if (!parent) return;
              while (span.firstChild) parent.insertBefore(span.firstChild, span);
              span.remove();
              parent.normalize();
            });
            const diffP = document.querySelector("#consensusResponse .consensus-differences p");
            if (diffP) diffP.hidden = false;
          }

          function renderConsensusInsights(data, includedCount) {
            resetConsensusInsights();
            if (!data || typeof data !== "object") return false;

            const claims = (Array.isArray(data.claims) ? data.claims : [])
              .filter(c => c && c.anchor && Array.isArray(c.agree) && Array.isArray(c.dissent));
            const differences = (Array.isArray(data.differences) ? data.differences : [])
              .filter(d => d && d.claim && Array.isArray(d.positions) && d.positions.length);
            const modelCount = (Array.isArray(data.models_compared) && data.models_compared.length)
              || includedCount || 0;
            const modelsCompared = Array.isArray(data.models_compared)
              ? data.models_compared : [];
            const agreement = (data.agreement && typeof data.agreement === "object") ? data.agreement : null;
            const judge = (data.judges && typeof data.judges === "object"
              && data.judges.differences && typeof data.judges.differences === "object")
              ? data.judges.differences : null;

            renderVerdictHeader(differences, modelCount, agreement, judge);
            // Karten zuerst: die Inline-Marker verlinken per Index auf sie.
            renderDifferenceCards(differences, modelCount);
            const marks = renderInlineMarkers(claims, differences, modelsCompared) || {};
            window.trackUmamiEvent?.("app_consensus_insights_rendered", {
              claims: claims.length,
              // Wie viele Claims wirklich IM Text markiert wurden statt nur in
              // der Fallback-Liste zu landen: die eine Zahl, an der sich eine
              // Aenderung an Ankern oder Ankersuche messen laesst.
              claims_anchored: marks.claims_anchored ?? null,
              claims_unanchored: marks.claims_unanchored ?? null,
              diffs_unanchored: marks.diffs_unanchored ?? null,
              differences: differences.length,
              contradictions: differences.filter(d => d.type === "contradiction").length,
              major_contradictions: differences.filter(d => d.type === "contradiction" && d.severity === "major").length,
              agreement_score: agreement ? agreement.score : null
            });
            return true;
          }

          window.renderConsensusInsights = renderConsensusInsights;
          window.renderStoredConsensusClaims = renderStoredConsensusClaims;
          window.renderStoredDifferenceCards = renderStoredDifferenceCards;
          window.resetConsensusInsights = resetConsensusInsights;
          window.jumpToModelAnswer = jumpToModelAnswer;
        })();
