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

          function isMobileViewport() {
            return window.matchMedia("(max-width: 768px)").matches;
          }

          // --- Textsuche: Whitespace kollabieren, Anführungszeichen vereinheitlichen
          function normalizeForSearch(value) {
            return String(value || "")
              .toLowerCase()
              .replace(/[“”„‘’«»"]/g, '"')
              .replace(/\s+/g, " ")
              .trim();
          }

          // Sucht den normalisierten Needle in einem Textknoten und liefert die
          // Original-Offsets (für splitText/Range) zurück.
          function findRangeInTextNode(node, normNeedle) {
            const raw = node.nodeValue || "";
            let norm = "";
            const map = [];
            for (let i = 0; i < raw.length; i++) {
              let ch = raw[i].toLowerCase();
              if (/[“”„‘’«»"]/.test(ch)) ch = '"';
              if (/\s/.test(ch)) {
                if (norm.endsWith(" ") || norm === "") continue;
                ch = " ";
              }
              norm += ch;
              map.push(i);
            }
            const idx = norm.indexOf(normNeedle);
            if (idx === -1 || !normNeedle) return null;
            return { start: map[idx], end: map[idx + normNeedle.length - 1] + 1 };
          }

          function searchVariants(text) {
            const norm = normalizeForSearch(text).replace(/^(\.{3}|…)\s*/, "").replace(/\s*(\.{3}|…)$/, "");
            if (!norm) return [];
            const variants = [norm];
            const words = norm.split(" ");
            if (words.length > 8) variants.push(words.slice(0, 8).join(" "));
            return variants;
          }

          function findAnchorTarget(container, anchor) {
            for (const needle of searchVariants(anchor)) {
              const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
              let node;
              while ((node = walker.nextNode())) {
                if (!node.nodeValue || node.parentElement?.closest(".claim-badge, code, pre")) continue;
                const range = findRangeInTextNode(node, needle);
                if (range) return { type: "exact", node, range };
              }
              const blocks = container.querySelectorAll("p, li, h1, h2, h3, h4, td, blockquote");
              for (const block of blocks) {
                if (normalizeForSearch(block.textContent).includes(needle)) {
                  return { type: "block", block };
                }
              }
            }
            return null;
          }

          // --- Popover / Bottom Sheet -------------------------------------
          function onDocClick(event) {
            const pop = $("claimPopover");
            if (!pop || pop.hidden) return;
            if (pop.contains(event.target)) return;
            if (event.target.closest?.(".claim-badge")) return;
            closeClaimPopover();
          }

          function onKeyDown(event) {
            if (event.key === "Escape") closeClaimPopover();
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

          function closeClaimPopover() {
            const pop = $("claimPopover");
            const backdrop = $("claimSheetBackdrop");
            if (pop) {
              pop.hidden = true;
              pop.classList.remove("is-modal");
              pop.innerHTML = "";
              pop.style.left = pop.style.top = pop.style.width = "";
            }
            if (backdrop) backdrop.hidden = true;
            document.removeEventListener("click", onDocClick, true);
            document.removeEventListener("keydown", onKeyDown, true);
          }

          function buildModelRow(model, quote, agreeing) {
            const row = document.createElement("div");
            row.className = "claim-model-row " + (agreeing ? "is-agree" : "is-dissent");

            const head = document.createElement("div");
            head.className = "claim-model-head";
            const name = document.createElement("span");
            name.className = "claim-model-name";
            name.textContent = modelDisplayName(model);
            head.appendChild(name);
            if (MODEL_BOX_IDS[model]) {
              const jump = document.createElement("button");
              jump.type = "button";
              jump.className = "claim-jump-link";
              jump.textContent = "View answer";
              jump.addEventListener("click", function () {
                closeClaimPopover();
                jumpToModelAnswer(model, quote);
              });
              head.appendChild(jump);
            }
            row.appendChild(head);

            if (quote) {
              const q = document.createElement("blockquote");
              q.className = "claim-model-quote";
              q.textContent = quote;
              row.appendChild(q);
            }
            return row;
          }

          function openClaimPopover(claim, anchorEl) {
            const pop = ensureOverlayOnBody($("claimPopover"));
            const backdrop = ensureOverlayOnBody($("claimSheetBackdrop"));
            if (!pop) return;
            closeClaimPopover();

            const agreeCount = claim.agree.length;
            const total = agreeCount + claim.dissent.length;

            const header = document.createElement("div");
            header.className = "claim-popover-header";
            const title = document.createElement("span");
            title.className = "claim-popover-title";
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
            claimText.textContent = "“" + claim.anchor + "”";
            pop.appendChild(claimText);

            if (claim.agree.length) {
              const section = document.createElement("div");
              section.className = "claim-popover-section";
              const label = document.createElement("div");
              label.className = "claim-section-label is-agree";
              label.textContent = "Agree";
              section.appendChild(label);
              claim.agree.forEach(model => section.appendChild(buildModelRow(model, "", true)));
              pop.appendChild(section);
            }
            if (claim.dissent.length) {
              const section = document.createElement("div");
              section.className = "claim-popover-section";
              const label = document.createElement("div");
              label.className = "claim-section-label is-dissent";
              label.textContent = "Deviate";
              section.appendChild(label);
              claim.dissent.forEach(item => section.appendChild(buildModelRow(item.model, item.quote, false)));
              pop.appendChild(section);
            }

            const asModal = isMobileViewport();
            pop.classList.toggle("is-modal", asModal);
            pop.hidden = false;
            if (asModal) {
              if (backdrop) {
                backdrop.hidden = false;
                backdrop.addEventListener("click", closeClaimPopover, { once: true });
              }
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

            let delay = 0;
            if (window.isAgentModeEnabled?.()) {
              // Agent Mode blendet die Antwort-Boxen aus — erst deaktivieren,
              // dann nach dem Layout-Übergang springen.
              window.setAgentMode?.(false, { persist: true });
              delay = 400;
            }

            window.setTimeout(function () {
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
            }, delay);
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
            const modelsLabel = modelCount + " model" + (modelCount === 1 ? "" : "s") + " compared";
            const hasScore = agreement && typeof agreement.score === "number";

            const cls = contradictions === 0 ? "is-calm" : "is-warn";
            verdict.classList.remove("is-calm", "is-warn");
            verdict.classList.add(cls);
            verdict.innerHTML = "";

            // Score-Ring; alte Bookmarks ohne Score behalten den kleinen Punkt.
            if (hasScore) {
              const score = Math.max(0, Math.min(100, agreement.score));
              const ring = document.createElement("span");
              ring.className = "verdict-score";
              ring.style.setProperty("--val", String(score));
              ring.title = "Agreement score " + agreement.score + "/100";
              const fill = document.createElement("span");
              fill.className = "verdict-score-ring";
              fill.setAttribute("aria-hidden", "true");
              const num = document.createElement("span");
              num.className = "verdict-score-num";
              num.textContent = String(agreement.score);
              // Screenreader (und E2E-Check) lesen den vollen Score.
              const unit = document.createElement("span");
              unit.className = "visually-hidden";
              unit.textContent = "/100 agreement score";
              num.appendChild(unit);
              ring.append(fill, num);
              verdict.appendChild(ring);
            } else {
              const icon = document.createElement("span");
              icon.className = "verdict-icon";
              icon.setAttribute("aria-hidden", "true");
              verdict.appendChild(icon);
            }

            const main = document.createElement("span");
            main.className = "verdict-main";
            const headline = document.createElement("span");
            headline.className = "verdict-headline";
            headline.textContent = contradictions === 0
              ? "High agreement"
              : "The models contradict each other on " + contradictions
                + " point" + (contradictions === 1 ? "" : "s");
            main.appendChild(headline);

            const detail = document.createElement("span");
            detail.className = "verdict-detail";
            if (contradictions === 0) {
              const note = emphases > 0
                ? emphases + " difference" + (emphases === 1 ? "" : "s") + " in emphasis, no contradictions"
                : "no contradictions found";
              detail.textContent = note + " — " + modelsLabel;
            } else if (hasSeverity && critical > 0) {
              const crit = document.createElement("span");
              crit.className = "verdict-detail-crit";
              crit.textContent = critical + " critical";
              detail.appendChild(crit);
              const minor = contradictions - critical;
              detail.appendChild(document.createTextNode(
                (minor > 0 ? " · " + minor + " minor detail" + (minor === 1 ? "" : "s") : "")
                + " — " + modelsLabel));
            } else if (hasSeverity) {
              detail.textContent = "minor details — " + modelsLabel;
            } else {
              detail.textContent = modelsLabel;
            }
            main.appendChild(detail);
            verdict.appendChild(main);

            // Transparenz: welche (unabhängige) Modellfamilie die Analyse
            // geliefert hat. Alte Bookmarks/Snapshots ohne judges-Feld zeigen
            // schlicht keine Fußnote.
            if (judge && judge.provider) {
              const note = document.createElement("span");
              note.className = "verdict-judge";
              const provider = document.createElement("span");
              provider.textContent = "Analysis by " + judge.provider
                + (judge.tier === "pro" ? " (Pro)" : "");
              const sub = document.createElement("span");
              sub.className = "verdict-judge-sub";
              sub.textContent = "independent of the consensus engine";
              note.append(provider, sub);
              verdict.appendChild(note);
            }
            verdict.hidden = false;
          }

          // --- Agreement-Badges in der Konsens-Antwort -----------------------
          function makeBadge(claim) {
            const agreeCount = claim.agree.length;
            const total = agreeCount + claim.dissent.length;
            const badge = document.createElement("button");
            badge.type = "button";
            badge.className = "claim-badge" + (claim.dissent.length ? " has-dissent" : "");
            // Dissens nur als kleiner Amber-Punkt im monochromen Chip,
            // nicht mehr als gefüllte Pill.
            if (claim.dissent.length) {
              const dot = document.createElement("span");
              dot.className = "claim-dot";
              dot.setAttribute("aria-hidden", "true");
              badge.appendChild(dot);
            }
            badge.appendChild(document.createTextNode(agreeCount + "/" + total));
            badge.title = claim.dissent.length
              ? agreeCount + " of " + total + " models support this — tap for details"
              : "All " + total + " models that address this agree — tap for details";
            badge.setAttribute("aria-haspopup", "dialog");
            badge.addEventListener("click", function (event) {
              event.stopPropagation();
              openClaimPopover(claim, badge);
            });
            return badge;
          }

          function renderClaimBadges(claims) {
            const mainP = document.querySelector("#consensusResponse .consensus-main p");
            const fallbackBox = $("consensusClaimsFallback");
            if (!mainP || !fallbackBox) return;

            const unanchored = [];
            claims.forEach(function (claim) {
              const target = findAnchorTarget(mainP, claim.anchor);
              if (!target) {
                unanchored.push(claim);
                return;
              }
              const badge = makeBadge(claim);
              if (target.type === "exact") {
                const node = target.node;
                const offset = Math.min(target.range.end, node.nodeValue.length);
                const rest = node.splitText(offset);
                node.parentNode.insertBefore(badge, rest);
              } else {
                target.block.appendChild(badge);
              }
            });

            if (unanchored.length) {
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
                text.textContent = claim.anchor;
                row.append(text, makeBadge(claim));
                fallbackBox.appendChild(row);
              });
              fallbackBox.hidden = false;
            }
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
              payload.differencesData
            );
          }

          function showResolveProTeaser() {
            window.trackUmamiEvent?.("app_resolve_pro_teaser_click");
            const shown = window.App?.showProFeatureModal?.("Resolve");
            if (!shown) {
              window.App?.showPopup?.("Resolve rounds are a Pro feature.");
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

              if (data.free_usage_remaining !== undefined) {
                const usageEl = $("freeUsageDisplay");
                if (usageEl) usageEl.innerText = "Requests: " + data.free_usage_remaining + " / " + window.currentMaxLimit;
              }

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
            // Pro-Chip immer zeigen: Free-Nutzer sehen den Teaser (Klick öffnet
            // das Upgrade-Modal), Pro-Nutzer die gleiche klare Kennzeichnung.
            const chip = document.createElement("span");
            chip.className = "pro-badge diff-resolve-pro-chip";
            chip.textContent = "Pro";
            btn.appendChild(chip);
            if (!window.isUserPro) {
              btn.classList.add("is-pro-locked");
              btn.title = "Resolve rounds are a Pro feature";
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

          // --- Differences-Karten --------------------------------------------
          function renderDifferenceCards(differences, modelCount) {
            const cards = $("differencesCards");
            const diffP = document.querySelector("#consensusResponse .consensus-differences p");
            if (!cards) return;
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
                  label.textContent = pos.models.map(modelDisplayName).join(", ");
                  posEl.appendChild(label);

                  if (pos.stance) {
                    const stance = document.createElement("div");
                    stance.className = "diff-position-stance";
                    stance.textContent = pos.stance;
                    posEl.appendChild(stance);
                  }
                  if (pos.quote) {
                    const quote = document.createElement("blockquote");
                    quote.className = "diff-position-quote";
                    quote.textContent = pos.quote;
                    posEl.appendChild(quote);
                  }

                  // Schlichte Textlinks (Modellname) statt Pill-Buttons.
                  const links = document.createElement("div");
                  links.className = "diff-position-links";
                  pos.models.forEach(function (model) {
                    if (!MODEL_BOX_IDS[model]) return;
                    const jump = document.createElement("button");
                    jump.type = "button";
                    jump.className = "diff-jump-link";
                    jump.textContent = modelDisplayName(model);
                    jump.title = "Jump to the full answer from " + modelDisplayName(model);
                    jump.addEventListener("click", function () { jumpToModelAnswer(model, pos.quote); });
                    links.appendChild(jump);
                  });
                  if (links.childNodes.length) posEl.appendChild(links);
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
                const resolveSection = buildResolveSection(diff);
                if (resolveSection) body.appendChild(resolveSection);
                card.appendChild(body);
                cards.appendChild(card);
                // Persistierte Resolve-Runde (z. B. aus einem Bookmark): Karte
                // direkt als gelöst/bestätigt kennzeichnen.
                if (diff.resolution) markCardResolved(card, diff.resolution.outcome);
              });
            }

            cards.hidden = false;
            if (diffP) {
              diffP.innerHTML = "";
              diffP.hidden = true;
            }
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
            document.querySelectorAll(".claim-badge").forEach(function (badge) { badge.remove(); });
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
            const agreement = (data.agreement && typeof data.agreement === "object") ? data.agreement : null;
            const judge = (data.judges && typeof data.judges === "object"
              && data.judges.differences && typeof data.judges.differences === "object")
              ? data.judges.differences : null;

            renderVerdictHeader(differences, modelCount, agreement, judge);
            renderClaimBadges(claims);
            renderDifferenceCards(differences, modelCount);
            window.trackUmamiEvent?.("app_consensus_insights_rendered", {
              claims: claims.length,
              differences: differences.length,
              contradictions: differences.filter(d => d.type === "contradiction").length,
              major_contradictions: differences.filter(d => d.type === "contradiction" && d.severity === "major").length,
              agreement_score: agreement ? agreement.score : null
            });
            return true;
          }

          window.renderConsensusInsights = renderConsensusInsights;
          window.resetConsensusInsights = resetConsensusInsights;
          window.jumpToModelAnswer = jumpToModelAnswer;
        })();

        // --------------------------------------------------------------------
        // Konsens-Layout: Breite des Differences-Bereichs dynamisch an die
        // Inhaltsmenge anpassen. Ist die Consensus-Antwort kurz, die
        // Differences aber lang, bekommt Differences mehr Breite (statt schmal
        // und sehr hoch zu werden). Wir gleichen die natürlichen Inhaltshöhen
        // beider Spalten an, indem die Breiten proportional zur Inhaltsfläche
        // (natürliche Höhe × aktuelle Breite) verteilt werden.
        // --------------------------------------------------------------------
        (function setupConsensusColumnBalancer() {
          const box = document.getElementById("consensusResponse");
          if (!box) return;
          const main = box.querySelector(".consensus-main");
          const diff = box.querySelector(".consensus-differences");
          if (!main || !diff || typeof ResizeObserver === "undefined") return;

          const MIN_DIFF = 0.24; // Differences nie schmaler als ~24 % (Redesign 2026-07: breitere Spalte)
          const MAX_DIFF = 0.5;  // ...und nie breiter als die Antwortspalte
          let appliedFrac = null;
          let prevAppliedFrac = null;
          let scheduled = false;

          const resetColumns = () => {
            main.style.flex = "";
            diff.style.flex = "";
            appliedFrac = null;
            prevAppliedFrac = null;
          };

          // Natürliche Inhaltshöhe messen: align-items:stretch zwingt beide
          // Spalten sonst auf dieselbe (gestreckte) Höhe, daher kurz auf
          // flex-start setzen, messen und zurücksetzen.
          const naturalHeight = (col) => {
            const prev = col.style.alignSelf;
            col.style.alignSelf = "flex-start";
            const h = col.scrollHeight;
            col.style.alignSelf = prev;
            return h;
          };

          const balance = () => {
            scheduled = false;
            // Gestapeltes Mobil-Layout, ausgeblendet oder im Lade-/Synthese-
            // Zustand: das Stylesheet entscheiden lassen.
            if (!box.offsetParent
                || box.classList.contains("is-synthesizing")
                || getComputedStyle(box).flexDirection !== "row") {
              if (appliedFrac !== null) resetColumns();
              return;
            }
            const wMain = main.clientWidth;
            const wDiff = diff.clientWidth;
            if (wMain <= 0 || wDiff <= 0) return;

            const areaMain = naturalHeight(main) * wMain;
            const areaDiff = naturalHeight(diff) * wDiff;
            if (areaMain <= 0 || areaDiff <= 0) return;

            let frac = areaDiff / (areaMain + areaDiff);
            frac = Math.max(MIN_DIFF, Math.min(MAX_DIFF, frac));

            // Rückkopplungsschleife vermeiden: nur bei spürbarer Änderung neu
            // schreiben (Breitenänderung triggert den ResizeObserver erneut).
            // Hysterese bewusst größer als das Reflow-Rauschen: Umbrüche nach
            // einer Breitenänderung verschieben die gemessene Inhaltsfläche
            // leicht, was sonst zwischen zwei Breiten oszilliert (Flackern).
            if (appliedFrac !== null && Math.abs(frac - appliedFrac) < 0.05) return;
            // Bounce-Guard: springt der Wert zurück auf die vorletzte Breite
            // (A -> B -> A ...), liegt eine Mess-Oszillation vor - einfrieren.
            if (prevAppliedFrac !== null && Math.abs(frac - prevAppliedFrac) < 0.01) return;
            prevAppliedFrac = appliedFrac;
            appliedFrac = frac;
            main.style.flex = (1 - frac).toFixed(4);
            diff.style.flex = frac.toFixed(4);
          };

          const schedule = () => {
            if (scheduled) return;
            scheduled = true;
            requestAnimationFrame(balance);
          };

          const ro = new ResizeObserver(schedule);
          ro.observe(main);
          ro.observe(diff);
          window.addEventListener("resize", schedule);
          window.balanceConsensusColumns = schedule;
        })();

