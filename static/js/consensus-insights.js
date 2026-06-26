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

          // --- Verdict-Header ------------------------------------------------
          function renderVerdictHeader(differences, modelCount) {
            const verdict = $("consensusVerdict");
            if (!verdict) return;
            const contradictions = differences.filter(d => d.type === "contradiction").length;
            const emphases = differences.length - contradictions;
            const modelsLabel = modelCount + " model" + (modelCount === 1 ? "" : "s");

            let cls, text;
            if (contradictions === 0) {
              cls = "is-calm";
              const detail = emphases > 0
                ? emphases + " difference" + (emphases === 1 ? "" : "s") + " in emphasis, no contradictions"
                : "no contradictions found";
              text = "High agreement — " + modelsLabel + ", " + detail;
            } else {
              cls = "is-warn";
              text = "Caution: the models contradict each other on "
                + contradictions + " point" + (contradictions === 1 ? "" : "s")
                + " — " + modelsLabel;
            }

            verdict.classList.remove("is-calm", "is-warn");
            verdict.classList.add(cls);
            verdict.innerHTML = "";
            const icon = document.createElement("span");
            icon.className = "verdict-icon";
            icon.setAttribute("aria-hidden", "true");
            const label = document.createElement("span");
            label.className = "verdict-text";
            label.textContent = text;
            verdict.append(icon, label);
            verdict.hidden = false;
          }

          // --- Agreement-Badges in der Konsens-Antwort -----------------------
          function makeBadge(claim) {
            const agreeCount = claim.agree.length;
            const total = agreeCount + claim.dissent.length;
            const badge = document.createElement("button");
            badge.type = "button";
            badge.className = "claim-badge" + (claim.dissent.length ? " has-dissent" : "");
            badge.textContent = agreeCount + "/" + total;
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

          // --- Differences-Karten --------------------------------------------
          function renderDifferenceCards(differences, modelCount) {
            const cards = $("differencesCards");
            const diffP = document.querySelector("#consensusResponse .consensus-differences p");
            if (!cards) return;
            cards.innerHTML = "";

            if (!differences.length) {
              const empty = document.createElement("div");
              empty.className = "diff-empty-state";
              const headline = document.createElement("div");
              headline.className = "diff-empty-headline";
              headline.textContent = "No substantive contradictions found across the "
                + modelCount + " answers.";
              const note = document.createElement("div");
              note.className = "diff-empty-note";
              note.textContent = "Agreement is a good signal, but not a guarantee of correctness.";
              empty.append(headline, note);
              cards.appendChild(empty);
            } else {
              const openByDefault = !isMobileViewport();
              differences.forEach(function (diff) {
                const card = document.createElement("details");
                card.className = "diff-card " + (diff.type === "contradiction" ? "is-contradiction" : "is-emphasis");
                if (openByDefault) card.open = true;

                const summary = document.createElement("summary");
                summary.className = "diff-card-summary";
                const typeTag = document.createElement("span");
                typeTag.className = "diff-type-tag";
                typeTag.textContent = diff.type === "contradiction" ? "Contradiction" : "Different emphasis";
                const claimEl = document.createElement("span");
                claimEl.className = "diff-card-claim";
                claimEl.textContent = diff.claim;
                summary.append(typeTag, claimEl);
                card.appendChild(summary);

                const body = document.createElement("div");
                body.className = "diff-card-body";
                diff.positions.forEach(function (pos, index) {
                  const posEl = document.createElement("div");
                  posEl.className = "diff-position";

                  const label = document.createElement("div");
                  label.className = "diff-position-label";
                  const names = pos.models.map(modelDisplayName).join(", ");
                  label.textContent = "Position " + String.fromCharCode(65 + index) + " ("
                    + (pos.models.length === 1 ? names : pos.models.length + " models: " + names) + ")";
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

                  const links = document.createElement("div");
                  links.className = "diff-position-links";
                  pos.models.forEach(function (model) {
                    if (!MODEL_BOX_IDS[model]) return;
                    const jump = document.createElement("button");
                    jump.type = "button";
                    jump.className = "claim-jump-link";
                    jump.textContent = "Jump to " + modelDisplayName(model);
                    jump.addEventListener("click", function () { jumpToModelAnswer(model, pos.quote); });
                    links.appendChild(jump);
                  });
                  if (links.childNodes.length) posEl.appendChild(links);
                  body.appendChild(posEl);
                });

                if (diff.verify) {
                  const verify = document.createElement("div");
                  verify.className = "diff-verify";
                  verify.textContent = "Worth verifying: " + diff.verify;
                  body.appendChild(verify);
                }
                card.appendChild(body);
                cards.appendChild(card);
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

            renderVerdictHeader(differences, modelCount);
            renderClaimBadges(claims);
            renderDifferenceCards(differences, modelCount);
            window.trackUmamiEvent?.("app_consensus_insights_rendered", {
              claims: claims.length,
              differences: differences.length,
              contradictions: differences.filter(d => d.type === "contradiction").length
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

          const MIN_DIFF = 0.18; // Differences nie schmaler als ~18 %
          const MAX_DIFF = 0.5;  // ...und nie breiter als die Antwortspalte
          let appliedFrac = null;
          let scheduled = false;

          const resetColumns = () => {
            main.style.flex = "";
            diff.style.flex = "";
            appliedFrac = null;
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
            if (appliedFrac !== null && Math.abs(frac - appliedFrac) < 0.02) return;
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

