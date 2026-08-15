/* ==========================================================================
   consens.io — landing page: the marks in the mockups actually work

   The consensus mockups on the landing page are exact rebuilds of the /app
   surfaces, but until now only in appearance: the agreement ratio behind a
   sentence and the dot beside a marked passage were rendered and then did
   nothing. That is the one gesture the whole product is built around — "who
   said that, and who didn't?" — so a visitor who tried it on the landing page
   learned that it is a picture.

   This file gives those marks the same two behaviours they have in
   static/js/consensus-insights.js:

     hover (pointer devices)  short version in an .insight-preview card
     click / Enter            the full breakdown

   For a ratio badge the full breakdown is the .claim-popover (anchored on a
   desktop, a centred modal on a phone), built from data- attributes on the
   badge itself. For a difference marker it stays what it already was: the
   differences panel opens and the matching card flashes — its hover preview
   is read straight out of that card, so the two can never drift apart.

   Deliberately NOT ported from the app: the "View answer" jump inside a model
   row. There are no model answers behind this mockup, and a button that
   promises one would be the exact dishonesty the page is arguing against.
   ========================================================================== */
(function () {
  "use strict";

  const HOVER_DELAY_MS = 130;
  const MOBILE_QUERY = "(max-width: 700px)";

  const canHover = () =>
    !!(window.matchMedia && window.matchMedia("(hover: hover) and (pointer: fine)").matches);
  const isPhone = () =>
    !!(window.matchMedia && window.matchMedia(MOBILE_QUERY).matches);

  /* "ChatGPT, Claude, Gemini" -> ["ChatGPT", "Claude", "Gemini"] */
  const list = (value) =>
    (value || "").split(",").map(s => s.trim()).filter(Boolean);

  /* "Mistral: they say X | Grok: they say Y" -> [{model, quote}, ...]
     The quote is optional; everything after the first colon belongs to it. */
  const positions = (value) => (value || "")
    .split("|")
    .map(entry => entry.trim())
    .filter(Boolean)
    .map(entry => {
      const at = entry.indexOf(":");
      return at === -1
        ? { model: entry, quote: "" }
        : { model: entry.slice(0, at).trim(), quote: entry.slice(at + 1).trim() };
    });

  // ---------------------------------------------------------------- preview
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

  function hidePreview() {
    window.clearTimeout(hoverTimer);
    hoverTimer = null;
    if (hoverCard) {
      hoverCard.hidden = true;
      hoverCard.replaceChildren();
    }
  }

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

  /* Anchored under the element that owns the mark, flipped above it when the
     bottom of the window is closer than the card is tall. */
  function placePreview(card, anchor) {
    const rect = anchor.getBoundingClientRect();
    const width = Math.min(340, window.innerWidth - 24);
    card.style.width = width + "px";

    const centred = rect.left + rect.width / 2 - width / 2;
    const left = Math.max(12, Math.min(centred, window.innerWidth - width - 12));
    card.style.left = (left + window.scrollX) + "px";

    const height = card.offsetHeight;
    const below = rect.bottom + 8;
    const fitsBelow = below + height <= window.innerHeight - 12;
    const top = fitsBelow ? below : Math.max(12, rect.top - height - 8);
    card.style.top = (top + window.scrollY) + "px";
  }

  function schedulePreview(anchor, build) {
    window.clearTimeout(hoverTimer);
    hoverTimer = window.setTimeout(function () {
      const content = build();
      if (!content) return;
      const card = ensureHoverCard();
      card.replaceChildren(content);
      card.hidden = false;
      placePreview(card, anchor);
    }, HOVER_DELAY_MS);
  }

  // A card pinned to a screen position is wrong the moment anything moves.
  window.addEventListener("scroll", hidePreview, true);
  window.addEventListener("resize", hidePreview);

  // ---------------------------------------------------------------- popover
  let popover = null;
  let backdrop = null;
  let popoverTrigger = null;

  function ensurePopover() {
    if (popover && popover.isConnected) return popover;
    backdrop = document.createElement("div");
    backdrop.className = "claim-sheet-backdrop";
    backdrop.hidden = true;
    backdrop.addEventListener("click", () => closePopover());

    popover = document.createElement("div");
    popover.className = "claim-popover";
    popover.setAttribute("role", "dialog");
    popover.hidden = true;

    document.body.append(backdrop, popover);
    return popover;
  }

  function closePopover(options) {
    const restoreFocus = !options || options.restoreFocus !== false;
    if (popover) {
      popover.hidden = true;
      popover.replaceChildren();
      popover.classList.remove("is-modal");
      popover.removeAttribute("aria-modal");
      popover.style.width = "";
    }
    if (backdrop) backdrop.hidden = true;
    document.removeEventListener("keydown", onKeyDown, true);
    document.removeEventListener("click", onDocClick, true);
    if (restoreFocus && popoverTrigger && popoverTrigger.isConnected) {
      popoverTrigger.focus();
    }
    popoverTrigger = null;
  }

  function onKeyDown(event) {
    if (event.key === "Escape") {
      event.stopPropagation();
      closePopover();
    }
  }

  function onDocClick(event) {
    if (!popover || popover.hidden) return;
    if (popover.contains(event.target)) return;
    if (popoverTrigger && popoverTrigger.contains(event.target)) return;
    closePopover({ restoreFocus: false });
  }

  function modelRow(model, quote) {
    const row = document.createElement("div");
    row.className = "claim-model-row";
    const name = document.createElement("div");
    name.className = "claim-model-name";
    name.textContent = model;
    row.appendChild(name);
    if (quote) {
      const q = document.createElement("blockquote");
      q.className = "claim-model-quote";
      q.textContent = quote;
      row.appendChild(q);
    }
    return row;
  }

  function popoverSection(label, labelClass, rows) {
    const section = document.createElement("div");
    section.className = "claim-popover-section";
    const head = document.createElement("div");
    head.className = "claim-section-label" + (labelClass ? " " + labelClass : "");
    head.textContent = label;
    section.appendChild(head);
    rows.forEach(row => section.appendChild(row));
    return section;
  }

  function openClaimPopover(claim, anchor) {
    const pop = ensurePopover();
    closePopover({ restoreFocus: false });
    hidePreview();
    popoverTrigger = anchor;

    const header = document.createElement("div");
    header.className = "claim-popover-header";
    const title = document.createElement("span");
    title.className = "claim-popover-title";
    title.textContent = claim.dissent.length
      ? claim.agree.length + " of " + (claim.agree.length + claim.dissent.length) + " models agree"
      : "All " + claim.agree.length + " models agree";
    const close = document.createElement("button");
    close.type = "button";
    close.className = "claim-popover-close";
    close.setAttribute("aria-label", "Close");
    close.innerHTML = "&times;";
    close.addEventListener("click", () => closePopover());
    header.append(title, close);
    pop.appendChild(header);

    if (claim.anchor) {
      const text = document.createElement("div");
      text.className = "claim-popover-claim";
      text.textContent = "“" + claim.anchor + "”";
      pop.appendChild(text);
    }

    if (claim.agree.length) {
      pop.appendChild(popoverSection("Agree", "is-agree",
        claim.agree.map(model => modelRow(model, ""))));
    }
    if (claim.dissent.length) {
      pop.appendChild(popoverSection("Deviate", "is-dissent",
        claim.dissent.map(item => modelRow(item.model, item.quote))));
    }
    if (claim.missing.length) {
      pop.appendChild(popoverSection("Not addressed", "",
        claim.missing.map(model => modelRow(model, ""))));
    }

    const asModal = isPhone();
    pop.classList.toggle("is-modal", asModal);
    pop.hidden = false;

    if (asModal) {
      pop.setAttribute("aria-modal", "true");
      if (backdrop) backdrop.hidden = false;
      window.requestAnimationFrame(() => close.focus());
    } else {
      const rect = anchor.getBoundingClientRect();
      const width = Math.min(340, window.innerWidth - 24);
      pop.style.width = width + "px";
      const centred = rect.left + rect.width / 2 - width / 2;
      const left = Math.max(12, Math.min(centred, window.innerWidth - width - 12));
      pop.style.left = (left + window.scrollX) + "px";
      // Under the badge while there is room under it, above it otherwise —
      // a panel that opens off the bottom of the window reads as broken.
      const height = pop.offsetHeight;
      const fitsBelow = rect.bottom + 8 + height <= window.innerHeight - 12;
      const top = fitsBelow
        ? rect.bottom + 8
        : Math.max(12, rect.top - height - 8);
      pop.style.top = (top + window.scrollY) + "px";
    }

    document.addEventListener("keydown", onKeyDown, true);
    // Deferred, so the click that opened it does not immediately close it.
    window.setTimeout(() => document.addEventListener("click", onDocClick, true), 0);
  }

  // ------------------------------------------------------------------ wiring
  function readClaim(badge) {
    return {
      anchor: badge.dataset.claimAnchor || "",
      agree: list(badge.dataset.claimAgree),
      dissent: positions(badge.dataset.claimDissent),
      missing: list(badge.dataset.claimMissing)
    };
  }

  function buildClaimPreview(claim) {
    const frag = document.createDocumentFragment();
    const total = claim.agree.length + claim.dissent.length;
    frag.appendChild(previewHead(
      claim.dissent.length
        ? claim.agree.length + " of " + total + " models support this"
        : "All " + total + " models agree",
      claim.dissent.length ? "is-warn" : null
    ));
    if (claim.agree.length) frag.appendChild(previewRow("Agree", claim.agree.join(", ")));
    if (claim.dissent.length) {
      frag.appendChild(previewRow("Deviate", claim.dissent.map(d => d.model).join(", ")));
      const quoted = claim.dissent.find(d => d.quote);
      if (quoted) {
        const q = document.createElement("blockquote");
        q.className = "insight-preview-quote";
        q.textContent = quoted.quote;
        frag.appendChild(q);
      }
    }
    if (claim.missing.length) {
      frag.appendChild(previewRow("Not addressed", claim.missing.join(", ")));
    }
    const foot = document.createElement("div");
    foot.className = "insight-preview-foot";
    foot.textContent = "Click for the full breakdown";
    frag.appendChild(foot);
    return frag;
  }

  /* The marker's preview is read out of the card it opens, so the summary can
     never say something the card below does not. */
  function buildDiffPreview(cardId) {
    const card = document.getElementById(cardId);
    if (!card) return null;
    const frag = document.createDocumentFragment();
    const sev = card.querySelector(".sev-dot");
    const sevClass = sev && sev.classList.contains("is-crit") ? "is-crit" : "is-warn";
    frag.appendChild(previewHead(
      (card.querySelector(".diff-type-tag") || {}).textContent || "Difference", sevClass));

    const claimText = card.querySelector(".diff-card-claim");
    if (claimText) {
      const el = document.createElement("div");
      el.className = "insight-preview-claim";
      el.textContent = claimText.textContent;
      frag.appendChild(el);
    }

    Array.from(card.querySelectorAll(".diff-position")).slice(0, 2).forEach(pos => {
      const models = pos.querySelector(".diff-position-label");
      const stance = pos.querySelector(".diff-position-stance");
      frag.appendChild(previewRow(
        models ? models.textContent : "",
        stance ? stance.textContent : ""));
    });

    const foot = document.createElement("div");
    foot.className = "insight-preview-foot";
    foot.textContent = "Click to open the difference";
    frag.appendChild(foot);
    return frag;
  }

  /* A badge and the highlighted sentence it sits behind light up together.
     The sentence is the one before the badge in the same block. */
  function linkedPassage(badge) {
    const previous = badge.previousElementSibling;
    return previous && previous.classList.contains("cx-claim") ? previous : null;
  }

  function wireBadge(badge) {
    if (!badge.dataset.claimAgree && !badge.dataset.claimDissent) return;
    const claim = readClaim(badge);
    const passage = linkedPassage(badge);

    badge.removeAttribute("tabindex");
    badge.setAttribute("aria-haspopup", "dialog");

    badge.addEventListener("click", function (event) {
      event.preventDefault();
      event.stopPropagation();
      openClaimPopover(claim, badge);
      if (window.umami && typeof window.umami.track === "function") {
        window.umami.track("landing_claim_opened");
      }
    });

    if (!canHover()) return;
    badge.addEventListener("mouseenter", function () {
      if (passage) passage.classList.add("is-hovered");
      schedulePreview(badge, () => buildClaimPreview(claim));
    });
    badge.addEventListener("mouseleave", function () {
      if (passage) passage.classList.remove("is-hovered");
      hidePreview();
    });

    // The other direction, so it stays obvious which ratio belongs to which
    // sentence when a sentence carries both a mark and a count.
    if (passage) {
      passage.addEventListener("mouseenter", () => badge.classList.add("is-linked-hover"));
      passage.addEventListener("mouseleave", () => badge.classList.remove("is-linked-hover"));
    }
  }

  function wireDiffTrigger(trigger) {
    if (!canHover()) return;
    const cardId = trigger.dataset.diffOpen;
    trigger.addEventListener("mouseenter", () =>
      schedulePreview(trigger, () => buildDiffPreview(cardId)));
    trigger.addEventListener("mouseleave", hidePreview);
  }

  function init() {
    document.querySelectorAll(".claim-badge").forEach(wireBadge);
    document.querySelectorAll("[data-diff-open]").forEach(wireDiffTrigger);
    // Clicking a marked passage means "show me the card", not "keep the
    // summary" — landing.html handles the opening, this only clears the hint.
    document.addEventListener("click", hidePreview, true);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
