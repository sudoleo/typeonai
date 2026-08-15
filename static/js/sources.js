// =====================================================================
// sources.js
// Evidence-/Quellen-Tags: Chips, [S1]-Linkifizierung, Merge & Rewrite der
// Quell-IDs ueber alle Modellantworten hinweg.
// Extrahiert aus templates/index.html (initApp-Closure).
// Exporte: window.linkifySourceTags, window.mergeEvidenceSources,
// window.rewriteSourceTags, window.registerResponseSources,
// window.prepareResponseSources, window.renderModelResponseWithSources.
// Call-time-Abhaengigkeiten: window.injectMarkdown, window.renderEvidenceSources,
// window.currentEvidenceSources (geteilter State).
// =====================================================================

function getSourceSiteName(src) {
  const rawUrl = src && src.url ? String(src.url) : "";
  if (!rawUrl && !(src && src.title)) return "";
  try {
    const url = new URL(rawUrl);
    const hostParts = url.hostname
      .toLowerCase()
      .replace(/^(www|m|amp)\./, "")
      .split(".");
    const sldSuffixes = new Set(["co", "com", "org", "net", "ac", "gov"]);
    const nameIndex = hostParts.length >= 3 && sldSuffixes.has(hostParts[hostParts.length - 2])
      ? hostParts.length - 3
      : hostParts.length - 2;
    return hostParts[Math.max(0, nameIndex)] || rawUrl;
  } catch (e) {
    return (src && (src.title || src.url)) ? String(src.title || src.url) : "source";
  }
}

function getSafeSourceHref(src) {
  if (!src || !src.url) return "";
  try {
    const url = new URL(String(src.url));
    return ["http:", "https:"].includes(url.protocol) ? url.href : "";
  } catch (e) {
    return "";
  }
}

function getSourceTitle(src, fallbackLabel) {
  return (src && (src.title || src.url)) ? String(src.title || src.url) : fallbackLabel;
}

function getSourceHost(src) {
  if (!src || !src.url) return "";
  try {
    return new URL(String(src.url)).hostname.replace(/^www\./, "");
  } catch (e) {
    return "";
  }
}

// Mirror the public Topic/Share pages: prepend a real site favicon (fetched
// through our privacy-preserving proxy) so inline citation chips carry the
// source's identity, not just its name. The icon is decorative; if it fails to
// load we drop it and the chip falls back to text only.
function prependSourceFavicon(el, src) {
  const host = getSourceHost(src);
  if (!host) return;
  const fav = document.createElement("img");
  fav.className = "source-favicon";
  fav.src = "/api/topics/favicon?d=" + encodeURIComponent(host);
  fav.alt = "";
  fav.setAttribute("aria-hidden", "true");
  fav.setAttribute("referrerpolicy", "no-referrer");
  fav.loading = "lazy";
  fav.width = 13;
  fav.height = 13;
  fav.addEventListener("error", function () {
    fav.remove();
    el.classList.remove("has-favicon");
  });
  el.insertBefore(fav, el.firstChild);
  el.classList.add("has-favicon");
}

function createSourceChip(src, fallbackLabel) {
  const href = getSafeSourceHref(src);
  const el = href ? document.createElement("a") : document.createElement("span");
  el.className = "source-link";
  el.textContent = getSourceSiteName(src) || fallbackLabel;
  el.title = getSourceTitle(src, fallbackLabel);
  el.setAttribute("aria-label", `Source: ${el.title}`);

  if (href) {
    el.href = href;
    el.target = "_blank";
    el.rel = "noopener noreferrer";
  }

  prependSourceFavicon(el, src);

  return el;
}

function getSourceRefs(sourceText, sources) {
  const refs = [];
  const sourceTagRegex = /\[((?:S?\d+)(?:,\s*S?\d+)*)\]/g;

  String(sourceText || "").replace(sourceTagRegex, (match, innerContent) => {
    innerContent.split(",").forEach(part => {
      const token = part.trim();
      const idNum = parseInt(token.replace(/^S/i, ""), 10);
      refs.push({
        token,
        src: Number.isFinite(idNum) ? sources[idNum - 1] : null
      });
    });
    return match;
  });

  return refs;
}

function appendInlineSourceRefs(fragment, refs) {
  refs.forEach((ref, idx) => {
    if (idx > 0) fragment.appendChild(document.createTextNode(" "));
    fragment.appendChild(createSourceChip(ref.src, ref.token));
  });
}

// --- Numbered citations in the consensus answer ---------------------------
// The consensus is a page of prose, and a favicon chip in the middle of a
// sentence is a piece of furniture inside it. There it gets what printed
// prose uses: a raised number that points at the list underneath. The model
// answers keep the chips — they are scannable evidence, not a read.

function sourceNumberFromToken(token) {
  const num = parseInt(String(token || "").replace(/^S/i, ""), 10);
  return Number.isFinite(num) && num > 0 ? num : null;
}

function createSourceRef(ref) {
  const number = sourceNumberFromToken(ref.token);
  const href = getSafeSourceHref(ref.src);
  const el = href ? document.createElement("a") : document.createElement("span");
  el.className = "src-ref";
  el.textContent = number ? String(number) : ref.token;
  el.dataset.sourceNumber = number ? String(number) : "";
  // The teaser is the explanation; the native tooltip is the fallback for
  // keyboard and touch, where no hover exists.
  el.title = getSourceTitle(ref.src, ref.token);
  el.setAttribute("aria-label", `Source ${number || ref.token}: ${el.title}`);
  // Die Nummer allein ist keine Identitaet: ein archivierter Turn nummeriert
  // seine EIGENE Quellenliste, waehrend window.currentEvidenceSources schon
  // dem naechsten Lauf gehoert. Der Teaser liest deshalb die aufgeloeste
  // Quelle vom Element und nicht noch einmal die Nummer nach.
  el.sourceData = ref.src || null;

  if (href) {
    el.href = href;
    el.target = "_blank";
    el.rel = "noopener noreferrer";
  }

  return el;
}

function appendNumberedSourceRefs(fragment, refs) {
  const seen = new Set();
  let written = 0;
  refs.forEach(ref => {
    const number = sourceNumberFromToken(ref.token);
    const key = number || ref.token;
    if (seen.has(key)) return;
    seen.add(key);
    if (written > 0) {
      const sep = document.createElement("span");
      sep.className = "src-ref-sep";
      sep.textContent = ",";
      fragment.appendChild(sep);
    }
    fragment.appendChild(createSourceRef(ref));
    written += 1;
  });
}

// Die hochgestellte Ziffer gehoert dem Konsens-Fliesstext — und der hat mehr
// als eine Adresse: die ID gibt es nur einmal (der Live-Lauf), die Klasse
// tragen auch die archivierten Turns im Thread. Die Kernaussagen-Liste
// darunter zitiert woertlich denselben Text und muss deshalb dieselbe Form
// sprechen; ein Favicon-Chip mitten in einer Claim-Zeile waere ein zweites
// Vokabular fuer dieselbe Fussnote.
const NUMBERED_REF_SELECTOR =
  "#consensusAnswerBody, .consensus-answer-body, .consensus-claims-fallback";

function wantsNumberedRefs(containerEl) {
  if (!containerEl || typeof containerEl.closest !== "function") return false;
  return Boolean(
    containerEl.matches?.(NUMBERED_REF_SELECTOR)
    || containerEl.closest(NUMBERED_REF_SELECTOR)
    || containerEl.querySelector?.(NUMBERED_REF_SELECTOR)
  );
}

// Modelle setzen Quellen-Tags nicht immer typografisch korrekt: häufig kommt
// `[S1].`, obwohl eine Fussnote am Satzende hinter Punkt/Frage-/Ausrufezeichen
// steht. Normalisiert nur echte Satzendzeichen und lässt Code sowie
// satzinterne Referenzen unangetastet.
function normalizeTerminalSourceTagOrder(markdown) {
  const sourceRun = String.raw`\[((?:S?\d+)(?:,\s*S?\d+)*)\]`;
  const pattern = new RegExp(
    String.raw`[ \t]*(${sourceRun})([.!?]+(?:["'”’)\]}]+)?)(?=\s|$)`,
    "gi"
  );
  return String(markdown || "").replace(pattern, "$3$1");
}

function createSourceListCluster(refs) {
  const uniqueRefs = [];
  const seen = new Set();

  refs.forEach(ref => {
    const key = normalizeEvidenceUrl(ref.src && ref.src.url) || String(ref.src?.title || ref.token || "").trim().toLowerCase();
    if (key && seen.has(key)) return;
    if (key) seen.add(key);
    uniqueRefs.push(ref);
  });

  const details = document.createElement("details");
  details.className = "source-list-cluster";
  details.open = true;

  const summary = document.createElement("summary");
  summary.className = "source-list-summary";
  summary.textContent = `${uniqueRefs.length} sources`;
  details.appendChild(summary);

  const list = document.createElement("ol");
  list.className = "source-list";

  uniqueRefs.forEach(ref => {
    const item = document.createElement("li");
    item.className = "source-list-item";

    const href = getSafeSourceHref(ref.src);
    const title = getSourceTitle(ref.src, ref.token);
    const link = href ? document.createElement("a") : document.createElement("span");
    link.className = "source-list-link";
    link.textContent = title;
    link.title = title;

    if (href) {
      link.href = href;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
    }

    item.appendChild(link);

    const siteName = getSourceSiteName(ref.src);
    if (siteName && siteName !== title) {
      const meta = document.createElement("span");
      meta.className = "source-list-meta";
      meta.textContent = siteName;
      item.appendChild(meta);
    }

    list.appendChild(item);
  });

  details.appendChild(list);
  return details;
}

function linkifySourceTags(containerEl, sources) {
  if (!containerEl || !sources || !sources.length) return;

  const numbered = wantsNumberedRefs(containerEl);
  const ignoredParents = new Set(["A", "CODE", "PRE", "SCRIPT", "STYLE", "TEXTAREA"]);
  const sourceRunRegex = /(?:\[((?:S?\d+)(?:,\s*S?\d+)*)\](?:[\s,;:]*(?=\[S?\d))?)+/gi;
  const sourceGroupThreshold = 6;
  const walker = document.createTreeWalker(containerEl, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      if (!sourceRunRegex.test(node.nodeValue || "")) {
        sourceRunRegex.lastIndex = 0;
        return NodeFilter.FILTER_REJECT;
      }
      sourceRunRegex.lastIndex = 0;
      let parent = node.parentElement;
      while (parent && parent !== containerEl) {
        if (ignoredParents.has(parent.tagName)) return NodeFilter.FILTER_REJECT;
        parent = parent.parentElement;
      }
      return NodeFilter.FILTER_ACCEPT;
    }
  });

  const textNodes = [];
  while (walker.nextNode()) textNodes.push(walker.currentNode);

  textNodes.forEach(node => {
    const text = node.nodeValue || "";
    const fragment = document.createDocumentFragment();
    let lastIndex = 0;

    text.replace(sourceRunRegex, (match, innerContent, offset) => {
      if (offset > lastIndex) {
        fragment.appendChild(document.createTextNode(text.slice(lastIndex, offset)));
      }

      const refs = getSourceRefs(match, sources);
      if (numbered) {
        // Fallback fuer alte Bookmarks/Snapshots, deren Markdown noch
        // `Aussage [S1].` enthaelt: Satzzeichen im selben Textknoten vor die
        // hochgestellte Referenz ziehen und den Leerraum davor entfernen.
        const tail = text.slice(offset + match.length);
        const punctuation = tail.match(/^([.!?]+(?:["'”’)\]}]+)?)(?=\s|$)/);
        if (punctuation) {
          const previous = fragment.lastChild;
          if (previous?.nodeType === Node.TEXT_NODE) {
            previous.nodeValue = previous.nodeValue.replace(/[ \t]+$/, "");
          }
          fragment.appendChild(document.createTextNode(punctuation[1]));
        }
        // Numbers never need a cluster: twelve raised digits still read as
        // one citation, twelve chips are a paragraph of their own.
        appendNumberedSourceRefs(fragment, refs);
        if (punctuation) {
          lastIndex = offset + match.length + punctuation[1].length;
          return match;
        }
      } else if (refs.length >= sourceGroupThreshold) {
        fragment.appendChild(createSourceListCluster(refs));
      } else {
        appendInlineSourceRefs(fragment, refs);
      }

      lastIndex = offset + match.length;
      return match;
    });

    if (lastIndex < text.length) {
      fragment.appendChild(document.createTextNode(text.slice(lastIndex)));
    }

    node.parentNode.replaceChild(fragment, node);
  });
}
window.linkifySourceTags = linkifySourceTags;

function normalizeEvidenceUrl(url) {
  if (!url) return "";
  try {
    const u = new URL(url);
    u.hash = "";
    u.pathname = u.pathname.replace(/\/$/, "");
    return u.toString().toLowerCase();
  } catch (e) {
    return String(url).trim().toLowerCase();
  }
}

function mergeEvidenceSources(incomingSources) {
  const evidenceSources = Array.isArray(window.currentEvidenceSources)
    ? [...window.currentEvidenceSources]
    : [];
  const idMap = {};
  (incomingSources || []).forEach((src, idx) => {
    const localId = String(src.id || `S${idx + 1}`);
    const key = normalizeEvidenceUrl(src.url) || String(src.title || "").trim().toLowerCase();
    let existingIndex = evidenceSources.findIndex(existing => {
      const existingKey = normalizeEvidenceUrl(existing.url) || String(existing.title || "").trim().toLowerCase();
      return existingKey && existingKey === key;
    });

    if (existingIndex === -1) {
      existingIndex = evidenceSources.length;
      evidenceSources.push({
        ...src,
        id: `S${existingIndex + 1}`
      });
    }

    const globalNumber = existingIndex + 1;
    idMap[localId] = globalNumber;
    idMap[localId.replace(/^S/i, "")] = globalNumber;
    idMap[`S${idx + 1}`] = globalNumber;
    idMap[String(idx + 1)] = globalNumber;
  });

  window.App.state.set("currentEvidenceSources", evidenceSources, "evidence");
  if (window.renderEvidenceSources) {
    window.renderEvidenceSources(evidenceSources);
  }
  return idMap;
}

function rewriteSourceTags(markdown, idMap) {
  if (!markdown || !idMap || !Object.keys(idMap).length) {
    return normalizeTerminalSourceTagOrder(markdown);
  }
  const rewritten = markdown.replace(/\[((?:S?\d+)(?:,\s*S?\d+)*)\]/g, (match, inner) => {
    const mapped = inner.split(",").map(part => {
      const token = part.trim();
      const numeric = token.replace(/^S/i, "");
      return idMap[token] || idMap[numeric] || null;
    }).filter(Boolean);
    return mapped.length ? `[${mapped.join(", ")}]` : match;
  });
  return normalizeTerminalSourceTagOrder(rewritten);
}

function registerResponseSources(markdown, incomingSources) {
  const idMap = mergeEvidenceSources(incomingSources || []);
  return rewriteSourceTags(markdown || "", idMap);
}

function prepareResponseSources(markdown, incomingSources) {
  const sources = Array.isArray(incomingSources) ? incomingSources : [];
  const idMap = mergeEvidenceSources(sources);
  const mappedSources = [];
  const seen = new Set();

  sources.forEach((src, idx) => {
    if (!src || typeof src !== "object") return;
    const localId = String(src.id || `S${idx + 1}`);
    const numericId = localId.replace(/^S/i, "");
    const globalNumber =
      idMap[localId] ||
      idMap[numericId] ||
      idMap[`S${idx + 1}`] ||
      idMap[String(idx + 1)];
    const mapped = {
      id: globalNumber ? `S${globalNumber}` : localId,
      title: src.title || src.url || "",
      url: src.url || "",
      provider: src.provider || ""
    };
    const key = normalizeEvidenceUrl(mapped.url) || String(mapped.title || mapped.id || "").trim().toLowerCase();
    if (!key || seen.has(key)) return;
    seen.add(key);
    mappedSources.push(mapped);
  });

  return {
    markdown: rewriteSourceTags(markdown || "", idMap),
    sources: mappedSources
  };
}

function renderModelResponseWithSources(outputEl, markdown, incomingSources) {
  const prepared = prepareResponseSources(markdown, incomingSources || []);
  const box = outputEl?.closest?.(".response-box");
  if (box) {
    box.dataset.consensusAnswer = prepared.markdown || "";
    box.dataset.consensusSources = JSON.stringify(prepared.sources || []);
  }
  window.injectMarkdown(outputEl, prepared.markdown);
  return prepared.markdown;
}

// --- Teaser on hover ------------------------------------------------------
// A raised number says "there is a source", not "which one". Hovering it
// answers that without leaving the sentence — the same bargain the marked
// passages in the consensus already make: look closer, stay in place. Click
// still opens the source, and the title attribute covers keyboard and touch,
// where there is no hover to lean on.

const sourceTeaser = (function () {
  let el = null;
  let anchor = null;
  let hideTimer = null;

  function ensure() {
    if (el) return el;
    el = document.createElement("div");
    el.id = "sourceTeaser";
    el.className = "source-teaser";
    el.setAttribute("role", "tooltip");
    el.hidden = true;
    document.body.appendChild(el);
    return el;
  }

  function lookup(number) {
    const list = Array.isArray(window.currentEvidenceSources) ? window.currentEvidenceSources : [];
    return list[number - 1] || null;
  }

  function fill(node, src, number) {
    node.innerHTML = "";

    const head = document.createElement("div");
    head.className = "source-teaser-head";

    const index = document.createElement("span");
    index.className = "source-teaser-index";
    index.textContent = String(number);
    head.appendChild(index);

    const host = getSourceHost(src);
    if (host) {
      const fav = document.createElement("img");
      fav.className = "source-teaser-favicon";
      fav.src = "/api/topics/favicon?d=" + encodeURIComponent(host);
      fav.alt = "";
      fav.setAttribute("aria-hidden", "true");
      fav.setAttribute("referrerpolicy", "no-referrer");
      fav.width = 14;
      fav.height = 14;
      fav.addEventListener("error", () => fav.remove());
      head.appendChild(fav);

      const hostEl = document.createElement("span");
      hostEl.className = "source-teaser-host";
      hostEl.textContent = host;
      head.appendChild(hostEl);
    }

    node.appendChild(head);

    const title = document.createElement("div");
    title.className = "source-teaser-title";
    title.textContent = getSourceTitle(src, "Source " + number);
    node.appendChild(title);

    const snippet = src && (src.snippet || src.text);
    if (snippet) {
      const body = document.createElement("div");
      body.className = "source-teaser-snippet";
      body.textContent = String(snippet);
      node.appendChild(body);
    }
  }

  function place(node, target) {
    const rect = target.getBoundingClientRect();
    node.style.visibility = "hidden";
    node.hidden = false;
    const width = node.offsetWidth;
    const height = node.offsetHeight;
    const margin = 8;

    let left = rect.left + rect.width / 2 - width / 2;
    left = Math.max(margin, Math.min(left, window.innerWidth - width - margin));

    // Above the citation by default; below it when the top of the viewport
    // is closer than the popup is tall.
    let top = rect.top - height - 8;
    node.classList.toggle("is-below", top < margin);
    if (top < margin) top = rect.bottom + 8;

    node.style.left = Math.round(left + window.scrollX) + "px";
    node.style.top = Math.round(top + window.scrollY) + "px";
    node.style.visibility = "";
  }

  function show(target) {
    const number = parseInt(target.dataset.sourceNumber || "", 10);
    if (!Number.isFinite(number)) return;
    const src = target.sourceData || lookup(number);
    if (!src) return;

    window.clearTimeout(hideTimer);
    hideTimer = null;
    anchor = target;
    const node = ensure();
    fill(node, src, number);
    place(node, target);
    node.classList.add("is-visible");
  }

  function hide() {
    if (!el) return;
    anchor = null;
    el.classList.remove("is-visible");
    window.clearTimeout(hideTimer);
    hideTimer = window.setTimeout(() => {
      if (el && !el.classList.contains("is-visible")) el.hidden = true;
    }, 160);
  }

  document.addEventListener("pointerover", event => {
    const ref = event.target.closest?.(".src-ref");
    if (ref) {
      if (ref !== anchor) show(ref);
      return;
    }
    if (anchor && !event.target.closest?.("#sourceTeaser")) hide();
  });

  document.addEventListener("pointerdown", () => hide());
  window.addEventListener("scroll", () => { if (anchor) hide(); }, { passive: true });
  window.addEventListener("resize", () => { if (anchor) hide(); });

  return { hide };
})();

window.mergeEvidenceSources = mergeEvidenceSources;
window.rewriteSourceTags = rewriteSourceTags;
window.registerResponseSources = registerResponseSources;
window.prepareResponseSources = prepareResponseSources;
window.renderModelResponseWithSources = renderModelResponseWithSources;
window.hideSourceTeaser = sourceTeaser.hide;
