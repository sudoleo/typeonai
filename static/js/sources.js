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
      if (refs.length >= sourceGroupThreshold) {
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
  if (!Array.isArray(window.currentEvidenceSources)) {
    window.currentEvidenceSources = [];
  }
  const idMap = {};
  (incomingSources || []).forEach((src, idx) => {
    const localId = String(src.id || `S${idx + 1}`);
    const key = normalizeEvidenceUrl(src.url) || String(src.title || "").trim().toLowerCase();
    let existingIndex = window.currentEvidenceSources.findIndex(existing => {
      const existingKey = normalizeEvidenceUrl(existing.url) || String(existing.title || "").trim().toLowerCase();
      return existingKey && existingKey === key;
    });

    if (existingIndex === -1) {
      existingIndex = window.currentEvidenceSources.length;
      window.currentEvidenceSources.push({
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

  if (window.renderEvidenceSources) {
    window.renderEvidenceSources(window.currentEvidenceSources);
  }
  return idMap;
}

function rewriteSourceTags(markdown, idMap) {
  if (!markdown || !idMap || !Object.keys(idMap).length) return markdown;
  return markdown.replace(/\[((?:S?\d+)(?:,\s*S?\d+)*)\]/g, (match, inner) => {
    const mapped = inner.split(",").map(part => {
      const token = part.trim();
      const numeric = token.replace(/^S/i, "");
      return idMap[token] || idMap[numeric] || null;
    }).filter(Boolean);
    return mapped.length ? `[${mapped.join(", ")}]` : match;
  });
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

window.mergeEvidenceSources = mergeEvidenceSources;
window.rewriteSourceTags = rewriteSourceTags;
window.registerResponseSources = registerResponseSources;
window.prepareResponseSources = prepareResponseSources;
window.renderModelResponseWithSources = renderModelResponseWithSources;
