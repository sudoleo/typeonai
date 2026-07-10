// =====================================================================
// markdown-stream.js
// Markdown-Rendering (sanitised) + SSE-Streaming-Helfer.
// Extrahiert aus templates/index.html (initApp-Closure).
// Exporte: window.injectMarkdown, window.createStreamRenderer,
// window.streamSSERequest. (readSSEStream bleibt modul-privat.)
// Call-time-Abhaengigkeiten: DOMPurify, marked (CDN), window.addCopyButtons,
// window.addNewTabToLinks, window.linkifySourceTags, window.currentEvidenceSources.
// =====================================================================

// Utils: Markdown → HTML (sanitised) + deine Addons
function injectMarkdown(el, md) {
  el.innerHTML = DOMPurify.sanitize(marked.parse(md || ""));

  if (window.addCopyButtons) window.addCopyButtons(el);
  if (window.addNewTabToLinks) window.addNewTabToLinks(el);

  if (window.currentEvidenceSources && window.currentEvidenceSources.length && window.linkifySourceTags) {
    window.linkifySourceTags(el, window.currentEvidenceSources);
  }
}

window.injectMarkdown = injectMarkdown;

// === Streaming (SSE) Helpers ===
// Rendert eintreffende Text-Deltas gedrosselt als Markdown in ein Element.
function createStreamRenderer(outputEl, isActiveFn) {
  const RENDER_INTERVAL_MS = 120;
  let text = "";
  let renderTimer = null;
  let lastRenderAt = 0;
  let started = false;

  function render() {
    renderTimer = null;
    if (isActiveFn && !isActiveFn()) return;
    lastRenderAt = Date.now();
    outputEl.innerHTML = DOMPurify.sanitize(marked.parse(text || ""));
  }

  return {
    append(chunk) {
      if (!chunk) return;
      if (isActiveFn && !isActiveFn()) return;
      if (!started) {
        started = true;
        outputEl.classList.add("is-streaming");
      }
      text += chunk;
      const elapsed = Date.now() - lastRenderAt;
      if (elapsed >= RENDER_INTERVAL_MS) {
        if (renderTimer) clearTimeout(renderTimer);
        render();
      } else if (!renderTimer) {
        renderTimer = setTimeout(render, RENDER_INTERVAL_MS - elapsed);
      }
    },
    // Reasoning-Modelle: solange noch kein Antworttext eintrifft, den
    // "Typing"-Indikator auf "Reasoning" umstellen, damit sichtbar ist,
    // dass das Modell arbeitet (statt scheinbar zu haengen).
    markReasoning() {
      if (started) return;
      if (isActiveFn && !isActiveFn()) return;
      const label = outputEl.querySelector(".thinking.typing-indicator");
      if (!label || label.dataset.text === "Reasoning") return;
      label.dataset.text = "Reasoning";
      label.setAttribute("aria-label", "Reasoning");
      if (label.firstChild && label.firstChild.nodeType === Node.TEXT_NODE) {
        label.firstChild.nodeValue = "Reasoning";
      }
    },
    stop() {
      if (renderTimer) {
        clearTimeout(renderTimer);
        renderTimer = null;
      }
      outputEl.classList.remove("is-streaming");
    }
  };
}
window.createStreamRenderer = createStreamRenderer;

async function readSSEStream(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  function dispatch(rawEvent) {
    let eventName = "message";
    const dataLines = [];
    rawEvent.split("\n").forEach(line => {
      if (line.startsWith("event:")) {
        eventName = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).replace(/^\s/, ""));
      }
    });
    if (!dataLines.length) return;
    let parsed;
    try {
      parsed = JSON.parse(dataLines.join("\n"));
    } catch (_) {
      return;
    }
    onEvent(eventName, parsed);
  }

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let separatorIndex;
    while ((separatorIndex = buffer.indexOf("\n\n")) !== -1) {
      const rawEvent = buffer.slice(0, separatorIndex);
      buffer = buffer.slice(separatorIndex + 2);
      dispatch(rawEvent);
    }
  }
  if (buffer.trim()) dispatch(buffer);
}

// Führt einen POST-Request aus, der wahlweise als SSE-Stream (stream:true)
// oder als normales JSON beantwortet wird (z. B. Fehler/Limits vor Streamstart).
// deltaRenderers: { eventName: streamRenderer } für die Live-Anzeige.
// Rückgabe: { ok, status, data, streamed } – data hat dieselbe Struktur wie
// die bisherige JSON-Antwort (final-Event des Streams bzw. JSON-Body).
async function streamSSERequest(url, payload, signal, deltaRenderers) {
  const renderers = deltaRenderers || {};
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...payload, stream: true }),
      signal
    });

    const contentType = (response.headers.get("content-type") || "").toLowerCase();
    if (!contentType.includes("text/event-stream") || !response.body) {
      let data = {};
      try {
        data = await response.json();
      } catch (_) { /* keine JSON-Antwort -> leeres Objekt */ }
      return { ok: response.ok, status: response.status, data, streamed: false };
    }

    let finalData = null;
    await readSSEStream(response, (eventName, data) => {
      if (eventName === "final" || eventName === "error") {
        finalData = data;
        return;
      }
      if (eventName === "reasoning") {
        Object.values(renderers).forEach(renderer => renderer && renderer.markReasoning && renderer.markReasoning());
        return;
      }
      const renderer = renderers[eventName];
      if (!renderer || !data) return;
      if (data.text) {
        renderer.append(data.text);
      } else if (data.reasoning && renderer.markReasoning) {
        // Reasoning-Marker auf einem benannten Event (z. B. consensus.delta):
        // nur den zugehörigen Renderer markieren, nicht alle.
        renderer.markReasoning();
      }
    });

    if (!finalData) {
      finalData = { error: "Connection lost before the response was completed." };
    }
    return { ok: true, status: response.status, data: finalData, streamed: true };
  } finally {
    Object.values(renderers).forEach(renderer => renderer && renderer.stop());
  }
}
window.streamSSERequest = streamSSERequest;
