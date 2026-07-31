// Critical browser failures only. Expected API errors and deliberate
// AbortController cancellations are reported by neither this module nor the
// global console; run modules opt in when a complete user flow has failed.
(function () {
  "use strict";

  window.App = window.App || {};

  const recentReports = new Map();
  const DEDUPE_MS = 5 * 60 * 1000;
  const MAX_RECENT = 40;

  function isExpectedAbort(value) {
    if (!value) return false;
    if (value.name === "AbortError") return true;
    return value instanceof DOMException && value.name === "AbortError";
  }

  function asText(value, fallback) {
    if (typeof value === "string") return value;
    if (value && typeof value.message === "string") return value.message;
    try {
      return JSON.stringify(value);
    } catch (_) {
      return fallback;
    }
  }

  function compactDetails(value) {
    if (!value) return "";
    if (typeof value === "string") return value;
    try {
      return JSON.stringify(value);
    } catch (_) {
      return String(value);
    }
  }

  function shouldSend(report) {
    const now = Date.now();
    const key = [report.type, report.phase, report.message, report.path].join("|");
    const previous = recentReports.get(key) || 0;
    if (previous && now - previous < DEDUPE_MS) return false;
    recentReports.set(key, now);
    if (recentReports.size > MAX_RECENT) {
      for (const [storedKey, storedAt] of recentReports) {
        if (now - storedAt >= DEDUPE_MS || recentReports.size > MAX_RECENT) {
          recentReports.delete(storedKey);
        }
      }
    }
    return true;
  }

  function reportCriticalError(input) {
    const value = input || {};
    if (isExpectedAbort(value.error || value.reason)) return;
    const report = {
      type: String(value.type || "unhandled_error"),
      phase: String(value.phase || "browser"),
      message: asText(value.message || value.error || value.reason, "Unknown browser error"),
      details: compactDetails(value.details),
      stack: String(value.stack || value.error?.stack || value.reason?.stack || ""),
      path: window.location.pathname
    };
    if (!shouldSend(report)) return;
    fetch("/api/client-errors", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(report),
      credentials: "same-origin",
      keepalive: true
    }).catch(function () {
      // Reporting must never create another unhandled rejection.
    });
  }

  function safeResourcePath(target) {
    const raw = target?.src || target?.href || "";
    if (!raw) return "unknown resource";
    try {
      const url = new URL(raw, window.location.href);
      if (![window.location.host, "cdn.jsdelivr.net", "www.gstatic.com"].includes(url.host)) {
        return "ignored external resource";
      }
      return `${url.origin}${url.pathname}`;
    } catch (_) {
      return "unknown resource";
    }
  }

  window.addEventListener("error", function (event) {
    if (event.target && event.target !== window) {
      const resource = safeResourcePath(event.target);
      if (resource === "ignored external resource") return;
      reportCriticalError({
        type: "resource_load_failed",
        phase: "asset_load",
        message: `Failed to load ${event.target.tagName || "resource"}`,
        details: resource
      });
      return;
    }
    reportCriticalError({
      type: "unhandled_error",
      phase: "browser_runtime",
      message: event.message || "Unhandled browser error",
      error: event.error,
      stack: event.error?.stack || "",
      details: event.filename
        ? `${event.filename.split("?")[0]}:${event.lineno || 0}:${event.colno || 0}`
        : ""
    });
  }, true);

  window.addEventListener("unhandledrejection", function (event) {
    if (isExpectedAbort(event.reason)) return;
    reportCriticalError({
      type: "unhandled_rejection",
      phase: "browser_promise",
      message: asText(event.reason, "Unhandled promise rejection"),
      reason: event.reason,
      stack: event.reason?.stack || ""
    });
  });

  window.App.reportCriticalError = reportCriticalError;
})();
