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
    const key = [
      report.type,
      report.phase,
      report.resource_class,
      report.message,
      report.path
    ].join("|");
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
      path: window.location.pathname
    };
    const resourceClass = String(value.resource_class || "");
    const details = compactDetails(value.details);
    const stack = String(value.stack || value.error?.stack || value.reason?.stack || "");
    if (resourceClass) report.resource_class = resourceClass;
    if (details) report.details = details;
    if (stack) report.stack = stack;
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

  function criticalResourceClass(target) {
    const tag = String(target?.tagName || "").toUpperCase();
    const rel = String(target?.rel || "").toLowerCase().split(/\s+/);
    const explicitlyCritical = target?.dataset?.criticalResource === "true";
    const isRequiredScript = tag === "SCRIPT";
    const isRequiredStylesheet = tag === "LINK" && rel.includes("stylesheet");

    // Images, favicons, media and preloads are optional decoration unless a
    // future caller marks one explicitly. Their own components own fallbacks.
    if (!explicitlyCritical && !isRequiredScript && !isRequiredStylesheet) return "";

    const raw = target?.src || target?.href || "";
    if (!raw) return "unknown_resource";
    try {
      const url = new URL(raw, window.location.href);
      if (url.host === window.location.host) {
        if (url.pathname.startsWith("/static/dist/")) return "app_bundle";
        if (url.pathname.startsWith("/static/")) return "static_asset";
        return "same_origin_resource";
      }
      if (url.host === "cdn.jsdelivr.net") return "jsdelivr_dependency";
      if (url.host === "www.gstatic.com") return "firebase_dependency";
      return "";
    } catch (_) {
      return "unknown_resource";
    }
  }

  window.addEventListener("error", function (event) {
    if (event.target && event.target !== window) {
      const resourceClass = criticalResourceClass(event.target);
      if (!resourceClass) return;
      reportCriticalError({
        type: "resource_load_failed",
        phase: "asset_load",
        message: `Failed to load ${event.target.tagName || "resource"}`,
        resource_class: resourceClass
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
