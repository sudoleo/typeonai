import { describe, expect, it } from "vitest";

import { loadScripts } from "./helpers/appWindow.mjs";

function boot() {
  const reports = [];
  const loaded = loadScripts(["static/js/error-reporter.js"], {
    before(window) {
      window.fetch = (_url, options) => {
        reports.push(JSON.parse(options.body));
        return Promise.resolve({ ok: true });
      };
    },
  });
  return { ...loaded, reports };
}

function fail(window, element) {
  window.document.head.appendChild(element);
  element.dispatchEvent(new window.Event("error"));
}

describe("critical resource reporting", () => {
  it("ignores optional source favicons", () => {
    const { window, document, reports } = boot();
    const image = document.createElement("img");
    image.src = "/api/topics/favicon?d=example.com";

    fail(window, image);

    expect(reports).toEqual([]);
  });

  it("ignores document favicons", () => {
    const { window, document, reports } = boot();
    const link = document.createElement("link");
    link.rel = "icon";
    link.href = "/static/favicon.svg";

    fail(window, link);

    expect(reports).toEqual([]);
  });

  it("reports a failed app script without sending its URL", () => {
    const { window, document, reports } = boot();
    const script = document.createElement("script");
    script.src = "/static/dist/app.abc123.js";

    fail(window, script);

    expect(reports).toHaveLength(1);
    expect(reports[0]).toMatchObject({
      type: "resource_load_failed",
      phase: "asset_load",
      resource_class: "app_bundle",
      path: "/app",
    });
    expect(reports[0]).not.toHaveProperty("details");
  });

  it("classifies a failed CDN stylesheet", () => {
    const { window, document, reports } = boot();
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "https://cdn.jsdelivr.net/npm/katex/dist/katex.min.css";

    fail(window, link);

    expect(reports).toHaveLength(1);
    expect(reports[0].resource_class).toBe("jsdelivr_dependency");
  });
});
