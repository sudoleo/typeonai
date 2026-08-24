/**
 * The SEO alert strip is the only thing standing between a dead pipeline and
 * five silent weeks.
 *
 * Configuration and the collection run used to live in two quiet summary cards.
 * When the Search Console credentials sat under the wrong variable name, every
 * weekly run aborted as collection_failed and nothing in the admin said so.
 * The redesign moved both cards behind a "Diagnostics & configuration" fold,
 * which is only defensible while the strip above really catches every failure
 * state. This drives the real production code from templates/admin.html and
 * static/js/admin.js against constructed data states and pins that.
 */

import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";
import { JSDOM } from "jsdom";

import { ROOT } from "./helpers/appWindow.mjs";

function seoTabMarkup() {
    const html = readFileSync(path.join(ROOT, "templates/admin.html"), "utf8");
    const start = html.indexOf('<section id="tab-seo"');
    const end = html.indexOf("</section>", start) + "</section>".length;
    if (start < 0 || end <= start) throw new Error("#tab-seo not found in templates/admin.html");
    // The panel ships hidden; the test needs it queryable, not visible.
    return html.slice(start, end).replace(" hidden>", ">");
}

/**
 * admin.js is an ES module that imports the Firebase SDK from a CDN, so it
 * cannot be imported here. The SEO half of it is plain global-scope code, so it
 * runs as a classic script once the module boilerplate around it is dropped.
 */
function seoModuleSource() {
    const js = readFileSync(path.join(ROOT, "static/js/admin.js"), "utf8");
    const start = js.indexOf("// === Read-only SEO data foundation ===");
    const end = js.indexOf("onAuthStateChanged(auth");
    if (start < 0 || end <= start) throw new Error("SEO section not found in static/js/admin.js");
    return js.slice(start, end);
}

function bootSeoTab() {
    const dom = new JSDOM(
        `<!doctype html><html><head></head><body>${seoTabMarkup()}</body></html>`,
        { runScripts: "dangerously", url: "https://consens.io/admin" }
    );
    const { window } = dom;
    window.confirm = () => false;
    const script = window.document.createElement("script");
    script.textContent = [
        "window.shareAdminRequest = async function () { return {}; };",
        "window.loadPublisherConfig = async function () {};",
        seoModuleSource(),
    ].join("\n");
    window.document.body.appendChild(script);
    return window;
}

function overview(extra = {}) {
    return {
        configuration: { configured: true, status: "configured", message: "" },
        last_run: {
            status: "success",
            started_at: "2026-08-20T09:00:00.000Z",
            metrics_written: 120,
            message: "",
        },
        captured_urls: 40,
        eligible_urls: 55,
        final_date: "2026-08-21",
        rows: [],
        status_rules: {},
        content_judge: { configured: false },
        weekly_review: {
            config: {},
            judge: {},
            publisher_watches: {},
            latest_review: {
                run_id: "run-1",
                status: "completed",
                judge_called: true,
                judge_error: "",
                groups: {},
                pages: [],
                collection: { status: "success", metrics_written: 120 },
            },
        },
        ...extra,
    };
}

function alertsOf(window) {
    return [...window.document.querySelectorAll("#seoAlerts .seo-alert")].map((node) => ({
        tone: node.className.replace("seo-alert is-", ""),
        text: node.textContent,
    }));
}

describe("admin SEO alert strip", () => {
    it("stays silent only when everything is verified healthy", () => {
        const window = bootSeoTab();
        window.renderSeoOverview(overview());
        // Nothing has been checked yet, so the strip still speaks up once.
        expect(alertsOf(window).map((item) => item.tone)).toEqual(["notice"]);
        expect(alertsOf(window)[0].text).toContain("Connection not verified");

        window.renderSeoConnection({ connected: true, status: "connected", message: "ok" });
        expect(alertsOf(window)).toEqual([]);
        expect(window.document.getElementById("seoAlerts").hidden).toBe(true);
    });

    it("catches credentials that are not configured", () => {
        const window = bootSeoTab();
        window.renderSeoOverview(overview({
            configuration: {
                configured: false,
                status: "not_configured",
                message: "GSC_SERVICE_ACCOUNT_FILE is not set.",
            },
        }));
        const alert = alertsOf(window).find((item) => item.text.includes("not configured"));
        expect(alert.tone).toBe("error");
        expect(alert.text).toContain("GSC_SERVICE_ACCOUNT_FILE is not set.");
    });

    it("catches a failed connection check and drops the unverified notice", () => {
        const window = bootSeoTab();
        window.renderSeoOverview(overview());
        window.renderSeoConnection({
            connected: false,
            status: "permission_denied",
            message: "The service account cannot read the property.",
        });
        const alerts = alertsOf(window);
        expect(alerts.some((item) => item.text.includes("Connection not verified"))).toBe(false);
        const failure = alerts.find((item) => item.text.includes("connection failed"));
        expect(failure.tone).toBe("error");
        expect(failure.text).toContain("The service account cannot read the property.");
    });

    it("catches every collection run that is neither success nor partial", () => {
        for (const status of ["not_configured", "permission_denied", "error", "transport_failed"]) {
            const window = bootSeoTab();
            window.renderSeoOverview(overview({
                last_run: {
                    status,
                    started_at: "2026-07-14T09:00:00.000Z",
                    message: "credentials missing",
                },
            }));
            const alert = alertsOf(window).find((item) => item.text.includes("did not succeed"));
            expect(alert, status).toBeTruthy();
            expect(alert.tone).toBe("error");
            expect(alert.text).toContain(status);
            expect(alert.text).toContain("credentials missing");
        }
    });

    it("keeps partial and success quiet, and calls a running collection out as a notice", () => {
        for (const status of ["success", "partial"]) {
            const window = bootSeoTab();
            window.renderSeoOverview(overview({
                last_run: { status, started_at: "2026-08-20T09:00:00.000Z" },
            }));
            expect(alertsOf(window).some((item) => item.text.includes("collection run")), status)
                .toBe(false);
        }
        const window = bootSeoTab();
        window.renderSeoOverview(overview({
            last_run: { status: "running", started_at: "2026-08-24T09:00:00.000Z" },
        }));
        const alert = alertsOf(window).find((item) => item.text.includes("still in progress"));
        expect(alert.tone).toBe("notice");
    });

    it("catches a portfolio that has never been collected at all", () => {
        const window = bootSeoTab();
        window.renderSeoOverview(overview({ last_run: {} }));
        const alert = alertsOf(window).find((item) => item.text.includes("has ever run"));
        expect(alert.tone).toBe("error");
    });

    it("catches a weekly review that ended as collection_failed or error", () => {
        for (const status of ["collection_failed", "error"]) {
            const window = bootSeoTab();
            const data = overview();
            data.weekly_review.latest_review = {
                ...data.weekly_review.latest_review,
                status,
                summary: "Search Console collection failed; the portfolio judge was not called.",
            };
            window.renderSeoOverview(data);
            const alert = alertsOf(window).find((item) => item.text.includes("weekly review ended"));
            expect(alert, status).toBeTruthy();
            expect(alert.tone).toBe("error");
            expect(alert.text).toContain("the portfolio judge was not called");
        }
    });

    it("catches a judge that did not answer", () => {
        const window = bootSeoTab();
        const data = overview();
        data.weekly_review.latest_review = {
            ...data.weekly_review.latest_review,
            judge_called: false,
            judge_error: "Judge request timed out.",
        };
        window.renderSeoOverview(data);
        const alert = alertsOf(window).find((item) => item.text.includes("portfolio judge did not answer"));
        expect(alert.tone).toBe("warning");
        expect(alert.text).toContain("Judge request timed out.");
    });

    it("catches a portfolio where no page has any stored data", () => {
        const window = bootSeoTab();
        window.renderSeoOverview(overview({ captured_urls: 0, eligible_urls: 55 }));
        const alert = alertsOf(window).find((item) => item.text.includes("stored Search Console data"));
        expect(alert.tone).toBe("error");
        // insufficient_data is a data gap, never a traffic statement.
        expect(alert.text).toContain("not because traffic is low");
    });

    it("reproduces the five silent weeks: several failures stack instead of hiding", () => {
        const window = bootSeoTab();
        const data = overview({
            configuration: { configured: false, status: "not_configured", message: "no credentials" },
            captured_urls: 0,
            eligible_urls: 55,
            last_run: { status: "not_configured", started_at: "2026-07-14T09:00:00.000Z" },
        });
        data.weekly_review.latest_review = {
            ...data.weekly_review.latest_review,
            status: "collection_failed",
            judge_called: false,
            judge_error: "",
        };
        window.renderSeoOverview(data);
        const alerts = alertsOf(window);
        expect(alerts.length).toBe(4);
        expect(alerts.every((item) => item.tone === "error")).toBe(true);
        expect(window.document.getElementById("seoAlerts").hidden).toBe(false);
    });
});
