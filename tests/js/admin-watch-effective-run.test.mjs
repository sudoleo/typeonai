/**
 * A filter that fires at run time has to be visible where the choice is made.
 *
 * The Watch model grid used to show three configured providers while the runs
 * only ever used two: Publisher watches carried a hardcoded DeepSeek exclusion
 * that nothing in the UI mentioned. The exclusion is gone, but the class of bug
 * is not — a provider without a server credential still drops out silently. The
 * "Actually runs" row is what makes that visible, so it is pinned here against
 * the real production code from templates/admin.html and static/js/admin.js.
 */

import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";
import { JSDOM } from "jsdom";

import { ROOT } from "./helpers/appWindow.mjs";

/**
 * admin.js imports the Firebase SDK from a CDN and cannot be imported here.
 * The Watch model section is plain global-scope code and runs as a classic
 * script once the surrounding module boilerplate is dropped.
 */
function watchModelSource() {
    const js = readFileSync(path.join(ROOT, "static/js/admin.js"), "utf8");
    const start = js.indexOf("function meta() {");
    const end = js.indexOf("function currentPresetModels() {");
    if (start < 0 || end <= start) {
        throw new Error("Watch model section not found in static/js/admin.js");
    }
    return js.slice(start, end);
}

function boot(modelsData) {
    const dom = new JSDOM(
        '<!doctype html><html><body><div id="watchModelConfig"></div></body></html>',
        { runScripts: "dangerously", url: "https://consens.io/admin" }
    );
    const { window } = dom;
    const script = window.document.createElement("script");
    script.textContent = [
        "const providers = ['openai', 'mistral', 'anthropic', 'gemini', 'deepseek', 'grok'];",
        `let globalModelsData = ${JSON.stringify(modelsData)};`,
        "function markDirty() {}",
        "function consensusListValues() { return ['OpenAI']; }",
        "function isLockedConsensusModel() { return false; }",
        watchModelSource(),
        "renderWatchModelConfig();",
    ].join("\n");
    window.document.body.appendChild(script);
    return window;
}

function tierCells(window) {
    const cells = window.document.querySelectorAll(".watch-effective-run");
    return { free: cells[0], pro: cells[1] };
}

function chipTexts(cell) {
    return Array.from(cell.querySelectorAll(".admin-chip")).map(node => node.textContent);
}

const BASE = {
    openai: ["gpt-5.6-luna"],
    mistral: ["mistral-small-latest"],
    anthropic: [],
    gemini: ["gemini-3.7-flash"],
    deepseek: ["deepseek-v4-flash", "deepseek-v4-pro"],
    grok: [],
    premium: ["deepseek-v4-pro"],
    watch_consensus_models: { free: "OpenAI", pro: "OpenAI" },
};

describe("watch model configuration", () => {
    it("names every provider the next run will really use", () => {
        const window = boot({
            ...BASE,
            watch_models: {
                free: { openai: "gpt-5.6-luna", gemini: "gemini-3.7-flash", deepseek: "deepseek-v4-flash" },
                pro: { openai: "gpt-5.6-luna", gemini: "gemini-3.7-flash", deepseek: "deepseek-v4-flash" },
            },
            _meta: { provider_credentials: { openai: true, gemini: true, deepseek: true } },
        });

        const { free } = tierCells(window);
        expect(free.querySelector(".watch-effective-summary").textContent).toBe(
            "3 providers: openai, gemini, deepseek"
        );
        expect(chipTexts(free)).toEqual([]);
    });

    it("names the skipped provider when the server has no credential for it", () => {
        const window = boot({
            ...BASE,
            watch_models: {
                free: { openai: "gpt-5.6-luna", gemini: "gemini-3.7-flash", deepseek: "deepseek-v4-flash" },
                pro: { openai: "gpt-5.6-luna", gemini: "gemini-3.7-flash", deepseek: "deepseek-v4-flash" },
            },
            _meta: { provider_credentials: { openai: true, gemini: true, deepseek: false } },
        });

        const { free } = tierCells(window);
        expect(free.querySelector(".watch-effective-summary").textContent).toBe(
            "2 providers: openai, gemini"
        );
        expect(chipTexts(free)).toEqual(["deepseek skipped"]);
        expect(free.querySelector(".admin-chip").title).toContain("no server credential");
    });

    it("flags a Free tier pick that is locked to Pro instead of dropping it silently", () => {
        const window = boot({
            ...BASE,
            watch_models: {
                free: { openai: "gpt-5.6-luna", deepseek: "deepseek-v4-pro" },
                pro: { openai: "gpt-5.6-luna", deepseek: "deepseek-v4-pro" },
            },
            _meta: { provider_credentials: { openai: true, deepseek: true } },
        });

        const { free, pro } = tierCells(window);
        expect(chipTexts(free)).toEqual(["Needs at least 2", "deepseek skipped"]);
        expect(chipTexts(pro)).toEqual([]);
        expect(pro.querySelector(".watch-effective-summary").textContent).toBe(
            "2 providers: openai, deepseek"
        );
    });

    it("claims nothing while the server reports no credential state at all", () => {
        const window = boot({
            ...BASE,
            watch_models: {
                free: { openai: "gpt-5.6-luna", gemini: "gemini-3.7-flash" },
                pro: { openai: "gpt-5.6-luna", gemini: "gemini-3.7-flash" },
            },
            _meta: {},
        });

        expect(chipTexts(tierCells(window).free)).toEqual([]);
    });
});
