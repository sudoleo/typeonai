/**
 * Load classic scripts into a throwaway jsdom, the way the browser does.
 *
 * These modules are not ES modules: they run in the global scope and hand each
 * other work through window.* contracts. So they cannot be `import`ed -- they
 * have to be executed as <script> content against a real document, which is
 * exactly what this does. Every call gets a fresh window, so a module that
 * installs listeners or freezes globals cannot leak into the next test.
 */

import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { JSDOM } from "jsdom";

const ROOT = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "..",
  "..",
  ".."
);

const BLANK = "<!doctype html><html><head></head><body></body></html>";

/**
 * @param {string[]} files      repo-relative paths, in load order
 * @param {object}   [options]
 * @param {string}   [options.body]    markup the scripts expect to find
 * @param {Function} [options.before]  runs against window before the scripts
 * @returns {{ window: Window, document: Document, dom: JSDOM }}
 */
export function loadScripts(files, { body, before } = {}) {
  const html = body
    ? `<!doctype html><html><head></head><body>${body}</body></html>`
    : BLANK;

  const dom = new JSDOM(html, {
    runScripts: "dangerously",
    url: "https://consens.io/app",
    pretendToBeVisual: true,
  });

  const { window } = dom;
  // jsdom has no rAF under some configurations, and several modules schedule
  // their DOM work through it. Run it synchronously so tests stay deterministic.
  window.requestAnimationFrame = (fn) => {
    fn(0);
    return 0;
  };
  window.cancelAnimationFrame = () => {};

  before?.(window);

  for (const relative of files) {
    const code = readFileSync(path.join(ROOT, relative), "utf8");
    const element = window.document.createElement("script");
    element.textContent = code;
    window.document.head.appendChild(element);
  }

  return { window, document: window.document, dom };
}

export { ROOT };
