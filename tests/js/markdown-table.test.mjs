import { beforeEach, describe, expect, it, vi } from "vitest";

import { loadScripts } from "./helpers/appWindow.mjs";

const TABLE_HTML = `
  <table>
    <thead><tr><th>Stage</th><th>Subject</th><th>Units</th></tr></thead>
    <tbody>
      <tr><td>1</td><td>Rendering fundamentals</td><td>8</td></tr>
      <tr><td>2</td><td>Information design</td><td>11</td></tr>
      <tr><td>Total</td><td></td><td>19</td></tr>
    </tbody>
  </table>`;

function boot() {
  return loadScripts(["static/js/markdown-stream.js"], {
    before(window) {
      window.marked = { parse: vi.fn(() => TABLE_HTML) };
      window.DOMPurify = { sanitize: vi.fn(value => value) };
    },
  });
}

describe("Markdown tables", () => {
  let window;
  let document;

  beforeEach(() => {
    ({ window, document } = boot());
  });

  it("keeps the semantic table inside an accessible scroll region", () => {
    const output = document.createElement("div");
    window.injectMarkdown(output, "| Stage | Subject | Units |");

    const wrapper = output.querySelector(".markdown-table-wrap");
    const table = wrapper?.querySelector("table.markdown-table");

    expect(wrapper).not.toBeNull();
    expect(wrapper.getAttribute("role")).toBe("region");
    expect(wrapper.getAttribute("aria-label")).toBe("Table: Stage, Subject, Units");
    expect(wrapper.tabIndex).toBe(0);
    expect(table?.querySelectorAll("thead th")).toHaveLength(3);
  });

  it("aligns predominantly numeric columns without changing text columns", () => {
    const output = document.createElement("div");
    window.injectMarkdown(output, "table");

    const headings = output.querySelectorAll("th");
    const firstRow = output.querySelector("tbody tr");
    const totalRow = output.querySelector("tbody tr:last-child");

    expect(headings[0].classList.contains("is-numeric")).toBe(true);
    expect(headings[1].classList.contains("is-numeric")).toBe(false);
    expect(headings[2].classList.contains("is-numeric")).toBe(true);
    expect(firstRow.cells[0].classList.contains("is-numeric")).toBe(true);
    expect(firstRow.cells[2].classList.contains("is-numeric")).toBe(true);
    expect(totalRow.cells[0].classList.contains("is-numeric")).toBe(false);
    expect(totalRow.cells[2].classList.contains("is-numeric")).toBe(true);
  });
});
