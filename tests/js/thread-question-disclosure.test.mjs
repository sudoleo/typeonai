import { describe, expect, it } from "vitest";

import { loadScripts } from "./helpers/appWindow.mjs";

const BODY = `
<div id="threadAsk" class="thread-ask" hidden>
  <div class="thread-ask-text" id="threadAskText"></div>
  <div id="threadAskAttachments" hidden></div>
  <button type="button" class="thread-ask-more">Show full question</button>
</div>
<div id="threadPendingAsk" class="thread-ask" hidden>
  <div class="thread-ask-text" id="threadPendingAskText"></div>
  <div id="threadPendingAskAttachments" hidden></div>
  <button type="button" class="thread-ask-more">Show full question</button>
</div>
`;

function boot() {
  return loadScripts(["static/js/app-core.js"], {
    body: BODY,
    before: window => {
      window.matchMedia = () => ({
        matches: false,
        addEventListener() {},
        removeEventListener() {},
        addListener() {},
        removeListener() {}
      });
    }
  });
}

describe("thread question disclosure", () => {
  it("stays open when a running context projects the same question again", () => {
    const { window, document } = boot();
    const question = "A long question that is projected again while its answers stream.";

    window.App.setThreadQuestion(question);
    const wrap = document.getElementById("threadAsk");
    const more = wrap.querySelector(".thread-ask-more");
    wrap.classList.add("is-long");
    more.click();

    expect(wrap.classList.contains("is-open")).toBe(true);
    expect(more.textContent).toBe("Collapse question");
    expect(more.getAttribute("aria-expanded")).toBe("true");

    // run-view.js does this repeatedly for the visible run while provider
    // deltas arrive. It must not be interpreted as a new question.
    window.App.setThreadQuestion(question);

    expect(wrap.classList.contains("is-open")).toBe(true);
    expect(wrap.classList.contains("is-long")).toBe(true);
    expect(more.textContent).toBe("Collapse question");
    expect(more.getAttribute("aria-expanded")).toBe("true");
  });

  it("resets the disclosure when the projected question actually changes", () => {
    const { window, document } = boot();
    const wrap = document.getElementById("threadAsk");
    const more = wrap.querySelector(".thread-ask-more");

    window.App.setThreadQuestion("First long question");
    wrap.classList.add("is-long");
    more.click();
    window.App.setThreadQuestion("Second long question");

    expect(wrap.classList.contains("is-open")).toBe(false);
    expect(wrap.classList.contains("is-long")).toBe(false);
    expect(more.textContent).toBe("Show full question");
    expect(more.getAttribute("aria-expanded")).toBe("false");
  });
});
