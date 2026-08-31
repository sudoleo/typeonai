import { describe, expect, it, vi } from "vitest";

import { loadScripts } from "./helpers/appWindow.mjs";

const BODY = `
  <div id="bookmarksContainer"></div>
  <button id="sendButton"></button>
  <span id="openaiModelText"></span>
  <div id="openaiResponse" class="response-box"><div class="collapsible-content"></div></div>
  <span id="mistralModelText"></span>
  <div id="mistralResponse" class="response-box"><div class="collapsible-content"></div></div>
  <span id="claudeModelText"></span>
  <div id="claudeResponse" class="response-box"><div class="collapsible-content"></div></div>
  <span id="geminiModelText"></span>
  <div id="geminiResponse" class="response-box"><div class="collapsible-content"></div></div>
  <span id="deepseekModelText"></span>
  <div id="deepseekResponse" class="response-box"><div class="collapsible-content"></div></div>
  <span id="grokModelText"></span>
  <div id="grokResponse" class="response-box"><div class="collapsible-content"></div></div>
  <section id="consensusOutput" class="is-hidden">
    <div id="consensusResponse">
      <div id="consensusAnswerBody"></div>
      <div class="consensus-differences"><p></p></div>
    </div>
  </section>
`;

function boot() {
  const user = { uid: "view-user", getIdToken: vi.fn(async () => "token") };
  const state = new Map();
  const harness = loadScripts(
    ["static/js/run-registry.js", "static/js/run-view.js"],
    {
      body: BODY,
      before(window) {
        window.auth = { currentUser: user };
        window.App = {
          // Familienliste der App (im Browser aus window.MODEL_FAMILIES,
          // serverseitig cfg.PROVIDERS). run-view projiziert genau diese.
          modelPrefs: [
            ["OpenAI", "openai"], ["Mistral", "mistral"], ["Anthropic", "claude"],
            ["Gemini", "gemini"], ["DeepSeek", "deepseek"], ["Grok", "grok"]
          ].map(([key, dom]) => ({
            key,
            responseId: `${dom}Response`,
            textId: `${dom}ModelText`
          })),
          authState: {
            generation: 3,
            snapshot: () => ({ uid: user.uid, generation: 3 })
          },
          state: { set: (key, value) => state.set(key, value) },
          // Im Browser kommt das aus app-state.js (head-Bundle); run-view
          // vergleicht damit die Stufe des Laufs gegen die auf dem Schirm.
          normalizeTier: (value) => {
            if (value === true) return "pro";
            if (value === false || value === null || value === undefined) return "free";
            const text = String(value).trim().toLowerCase();
            if (text === "pro" || text === "premium") return "pro";
            if (text === "plus") return "plus";
            return "free";
          },
          consensusBodyEl: root => root?.querySelector("#consensusAnswerBody"),
          setAppTitle: vi.fn(),
          setThreadQuestion: vi.fn(),
          setThreadQuestionAttachments: vi.fn(),
          bookmarkSession: { restore: vi.fn() },
          chatSession: { reset: vi.fn(), restoreCompletedChat: vi.fn() },
          followup: {
            renderStoredTurns: vi.fn(),
            reset: vi.fn(),
            offer: vi.fn()
          },
          consensusPipeline: {
            dismiss: vi.fn(),
            onPrepare: vi.fn(),
            onQueryStatus: vi.fn(),
            onConsensusStart: vi.fn(),
            onDifferencesStart: vi.fn(),
            onConsensusEnd: vi.fn(),
            renderProvenance: vi.fn()
          },
          differencesPanel: {
            setSynthesizing: vi.fn(),
            expandForFallback: vi.fn()
          },
          syncSendButtonRunning: vi.fn()
        };
        window.injectMarkdown = (element, markdown) => { element.textContent = markdown; };
        window.renderEvidenceSources = vi.fn();
        window.resetConsensusInsights = vi.fn();
        window.resetCredibilityFrame = vi.fn();
        window.updateAgentModeUI = vi.fn();
        window.syncHeroResponseAccess = vi.fn();
        window.exitHeroMode = vi.fn();
        window.enterDirectComparisonView = vi.fn();
        window.hideConsensusOutput = vi.fn();
        window.spinnerHTML = "loading";
        window.consensusSpinnerHTML = "consensus loading";
      }
    }
  );
  return { ...harness, registry: harness.window.App.runRegistry, state };
}

function createRunning(registry, question) {
  const context = registry.create({
    question,
    mode: "Standard",
    bookmarkId: `bookmark-${question}`,
    config: {
      agentMode: true,
      providers: [{ provider: "OpenAI", modelId: `model-${question}`, modelLabel: `Model ${question}` }]
    }
  });
  context.modelResults.OpenAI = {
    provider: "OpenAI",
    status: "complete",
    text: `answer ${question}`,
    streamText: `answer ${question}`,
    sources: []
  };
  context.progress.totalModels = 1;
  context.phase = "answers";
  registry.setStatus(context.runId, "running");
  return context;
}

describe("selected RunContext projection", () => {
  it("keeps late background updates out of the visible DOM and restores either run from its row", () => {
    const { registry, document, dom } = boot();
    const runA = createRunning(registry, "A");
    const runB = createRunning(registry, "B");
    const output = document.querySelector("#openaiResponse .collapsible-content");

    expect(registry.visible().runId).toBe(runB.runId);
    expect(output.textContent).toBe("answer B");

    registry.update(runA.runId, context => {
      context.modelResults.OpenAI.text = "late answer A";
      context.modelResults.OpenAI.streamText = "late answer A";
    });

    expect(output.textContent).toBe("answer B");
    expect(document.querySelectorAll(".bookmark.run-entry")).toHaveLength(2);
    expect(document.querySelector(`[data-run-id="${runA.runId}"]`)?.textContent).toContain("Models answering");

    document.querySelector(`[data-run-id="${runA.runId}"]`).click();
    expect(registry.visible().runId).toBe(runA.runId);
    expect(output.textContent).toBe("late answer A");

    registry.cancel(runB.runId, "user");
    expect(registry.visible().runId).toBe(runA.runId);
    expect(output.textContent).toBe("late answer A");
    expect(document.querySelector(`[data-run-id="${runB.runId}"]`)?.textContent).toContain("Canceled");

    document.querySelector(`[data-run-id="${runB.runId}"]`).click();
    expect(registry.visible().runId).toBe(runB.runId);
    expect(output.textContent).toBe("answer B");
    dom.window.close();
  });
});

describe("projection re-entrancy", () => {
  it("survives a surface that renders the visible run back at it", () => {
    const { registry, window, document, dom } = boot();
    let tierCalls = 0;
    // The real chain is updateUserTierUI -> restoreModelSelections ->
    // setModelSelectionState -> renderVisible. Any such feedback used to
    // recurse until "Maximum call stack size exceeded", which the run then
    // reported as a failure right after /prepare.
    window.updateUserTierUI = () => {
      tierCalls += 1;
      registry.renderVisible();
    };

    const run = createRunning(registry, "A");
    run.usage = { isProUser: true };
    expect(() => registry.update(run.runId, () => {})).not.toThrow();
    expect(tierCalls).toBe(1);
    expect(document.querySelector("#openaiResponse .collapsible-content").textContent)
      .toBe("answer A");

    // A tier the view already shows is not pushed again on every stream tick.
    // app-state.js exposes window.userTier as a read-only view, so mimic that
    // shape rather than assigning (a plain write throws in the real app).
    Object.defineProperty(window, "userTier", { get: () => "pro", configurable: true });
    registry.update(run.runId, () => {});
    expect(tierCalls).toBe(1);
    dom.window.close();
  });
});

describe("run-local evidence mapping", () => {
  it("rewrites source numbers against the supplied run without touching the visible global", () => {
    const stateSet = vi.fn();
    const renderEvidence = vi.fn();
    const { window, dom } = loadScripts(["static/js/sources.js"], {
      before(target) {
        target.App = { state: { set: stateSet } };
        target.currentEvidenceSources = [{ id: "S1", url: "https://visible.example/source" }];
        target.renderEvidenceSources = renderEvidence;
      }
    });

    const runAFirst = window.App.prepareResponseSourcesForEvidence(
      "A [S1]",
      [{ id: "S1", url: "https://a.example/one", title: "A one" }],
      []
    );
    const runBFirst = window.App.prepareResponseSourcesForEvidence(
      "B [S1]",
      [{ id: "S1", url: "https://b.example/one", title: "B one" }],
      []
    );
    const runASecond = window.App.prepareResponseSourcesForEvidence(
      "A again [S1]",
      [{ id: "S1", url: "https://a.example/two", title: "A two" }],
      runAFirst.evidenceSources
    );

    expect(runAFirst.markdown).toBe("A [1]");
    expect(runBFirst.markdown).toBe("B [1]");
    expect(runASecond.markdown).toBe("A again [2]");
    expect(runASecond.evidenceSources.map(source => source.url)).toEqual([
      "https://a.example/one",
      "https://a.example/two"
    ]);
    expect(runBFirst.evidenceSources.map(source => source.url)).toEqual([
      "https://b.example/one"
    ]);
    expect(window.currentEvidenceSources).toEqual([
      { id: "S1", url: "https://visible.example/source" }
    ]);
    expect(stateSet).not.toHaveBeenCalled();
    expect(renderEvidence).not.toHaveBeenCalled();
    dom.window.close();
  });

  it("moves a legacy leading source block behind the first model claim", () => {
    const { window, dom } = loadScripts(["static/js/sources.js"], {
      before(target) {
        target.App = { state: { set: vi.fn() } };
        target.currentEvidenceSources = [];
      }
    });

    expect(window.rewriteSourceTags(
      "[S1] [S2] Die erste Aussage ist belegt. Danach folgt mehr Text.",
      { S1: 3, 1: 3, S2: 4, 2: 4 }
    )).toBe("Die erste Aussage ist belegt. [3] [4] Danach folgt mehr Text.");

    expect(window.rewriteSourceTags(
      "Die Quellen stehen bereits richtig.[S1]",
      { S1: 1, 1: 1 }
    )).toBe("Die Quellen stehen bereits richtig.[1]");
    dom.window.close();
  });
});
