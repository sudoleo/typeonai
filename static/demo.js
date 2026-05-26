// static/demo.js

window.spinnerHTML = `
  <span class="thinking-wrap" role="status" aria-live="polite" aria-busy="true">
    <span class="thinking typing-indicator" data-text="Typing" aria-label="Typing">Typing<span class="typing-dots" aria-hidden="true"><span>.</span><span>.</span><span>.</span></span></span>
  </span>
`;
window.currentEvidenceSources = window.currentEvidenceSources || [];

/* === DEMO: Data & Utilities ======================================= */
const DEMO_SCENARIO_PROMPT =
  "Should vegetarians take supplements? If yes, which ones?";

const DEMO_SOURCE_LIBRARY = {
  b12: {
    id: "S1",
    title: "NIH ODS: Vitamin B12 Fact Sheet",
    url: "https://ods.od.nih.gov/factsheets/VitaminB12-Consumer/",
    provider: "NIH Office of Dietary Supplements"
  },
  vitaminD: {
    id: "S2",
    title: "NIH ODS: Vitamin D Fact Sheet",
    url: "https://ods.od.nih.gov/factsheets/VitaminD-HealthProfessional/",
    provider: "NIH Office of Dietary Supplements"
  },
  iodine: {
    id: "S3",
    title: "NIH ODS: Iodine Fact Sheet",
    url: "https://ods.od.nih.gov/factsheets/Iodine-Consumer/",
    provider: "NIH Office of Dietary Supplements"
  },
  omega3: {
    id: "S4",
    title: "NIH ODS: Omega-3 Fatty Acids Fact Sheet",
    url: "https://ods.od.nih.gov/factsheets/Omega3FattyAcids-HealthProfessional/",
    provider: "NIH Office of Dietary Supplements"
  },
  iron: {
    id: "S5",
    title: "NIH ODS: Iron Fact Sheet",
    url: "https://ods.od.nih.gov/factsheets/Iron-Consumer/",
    provider: "NIH Office of Dietary Supplements"
  },
  calcium: {
    id: "S6",
    title: "NIH ODS: Calcium Fact Sheet",
    url: "https://ods.od.nih.gov/factsheets/Calcium-Consumer/",
    provider: "NIH Office of Dietary Supplements"
  },
  zinc: {
    id: "S7",
    title: "NIH ODS: Zinc Fact Sheet",
    url: "https://ods.od.nih.gov/factsheets/Zinc-HealthProfessional/",
    provider: "NIH Office of Dietary Supplements"
  },
  creatine: {
    id: "S8",
    title: "ISSN position stand: creatine supplementation",
    url: "https://jissn.biomedcentral.com/articles/10.1186/s12970-017-0173-z",
    provider: "Journal of the International Society of Sports Nutrition"
  }
};

function demoSources(keys) {
  return keys.map(key => DEMO_SOURCE_LIBRARY[key]).filter(Boolean);
}

const DEMO_DATA = {
  delays: { OpenAI: 1400, Mistral: 2500, Anthropic: 2900, Gemini: 3600, DeepSeek: 4300, Grok: 5000 },
  sourceKeys: {
    OpenAI: ["b12", "vitaminD", "iodine", "omega3", "iron", "calcium", "creatine"],
    Mistral: ["b12", "vitaminD", "omega3", "iodine", "iron", "creatine", "calcium"],
    Anthropic: ["b12", "iodine", "vitaminD", "omega3", "iron", "zinc", "creatine"],
    Gemini: ["b12", "vitaminD", "creatine", "iron", "iodine", "calcium", "zinc"],
    DeepSeek: ["iron", "calcium", "b12", "vitaminD", "iodine", "omega3", "creatine"],
    Grok: ["b12", "vitaminD", "omega3", "iodine", "iron", "creatine"]
  },
  responses: {
    OpenAI:
`<div class="ai-block">
  <p>Quick overview</p>
  <p>For many vegetarians, a targeted supplement plan is more useful than a broad multivitamin. Vitamin B12 is the most consistent baseline because people eating little or no animal food can fall short [S1].</p>
  <h4>Likely baseline</h4>
  <ul>
    <li>Vitamin B12: use a regular supplement or reliably fortified foods; many people choose a daily low dose or a larger weekly dose [S1].</li>
  </ul>
  <h4>Often context-dependent</h4>
  <ul>
    <li>Vitamin D: consider it when sun exposure is low, especially in winter or mostly indoor routines [S2].</li>
    <li>Iodine: check whether iodized salt, dairy, eggs, seafood or seaweed are actually present in your diet [S3].</li>
    <li>Omega-3: algae-based EPA/DHA is the vegetarian route when fish is absent [S4].</li>
  </ul>
  <h4>Use labs or diet tracking</h4>
  <ul>
    <li>Iron: supplement when deficiency is shown or a clinician recommends it, not as a default [S5].</li>
    <li>Calcium: aim to meet the daily target from food first; fortified plant milks, tofu and dairy can change the answer [S6].</li>
    <li>Creatine: optional, mainly for strength or high-intensity training goals [S7].</li>
  </ul>
  <p><small>General information only; individual needs depend on labs, medical history, medication and pregnancy status.</small></p>
</div>`,

    Mistral:
`<div class="ai-block">
  <p>Decision path</p>
  <ol>
    <li>If you are vegetarian most days, make B12 the first check. Food sources can be inconsistent unless you regularly use fortified foods [S1].</li>
    <li>If you get little sun or live through long winters, vitamin D becomes a practical candidate; a 25-OH-D blood test can guide dosing [S2].</li>
    <li>If you never eat fish, algae oil is the direct EPA/DHA option; flax, chia and walnuts mainly provide ALA, not much preformed EPA/DHA [S3].</li>
    <li>If your salt is non-iodized and seaweed is rare, iodine may be a gap, especially around pregnancy planning [S4].</li>
    <li>If fatigue or heavy periods are part of the picture, check ferritin or an iron panel before taking iron [S5].</li>
    <li>If you train hard, creatine monohydrate at a small daily dose is a reasonable performance-oriented add-on [S6].</li>
    <li>If dairy and fortified plant drinks are low, calculate calcium intake before buying a pill [S7].</li>
  </ol>
  <p><small>Start with the smallest set that solves a real gap, then review symptoms and labs after several weeks.</small></p>
</div>`,

    Anthropic:
`<div class="ai-block">
  <p>Clinician-style framing</p>
  <p>A vegetarian diet can be nutritionally complete, but it moves several nutrients from automatic intake to active monitoring. The strongest routine recommendation is B12 because natural plant foods are not dependable B12 sources [S1].</p>
  <ul>
    <li>Core: B12 through supplement or fortified foods.</li>
    <li>Common gaps: iodine, vitamin D and long-chain omega-3, depending on salt choice, sun exposure and fish avoidance [S2] [S3] [S4].</li>
    <li>Lab-driven: iron status, ferritin and sometimes zinc should guide extra supplementation [S5] [S6].</li>
    <li>Optional performance layer: creatine may help people doing resistance training or repeated high-intensity work [S7].</li>
  </ul>
  <p>The practical workflow is diet review, targeted labs, one or two changes, then recheck rather than stacking many products at once.</p>
  <p><small>This is educational context and not a diagnosis or prescription.</small></p>
</div>`,

    Gemini:
`<div class="ai-block">
  <p>Pick the profile that fits best</p>
  <ul>
    <li>Busy office vegetarian: B12 as the anchor; vitamin D is worth checking when most daylight hours are indoors [S1] [S2].</li>
    <li>Strength or endurance athlete: keep the nutrition basics, then consider creatine; monitor iron if performance drops or recovery worsens [S3] [S4].</li>
    <li>Pregnancy planning or pregnant: discuss a prenatal approach with folate, B12 and iodine; iron should follow labs and clinician advice [S1] [S5].</li>
    <li>Dairy-light or dairy-free: count calcium from fortified drinks, tofu, dairy alternatives and greens before adding a supplement [S6].</li>
    <li>Grain-and-legume-heavy pattern: zinc absorption can be lower in high-phytate diets, so food planning matters [S7].</li>
  </ul>
  <p><small>Most people are a mix of profiles. The useful answer is the one that matches your diet and labs.</small></p>
</div>`,

    DeepSeek:
`<div class="ai-block">
  <p>Risk and safety checklist</p>
  <h4>Before buying</h4>
  <ul>
    <li>List medication, thyroid history and recent labs. Iron and calcium can interact with some medicines, so timing matters [S1] [S2].</li>
    <li>Write down a normal week of meals; this quickly reveals whether B12, D, iodine, omega-3 or calcium are actually low [S3] [S4] [S5] [S6].</li>
  </ul>
  <h4>Commonly sensible</h4>
  <ul>
    <li>B12 for most vegetarians, D when sun exposure is low, iodine when iodized salt and seaweed are absent, and algae EPA/DHA for fish-free diets [S3] [S4] [S5] [S6].</li>
    <li>Creatine is a specific performance tool, not a universal health requirement [S7].</li>
  </ul>
  <h4>Avoid</h4>
  <ul>
    <li>High-dose shotgun multis without a reason.</li>
    <li>Iron just in case; overdose risk and side effects make labs important [S1].</li>
    <li>Overlapping products that quietly add multiple doses of the same nutrient.</li>
  </ul>
  <p><small>Use supplements as precise tools: one gap, one intervention, one follow-up check.</small></p>
</div>`,

    Grok:
`<div class="ai-block">
  <p>Plain-language take</p>
  <ul>
    <li>B12: yes, treat it as the boring baseline. Vegetarian diets can miss it unless fortified foods are deliberate [S1].</li>
    <li>Vitamin D: if your lifestyle is mostly indoors or your winters are long, test or supplement thoughtfully [S2].</li>
    <li>Omega-3: if fish is off the menu, algae oil is the direct EPA/DHA route [S3].</li>
    <li>Iodine: trendy non-iodized salts do not help your thyroid; check your actual iodine sources [S4].</li>
    <li>Iron: blood work first. Guessing with iron is unnecessary risk [S5].</li>
    <li>Creatine: useful if you lift, sprint or train hard; not mandatory for everyone [S6].</li>
  </ul>
  <p><small>Internet guidance can help you ask better questions; your labs and clinician still settle the personal answer.</small></p>
</div>`
  },
  consensusSources: demoSources(["b12", "vitaminD", "iodine", "omega3", "iron", "calcium", "zinc", "creatine"]),
  consensus:
`<div class="ai-consensus">
  <p>Consensus: vegetarian supplement priorities</p>
  <p>The models converge on a targeted approach. The most consistent baseline is B12 because vegetarian and especially vegan patterns can provide too little without fortified foods or supplements [S1]. After that, the answer depends on sun exposure, iodine sources, fish avoidance, calcium intake and labs.</p>
  <ul>
    <li>B12: routine supplement or reliably fortified foods [S1].</li>
    <li>Vitamin D: most relevant with low sun exposure or low measured 25-OH-D [S2].</li>
    <li>Iodine: consider it when iodized salt, dairy, eggs, seafood and seaweed are not regular parts of the diet [S3].</li>
    <li>Omega-3: algae EPA/DHA is the direct fish-free option [S4].</li>
  </ul>
  <p>Other nutrients are more conditional. Iron should follow ferritin or an iron panel rather than habit [S5]. Calcium is best handled by counting food and fortified products before topping up [S6]. Zinc can matter in monotonous high-grain or legume-heavy diets [S7]. Creatine is optional and most relevant for training or performance goals [S8].</p>
  <p>Overall recommendation: choose the smallest targeted set, avoid overlapping high-dose products, and recheck labs after a defined interval. This remains general information, not personal medical advice.</p>
</div>`,

  differences:
`The consensus answer is largely credible.

Most models agree on the important hierarchy: B12 first; vitamin D, iodine and algae EPA/DHA when lifestyle or diet indicates; iron only with labs; calcium by intake calculation; creatine mainly for training goals. The answers differ in tone and workflow, but not in the main practical advice.

BestModel: Anthropic`
};

/* === DEMO: Timing & Typing Configuration =============================== */
const DEMO_PHASES = {
  preType: true,
  order: ["OpenAI", "Anthropic", "Gemini", "Mistral", "DeepSeek", "Grok"],
  typeChars: 90,
  typeSpeed: 40,
  gapBetweenModels: 540,
  pauseAfterTypingAll: 650
};

const DEMO_CONSENSUS_DELAY_MS = 4200;
const DEMO_CONSENSUS_JITTER_MS = 600;
const DEMO_DELAY_BOOST_MS = 1800;

Object.keys(DEMO_DATA.delays).forEach(key => {
  DEMO_DATA.delays[key] = (DEMO_DATA.delays[key] || 1500) + DEMO_DELAY_BOOST_MS;
});

const MODEL_TO_BOX = {
  OpenAI: "openaiResponse",
  Mistral: "mistralResponse",
  Anthropic: "claudeResponse",
  Gemini: "geminiResponse",
  DeepSeek: "deepseekResponse",
  Grok: "grokResponse"
};

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

function getDemoStorage() {
  try {
    return window.localStorage || null;
  } catch (e) {
    return null;
  }
}

function shouldAvoidDemoInputFocus() {
  return window.matchMedia?.("(hover: none) and (pointer: coarse)")?.matches ||
    window.matchMedia?.("(max-width: 768px)")?.matches;
}

async function typeIntoInput(inputEl, text, speed = 14, options = {}) {
  if (!inputEl) return;
  const allowFocus = options.allowFocus ?? !shouldAvoidDemoInputFocus();
  if (allowFocus) {
    inputEl.focus({ preventScroll: true });
  } else if (document.activeElement === inputEl) {
    inputEl.blur();
  }

  inputEl.value = "";
  inputEl.dispatchEvent(new Event("input", { bubbles: true }));
  for (let i = 0; i < text.length; i++) {
    inputEl.value += text[i];
    inputEl.dispatchEvent(new Event("input", { bubbles: true }));
    const jitter = Math.random() * 6 - 3;
    await sleep(Math.max(4, speed + jitter));
    if (typeof inputEl.scrollTop === "number") inputEl.scrollTop = inputEl.scrollHeight;
  }

  if (!allowFocus && document.activeElement === inputEl) {
    inputEl.blur();
  }
}

function getBox(model) {
  const id = MODEL_TO_BOX[model];
  const box = document.getElementById(id);
  if (!box || box.classList.contains("excluded") || box.style.display === "none") return null;
  return box;
}

function setSpinnerEl(box) {
  const p = box.querySelector(".collapsible-content");
  if (p) p.innerHTML = window.spinnerHTML;
}

window.setSpinnerEl = setSpinnerEl;

function getDemoSourcesForModel(model) {
  return demoSources(DEMO_DATA.sourceKeys[model] || []);
}

function renderDemoModelResponse(model, outputEl) {
  const markdown = DEMO_DATA.responses[model] || "";
  const sources = getDemoSourcesForModel(model);
  if (window.renderModelResponseWithSources) {
    window.renderModelResponseWithSources(outputEl, markdown, sources);
    return;
  }
  if (window.injectMarkdown) window.injectMarkdown(outputEl, markdown);
}

function renderDemoConsensus(mainP, diffP) {
  let consensusMarkdown = DEMO_DATA.consensus;
  if (window.registerResponseSources) {
    consensusMarkdown = window.registerResponseSources(consensusMarkdown, DEMO_DATA.consensusSources);
  } else if (window.mergeEvidenceSources) {
    window.mergeEvidenceSources(DEMO_DATA.consensusSources);
  }

  if (mainP && window.injectMarkdown) {
    window.injectMarkdown(mainP, consensusMarkdown);
  }
  if (diffP) {
    window.applyCredibilityFrame?.(diffP, DEMO_DATA.differences);
    const html = marked.parse(
      (window.colorizeCredibility?.(DEMO_DATA.differences) ?? DEMO_DATA.differences)
    );
    diffP.innerHTML = DOMPurify.sanitize(html);
  }
  const best = (DEMO_DATA.differences.match(/BestModel:\s*(.*)/i)?.[1] || "").trim();
  if (best) window.recordModelVote?.(best, "BestModel");
}

async function runDemoFlow() {
  const sendBtn = document.getElementById("sendButton");
  const consensusBtn = document.getElementById("consensusButton");
  if (sendBtn) sendBtn.disabled = true;
  if (consensusBtn) consensusBtn.disabled = true;
  window.setAgentModeStatus?.("running");

  window.currentEvidenceSources = [];
  window.renderEvidenceSources?.([]);

  const qi = document.getElementById("questionInput");
  if (qi && !qi.value.trim()) qi.value = DEMO_SCENARIO_PROMPT;

  if (DEMO_PHASES.preType) {
    const qiEl = document.getElementById("questionInput");
    const snippet =
      DEMO_SCENARIO_PROMPT.slice(0, DEMO_PHASES.typeChars) +
      (DEMO_SCENARIO_PROMPT.length > DEMO_PHASES.typeChars ? "..." : "");
    await typeIntoInput(qiEl, snippet, DEMO_PHASES.typeSpeed);
    await sleep(DEMO_PHASES.pauseAfterTypingAll);
  }

  window.setAgentModeStatus?.("running");
  Object.keys(MODEL_TO_BOX).forEach(key => {
    const box = getBox(key);
    if (box) setSpinnerEl(box);
  });

  await Promise.all(Object.keys(MODEL_TO_BOX).map(model =>
    new Promise(resolve => {
      setTimeout(() => {
        const box = getBox(model);
        if (box) {
          const p = box.querySelector(".collapsible-content");
          if (p) renderDemoModelResponse(model, p);
        }
        resolve();
      }, DEMO_DATA.delays[model] || 1800);
    })
  ));

  window.setAgentModeStatus?.("complete");

  const consensusDiv = document.getElementById("consensusResponse");
  const mainP = consensusDiv?.querySelector(".consensus-main p");
  const diffP = consensusDiv?.querySelector(".consensus-differences p");
  const auto = document.getElementById("autoConsensusToggle")?.checked;

  if (auto) {
    window.resetCredibilityFrame?.(consensusDiv?.querySelector(".consensus-differences"));
    if (mainP) mainP.innerHTML = window.spinnerHTML;
    if (diffP) diffP.innerHTML = window.spinnerHTML;
    setTimeout(
      () => renderDemoConsensus(mainP, diffP),
      DEMO_CONSENSUS_DELAY_MS + Math.floor(Math.random() * DEMO_CONSENSUS_JITTER_MS)
    );
  } else if (consensusBtn) {
    const originalOnclick = consensusBtn.onclick;
    consensusBtn.onclick = () => {
      if (mainP) mainP.innerHTML = window.spinnerHTML;
      if (diffP) diffP.innerHTML = window.spinnerHTML;
      setTimeout(
        () => renderDemoConsensus(mainP, diffP),
        DEMO_CONSENSUS_DELAY_MS + Math.floor(Math.random() * DEMO_CONSENSUS_JITTER_MS)
      );
      consensusBtn.onclick = originalOnclick;
    };
  }

  if (sendBtn) sendBtn.disabled = false;
  if (consensusBtn) consensusBtn.disabled = false;
}

function createStartDemoChip() {
  const storage = getDemoStorage();
  if (storage?.getItem("demoChipDismissed")) return;
  const container = document.querySelector(".chat-input-container");
  if (!container || container.querySelector(".demo-chip")) return;

  const btn = document.createElement("button");
  btn.className = "demo-chip";
  btn.type = "button";
  btn.setAttribute("aria-label", "Start interactive demo");
  btn.textContent = "Try Demo";

  btn.addEventListener("click", async () => {
    storage?.setItem("demoChipDismissed", "1");
    btn.remove();
    await runDemoFlow();
  });

  container.appendChild(btn);
}

window.runDemoFlow = runDemoFlow;
window.createStartDemoChip = createStartDemoChip;
createStartDemoChip();

function toggleSettingsCollapse(contentId, arrowId) {
  const content = document.getElementById(contentId);
  const arrow = document.getElementById(arrowId);

  if (content.style.display === "none") {
    content.style.display = "block";
    if (arrow) arrow.classList.add("rotated");
    if (arrow) arrow.innerHTML = "&#9650;";
  } else {
    content.style.display = "none";
    if (arrow) arrow.classList.remove("rotated");
    if (arrow) arrow.innerHTML = "&#9660;";
  }
}

window.toggleSettingsCollapse = toggleSettingsCollapse;
