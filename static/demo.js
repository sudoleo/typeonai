// static/demo.js

// Alles in DOMContentLoaded kapseln, damit das DOM fertig ist,
// wenn wir auf Elemente zugreifen
  // Globale Spinner-HTML, falls du sie nur für die Demo brauchst
  window.spinnerHTML = `
    <span class="thinking-wrap" role="status" aria-live="polite" aria-busy="true">
      <span class="spinner" aria-hidden="true"></span>
      <span class="thinking" data-text="Typing …">Typing&nbsp;…</span>
    </span>
  `;
  window.currentEvidenceSources = [];

  /* === DEMO: Data & Utilities ======================================= */
  const DEMO_SCENARIO_PROMPT =
    "Should vegetarians take supplements? If yes, which ones?";

      const DEMO_DATA = {
        // realistic, staggered loading
        delays: { OpenAI: 1400, Mistral: 2500, Anthropic: 2900, Gemini: 3600, DeepSeek: 4300, Grok: 5000 },

        // short but differentiated answers to create real signal for Consensus & Differences
        // short but differentiated answers to create real signal for Consensus & Differences
        responses: {
          OpenAI:
        `<div class="ai-block">
            <p><u>Quick overview: do vegetarians need supplements?</u></p>
            <p>Short answer: very often <strong>yes</strong> for B12, sometimes for D, iodine and omega-3, and the rest depends on your lab work and diet.</p>
            <h4>1. Almost always needed</h4>
            <ul>
                <li><strong>Vitamin B12:</strong> 50–100 µg/day or 1,000 µg once weekly (cyanocobalamin is fine for most). Re-check B12 or MMA after ~8–12 weeks.</li>
            </ul>
            <h4>2. Very commonly useful</h4>
            <ul>
                <li><strong>Vitamin D3:</strong> 1,000–2,000 IU/day if you live far from the equator, have darker skin, or are mostly indoors.</li>
                <li><strong>Iodine:</strong> ~150 µg/day if you rarely use iodized salt or seaweed (talk to your doctor if you have thyroid disease).</li>
                <li><strong>Omega-3 (EPA+DHA from algae):</strong> 250–500 mg/day if you don’t eat fish.</li>
            </ul>
            <h4>3. Sometimes helpful (case-by-case)</h4>
            <ul>
                <li><strong>Iron:</strong> only if blood tests show low ferritin/iron.</li>
                <li><strong>Calcium:</strong> aim for ~1,000 mg/day total (food + supplements).</li>
                <li><strong>Zinc &amp; Selenium:</strong> small daily doses can make sense if intake is uncertain and your diet is very grain/legume-heavy.</li>
                <li><strong>Creatine:</strong> 3–5 g/day for strength, performance or cognition – vegetarians often respond well.</li>
            </ul>
            <p><small>This is general information only and does not replace personal medical advice, lab testing or a consultation.</small></p>
            </div>
        `,

          Mistral:
        `<div class="ai-block">
            <p><u>Flowchart style: walk through your situation</u></p>
            <ol>
                <li><strong>Fully vegetarian or vegan &gt; 80% of the time?</strong><br>
                    → Yes: add <strong>B12</strong> (50–100 µg/day or 1,000 µg/week).<br>
                    → No: discuss B12 with a clinician, still often recommended.</li>
                <li><strong>Little sun (office job, winter, north of ~35° latitude)?</strong><br>
                    → Consider <strong>Vitamin D3</strong> 1,000–2,000 IU/day, check 25-OH-D after ~3 months.</li>
                <li><strong>No fish at all + rarely flax/chia/walnuts?</strong><br>
                    → Add <strong>algae EPA+DHA</strong> 250–500 mg/day.</li>
                <li><strong>Use non-iodized “gourmet” salts and little seaweed?</strong><br>
                    → Iodine ~150 µg/day may close a gap (especially relevant in pregnancy).</li>
                <li><strong>Heavy periods, fatigue, pale skin, shortness of breath?</strong><br>
                    → <em>Labs first.</em> Only supplement <strong>iron</strong> if a deficiency is confirmed.</li>
                <li><strong>Train hard (gym, sprinting, CrossFit)?</strong><br>
                    → <strong>Creatine monohydrate</strong> 3 g/day can support performance and strength.</li>
                <li><strong>Minimal dairy/fortified plant milks?</strong><br>
                    → Check if you reach ~1,000 mg/day of <strong>calcium</strong>; top up 300–500 mg if needed.</li>
            </ol>
            <p><small>General rule: start with 1–3 targeted supplements, track how you feel, and review with a professional after 8–12 weeks.</small></p>
            </div>
        `,

          Anthropic:
        `<div class="ai-block">
            <p><u>What a cautious clinician might consider</u></p>
            <p><strong>Baseline assumption:</strong> a vegetarian diet can be healthy, but it shifts several nutrients from “almost automatic” to “must be monitored”.</p>
            <ul>
                <li><strong>Non-negotiable for most:</strong> Vitamin B12 in supplement or reliably fortified foods.</li>
                <li><strong>Frequently indicated:</strong> Vitamin D3, iodine, and algae-derived omega-3, especially in low-sun regions and when fish and iodized salt are absent.</li>
                <li><strong>Lab-driven:</strong> Iron, ferritin, B12, folate, sometimes zinc and thyroid markers guide whether additional supplements are needed.</li>
                <li><strong>Performance/cognition:</strong> Creatine 3–5 g/day is often considered in vegetarians with high physical or cognitive demands.</li>
            </ul>
            <p>The clinical workflow is usually: assess diet → check relevant labs → introduce a small number of well-chosen supplements → re-evaluate symptoms and lab values after ~8–12 weeks.</p>
            <p><small>This is an educational summary of common medical practice patterns, not individual medical advice.</small></p>
            </div>
        `,

          Gemini:
        `<div class="ai-block">
            <p><u>Pick your profile: which one sounds like you?</u></p>
            <ul>
                <li><strong>1. Busy office vegetarian</strong><br>
                    • B12 routinely<br>
                    • D3 in winter or with low sun exposure<br>
                    • Optional algae EPA+DHA if fish is off the menu</li>
                <li><strong>2. Strength / endurance athlete</strong><br>
                    • B12 + D3 as needed<br>
                    • Creatine 3–5 g/day<br>
                    • Check iron and ferritin if fatigue or performance drops</li>
                <li><strong>3. Planning pregnancy / pregnant</strong><br>
                    • Prenatal with folate, B12 and iodine (often ~150 µg/day)<br>
                    • Iron according to lab results<br>
                    • Discuss omega-3 and vitamin D with your doctor</li>
                <li><strong>4. Dairy-light or dairy-free</strong><br>
                    • Focus on calcium from fortified plant milks, tofu, greens<br>
                    • Add 300–500 mg calcium if you consistently fall short</li>
                <li><strong>5. “Everything from grains and legumes” pattern</strong><br>
                    • Watch zinc and iron status<br>
                    • Consider small doses of zinc + selenium if diet is very monotonous</li>
            </ul>
            <p><small>Real life is messy – you may be a mix of several profiles. When in doubt, get a basic blood panel and tailor things with a professional.</small></p>
            </div>
        `,

          DeepSeek:
        `<div class="ai-block">
            <p><u>Risk &amp; safety oriented checklist</u></p>
            <h4>A. Before you buy anything</h4>
            <ul>
                <li>List your meds (e.g., levothyroxine, anticoagulants, metformin).</li>
                <li>Collect recent lab results (B12, ferritin, 25-OH-D, thyroid, etc.).</li>
                <li>Write down your typical week of food (2–3 days is better than guessing).</li>
            </ul>
            <h4>B. Smart additions for many vegetarians</h4>
            <ul>
                <li>B12 supplement as a default.</li>
                <li>D3 if sun exposure is low.</li>
                <li>Iodine if salt is not iodized and seaweed is rare.</li>
                <li>Algae EPA+DHA if you never eat fish.</li>
                <li>Creatine 3 g/day if you train seriously.</li>
            </ul>
            <h4>C. Things to actively avoid</h4>
            <ul>
                <li>High-dose “shotgun” multivitamins without a clear reason.</li>
                <li>Taking iron “just in case” without lab-confirmed deficiency.</li>
                <li>Stacking several products with overlapping ingredients (e.g., multiple D or A sources).</li>
                <li>Ignoring timing: keep iron and calcium away from levothyroxine by ≥4 hours.</li>
            </ul>
            <p><small>Think of supplements as tools, not a lifestyle. Use the smallest, safest set that actually solves a defined problem.</small></p>
            </div>
        `,

          Grok:
        `<div class="ai-block">
            <p><u>No-BS take on vegetarian supplements</u></p>
            <ul>
                <li><strong>B12:</strong> Yes. Just do it. Your brain and nerves like it. Food sources in vegetarian diets are unreliable.</li>
                <li><strong>Vitamin D:</strong> If you live in a place with winter and have a job that involves “indoors”, there’s a good chance you’re low.</li>
                <li><strong>Omega-3 (algae):</strong> Fish on pause? Algae oil is the vegetarian detour.</li>
                <li><strong>Iodine:</strong> Fancy pink salt looks nice on Instagram, not so great for your thyroid if it’s not iodized.</li>
                <li><strong>Iron:</strong> Only with blood work. Guessing with iron is a bad hobby.</li>
                <li><strong>Creatine:</strong> Take it if you lift, sprint or game hard. Otherwise, it’s not mandatory.</li>
            </ul>
            <p><small>Reminder: this is information, not medical orders. Internet advice is great for questions, not for diagnosing you.</small></p>
            </div>
        `
    },
        consensus:
        `<div class="ai-consensus">
          <p><u>Consensus (Demo — vegetarian focus)</u></p>
          <p>Vegetarians benefit from a <em>targeted</em> supplement approach rather than large stacks. The near-universal baseline is <span>B12 50–100 µg/day</span> or <span>1,000 µg weekly</span>. Add context-dependent items:</p>
        <ul>
            <li><strong>Vitamin B12</strong>: almost universally advised (50–100 µg/day or ~1,000 µg weekly).</li>
            <li><strong>Vitamin D3</strong>: 1,000–2,000 IU/day, especially with low sun exposure.</li>
            <li><strong>Iodine</strong>: ~150 µg/day if iodized salt or seaweed aren’t regular parts of the diet.</li>
            <li><strong>Omega-3 (algae EPA+DHA)</strong>: 250–500 mg/day for fully fish-free diets.</li>
        </ul>
        <p>Context-dependent items</p>
        <ul>
            <li><strong>Iron</strong>: only when confirmed by labs (ferritin/iron panel).</li>
            <li><strong>Calcium</strong>: aim for ~1,000 mg/day total; add 300–500 mg if intake is low.</li>
            <li><strong>Zinc / Selenium</strong>: mostly relevant for grain/legume-heavy patterns.</li>
            <li><strong>Creatine</strong>: 3–5 g/day for strength, endurance or cognitive performance goals.</li>
        </ul>
        <p>All models emphasize: pick high-quality, third-party tested products, 
        watch interactions (e.g., keeping levothyroxine away from iron/calcium), and 
        recheck labs after ~8–12 weeks. This is general information, not medical advice.</p>
    </div>`,

    differences:
      `<div class="ai-differences">
        <p><span class="cred-badge cred-largely">The consensus answer is largely credible.</span></p>
        <hr>
        <p>Most models agree on the core stack (B12 baseline; D3, iodine, algae EPA+DHA as context; iron only with labs; calcium to ~1,000 mg; creatine for athletes). Minor content nuances appear (e.g., zinc/copper balance, magnesium for sleep/performance, process/QA details) without contradictions.</p>
        <p><br>BestModel: Anthropic</p>
      </div>`
      };

      /* === DEMO: Timing & Typing Configuration =============================== */
      const DEMO_PHASES = {
        preType: true,
        order: ["OpenAI","Anthropic","Gemini","Mistral","DeepSeek","Grok"],
        typeChars: 90,         // how many prompt characters to "type"
        typeSpeed: 40,         // ms per character
        gapBetweenModels: 540, // pause between models
        pauseAfterTypingAll: 650
      };

      const DEMO_CONSENSUS_DELAY_MS = 4200;
      const DEMO_CONSENSUS_JITTER_MS = 600;

      // Ladezeiten etwas länger machen (ohne alles neu zu tippen)
      const DEMO_DELAY_BOOST_MS = 1800;
      Object.keys(DEMO_DATA.delays).forEach(k => {
        DEMO_DATA.delays[k] = (DEMO_DATA.delays[k] || 1500) + DEMO_DELAY_BOOST_MS;
      });

      const MODEL_TO_BOX = {
        OpenAI: "openaiResponse",
        Mistral: "mistralResponse",
        Anthropic: "claudeResponse",
        Gemini: "geminiResponse",
        DeepSeek: "deepseekResponse",
        Grok: "grokResponse"
      };

      const sleep = (ms)=> new Promise(r=>setTimeout(r, ms));

      // Tippt Text in das Input/Textarea-Feld
      async function typeIntoInput(inputEl, text, speed = 14) {
        if (!inputEl) return;
        inputEl.focus();
        inputEl.value = "";
        inputEl.dispatchEvent(new Event("input", { bubbles: true }));
        for (let i = 0; i < text.length; i++) {
          inputEl.value += text[i];
          inputEl.dispatchEvent(new Event("input", { bubbles: true }));
          const jitter = Math.random() * 6 - 3; // -3..+3ms
          await sleep(Math.max(4, speed + jitter));
          if (typeof inputEl.scrollTop === "number") inputEl.scrollTop = inputEl.scrollHeight;
        }
      }

      function getBox(model){
        const id = MODEL_TO_BOX[model];
        const box = document.getElementById(id);
        if (!box || box.classList.contains("excluded") || box.style.display === "none") return null;
        return box;
      }

      async function typeInto(el, text, speed){
        el.textContent = "";
        for (let i=0;i<text.length;i++){
          el.textContent += text[i];
          // kleine Randomisierung wirkt natürlicher
          const jitter = Math.random()*6 - 3; // -3..+3
          await sleep(Math.max(4, speed + jitter));
        }
      }

        function setSpinnerEl(box){
            const p = box.querySelector(".collapsible-content");
            if (p) p.innerHTML = window.spinnerHTML;
        }

        window.setSpinnerEl = setSpinnerEl;

      // Der eigentliche Demo-Lauf
      async function runDemoFlow() {
        // Buttons blocken
        const sendBtn = document.getElementById("sendButton");
        const consensusBtn = document.getElementById("consensusButton");
        if (sendBtn) sendBtn.disabled = true;
        if (consensusBtn) consensusBtn.disabled = true;

        // Prompt in das Eingabefeld setzen
        const qi = document.getElementById("questionInput");
        if (qi && !qi.value.trim()) qi.value = DEMO_SCENARIO_PROMPT;

        // === 1) Vorphase: so tun, als würden wir an Modelle „eingeben“ ==========
        // === 1) Vorphase: Prompt in das Input-Feld „tippen“ ======================
        if (DEMO_PHASES.preType) {
          const qiEl = document.getElementById("questionInput");
          const snippet =
            DEMO_SCENARIO_PROMPT.slice(0, DEMO_PHASES.typeChars) +
            (DEMO_SCENARIO_PROMPT.length > DEMO_PHASES.typeChars ? "…" : "");
          await typeIntoInput(qiEl, snippet, DEMO_PHASES.typeSpeed);
          await sleep(DEMO_PHASES.pauseAfterTypingAll);
        }

        // === 2) Jetzt erst Spinners in alle aktiven Boxen =======================
        Object.keys(MODEL_TO_BOX).forEach(key => {
          const box = getBox(key);
          if (box) setSpinnerEl(box);
        });

        // === 3) Gestaffelte Antworten „einlaufen“ lassen ========================
        await Promise.all(Object.keys(MODEL_TO_BOX).map(model =>
          new Promise(resolve => {
            setTimeout(() => {
              const box = getBox(model);
                if (box){
                const p = box.querySelector(".collapsible-content");
                if (p && window.injectMarkdown) {
                    window.injectMarkdown(p, DEMO_DATA.responses[model]);
                }
                }
              resolve();
            }, DEMO_DATA.delays[model] || 1800);
          })
        ));

        // === 4) Consensus/Differences – wie gehabt ==============================
        const consensusDiv = document.getElementById("consensusResponse");
        const mainP = consensusDiv?.querySelector(".consensus-main p");
        const diffP = consensusDiv?.querySelector(".consensus-differences p");
        const auto = document.getElementById("autoConsensusToggle")?.checked;

        const renderConsensus = () => {
        if (mainP && window.injectMarkdown) {
            window.injectMarkdown(mainP, DEMO_DATA.consensus);
        }
        if (diffP) {
            const html = marked.parse(
            (window.colorizeCredibility?.(DEMO_DATA.differences) ?? DEMO_DATA.differences)
            );
            diffP.innerHTML = DOMPurify.sanitize(html);
        }
        const best = (DEMO_DATA.differences.match(/BestModel:\s*(.*)/i)?.[1] || "").trim();
        if (best) window.recordModelVote?.(best, "BestModel");
        };

        if (auto){
          if (mainP) mainP.innerHTML = window.spinnerHTML;
          if (diffP) diffP.innerHTML = window.spinnerHTML;
          setTimeout(
            renderConsensus,
            DEMO_CONSENSUS_DELAY_MS + Math.floor(Math.random() * DEMO_CONSENSUS_JITTER_MS)
          );

        } else if (consensusBtn){
          const originalOnclick = consensusBtn.onclick;
          consensusBtn.onclick = () => {
            if (mainP) mainP.innerHTML = window.spinnerHTML;
            if (diffP) diffP.innerHTML = window.spinnerHTML;
            setTimeout(
              renderConsensus,
              DEMO_CONSENSUS_DELAY_MS + Math.floor(Math.random() * DEMO_CONSENSUS_JITTER_MS)
            );
            consensusBtn.onclick = originalOnclick;
          };
        }

        // Buttons wieder freigeben
        if (sendBtn) sendBtn.disabled = false;
        if (consensusBtn) consensusBtn.disabled = false;
      }

  function createStartDemoChip() {
    if (localStorage.getItem("demoChipDismissed")) return;
    const container = document.querySelector(".chat-input-container");
    if (!container || container.querySelector(".demo-chip")) return;

    const btn = document.createElement("button");
    btn.className = "demo-chip";
    btn.type = "button";
    btn.setAttribute("aria-label","Start interactive demo");
    btn.textContent = "Start Demo";

    btn.addEventListener("click", async () => {
      localStorage.setItem("demoChipDismissed","1");
      btn.remove();
      await runDemoFlow();
    });

    container.appendChild(btn);
  }
  window.createStartDemoChip = createStartDemoChip;
  createStartDemoChip();
  window.runDemoFlow = runDemoFlow;

function toggleSettingsCollapse(contentId, arrowId) {
  const content = document.getElementById(contentId);
  const arrow = document.getElementById(arrowId);
  
  if (content.style.display === "none") {
    // Einblenden
    content.style.display = "block";
    if(arrow) arrow.classList.add("rotated"); // Pfeil drehen (optional, siehe CSS)
    if(arrow) arrow.innerHTML = "&#9650;";    // Oder Pfeil-Zeichen ändern (hoch)
  } else {
    // Ausblenden
    content.style.display = "none";
    if(arrow) arrow.classList.remove("rotated");
    if(arrow) arrow.innerHTML = "&#9660;";    // Pfeil-Zeichen ändern (runter)
  }
}

window.toggleSettingsCollapse = toggleSettingsCollapse;
  