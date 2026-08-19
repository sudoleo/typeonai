// static/demo.js

window.App.state.set("spinnerHTML", `
  <span class="thinking-wrap" role="status" aria-live="polite" aria-busy="true">
    <span class="thinking typing-indicator" data-text="Typing" aria-label="Typing">Typing<span class="typing-dots" aria-hidden="true"><span>.</span><span>.</span><span>.</span></span></span>
  </span>
`, "runUi");
if (!Array.isArray(window.currentEvidenceSources)) {
  window.App.state.set("currentEvidenceSources", [], "evidence");
}

/* === DEMO: Data & Utilities =======================================
   Das Szenario ist bewusst kein Faktencheck: Wer eine heikle Nachricht
   vor dem Absenden prueft, holt sich im echten Leben zwei, drei Meinungen
   ein — und bekommt sie auch. Genau dort ist Uneinigkeit die Nachricht
   und nicht der Fehler. Deshalb hat der Lauf drei strittige Stellen
   (ein kritischer Widerspruch, ein kleiner, eine andere Gewichtung)
   statt einer einzigen Randnotiz, und der Score liegt entsprechend im
   mittleren Bereich. Quellenlisten gibt es hier bewusst nicht: Auf
   "kann ich das so schreiben?" zitiert kein Modell eine Studie, und
   erfundene Belege waeren die schlechtere Demo.
   ================================================================= */
const DEMO_SCENARIO_PROMPT =
  "I have to tell a client that our launch slips by two weeks. Can I send this as it is?\n\n" +
  "“Hi Anna, quick heads-up: we won’t make the 15th. We’re aiming for the 29th now. " +
  "A few things came up on our side, sorry about that. Let me know if that’s a problem.”";

// Der Entwurf selbst wird nicht getippt, sondern eingefuegt — so macht es
// jeder, der eine fertige Nachricht pruefen laesst.
const DEMO_TYPED_QUESTION = DEMO_SCENARIO_PROMPT.split("\n")[0];

const DEMO_DATA = {
  delays: { OpenAI: 1400, Mistral: 2500, Anthropic: 2900, Gemini: 3600, DeepSeek: 4300, Grok: 5000 },
  responses: {
    OpenAI:
`<div class="ai-block">
  <p>Short version: send it — but not in this order, and two of the four sentences are doing work you did not intend.</p>
  <h4>What already works</h4>
  <ul>
    <li>You are telling her before the 15th, not on it. That is the part most people get wrong.</li>
    <li>The tone is fine. Nothing here is rude, and nothing needs softening.</li>
  </ul>
  <h4>What I would change</h4>
  <ul>
    <li>Lead with the new date. Right now the 29th arrives in the second sentence, after the bad news; put it first so she can act on it immediately.</li>
    <li>Replace “a few things came up on our side” with the actual cause in one clause. Vagueness is the thing she will ask about.</li>
    <li>Say what she gets on the 15th. Two silent weeks are harder to plan around than the delay itself.</li>
  </ul>
  <h4>What I would keep, against the usual advice</h4>
  <p>Keep the closing question. It is the only line in the draft that gives Anna a way in, and a delay announced without one reads like a decision taken for her.</p>
  <p>One thing I would not add: a fixed check-in date. A promised status note is a second date you now also have to hit, and you are already explaining a missed one.</p>
</div>`,

    Mistral:
`<div class="ai-block">
  <p>Work through it line by line.</p>
  <ol>
    <li>First line: the new date. “The launch moves to the 29th” is the sentence she needs; everything else is context for it.</li>
    <li>Second line: the reason, kept to a clause. A detailed reason moves the conversation to your process, which is the one place you do not want it.</li>
    <li>Third line: what she has on the 15th anyway. A delay with something in it is a status update; a delay with nothing in it is a gap she has to explain to someone else.</li>
    <li>Fourth line: the apology — drop it. It asks her to absolve you, while a concrete plan tells her what you are doing about it. Only one of those helps her.</li>
    <li>Last line: cut “let me know if that’s a problem”. You are not offering her a choice, so do not phrase it as one; she will take you up on it and you will have to say no twice.</li>
  </ol>
  <p>Four short lines are enough. If you cannot get it under six, the message has turned into a defence.</p>
</div>`,

    Anthropic:
`<div class="ai-block">
  <p>What the message is actually asking her to do</p>
  <p>The draft asks Anna for two things: to absorb a two-week slip, and to tell you whether that is acceptable. The second request is the problem. You are asking her to rule on something you cannot change, and the honest answer to that question is yes, it is a problem — which leaves you both in a conversation with no move left in it.</p>
  <ul>
    <li>Open with the new date, then the cause, then what she has in the meantime.</li>
    <li>Name the cause. “A few things” is the only sentence in the draft she cannot check, and it will be the one she remembers.</li>
    <li>Replace the apology with the remedy. An apology asks something of her; a plan gives her something.</li>
    <li>Close with a commitment instead of a question: the day you will confirm the 29th.</li>
  </ul>
  <p>Same facts, different posture. The draft reports a slip and asks for permission; the version above reports a slip and takes responsibility for the next step.</p>
</div>`,

    Gemini:
`<div class="ai-block">
  <p>How it reads on her side, sentence by sentence</p>
  <ul>
    <li><b>“quick heads-up”</b> — signals something small. A two-week slip is not small, and the mismatch is the first thing she notices.</li>
    <li><b>“we won’t make the 15th”</b> — the loss arrives before the fix. Reverse it: the 29th first, the miss second.</li>
    <li><b>“a few things came up on our side”</b> — fine as it stands. She is not going to audit your sprint; she needs to know this is not the start of a pattern.</li>
    <li><b>“sorry about that”</b> — keep it exactly as it is. One apology, early and unqualified, is what makes the rest of the message read as news rather than as a defence.</li>
    <li><b>“let me know if that’s a problem”</b> — keep this one too, but only if you can live with the answer. It is the line that turns an announcement into a conversation.</li>
  </ul>
  <p>What is missing entirely: what she can show her own stakeholders on the 15th. That is the question she will be asked within an hour of forwarding your message.</p>
</div>`,

    DeepSeek:
`<div class="ai-block">
  <p>Before you send</p>
  <h4>Check</h4>
  <ul>
    <li>Is the 29th a date you would bet on? If it is a hope, you will be writing this message twice.</li>
    <li>Does anything of hers hang on the 15th — a campaign, a client of her own, a contract date? If so, the message needs a line about that, not about you.</li>
  </ul>
  <h4>Fix</h4>
  <ul>
    <li>New date in the first line.</li>
    <li>Name the cause in one clause. A vague cause reads as a cause you do not want to name, and that is the line she will come back to.</li>
    <li>Cut the closing question. It invites an escalation you cannot answer and implies the 29th is negotiable when it is not.</li>
    <li>Say what lands on the 15th, and name the day you will confirm the 29th.</li>
  </ul>
  <h4>Do not</h4>
  <ul>
    <li>Do not apologise twice. One acknowledgment, then move to the plan.</li>
    <li>Do not leave it in a chat window if the 15th sits in a contract. The same three lines in an email are the ones that count later.</li>
  </ul>
</div>`,

    Grok:
`<div class="ai-block">
  <p>Plain take: it is fine, and it is also three small edits away from being good.</p>
  <ul>
    <li>Put the 29th first. She is scanning for a date, not for context.</li>
    <li>Leave the reason vague. “A few things came up” is what everyone writes, and nobody has ever won a client back with a root-cause analysis.</li>
    <li>Keep the “sorry”. It costs you nothing and it is the only human line in there.</li>
    <li>Keep the closing question. If the 29th genuinely wrecks something on her side, you want to hear that now and not on the 28th.</li>
  </ul>
  <p>Two things I would not do. Don’t stack a new promise on top of one you just broke — that includes promising a check-in date. And if you normally talk to this client on the phone, call first and send the same three lines right after; a two-week slip that arrives only as text is how a working relationship gets formal.</p>
</div>`
  },
  consensus:
`<div class="ai-consensus">
  <p>Consensus: send it — after two fixes, and after you decide one thing yourself</p>
  <p>All six models read the draft as close to sendable, and not one of them objects to the tone. Nothing in the draft is impolite, and that is not where the risk sits. The risk is in three sentences, and on one of them the models split three against three.</p>
  <h4>Fix before you send</h4>
  <ul>
    <li>The new date belongs in the first line, ahead of the miss, the cause and the apology. She is scanning for a date.</li>
    <li>Send it today, not on the 15th: two weeks of warning is a different message than a same-day cancellation, even though the delay is identical.</li>
    <li>Say what Anna actually gets on the 15th instead of leaving the two weeks blank — that is the question she will be asked as soon as she forwards your message.</li>
    <li>Put it in writing, so she can forward it to whoever planned around the 15th.</li>
    <li>Name the day you will confirm the 29th, so the next update does not arrive as another surprise.</li>
  </ul>
  <h4>Decide for yourself</h4>
  <ul>
    <li>The closing line splits the models down the middle: three read it as the only sentence that gives her a way in, three as an invitation to reopen a date you cannot move.</li>
    <li>A few things came up on our side is the weakest sentence in the draft: half the models want the actual cause in one clause, half want it left exactly as vague as it is.</li>
    <li>The apology itself is not disputed, only who does the work — whether the apology or the plan carries it.</li>
  </ul>
  <h4>What that looks like</h4>
  <blockquote>Hi Anna, the launch moves to the 29th — we will not make the 15th. [One clause on the cause.] What you will have on the 15th is the checkout flow on staging, so your team can start testing on schedule. I will confirm the 29th by the 22nd at the latest. [Your closing line.]</blockquote>
  <p>Both bracketed parts are the ones the models could not settle for you, and both turn on something only you know: whether the 29th is still negotiable, and whether this client reads a named cause as openness or as an excuse.</p>
</div>`,

  // Strukturierte Auswertung – exakt das Schema, das eine echte Consensus-Query
  // liefert. Treibt Verdict-Header, Agreement-Badges und die Differences-Karten
  // (inkl. Contradiction) über window.renderConsensusInsights.
  //
  // Der Score ist nicht gegriffen, sondern die Rechnung aus
  // app/services/llm/consensus_scoring.py auf genau diese Daten:
  // Claim-Schnitt 5.5/6 = 0.9167, minus 0.25 (major) - 0.10 (minor)
  // - 0.05 (emphasis) = 0.5167 -> 52, Deckel 0.64 greift nicht.
  differencesData: {
    models_compared: ["OpenAI", "Mistral", "Anthropic", "Gemini", "DeepSeek", "Grok"],
    best_model: "Anthropic",
    judges: { differences: { provider: "Gemini" } },
    agreement: {
      score: 52,
      level: "partially",
      model_count: 6,
      major_contradictions: 1,
      minor_contradictions: 1,
      emphases: 1
    },
    claims: [
      {
        anchor: "The new date belongs in the first line",
        agree: ["OpenAI", "Mistral", "Anthropic", "Gemini", "DeepSeek", "Grok"],
        dissent: []
      },
      {
        anchor: "Send it today, not on the 15th",
        agree: ["OpenAI", "Mistral", "Anthropic", "Gemini", "DeepSeek", "Grok"],
        dissent: []
      },
      {
        anchor: "Say what Anna actually gets on the 15th",
        agree: ["OpenAI", "Mistral", "Anthropic", "Gemini", "DeepSeek", "Grok"],
        dissent: []
      },
      {
        anchor: "Nothing in the draft is impolite",
        agree: ["OpenAI", "Mistral", "Anthropic", "Gemini", "DeepSeek", "Grok"],
        dissent: []
      },
      {
        anchor: "Put it in writing, so she can forward it",
        agree: ["OpenAI", "Mistral", "Anthropic", "Gemini", "DeepSeek"],
        dissent: [
          {
            model: "Grok",
            quote: "if you normally talk to this client on the phone, call first and send the same three lines right after"
          }
        ]
      },
      {
        anchor: "Name the day you will confirm the 29th",
        agree: ["Mistral", "Anthropic", "Gemini", "DeepSeek"],
        dissent: [
          {
            model: "OpenAI",
            quote: "A promised status note is a second date you now also have to hit"
          },
          {
            model: "Grok",
            quote: "Don’t stack a new promise on top of one you just broke"
          }
        ]
      }
    ],
    differences: [
      {
        claim: "Keep or cut the line asking if the delay is a problem?",
        // Stelle im Konsenstext, die inline markiert wird (Wellenlinie +
        // Marker). Muss woertlich im Antworttext oben vorkommen.
        consensus_anchor: "an invitation to reopen a date you cannot move",
        type: "contradiction",
        severity: "major",
        positions: [
          {
            stance: "Keep it. It is the only opening the message offers her.",
            models: ["OpenAI", "Gemini", "Grok"],
            quote: "It is the only line in the draft that gives Anna a way in, and a delay announced without one reads like a decision taken for her."
          },
          {
            stance: "Cut it. It asks her to rule on something you cannot change.",
            models: ["Anthropic", "Mistral", "DeepSeek"],
            quote: "You are asking her to rule on something you cannot change, and the honest answer to that question is yes, it is a problem"
          }
        ],
        verify: "Decide first whether the 29th is still negotiable. If it is not, do not ask a question that implies it is."
      },
      {
        claim: "Name the cause, or leave it vague?",
        consensus_anchor: "the weakest sentence in the draft",
        type: "contradiction",
        severity: "minor",
        positions: [
          {
            stance: "Name it in one clause; vagueness is what she will question.",
            models: ["OpenAI", "Anthropic", "DeepSeek"],
            quote: "A vague cause reads as a cause you do not want to name, and that is the line she will come back to."
          },
          {
            stance: "Leave it. A named cause moves the talk to your process.",
            models: ["Mistral", "Gemini", "Grok"],
            quote: "A detailed reason moves the conversation to your process, which is the one place you do not want it."
          }
        ],
        verify: "Ask whether the cause changes anything for her. If it does not, it is an explanation for you, not for her."
      },
      {
        claim: "Apologise, or say what you are doing about it?",
        consensus_anchor: "whether the apology or the plan carries it",
        type: "emphasis",
        severity: "minor",
        positions: [
          {
            stance: "One short apology, early — it is what makes the rest readable.",
            models: ["OpenAI", "Gemini", "Grok"],
            quote: "One apology, early and unqualified, is what makes the rest of the message read as news rather than as a defence."
          },
          {
            stance: "Let the plan do it — an apology puts the work on her.",
            models: ["Anthropic", "Mistral", "DeepSeek"],
            quote: "An apology asks something of her; a plan gives her something."
          }
        ],
        verify: "Both camps keep an acknowledgment. The question is only whether it stands alone or comes with the next step attached."
      }
    ]
  },

  differences:
`The consensus answer is partially credible.

All six models agree on the mechanics: new date first, say what she gets on the 15th, put it in writing, and leave the tone alone. They contradict each other on the closing question — three would keep it as the only opening the message offers her, three would cut it as an invitation to reopen a date that is not negotiable — and again, more mildly, on whether the cause should be named or left vague.

BestModel: Anthropic`
};

/* === DEMO: Timing & Typing Configuration =============================== */
const DEMO_PHASES = {
  preType: true,
  order: ["OpenAI", "Anthropic", "Gemini", "Mistral", "DeepSeek", "Grok"],
  // Obergrenze fuer die getippte Frage (ohne den eingefuegten Entwurf).
  typeChars: 140,
  typeSpeed: 40,
  gapBetweenModels: 540,
  pauseAfterTypingAll: 650
};

const DEMO_CONSENSUS_DELAY_MS = 4200;
const DEMO_CONSENSUS_JITTER_MS = 600;
// The structured contradiction check is synchronous in the local demo. Give
// its announced UI phase one deliberate beat so "Checking for contradictions"
// does not flash and disappear between two paints.
const DEMO_DIFFERENCES_REVIEW_MS = 1100;
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

// Staffel-Startzeiten + Tempo für das Token-Streaming der Demo-Antworten.
const DEMO_STREAM_STARTS = { OpenAI: 500, Anthropic: 1150, Gemini: 1800, Mistral: 2450, DeepSeek: 3100, Grok: 3750 };
const DEMO_RESPONSE_STREAM = { wordsPerTick: 2, tickMs: 55 };
const DEMO_CONSENSUS_STREAM = { wordsPerTick: 2, tickMs: 46 };

// Läuft hoch, sobald ein neuer Demo-Durchlauf startet, damit alte
// Streaming-Timer aus einem vorherigen Lauf sauber abbrechen.
let demoRunId = 0;

// Zerlegt eine HTML-Antwort in Tokens: Tags bleiben ganz, Text wird in
// Wörter/Whitespace gesplittet, damit beim schrittweisen Aufbau nie ein
// halbes Tag im DOM landet.
function tokenizeForStream(html) {
  const tokens = [];
  const re = /<[^>]+>|[^<]+/g;
  let match;
  while ((match = re.exec(html))) {
    const part = match[0];
    if (part[0] === "<") {
      tokens.push(part);
    } else {
      const pieces = part.match(/\s+|[^\s]+/g) || [];
      for (const piece of pieces) tokens.push(piece);
    }
  }
  return tokens;
}

// Baut die Antwort wortweise auf – wie ein echter Streaming-Response.
// Tags zählen nicht gegen das Wort-Budget, der Browser schließt offene
// Tags beim Zuweisen von innerHTML automatisch, daher bleibt das Markup gültig.
function streamDemoInto(outputEl, html, runId, opts = {}) {
  return new Promise(resolve => {
    if (!outputEl) { resolve(); return; }
    const wordsPerTick = opts.wordsPerTick || 3;
    const tickMs = opts.tickMs || 38;
    const tokens = tokenizeForStream(html || "");
    let index = 0;
    let acc = "";
    outputEl.innerHTML = "";
    outputEl.classList.add("is-streaming");

    const finish = () => {
      outputEl.classList.remove("is-streaming");
      resolve();
    };

    const tick = () => {
      if (runId !== demoRunId) { finish(); return; }
      let added = 0;
      while (index < tokens.length && added < wordsPerTick) {
        const token = tokens[index++];
        acc += token;
        if (token[0] !== "<" && token.trim()) added++;
      }
      outputEl.innerHTML = acc;
      if (typeof outputEl.scrollTop === "number") outputEl.scrollTop = outputEl.scrollHeight;
      if (index < tokens.length) {
        setTimeout(tick, tickMs);
      } else {
        finish();
      }
    };

    tick();
  });
}

function getDemoStorage() {
  try {
    return window.localStorage || null;
  } catch (e) {
    return null;
  }
}

function showPostDemoLoginPrompt() {
  const prompt = document.getElementById("postDemoLoginPrompt");
  if (!prompt || window.auth?.currentUser) return;

  prompt.hidden = false;
  prompt.classList.remove("is-visible");
  requestAnimationFrame(() => prompt.classList.add("is-visible"));
  window.trackAppEvent?.("app_demo_login_prompt_shown");
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

function renderDemoModelResponse(model, outputEl) {
  const markdown = DEMO_DATA.responses[model] || "";
  // Ohne Quellen, aber ueber denselben Weg wie eine echte Antwort: der
  // Renderer legt nebenbei die Kopier-/Bestantwort-Daten am Kasten ab.
  if (window.renderModelResponseWithSources) {
    window.renderModelResponseWithSources(outputEl, markdown, []);
    return;
  }
  if (window.injectMarkdown) window.injectMarkdown(outputEl, markdown);
}

async function renderDemoConsensus(mainP, diffP) {
  const runId = demoRunId;
  // Der gefuehrte Lauf kennt bei einer echten Query drei Schritte: Antworten,
  // Konsens, Differences. Die Demo hat bisher nur Anfang und Ende gemeldet —
  // dadurch lief nach 6 s der Notausstieg (settleWithoutConsensus) und der
  // Lauf stand auf "Done", waehrend der Konsenstext noch geschrieben wurde.
  // Hier meldet die Demo dieselben Uebergaenge wie ein echter Lauf.
  window.App?.consensusPipeline?.onConsensusStart?.();

  // Konsens-Antwort als Streaming-Response aufbauen, danach sauber rendern,
  // damit Copy-Buttons und die Inline-Marker auf fertigem Markup sitzen.
  if (mainP) {
    await streamDemoInto(mainP, DEMO_DATA.consensus, runId, DEMO_CONSENSUS_STREAM);
    if (runId !== demoRunId) return;
    if (window.injectMarkdown) window.injectMarkdown(mainP, DEMO_DATA.consensus);
  }

  // Konsenstext steht: ab hier prueft die Auswertung auf Widersprueche.
  window.App?.consensusPipeline?.onDifferencesStart?.();
  await sleep(DEMO_DIFFERENCES_REVIEW_MS);
  if (runId !== demoRunId) return;

  // Differences exakt wie bei echten Queries: strukturierte Auswertung mit
  // Verdict-Header, Agreement-Badges und Contradiction-Karten. Nur wenn die
  // strukturierten Daten fehlen, greift der Legacy-Freitext.
  // Demo-Daten gehören zu keinem Bookmark: Resolve-Persistenz-Payload leeren,
  // damit eine Resolve-Runde hier nie ein altes Bookmark überschreibt.
  window.lastConsensusBookmarkPayload = null;
  const includedCount = (DEMO_DATA.differencesData?.models_compared || []).length || 6;
  const structuredRendered = window.renderConsensusInsights
    ? window.renderConsensusInsights(DEMO_DATA.differencesData, includedCount)
    : false;

  if (!structuredRendered && diffP) {
    window.App.differencesPanel?.expandForFallback?.();
    window.applyCredibilityFrame?.(diffP, DEMO_DATA.differences);
    const differences = window.colorizeCredibility?.(DEMO_DATA.differences)
      ?? DEMO_DATA.differences;
    if (window.injectMarkdown) {
      window.injectMarkdown(diffP, differences);
    } else {
      // This only happens while the app's deferred helpers are not available.
      // Keep the demo readable instead of depending on a CDN global directly.
      diffP.textContent = differences;
    }
  }

  // Demo-Ergebnisse sind reine lokale Produktvorschau. Sie dürfen weder das
  // Best-answer-Nutzungssignal noch die serverseitige Differences-Telemetrie
  // beeinflussen; deshalb gibt es hier bewusst keinen Persistenz-/Vote-Aufruf.
  window.App?.consensusPipeline?.onConsensusEnd?.();
  showPostDemoLoginPrompt();
}

async function runDemoFlow() {
  const agentModeEnabled = window.isAgentModeEnabled?.() === true;
  // Auch die lokale Demo respektiert den Produktmodus: Agent Mode baut den
  // Thread auf, der Direktvergleich bleibt bei den sechs Antwortfenstern.
  if (agentModeEnabled) {
    window.exitHeroMode?.();
  } else {
    document.body.classList.add("is-hero", "direct-comparison-active");
    window.App?.setThreadQuestion?.("");
    window.syncHeroResponseAccess?.();
    window.App?.consensusPipeline?.dismiss?.();
  }
  const runId = ++demoRunId;
  const sendBtn = document.getElementById("sendButton");
  if (sendBtn) sendBtn.disabled = true;
  // Neue Demo-Runde: Konsens-Bereich zunächst ausblenden.
  window.hideConsensusOutput?.();

  window.App.state.set("currentEvidenceSources", [], "evidence");
  window.renderEvidenceSources?.([]);

  const qi = document.getElementById("questionInput");
  if (qi && !qi.value.trim()) qi.value = DEMO_SCENARIO_PROMPT;

  if (DEMO_PHASES.preType) {
    const qiEl = document.getElementById("questionInput");
    await typeIntoInput(
      qiEl,
      DEMO_TYPED_QUESTION.slice(0, DEMO_PHASES.typeChars),
      DEMO_PHASES.typeSpeed
    );
    // Den Entwurf tippt niemand ab; er wird eingefuegt. Deshalb erscheint er
    // in einem Zug, mit einer kurzen Pause davor, damit sichtbar bleibt, dass
    // Frage und Nachricht zwei verschiedene Dinge sind.
    if (qiEl) {
      await sleep(340);
      qiEl.value = DEMO_SCENARIO_PROMPT;
      qiEl.dispatchEvent(new Event("input", { bubbles: true }));
    }
    await sleep(DEMO_PHASES.pauseAfterTypingAll);
  }

  // Die fertig getippte Frage wird jetzt "abgeschickt": Sie wandert in den
  // Thread-Kopf und verschwindet wie bei einem echten Lauf aus dem Composer.
  // Erst danach beginnen Fortschrittsanzeige und Modell-Spinner.
  if (agentModeEnabled) {
    window.App?.setThreadQuestion?.(DEMO_SCENARIO_PROMPT);
  }
  if (qi) {
    qi.value = "";
    qi.dispatchEvent(new Event("input", { bubbles: true }));
    window.syncDemoChipState?.();
  }

  window.setAgentModeStatus?.("running");
  Object.keys(MODEL_TO_BOX).forEach(key => {
    const box = getBox(key);
    if (box) setSpinnerEl(box);
  });

  // Jede Modellantwort läuft zeitversetzt als Streaming-Response ein und wird
  // danach sauber gerendert (für [S1]-Quellenlinks und Copy-Buttons).
  await Promise.all(Object.keys(MODEL_TO_BOX).map(model =>
    new Promise(resolve => {
      const start = DEMO_STREAM_STARTS[model] ?? (DEMO_DATA.delays[model] || 1800);
      setTimeout(async () => {
        const box = getBox(model);
        const p = box?.querySelector(".collapsible-content");
        if (!p) { resolve(); return; }
        await streamDemoInto(p, DEMO_DATA.responses[model] || "", runId, DEMO_RESPONSE_STREAM);
        if (runId === demoRunId) renderDemoModelResponse(model, p);
        resolve();
      }, start);
    })
  ));

  if (runId !== demoRunId) return;

  window.setAgentModeStatus?.("complete");

  const consensusDiv = document.getElementById("consensusResponse");
  const mainP = window.App.consensusBodyEl(consensusDiv);
  const diffP = consensusDiv?.querySelector(".consensus-differences p");
  // Consensus/Differences sind Teil des Agent Mode. Im Direktvergleich endet
  // die Demo nach den sechs Modellantworten.
  const auto = window.isAgentModeEnabled?.() === true
    && document.getElementById("autoConsensusToggle")?.checked !== false;

  if (auto) {
    window.resetConsensusInsights?.();
    window.resetCredibilityFrame?.(consensusDiv?.querySelector(".consensus-differences"));
    // Rahmenlosen Konsens-Bereich sanft einblenden, sobald alle Antworten fertig sind.
    window.revealConsensusOutput?.();
    if (mainP) mainP.innerHTML = window.spinnerHTML;
    if (diffP) diffP.innerHTML = window.spinnerHTML;
    setTimeout(
      () => renderDemoConsensus(mainP, diffP),
      DEMO_CONSENSUS_DELAY_MS + Math.floor(Math.random() * DEMO_CONSENSUS_JITTER_MS)
    );
  } else {
    showPostDemoLoginPrompt();
  }

  if (sendBtn) sendBtn.disabled = false;
}

function createStartDemoChip() {
  const storage = getDemoStorage();
  if (storage?.getItem("demoChipDismissed")) return;
  const container = document.querySelector(".chat-input-container");
  if (!container || container.querySelector(".demo-chip")) return;
  const questionInput = document.getElementById("questionInput");

  const btn = document.createElement("button");
  btn.className = "demo-chip";
  btn.type = "button";
  btn.setAttribute("aria-label", "Start interactive demo");
  // Zwei Beschriftungen, immer genau eine sichtbar. Auf einem 375er Schirm
  // teilen sich (+), Lauf-Schalter, dieser Knopf und Senden 315 px — mit
  // "Watch demo" passte das nicht mehr in eine Zeile und der Senden-Knopf
  // rutschte allein in eine zweite. Welche Beschriftung gilt, entscheidet
  // components-misc.css; der aria-Name bleibt in beiden Faellen derselbe.
  btn.innerHTML =
    '<span class="demo-chip-label demo-chip-label-full">Watch demo</span>' +
    '<span class="demo-chip-label demo-chip-label-short">Demo</span>';

  const inputActions = container.querySelector(".input-actions-container");
  if (inputActions) {
    inputActions.prepend(btn);
  } else {
    container.appendChild(btn);
  }

  const syncChipState = () => {
    const hasQuestionText = Boolean(questionInput?.value.length);
    container.classList.toggle("has-question-input", hasQuestionText);
    btn.hidden = hasQuestionText;
    btn.tabIndex = hasQuestionText ? -1 : 0;
  };

  window.syncDemoChipState = syncChipState;

  if (questionInput) {
    questionInput.addEventListener("input", event => {
      syncChipState();
      if (questionInput.value.length && event.isTrusted) {
        storage?.setItem("demoChipDismissed", "1");
        btn.remove();
      }
    });
    questionInput.addEventListener("change", syncChipState);
  }

  btn.addEventListener("click", async () => {
    storage?.setItem("demoChipDismissed", "1");
    btn.remove();
    await runDemoFlow();
  });

  syncChipState();
}

window.runDemoFlow = runDemoFlow;
window.createStartDemoChip = createStartDemoChip;
createStartDemoChip();

/* === DEMO: Auto-Start von der Landingpage (/app?demo=1) ================ */
// Der Hero der Landingpage verlinkt auf /app?demo=1: Die Demo startet dann
// automatisch in der echten App-Oberfläche, statt auf der Landingpage einen
// zweiten App-Nachbau zu pflegen. Nach der Demo ist der Nutzer bereits in der
// App und sieht den bestehenden Post-Demo-Login-Prompt.
function maybeAutoStartDemo() {
  let shouldStart = false;
  try {
    const params = new URLSearchParams(window.location.search);
    if (params.get("demo") === "1") {
      shouldStart = true;
      // Parameter entfernen, damit Reload/Bookmark die Demo nicht erneut startet.
      params.delete("demo");
      const query = params.toString();
      window.history.replaceState(
        null, "",
        window.location.pathname + (query ? "?" + query : "") + window.location.hash
      );
    }
  } catch (e) {
    return;
  }
  if (!shouldStart) return;

  // Auto-Start ersetzt den Chip-Klick: Chip einmalig als erledigt markieren.
  getDemoStorage()?.setItem("demoChipDismissed", "1");

  const start = () => {
    document.querySelector(".chat-input-container .demo-chip")?.remove();
    window.trackAppEvent?.("app_demo_autostart", { source: "landing" });
    // Kurze Pause: Erst rendert der Hero, dann beginnt die Demo zu tippen.
    // Das Tippen selbst überbrückt die restliche Initialisierung der App.
    setTimeout(() => runDemoFlow(), 450);
  };
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start, { once: true });
  } else {
    start();
  }
}
maybeAutoStartDemo();

document.getElementById("postDemoLoginButton")?.addEventListener("click", () => {
  const modal = document.getElementById("loginModal");
  if (!modal) return;

  modal.style.display = "block";
  window.trackAppEvent?.("auth_modal_open", { source: "post_demo" });
  requestAnimationFrame(() => document.getElementById("loginEmail")?.focus());
});

function toggleSettingsCollapse(contentId, arrowId) {
  const content = document.getElementById(contentId);
  const arrow = document.getElementById(arrowId);
  if (!content) return;

  // The initial closed state comes from a template CSS class. Inspect the
  // effective value so the first click opens it even before an inline value
  // has ever been written.
  if (window.getComputedStyle(content).display === "none") {
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
