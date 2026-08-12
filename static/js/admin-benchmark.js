import { initializeApp } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-app.js";
import { getAuth, onAuthStateChanged } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-auth.js";

const app = initializeApp(window.FIREBASE_CONFIG);
const auth = getAuth(app);

const SYSTEM_LABELS = {
  "model:openai": "OpenAI",
  "model:mistral": "Mistral",
  "model:anthropic": "Anthropic",
  "model:gemini": "Gemini",
  "model:deepseek": "DeepSeek",
  "model:grok": "Grok",
  majority_vote: "Majority Vote",
  consensus: "Consensus",
  synth_alone: "Synthesizer alone",
};
const AGGREGATE_KEYS = new Set(["majority_vote", "consensus", "synth_alone"]);

function setStatus(message, isError = false) {
  const status = document.getElementById("bmStatus");
  status.textContent = message || "";
  status.className = `bm-status${isError ? " error" : ""}`;
}

async function api(path) {
  const user = auth.currentUser;
  if (!user) throw new Error("Not logged in");
  const idToken = await user.getIdToken();
  const response = await fetch(path, {
    headers: { Authorization: `Bearer ${idToken}` },
  });
  let data = {};
  try {
    data = await response.json();
  } catch (_error) {
    // Preserve the HTTP status when an intermediary did not return JSON.
  }
  if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
  return data;
}

function fmtPct(value) {
  return value === null || value === undefined ? "–" : `${(value * 100).toFixed(1)}%`;
}

function fmtCi(interval) {
  if (!Array.isArray(interval)) return "";
  return ` [${(interval[0] * 100).toFixed(0)}–${(interval[1] * 100).toFixed(0)}]`;
}

function fmtUsd(value) {
  return value === null || value === undefined ? "–" : `$${Number(value).toFixed(4)}`;
}

function fmtMs(value) {
  return value === null || value === undefined ? "–" : `${Math.round(value)} ms`;
}

function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  Object.assign(node, props);
  for (const child of [].concat(children)) {
    if (child == null) continue;
    node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return node;
}

async function loadRuns() {
  setStatus("Loading runs…");
  let data;
  try {
    data = await api("/api/admin/benchmark/runs");
  } catch (error) {
    setStatus(error.message, true);
    return;
  }
  const runs = data.runs || [];
  const select = document.getElementById("runSelect");
  select.replaceChildren();
  if (!runs.length) {
    setStatus("No benchmark runs found.", true);
    document.getElementById("runDetail").replaceChildren();
    return;
  }
  for (const run of runs) {
    const cost = run.total_cost_usd != null ? ` · ${fmtUsd(run.total_cost_usd)}` : "";
    const count = run.n_questions != null ? ` · ${run.n_questions}Q` : "";
    const source = run.source ? ` · ${run.source}` : "";
    select.appendChild(
      el("option", {
        value: run.run_id,
        textContent: `${run.run_id} (${run.sample_role || "?"}${count}${cost}${source})`,
      }),
    );
  }
  setStatus("");
  await loadRun(select.value);
}

async function loadRun(runId) {
  if (!runId) return;
  setStatus(`Loading ${runId}…`);
  let data;
  try {
    data = await api(`/api/admin/benchmark/runs/${encodeURIComponent(runId)}`);
  } catch (error) {
    setStatus(error.message, true);
    return;
  }
  setStatus("");
  renderRun(data.run);
}

function renderRun(run) {
  const root = document.getElementById("runDetail");
  root.replaceChildren();
  if (!run) {
    root.appendChild(el("p", { className: "bm-empty", textContent: "No data." }));
    return;
  }
  root.appendChild(renderManifest(run.manifest, run.run_id));
  if (run.results) {
    root.appendChild(renderAccuracyBars(run.results));
    root.appendChild(renderComparisonTable(run.results));
  } else {
    root.appendChild(
      el(
        "div",
        { className: "bm-card" },
        el("p", {
          className: "bm-empty",
          textContent: "No results.json yet (run not finished or dry-run only).",
        }),
      ),
    );
  }
  if (run.audits) root.appendChild(renderAudits(run.audits));
  if (run.questions?.length) root.appendChild(renderMatrix(run.questions));
}

function metaItem(label, value) {
  return el("div", {}, [
    el("span", { textContent: label }),
    document.createTextNode(value == null ? "–" : String(value)),
  ]);
}

function renderManifest(manifest, runId) {
  const card = el("div", { className: "bm-card" }, el("h3", { textContent: `Run · ${runId}` }));
  const data = manifest || {};
  card.appendChild(
    el("div", { className: "bm-meta-grid" }, [
      metaItem("Sample", data.sample_role),
      metaItem("Label mode", data.label_mode),
      metaItem("Consensus model", data.consensus_model),
      metaItem("Temperature", data.temperature),
      metaItem("Output token limit", data.output_token_limit),
      metaItem("Consensus token limit", data.consensus_output_token_limit),
      metaItem("Synth alone", String(data.include_synth_alone)),
      metaItem("Created", data.created),
    ]),
  );
  if (Array.isArray(data.models) && data.models.length) {
    const table = el("table", { className: "bm-table" });
    table.appendChild(
      el("thead", {}, el("tr", {}, [
        el("th", { textContent: "Provider" }),
        el("th", { textContent: "internal_id" }),
        el("th", { textContent: "resolved api_model" }),
        el("th", { textContent: "output limit" }),
        el("th", { textContent: "temperature" }),
      ])),
    );
    const body = el("tbody");
    for (const row of data.models) {
      body.appendChild(el("tr", {}, [
        el("td", { textContent: row.provider }),
        el("td", { textContent: row.internal_id }),
        el("td", { textContent: row.resolved_api_model }),
        el("td", { textContent: row.output_token_limit }),
        el("td", { textContent: row.temperature == null ? "provider default" : row.temperature }),
      ]));
    }
    table.appendChild(body);
    card.appendChild(el("div", { className: "bm-model-table-wrap" }, table));
  }
  if (data.system_prompt) {
    card.appendChild(promptBlock("Closed-book system prompt (sent to all 6 models)", data.system_prompt));
  }
  if (data.consensus_prompt_template) {
    card.appendChild(
      promptBlock(
        "Consensus synthesis prompt — V0 (template; {…} filled per question)",
        data.consensus_prompt_template,
      ),
    );
  }
  return card;
}

function promptBlock(label, text) {
  return el("details", { className: "bm-prompt" }, [
    el("summary", { textContent: label }),
    el("pre", { textContent: text }),
  ]);
}

function orderedSystemKeys(systems) {
  const models = Object.keys(systems).filter((key) => key.startsWith("model:"));
  const aggregates = ["majority_vote", "consensus", "synth_alone"].filter((key) => key in systems);
  return models.concat(aggregates);
}

function renderAccuracyBars(results) {
  const card = el("div", { className: "bm-card" }, el("h3", {
    textContent: `Accuracy (overall) · ${results.n_questions ?? "?"} questions · ${results.n_disagreement ?? 0} disagreement`,
  }));
  const systems = results.systems || {};
  for (const key of orderedSystemKeys(systems)) {
    const accuracy = systems[key].accuracy_overall;
    const percent = accuracy == null ? 0 : accuracy * 100;
    const fill = el("div", { className: `bm-bar-fill${key === "consensus" ? " consensus" : ""}` });
    fill.style.width = `${percent.toFixed(1)}%`;
    card.appendChild(el("div", { className: "bm-bar-wrap" }, [
      el("div", { className: "bm-bar-label", textContent: SYSTEM_LABELS[key] || key }),
      el("div", { className: "bm-bar-track" }, fill),
      el("div", { className: "bm-bar-val", textContent: fmtPct(accuracy) }),
    ]));
  }
  return card;
}

function renderComparisonTable(results) {
  const card = el("div", { className: "bm-card" }, el("h3", { textContent: "System comparison" }));
  const systems = results.systems || {};
  const table = el("table", { className: "bm-table" });
  table.appendChild(
    el("thead", {}, el("tr", {}, [
      "System",
      "Acc. overall (95% CI)",
      "Acc. disagreement (95% CI)",
      "Correct",
      "Parse rate",
      "Abstain",
      "Errors",
      "Cost",
      "Latency avg",
    ].map((heading) => el("th", { textContent: heading })))),
  );
  const body = el("tbody");
  for (const key of orderedSystemKeys(systems)) {
    const system = systems[key];
    const latency = (system.latency_ms || {}).avg;
    body.appendChild(el("tr", AGGREGATE_KEYS.has(key) ? { className: "bm-aggregate" } : {}, [
      el("td", { textContent: SYSTEM_LABELS[key] || key }),
      el("td", { textContent: fmtPct(system.accuracy_overall) + fmtCi(system.accuracy_overall_ci) }),
      el("td", { textContent: fmtPct(system.accuracy_disagreement) + fmtCi(system.accuracy_disagreement_ci) }),
      el("td", { textContent: `${system.correct ?? "–"}/${system.total ?? "–"}` }),
      el("td", { textContent: system.parse_rate == null ? "–" : fmtPct(system.parse_rate) }),
      el("td", {
        textContent: system.abstain ?? (key === "majority_vote" ? `${system.no_majority ?? 0} no-maj` : "–"),
      }),
      el("td", { textContent: system.error ?? "–" }),
      el("td", { textContent: system.cost_usd == null ? "–" : fmtUsd(system.cost_usd) }),
      el("td", { textContent: latency == null ? "–" : fmtMs(latency) }),
    ]));
  }
  table.appendChild(body);
  card.appendChild(table);
  const totals = results.totals || {};
  card.appendChild(el("p", {
    className: "bm-status bm-total",
    textContent: `Total: ${totals.cells ?? "–"} cells · ${fmtUsd(totals.cost_usd)} · ${totals.errors ?? 0} errors`,
  }));
  return card;
}

function auditPill(passed, total) {
  const kind = total === 0 ? "warn" : passed === total ? "ok" : "bad";
  return el("span", { className: `bm-pill ${kind}`, textContent: `${passed}/${total}` });
}

function renderAudits(audits) {
  const card = el("div", { className: "bm-card" }, el("h3", { textContent: "E4 audits" }));
  const permutation = audits.option_permutation || {};
  const order = audits.consensus_order || {};

  const permutationRow = el("div", { className: "bm-bar-wrap" }, [
    el("div", { className: "bm-bar-label", textContent: "Option permutation" }),
  ]);
  if (permutation.enabled === false) {
    permutationRow.appendChild(el("span", {
      className: "bm-pill warn",
      textContent: `disabled (${permutation.reason || "n/a"})`,
    }));
  } else {
    permutationRow.appendChild(
      auditPill(permutation.consistent ?? 0, permutation.conclusive ?? permutation.total ?? 0),
    );
    permutationRow.appendChild(el("span", {
      className: "bm-status",
      textContent: ` consistent of ${permutation.conclusive ?? 0} conclusive (${permutation.total ?? 0} checks)`,
    }));
  }
  card.appendChild(permutationRow);

  const orderRow = el("div", { className: "bm-bar-wrap" }, [
    el("div", { className: "bm-bar-label", textContent: "Consensus order" }),
  ]);
  if (order.enabled === false) {
    orderRow.appendChild(el("span", {
      className: "bm-pill warn",
      textContent: `disabled (${order.reason || "n/a"})`,
    }));
  } else {
    orderRow.appendChild(auditPill(order.stable ?? 0, order.total ?? 0));
    orderRow.appendChild(el("span", {
      className: "bm-status",
      textContent: " stable across normal/reversed/shuffled",
    }));
  }
  card.appendChild(orderRow);
  return card;
}

function letterCell(cell) {
  if (!cell) return el("td", { className: "cell-abstain", textContent: "·" });
  if (cell.error) return el("td", { className: "cell-error", textContent: "err" });
  if (cell.abstain || !cell.letter) return el("td", { className: "cell-abstain", textContent: "–" });
  return el("td", {
    className: cell.correct ? "cell-correct" : "cell-wrong",
    textContent: cell.letter,
  });
}

function renderMatrix(questions) {
  const card = el("div", { className: "bm-card" }, el("h3", { textContent: "Per-question matrix" }));
  const providers = [];
  for (const question of questions) {
    for (const provider of Object.keys(question.models || {})) {
      if (!providers.includes(provider)) providers.push(provider);
    }
  }

  const wrap = el("div", { className: "bm-matrix-wrap" });
  const table = el("table", { className: "bm-matrix" });
  const headings = ["#", "Category", "GT"]
    .concat(providers.map((provider) => provider.slice(0, 4)))
    .concat(["Maj", "Cons", "Synth", "≠"]);
  table.appendChild(
    el("thead", {}, el("tr", {}, headings.map((heading) => el("th", { textContent: heading })))),
  );
  const body = el("tbody");
  for (const question of questions) {
    const row = el("tr", {}, [
      el("td", { textContent: question.question_id }),
      el("td", {
        className: "qcat",
        textContent: question.category || "",
        title: question.category || "",
      }),
      el("td", { textContent: question.ground_truth || "" }),
    ]);
    for (const provider of providers) row.appendChild(letterCell((question.models || {})[provider]));
    row.appendChild(letterCell(question.majority));
    row.appendChild(letterCell(question.consensus));
    row.appendChild(letterCell(question.synth_alone));
    row.appendChild(el("td", { textContent: question.disagreement ? "✓" : "" }));
    body.appendChild(row);
  }
  table.appendChild(body);
  wrap.appendChild(table);
  card.appendChild(wrap);
  card.appendChild(el("div", { className: "bm-legend" }, [
    legendSwatch("cell-correct", "correct"),
    legendSwatch("cell-wrong", "wrong"),
    legendSwatch("cell-abstain", "abstain / no answer"),
    legendSwatch("cell-error", "error"),
  ]));
  return card;
}

function legendSwatch(className, label) {
  return el("span", {}, [
    el("span", { className: `bm-swatch ${className}` }),
    document.createTextNode(label),
  ]);
}

document.getElementById("runSelect").addEventListener("change", (event) => {
  void loadRun(event.target.value);
});
document.getElementById("refreshBtn").addEventListener("click", () => {
  void loadRuns();
});

onAuthStateChanged(auth, (user) => {
  if (user) {
    void loadRuns();
  } else {
    setStatus("Please log in as an admin to view benchmark runs.", true);
    window.location.href = "/";
  }
});
