// Dedicated model-pulse ranking. Reads the public, anonymized provider-family
// totals and renders them without exposing prompt or user data.

(function () {
  const list = document.getElementById("modelLeaderboardRows");
  const total = document.getElementById("modelLeaderboardTotal");
  if (!list || !total) return;

  const number = new Intl.NumberFormat(document.documentElement.lang || "en");
  const icons = [
    [/anthropic|claude/i, "/static/icons/chat_icons/claude.png"],
    [/openai|chatgpt/i, "/static/icons/chat_icons/chatgpt.png"],
    [/google|gemini/i, "/static/icons/chat_icons/gemini-icon.png"],
    [/mistral/i, "/static/icons/chat_icons/mistral.png"],
    [/deepseek/i, "/static/icons/chat_icons/deepseek.png"],
    [/grok|xai/i, "/static/icons/chat_icons/grok.png"]
  ];

  function iconFor(family) {
    return icons.find(([pattern]) => pattern.test(family))?.[1] || "/static/favicon.png";
  }

  function renderEmpty(message) {
    const empty = document.createElement("p");
    empty.className = "lp-model-pulse-empty";
    empty.textContent = message;
    list.replaceChildren(empty);
    list.classList.remove("is-loading");
    list.setAttribute("aria-busy", "false");
  }

  function renderRows(rows, totalSelections) {
    const maxSelections = Math.max(...rows.map(row => row.selections), 1);
    const fragment = document.createDocumentFragment();

    rows.slice(0, 6).forEach((row, index) => {
      const item = document.createElement("div");
      item.className = "lp-model-pulse-row";
      item.setAttribute("role", "listitem");

      const rank = document.createElement("span");
      rank.className = "lp-model-pulse-rank";
      rank.textContent = String(index + 1).padStart(2, "0");

      const iconWrap = document.createElement("span");
      iconWrap.className = "lp-model-pulse-icon";
      const icon = document.createElement("img");
      icon.src = iconFor(row.family);
      icon.alt = "";
      iconWrap.appendChild(icon);

      const copy = document.createElement("span");
      copy.className = "lp-model-pulse-copy";
      const name = document.createElement("span");
      name.className = "lp-model-pulse-name";
      name.textContent = row.family;
      const meter = document.createElement("span");
      meter.className = "lp-model-pulse-meter";
      meter.style.setProperty("--pulse-share", String((row.selections / maxSelections) * 100));
      meter.appendChild(document.createElement("i"));
      copy.append(name, meter);

      const value = document.createElement("span");
      value.className = "lp-model-pulse-value";
      value.textContent = number.format(row.selections);
      const unit = document.createElement("small");
      unit.textContent = row.selections === 1 ? "pick" : "picks";
      value.appendChild(unit);

      item.append(rank, iconWrap, copy, value);
      fragment.appendChild(item);
    });

    list.replaceChildren(fragment);
    list.setAttribute("role", "list");
    list.classList.remove("is-loading");
    list.setAttribute("aria-busy", "false");
    total.textContent = `${number.format(totalSelections)} judge selections from real runs.`;
    requestAnimationFrame(() => list.classList.add("is-ready"));
  }

  fetch("/api/model-leaderboard", {
    headers: { Accept: "application/json" },
    credentials: "same-origin"
  })
    .then(response => {
      if (!response.ok) throw new Error(`Leaderboard request failed (${response.status})`);
      return response.json();
    })
    .then(data => {
      const rows = Array.isArray(data.rows)
        ? data.rows.filter(row => row && typeof row.family === "string" && Number(row.selections) > 0)
        : [];
      if (!rows.length) {
        total.textContent = "The live tally starts with the first recorded selection.";
        renderEmpty("No best-answer selections have been recorded yet.");
        return;
      }
      renderRows(rows, Number(data.total_selections) || rows.reduce((sum, row) => sum + Number(row.selections), 0));
    })
    .catch(error => {
      console.warn("Model pulse unavailable:", error);
      total.textContent = "The live tally is temporarily unavailable.";
      renderEmpty("Model pulse unavailable right now.");
    });
})();
