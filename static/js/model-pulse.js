// Dedicated model-pulse ranking. Reads the public, anonymized provider-family
// totals and renders them without exposing prompt or user data.

(function () {
  const list = document.getElementById("modelLeaderboardRows");
  const total = document.getElementById("modelLeaderboardTotal");
  const periodLabel = document.getElementById("modelLeaderboardPeriod");
  const periodButtons = Array.from(document.querySelectorAll("[data-model-pulse-period]"));
  if (!list || !total) return;

  const number = new Intl.NumberFormat(document.documentElement.lang || "en");
  let requestVersion = 0;

  function renderEmpty(message) {
    const empty = document.createElement("p");
    empty.className = "lp-model-pulse-empty";
    empty.textContent = message;
    list.replaceChildren(empty);
    list.classList.remove("is-loading");
    list.setAttribute("aria-busy", "false");
  }

  function renderRows(rows, totalSelections, period) {
    const maxSelections = Math.max(...rows.map(row => row.selections), 1);
    const fragment = document.createDocumentFragment();

    rows.slice(0, 9).forEach((row, index) => {
      const item = document.createElement("div");
      item.className = "lp-model-pulse-row";
      item.setAttribute("role", "listitem");

      const rank = document.createElement("span");
      rank.className = "lp-model-pulse-rank";
      rank.textContent = String(index + 1).padStart(2, "0");

      const iconWrap = document.createElement("span");
      iconWrap.className = "lp-model-pulse-icon";
      const icon = document.createElement("img");
      icon.src = typeof row.icon === "string" ? row.icon : "/static/favicon.png";
      icon.alt = "";
      iconWrap.appendChild(icon);

      const copy = document.createElement("span");
      copy.className = "lp-model-pulse-copy";
      const name = document.createElement("span");
      name.className = "lp-model-pulse-name";
      name.textContent = row.family;
      if (row.available_since) {
        const available = document.createElement("small");
        available.className = "lp-model-pulse-since";
        available.textContent = "tracked since 31 Aug 2026";
        name.appendChild(available);
      }
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
    total.textContent = totalSelections
      ? `${number.format(totalSelections)} judge selections ${period === "all" ? "from all real runs" : "since 31 Aug 2026"}.`
      : `No judge selections recorded ${period === "all" ? "yet" : "since 31 Aug 2026"}.`;
    requestAnimationFrame(() => list.classList.add("is-ready"));
  }

  function setLoading(period) {
    list.classList.add("is-loading");
    list.classList.remove("is-ready");
    list.setAttribute("aria-busy", "true");
    periodButtons.forEach(button => {
      const active = button.dataset.modelPulsePeriod === period;
      button.setAttribute("aria-pressed", String(active));
      button.disabled = true;
    });
    if (periodLabel) {
      periodLabel.textContent = period === "all" ? "All-time ranking" : "Shared-window ranking";
    }
  }

  function loadPeriod(period) {
    const currentRequest = ++requestVersion;
    setLoading(period);
    fetch(`/api/model-leaderboard?period=${encodeURIComponent(period)}`, {
      headers: { Accept: "application/json" },
      credentials: "same-origin"
    })
      .then(response => {
        if (!response.ok) throw new Error(`Leaderboard request failed (${response.status})`);
        return response.json();
      })
      .then(data => {
        if (currentRequest !== requestVersion) return;
        const rows = Array.isArray(data.rows)
          ? data.rows.filter(row => row && typeof row.family === "string" && Number(row.selections) >= 0)
          : [];
        if (!rows.length) {
          total.textContent = "The live tally starts with the first recorded selection.";
          renderEmpty("No model families are available in this view.");
          return;
        }
        renderRows(
          rows,
          Number(data.total_selections) || rows.reduce((sum, row) => sum + Number(row.selections), 0),
          data.period === "all" ? "all" : "since-2026-08-31"
        );
      })
      .catch(error => {
        if (currentRequest !== requestVersion) return;
        console.warn("Model pulse unavailable:", error);
        total.textContent = "The live tally is temporarily unavailable.";
        renderEmpty("Model pulse unavailable right now.");
      })
      .finally(() => {
        if (currentRequest !== requestVersion) return;
        periodButtons.forEach(button => { button.disabled = false; });
      });
  }

  periodButtons.forEach(button => {
    button.addEventListener("click", () => loadPeriod(button.dataset.modelPulsePeriod || "all"));
  });

  loadPeriod("all");
})();
