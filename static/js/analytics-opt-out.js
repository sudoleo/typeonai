(function () {
  try {
    const flag = new URLSearchParams(window.location.search).get("notrack");
    if (flag === "1") localStorage.setItem("umami.disabled", "1");
    else if (flag === "0") localStorage.removeItem("umami.disabled");
  } catch (_) { /* storage unavailable */ }
})();

