(function () {
  document.addEventListener("click", (event) => {
    const exclude = event.target.closest("[data-exclude-target]");
    if (exclude) return window.toggleExclude?.(exclude.dataset.excludeTarget);
    if (event.target.closest("#bookmarksToggle")) return window.toggleBookmarks?.();
    if (event.target.closest("#sendButton")) return window.sendQuestion?.();
    if (event.target.closest("#apiSettingsToggle")) {
      return window.toggleSettingsCollapse?.("apiSettingsArea", "apiArrow");
    }
    if (event.target.closest("#testAllKeysButton")) return window.testAllKeys?.();
  });
  document.addEventListener("keydown", (event) => {
    if (!event.target.closest("#apiSettingsToggle")) return;
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    window.toggleSettingsCollapse?.("apiSettingsArea", "apiArrow");
  });
})();
