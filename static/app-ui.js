const DEFAULT_SYSTEM_PROMPT = "Please respond briefly and precisely, focusing only on the essentials.";

function getStoredSystemPrompt() {
  const stored = localStorage.getItem("systemPrompt");
  if (stored !== null) {
    return stored;
  }
  localStorage.setItem("systemPrompt", DEFAULT_SYSTEM_PROMPT);
  return DEFAULT_SYSTEM_PROMPT;
}

function openSettingsModal() {
  const modal = document.getElementById("systemPromptModal");
  const textarea = document.getElementById("systemPromptInput");
  if (!modal || !textarea) return;

  textarea.value = getStoredSystemPrompt();
  modal.style.display = "block";
}

function closeSettingsModal() {
  const modal = document.getElementById("systemPromptModal");
  if (modal) {
    modal.style.display = "none";
  }
}

function saveSystemPrompt() {
  const textarea = document.getElementById("systemPromptInput");
  if (textarea) {
    localStorage.setItem("systemPrompt", textarea.value.trim());
  }
  closeSettingsModal();
}

function openHelpModal() {
  const helpModal = document.getElementById("helpModal");
  if (helpModal) {
    helpModal.style.display = "block";
  }
}

function closeHelpModal() {
  const helpModal = document.getElementById("helpModal");
  if (helpModal) {
    helpModal.style.display = "none";
  }
}

function bindSettingsModalControls() {
  if (window.__settingsModalControlsBound) return;
  window.__settingsModalControlsBound = true;

  document.getElementById("editSystemPromptBtn")?.addEventListener("click", openSettingsModal);
  document.getElementById("closeSystemPromptModal")?.addEventListener("click", closeSettingsModal);
  document.getElementById("saveSystemPromptBtn")?.addEventListener("click", saveSystemPrompt);
  document.getElementById("helpButton")?.addEventListener("click", openHelpModal);
  document.getElementById("closeHelpModal")?.addEventListener("click", closeHelpModal);

  window.addEventListener("click", (event) => {
    if (event.target === document.getElementById("systemPromptModal")) {
      closeSettingsModal();
    }
    if (event.target === document.getElementById("helpModal")) {
      closeHelpModal();
    }
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", bindSettingsModalControls);
} else {
  bindSettingsModalControls();
}
