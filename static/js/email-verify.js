(function () {
"use strict";

// =====================================================================
// email-verify.js
// Der Streifen fuer den Zustand "angemeldet, aber E-Mail noch nicht
// bestaetigt" (#verifyBanner in templates/index.html).
//
// Frueher war dieser Zustand ein Rauswurf: onIdTokenChanged hat unbestaetigte
// Nutzer sofort ausgeloggt. Wer den Bestaetigungslink nicht im selben Moment
// fand, stand wieder vor der Login-Maske — mit verlorener Frage. Jetzt bleibt
// die Session stehen, die App ist sichtbar, und dieser Streifen sagt, was
// fehlt: ein Klick im Postfach.
//
// Dieses Modul kennt Firebase NICHT. firebase.js reicht die drei Aktionen
// (erneut senden, Status neu laden, ausloggen) als Callbacks herein.
// =====================================================================

const RESEND_COOLDOWN_MS = 60_000;
// Der Nutzer bestaetigt typischerweise in einem ANDEREN Tab. Kommt er
// zurueck, soll die App das von selbst merken, statt auf einen Klick zu
// warten — aber ohne Firebase mit reload()-Aufrufen zu bombardieren.
const RECHECK_THROTTLE_MS = 4_000;

let handlers = null;
let cooldownTimer = null;
let cooldownUntil = 0;
let lastRecheckAt = 0;
let wired = false;

function el(id) {
  return document.getElementById(id);
}

function setStatus(message, tone) {
  const status = el("verifyBannerStatus");
  if (!status) return;
  if (!message) {
    status.hidden = true;
    status.textContent = "";
    status.classList.remove("is-error", "is-success");
    return;
  }
  status.hidden = false;
  status.textContent = message;
  status.classList.toggle("is-error", tone === "error");
  status.classList.toggle("is-success", tone === "success");
}

function syncResendButton() {
  const button = el("verifyBannerResend");
  if (!button) return;
  const remaining = Math.ceil((cooldownUntil - Date.now()) / 1000);
  if (remaining > 0) {
    button.disabled = true;
    button.textContent = `Resend in ${remaining}s`;
    return;
  }
  button.disabled = false;
  button.textContent = "Resend the link";
  if (cooldownTimer) {
    clearInterval(cooldownTimer);
    cooldownTimer = null;
  }
}

function startCooldown() {
  cooldownUntil = Date.now() + RESEND_COOLDOWN_MS;
  if (cooldownTimer) clearInterval(cooldownTimer);
  cooldownTimer = setInterval(syncResendButton, 1000);
  syncResendButton();
}

async function handleResend() {
  if (!handlers?.onResend || Date.now() < cooldownUntil) return;
  setStatus("Sending…");
  startCooldown();
  try {
    await handlers.onResend();
    setStatus("Link sent. It can take a minute — check spam and promotions too.", "success");
  } catch (error) {
    // Firebase drosselt haeufiges Neusenden selbst; das ist kein Fehler des
    // Nutzers, also sagen wir, was zu tun ist, statt einen Code zu zeigen.
    const tooMany = String(error?.code || "").includes("too-many-requests");
    setStatus(
      tooMany
        ? "Too many attempts for now. The last link is still valid — try that one."
        : "The e-mail could not be sent right now. Please try again in a moment.",
      "error"
    );
  }
}

// Silent: der Hintergrund-Check beim Zurueckwechseln in den Tab. Der bleibt
// stumm, solange nichts passiert ist — sonst blinkt bei jedem Tabwechsel eine
// Fehlermeldung auf, obwohl der Nutzer gar nichts getan hat.
async function recheck(silent) {
  if (!handlers?.onRecheck) return false;
  const now = Date.now();
  if (silent && now - lastRecheckAt < RECHECK_THROTTLE_MS) return false;
  lastRecheckAt = now;
  if (!silent) setStatus("Checking…");
  let verified = false;
  try {
    verified = await handlers.onRecheck();
  } catch (_) {
    if (!silent) {
      setStatus("Could not check right now. Please try again in a moment.", "error");
    }
    return false;
  }
  if (verified) {
    setStatus("Confirmed. Loading your app…", "success");
    return true;
  }
  if (!silent) {
    setStatus("Not confirmed yet. Open the link in the e-mail, then try again.", "error");
  }
  return false;
}

function wire() {
  if (wired) return;
  const banner = el("verifyBanner");
  if (!banner) return;
  wired = true;
  el("verifyBannerResend")?.addEventListener("click", handleResend);
  el("verifyBannerRecheck")?.addEventListener("click", () => recheck(false));
  el("verifyBannerSignOut")?.addEventListener("click", () => handlers?.onSignOut?.());
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible" && !banner.hidden) recheck(true);
  });
  window.addEventListener("focus", () => {
    if (!banner.hidden) recheck(true);
  });
}

function showEmailVerificationGate(options) {
  handlers = options || {};
  wire();
  const banner = el("verifyBanner");
  if (!banner) return;
  const email = el("verifyBannerEmail");
  if (email) email.textContent = handlers.email || "your address";
  banner.hidden = false;
  document.body.classList.add("is-unverified");
  syncResendButton();
}

function hideEmailVerificationGate() {
  const banner = el("verifyBanner");
  if (banner) banner.hidden = true;
  document.body.classList.remove("is-unverified");
  setStatus("");
  handlers = null;
}

window.App = window.App || {};
window.App.emailVerification = Object.freeze({
  showEmailVerificationGate,
  hideEmailVerificationGate
});
})();
