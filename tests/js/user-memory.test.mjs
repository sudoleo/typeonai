/**
 * user-memory.js -- der Schalter "Use my memory".
 *
 * Der Server antwortet mit dem gespeicherten Profil INKLUSIVE seiner
 * schema_version, und PUT /api/my/memory verbietet unbekannte Felder. Wer die
 * Antwort ungefiltert zurueckschickt, bekommt 422 -- und der Schalter sprang
 * genau deshalb in seine alte Stellung zurueck. Diese Tests halten fest, dass
 * der Body nur die Felder der Schnittstelle enthaelt.
 */

import { beforeEach, describe, expect, it } from "vitest";

import { loadScripts } from "./helpers/appWindow.mjs";

const SETTINGS = `
  <section id="memorySettingsSection">
    <label for="memoryEnabledSwitch"><input type="checkbox" id="memoryEnabledSwitch"></label>
    <textarea id="memoryRoleInput" maxlength="250"></textarea>
    <textarea id="memoryFocusInput" maxlength="250"></textarea>
    <textarea id="memoryStyleInput" maxlength="250"></textarea>
    <textarea id="memoryConstraintsInput" maxlength="250"></textarea>
    <textarea id="memoryNotesInput" maxlength="800"></textarea>
    <button id="saveMemoryBtn" type="button">Save memory</button>
    <button id="clearMemoryBtn" type="button">Clear</button>
    <p id="memoryStatus"></p>
  </section>
  <button id="editSystemPromptBtn" type="button">Settings</button>
`;

// Was der Server tatsaechlich liefert: sanitize_profile() setzt schema_version.
function storedProfile(enabled) {
  return {
    schema_version: 2,
    enabled,
    role: "Anaesthetist",
    focus: "",
    style: "",
    constraints: "",
    notes: "A note the user maintains."
  };
}

function boot() {
  const calls = [];
  const { window, document } = loadScripts(["static/js/user-memory.js"], {
    body: SETTINGS,
    before(win) {
      win.auth = { currentUser: { uid: "uid-1", getIdToken: async () => "token" } };
      win.fetch = async (url, options = {}) => {
        const body = options.body ? JSON.parse(options.body) : null;
        calls.push({ url, method: options.method, body });
        // Der Server spiegelt nur bekannte Felder zurueck; unbekannte lehnt er
        // mit 422 ab. Genau dieses Verhalten bildet der Stub nach.
        if (options.method === "PUT") {
          const unknown = Object.keys(body).filter(
            key => !["enabled", "role", "focus", "style", "constraints", "notes"].includes(key)
          );
          if (unknown.length) {
            return {
              ok: false,
              status: 422,
              json: async () => ({
                detail: unknown.map(key => ({ type: "extra_forbidden", loc: ["body", key] }))
              })
            };
          }
          return {
            ok: true,
            status: 200,
            json: async () => ({ memory: storedProfile(body.enabled), limits: { notes_chars: 800 } })
          };
        }
        return {
          ok: true,
          status: 200,
          json: async () => ({ memory: storedProfile(true), limits: { notes_chars: 800 } })
        };
      };
    }
  });
  return { window, document, calls };
}

async function settle() {
  for (let i = 0; i < 10; i += 1) await Promise.resolve();
}

describe("memory switch", () => {
  let ctx;
  beforeEach(async () => {
    ctx = boot();
    // jsdom steht beim Einfuegen der Skripte noch auf readyState "loading";
    // das Modul bindet sich erst mit DOMContentLoaded, genau wie im Browser.
    ctx.document.dispatchEvent(new ctx.window.Event("DOMContentLoaded"));
    ctx.document.getElementById("editSystemPromptBtn").click();
    await settle();
  });

  it("loads the stored profile into the form", () => {
    expect(ctx.document.getElementById("memoryEnabledSwitch").checked).toBe(true);
    expect(ctx.document.getElementById("memoryRoleInput").value).toBe("Anaesthetist");
  });

  it("turns memory off and stays off", async () => {
    const box = ctx.document.getElementById("memoryEnabledSwitch");
    box.checked = false;
    box.dispatchEvent(new ctx.window.Event("change"));
    await settle();

    const put = ctx.calls.find(call => call.method === "PUT");
    expect(put).toBeTruthy();
    expect(put.body.enabled).toBe(false);
    expect(Object.keys(put.body).sort()).toEqual(
      ["constraints", "enabled", "focus", "notes", "role", "style"]
    );
    expect(box.checked).toBe(false);
  });

  it("keeps the saved text when only the switch is written", async () => {
    const box = ctx.document.getElementById("memoryEnabledSwitch");
    box.checked = false;
    box.dispatchEvent(new ctx.window.Event("change"));
    await settle();

    const put = ctx.calls.find(call => call.method === "PUT");
    expect(put.body.role).toBe("Anaesthetist");
    expect(put.body.notes).toBe("A note the user maintains.");
  });
});
