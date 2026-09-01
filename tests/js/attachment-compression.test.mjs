/**
 * Ein Anhang geht an bis zu sechs Familien GLEICHZEITIG raus, base64 legt ein
 * Drittel obendrauf: ein 4-MB-Foto ist ein 32-MB-Ausgang plus Bildtokens in
 * jedem der sechs Prompts. Deshalb verkleinert der Client jedes grosse Bild,
 * BEVOR es kodiert wird -- und nur was der Composer wirklich verschickt, steht
 * anschliessend im Chip.
 *
 * Getestet wird der echte Weg (Dateifeld -> addFiles -> pendingAttachments)
 * gegen gestellte Browser-Bausteine: jsdom hat weder Bilddekoder noch Canvas.
 */

import { describe, expect, it } from "vitest";

import { loadScripts } from "./helpers/appWindow.mjs";

const BODY = `
<div class="chat-input-container">
  <div id="attachmentBar" class="attachment-bar" hidden></div>
  <textarea id="questionInput"></textarea>
  <button id="attachTrigger"></button>
  <div id="attachMenu" hidden><button id="attachUploadOption"></button></div>
  <input id="attachFileInput" type="file">
</div>
<label class="switch deep-switch"><input type="checkbox" id="deepSearchToggle"></label>
`;

// Groesse der Blobs, die das gestellte Canvas ausspuckt: ein verkleinertes
// Bild ist immer ein Bruchteil des Originals.
const ENCODED_BYTES = 140 * 1024;

function boot({ withCanvas = true } = {}) {
  const alerts = [];
  const drawn = [];

  const harness = loadScripts(
    ["static/js/app-state.js", "static/js/app-core.js", "static/js/attachments.js"],
    {
      body: BODY,
      before(window) {
        window.matchMedia = () => ({
          matches: false,
          addEventListener() {}, removeEventListener() {},
          addListener() {}, removeListener() {}
        });
        window.alert = (message) => alerts.push(String(message));
        window.URL.createObjectURL = () => "blob:image";
        window.URL.revokeObjectURL = () => {};

        // Ein Bild, dessen Masse der Test vorgibt. jsdom laedt nichts, also
        // meldet der Stub den Ladeerfolg selbst.
        window.Image = class {
          set src(value) {
            this.naturalWidth = window.__imageSize[0];
            this.naturalHeight = window.__imageSize[1];
            setTimeout(() => this.onload && this.onload(), 0);
          }
        };
        window.__imageSize = [0, 0];

        const createElement = window.document.createElement.bind(window.document);
        window.document.createElement = (tag) => {
          if (String(tag).toLowerCase() !== "canvas") return createElement(tag);
          const canvas = { width: 0, height: 0 };
          canvas.getContext = () => ({
            fillStyle: "",
            fillRect() {},
            drawImage() { drawn.push([canvas.width, canvas.height]); }
          });
          if (withCanvas) {
            canvas.toBlob = (callback, type) => {
              callback(new window.Blob([new Uint8Array(ENCODED_BYTES)], { type }));
            };
          }
          return canvas;
        };
      }
    }
  );

  harness.window.App.state.set("isUserPlus", true, "userTier");
  harness.alerts = alerts;
  harness.drawn = drawn;
  return harness;
}

// Eine Datei mit vorgegebener Groesse, ohne sie wirklich zu belegen: geprueft
// wird `size`, gelesen werden nur die paar echten Bytes.
function fakeFile(window, name, type, size) {
  const file = new window.File([new Uint8Array(8)], name, { type });
  Object.defineProperty(file, "size", { value: size });
  return file;
}

async function pick(harness, file, imageSize) {
  const { window, document } = harness;
  window.__imageSize = imageSize || [0, 0];
  const input = document.getElementById("attachFileInput");
  Object.defineProperty(input, "files", { value: [file], configurable: true });
  input.dispatchEvent(new window.Event("change"));
  // Verkleinern und Kodieren sind beide asynchron.
  for (let i = 0; i < 12; i += 1) await new Promise((resolve) => setTimeout(resolve, 0));
  return window.pendingAttachments;
}

describe("client-side image compression", () => {
  it("shrinks an oversized photo before it is encoded", async () => {
    const harness = boot();
    const attachments = await pick(
      harness,
      fakeFile(harness.window, "photo.png", "image/png", 4 * 1024 * 1024),
      [3000, 2000]
    );

    expect(attachments).toHaveLength(1);
    expect(attachments[0].mime).toBe("image/jpeg");
    // Der Name darf keinen Typ mehr behaupten, den die Datei nicht mehr hat.
    expect(attachments[0].name).toBe("photo.jpg");
    expect(attachments[0].size).toBe(ENCODED_BYTES);
    // Laengste Kante auf 1568, Seitenverhaeltnis erhalten.
    expect(harness.drawn).toEqual([[1568, 1045]]);
  });

  it("leaves a small screenshot untouched", async () => {
    const harness = boot();
    const attachments = await pick(
      harness,
      fakeFile(harness.window, "screenshot.png", "image/png", 20 * 1024),
      [400, 300]
    );

    expect(attachments).toHaveLength(1);
    expect(attachments[0].mime).toBe("image/png");
    expect(attachments[0].name).toBe("screenshot.png");
    expect(harness.drawn).toEqual([]);
  });

  it("still attaches the file when the browser has no canvas", async () => {
    const harness = boot({ withCanvas: false });
    const attachments = await pick(
      harness,
      fakeFile(harness.window, "photo.png", "image/png", 4 * 1024 * 1024),
      [3000, 2000]
    );

    expect(attachments).toHaveLength(1);
    expect(attachments[0].mime).toBe("image/png");
    expect(harness.alerts).toEqual([]);
  });

  it("refuses an image far above the input limit", async () => {
    const harness = boot();
    const attachments = await pick(
      harness,
      fakeFile(harness.window, "raw.png", "image/png", 16 * 1024 * 1024),
      [6000, 4000]
    );

    expect(attachments).toHaveLength(0);
    expect(harness.alerts.join(" ")).toContain("15 MB per file");
  });

  it("keeps the tighter limit for files that cannot be shrunk", async () => {
    const harness = boot();
    const attachments = await pick(
      harness,
      fakeFile(harness.window, "report.pdf", "application/pdf", 6 * 1024 * 1024)
    );

    expect(attachments).toHaveLength(0);
    expect(harness.alerts.join(" ")).toContain("5 MB per file");
  });
});
