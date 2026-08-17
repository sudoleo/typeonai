#!/usr/bin/env node
/**
 * Frontend build for the classic-script app in static/.
 *
 * Reads the load order from static/js/bundles.json, concatenates each group,
 * minifies it, and writes content-hashed files to static/dist/ plus a manifest
 * that app/core/assets.py renders the tags from.
 *
 * Concatenation -- not module bundling -- is deliberate. These files share one
 * global scope and talk to each other through ~85 window.* contracts; wrapping
 * them in module scopes would silently break every implicit global. Joining
 * them in the documented order reproduces exactly what the browser does today
 * with separate <script> tags.
 *
 * Usage:
 *   node scripts/build_frontend.mjs           build into static/dist
 *   node scripts/build_frontend.mjs --check   verify dist matches the sources
 */

import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import * as esbuild from "esbuild";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const DIST = path.join(ROOT, "static", "dist");
const BUNDLES = path.join(ROOT, "static", "js", "bundles.json");
const MANIFEST = path.join(DIST, "manifest.json");
const BUILD_SCRIPT = "scripts/build_frontend.mjs";

const CHECK_ONLY = process.argv.includes("--check");

const TARGET = "es2020";

function hash(content) {
  return createHash("sha256").update(content).digest("hex").slice(0, 12);
}

function entryPath(entry) {
  return typeof entry === "string" ? entry : entry.file;
}

async function read(relative) {
  const absolute = path.join(ROOT, relative);
  if (!existsSync(absolute)) {
    throw new Error(`bundles.json lists a file that does not exist: ${relative}`);
  }
  return fs.readFile(absolute, "utf8");
}

async function readBytes(relative) {
  const absolute = path.join(ROOT, relative);
  if (!existsSync(absolute)) {
    throw new Error(`build input does not exist: ${relative}`);
  }
  return fs.readFile(absolute);
}

/**
 * Concatenate classic scripts.
 *
 * The `;` between files is an ASI guard: a file ending without a semicolon
 * followed by one starting with `(` or `[` would otherwise be parsed as a call
 * or index expression across the seam.
 */
async function buildClassicGroup(group) {
  const parts = [];
  for (const entry of group.files) {
    const relative = entryPath(entry);
    parts.push(`/* ${relative} */\n${await read(relative)}\n;`);
  }
  const source = parts.join("\n");

  // No `format` and no `bundle`: esbuild then treats the input as a classic
  // script and leaves top-level names alone, which is what keeps the window.*
  // contracts working. keepNames guards code that reads fn.name.
  const result = await esbuild.transform(source, {
    minify: true,
    keepNames: true,
    target: TARGET,
    legalComments: "none",
  });
  return {
    code: result.code,
    inputs: group.files.map(entryPath),
  };
}

/**
 * Bundle an ES module entry (firebase.js, demo.js).
 *
 * CDN imports stay external -- the Firebase SDK must keep coming from gstatic.
 * Root-absolute /static/... specifiers are resolved against the project root,
 * which is how the browser reads them but not how node/esbuild would.
 */
const staticResolver = {
  name: "static-root",
  setup(build) {
    build.onResolve({ filter: /^https?:\/\// }, (args) => ({
      path: args.path,
      external: true,
    }));
    build.onResolve({ filter: /^\/static\// }, (args) => ({
      path: path.join(ROOT, args.path.split("?")[0]),
    }));
  },
};

async function buildModuleGroup(group) {
  const result = await esbuild.build({
    absWorkingDir: ROOT,
    entryPoints: [group.entry],
    bundle: true,
    write: false,
    format: "esm",
    minify: true,
    keepNames: true,
    target: TARGET,
    legalComments: "none",
    metafile: true,
    plugins: [staticResolver],
  });
  return {
    code: result.outputFiles[0].text,
    // External CDN modules are absent from the metafile. Normalize the local
    // paths so Python can recompute this exact input set on every platform.
    inputs: Object.keys(result.metafile.inputs).map((input) =>
      path.relative(ROOT, path.resolve(ROOT, input)).replace(/\\/g, "/")
    ),
  };
}

/**
 * Inline the @import chain of style.css in cascade order.
 *
 * esbuild's own CSS bundler would rewrite url() references and copy the
 * referenced assets; inlining by hand keeps the emitted file a sibling of
 * static/css/, so every ../fonts and ../icons path still resolves untouched.
 */
const IMPORT_RE = /@import\s+url\(\s*['"]([^'"]+)['"]\s*\)\s*;/g;

async function inlineCss(relative, seen = new Set()) {
  const normalized = path.normalize(relative).replace(/\\/g, "/");
  if (seen.has(normalized)) return "";
  seen.add(normalized);

  const source = await read(normalized);
  const dir = path.posix.dirname(normalized);

  let out = "";
  let cursor = 0;
  for (const match of source.matchAll(IMPORT_RE)) {
    out += source.slice(cursor, match.index);
    const target = path.posix.normalize(path.posix.join(dir, match[1].split("?")[0]));
    out += await inlineCss(target, seen);
    cursor = match.index + match[0].length;
  }
  out += source.slice(cursor);
  return out;
}

async function buildStyle(style) {
  const source = await inlineCss(style.entry);
  if (IMPORT_RE.test(source)) {
    throw new Error(`unresolved @import left in ${style.entry}`);
  }
  const result = await esbuild.transform(source, {
    loader: "css",
    minify: true,
    target: TARGET,
    legalComments: "none",
  });
  return result.code;
}

/**
 * Fingerprint of every file that feeds the build.
 *
 * Stored in the manifest so the Python suite can tell -- without node -- that
 * someone edited a source file and forgot to rebuild. That is the same failure
 * the manual ?v= marks had; it must not come back through the back door.
 */
async function sourceFingerprint(inputFiles) {
  // bundles.json is itself executable configuration: changing defer/module
  // metadata can alter the rendered page without changing any source file.
  // The build script is included too, so a build-behaviour fix cannot leave a
  // previous artifact looking current to the Python-only deployment check.
  const files = [...new Set([
    "static/js/bundles.json",
    BUILD_SCRIPT,
    "package-lock.json",
    ...inputFiles,
  ])].sort();
  const digest = createHash("sha256");
  for (const relative of files) {
    digest.update(relative);
    digest.update("\0");
    digest.update(await readBytes(relative));
    digest.update("\0");
  }
  return {
    hash: digest.digest("hex").slice(0, 12),
    inputs: files,
  };
}

async function build() {
  const config = JSON.parse(await fs.readFile(BUNDLES, "utf8"));
  const outputs = new Map(); // filename -> content
  const manifest = {
    sources: "",
    inputs: [],
    scripts: [],
    styles: {},
  };
  const sourceInputs = [];

  for (const group of config.groups) {
    const built =
      group.kind === "module"
        ? await buildModuleGroup(group)
        : await buildClassicGroup(group);
    const code = built.code;
    sourceInputs.push(...built.inputs);
    const filename = `${group.name}.${hash(code)}.js`;
    outputs.set(filename, code);
    manifest.scripts.push({
      name: group.name,
      src: `/static/dist/${filename}`,
      module: group.kind === "module",
      defer: group.kind === "module" ? false : Boolean(group.defer),
    });
  }

  for (const style of config.styles) {
    const code = await buildStyle(style);
    sourceInputs.push(style.entry);
    const filename = `${style.name}.${hash(code)}.css`;
    outputs.set(filename, code);
    manifest.styles[style.name] = `/static/dist/${filename}`;
  }

  // Source mode hashes the complete CSS directory because style.css is an
  // @import aggregator. Keep the committed-build guard equally conservative.
  for (const sheet of (await fs.readdir(path.join(ROOT, "static", "css"))).sort()) {
    if (sheet.endsWith(".css")) sourceInputs.push(`static/css/${sheet}`);
  }
  const fingerprint = await sourceFingerprint(sourceInputs);
  manifest.sources = fingerprint.hash;
  manifest.inputs = fingerprint.inputs;

  const manifestJson = `${JSON.stringify(manifest, null, 2)}\n`;

  if (CHECK_ONLY) {
    return verify(outputs, manifestJson);
  }

  await fs.mkdir(DIST, { recursive: true });
  for (const stale of await fs.readdir(DIST).catch(() => [])) {
    if (!outputs.has(stale) && stale !== "manifest.json") {
      await fs.rm(path.join(DIST, stale), { force: true });
    }
  }
  for (const [filename, content] of outputs) {
    await fs.writeFile(path.join(DIST, filename), content, "utf8");
  }
  await fs.writeFile(MANIFEST, manifestJson, "utf8");

  report(outputs);
  return 0;
}

async function verify(outputs, manifestJson) {
  const current = await fs.readFile(MANIFEST, "utf8").catch(() => null);
  if (current !== manifestJson) {
    console.error("static/dist is stale. Run: npm run build");
    return 1;
  }
  for (const [filename, content] of outputs) {
    const onDisk = await fs.readFile(path.join(DIST, filename), "utf8").catch(() => null);
    if (onDisk !== content) {
      console.error(`static/dist/${filename} is stale. Run: npm run build`);
      return 1;
    }
  }
  console.log("static/dist is up to date.");
  return 0;
}

function report(outputs) {
  let total = 0;
  for (const [filename, content] of outputs) {
    const size = Buffer.byteLength(content);
    total += size;
    console.log(`  ${filename.padEnd(28)} ${(size / 1024).toFixed(1)} KB`);
  }
  console.log(`  ${"total".padEnd(28)} ${(total / 1024).toFixed(1)} KB`);
}

process.exit(await build());
