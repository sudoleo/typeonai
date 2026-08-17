"""Content-hashed asset URLs for the /app frontend.

Two modes, one source of truth (``static/js/bundles.json``):

* **built** -- ``static/dist/manifest.json`` exists: the templates get the
  minified, content-hashed bundles produced by ``npm run build``.
* **source** -- no build output, or ``FRONTEND_DEV=1``: the templates get the
  individual files in the same order, each with a ``?v=<content hash>`` query.
  Editing a file changes its hash, so a stale cached copy is impossible.

Either way no version mark is maintained by hand. The 30 ``?v=20260817-...``
strings this replaces had to be bumped manually on every edit, and forgetting
one served stale JS or CSS to returning users.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = ROOT / "static"
BUNDLES_FILE = STATIC_DIR / "js" / "bundles.json"
MANIFEST_FILE = STATIC_DIR / "dist" / "manifest.json"

_HASH_LENGTH = 12


@dataclass(frozen=True)
class ScriptTag:
    src: str
    # Group name from bundles.json. index.html renders the "head" group above
    # the CDN libraries and everything else below them, so the relative order
    # of local and third-party scripts is preserved in both modes.
    group: str
    defer: bool = False
    module: bool = False


@dataclass(frozen=True)
class FrontendAssets:
    scripts: tuple[ScriptTag, ...]
    style: str
    built: bool


# Cache keyed by the inputs' (mtime_ns, size) so a source edit invalidates it
# without a restart, while a warm process never re-hashes on every request.
_cache: dict[str, tuple[object, object]] = {}


def _fingerprint(paths: list[Path]) -> tuple:
    stamps = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            stamps.append((str(path), None))
        else:
            stamps.append((str(path), stat.st_mtime_ns, stat.st_size))
    return tuple(stamps)


def _memoize(key: str, paths: list[Path], build):
    stamp = _fingerprint(paths)
    cached = _cache.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    value = build()
    _cache[key] = (stamp, value)
    return value


def _content_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        try:
            digest.update(path.read_bytes())
        except OSError:
            # A missing file must still produce a stable, distinct hash rather
            # than crash the page render.
            digest.update(b"\0missing\0")
        digest.update(path.name.encode("utf-8"))
    return digest.hexdigest()[:_HASH_LENGTH]


def dev_mode() -> bool:
    return os.environ.get("FRONTEND_DEV") == "1"


def _load_bundles() -> dict:
    return _memoize(
        "bundles",
        [BUNDLES_FILE],
        lambda: json.loads(BUNDLES_FILE.read_text(encoding="utf-8")),
    )


def _group_files(group: dict) -> list[str]:
    files = []
    for entry in group.get("files", []):
        files.append(entry if isinstance(entry, str) else entry["file"])
    return files


def _source_assets() -> FrontendAssets:
    config = _load_bundles()
    scripts: list[ScriptTag] = []

    for group in config["groups"]:
        name = group["name"]
        if group.get("kind") == "module":
            relative = group["entry"]
            scripts.append(
                ScriptTag(
                    src=asset_url(f"/{relative}"), group=name, defer=False, module=True
                )
            )
            continue
        defer = bool(group.get("defer"))
        for relative in _group_files(group):
            scripts.append(
                ScriptTag(
                    src=asset_url(f"/{relative}"), group=name, defer=defer, module=False
                )
            )

    return FrontendAssets(scripts=tuple(scripts), style=_source_style_url(), built=False)


def _source_style_url() -> str:
    """Hash style.css together with every file it can pull in.

    style.css is an ``@import`` aggregator: its own bytes never change when one
    of the imported component sheets is edited. Hashing the whole directory is
    coarse -- any CSS edit busts the whole sheet -- but in source mode that is
    exactly the intent.
    """

    entry = STATIC_DIR / "style.css"
    sheets = sorted((STATIC_DIR / "css").glob("*.css"))
    paths = [entry, *sheets]
    digest = _memoize("style-sources", paths, lambda: _content_hash(paths))
    return f"/static/style.css?v={digest}"


def _built_assets() -> FrontendAssets:
    manifest = _memoize(
        "manifest",
        [MANIFEST_FILE],
        lambda: json.loads(MANIFEST_FILE.read_text(encoding="utf-8")),
    )
    scripts = tuple(
        ScriptTag(
            src=item["src"],
            group=item["name"],
            defer=bool(item.get("defer")),
            module=bool(item.get("module")),
        )
        for item in manifest["scripts"]
    )
    return FrontendAssets(scripts=scripts, style=manifest["styles"]["app"], built=True)


def frontend_assets() -> FrontendAssets:
    """Scripts and stylesheet for index.html, in load order."""

    if not dev_mode() and MANIFEST_FILE.exists():
        return _built_assets()
    return _source_assets()


def source_fingerprint() -> str:
    """Hash of every file the build consumes, in the build's own order.

    Must stay byte-identical to ``sourceFingerprint`` in
    scripts/build_frontend.mjs -- comparing it against the value recorded in the
    manifest is how a forgotten ``npm run build`` is caught without node.
    """

    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    relatives = manifest.get("inputs")
    if not isinstance(relatives, list) or not all(
        isinstance(relative, str) for relative in relatives
    ):
        # Compatibility with a pre-input-list manifest: enough to report it as
        # stale and direct the developer to rebuild with the current script.
        config = _load_bundles()
        relatives = ["static/js/bundles.json", "scripts/build_frontend.mjs"]
        if (ROOT / "package-lock.json").is_file():
            relatives.append("package-lock.json")
        for group in config["groups"]:
            if group.get("kind") == "module":
                relatives.append(group["entry"])
            else:
                relatives.extend(_group_files(group))
        for style in config["styles"]:
            relatives.append(style["entry"])
        relatives.extend(
            f"static/css/{sheet.name}"
            for sheet in sorted((STATIC_DIR / "css").glob("*.css"))
        )
        relatives = sorted(set(relatives))

    digest = hashlib.sha256()
    for relative in relatives:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update((ROOT / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:_HASH_LENGTH]


def build_is_stale() -> bool:
    """True when static/dist no longer matches the sources it was built from."""

    if not MANIFEST_FILE.exists():
        return False
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    return manifest.get("sources") != source_fingerprint()


def register_asset_globals(templates) -> None:
    """Expose the /app asset helpers to its Jinja2Templates instance."""

    templates.env.globals["asset_url"] = asset_url
    templates.env.globals["frontend_assets"] = frontend_assets


def asset_url(path: str) -> str:
    """Append a content hash to a ``/static/...`` URL.

    Used for one-off assets (favicons, images) that are not part of a bundle.
    An unknown path is returned unchanged rather than raising -- a missing
    favicon must not take the page down.
    """

    clean = path.split("?", 1)[0]
    if not clean.startswith("/static/"):
        return path
    target = ROOT / clean.lstrip("/")
    if not target.is_file():
        return clean
    digest = _memoize(f"asset:{clean}", [target], lambda: _content_hash([target]))
    return f"{clean}?v={digest}"
