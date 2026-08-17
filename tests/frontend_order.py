"""Load-order helper for the frontend source contracts.

Script order used to be asserted by comparing ``str.index()`` positions inside
templates/index.html. The template no longer lists the files -- the order lives
in static/js/bundles.json, which is what both the build and the renderer read.
Asserting against that file checks the actual contract instead of the byte
offsets of a rendering detail.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLES = ROOT / "static" / "js" / "bundles.json"


def script_order() -> list[str]:
    """Every local script of /app, in the order the browser executes it."""

    config = json.loads(BUNDLES.read_text(encoding="utf-8"))
    order: list[str] = []
    for group in config["groups"]:
        if group.get("kind") == "module":
            order.append(group["entry"])
            continue
        for entry in group["files"]:
            order.append(entry if isinstance(entry, str) else entry["file"])
    return order


def position(script: str) -> int:
    """Index of ``script`` in load order. Accepts a bare filename or a path."""

    order = script_order()
    for index, relative in enumerate(order):
        if relative == script or Path(relative).name == script:
            return index
    raise AssertionError(f"{script} is not part of the /app bundle: {order}")


def loads_before(first: str, second: str) -> bool:
    return position(first) < position(second)


def group_of(script: str) -> str:
    config = json.loads(BUNDLES.read_text(encoding="utf-8"))
    for group in config["groups"]:
        if group.get("kind") == "module":
            if Path(group["entry"]).name == script or group["entry"] == script:
                return group["name"]
            continue
        for entry in group["files"]:
            relative = entry if isinstance(entry, str) else entry["file"]
            if relative == script or Path(relative).name == script:
                return group["name"]
    raise AssertionError(f"{script} is not part of the /app bundle")
