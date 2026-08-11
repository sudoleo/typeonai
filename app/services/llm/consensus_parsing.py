"""Provider-agnostic JSON extraction and repair for Consensus judges."""

from __future__ import annotations

import json
import logging
import re


def extract_json_object(raw: str, *, with_repair_flag: bool = False):
    result = extract_json_object_inner(raw)
    return result if with_repair_flag else result[0]


def extract_json_object_inner(raw: str):
    text = str(raw or "").strip()
    if not text:
        return None, False
    candidates = []
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        candidates.append(fence.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed, False
    if start != -1:
        repaired = repair_truncated_json(text[start:])
        if repaired is not None:
            return repaired, True
    return None, False


def close_open_json(text: str):
    stack = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack:
                return None
            open_char = stack.pop()
            if (open_char == "{") != (char == "}"):
                return None
    repaired = text
    if escaped:
        repaired = repaired[:-1]
    if in_string:
        repaired += '"'
    stripped = repaired.rstrip()
    if stripped.endswith(","):
        stripped = stripped[:-1].rstrip()
    if stripped.endswith(":"):
        return None
    return stripped + "".join("}" if char == "{" else "]" for char in reversed(stack))


def repair_truncated_json(fragment: str):
    text = fragment
    for _ in range(40):
        repaired = close_open_json(text)
        if repaired is not None:
            try:
                parsed = json.loads(repaired)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict):
                logging.info("Differences engine output was truncated; repaired JSON tail.")
                return parsed
        cut = max(text.rfind(","), text.rfind("{"), text.rfind("["))
        if cut <= 0:
            return None
        text = text[:cut]
    return None

