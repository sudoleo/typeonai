"""Serverseitiges Markdown-Rendering für öffentliche Share-Seiten.

Die App rendert Markdown clientseitig mit marked + DOMPurify; auf der
öffentlichen Seite passiert beides serverseitig, damit Crawler den vollen
Inhalt sehen und kein unsanitisiertes LLM-HTML ausgeliefert wird:

  1. markdown-it-py mit html=False – rohes HTML im LLM-Text wird escaped,
     nie interpretiert.
  2. nh3 (ammonia) als zweite Schicht über dem erzeugten HTML; Links nur
     mit http(s)/mailto-Schema und rel="nofollow noopener noreferrer".

[S1]-Quellen-Tags werden vor dem Rendern zu Links auf die Anker der
Quellenliste (#src-1) umgeschrieben – das Python-Pendant zu
linkifySourceTags() in index.html.
"""

import re

import nh3
from markdown_it import MarkdownIt

_MD = (
    MarkdownIt("commonmark", {"html": False, "linkify": False, "typographer": False})
    .enable("table")
    .enable("strikethrough")
)

_URL_SCHEMES = {"http", "https", "mailto"}
_LINK_REL = "nofollow noopener noreferrer"

# Läufe wie "[S1]", "[S1, S2]" oder "[1, 3]" (gleiches Muster wie im Frontend).
_SOURCE_RUN_RE = re.compile(r"\[((?:S?\d+)(?:\s*,\s*S?\d+)*)\]", re.IGNORECASE)
# Fenced-Code-Blöcke und Inline-Code: dort keine Quellen-Tags ersetzen.
_CODE_SEGMENT_RE = re.compile(r"(```.*?(?:```|$)|`[^`\n]*`)", re.DOTALL)


def _known_source_numbers(sources):
    numbers = set()
    for source in sources or []:
        match = re.match(r"^S?(\d+)$", str(source.get("id") or ""), re.IGNORECASE)
        if match:
            numbers.add(match.group(1).lstrip("0") or "0")
    return numbers


def _link_source_tags(md_text, known_numbers):
    def replace_run(match):
        tokens = [token.strip() for token in match.group(1).split(",")]
        numbers = []
        for token in tokens:
            number = token[1:] if token[:1] in ("s", "S") else token
            number = number.lstrip("0") or "0"
            if number not in known_numbers:
                return match.group(0)  # unbekannte Referenz: Lauf unverändert lassen
            numbers.append(number)
        return ", ".join("[\\[S%s\\]](#src-%s)" % (n, n) for n in numbers)

    def replace_outside_code(segment):
        return _SOURCE_RUN_RE.sub(replace_run, segment)

    parts = _CODE_SEGMENT_RE.split(md_text)
    return "".join(
        part if index % 2 else replace_outside_code(part)
        for index, part in enumerate(parts)
    )


def render_public_markdown(md_text, sources=None):
    """Markdown → sanitisiertes HTML (sicher für `| safe` im Template)."""
    text = str(md_text or "")
    known_numbers = _known_source_numbers(sources)
    if known_numbers:
        text = _link_source_tags(text, known_numbers)
    html = _MD.render(text)
    return nh3.clean(html, url_schemes=_URL_SCHEMES, link_rel=_LINK_REL)


def markdown_to_plaintext(md_text, limit=None):
    """Reiner Text (für Meta-Descriptions/Anrisse): rendert und strippt alle Tags."""
    html = _MD.render(str(md_text or ""))
    text = nh3.clean(html, tags=set())
    text = re.sub(r"\s+", " ", text).strip()
    if limit is not None and len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text
