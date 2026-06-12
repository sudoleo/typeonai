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
linkifySourceTags() in index.html. Als Link-Text dient wie in der App der
erkennbare Site-Name der Quelle (z. B. "wikipedia") statt des technischen
Markers "[S1]"; nur ohne verwertbare URL bleibt "S1" als Fallback.
"""

import re
from urllib.parse import urlsplit

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

# Second-Level-Suffixe wie in getSourceSiteName() im Frontend (z. B. bbc.co.uk).
_SLD_SUFFIXES = {"co", "com", "org", "net", "ac", "gov"}
# Nur harmlose Zeichen im Markdown-Link-Text (keine [, ], ( , ) usw.).
_LABEL_SAFE_RE = re.compile(r"[^A-Za-z0-9._äöü-]+")


def source_site_name(url):
    """Kurzer, erkennbarer Site-Name einer Quelle ("en.wikipedia.org" → "wikipedia").

    Python-Pendant zu getSourceSiteName() in index.html, damit Inline-
    Citations auf der Share-Seite genauso aussehen wie im Consensus-Bereich.
    """
    try:
        hostname = (urlsplit(str(url or "")).hostname or "").lower()
    except ValueError:
        return ""
    hostname = re.sub(r"^(www|m|amp)\.", "", hostname)
    parts = [p for p in hostname.split(".") if p]
    if len(parts) < 2:
        return parts[0] if parts else ""
    if len(parts) >= 3 and parts[-2] in _SLD_SUFFIXES:
        return parts[-3]
    return parts[-2]


def _source_labels(sources):
    """Map Quellen-Nummer → (Anzeige-Label) für die Inline-Citations."""
    labels = {}
    for source in sources or []:
        match = re.match(r"^S?(\d+)$", str(source.get("id") or ""), re.IGNORECASE)
        if not match:
            continue
        number = match.group(1).lstrip("0") or "0"
        label = source_site_name(source.get("url"))
        labels[number] = _LABEL_SAFE_RE.sub("", label) if label else ""
    return labels


def _link_source_tags(md_text, labels):
    def replace_run(match):
        tokens = [token.strip() for token in match.group(1).split(",")]
        numbers = []
        for token in tokens:
            number = token[1:] if token[:1] in ("s", "S") else token
            number = number.lstrip("0") or "0"
            if number not in labels:
                return match.group(0)  # unbekannte Referenz: Lauf unverändert lassen
            numbers.append(number)
        return " ".join(
            "[%s](#src-%s)" % (labels[n] or ("S%s" % n), n) for n in numbers
        )

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
    labels = _source_labels(sources)
    if labels:
        text = _link_source_tags(text, labels)
    html = _MD.render(text)
    return nh3.clean(html, url_schemes=_URL_SCHEMES, link_rel=_LINK_REL)


def markdown_to_plaintext(md_text, limit=None):
    """Reiner Text (für Meta-Descriptions/Anrisse): rendert und strippt alle Tags.

    Quellen-Marker wie "[S1]" werden entfernt – in einem SEO-Snippet sind sie
    nur technisches Rauschen.
    """
    text = _SOURCE_RUN_RE.sub("", str(md_text or ""))
    html = _MD.render(text)
    text = nh3.clean(html, tags=set())
    text = re.sub(r"\s+", " ", text).strip()
    if limit is not None and len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text
