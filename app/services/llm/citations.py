from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple
from urllib.parse import urlsplit, urlunsplit


Source = Dict[str, Any]
LLMResult = Dict[str, Any]


def make_llm_result(text: str, sources: Iterable[Source] | None = None) -> LLMResult:
    return {
        "text": coerce_text(text).strip(),
        "sources": list(sources or []),
    }


def coerce_text(value: Any) -> str:
    """Normalisiert Provider-Content-Blöcke zu sichtbarem Antworttext.

    Neue Modellversionen liefern Text je nach SDK als String, Liste von Blöcken
    oder verschachteltes Objekt. Niemals das Objekt selbst stringifizieren:
    im Browser wuerde daraus `[object Object]`.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "".join(coerce_text(item) for item in value)
    if isinstance(value, dict):
        for key in ("text", "output_text", "content", "delta", "message", "error"):
            if key in value:
                text = coerce_text(value.get(key))
                if text:
                    return text
        return ""
    plain = to_plain(value)
    if plain is not value:
        return coerce_text(plain)
    return str(value) if isinstance(value, (int, float, bool)) else ""


def result_text(result: Any) -> str:
    if isinstance(result, dict):
        return coerce_text(result.get("text") or result.get("response"))
    return coerce_text(result)


def result_sources(result: Any) -> List[Source]:
    if isinstance(result, dict):
        sources = result.get("sources")
        return sources if isinstance(sources, list) else []
    return []


def source_response(result: Any, **extra: Any) -> Dict[str, Any]:
    if isinstance(result, dict) and result.get("error"):
        payload = {
            "error": coerce_text(result.get("error")) or "This model could not complete the request.",
            "error_code": str(result.get("error_code") or "provider_request_failed"),
            "response": "",
            "sources": result_sources(result),
        }
        payload.update(extra)
        return payload

    payload = {
        "response": result_text(result),
        "sources": result_sources(result),
    }
    payload.update(extra)
    return payload


def to_plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {k: to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return to_plain(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            return to_plain(value.to_dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return {
            k: to_plain(v)
            for k, v in vars(value).items()
            if not k.startswith("_")
        }
    return value


def normalize_url(url: str | None) -> str:
    if not url:
        return ""
    try:
        parts = urlsplit(str(url).strip())
        netloc = parts.netloc.lower()
        path = parts.path.rstrip("/") or parts.path
        return urlunsplit((parts.scheme.lower(), netloc, path, parts.query, ""))
    except Exception:
        return str(url).strip()


def _source_key(url: str | None, title: str | None = None) -> str:
    return normalize_url(url) or (title or "").strip().lower()


# OpenRouter legt in `url_citation.content` den ausgelesenen Seitentext ab,
# nicht den zitierten Satz: gemessen bis ~4.000 Zeichen je Quelle und
# 5.000-10.000 je Antwort. Ungekuerzt stand damit die halbe Seite in der
# Quellenliste -- der Hover-Teaser klemmt per CSS auf drei Zeilen, die Liste
# unter der Antwort tut das nicht -- und jeder SSE-Frame trug sie mit. Ein
# Teaser braucht einen Satz, keine Seite; der Rest steht hinter dem Link.
SOURCE_SNIPPET_MAX_CHARS = 300


def clip_snippet(value: Any) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= SOURCE_SNIPPET_MAX_CHARS:
        return text
    cut = text[:SOURCE_SNIPPET_MAX_CHARS]
    # An der letzten Wortgrenze schneiden, solange dabei nicht mehr als ein
    # Fuenftel des Teasers verloren geht.
    space = cut.rfind(" ")
    if space >= SOURCE_SNIPPET_MAX_CHARS * 0.8:
        cut = cut[:space]
    return cut.rstrip(" ,;:.-") + "…"


def _fallback_title(url: str | None, title: str | None = None) -> str:
    if title:
        return str(title).strip()
    if not url:
        return "Source"
    try:
        host = urlsplit(url).netloc
        return host or url
    except Exception:
        return url


def _ensure_source(
    sources: List[Source],
    index_by_key: Dict[str, int],
    *,
    url: str | None,
    title: str | None = None,
    snippet: str | None = None,
    provider: str | None = None,
) -> str:
    key = _source_key(url, title)
    if not key:
        key = f"source:{len(sources) + 1}"
    if key in index_by_key:
        return f"S{index_by_key[key]}"

    source_id = f"S{len(sources) + 1}"
    index_by_key[key] = len(sources) + 1
    source: Source = {
        "id": source_id,
        "title": _fallback_title(url, title),
        "url": url or "",
    }
    clipped = clip_snippet(snippet) if snippet else ""
    if clipped:
        source["snippet"] = clipped
        source["extract"] = clipped
    if provider:
        source["provider"] = provider
    sources.append(source)
    return source_id


_TERMINAL_CITATION_BOUNDARY_RE = re.compile(
    r"[.!?\u2026](?:[\"'\u201d\u2019)\]}]+)?(?=\s|$)|\n"
)


def _integer_index(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _fallback_citation_end(text: str, hint: Any) -> int:
    """Place an unpositioned citation after the nearest complete claim.

    Some OpenRouter native-search adapters emit a zero-width annotation
    (``start_index == end_index == 0``).  In a stream, the caller records how
    much answer text had arrived with that annotation.  That point is a useful
    anchor, but it can fall in the middle of a streamed token, so advance to
    the next sentence or line boundary.  Without a usable stream hint the only
    honest placement is the end of the answer -- never its beginning.
    """
    if not text:
        return 0

    position = _integer_index(hint)
    if position is None or position <= 0:
        return len(text)
    position = min(position, len(text))
    if position >= len(text):
        return len(text)

    prefix = text[:position].rstrip()
    if prefix and prefix[-1] in ".!?\u2026\n":
        return len(prefix)

    boundary = _TERMINAL_CITATION_BOUNDARY_RE.search(text, position)
    return boundary.end() if boundary else len(text)


def _citation_end(text: str, citation: Dict[str, Any]) -> int:
    start = _integer_index(citation.get("start_index"))
    end = _integer_index(citation.get("end_index"))

    # A citation must cover at least one character.  Zero-width ranges are a
    # real OpenRouter/provider failure mode and used to create the row of chips
    # before the first word of the model answer.
    valid_range = (
        end is not None
        and 0 < end <= len(text)
        and (start is None or 0 <= start < end)
    )
    if not valid_range:
        return _fallback_citation_end(text, citation.get("fallback_end_index"))

    # Defensive guard for offsets counted slightly differently by an upstream
    # adapter: never split a visible word with a source chip.
    while end < len(text) and text[end - 1].isalnum() and text[end].isalnum():
        end += 1
    return end


def insert_source_tags(
    text: str,
    citations: Iterable[Dict[str, Any]],
    provider: str,
) -> LLMResult:
    text = text or ""
    sources: List[Source] = []
    index_by_key: Dict[str, int] = {}
    tags_by_end: Dict[int, List[str]] = {}

    for citation in citations or []:
        url = citation.get("url")
        title = citation.get("title")
        snippet = citation.get("snippet") or citation.get("cited_text")
        source_id = _ensure_source(
            sources,
            index_by_key,
            url=url,
            title=title,
            snippet=snippet,
            provider=provider,
        )
        end = _citation_end(text, citation)
        tags_by_end.setdefault(end, [])
        if source_id not in tags_by_end[end]:
            tags_by_end[end].append(source_id)

    for end, source_ids in sorted(tags_by_end.items(), reverse=True):
        tag = "[" + ", ".join(source_ids) + "]"
        prefix = "" if end > 0 and text[end - 1].isspace() else " "
        text = text[:end] + prefix + tag + text[end:]

    return make_llm_result(text, sources)


_MARKDOWN_CITATION_RE = re.compile(r"\[\[?(\d+)\]?\]\((https?://[^)\s]+)\)")


def convert_markdown_citations(text: str, provider: str) -> LLMResult:
    text = text or ""
    sources: List[Source] = []
    index_by_key: Dict[str, int] = {}

    def repl(match: re.Match[str]) -> str:
        url = match.group(2)
        source_id = _ensure_source(
            sources,
            index_by_key,
            url=url,
            title=None,
            snippet=None,
            provider=provider,
        )
        return f"[{source_id}]"

    converted = _MARKDOWN_CITATION_RE.sub(repl, text)
    return make_llm_result(converted, sources)


def parse_openrouter_response(
    text: Any, annotations: Iterable[Dict[str, Any]] | None = None,
    provider: str = "openrouter",
) -> LLMResult:
    """Convert OpenRouter answer text and URL annotations to an ``LLMResult``.

    The caller extracts ``choices[0].message.content`` and
    ``choices[0].message.annotations`` from the chat-completions response. A
    URL annotation has the shape
    ``{"type": "url_citation", "url_citation": {"url", "title",
    "content", "start_index", "end_index"}}``.  Offsets are applied to the
    unmodified answer, then source tags are inserted from right to left so
    earlier offsets remain valid.
    """
    text = coerce_text(text)
    citations: List[Dict[str, Any]] = []
    for annotation in annotations or []:
        if not isinstance(annotation, dict) or annotation.get("type") != "url_citation":
            continue
        citation = annotation.get("url_citation")
        if not isinstance(citation, dict) or not citation.get("url"):
            continue
        citations.append({
            "url": citation.get("url"),
            "title": citation.get("title"),
            "snippet": citation.get("content"),
            "start_index": citation.get("start_index"),
            "end_index": citation.get("end_index"),
            "fallback_end_index": citation.get("_stream_text_end_index"),
        })
    return insert_source_tags(text, citations, provider)
