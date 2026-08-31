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
        end_index = citation.get("end_index")
        try:
            end = int(end_index)
        except (TypeError, ValueError):
            end = len(text)
        end = max(0, min(len(text), end))
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
        })
    return insert_source_tags(text, citations, provider)
