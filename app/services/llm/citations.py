from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple
from urllib.parse import urlsplit, urlunsplit


Source = Dict[str, Any]
LLMResult = Dict[str, Any]


def make_llm_result(text: str, sources: Iterable[Source] | None = None) -> LLMResult:
    return {
        "text": (text or "").strip(),
        "sources": list(sources or []),
    }


def result_text(result: Any) -> str:
    if isinstance(result, dict):
        return str(result.get("text") or result.get("response") or "")
    return str(result or "")


def result_sources(result: Any) -> List[Source]:
    if isinstance(result, dict):
        sources = result.get("sources")
        return sources if isinstance(sources, list) else []
    return []


def source_response(result: Any, **extra: Any) -> Dict[str, Any]:
    if isinstance(result, dict) and result.get("error"):
        payload = {
            "error": str(result.get("error") or "This model could not complete the request."),
            "error_detail": str(result.get("error_detail") or ""),
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
    if snippet:
        source["snippet"] = str(snippet).strip()
        source["extract"] = str(snippet).strip()
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


def merge_results(primary: LLMResult, secondary: LLMResult) -> LLMResult:
    if result_sources(primary):
        return primary
    if result_sources(secondary):
        return make_llm_result(result_text(primary), result_sources(secondary))
    return primary


def parse_openai_response(data: Dict[str, Any], provider: str = "openai") -> LLMResult:
    text_parts: List[str] = []
    citations: List[Dict[str, Any]] = []
    fallback_citations: List[Dict[str, Any]] = []
    offset = 0

    for item in data.get("output", []) or []:
        if item.get("type") == "web_search_call":
            action = item.get("action") or {}
            for source in action.get("sources") or []:
                url = source.get("url") if isinstance(source, dict) else None
                if url:
                    fallback_citations.append({
                        "url": url,
                        "title": source.get("title") if isinstance(source, dict) else None,
                        "end_index": len("".join(text_parts)) or None,
                    })
            continue
        if item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if content.get("type") not in {"output_text", "text"}:
                continue
            part_text = content.get("text") or ""
            for annotation in content.get("annotations", []) or []:
                if annotation.get("type") in {"url_citation", "citation"}:
                    start = annotation.get("start_index")
                    end = annotation.get("end_index")
                    citations.append({
                        "url": annotation.get("url"),
                        "title": annotation.get("title"),
                        "start_index": (offset + int(start)) if isinstance(start, int) else None,
                        "end_index": (offset + int(end)) if isinstance(end, int) else offset + len(part_text),
                    })
            text_parts.append(part_text)
            offset += len(part_text)

    text = "".join(text_parts) or data.get("output_text") or ""
    for url in data.get("citations") or []:
        if isinstance(url, str):
            fallback_citations.append({"url": url, "end_index": len(text)})
        elif isinstance(url, dict):
            fallback_citations.append({
                "url": url.get("url"),
                "title": url.get("title"),
                "end_index": len(text),
            })
    if not citations:
        citations = fallback_citations
    markdown = convert_markdown_citations(text, provider)
    annotated = insert_source_tags(result_text(markdown), citations, provider)
    return merge_results(annotated, markdown)


def parse_anthropic_response(data: Dict[str, Any]) -> LLMResult:
    text_parts: List[str] = []
    citations: List[Dict[str, Any]] = []
    offset = 0

    for block in data.get("content", []) or []:
        if block.get("type") != "text":
            continue
        part_text = block.get("text") or ""
        block_citations = block.get("citations", []) or []
        if block_citations:
            end = offset + len(part_text)
            for citation in block_citations:
                citations.append({
                    "url": citation.get("url"),
                    "title": citation.get("title"),
                    "snippet": citation.get("cited_text"),
                    "end_index": end,
                })
        text_parts.append(part_text)
        offset += len(part_text)

    return insert_source_tags("".join(text_parts), citations, "anthropic")


def parse_gemini_response(resp: Any, fallback_text: str = "") -> LLMResult:
    data = to_plain(resp) or {}
    text = (getattr(resp, "text", None) or fallback_text or "").strip()
    candidates = data.get("candidates") if isinstance(data, dict) else None
    cand = (candidates or [None])[0]
    if isinstance(cand, dict) and not text:
        parts = (((cand.get("content") or {}).get("parts")) or [])
        text = "".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
    metadata = None
    if isinstance(cand, dict):
        metadata = cand.get("grounding_metadata") or cand.get("groundingMetadata")
    else:
        metadata = getattr(cand, "grounding_metadata", None) or getattr(cand, "groundingMetadata", None)
    metadata = to_plain(metadata) or {}
    chunks = metadata.get("grounding_chunks") or metadata.get("groundingChunks") or []
    supports = metadata.get("grounding_supports") or metadata.get("groundingSupports") or []

    citations: List[Dict[str, Any]] = []
    chunk_data = [to_plain(chunk) for chunk in chunks]

    for support in supports:
        support_data = to_plain(support) or {}
        segment = support_data.get("segment") or {}
        end = segment.get("end_index", segment.get("endIndex"))
        indices = (
            support_data.get("grounding_chunk_indices")
            or support_data.get("groundingChunkIndices")
            or []
        )
        for idx in indices:
            try:
                chunk = chunk_data[int(idx)]
            except (TypeError, ValueError, IndexError):
                continue
            web = chunk.get("web") or {}
            citations.append({
                "url": web.get("uri"),
                "title": web.get("title"),
                "end_index": end,
            })

    return insert_source_tags(text, citations, "gemini")


def parse_mistral_content(content: Any) -> LLMResult:
    if isinstance(content, str):
        return convert_markdown_citations(content, "mistral")

    sources: List[Source] = []
    index_by_key: Dict[str, int] = {}
    text_parts: List[str] = []

    for item in content or []:
        data = to_plain(item) or {}
        item_type = data.get("type")
        if item_type == "text":
            text_parts.append(data.get("text") or "")
            continue
        if item_type in {"tool_reference", "reference"}:
            source_id = _ensure_source(
                sources,
                index_by_key,
                url=data.get("url"),
                title=data.get("title"),
                snippet=data.get("source"),
                provider="mistral",
            )
            text_parts.append(f"[{source_id}]")

    text = "".join(text_parts).strip()
    if not text:
        text = str(content or "")
    return make_llm_result(text, sources)
