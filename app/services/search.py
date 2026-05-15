import os
import requests
import re
from fastapi import HTTPException

def exa_search(query: str, num_results: int = 5):
    search_url = "https://api.exa.ai/search"
    headers = {"Content-Type": "application/json", "x-api-key": os.getenv("DEVELOPER_EXA_API_KEY")}
    payload = {"query": query, "num_results": num_results}

    resp = requests.post(search_url, json=payload, headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])
    if not results:
        return {"results": []}

    ids = [r["id"] for r in results]

    contents_resp = requests.post(
        "https://api.exa.ai/contents",
        json={"ids": ids, "contents": {"max_characters": 1200, "include_html": False}},
        headers=headers,
        timeout=10
    )
    contents_resp.raise_for_status()
    contents_data = contents_resp.json()
    contents_by_id = {c["id"]: c for c in contents_data.get("results", contents_data.get("contents", []))}

    merged = []
    for r in results:
        cid = r["id"]
        c = contents_by_id.get(cid, {})
        merged.append({
            "id": r["id"],
            "title": r.get("title"),
            "url": r.get("url"),
            "text": c.get("text") or c.get("content") or c.get("snippet") or ""
        })
    return {"results": merged}


def clean_exa_text(raw: str) -> str:
    if not raw:
        return ""
    text = raw.replace("\r", "\n").strip()

    drop_prefixes = (
        "[Skip to", "- [Skip to", "[Jump to", "- [Jump to",
        "[LIVING ROOM IDEAS", "[HALLWAY IDEAS"
    )
    lines = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            continue
        if any(s.startswith(p) for p in drop_prefixes):
            continue
        lines.append(s)

    text = " ".join(lines)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def build_evidence_block(exa_results, max_sources: int = 5):
    sources = []
    for i, r in enumerate(exa_results.get("results", [])[:max_sources], start=1):
        cleaned = clean_exa_text(r.get("text", ""))
        extract = cleaned[:800]
        sources.append({
            "id": f"S{i}",
            "title": r["title"],
            "url": r["url"],
            "extract": extract
        })

    block_lines = ["Relevant web sources:"]
    for s in sources:
        block_lines.append(f"[{s['id']}] {s['title']} – {s['url']}\n{s['extract']}\n")

    return "\n".join(block_lines), sources


def prepare_prompt_with_websearch(question: str, search_mode: bool, base_system_prompt: str):
    if not search_mode:
        return base_system_prompt, None

    exa_key = os.getenv("DEVELOPER_EXA_API_KEY")
    if not exa_key:
        raise HTTPException(status_code=500, detail="Exa API key missing")

    raw = exa_search(question, num_results=6)
    evidence_block, sources = build_evidence_block(raw)

    enriched_prompt = f"""
        Synthesize the web sources below into a natural, coherent answer.

        Guidelines:
        - Focus on answering the question; do not simply list links or snippets.
        - Integrate facts fluently and cite them strictly as [S1], [S2].
        - Base your factual claims on the provided sources.

        Web sources:
        {evidence_block}

        Original instructions:
        {base_system_prompt}
        """.strip()

    return enriched_prompt, sources
