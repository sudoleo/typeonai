#!/usr/bin/env python3
"""Choose, run, publish and optionally index one Consensus question.

Designed for cron/GitHub Actions and intentionally uses only the Python
standard library. Set CONSENSUS_QUESTION to skip the OpenAI topic-selection
call for manual or deterministic runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class PublisherError(RuntimeError):
    pass


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def http_json(method: str, url: str, *, headers=None, payload=None, timeout=60):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request_headers = {"Accept": "application/json", **(headers or {})}
    if body is not None:
        request_headers["Content-Type"] = "application/json"
    request = Request(url, data=body, headers=request_headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
            return response.status, json.loads(raw) if raw else None
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(raw)
        except json.JSONDecodeError:
            detail = raw or exc.reason
        raise PublisherError(f"{method} {url} failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise PublisherError(f"{method} {url} failed: {exc.reason}") from exc


def response_output_text(response: dict) -> str:
    chunks = []
    for item in response.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if isinstance(content, dict) and content.get("type") == "output_text":
                chunks.append(str(content.get("text") or ""))
    return "\n".join(chunks).strip()


def recent_questions(api_base: str, api_key: str) -> list[str]:
    try:
        _status, payload = http_json(
            "GET",
            f"{api_base}/api/v1/shares?limit=20",
            headers={"X-API-Key": api_key},
        )
    except PublisherError as exc:
        print(f"Warning: could not load recent published questions: {exc}", file=sys.stderr)
        return []
    return [
        str(item.get("question") or "").strip()
        for item in (payload or {}).get("shares") or []
        if isinstance(item, dict) and str(item.get("question") or "").strip()
    ]


def choose_question(api_base: str, consensus_key: str) -> str:
    explicit = os.environ.get("CONSENSUS_QUESTION", "").strip()
    if explicit:
        return validate_question(explicit)

    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not openai_key:
        raise PublisherError("OPENAI_API_KEY is required when CONSENSUS_QUESTION is empty")
    model = os.environ.get("OPENAI_TOPIC_MODEL", "gpt-5.6-luna").strip()
    default_brief = (
        "Choose one timely, evidence-rich public-interest topic in science, technology, "
        "economics, environment, or society. It should benefit from comparing multiple AI "
        "models and support a substantial answer with several credible web sources. Avoid "
        "personal medical, legal, or financial advice, sensationalism, and pure opinion polls."
    )
    brief = (os.environ.get("CONSENSUS_TOPIC_BRIEF") or default_brief).strip()
    previous = recent_questions(api_base, consensus_key)
    avoid = "\n".join(f"- {question}" for question in previous) or "- none"
    today = datetime.now(timezone.utc).date().isoformat()
    prompt = f"""Today is {today}.

{brief}

Do not repeat or closely paraphrase these recently published questions:
{avoid}

Return exactly one neutral English question, with no quotation marks, preface, markdown, or explanation.
The question must be 15 to 300 characters long and understandable without additional context."""
    _status, response = http_json(
        "POST",
        "https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {openai_key}"},
        payload={
            "model": model,
            "input": prompt,
            "tools": [{"type": "web_search"}],
            "reasoning": {"effort": "low"},
            "max_output_tokens": 800,
        },
        timeout=180,
    )
    return validate_question(response_output_text(response or {}))


def validate_question(value: str) -> str:
    question = " ".join(str(value or "").strip().strip('"\'').split())
    if len(question) < 15 or len(question) > 300:
        raise PublisherError(
            f"Selected question must contain 15-300 characters; received {len(question)}"
        )
    if not question.endswith("?"):
        question += "?"
    if len(question) > 300:
        raise PublisherError("Selected question exceeds 300 characters after normalization")
    return question


def idempotency_key(question: str) -> str:
    configured = os.environ.get("CONSENSUS_IDEMPOTENCY_KEY", "").strip()
    if configured:
        return configured
    slot = os.environ.get("GITHUB_RUN_ID") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    digest = hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]
    return f"scheduled-publisher-{slot}-{digest}"


def wait_for_run(api_base: str, api_key: str, run_id: str) -> dict:
    poll_seconds = max(1.0, float(os.environ.get("CONSENSUS_POLL_SECONDS", "5")))
    timeout_seconds = max(30.0, float(os.environ.get("CONSENSUS_TIMEOUT_SECONDS", "1800")))
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        _status, run = http_json(
            "GET",
            f"{api_base}/api/v1/consensus/runs/{run_id}",
            headers={"X-API-Key": api_key},
        )
        state = (run or {}).get("status")
        print(f"Consensus run {run_id}: {state}")
        if state == "succeeded":
            return run
        if state == "failed":
            raise PublisherError(f"Consensus run failed: {(run or {}).get('error')}")
        time.sleep(poll_seconds)
    raise PublisherError(f"Consensus run {run_id} did not finish within {timeout_seconds:.0f}s")


def write_github_summary(question: str, share: dict) -> None:
    target = os.environ.get("GITHUB_STEP_SUMMARY", "").strip()
    if not target:
        return
    lines = [
        "## Published Consensus",
        "",
        f"- Question: {question}",
        f"- URL: {share.get('url')}",
        f"- Indexing: {share.get('indexing_status')}",
        f"- In sitemap: {share.get('in_sitemap')}",
        "",
    ]
    with Path(target).open("a", encoding="utf-8") as summary:
        summary.write("\n".join(lines))


def main() -> int:
    api_base = os.environ.get("CONSENSUS_API_BASE_URL", "https://www.consens.io").rstrip("/")
    api_key = os.environ.get("CONSENSUS_API_KEY", "").strip()
    if not api_key:
        raise PublisherError("CONSENSUS_API_KEY is required")

    question = choose_question(api_base, api_key)
    print(f"Selected question: {question}")
    headers = {
        "X-API-Key": api_key,
        "Idempotency-Key": idempotency_key(question),
    }
    _status, run = http_json(
        "POST",
        f"{api_base}/api/v1/consensus/runs",
        headers=headers,
        payload={
            "question": question,
            "deep_think": env_bool("CONSENSUS_DEEP_THINK", False),
        },
    )
    run_id = str((run or {}).get("run_id") or "")
    if not run_id:
        raise PublisherError("Consensus API did not return a run_id")
    wait_for_run(api_base, api_key, run_id)

    _status, share = http_json(
        "POST",
        f"{api_base}/api/v1/consensus/runs/{run_id}/share",
        headers={"X-API-Key": api_key},
    )
    if env_bool("CONSENSUS_AUTO_INDEX", True):
        _status, share = http_json(
            "PUT",
            f"{api_base}/api/v1/shares/{share['share_id']}/indexing",
            headers={"X-API-Key": api_key},
            payload={"indexed": True},
        )

    print(json.dumps({"question": question, "run_id": run_id, "share": share}, indent=2))
    write_github_summary(question, share)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PublisherError as exc:
        print(f"Publisher failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
