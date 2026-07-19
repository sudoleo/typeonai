#!/usr/bin/env python3
"""Choose, run, publish, watch and optionally index one Consensus question.

Designed for cron/GitHub Actions and intentionally uses only the Python
standard library. Set CONSENSUS_QUESTION to skip the OpenAI topic-selection
call for manual or deterministic runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class PublisherError(RuntimeError):
    pass


DEFAULT_TOPIC_BRIEF = (
    "Choose one highly current, evidence-rich AI topic that people are beginning to search "
    "for now. Focus on named AI models, products, features, subscriptions, developer tools, "
    "release timing, availability, surprising product behavior, or a credible emerging "
    "rumor. Favor a narrow exact-intent query while the news window is still young and "
    "dedicated coverage is sparse.\n\n"
    "The question should help someone verify a claim, understand what just changed, or "
    "decide whether to wait for or use a specific AI product. It must benefit from comparing "
    "multiple AI models and support a substantial answer from several credible web sources. "
    "Careful speculation is welcome only when it is clearly framed and anchored in official "
    "announcements, documentation, changelogs, observed product behavior, or credible reporting.\n\n"
    "Avoid government policy, regulation, legislation, elections, broad evergreen explainers, "
    "generic AI trend pieces, personal medical/legal/financial advice, sensationalism, and "
    "unsupported rumors."
)


# These requirements are appended even when an older Admin topic brief is still stored.
# They encode the Search Console pattern behind the publisher's current acquisition strategy.
SEARCH_OPPORTUNITY_RULES = (
    "Search-opportunity requirements:\n"
    "- Work in the high-current AI product/news lane. Prefer a named model, company, feature, "
    "plan, coding tool, release, leak, or rumor with a new signal from roughly the last seven days.\n"
    "- Favor queries shaped like release/availability checks, rumor verification, product-name "
    "clarification, or what a just-announced change means for users.\n"
    "- Use web search to compare at least five candidate queries before choosing. Reject a "
    "candidate when its exact search intent is already answered by many established, high-ranking "
    "news, government, legal, or corporate explainer pages. Choose the candidate with the best "
    "combination of freshness, plausible search demand, low exact-intent competition, and sources.\n"
    "- Do not select government policy, grants, federal/state law, regulation, enforcement, "
    "elections, or broad societal impact as the main intent.\n"
    "- Do not invent a release or rumor. A speculative question needs at least one current, "
    "checkable signal and must make uncertainty explicit."
)


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


def load_publisher_config(api_base: str, api_key: str) -> dict:
    _status, payload = http_json(
        "GET",
        f"{api_base}/api/v1/publisher/config",
        headers={"X-API-Key": api_key},
    )
    if not isinstance(payload, dict):
        raise PublisherError("Consensus API returned an invalid Publisher configuration")
    return payload


def choose_question(api_base: str, consensus_key: str, *, topic_brief="") -> str:
    explicit = os.environ.get("CONSENSUS_QUESTION", "").strip()
    if explicit:
        return validate_question(explicit)

    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not openai_key:
        raise PublisherError("OPENAI_API_KEY is required when CONSENSUS_QUESTION is empty")
    model = os.environ.get("OPENAI_TOPIC_MODEL", "gpt-5.6-luna").strip()
    brief = (
        os.environ.get("CONSENSUS_TOPIC_BRIEF") or topic_brief or DEFAULT_TOPIC_BRIEF
    ).strip()
    previous = recent_questions(api_base, consensus_key)
    avoid = "\n".join(f"- {question}" for question in previous) or "- none"
    today = datetime.now(timezone.utc).date().isoformat()
    feedback = ""
    last_error = None

    for _attempt in range(3):
        prompt = f"""Today is {today}.

{brief}

{SEARCH_OPPORTUNITY_RULES}

Do not repeat or closely paraphrase these recently published questions:
{avoid}

Write the final question as a Google-style search query and clickable page title:
- use 6 to 16 words and no more than 110 characters;
- express one clear search intent in plain, concrete language;
- put the main subject and outcome directly in the question;
- do not begin with a date or "As of";
- do not combine competing theses, trade-offs, or multiple questions in one sentence;
- avoid subordinate clauses introduced by when, while, whereas, although, or despite;
- include a year only when it is essential to identify a policy, event, or product.

For example, prefer "Do AI data centers raise household electricity prices?" over a long
question that asks whether benefits are justified while also discussing costs and climate goals.

Return exactly one neutral English question, with no quotation marks, preface, markdown, or explanation.
{feedback}"""
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
        try:
            return validate_generated_question(response_output_text(response or {}))
        except PublisherError as exc:
            last_error = exc
            feedback = (
                f"\nThe previous candidate failed the title check: {exc}. "
                "Produce a simpler replacement."
            )
    raise PublisherError(f"Topic model did not produce a search-ready question: {last_error}")


def validate_question(value: str) -> str:
    question = str(value or "").strip().strip('"\'')
    question = re.sub(r"^(?:#{1,6}\s*|[*_`]{1,3})", "", question)
    question = re.sub(r"(?:[*_`]{1,3})$", "", question)
    question = " ".join(question.split())
    if len(question) < 15 or len(question) > 300:
        raise PublisherError(
            f"Selected question must contain 15-300 characters; received {len(question)}"
        )
    if not question.endswith("?"):
        question += "?"
    if len(question) > 300:
        raise PublisherError("Selected question exceeds 300 characters after normalization")
    return question


def validate_search_question(value: str) -> str:
    question = validate_question(value)
    words = re.findall(r"\b[\w'-]+\b", question, flags=re.UNICODE)
    issues = []
    if not 6 <= len(words) <= 16:
        issues.append("use 6-16 words")
    if len(question) > 110:
        issues.append("use at most 110 characters")
    lowered = question.lower()
    if lowered.startswith("as of ") or re.match(r"^(?:in|by)\s+20\d{2}\b", lowered):
        issues.append("do not lead with a date")
    if any(
        marker in lowered
        for marker in (" when ", " while ", " whereas ", " although ", " despite ")
    ):
        issues.append("remove subordinate trade-off clauses")
    if any(mark in question for mark in (";", ":", "—")) or question.count(",") > 1:
        issues.append("use one simple clause")
    if question.count("?") != 1:
        issues.append("ask exactly one question")
    if issues:
        raise PublisherError("; ".join(issues))
    return question


def validate_generated_question(value: str) -> str:
    """Apply hard topic exclusions to automatically selected search titles."""
    question = validate_search_question(value)
    lowered = question.lower()
    excluded_patterns = (
        r"\bpolitical appointees?\b",
        r"\bfederal\b",
        r"\bstate (?:chatbot |ai )?laws?\b",
        r"\bgovernment (?:policy|rule|regulation)s?\b",
        r"\b(?:ftc|fcc|sec|congress)\b",
        r"\b(?:legislation|regulatory|regulation|election)s?\b",
    )
    if any(re.search(pattern, lowered) for pattern in excluded_patterns):
        raise PublisherError("choose a current AI product/news query, not government policy")
    ai_signal = re.search(
        r"\b(?:ai|artificial intelligence|llms?|large language models?|chatbots?|"
        r"openai|chatgpt|gpt|anthropic|claude|google gemini|gemini|mistral|le chat|"
        r"deepseek|grok|xai|cursor|copilot|perplexity|hugging face|midjourney|"
        r"runway|sora)\b",
        lowered,
    )
    if not ai_signal:
        raise PublisherError("name a specific AI model, product, company, or tool")
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


def write_github_summary(question: str, share: dict, watch: dict | None = None) -> None:
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
    ]
    if watch:
        lines.extend(
            [
                f"- Watch: {watch.get('interval')} / {watch.get('model_tier')} models",
                f"- Next Watch run: {watch.get('next_run_at')}",
            ]
        )
    lines.append("")
    with Path(target).open("a", encoding="utf-8") as summary:
        summary.write("\n".join(lines))


def send_telegram_notification(question: str, share: dict) -> bool:
    """Send the published page to Telegram without failing the publisher run."""
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not bot_token and not chat_id:
        return False
    if not bot_token or not chat_id:
        print(
            "Warning: Telegram notification skipped because TELEGRAM_BOT_TOKEN "
            "and TELEGRAM_CHAT_ID must both be configured.",
            file=sys.stderr,
        )
        return False

    url = str(share.get("url") or "").strip()
    if not url:
        print(
            "Warning: Telegram notification skipped because the share URL is missing.",
            file=sys.stderr,
        )
        return False

    message = f"New Consensus published\n\n{question}\n\n{url}"
    payload = json.dumps(
        {
            "chat_id": chat_id,
            "text": message,
        }
    ).encode("utf-8")
    request = Request(
        f"https://api.telegram.org/bot{bot_token}/sendMessage",
        data=payload,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            response.read()
    except HTTPError as exc:
        # Never include the request URL here: it contains the bot token.
        print(
            f"Warning: Telegram notification failed with HTTP {exc.code}.",
            file=sys.stderr,
        )
        return False
    except URLError:
        # Keep network exception details out of logs in case a runtime embeds the URL.
        print("Warning: Telegram notification failed with a network error.", file=sys.stderr)
        return False

    print("Telegram notification sent.")
    return True


def main() -> int:
    api_base = os.environ.get("CONSENSUS_API_BASE_URL", "https://www.consens.io").rstrip("/")
    api_key = os.environ.get("CONSENSUS_API_KEY", "").strip()
    if not api_key:
        raise PublisherError("CONSENSUS_API_KEY is required")

    config = load_publisher_config(api_base, api_key)
    if not config.get("enabled"):
        print("Scheduled Consensus Publisher is disabled in Admin; nothing to do.")
        return 0

    question = choose_question(
        api_base, api_key, topic_brief=str(config.get("topic_brief") or "")
    )
    print(f"Selected question: {question}")
    headers = {
        "X-API-Key": api_key,
        "X-Consensus-Publisher": "true",
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
    watch = None
    if config.get("weekly_watch_enabled"):
        _status, watch_payload = http_json(
            "POST",
            f"{api_base}/api/v1/shares/{share['share_id']}/watch",
            headers={"X-API-Key": api_key},
        )
        watch = (watch_payload or {}).get("watch")

    if env_bool("CONSENSUS_AUTO_INDEX", bool(config.get("auto_index"))):
        _status, share = http_json(
            "PUT",
            f"{api_base}/api/v1/shares/{share['share_id']}/indexing",
            headers={"X-API-Key": api_key},
            payload={"indexed": True},
        )

    print(
        json.dumps(
            {"question": question, "run_id": run_id, "share": share, "watch": watch},
            indent=2,
        )
    )
    write_github_summary(question, share, watch)
    send_telegram_notification(question, share)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PublisherError as exc:
        print(f"Publisher failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
