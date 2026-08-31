"""DeepSeek uses the same OpenRouter web-search and citation contract."""

import app.core.config as cfg
from app.services.llm import engines
from app.services.llm.citations import parse_openrouter_response
from app.services.llm.engines import build_provider_payload


def test_deepseek_payload_uses_openrouter_search_without_provider_pinning():
    built = build_provider_payload(
        "deepseek",
        question="Who won the last race?",
        system_prompt="Answer with sources.",
        model_override=cfg.DEEPSEEK_FLASH_MODEL,
        max_output_tokens=500,
    )

    payload = built["payload"]
    assert built["api_model"] == f"deepseek/{cfg.DEEPSEEK_FLASH_MODEL}"
    assert payload["tools"] == [{
        "type": "openrouter:web_search",
        "parameters": {"engine": "auto", "max_uses": 2},
    }]
    assert payload["provider"] == {"zdr": True}
    assert "order" not in payload["provider"]
    assert "only" not in payload["provider"]


def test_deepseek_url_citations_keep_the_logical_provider_label():
    text = "Lando Norris won the race."
    parsed = parse_openrouter_response(text, [{
        "type": "url_citation",
        "url_citation": {
            "url": "https://example.test/results",
            "title": "Race results",
            "content": "Lando Norris won",
            "start_index": 0,
            "end_index": len(text),
        },
    }], "deepseek")

    assert parsed["text"].endswith("[S1]")
    assert parsed["sources"] == [{
        "id": "S1",
        "title": "Race results",
        "url": "https://example.test/results",
        "snippet": "Lando Norris won",
        "extract": "Lando Norris won",
        "provider": "deepseek",
    }]


def test_successful_query_model_parses_openrouter_response(monkeypatch):
    class Response:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {
                "content": "Verified answer.",
                "annotations": [{
                    "type": "url_citation",
                    "url_citation": {
                        "url": "https://example.test/source",
                        "title": "Example source",
                        "end_index": 16,
                    },
                }],
            }}]}

        def close(self):
            pass

    monkeypatch.setattr(engines.requests, "post", lambda *args, **kwargs: Response())

    result = engines.query_model("deepseek", "question", "sk-or-test")

    assert result["text"] == "Verified answer. [S1]"
    assert result["sources"][0]["provider"] == "deepseek"
