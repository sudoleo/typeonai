import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "publish_consensus.py"
SPEC = importlib.util.spec_from_file_location("publish_consensus", SCRIPT_PATH)
publisher = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(publisher)


def test_response_output_text_and_question_validation():
    response = {
        "output": [
            {"type": "web_search_call"},
            {
                "type": "message",
                "content": [{"type": "output_text", "text": '"What should we compare"'}],
            },
        ]
    }

    assert publisher.response_output_text(response) == '"What should we compare"'
    assert publisher.validate_question('"What should we compare"') == "What should we compare?"


def test_main_runs_publish_and_index_flow_without_topic_call(monkeypatch, capsys):
    monkeypatch.setenv("CONSENSUS_API_KEY", "cns_test")
    monkeypatch.setenv("CONSENSUS_API_BASE_URL", "https://consensus.example")
    monkeypatch.setenv("CONSENSUS_QUESTION", "Which technologies should be compared?")
    monkeypatch.setenv("CONSENSUS_IDEMPOTENCY_KEY", "scheduled-test")
    monkeypatch.setenv("CONSENSUS_AUTO_INDEX", "true")
    calls = []

    def fake_http(method, url, *, headers=None, payload=None, timeout=60):
        calls.append((method, url, headers, payload))
        if url.endswith("/api/v1/consensus/runs"):
            return 202, {"run_id": "a" * 32, "status": "reserved"}
        if url.endswith("/api/v1/consensus/runs/" + "a" * 32):
            return 200, {"run_id": "a" * 32, "status": "succeeded"}
        if url.endswith("/share"):
            return 201, {"share_id": "B" * 16, "indexing_status": "noindex"}
        if url.endswith("/indexing"):
            return 200, {
                "share_id": "B" * 16,
                "url": "https://consensus.example/s/topic-" + "B" * 16,
                "indexing_status": "indexed",
                "in_sitemap": True,
            }
        raise AssertionError(url)

    monkeypatch.setattr(publisher, "http_json", fake_http)

    assert publisher.main() == 0
    output = capsys.readouterr().out
    assert "Which technologies should be compared?" in output
    assert '"indexing_status": "indexed"' in output
    assert calls[0][2]["Idempotency-Key"] == "scheduled-test"
    assert calls[-1][3] == {"indexed": True}


def test_topic_selection_uses_web_search_and_recent_question_history(monkeypatch):
    monkeypatch.delenv("CONSENSUS_QUESTION", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("CONSENSUS_TOPIC_BRIEF", "")
    monkeypatch.setattr(
        publisher,
        "recent_questions",
        lambda api_base, api_key: ["What was already published?"],
    )
    captured = {}

    def fake_http(method, url, *, headers=None, payload=None, timeout=60):
        captured.update(method=method, url=url, headers=headers, payload=payload)
        return 200, {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Which grid technologies best support renewable energy?",
                        }
                    ],
                }
            ]
        }

    monkeypatch.setattr(publisher, "http_json", fake_http)
    question = publisher.choose_question("https://consensus.example", "cns_test")

    assert question.endswith("?")
    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert captured["payload"]["model"] == "gpt-5.6-luna"
    assert captured["payload"]["tools"] == [{"type": "web_search"}]
    assert "What was already published?" in captured["payload"]["input"]
    assert "evidence-rich public-interest topic" in captured["payload"]["input"]
