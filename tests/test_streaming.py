import json
import unittest
from unittest import mock

import anyio

from types import SimpleNamespace

from app.services.llm.citations import make_llm_result
from app.services.llm.streaming import (
    _stream_chat_completions,
    _stream_openai_responses,
    iter_sse_events,
    sse_pack,
    stream_grok_query,
    streaming_model_response,
)
from app.services.llm.consensus_engine import (
    is_consensus_error_text,
    stream_consensus,
    stream_differences,
)


class FakeSSEResponse:
    """Minimaler Ersatz für requests.Response mit iter_lines()."""

    def __init__(self, raw: str):
        self._raw = raw

    def iter_lines(self):
        for line in self._raw.split("\n"):
            yield line.encode("utf-8")


def collect_sse_body(response) -> str:
    async def consume():
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk if isinstance(chunk, str) else chunk.decode("utf-8"))
        return "".join(chunks)

    return anyio.run(consume)


def parse_sse_text(raw: str):
    events = []
    fake = FakeSSEResponse(raw)
    for event_name, data_str in iter_sse_events(fake):
        events.append((event_name, json.loads(data_str)))
    return events


class SSEPackTests(unittest.TestCase):
    def test_pack_roundtrip(self):
        packed = sse_pack("delta", {"text": "Hällo\nWelt"})
        self.assertTrue(packed.startswith("event: delta\ndata: "))
        self.assertTrue(packed.endswith("\n\n"))
        events = parse_sse_text(packed)
        self.assertEqual(events, [("delta", {"text": "Hällo\nWelt"})])

    def test_iter_sse_events_multiple(self):
        raw = (
            "event: delta\ndata: {\"text\": \"a\"}\n\n"
            ": keepalive comment\n\n"
            "event: final\ndata: {\"response\": \"ab\"}\n\n"
        )
        events = parse_sse_text(raw)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0], ("delta", {"text": "a"}))
        self.assertEqual(events[1], ("final", {"response": "ab"}))


class StreamingModelResponseTests(unittest.TestCase):
    def test_delta_and_final_events(self):
        def gen():
            yield {"type": "delta", "text": "Hel"}
            yield {"type": "delta", "text": "lo"}
            yield {
                "type": "final",
                "result": make_llm_result("Hello [S1]", [{"id": "S1", "title": "T", "url": "https://example.com"}]),
            }

        response = streaming_model_response(gen(), "OpenAI", {"free_usage_remaining": 5, "is_pro_user": False})
        self.assertEqual(response.media_type, "text/event-stream")
        self.assertEqual(response.headers.get("x-accel-buffering"), "no")

        events = parse_sse_text(collect_sse_body(response))
        self.assertEqual([name for name, _ in events], ["delta", "delta", "final"])
        final = events[-1][1]
        self.assertEqual(final["response"], "Hello [S1]")
        self.assertEqual(final["free_usage_remaining"], 5)
        self.assertEqual(final["sources"][0]["id"], "S1")

    def test_error_result_final_event(self):
        def gen():
            yield {"type": "delta", "text": "partial"}
            yield {
                "type": "final",
                "result": {"text": "", "sources": [], "error": "OpenAI could not complete this request. Please try again later.", "error_code": "provider_request_failed"},
            }

        response = streaming_model_response(gen(), "OpenAI", {"free_usage_remaining": 1})
        events = parse_sse_text(collect_sse_body(response))
        final = events[-1][1]
        self.assertIn("error", final)
        self.assertEqual(final["response"], "")
        self.assertEqual(final["free_usage_remaining"], 1)

    def test_structured_content_blocks_never_serialize_as_object_object(self):
        def gen():
            yield {
                "type": "delta",
                "text": [{"type": "text", "text": "Structured answer"}],
            }
            yield {
                "type": "final",
                "result": {
                    "text": {"type": "text", "text": "Structured answer"},
                    "sources": [],
                },
            }

        response = streaming_model_response(gen(), "DeepSeek")
        events = parse_sse_text(collect_sse_body(response))
        self.assertEqual(events[0], ("delta", {"text": "Structured answer"}))
        self.assertEqual(events[-1][1]["response"], "Structured answer")
        self.assertNotIn("[object Object]", collect_sse_body(
            streaming_model_response(gen(), "DeepSeek")
        ))

    def test_generator_exception_yields_error_final(self):
        def gen():
            yield {"type": "delta", "text": "x"}
            raise RuntimeError("connection dropped")

        response = streaming_model_response(gen(), "Mistral", {"key_used": "User API Key"})
        events = parse_sse_text(collect_sse_body(response))
        self.assertEqual(events[-1][0], "final")
        final = events[-1][1]
        self.assertIn("Mistral could not complete this request", final["error"])
        self.assertEqual(final["error_code"], "provider_stream_failed")
        self.assertNotIn("error_detail", final)
        self.assertEqual(final["key_used"], "User API Key")


def _chunk(*, content=None, reasoning_content=None, finish_reason=None):
    delta = SimpleNamespace(content=content, reasoning_content=reasoning_content)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


class FakeChatCompletions:
    def __init__(self, chunks):
        self._chunks = chunks

    def create(self, **kwargs):
        return iter(self._chunks)


class FakeOpenAIClient:
    def __init__(self, chunks):
        self.chat = SimpleNamespace(completions=FakeChatCompletions(chunks))


class ChatCompletionsStreamTests(unittest.TestCase):
    """Deckt den DeepSeek-/OpenAI-kompatiblen Chat-Completions-Stream ab –
    speziell den Fall, dass ein Reasoning-Modell nur Reasoning liefert und
    nie eine Antwort ausgibt (früher: leerer String -> irreführender
    'Please log in'-Fallback im Frontend)."""

    def test_normal_answer_produces_deltas_and_text(self):
        client = FakeOpenAIClient([
            _chunk(content="Hel"),
            _chunk(content="lo"),
            _chunk(finish_reason="stop"),
        ])
        events = list(_stream_chat_completions(client=client, payload={"model": "x"}))
        self.assertEqual([e["type"] for e in events], ["delta", "delta", "final"])
        self.assertEqual(events[-1]["result"]["text"], "Hello")
        self.assertNotIn("error", events[-1]["result"])

    def test_reasoning_only_length_cutoff_yields_error_result(self):
        client = FakeOpenAIClient([
            _chunk(reasoning_content="thinking..."),
            _chunk(reasoning_content="still thinking..."),
            _chunk(finish_reason="length"),
        ])
        events = list(_stream_chat_completions(client=client, payload={"model": "x"}))
        self.assertEqual([e["type"] for e in events], ["reasoning", "reasoning", "final"])
        result = events[-1]["result"]
        self.assertEqual(result["text"], "")
        self.assertEqual(result["error_code"], "empty_reasoning_response")
        self.assertIn("ran out of output tokens", result["error"])

    def test_empty_response_without_length_yields_generic_error(self):
        client = FakeOpenAIClient([_chunk(finish_reason="stop")])
        events = list(_stream_chat_completions(client=client, payload={"model": "x"}))
        result = events[-1]["result"]
        self.assertEqual(result["error_code"], "empty_reasoning_response")
        self.assertIn("no answer", result["error"])


def _responses_sse(events) -> str:
    return "".join(f"data: {json.dumps(evt)}\n\n" for evt in events)


class FakeResponsesHTTP(FakeSSEResponse):
    status_code = 200
    text = ""


class ResponsesStreamTests(unittest.TestCase):
    """Responses-API-Stream (OpenAI/Grok): leere finale Antworten müssen als
    echter Fehler gemeldet werden, sonst zeigt das Frontend nur den
    irreführenden Generik-Text 'No response received / timed out'."""

    def _run(self, events):
        fake = FakeResponsesHTTP(_responses_sse(events))
        with mock.patch("app.services.llm.streaming.requests.post", return_value=fake):
            return list(_stream_openai_responses(
                api_key="k", base_url="https://api.x.ai/v1", payload={"model": "m"}, provider="grok",
            ))

    def test_incomplete_without_text_yields_max_tokens_error(self):
        events = self._run([
            {"type": "response.output_item.added", "item": {"type": "reasoning"}},
            {"type": "response.incomplete", "response": {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [],
            }},
        ])
        self.assertEqual([e["type"] for e in events], ["reasoning", "final"])
        result = events[-1]["result"]
        self.assertEqual(result["error_code"], "max_output_tokens")
        self.assertIn("output token budget", result["error"])

    def test_completed_with_text_stays_untouched(self):
        events = self._run([
            {"type": "response.output_text.delta", "delta": "Hi"},
            {"type": "response.completed", "response": {
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "Hi"}]}],
            }},
        ])
        self.assertEqual(events[-1]["result"]["text"], "Hi")
        self.assertNotIn("error", events[-1]["result"])


class GrokReasoningMarkerTests(unittest.TestCase):
    """Non-Reasoning-Grok-Varianten dürfen keine Reasoning-Marker durchreichen
    (xAI streamt trotzdem Reasoning-Items) — das Frontend zeigte sonst
    'Reasoning' für ein Modell mit dem Label 'No reasoning'."""

    GROK_EVENTS = [
        {"type": "response.output_item.added", "item": {"type": "reasoning"}},
        {"type": "response.output_text.delta", "delta": "Hi"},
        {"type": "response.completed", "response": {
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "Hi"}]}],
        }},
    ]

    def _run(self, model_override):
        fake = FakeResponsesHTTP(_responses_sse(self.GROK_EVENTS))
        with mock.patch("app.services.llm.streaming.requests.post", return_value=fake):
            return list(stream_grok_query("Q?", "key", model_override=model_override))

    def test_non_reasoning_model_suppresses_reasoning_markers(self):
        events = self._run("grok-4.20-non-reasoning")
        self.assertEqual([e["type"] for e in events], ["delta", "final"])
        self.assertEqual(events[-1]["result"]["text"], "Hi")

    def test_reasoning_model_keeps_reasoning_markers(self):
        events = self._run("grok-4.3")
        self.assertEqual([e["type"] for e in events], ["reasoning", "delta", "final"])


class ConsensusStreamTests(unittest.TestCase):
    def test_invalid_consensus_engine(self):
        events = list(stream_consensus(
            "Q?", "a", "b", None, None, None, None,
            excluded_models=[],
            consensus_model="DoesNotExist",
            api_keys={},
        ))
        self.assertEqual(events[-1]["type"], "final")
        self.assertEqual(events[-1]["text"], "Invalid consensus model selected: DoesNotExist")

    def test_differences_without_answers(self):
        events = list(stream_differences(
            None, None, None, None, None, None,
            consensus_answer="c",
            api_keys={},
            differences_model="OpenAI",
            excluded_models=[],
        ))
        self.assertEqual(events, [{"type": "final", "text": "Error in comparison: no model responses available.", "data": None}])

    def test_invalid_differences_engine(self):
        events = list(stream_differences(
            "answer one", "answer two", None, None, None, None,
            consensus_answer="c",
            api_keys={},
            differences_model="DoesNotExist",
            excluded_models=[],
        ))
        self.assertEqual(events[-1]["type"], "final")
        self.assertEqual(events[-1]["text"], "Invalid model selected for difference comparison.")

    def test_invalid_engine_final_is_flagged_as_error(self):
        events = list(stream_consensus(
            "Q?", "a", "b", None, None, None, None,
            excluded_models=[],
            consensus_model="DoesNotExist",
            api_keys={},
        ))
        self.assertTrue(events[-1].get("error"))


class ConsensusRetryTests(unittest.TestCase):
    def _run(self, fake_engine):
        # DEVELOPER_GEMINI_API_KEY leeren: mit nur einem OpenAI-Key darf kein
        # Fallback-Provider verfuegbar sein (sonst gaebe es einen 3. Versuch).
        with mock.patch.dict("os.environ", {"DEVELOPER_GEMINI_API_KEY": ""}), mock.patch(
            "app.services.llm.consensus_engine._stream_consensus_engine",
            side_effect=fake_engine,
        ) as patched:
            events = list(stream_consensus(
                "Q?", "a", "b", None, None, None, None,
                excluded_models=[],
                consensus_model="OpenAI",
                api_keys={"OpenAI": "sk-test"},
            ))
        return events, patched.call_count

    def test_transient_failure_is_retried(self):
        calls = []

        def fake_engine(consensus_model, api_keys, prompt):
            calls.append(1)
            if len(calls) == 1:
                yield {"type": "delta", "text": "partial "}
                raise RuntimeError("503 - UNAVAILABLE")
            yield {"type": "reasoning"}
            yield {"type": "delta", "text": "Recovered "}
            yield {"type": "delta", "text": "answer."}

        events, call_count = self._run(fake_engine)
        self.assertEqual(call_count, 2)
        self.assertEqual(events[-1], {"type": "final", "text": "Recovered answer."})

    def test_persistent_failure_yields_error_final(self):
        def fake_engine(consensus_model, api_keys, prompt):
            raise RuntimeError("503 - UNAVAILABLE")
            yield  # pragma: no cover - macht die Funktion zum Generator

        events, call_count = self._run(fake_engine)
        self.assertEqual(call_count, 2)
        self.assertEqual(events[-1]["text"], "Consensus error: 503 - UNAVAILABLE")
        self.assertTrue(events[-1]["error"])

    def test_empty_stream_counts_as_failure(self):
        def fake_engine(consensus_model, api_keys, prompt):
            return iter(())

        events, call_count = self._run(fake_engine)
        self.assertEqual(call_count, 2)
        self.assertEqual(events[-1]["text"], "Consensus error: empty response from consensus engine.")
        self.assertTrue(events[-1]["error"])


class ConsensusErrorTextTests(unittest.TestCase):
    def test_error_and_empty_texts_are_detected(self):
        self.assertTrue(is_consensus_error_text("Consensus error: 503 - UNAVAILABLE"))
        self.assertTrue(is_consensus_error_text("Invalid consensus model selected: X"))
        self.assertTrue(is_consensus_error_text(""))
        self.assertTrue(is_consensus_error_text("   "))
        self.assertTrue(is_consensus_error_text(None))

    def test_normal_answers_are_not_errors(self):
        self.assertFalse(is_consensus_error_text("The capital of France is Paris."))
        # Ein Konsens, der das Wort "error" nur enthält, ist kein Fehler
        self.assertFalse(is_consensus_error_text("Common error sources include ..."))


if __name__ == "__main__":
    unittest.main()
