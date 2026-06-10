import json
import unittest

import anyio

from app.services.llm.citations import make_llm_result
from app.services.llm.streaming import (
    iter_sse_events,
    sse_pack,
    streaming_model_response,
)
from app.services.llm.consensus_engine import stream_consensus, stream_differences


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
                "result": {"text": "", "sources": [], "error": "OpenAI could not complete this request. Please try again later.", "error_detail": "boom"},
            }

        response = streaming_model_response(gen(), "OpenAI", {"free_usage_remaining": 1})
        events = parse_sse_text(collect_sse_body(response))
        final = events[-1][1]
        self.assertIn("error", final)
        self.assertEqual(final["response"], "")
        self.assertEqual(final["free_usage_remaining"], 1)

    def test_generator_exception_yields_error_final(self):
        def gen():
            yield {"type": "delta", "text": "x"}
            raise RuntimeError("connection dropped")

        response = streaming_model_response(gen(), "Mistral", {"key_used": "User API Key"})
        events = parse_sse_text(collect_sse_body(response))
        self.assertEqual(events[-1][0], "final")
        final = events[-1][1]
        self.assertIn("Mistral could not complete this request", final["error"])
        self.assertEqual(final["error_detail"], "connection dropped")
        self.assertEqual(final["key_used"], "User API Key")


class ConsensusStreamTests(unittest.TestCase):
    def test_invalid_consensus_engine(self):
        events = list(stream_consensus(
            "Q?", "a", "b", None, None, None, None,
            best_model="",
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


if __name__ == "__main__":
    unittest.main()
