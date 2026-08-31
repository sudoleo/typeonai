import json
import threading
import unittest
from unittest import mock

import anyio
from google.api_core.datetime_helpers import DatetimeWithNanoseconds

from app.services.llm.citations import make_llm_result
from app.services.llm.streaming import (
    _iter_openrouter_chunks,
    _stream_openrouter_chat_completion,
    iter_sse_events,
    iter_sse_with_keepalive,
    keepalive_streaming_response,
    sse_pack,
    stream_model_query,
    streaming_model_response,
)
from app.services.llm.provider_runtime import (
    ProviderCancellation,
    ProviderCancelled,
    bind_provider_cancellation,
    managed_provider_resource,
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


class ProviderCancellationTests(unittest.TestCase):
    @staticmethod
    async def _disconnect_response(response, producer_entered):
        async def receive():
            await anyio.to_thread.run_sync(lambda: producer_entered.wait(timeout=1))
            return {"type": "http.disconnect"}

        async def send(_message):
            return None

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "method": "GET",
            "path": "/stream",
            "headers": [],
            "query_string": b"",
            "http_version": "1.1",
            "scheme": "http",
            "server": ("test", 80),
            "client": ("test", 123),
        }
        await response(scope, receive, send)

    def test_disconnect_closes_active_provider_resource(self):
        closed = threading.Event()
        producer_stopped = threading.Event()

        class BlockingResponse:
            def close(self):
                closed.set()

        def source():
            try:
                with managed_provider_resource(BlockingResponse()):
                    yield "event: delta\ndata: {}\n\n"
                    closed.wait(timeout=2)
            finally:
                producer_stopped.set()

        downstream = iter_sse_with_keepalive(source(), interval_seconds=0.01)
        self.assertIn("event: delta", next(downstream))
        downstream.close()

        self.assertTrue(closed.wait(timeout=1))
        self.assertTrue(producer_stopped.wait(timeout=1))

    def test_sse_response_is_closed_after_normal_exhaustion(self):
        response = FakeSSEResponse("event: final\ndata: {}\n\n")
        response.closed = False
        response.close = lambda: setattr(response, "closed", True)

        self.assertEqual(list(iter_sse_events(response)), [("final", "{}")])
        self.assertTrue(response.closed)

    def test_real_asgi_disconnect_closes_model_provider_resource(self):
        entered = threading.Event()
        closed = threading.Event()
        released = threading.Event()
        producer_stopped = threading.Event()

        class BlockingResponse:
            def close(self):
                closed.set()
                released.set()

        def source():
            try:
                with managed_provider_resource(BlockingResponse()):
                    entered.set()
                    released.wait(timeout=2)
                    yield {"type": "delta", "text": "late"}
            finally:
                producer_stopped.set()

        response = streaming_model_response(source(), "OpenAI")
        anyio.run(self._disconnect_response, response, entered)

        self.assertTrue(closed.wait(timeout=1))
        self.assertTrue(producer_stopped.wait(timeout=1))

    def test_real_asgi_disconnect_stops_consensus_before_followup_work(self):
        entered = threading.Event()
        closed = threading.Event()
        released = threading.Event()
        producer_stopped = threading.Event()
        followup_work = threading.Event()

        class BlockingResponse:
            def close(self):
                closed.set()
                released.set()

        def source():
            try:
                with managed_provider_resource(BlockingResponse()):
                    entered.set()
                    released.wait(timeout=2)
                    yield "event: delta\ndata: {}\n\n"
                    followup_work.set()
            finally:
                producer_stopped.set()

        response = keepalive_streaming_response(source())
        anyio.run(self._disconnect_response, response, entered)

        self.assertTrue(closed.wait(timeout=1))
        self.assertTrue(producer_stopped.wait(timeout=1))
        self.assertFalse(followup_work.is_set())

    def test_cancelled_provider_wrapper_does_not_emit_a_failure_result(self):
        cancellation = ProviderCancellation()
        cancellation.cancel()

        with bind_provider_cancellation(cancellation):
            with self.assertRaises(ProviderCancelled):
                list(stream_model_query("openai", "question", "key"))


class SSEPackTests(unittest.TestCase):
    def test_pack_roundtrip(self):
        packed = sse_pack("delta", {"text": "Hällo\nWelt"})
        self.assertTrue(packed.startswith("event: delta\ndata: "))
        self.assertTrue(packed.endswith("\n\n"))
        events = parse_sse_text(packed)
        self.assertEqual(events, [("delta", {"text": "Hällo\nWelt"})])

    def test_pack_normalizes_firestore_timestamp_in_final_metadata(self):
        timestamp = DatetimeWithNanoseconds.from_rfc3339(
            "2026-08-26T11:12:39.123456789Z"
        )

        packed = sse_pack("final", {
            "bookmark_persisted": True,
            "bookmark_meta": {"id": "b_run", "timestamp": timestamp},
        })

        events = parse_sse_text(packed)
        self.assertEqual(events[0][0], "final")
        self.assertTrue(events[0][1]["bookmark_persisted"])
        self.assertEqual(events[0][1]["bookmark_meta"]["id"], "b_run")
        self.assertEqual(
            events[0][1]["bookmark_meta"]["timestamp"],
            "2026-08-26T11:12:39.123456+00:00",
        )

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
        secret = "owner@example.test|provider-body-private"

        def gen():
            yield {"type": "delta", "text": "x"}
            raise RuntimeError(secret)

        response = streaming_model_response(gen(), "Mistral", {"key_used": "User API Key"})
        with self.assertLogs("app.services.llm.streaming", level="ERROR") as logs:
            raw = collect_sse_body(response)
        events = parse_sse_text(raw)
        self.assertEqual(events[-1][0], "final")
        final = events[-1][1]
        self.assertIn("Mistral could not complete this request", final["error"])
        self.assertEqual(final["error_code"], "provider_stream_failed")
        self.assertNotIn("error_detail", final)
        self.assertEqual(final["key_used"], "User API Key")
        self.assertNotIn(secret, raw)
        self.assertNotIn(secret, "\n".join(logs.output))
        self.assertIn("RuntimeError", "\n".join(logs.output))


def _openrouter_sse(events) -> str:
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"


class FakeOpenRouterHTTP(FakeSSEResponse):
    status_code = 200

    def __init__(self, events):
        super().__init__(_openrouter_sse(events))
        self.closed = False

    def close(self):
        self.closed = True


class OpenRouterStreamTests(unittest.TestCase):
    def _run(self, events, provider="openai"):
        fake = FakeOpenRouterHTTP(events)
        with mock.patch("app.services.llm.streaming.requests.post", return_value=fake) as post:
            result = list(_stream_openrouter_chat_completion(
                api_key="sk-or-test",
                payload={"model": "openai/gpt-5.4-mini", "provider": {"zdr": True}},
                provider=provider,
            ))
        return result, fake, post

    def test_delta_final_and_url_citation_use_the_common_contract(self):
        annotations = [{
            "type": "url_citation",
            "url_citation": {
                "url": "https://example.com/a",
                "title": "Example",
                "content": "Hello",
                "start_index": 0,
                "end_index": 5,
            },
        }]
        events, fake, post = self._run([
            {"choices": [{"delta": {"content": "Hel"}}]},
            {"choices": [{"delta": {"content": "lo", "annotations": annotations}, "finish_reason": "stop"}]},
        ])
        self.assertEqual([event["type"] for event in events], ["delta", "delta", "final"])
        self.assertEqual(events[-1]["result"]["text"], "Hello [S1]")
        self.assertEqual(events[-1]["result"]["sources"][0]["provider"], "openai")
        self.assertTrue(fake.closed)
        payload = post.call_args.kwargs["json"]
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["provider"], {"zdr": True})

    def test_reasoning_only_length_cutoff_is_a_structured_error(self):
        events, _, _ = self._run([
            {"choices": [{"delta": {"reasoning": "thinking"}}]},
            {"choices": [{"delta": {}, "finish_reason": "length"}]},
        ])
        self.assertEqual([event["type"] for event in events], ["reasoning", "final"])
        self.assertEqual(events[-1]["result"]["error_code"], "empty_reasoning_response")
        self.assertIn("ran out of output tokens", events[-1]["result"]["error"])

    def test_low_level_parser_accepts_content_blocks_and_done(self):
        fake = FakeOpenRouterHTTP([
            {"choices": [{"delta": {"content": [{"type": "text", "text": "Hi"}]}}]},
        ])
        with mock.patch("app.services.llm.streaming.requests.post", return_value=fake):
            events = list(_iter_openrouter_chunks(api_key="key", payload={"model": "m"}))
        self.assertEqual(events, [{"type": "delta", "text": "Hi"}])


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
        with mock.patch(
            "app.services.llm.consensus_engine._stream_consensus_engine",
            side_effect=fake_engine,
        ) as patched:
            events = list(stream_consensus(
                "Q?", "a", "b", None, None, None, None,
                excluded_models=[],
                consensus_model="OpenAI",
                api_keys={"OpenRouter": "sk-or-test"},
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
        self.assertEqual(call_count, 3)
        self.assertEqual(events[-1]["text"], "Consensus error: provider request failed.")
        self.assertTrue(events[-1]["error"])

    def test_empty_stream_counts_as_failure(self):
        def fake_engine(consensus_model, api_keys, prompt):
            return iter(())

        events, call_count = self._run(fake_engine)
        self.assertEqual(call_count, 3)
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
