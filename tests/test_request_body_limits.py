"""ASGI body cap must reject declared and chunked payloads before parsing."""

import asyncio

from starlette.responses import StreamingResponse

from app.core.request_limits import RequestBodyLimitMiddleware


def run_asgi(messages, *, headers=(), limit=8):
    received_by_app = []
    sent = []

    async def inner(scope, receive, send):
        received_by_app.append(await receive())
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    queue = list(messages)

    async def receive():
        if queue:
            return queue.pop(0)
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/json",
        "headers": list(headers),
    }
    asyncio.run(RequestBodyLimitMiddleware(inner, limit)(scope, receive, send))
    return sent, received_by_app


def test_declared_oversized_body_is_rejected_without_reading_or_parsing():
    sent, received = run_asgi(
        [{"type": "http.request", "body": b"ignored", "more_body": False}],
        headers=[(b"content-length", b"9")],
    )

    assert sent[0]["status"] == 413
    assert received == []


def test_chunked_body_is_counted_across_receive_messages():
    sent, received = run_asgi(
        [
            {"type": "http.request", "body": b"12345", "more_body": True},
            {"type": "http.request", "body": b"6789", "more_body": False},
        ]
    )

    assert sent[0]["status"] == 413
    assert received == []


def test_exact_limit_body_is_replayed_once_to_the_application():
    sent, received = run_asgi(
        [
            {"type": "http.request", "body": b"1234", "more_body": True},
            {"type": "http.request", "body": b"5678", "more_body": False},
        ]
    )

    assert sent[0]["status"] == 204
    assert received == [
        {"type": "http.request", "body": b"12345678", "more_body": False}
    ]


def test_replayed_body_does_not_synthesize_disconnect_for_delayed_stream():
    sent = []
    request_messages = [
        {"type": "http.request", "body": b"{}", "more_body": False}
    ]

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/ask_openai",
        "raw_path": b"/ask_openai",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-length", b"2")],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }

    async def run_streaming_scenario():
        connection_closed = asyncio.Event()

        async def receive():
            if request_messages:
                return request_messages.pop(0)
            await connection_closed.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)

        async def inner(inner_scope, inner_receive, inner_send):
            assert await inner_receive() == {
                "type": "http.request",
                "body": b"{}",
                "more_body": False,
            }

            async def delayed_sse():
                await asyncio.sleep(0.01)
                yield b'event: final\ndata: {"response":"ok"}\n\n'

            response = StreamingResponse(
                delayed_sse(), media_type="text/event-stream"
            )
            await response(inner_scope, inner_receive, inner_send)

        await RequestBodyLimitMiddleware(inner, max_body_bytes=8)(
            scope, receive, send
        )

    asyncio.run(run_streaming_scenario())

    body_messages = [
        message for message in sent if message["type"] == "http.response.body"
    ]
    assert body_messages == [
        {
            "type": "http.response.body",
            "body": b'event: final\ndata: {"response":"ok"}\n\n',
            "more_body": True,
        },
        {"type": "http.response.body", "body": b"", "more_body": False},
    ]
