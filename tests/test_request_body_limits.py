"""ASGI body cap must reject declared and chunked payloads before parsing."""

import asyncio

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
