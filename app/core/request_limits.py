"""ASGI request-body limits applied before framework JSON/form parsing."""

from __future__ import annotations

import json
import os


DEFAULT_MAX_REQUEST_BODY_BYTES = 16 * 1024 * 1024


def configured_max_request_body_bytes() -> int:
    raw = os.environ.get("MAX_REQUEST_BODY_BYTES", "").strip()
    if not raw:
        return DEFAULT_MAX_REQUEST_BODY_BYTES
    try:
        parsed = int(raw)
    except ValueError as exc:
        raise RuntimeError("MAX_REQUEST_BODY_BYTES must be an integer") from exc
    if parsed < 1024 or parsed > 32 * 1024 * 1024:
        raise RuntimeError(
            "MAX_REQUEST_BODY_BYTES must be between 1024 and 33554432"
        )
    return parsed


class RequestBodyLimitMiddleware:
    """Reject oversized Content-Length and chunked bodies with HTTP 413.

    The body is buffered only up to the configured cap and replayed once to
    Starlette. Therefore JSON parsing, authentication dependencies and route
    handlers never see an oversized payload.
    """

    def __init__(self, app, max_body_bytes: int | None = None):
        self.app = app
        self.max_body_bytes = (
            configured_max_request_body_bytes()
            if max_body_bytes is None
            else int(max_body_bytes)
        )
        if self.max_body_bytes < 1:
            raise ValueError("max_body_bytes must be positive")

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = {
            key.lower(): value
            for key, value in scope.get("headers", [])
        }
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                await self._reject(send, "Invalid Content-Length header.")
                return
            if declared < 0:
                await self._reject(send, "Invalid Content-Length header.")
                return
            if declared > self.max_body_bytes:
                await self._reject(send)
                return

        buffered = bytearray()
        disconnected = False
        while True:
            message = await receive()
            if message.get("type") == "http.disconnect":
                disconnected = True
                break
            if message.get("type") != "http.request":
                continue
            chunk = message.get("body", b"")
            if chunk:
                buffered.extend(chunk)
                if len(buffered) > self.max_body_bytes:
                    await self._reject(send)
                    return
            if not message.get("more_body", False):
                break

        delivered = False

        async def replay_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {
                    "type": "http.request",
                    "body": bytes(buffered),
                    "more_body": False,
                }
            return {"type": "http.disconnect"}

        if disconnected and not buffered:
            async def disconnected_receive():
                return {"type": "http.disconnect"}

            await self.app(scope, disconnected_receive, send)
            return
        await self.app(scope, replay_receive, send)

    async def _reject(self, send, detail: str = "Request body too large."):
        payload = json.dumps(
            {"error": detail}, separators=(",", ":")
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json; charset=utf-8"),
                    (b"content-length", str(len(payload)).encode("ascii")),
                    (b"cache-control", b"no-store"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": payload})
