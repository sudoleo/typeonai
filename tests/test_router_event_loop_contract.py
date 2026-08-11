"""Regression coverage for blocking SDK work in FastAPI route handlers."""

import ast
import asyncio
from pathlib import Path
import threading

import httpx

import main
from app.api.routers import chat


REPO_ROOT = Path(__file__).resolve().parents[1]
ROUTERS_DIR = REPO_ROOT / "app" / "api" / "routers"


def _is_route_handler(node: ast.AsyncFunctionDef) -> bool:
    return any(
        isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and isinstance(decorator.func.value, ast.Name)
        and decorator.func.value.id == "router"
        for decorator in node.decorator_list
    )


def test_async_route_handlers_have_real_async_work():
    """An ``async def`` without await would run blocking SDK I/O on uvicorn's loop."""
    violations = []
    for path in sorted(ROUTERS_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.AsyncFunctionDef) or not _is_route_handler(node):
                continue
            if not any(
                isinstance(child, (ast.Await, ast.AsyncFor, ast.AsyncWith))
                for child in ast.walk(node)
            ):
                violations.append(f"{path.name}:{node.lineno}:{node.name}")
    assert not violations, (
        "Async route handlers without await bypass FastAPI's worker threadpool: "
        f"{violations}"
    )


def test_prepare_firestore_bundle_does_not_serialize_the_event_loop(monkeypatch):
    """Two slow tier reads must enter the sync-handler threadpool concurrently."""
    rendezvous = threading.Barrier(2)

    monkeypatch.setattr(chat, "verify_user_token", lambda token: "phase-3-user")

    def slow_tier_read(uid):
        rendezvous.wait(timeout=2)
        return True

    monkeypatch.setattr(chat, "is_user_pro", slow_tier_read)

    async def exercise():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            payload = {
                "question": "Can these requests overlap?",
                "id_token": "token",
                "useOwnKeys": True,
            }
            return await asyncio.gather(
                client.post("/prepare", json=payload),
                client.post("/prepare", json=payload),
            )

    responses = asyncio.run(exercise())
    assert [response.status_code for response in responses] == [200, 200]

