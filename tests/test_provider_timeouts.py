"""Regressionstests: alle OpenRouter-HTTP-Calls muessen ein Timeout setzen.

Ein requests.post ohne timeout blockiert bei einem haengenden Upstream
dauerhaft einen Threadpool-Worker. Der
AST-Audit nagelt das fuer alle requests.post/get-Aufrufe unter
app/services/llm/ und app/api/routers/ fest; der Funktionstest prueft den
konkreten Anthropic-Pfad zusaetzlich zur Laufzeit.
"""

import ast
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from app.services.llm import engines, provider_runtime, streaming
from app.core import config as cfg

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDITED_DIRS = [
    REPO_ROOT / "app" / "services" / "llm",
    REPO_ROOT / "app" / "api" / "routers",
]


PROVIDERS = list(cfg.PROVIDERS)


def _call_direct_provider(provider: str):
    return engines.query_model(provider, "question", "key")


def _call_streaming_provider(provider: str):
    return list(streaming.stream_model_query(provider, "question", "key"))[-1]["result"]


@pytest.mark.parametrize(
    "provider", PROVIDERS
)
@pytest.mark.parametrize("status_code", [408, 504])
def test_real_provider_adapters_classify_http_timeout_status_without_body_leak(
    provider, status_code, caplog
):
    secret = "owner@example.test|private-provider-body"
    response = requests.Response()
    response.status_code = status_code
    response._content = secret.encode("utf-8")

    with patch.object(engines.requests, "post", return_value=response):
        result = _call_direct_provider(provider)

    assert result["error_code"] == "provider_timeout"
    assert secret not in caplog.text


@pytest.mark.parametrize(
    "provider", PROVIDERS
)
def test_real_provider_adapters_preserve_read_timeout_type(provider, caplog):
    secret = "owner@example.test|private-timeout-detail"

    with patch.object(
        engines.requests, "post", side_effect=requests.ReadTimeout(secret)
    ):
        result = _call_direct_provider(provider)

    assert result["error_code"] == "provider_timeout"
    assert secret not in caplog.text


@pytest.mark.parametrize(
    "provider", PROVIDERS
)
@pytest.mark.parametrize("status_code", [408, 504])
def test_real_streaming_adapters_classify_http_timeout_status(
    provider, status_code, caplog
):
    secret = "owner@example.test|private-stream-body"
    response = requests.Response()
    response.status_code = status_code
    response._content = secret.encode("utf-8")

    with patch.object(streaming.requests, "post", return_value=response):
        result = _call_streaming_provider(provider)

    assert result["error_code"] == "provider_timeout"
    assert secret not in caplog.text


@pytest.mark.parametrize(
    "provider", PROVIDERS
)
def test_real_streaming_adapters_preserve_read_timeout_type(provider, caplog):
    secret = "owner@example.test|private-stream-timeout"

    with patch.object(
        streaming.requests, "post", side_effect=requests.ReadTimeout(secret)
    ):
        result = _call_streaming_provider(provider)

    assert result["error_code"] == "provider_timeout"
    assert secret not in caplog.text


def test_query_model_sets_timeout():
    with patch.object(engines.requests, "post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "hi"}}]
        }
        engines.query_model("anthropic", "hello", api_key="sk-test")

    assert mock_post.called
    assert mock_post.call_args.kwargs.get("timeout") == provider_runtime.PROVIDER_HTTP_TIMEOUT


def test_openai_compatible_clients_disable_sdk_retries_and_set_timeout():
    with patch.object(provider_runtime.openai, "OpenAI") as constructor:
        provider_runtime.openai_client(
            api_key="sk-test", base_url="https://provider.invalid/v1"
        )

    constructor.assert_called_once()
    kwargs = constructor.call_args.kwargs
    assert kwargs["api_key"] == "sk-test"
    assert kwargs["base_url"] == "https://provider.invalid/v1"
    assert kwargs["max_retries"] == 0
    assert kwargs["timeout"].connect == provider_runtime.PROVIDER_CONNECT_TIMEOUT_SECONDS
    assert kwargs["timeout"].read == provider_runtime.PROVIDER_READ_TIMEOUT_SECONDS


def test_provider_modules_use_only_the_central_openai_client_factory():
    violations = []
    for directory in AUDITED_DIRS:
        for path in sorted(directory.rglob("*.py")):
            if path.name == "provider_runtime.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "OpenAI"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "openai"
                ):
                    violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert not violations, f"OpenAI SDK clients outside central factory: {violations}"


def _is_requests_call(node: ast.Call) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr in ("post", "get")
        and isinstance(func.value, ast.Name)
        and func.value.id == "requests"
    )


def _kwargs_dicts_with_timeout(tree: ast.AST) -> set:
    """Namen aller Variablen, denen ein Dict-Literal mit 'timeout'-Key
    zugewiesen wird (fuer requests.post(url, **request_kwargs)-Muster)."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):  # request_kwargs: Dict[...] = {...}
            targets = [node.target]
        else:
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        has_timeout = any(
            isinstance(key, ast.Constant) and key.value == "timeout"
            for key in node.value.keys
        )
        if not has_timeout:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


def _calls_without_timeout(path: Path) -> list:
    # utf-8-sig: einzelne Dateien (z.B. chat.py) tragen einen BOM.
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    kwargs_with_timeout = _kwargs_dicts_with_timeout(tree)
    violations = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _is_requests_call(node)):
            continue
        explicit = any(kw.arg == "timeout" for kw in node.keywords)
        via_kwargs = any(
            kw.arg is None
            and isinstance(kw.value, ast.Name)
            and kw.value.id in kwargs_with_timeout
            for kw in node.keywords
        )
        if not (explicit or via_kwargs):
            violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    return violations


def test_all_requests_calls_set_timeout():
    violations = []
    audited_files = 0
    for directory in AUDITED_DIRS:
        for path in sorted(directory.rglob("*.py")):
            audited_files += 1
            violations.extend(_calls_without_timeout(path))
    assert audited_files > 0
    assert not violations, (
        "requests.post/get ohne timeout= (blockiert Threadpool-Worker bei "
        f"haengendem Upstream): {violations}"
    )
