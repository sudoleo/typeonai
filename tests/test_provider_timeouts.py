"""Regressionstests: alle Provider-HTTP-Calls muessen ein Timeout setzen.

Ein requests.post ohne timeout blockiert bei einem haengenden Upstream
dauerhaft einen Threadpool-Worker (so geschehen bei query_claude). Der
AST-Audit nagelt das fuer alle requests.post/get-Aufrufe unter
app/services/llm/ und app/api/routers/ fest; der Funktionstest prueft den
konkreten Anthropic-Pfad zusaetzlich zur Laufzeit.
"""

import ast
from pathlib import Path
from unittest.mock import patch

from app.services.llm import engines

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDITED_DIRS = [
    REPO_ROOT / "app" / "services" / "llm",
    REPO_ROOT / "app" / "api" / "routers",
]


def test_query_claude_sets_timeout():
    with patch.object(engines.requests, "post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "content": [{"type": "text", "text": "hi"}]
        }
        engines.query_claude("hello", api_key="sk-test")

    assert mock_post.called
    assert mock_post.call_args.kwargs.get("timeout") == 120


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
