"""Source-level guardrails for content-free runtime exception logging."""

from __future__ import annotations

import ast
import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace

import main
from app.core.observability import safe_exception, safe_traceback
from app.services import mailer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = PROJECT_ROOT / "app"
LOG_METHODS = {"debug", "info", "warning", "error", "critical", "exception", "log"}
LOG_OBJECTS = {"logging", "logger", "log"}


def _is_logging_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in LOG_METHODS
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in LOG_OBJECTS
    )


# Die einzigen erlaubten Projektionen einer gefangenen Exception in ein Log:
# die Kategorie (safe_exception/type) und die Herkunftsframes
# (safe_traceback). Alles andere - str(exc), repr, exc_info, format_exc -
# transportiert die Message und damit potenziell Frage-, Modell- oder
# Provider-Text.
_SAFE_EXCEPTION_PROJECTIONS = {"safe_exception", "safe_traceback", "type"}


def _is_safe_exception_projection(node: ast.AST, exception_name: str) -> bool:
    if not isinstance(node, ast.Call) or len(node.args) != 1:
        return False
    if not isinstance(node.args[0], ast.Name) or node.args[0].id != exception_name:
        return False
    return (
        isinstance(node.func, ast.Name)
        and node.func.id in _SAFE_EXCEPTION_PROJECTIONS
    )


def _contains_unsafe_exception_reference(node: ast.AST, exception_name: str) -> bool:
    if _is_safe_exception_projection(node, exception_name):
        return False
    if isinstance(node, ast.Name) and node.id == exception_name:
        return True
    return any(
        _contains_unsafe_exception_reference(child, exception_name)
        for child in ast.iter_child_nodes(node)
    )


def test_runtime_never_emits_raw_exception_tracebacks_or_messages():
    violations = []
    runtime_paths = [PROJECT_ROOT / "main.py", *sorted(APP_ROOT.rglob("*.py"))]
    for path in runtime_paths:
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source, filename=str(path))
        relative = path.relative_to(PROJECT_ROOT)

        for node in ast.walk(tree):
            if _is_logging_call(node):
                if node.func.attr == "exception":
                    violations.append(f"{relative}:{node.lineno}: logging.exception")
                for keyword in node.keywords:
                    if keyword.arg in {"exc_info", "stack_info"} and not (
                        isinstance(keyword.value, ast.Constant)
                        and keyword.value.value in {False, None}
                    ):
                        violations.append(
                            f"{relative}:{node.lineno}: {keyword.arg}=..."
                        )

            if not isinstance(node, ast.ExceptHandler) or not node.name:
                continue
            for child in ast.walk(node):
                if not _is_logging_call(child):
                    continue
                arguments = [*child.args, *(kw.value for kw in child.keywords)]
                if any(
                    _contains_unsafe_exception_reference(arg, node.name)
                    for arg in arguments
                ):
                    violations.append(
                        f"{relative}:{child.lineno}: raw caught exception {node.name}"
                    )

    assert violations == [], "\n".join(violations)


def test_global_exception_handler_logs_and_alerts_only_safe_categories(
    monkeypatch, caplog
):
    secret = "private.user@example.test token=server-secret"
    captured = []
    monkeypatch.setattr(
        main,
        "send_critical_error_notification",
        lambda report: captured.append(report),
    )
    request = SimpleNamespace(
        method="GET",
        url=SimpleNamespace(path=f"/api/private/{secret}"),
        scope={"route": SimpleNamespace(path="/api/private/{item_id}")},
    )

    with caplog.at_level(logging.ERROR):
        response = asyncio.run(
            main.handle_unexpected_exception(request, RuntimeError(secret))
        )
        asyncio.run(response.background())

    assert response.status_code == 500
    assert captured == [{
        "source": "server",
        "type": "RuntimeError",
        "phase": "request",
        "message": "Unhandled server exception.",
        "path": "GET /api/private/{item_id}",
    }]
    assert secret not in caplog.text


def test_safe_traceback_reports_where_not_what():
    """Die Kategorie allein macht einen unerwarteten internen Fehler in
    Produktion unauffindbar. safe_traceback ergaenzt genau die fehlende
    Haelfte - Frame-Koordinaten im eigenen Code - und niemals die Message,
    die Argumente oder die Quellzeile."""
    secret = "private-question-text-and-provider-body"

    def inner():
        raise IndexError(secret)

    def outer():
        inner()

    try:
        outer()
    except IndexError as exc:
        rendered = safe_traceback(exc)

    assert secret not in rendered
    # Der innerste Frame steht am Ende: dort ist der Fehler entstanden.
    assert rendered.endswith("inner")
    assert "test_logging_redaction_contract.py:" in rendered
    # Nur Basenames, nie der ganze Pfad des Hosts.
    assert "\\" not in rendered and "/" not in rendered


def test_safe_traceback_is_bounded_and_survives_a_bare_exception():
    assert safe_traceback(RuntimeError("never raised")) == "-"

    def recurse(depth):
        if depth:
            return recurse(depth - 1)
        raise ValueError("deep")

    try:
        recurse(40)
    except ValueError as exc:
        rendered = safe_traceback(exc)

    assert len(rendered.split(">")) == 12


def test_safe_exception_never_projects_arbitrary_string_error_codes():
    secret = "private-user-token-123"
    error = RuntimeError("private provider body")
    error.code = secret

    assert safe_exception(error) == "RuntimeError"


def test_mail_delivery_error_log_redacts_recipient_credentials_and_provider_body(
    monkeypatch, caplog
):
    recipient = "private.recipient@example.test"
    password = "smtp-password-secret"
    provider_body = "550 rejected message body with private prompt"
    monkeypatch.setattr(mailer, "mock_llm_enabled", lambda: False)
    monkeypatch.setenv("SMTP_HOST", "smtp.example.test")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_USER", "mailer-user@example.test")
    monkeypatch.setenv("SMTP_PASSWORD", password)
    monkeypatch.setenv("MAIL_FROM", "sender@example.test")

    def fail_smtp(*_args, **_kwargs):
        raise RuntimeError(
            f"recipient={recipient} password={password} body={provider_body}"
        )

    monkeypatch.setattr(mailer.smtplib, "SMTP", fail_smtp)
    message = mailer._base_message(
        recipient,
        "Private subject",
        "Private plain body",
        "<p>Private HTML body</p>",
    )

    with caplog.at_level(logging.ERROR):
        assert mailer._deliver(message) is False

    logged = caplog.text
    assert "category=RuntimeError" in logged
    for secret in (
        recipient,
        password,
        provider_body,
        "mailer-user@example.test",
        "sender@example.test",
        "Private subject",
        "Private plain body",
        "Private HTML body",
    ):
        assert secret not in logged
