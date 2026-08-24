import json

from app.services import telegram_notifier


class Response:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return b"{}"


def test_seo_review_notification_is_sent_even_without_open_decisions(monkeypatch):
    captured = {}
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    monkeypatch.setenv("SEO_ADMIN_URL", "https://example.test/admin#seo")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(telegram_notifier, "urlopen", fake_urlopen)
    result = telegram_notifier.send_seo_review_notification({
        "status": "completed",
        "summary": "Everything is stable.",
        "pages": [{"page_id": "a"}],
        "groups": {"manual_improvement": []},
        "editorial_decisions": {},
        "proposed_topic_brief": None,
    })

    assert result["status"] == "sent"
    assert captured["url"].endswith("/bottest-token/sendMessage")
    assert "Everything is stable." in captured["payload"]["text"]
    assert "Editorial decisions open: 0" in captured["payload"]["text"]
    assert "https://example.test/admin#seo" in captured["payload"]["text"]


def test_seo_review_notification_names_the_judge_failure_and_findings(monkeypatch):
    captured = {}
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    # Only the critical-alert chat id is configured; the review must still land.
    monkeypatch.setenv("CRITICAL_ERROR_TELEGRAM_CHAT_ID", "456")

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)
        return Response()

    monkeypatch.setattr(telegram_notifier, "urlopen", fake_urlopen)
    result = telegram_notifier.send_seo_review_notification({
        "status": "completed",
        "summary": "Reviewed 12 SEO pages.",
        "pages": [{"page_id": "a"}],
        "groups": {"manual_improvement": []},
        "editorial_decisions": {},
        "judge_called": False,
        "judge_error": "The portfolio judge returned invalid JSON.",
        "findings": {"positive": ["One page is being shown."], "negative": ["Most pages are commodity answers."]},
        "status_counts": {"emerging": 11, "opportunity": 1},
        "delta": {"comparable": True, "changed": [{"title": "Agent pricing", "from": "emerging", "to": "opportunity"}], "new_pages": []},
    })

    text = captured["payload"]["text"]
    assert result["status"] == "sent"
    assert captured["payload"]["chat_id"] == "456"
    assert "Portfolio judge: FAILED" in text
    assert "Most pages are commodity answers." in text
    assert "emerging 11" in text
    assert "Agent pricing: emerging -> opportunity" in text


def test_seo_review_notification_is_recorded_as_skipped_without_config(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.delenv("CRITICAL_ERROR_TELEGRAM_CHAT_ID", raising=False)
    result = telegram_notifier.send_seo_review_notification({"status": "error"})
    assert result["status"] == "skipped_not_configured"


def test_critical_error_notification_redacts_and_deduplicates(monkeypatch):
    captured = []
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    monkeypatch.setenv("CRITICAL_ERROR_TELEGRAM_CHAT_ID", "456")
    telegram_notifier._critical_seen.clear()
    telegram_notifier._critical_sent_at.clear()

    def fake_urlopen(request, timeout):
        captured.append(json.loads(request.data))
        return Response()

    monkeypatch.setattr(telegram_notifier, "urlopen", fake_urlopen)
    report = {
        "source": "browser",
        "type": "run_failed",
        "phase": "model_fanout",
        "path": "/app",
        "message": "All models failed with token=secret-value",
        "details": "Authorization: Bearer abc.def.ghi",
    }

    first = telegram_notifier.send_critical_error_notification(report)
    second = telegram_notifier.send_critical_error_notification(report)

    assert first["status"] == "sent"
    assert second["status"] == "deduplicated"
    assert len(captured) == 1
    assert captured[0]["chat_id"] == "456"
    assert "secret-value" not in captured[0]["text"]
    assert "abc.def.ghi" not in captured[0]["text"]
    assert captured[0]["text"].startswith("🚨 consens.io critical error")


def test_new_user_registration_notification_is_pii_free(monkeypatch):
    captured = []
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    monkeypatch.setenv("CRITICAL_ERROR_TELEGRAM_CHAT_ID", "456")
    monkeypatch.setenv("ENVIRONMENT", "staging")
    telegram_notifier._registration_seen.clear()

    def fake_urlopen(request, timeout):
        captured.append(json.loads(request.data))
        return Response()

    monkeypatch.setattr(telegram_notifier, "urlopen", fake_urlopen)
    result = telegram_notifier.send_new_user_registration_notification(
        "email/password", "private-firebase-uid"
    )
    duplicate = telegram_notifier.send_new_user_registration_notification(
        "email/password", "private-firebase-uid"
    )

    assert result["status"] == "sent"
    assert duplicate["status"] == "deduplicated"
    assert len(captured) == 1
    assert captured[0]["chat_id"] == "456"
    assert captured[0]["text"].startswith("👤 consens.io new user registered")
    assert "Method: email/password" in captured[0]["text"]
    assert "Environment: staging" in captured[0]["text"]
    assert "@" not in captured[0]["text"]


def test_new_user_registration_notification_is_skipped_without_config(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.delenv("CRITICAL_ERROR_TELEGRAM_CHAT_ID", raising=False)

    result = telegram_notifier.send_new_user_registration_notification(
        "email/password", "private-firebase-uid"
    )

    assert result["status"] == "skipped_not_configured"
