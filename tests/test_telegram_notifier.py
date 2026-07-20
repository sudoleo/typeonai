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


def test_seo_review_notification_is_recorded_as_skipped_without_config(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    result = telegram_notifier.send_seo_review_notification({"status": "error"})
    assert result["status"] == "skipped_not_configured"
