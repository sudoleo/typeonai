import os
import unittest
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import app.core.config as cfg
from app.services import watch_service
from app.services import mailer, watch_scheduler
from app.services.watch_service import WatchError


class FakeSnap:
    def __init__(self, data):
        self._data = data

    @property
    def exists(self):
        return self._data is not None

    def to_dict(self):
        return dict(self._data) if self._data is not None else None


class FakeDocRef:
    def __init__(self, store, key):
        self.store = store
        self.id = key

    def get(self, transaction=None):
        return FakeSnap(self.store.get(self.id))

    def set(self, data):
        self.store[self.id] = dict(data)

    def update(self, data):
        self.store[self.id].update(data)

    def delete(self):
        self.store.pop(self.id, None)


class FakeQueryDoc:
    def __init__(self, store, key):
        self.store = store
        self.id = key
        self.reference = FakeDocRef(store, key)

    def to_dict(self):
        return dict(self.store[self.id])


class FakeQuery:
    def __init__(self, store, field, value):
        self.store = store
        self.field = field
        self.value = value

    def stream(self):
        for key, data in list(self.store.items()):
            if self.field is None or data.get(self.field) == self.value:
                yield FakeQueryDoc(self.store, key)


class FakeCollection:
    def __init__(self, store):
        self.store = store

    def document(self, key):
        return FakeDocRef(self.store, key)

    def where(self, field, op, value):
        assert op == "=="
        return FakeQuery(self.store, field, value)


class FakeDb:
    def __init__(self):
        self.stores = defaultdict(dict)

    def collection(self, name):
        return FakeCollection(self.stores[name])


class FakeTransaction:
    def update(self, ref, data):
        ref.update(data)

    def set(self, ref, data):
        ref.set(data)


def share(owner="u1", slug="question", score=60):
    return {
        "owner_uid": owner,
        "slug": slug,
        "status": "active",
        "question": "Will this consensus change?",
        "question_hash": "abc",
        "differences_data": {"agreement": {"score": score}},
    }


class WatchCrudTests(unittest.TestCase):
    def setUp(self):
        self.db = FakeDb()
        self.share_id = "A" * 16
        self.db.stores["shares"][self.share_id] = share()
        self.old_limits = cfg.get_limits_config()

    def tearDown(self):
        cfg.apply_limits(self.old_limits)

    def test_free_create_list_update_pause_delete(self):
        created = watch_service.create_watch("u1", share_id=self.share_id, interval="weekly", is_pro=False, db=self.db)
        self.assertEqual(created["status"], "active")
        self.assertEqual(created["last_agreement_score"], 60)
        self.assertEqual(len(watch_service.list_watches("u1", db=self.db)), 1)

        paused = watch_service.update_watch("u1", created["id"], {"status": "paused"}, False, db=self.db)
        self.assertEqual(paused["status"], "paused")
        monthly = watch_service.update_watch("u1", created["id"], {"interval": "monthly"}, False, db=self.db)
        self.assertEqual(monthly["interval"], "monthly")
        watch_service.delete_watch("u1", created["id"], db=self.db)
        self.assertEqual(watch_service.list_watches("u1", db=self.db), [])

    def test_free_daily_requires_pro(self):
        with self.assertRaisesRegex(WatchError, "Daily watches require Pro"):
            watch_service.create_watch("u1", share_id=self.share_id, interval="daily", is_pro=False, db=self.db)
        created = watch_service.create_watch("u1", share_id=self.share_id, interval="daily", is_pro=True, db=self.db)
        self.assertEqual(created["interval"], "daily")

    def test_free_and_pro_active_limits(self):
        cfg.apply_limits({**self.old_limits, "watch_free_active_limit": 1, "watch_pro_active_limit": 2})
        first = watch_service.create_watch("u1", share_id=self.share_id, interval="weekly", is_pro=False, db=self.db)
        second_share_id = "B" * 16
        self.db.stores["shares"][second_share_id] = share(slug="two")
        with self.assertRaisesRegex(WatchError, "limit"):
            watch_service.create_watch("u1", share_id=second_share_id, interval="weekly", is_pro=False, db=self.db)
        second = watch_service.create_watch("u1", share_id=second_share_id, interval="daily", is_pro=True, db=self.db)
        self.assertNotEqual(first["id"], second["id"])

    def test_cannot_watch_foreign_or_duplicate_share(self):
        foreign_id = "F" * 16
        self.db.stores["shares"][foreign_id] = share(owner="u2")
        with self.assertRaisesRegex(WatchError, "own shares"):
            watch_service.create_watch("u1", share_id=foreign_id, interval="weekly", is_pro=False, db=self.db)
        watch_service.create_watch("u1", share_id=self.share_id, interval="weekly", is_pro=False, db=self.db)
        with self.assertRaisesRegex(WatchError, "already watched"):
            watch_service.create_watch("u1", share_id=self.share_id, interval="weekly", is_pro=True, db=self.db)

    def test_update_rejects_unknown_fields_and_owner(self):
        created = watch_service.create_watch("u1", share_id=self.share_id, interval="weekly", is_pro=False, db=self.db)
        with self.assertRaises(WatchError):
            watch_service.update_watch("u1", created["id"], {"owner_uid": "u2"}, False, db=self.db)
        with self.assertRaisesRegex(WatchError, "own watches"):
            watch_service.update_watch("u2", created["id"], {"status": "paused"}, False, db=self.db)


class UnsubscribeTokenTests(unittest.TestCase):
    def setUp(self):
        self.db = FakeDb()
        self.now = datetime(2026, 7, 11, tzinfo=timezone.utc)
        self.env = patch.dict(os.environ, {"WATCH_UNSUBSCRIBE_SECRET": "test-secret"})
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_valid_token_pauses_without_login(self):
        self.db.stores["watches"]["w1"] = {"status": "active", "question": "Q"}
        token = watch_service.make_unsubscribe_token("w1", now=self.now)
        with patch.object(watch_service, "utcnow", return_value=self.now):
            result = watch_service.unsubscribe(token, db=self.db)
        self.assertEqual(result["watch_id"], "w1")
        self.assertEqual(self.db.stores["watches"]["w1"]["status"], "paused")

    def test_invalid_and_expired_tokens(self):
        with self.assertRaisesRegex(WatchError, "invalid"):
            watch_service.parse_unsubscribe_token("garbage", now=self.now)
        token = watch_service.make_unsubscribe_token("w1", now=self.now, max_age_days=1)
        with self.assertRaisesRegex(WatchError, "expired"):
            watch_service.parse_unsubscribe_token(token, now=self.now + timedelta(days=2))

    def test_tampered_token_is_invalid(self):
        token = watch_service.make_unsubscribe_token("w1", now=self.now)
        with self.assertRaisesRegex(WatchError, "invalid"):
            watch_service.parse_unsubscribe_token(token[:-1] + ("A" if token[-1] != "A" else "B"), now=self.now)


class SchedulerSafetyTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 7, 11, 10, tzinfo=timezone.utc)
        self.watch_store = {"w1": {
            "status": "active", "next_run_at": self.now - timedelta(minutes=1),
            "claimed_until": None, "consecutive_failures": 0, "interval": "weekly",
        }}
        self.budget_store = {}
        self.watch_ref = FakeDocRef(self.watch_store, "w1")
        self.budget_ref = FakeDocRef(self.budget_store, "day")

    def test_claim_transaction_prevents_double_run(self):
        first, reason = watch_service._claim_in_transaction(
            FakeTransaction(), self.watch_ref, self.budget_ref, self.now, 50
        )
        second, second_reason = watch_service._claim_in_transaction(
            FakeTransaction(), self.watch_ref, self.budget_ref, self.now, 50
        )
        self.assertEqual(reason, "claimed")
        self.assertIsNotNone(first)
        self.assertIsNone(second)
        self.assertEqual(second_reason, "claimed")
        self.assertEqual(self.budget_store["day"]["count"], 1)

    def test_daily_budget_leaves_watch_due(self):
        self.budget_store["day"] = {"count": 2}
        claimed, reason = watch_service._claim_in_transaction(
            FakeTransaction(), self.watch_ref, self.budget_ref, self.now, 2
        )
        self.assertIsNone(claimed)
        self.assertEqual(reason, "budget")
        self.assertLessEqual(self.watch_store["w1"]["next_run_at"], self.now)
        self.assertIsNone(self.watch_store["w1"]["claimed_until"])

    def test_auto_pause_only_on_third_failure(self):
        db = FakeDb()
        claimed = {
            "interval": "weekly", "consecutive_failures": 0,
            "status": "active", "claimed_until": self.now + timedelta(minutes=15),
        }
        db.stores["watches"]["w1"] = dict(claimed)
        self.assertFalse(watch_service.fail_watch_run("w1", claimed, now=self.now, db=db))
        claimed.update(db.stores["watches"]["w1"])
        self.assertFalse(watch_service.fail_watch_run("w1", claimed, now=self.now, db=db))
        claimed.update(db.stores["watches"]["w1"])
        self.assertTrue(watch_service.fail_watch_run("w1", claimed, now=self.now, db=db))
        self.assertEqual(db.stores["watches"]["w1"]["status"], "paused_error")

    def test_notification_threshold(self):
        self.assertTrue(watch_scheduler.should_notify(60, 61, True, "major"))
        self.assertTrue(watch_scheduler.should_notify(60, 75, False, "minor"))
        self.assertFalse(watch_scheduler.should_notify(60, 74, True, "minor"))

    def test_mock_llm_watch_pipeline(self):
        with patch.dict(os.environ, {"MOCK_LLM": "1"}):
            result = watch_scheduler.execute_watch(
                "When was the Eiffel Tower completed?", "An older consensus answer."
            )
        self.assertIn("Mock consensus", result["consensus"])
        self.assertIsInstance(result["agreement_score"], int)
        self.assertFalse(result["changed"])


class MailerTests(unittest.TestCase):
    def test_change_mail_is_multipart_with_unsubscribe(self):
        message = mailer.build_change_message(
            recipient="owner@example.test", question="A question", old_score=50,
            new_score=70, summary="A central conclusion changed.",
            share_url="https://consens.io/s/q-id", unsubscribe_url="https://consens.io/watch/unsubscribe?token=x",
        )
        self.assertTrue(message.is_multipart())
        self.assertIn("text/plain", [part.get_content_type() for part in message.walk()])
        self.assertIn("text/html", [part.get_content_type() for part in message.walk()])
        self.assertIn("unsubscribe?token=x", message.as_string())


class SchedulerLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_pause_mail_emitted_exactly_on_third_failure(self):
        claimed = {
            "owner_uid": "u1", "share_id": "A" * 16, "share_slug": "q",
            "question": "Q", "interval": "weekly", "consecutive_failures": 0,
        }
        with (
            patch.object(watch_service, "acquire_worker_lease", return_value=True),
            patch.object(watch_service, "release_worker_lease"),
            patch.object(watch_service, "list_due_watch_ids", return_value=["w1"]),
            patch.object(watch_service, "claim_watch", return_value=(claimed, "claimed")),
            patch.object(watch_scheduler, "execute_watch", side_effect=RuntimeError("provider failed")),
            patch.object(watch_service, "fail_watch_run", side_effect=[False, False, True]),
            patch.object(watch_scheduler, "_send_paused_mail", new_callable=AsyncMock) as send_paused,
        ):
            await watch_scheduler.run_watch_tick()
            await watch_scheduler.run_watch_tick()
            await watch_scheduler.run_watch_tick()
        send_paused.assert_awaited_once_with("w1", claimed)
