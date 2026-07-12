import asyncio
import os
import unittest
from types import SimpleNamespace
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.core.config as cfg
from app.api.routers import share as share_router
from app.api.routers import admin as admin_router
from app.core.rate_limit import limiter
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

    def stream(self):
        return FakeQuery(self.store, None, None).stream()


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
        self.assertEqual(created["email_mode"], "changes_only")
        self.assertEqual(created["last_agreement_score"], 60)
        self.assertEqual(len(watch_service.list_watches("u1", db=self.db)), 1)

        paused = watch_service.update_watch("u1", created["id"], {"status": "paused"}, False, db=self.db)
        self.assertEqual(paused["status"], "paused")
        monthly = watch_service.update_watch("u1", created["id"], {"interval": "monthly"}, False, db=self.db)
        self.assertEqual(monthly["interval"], "monthly")
        watch_service.delete_watch("u1", created["id"], db=self.db)
        self.assertEqual(watch_service.list_watches("u1", db=self.db), [])

    def test_every_run_email_mode_can_be_created_and_changed(self):
        created = watch_service.create_watch(
            "u1", share_id=self.share_id, interval="weekly", email_mode="every_run",
            is_pro=False, db=self.db,
        )
        self.assertEqual(created["email_mode"], "every_run")
        updated = watch_service.update_watch(
            "u1", created["id"], {"email_mode": "changes_only"}, False, db=self.db
        )
        self.assertEqual(updated["email_mode"], "changes_only")
        with self.assertRaisesRegex(WatchError, "Email mode"):
            watch_service.update_watch(
                "u1", created["id"], {"email_mode": "everything"}, False, db=self.db
            )

    def test_condition_mode_requires_condition_and_resets_state_when_edited(self):
        with self.assertRaisesRegex(WatchError, "condition"):
            watch_service.create_watch(
                "u1", share_id=self.share_id, interval="weekly",
                email_mode="condition", is_pro=False, db=self.db,
            )
        created = watch_service.create_watch(
            "u1", share_id=self.share_id, interval="weekly",
            email_mode="condition", condition="An official date is announced",
            is_pro=False, db=self.db,
        )
        self.assertEqual(created["condition"], "An official date is announced")
        self.db.stores["watches"][created["id"]]["last_condition_status"] = "met"
        updated = watch_service.update_watch(
            "u1", created["id"], {"condition": "The product is available"},
            False, db=self.db,
        )
        self.assertIsNone(updated["last_condition_status"])

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

    def test_admin_can_list_and_queue_active_watch(self):
        created = watch_service.create_watch(
            "u1", share_id=self.share_id, interval="weekly", is_pro=False, db=self.db
        )
        queued_at = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)
        listed = watch_service.list_watches_for_admin(db=self.db)
        self.assertEqual(listed[0]["owner_uid"], "u1")
        self.assertEqual(listed[0]["consecutive_failures"], 0)

        queued = watch_service.queue_watch_run(created["id"], now=queued_at, db=self.db)
        self.assertEqual(queued["next_run_at"], queued_at.isoformat())
        self.assertEqual(self.db.stores["watches"][created["id"]]["next_run_at"], queued_at)

    def test_admin_queue_rejects_paused_or_claimed_watch(self):
        created = watch_service.create_watch(
            "u1", share_id=self.share_id, interval="weekly", is_pro=False, db=self.db
        )
        watch_id = created["id"]
        now = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)
        self.db.stores["watches"][watch_id]["status"] = "paused"
        with self.assertRaisesRegex(WatchError, "active watch"):
            watch_service.queue_watch_run(watch_id, now=now, db=self.db)
        self.db.stores["watches"][watch_id].update(
            status="active", claimed_until=now + timedelta(minutes=10)
        )
        with self.assertRaisesRegex(WatchError, "currently running"):
            watch_service.queue_watch_run(watch_id, now=now, db=self.db)

    def test_update_rejects_unknown_fields_and_owner(self):
        created = watch_service.create_watch("u1", share_id=self.share_id, interval="weekly", is_pro=False, db=self.db)
        with self.assertRaises(WatchError):
            watch_service.update_watch("u1", created["id"], {"owner_uid": "u2"}, False, db=self.db)
        with self.assertRaisesRegex(WatchError, "own watches"):
            watch_service.update_watch("u2", created["id"], {"status": "paused"}, False, db=self.db)

    def test_result_id_uses_existing_share_flow(self):
        with patch.object(
            watch_service.share_snapshots, "create_share_from_pending",
            return_value={"share_id": self.share_id, "slug": "question", "created": True},
        ) as create_share:
            created = watch_service.create_watch(
                "u1", result_id="R" * 16, interval="weekly", is_pro=False, db=self.db
            )
        create_share.assert_called_once_with("u1", "R" * 16, db=self.db, visibility="public")
        self.assertEqual(created["share_id"], self.share_id)

    def test_private_watch_keeps_visibility_private(self):
        private_id = "P" * 16
        self.db.stores["shares"][private_id] = {**share(slug="private"), "visibility": "private"}
        created = watch_service.create_watch(
            "u1", share_id=private_id, interval="weekly", visibility="private",
            is_pro=False, db=self.db,
        )
        self.assertEqual(created["visibility"], "private")

    def test_watch_can_schedule_local_run_time(self):
        now = datetime(2026, 3, 28, 10, 0, tzinfo=timezone.utc)
        with patch.object(watch_service, "utcnow", return_value=now):
            created = watch_service.create_watch(
                "u1", share_id=self.share_id, interval="daily", is_pro=True,
                run_time="09:00", timezone_name="Europe/Berlin", db=self.db,
            )
        self.assertEqual(created["run_time"], "09:00")
        self.assertEqual(created["timezone"], "Europe/Berlin")
        # 29 March 2026 is after the DST switch: 09:00 Berlin == 07:00 UTC.
        self.assertEqual(
            self.db.stores["watches"][created["id"]]["next_run_at"],
            datetime(2026, 3, 29, 7, 0, tzinfo=timezone.utc),
        )

    def test_run_time_update_reschedules_and_rejects_invalid_values(self):
        created = watch_service.create_watch(
            "u1", share_id=self.share_id, interval="weekly", is_pro=False, db=self.db,
        )
        updated = watch_service.update_watch(
            "u1", created["id"],
            {"run_time": "18:45", "timezone": "Europe/Berlin"},
            False, db=self.db,
        )
        self.assertEqual(updated["run_time"], "18:45")
        self.assertEqual(updated["timezone"], "Europe/Berlin")
        with self.assertRaisesRegex(WatchError, "HH:MM"):
            watch_service.update_watch(
                "u1", created["id"],
                {"run_time": "25:00", "timezone": "Europe/Berlin"},
                False, db=self.db,
            )
        with self.assertRaisesRegex(WatchError, "IANA"):
            watch_service.update_watch(
                "u1", created["id"],
                {"run_time": "09:00", "timezone": "Mars/Olympus"},
                False, db=self.db,
            )


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
        unchanged = {"agreement_score": 61, "changed": False, "severity": "minor"}
        self.assertEqual(
            watch_scheduler.notification_kind(
                {"email_mode": "every_run", "last_agreement_score": 60}, unchanged
            ),
            "every_run",
        )
        self.assertIsNone(
            watch_scheduler.notification_kind(
                {"email_mode": "changes_only", "last_agreement_score": 60}, unchanged
            )
        )
        condition_result = {"agreement_score": 61, "condition_status": "met"}
        condition = "An official date is announced"
        condition_hash = watch_service.condition_hash(condition)
        self.assertEqual(
            watch_scheduler.notification_kind(
                {"email_mode": "condition", "condition": condition,
                 "last_condition_status": "not_met", "last_condition_hash": condition_hash},
                condition_result,
            ),
            "condition",
        )
        self.assertIsNone(watch_scheduler.notification_kind(
            {"email_mode": "condition", "condition": condition,
             "last_condition_status": "met", "last_condition_hash": condition_hash},
            condition_result,
        ))
        self.assertEqual(watch_scheduler.notification_kind(
            {"email_mode": "condition", "condition": "A different condition",
             "last_condition_status": "met", "last_condition_hash": condition_hash},
            condition_result,
        ), "condition")
        self.assertIsNone(watch_scheduler.notification_kind(
            {"email_mode": "condition", "condition": condition,
             "last_condition_status": "not_met", "last_condition_hash": condition_hash},
            {"agreement_score": 61, "condition_status": "unknown"},
        ))

    def test_mock_llm_watch_pipeline(self):
        with patch.dict(os.environ, {"MOCK_LLM": "1"}):
            result = watch_scheduler.execute_watch(
                "When was the Eiffel Tower completed?", "An older consensus answer."
            )
        self.assertIn("Mock consensus", result["consensus"])
        self.assertIsInstance(result["agreement_score"], int)
        self.assertFalse(result["changed"])

    def test_watch_uses_all_configured_models_for_the_selected_tier(self):
        configured = {
            "openai": cfg.DEFAULT_OPENAI_MODEL,
            "mistral": cfg.DEFAULT_MISTRAL_MODEL,
            "gemini": cfg.DEFAULT_GEMINI_MODEL,
            "anthropic": cfg.ANTHROPIC_PRO_MODEL,
        }
        keys = {label: "key" for label in watch_scheduler.PROVIDER_LABELS.values()}
        with patch.object(cfg, "get_watch_models", return_value=configured):
            selected = watch_scheduler._selected_models(keys, True)
        self.assertEqual(dict(selected), configured)
        self.assertEqual(len(selected), 4)


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

    def test_every_run_mail_contains_new_consensus(self):
        message = mailer.build_run_message(
            recipient="owner@example.test", question="A question", agreement_score=71,
            consensus="## New answer\n\nThe updated consensus content.", changed=False,
            severity="minor", summary="", share_url="https://consens.io/s/q-id",
            unsubscribe_url="https://consens.io/watch/unsubscribe?token=x",
        )
        rendered = message.as_string()
        self.assertIn("New consensus:", rendered)
        self.assertIn("The updated consensus content.", rendered)
        self.assertIn("71/100", rendered)

    def test_condition_mail_explains_trigger(self):
        message = mailer.build_condition_message(
            recipient="owner@example.test", question="A question",
            condition="An official date is announced", reason="The date is 15 September.",
            agreement_score=82, consensus="The launch is scheduled.",
            share_url="https://consens.io/s/q-id",
            unsubscribe_url="https://consens.io/watch/unsubscribe?token=x",
        )
        rendered = message.as_string()
        self.assertIn("Watch condition met", rendered)
        self.assertIn("15 September", rendered)
        self.assertIn("The launch is scheduled", rendered)

    def test_admin_test_mail_is_multipart_and_does_not_claim_a_watch(self):
        message = mailer.build_test_message(recipient="admin@example.test")
        self.assertTrue(message.is_multipart())
        self.assertEqual(message["Subject"], "Consensus Watch e-mail test")
        self.assertIn("No watch was executed", message.as_string())

    def test_public_watch_meta_contains_run_schedule_but_no_owner(self):
        db = FakeDb()
        now = datetime(2026, 7, 12, tzinfo=timezone.utc)
        db.stores["watches"]["w1"] = {
            "owner_uid": "private-owner", "share_id": "A" * 16,
            "status": "active", "interval": "weekly", "created_at": now,
            "run_time": "09:00", "timezone": "Europe/Berlin",
            "last_run_at": now, "next_run_at": now + timedelta(days=7),
        }
        meta = watch_service.get_public_watch_meta("A" * 16, db=db)
        self.assertEqual(meta["interval"], "weekly")
        self.assertEqual(meta["run_time"], "09:00")
        self.assertEqual(meta["timezone"], "Europe/Berlin")
        self.assertNotIn("owner_uid", meta)


class HistoryViewTests(unittest.TestCase):
    def test_svg_view_coordinates_and_change_events(self):
        points = [
            {"ts": datetime(2026, 7, 1, tzinfo=timezone.utc), "agreement_score": 25,
             "changed": False, "severity": "minor", "change_summary": "", "verdict": "hardly"},
            {"ts": datetime(2026, 7, 8, tzinfo=timezone.utc), "agreement_score": 80,
             "changed": True, "severity": "major", "change_summary": "Conclusion changed.", "verdict": "mostly"},
        ]
        view = share_router._build_watch_history_view(points)
        self.assertEqual(view["latest_score"], 80)
        self.assertTrue(view["path"].startswith("M 38.0"))
        self.assertEqual(len(view["events"]), 1)
        self.assertEqual(view["events"][0]["change_summary"], "Conclusion changed.")


class SchedulerLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_scheduler_wake_triggers_an_immediate_second_tick(self):
        calls = 0

        async def tick():
            nonlocal calls
            calls += 1
            if calls == 1:
                watch_scheduler.wake_watch_scheduler()
            else:
                raise asyncio.CancelledError

        with patch.object(watch_scheduler, "run_watch_tick", side_effect=tick):
            with self.assertRaises(asyncio.CancelledError):
                await watch_scheduler.watch_scheduler_loop()
        self.assertEqual(calls, 2)

    async def test_every_run_mail_targets_verified_watch_owner(self):
        watch = {
            "owner_uid": "u1", "share_id": "A" * 16, "share_slug": "question",
            "visibility": "public", "question": "Q",
        }
        result = {
            "consensus": "New consensus", "agreement_score": 72,
            "changed": False, "severity": "minor", "change_summary": "",
        }
        user = SimpleNamespace(email="owner@example.test", email_verified=True)
        with (
            patch.object(watch_scheduler.mailer, "is_configured", return_value=True),
            patch.object(watch_scheduler.auth, "get_user", return_value=user),
            patch.object(watch_service, "make_unsubscribe_token", return_value="token"),
            patch.object(watch_scheduler.mailer, "send_message", new_callable=AsyncMock, return_value=True) as send,
        ):
            await watch_scheduler._send_run_mail("w1", watch, result)
        message = send.await_args.args[0]
        self.assertEqual(message["To"], "owner@example.test")
        self.assertIn("New consensus", message.as_string())

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
            patch.object(watch_scheduler.security, "is_user_pro", return_value=False),
            patch.object(watch_scheduler, "execute_watch", side_effect=RuntimeError("provider failed")),
            patch.object(watch_service, "fail_watch_run", side_effect=[False, False, True]),
            patch.object(watch_scheduler, "_send_paused_mail", new_callable=AsyncMock) as send_paused,
        ):
            await watch_scheduler.run_watch_tick()
            await watch_scheduler.run_watch_tick()
            await watch_scheduler.run_watch_tick()
        send_paused.assert_awaited_once_with("w1", claimed)

    async def test_every_run_mode_sends_full_result_mail_without_change(self):
        claimed = {
            "owner_uid": "u1", "share_id": "A" * 16, "interval": "weekly",
            "email_mode": "every_run", "last_agreement_score": 60,
        }
        result = {
            "consensus": "New consensus content", "agreement_score": 61,
            "changed": False, "severity": "minor", "change_summary": "",
        }
        share_data = {"status": "active", "slug": "q", "question": "Q", "consensus_md": "Old"}
        with (
            patch.object(watch_service, "acquire_worker_lease", return_value=True),
            patch.object(watch_service, "release_worker_lease"),
            patch.object(watch_service, "list_due_watch_ids", return_value=["w1"]),
            patch.object(watch_service, "claim_watch", return_value=(claimed, "claimed")),
            patch.object(watch_scheduler.security, "is_user_pro", return_value=True) as pro_check,
            patch.object(watch_scheduler.share_snapshots, "get_share", return_value=share_data),
            patch.object(watch_scheduler, "execute_watch", return_value=result),
            patch.object(watch_service, "complete_watch_run"),
            patch.object(watch_scheduler, "_send_run_mail", new_callable=AsyncMock) as send_run,
            patch.object(watch_scheduler, "_send_change_mail", new_callable=AsyncMock) as send_change,
        ):
            completed = await watch_scheduler.run_watch_tick()
        self.assertEqual(completed, 1)
        pro_check.assert_called_once_with("u1")
        send_run.assert_awaited_once_with("w1", claimed, result)
        send_change.assert_not_awaited()


class WatchFrontendContractTests(unittest.TestCase):
    def test_user_menu_places_watched_after_shared_links(self):
        source = Path("static/firebase.js").read_text(encoding="utf-8")
        self.assertLess(source.index('id="sharedLinksButton"'), source.index('id="watchedLinksButton"'))
        self.assertIn('window.openWatchDialog("list")', source)

    def test_watch_ui_exposes_every_run_email_mode(self):
        source = Path("static/js/watch.js").read_text(encoding="utf-8")
        self.assertIn('value="every_run"', source)
        self.assertIn("Every new consensus (with content)", source)
        self.assertIn('value="condition"', source)
        self.assertIn('id="watchVisibility"', source)
        self.assertIn('id="watchRunTime"', source)
        self.assertIn("resolvedOptions().timeZone", source)

    def test_watch_modal_scrolls_instead_of_overflowing_actions(self):
        source = Path("templates/index.html").read_text(encoding="utf-8")
        self.assertIn("max-height: calc(100dvh - 32px)", source)
        self.assertIn("overflow-y: auto", source)


class AdminWatchRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = FastAPI()
        app.state.limiter = limiter
        app.include_router(admin_router.router)
        cls.client = TestClient(app)

    def _admin_patches(self, is_admin=True):
        return (
            patch.object(admin_router, "extract_id_token", return_value="tok"),
            patch.object(admin_router, "verify_user_token", return_value="admin-1"),
            patch.object(admin_router, "is_user_admin", return_value=is_admin),
        )

    def test_watch_diagnostics_requires_admin(self):
        token_patch, verify_patch, admin_patch = self._admin_patches(False)
        with token_patch, verify_patch, admin_patch:
            response = self.client.get("/api/admin/watches")
        self.assertEqual(response.status_code, 403)

    def test_watch_diagnostics_lists_and_starts_run(self):
        token_patch, verify_patch, admin_patch = self._admin_patches()
        with token_patch, verify_patch, admin_patch, \
                patch.object(admin_router.watch_service, "list_watches_for_admin", return_value=[]), \
                patch.object(admin_router.mailer, "is_configured", return_value=True):
            response = self.client.get("/api/admin/watches")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["smtp_configured"])

        token_patch, verify_patch, admin_patch = self._admin_patches()
        queued = {"id": "w1", "status": "active"}
        with token_patch, verify_patch, admin_patch, \
                patch.object(admin_router.watch_service, "queue_watch_run", return_value=queued) as queue, \
                patch.object(admin_router.watch_scheduler, "wake_watch_scheduler") as wake:
            response = self.client.post("/api/admin/watches/w1/run", json={})
        self.assertEqual(response.status_code, 200)
        queue.assert_called_once_with("w1")
        wake.assert_called_once_with()
        self.assertTrue(response.json()["run_requested"])

    def test_admin_test_email_uses_verified_admin_address(self):
        token_patch, verify_patch, admin_patch = self._admin_patches()
        user = SimpleNamespace(email="admin@example.test", email_verified=True)
        with token_patch, verify_patch, admin_patch, \
                patch.object(admin_router.mailer, "is_configured", return_value=True), \
                patch.object(admin_router.auth, "get_user", return_value=user), \
                patch.object(admin_router.mailer, "send_message", new_callable=AsyncMock, return_value=True) as send:
            response = self.client.post("/api/admin/watches/test-email", json={})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["recipient"], "admin@example.test")
        send.assert_awaited_once()
