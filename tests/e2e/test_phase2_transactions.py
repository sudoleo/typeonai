"""Phase-2 race regressions against the isolated Firestore emulator."""

from concurrent.futures import ThreadPoolExecutor
import os
import uuid

from google.cloud import firestore as google_firestore
from google.cloud.firestore_v1.base_query import FieldFilter

from app.core.e2e_profile import E2E_PROJECT_ID, assert_safe_e2e_environment
from app.services import chat_store, share_snapshots, watch_service


def _emulator_db():
    assert_safe_e2e_environment(os.environ)
    return google_firestore.Client(project=E2E_PROJECT_ID)


def _delete_collection(collection):
    for snapshot in collection.stream():
        snapshot.reference.delete()


def test_two_workers_cannot_exceed_owner_watch_limit(monkeypatch):
    db = _emulator_db()
    suffix = uuid.uuid4().hex
    uid = f"phase2-watch-{suffix}"
    share_ids = [
        share_snapshots.generate_share_id(),
        share_snapshots.generate_share_id(),
    ]
    for index, share_id in enumerate(share_ids):
        question = f"Will phase two race {index} remain bounded {suffix}?"
        db.collection("shares").document(share_id).set({
            "owner_uid": uid,
            "status": "active",
            "visibility": "public",
            "slug": f"phase-two-{index}",
            "question": question,
            "question_hash": share_snapshots.question_hash(question),
            "differences_data": {"agreement": {"score": 70}},
        })

    monkeypatch.setattr(
        watch_service.cfg,
        "get_watch_active_limit",
        lambda _is_pro: 1,
    )

    def create(share_id):
        try:
            watch = watch_service.create_watch(
                uid,
                share_id=share_id,
                interval="weekly",
                is_pro=False,
                db=db,
            )
            return "created", watch["id"]
        except watch_service.WatchError as exc:
            return exc.code, ""

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(create, share_ids))
        assert sorted(code for code, _watch_id in outcomes) == [
            "created",
            "limit_reached",
        ]
        watches = list(db.collection("watches").where(
            filter=FieldFilter("owner_uid", "==", uid)
        ).stream())
        assert len(watches) == 1
        state = (
            db.collection("users").document(uid).collection("watch_state")
            .document("quota").get().to_dict()
        )
        assert state["active_count"] == 1
    finally:
        for snapshot in db.collection("watches").where(
            filter=FieldFilter("owner_uid", "==", uid)
        ).stream():
            watch_service.delete_watch(uid, snapshot.id, db=db)
        for share_id in share_ids:
            db.collection("shares").document(share_id).delete()
        user_ref = db.collection("users").document(uid)
        _delete_collection(user_ref.collection("watch_state"))
        _delete_collection(user_ref.collection("watch_uniques"))
        user_ref.delete()


def test_two_workers_publish_one_pending_share_and_consume_one_quota():
    db = _emulator_db()
    suffix = uuid.uuid4().hex
    uid = f"phase2-share-{suffix}"
    result_id = share_snapshots.generate_share_id()
    pending_ref = db.collection("pending_results").document(result_id)
    pending_ref.set({
        "owner_uid": uid,
        "question": f"Can publication stay idempotent under a race {suffix}?",
        "consensus_md": "Yes. The transaction is the publication boundary.",
        "differences_data": {},
        "differences_text": "",
        "sources": [],
        "included_models": ["OpenAI", "Anthropic"],
        "consensus_model": "OpenAI",
    })

    def publish(_worker):
        return share_snapshots.create_share_from_pending(uid, result_id, db=db)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(publish, range(2)))
        assert len({outcome["share_id"] for outcome in outcomes}) == 1
        assert sorted(outcome["created"] for outcome in outcomes) == [False, True]
        quota = (
            db.collection("users").document(uid).collection("counters")
            .document("shares_daily").get().to_dict()
        )
        assert quota["count"] == 1
    finally:
        pending = pending_ref.get().to_dict() or {}
        for field in ("share_id", "public_share_id", "private_share_id"):
            share_id = str(pending.get(field) or "")
            if share_id:
                db.collection("shares").document(share_id).delete()
        pending_ref.delete()
        user_ref = db.collection("users").document(uid)
        _delete_collection(user_ref.collection("counters"))
        user_ref.delete()


def test_two_workers_cannot_exceed_owner_chat_limit(monkeypatch):
    db = _emulator_db()
    uid = f"phase2-chat-{uuid.uuid4().hex}"
    store = chat_store.ChatStore(db)
    monkeypatch.setattr(chat_store, "MAX_CHATS_PER_OWNER", 1)

    def create(_worker):
        try:
            return "created", store.create_chat(uid)["id"]
        except chat_store.ChatQuotaExceeded as exc:
            return exc.error_code, ""

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(create, range(2)))
        assert sorted(code for code, _chat_id in outcomes) == [
            "chat_limit_reached",
            "created",
        ]
        assert len(list(
            db.collection("users").document(uid).collection("chats").stream()
        )) == 1
    finally:
        store.delete_all_chats(uid)
        db.collection("users").document(uid).delete()


def test_parallel_reports_never_lose_increments_or_noindex_transition():
    db = _emulator_db()
    share_id = share_snapshots.generate_share_id()
    ref = db.collection("shares").document(share_id)
    ref.set({
        "status": "active",
        "visibility": "public",
        "indexed": True,
        "reports_count": 0,
    })

    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            counts = list(
                pool.map(
                    lambda _worker: share_snapshots.report_share(
                        share_id, "spam", db=db
                    ),
                    range(8),
                )
            )
        stored = ref.get().to_dict()
        assert sorted(counts) == list(range(1, 9))
        assert stored["reports_count"] == 8
        assert stored["report_reasons"] == {"spam": 8}
        assert stored["needs_review"] is True
        assert stored["indexed"] is False
    finally:
        ref.delete()
