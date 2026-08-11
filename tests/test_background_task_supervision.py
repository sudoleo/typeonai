import asyncio

import main
from app.core import background_tasks
from app.services import retention_maintenance


def setup_function():
    background_tasks.reset_task_health()


def test_supervisor_restarts_crashes_and_keeps_last_success_health():
    async def exercise():
        attempts = 0
        keep_running = asyncio.Event()

        async def worker():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise RuntimeError("transient failure")
            background_tasks.task_succeeded("test-worker", work=1)
            await keep_running.wait()

        task = asyncio.create_task(
            background_tasks.supervise_background_task(
                "test-worker",
                worker,
                initial_backoff_seconds=0.01,
                max_backoff_seconds=0.01,
            )
        )
        for _ in range(100):
            state = background_tasks.task_health_snapshot().get("test-worker", {})
            if state.get("last_success_at"):
                break
            await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return attempts, background_tasks.task_health_snapshot()["test-worker"]

    attempts, health = asyncio.run(exercise())
    assert attempts == 3
    assert health["restart_count"] == 2
    assert health["consecutive_failures"] == 0
    assert health["last_success_at"]
    assert health["state"] == "stopped"


def test_supervisor_alerts_after_repeated_failure():
    async def exercise():
        alerts = []

        async def worker():
            raise OSError("firestore unavailable")

        task = asyncio.create_task(
            background_tasks.supervise_background_task(
                "failing-worker",
                worker,
                alert=alerts.append,
                alert_after_failures=3,
                initial_backoff_seconds=0.01,
                max_backoff_seconds=0.01,
            )
        )
        for _ in range(100):
            if alerts:
                break
            await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return alerts

    alerts = asyncio.run(exercise())
    assert len(alerts) == 1
    assert alerts[0]["type"] == "background_task_repeated_failure"
    assert "failing-worker" in alerts[0]["message"]


def test_failed_one_shot_task_alerts_immediately():
    alerts = []

    async def exercise():
        async def worker():
            raise RuntimeError("startup maintenance failed")

        await background_tasks.supervise_background_task(
            "one-shot",
            worker,
            restart=False,
            alert=alerts.append,
        )

    asyncio.run(exercise())
    assert len(alerts) == 1
    assert background_tasks.task_health_snapshot()["one-shot"]["state"] == "failed"


def test_retention_loop_runs_cleanup_before_first_sleep(monkeypatch):
    calls = []
    monkeypatch.setattr(
        retention_maintenance,
        "cleanup_expired_pending",
        lambda: calls.append("pending") or 2,
    )
    monkeypatch.setattr(
        retention_maintenance,
        "cleanup_revoked_shares",
        lambda: calls.append("shares") or 3,
    )

    async def stop_after_first_tick(seconds):
        raise asyncio.CancelledError

    monkeypatch.setattr(retention_maintenance.asyncio, "sleep", stop_after_first_tick)

    async def exercise():
        try:
            await retention_maintenance.retention_maintenance_loop()
        except asyncio.CancelledError:
            pass

    asyncio.run(exercise())
    assert calls == ["pending", "shares"]
    health = background_tasks.task_health_snapshot()["retention-maintenance"]
    assert health["details"] == {
        "expired_pending_deleted": 2,
        "revoked_shares_deleted": 3,
    }


def test_maintenance_health_reports_degraded_task_state():
    background_tasks.mark_task_disabled("optional", "test")
    background_tasks._update("required", state="restarting")

    response = main.maintenance_health()

    assert response["status"] == "degraded"
    assert response["tasks"]["optional"]["state"] == "disabled"


def test_startup_readiness_only_loads_bounded_configuration(monkeypatch):
    calls = []
    monkeypatch.setattr(main, "load_models_from_db", lambda: calls.append("models"))

    main._load_startup_configuration()

    assert calls == ["models"]
