"""Periodic retention work that must not depend on process restarts."""

from __future__ import annotations

import asyncio
import os

from app.core.background_tasks import task_succeeded
from app.services.share_snapshots import cleanup_expired_pending, cleanup_revoked_shares


def _interval_seconds() -> int:
    try:
        value = int(os.environ.get("RETENTION_MAINTENANCE_INTERVAL_SECONDS", "3600"))
    except (TypeError, ValueError):
        value = 3600
    return max(60, min(value, 24 * 60 * 60))


RETENTION_MAINTENANCE_INTERVAL_SECONDS = _interval_seconds()
TASK_NAME = "retention-maintenance"


async def retention_maintenance_loop() -> None:
    while True:
        pending_deleted = await asyncio.to_thread(cleanup_expired_pending)
        shares_deleted = await asyncio.to_thread(cleanup_revoked_shares)
        task_succeeded(
            TASK_NAME,
            expired_pending_deleted=pending_deleted,
            revoked_shares_deleted=shares_deleted,
        )
        await asyncio.sleep(RETENTION_MAINTENANCE_INTERVAL_SECONDS)

