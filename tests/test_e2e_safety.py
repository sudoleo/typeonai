import asyncio
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from app.core.e2e_profile import (
    E2E_PROJECT_ID,
    assert_safe_e2e_environment,
)


def safe_env(**overrides):
    env = {
        "E2E_TEST_MODE": "1",
        "FIRESTORE_EMULATOR_HOST": "127.0.0.1:8085",
        "GOOGLE_CLOUD_PROJECT": E2E_PROJECT_ID,
        "GCLOUD_PROJECT": E2E_PROJECT_ID,
        "FIREBASE_PROJECT_ID": E2E_PROJECT_ID,
    }
    env.update(overrides)
    return env


def test_e2e_guard_accepts_only_the_local_demo_project():
    assert_safe_e2e_environment(safe_env())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"FIRESTORE_EMULATOR_HOST": ""}, "loopback"),
        ({"FIRESTORE_EMULATOR_HOST": "firestore.googleapis.com:443"}, "loopback"),
        ({"GOOGLE_CLOUD_PROJECT": "consensio-production"}, "not allowlisted"),
        ({"GCLOUD_PROJECT": "unknown-test-project"}, "not allowlisted"),
        ({"FIREBASE_PROJECT_ID": "consensio-production"}, "not allowlisted"),
    ],
)
def test_e2e_guard_rejects_missing_remote_or_unknown_targets(overrides, message):
    with pytest.raises(RuntimeError, match=message):
        assert_safe_e2e_environment(safe_env(**overrides))


def test_e2e_guard_is_a_noop_outside_the_explicit_profile():
    assert_safe_e2e_environment({"GOOGLE_CLOUD_PROJECT": "consensio-production"})


def test_real_security_import_refuses_a_production_project_before_firebase_init():
    env = os.environ.copy()
    env.update(safe_env(GOOGLE_CLOUD_PROJECT="consensio-production"))
    result = subprocess.run(
        [sys.executable, "-c", "import app.core.security"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "project ID is not allowlisted" in (result.stdout + result.stderr)


def test_unit_profile_import_needs_no_service_account_file(tmp_path):
    env = os.environ.copy()
    env.pop("E2E_TEST_MODE", None)
    env["UNIT_TEST_MODE"] = "1"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    result = subprocess.run(
        [sys.executable, "-c", "import app.core.security"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_e2e_lifespan_starts_no_maintenance_or_background_tasks(monkeypatch):
    import main

    monkeypatch.setenv("E2E_TEST_MODE", "1")

    async def enter_lifespan():
        with patch.object(main, "_run_startup_jobs") as startup_jobs, patch.object(
            main.asyncio, "create_task"
        ) as create_task:
            async with main.lifespan(main.app):
                pass
            startup_jobs.assert_not_called()
            create_task.assert_not_called()

    asyncio.run(enter_lifespan())
