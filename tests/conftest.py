import os

# Die Playwright-E2E-Suite (tests/e2e/) braucht einen Chromium-Browser und
# startet einen eigenen uvicorn-Server; sie darf die schnelle Backend-Baseline
# ("python -m pytest tests") nicht mit einsammeln. Lauf nur mit RUN_E2E=1
# (siehe tests/e2e/README.md).
if os.environ.get("RUN_E2E") != "1":
    collect_ignore = ["e2e"]
