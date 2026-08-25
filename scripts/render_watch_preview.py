"""Render a Watch share page from fixture data into _preview/ for design work.

No Firestore, no LLM: the share router is mounted on a bare FastAPI app and the
two snapshot lookups are patched, exactly the way the page tests do it. Serve
the repo root statically and open /_preview/watch-page.html, so /static/... and
the real templates are the ones under review.

    python scripts/render_watch_preview.py
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("UNIT_TEST_MODE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.api.routers import share as share_router  # noqa: E402
from app.core.rate_limit import limiter  # noqa: E402

SHARE_ID = "a1b2c3d4e5f6a7b8"
SLUG = "will-the-eu-ai-act-gpai-guidance-change-before-2027"
QUESTION = "Will the EU AI Act's GPAI guidance change before 2027?"
NOW = datetime(2026, 8, 25, 6, 0, tzinfo=timezone.utc)

CONSENSUS = (
    "The guidance will most likely be amended, but not repealed. [S1]\n\n"
    "The Commission has signalled a review of the general-purpose AI chapter for "
    "the second half of 2026, and three of the four models read that signal as an "
    "amendment rather than a delay. [S2]\n\n"
    "- **Timing**: an amended text is expected between Q3 2026 and Q1 2027.\n"
    "- **Scope**: the transparency obligations are the part under active revision.\n"
    "- **Open**: whether the compute threshold moves is genuinely unsettled."
)

DIFFERENCES = {
    "claims": [],
    "differences": [{
        "claim": "Whether the compute threshold will be lowered",
        "type": "contradiction",
        "severity": "major",
        "positions": [
            {"stance": "The threshold stays at 10^25 FLOP", "models": ["OpenAI", "Anthropic"], "quote": ""},
            {"stance": "A lower threshold is already drafted", "models": ["Google Gemini"], "quote": ""},
        ],
        "verify": "",
    }],
    "best_model": "",
    "models_compared": ["OpenAI", "Google Gemini", "Anthropic"],
    "agreement": {"score": 82, "level": "largely", "model_count": 3,
                  "major_contradictions": 0, "minor_contradictions": 1, "emphases": 1},
}

# Bewusst lang: genau so kam der Absatz auf der echten Seite an.
BASELINE_SUMMARY = (
    "Added clarifying details regarding public repository pricing (no per-committer "
    "charge), bot exclusions, and unique committer counting across an organization, "
    "while maintaining the same core conclusion and pricing model."
)

SOURCES = [
    {"id": "S1", "title": "Commission work programme 2026", "url": "https://ec.europa.eu/programme", "provider": "web"},
    {"id": "S2", "title": "AI Act implementation tracker", "url": "https://artificialintelligenceact.eu/tracker", "provider": "web"},
]


def opinion_map(shift):
    return {
        "schema_version": 1,
        "dimensions": [{
            "label": "Direction of the GPAI review",
            "positions": [
                {"stance": "Amendment, not repeal", "models": ["OpenAI", "Anthropic"]},
                {"stance": "Delay is more likely", "models": ["Google Gemini"]},
            ],
        }],
        "models": [
            {"provider": "OpenAI", "movement_score": shift, "moved": shift > 15, "summary": "Holds the amendment reading."},
            {"provider": "Google Gemini", "movement_score": shift, "moved": shift > 15, "summary": "Moved towards a delay."},
            {"provider": "Anthropic", "movement_score": 0, "moved": False, "summary": ""},
        ],
        "shift_score": shift,
        "shift_label": "Stable" if shift <= 15 else "Evolving",
        "center": ["Amendment, not repeal"],
    }


def history():
    rows = [
        (28, 90, False, "minor", "", 8),
        (21, 84, True, "minor", "A citation was swapped for a newer tracker entry.", 12),
        (14, 90, False, "minor", "", 6),
        (7, 90, True, "minor", "A qualification about the compute threshold was rephrased.", 11),
        (0, 75, False, "minor", "", 0),
    ]
    points = []
    for days, score, changed, severity, summary, shift in rows:
        points.append({
            "run_id": f"run{days:04d}0000000000",
            "ts": NOW - timedelta(days=days),
            "agreement_score": score,
            "verdict": "largely",
            "changed": changed,
            "severity": severity,
            "change_summary": summary,
            "trigger": "changed" if changed else "stable",
            "event_type": "watch.changed" if changed else "watch.checked",
            "baseline_changed": days == 0,
            "baseline_severity": "minor",
            "baseline_summary": (BASELINE_SUMMARY if days == 0 else ""),
            "has_snapshot": True,
            "opinion_map": opinion_map(shift),
        })
    return points


def share_doc():
    return {
        "schema_version": 1,
        "status": "active",
        "owner_uid": "preview-user",
        "slug": SLUG,
        "question": QUESTION,
        "consensus_md": "The guidance is expected to be amended. [S1]",
        "differences_data": DIFFERENCES,
        "differences_text": "",
        "sources": SOURCES,
        "included_models": ["OpenAI: gpt-5.2", "Google Gemini: 3.1 pro", "Anthropic: opus-5"],
        "consensus_model": "OpenAI",
        "answered_at": (NOW - timedelta(days=28)).isoformat(),
        "created_at": NOW - timedelta(days=28),
        "last_watch_run_at": NOW,
        "latest_watch_run_id": "run00000000000000",
        "index_eligible": True,
        "index_requested": True,
    }


def watch_version(points):
    latest = points[-1]
    return {
        "run_id": latest["run_id"],
        "ts": latest["ts"],
        "consensus_md": CONSENSUS,
        "differences_data": DIFFERENCES,
        "differences_text": "",
        "sources": SOURCES,
        "included_models": ["OpenAI: gpt-5.2", "Google Gemini: 3.1 pro", "Anthropic: opus-5"],
        "consensus_model": "OpenAI",
        "answered_at": latest["ts"].isoformat(),
        "agreement_score": latest["agreement_score"],
        "changed": latest["changed"],
        "severity": latest["severity"],
        "change_summary": latest["change_summary"],
        "trigger": latest["trigger"],
        "opinion_map": latest["opinion_map"],
    }


def main():
    points = history()
    meta = {
        "status": "active",
        "interval": "weekly",
        "run_time": "07:00",
        "run_weekday": "monday",
        "timezone": "Europe/Berlin",
        "last_run_at": NOW,
        "next_run_at": NOW + timedelta(days=7),
        "last_successful_run_id": points[-1]["run_id"],
    }
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(share_router.router)
    client = TestClient(app)
    path = f"/s/{SLUG}-{SHARE_ID}"
    with patch.object(share_router.snapshots, "get_share", return_value=share_doc()), \
            patch.object(share_router.snapshots, "get_share_cached", return_value=share_doc()), \
            patch.object(share_router.snapshots, "list_related_shares", return_value=[]), \
            patch.object(share_router.snapshots, "find_canonical_share", return_value=None), \
            patch.object(share_router.snapshots, "list_watch_history", return_value=points), \
            patch.object(share_router.watch_service, "get_public_watch_meta", return_value=meta), \
            patch.object(share_router.snapshots, "get_watch_version", return_value=watch_version(points)):
        response = client.get(path)
    response.raise_for_status()
    out = ROOT / "_preview"
    out.mkdir(exist_ok=True)
    target = out / "watch-page.html"
    target.write_text(response.text, encoding="utf-8")
    print(f"wrote {target.relative_to(ROOT)} ({len(response.text)} bytes)")


if __name__ == "__main__":
    main()
