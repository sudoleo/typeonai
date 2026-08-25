"""Claim-level record across every stored snapshot of one tracked question.

A Topic re-asks the same question on a schedule and keeps every answer whole.
Read run by run, that archive is close to unreadable: most runs differ only in
wording, so a twenty-snapshot Topic shows twenty near-identical summaries and
buries the two or three moments that actually mattered.

This module re-projects the stored runs onto the unit a reader cares about --
the individual claim. Every run already carries a structured Position Map whose
dimensions are the claims the cross-family judge extracted, so the ledger is
*derived* from what is already saved: no additional model call, no migration,
and an existing Topic gains its full claim history on the next page render.

Claims are chained across runs with the same token-overlap comparison the
Position Map uses between neighbouring runs. Matching always compares against
a claim's most recent wording, so slow rephrasing over many runs still tracks
as one claim instead of splitting into a new entry every week.
"""

from __future__ import annotations

import re
from datetime import datetime

from app.services import drift_signal
from app.services.opinion_map import similarity
from app.services.public_markdown import markdown_to_plaintext


# Same threshold the Position Map uses to call two generated labels the same
# dimension. Keeping one value means a claim that counts as "moved" there
# cannot silently count as a different claim here.
MATCH_THRESHOLD = 0.34
NEW_WINDOW_CHECKS = 3
MAX_LIFELINE_TICKS = 40
MAX_RETIRED_CLAIMS = 6
MAX_RETIRED_SOURCES = 8
MAX_RECORD_SITES = 6
# Movement, in the sense the record strip and the change badge use, is what
# app.services.drift_signal grades as material: the Change Judge's "major".
# A "minor" grade is the Judge saying the answer was restated, not moved, and
# a record that anchors on it reports a change after every single check.
MATERIAL_CHANGE_TYPES = {drift_signal.MATERIAL_SEVERITY}
RESTATED_CHANGE_TYPES = {"minor"}
# A run only counts as an inventory of the answer when the cross-check
# produced more than one claim. Runs below that line say nothing about the
# claims they omit, so they must not be read as an absence.
ENUMERATING_MIN_CLAIMS = 2
MIN_CLAIM_TOKENS = 4
_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
# The Differences judge sometimes quotes half a sentence ("recent releases
# have focused on the"). A fragment cannot be tracked as a claim, and it ends
# on a word no finished statement ends on.
_DANGLING_TAIL = {
    "the", "and", "such", "with", "from", "that", "this", "which", "still",
    "are", "was", "were", "appears", "including", "like", "about", "between",
    "because", "while", "whereas", "than", "then", "into", "its", "their",
    "these", "those", "not", "has", "have", "had", "but",
}


def _date_display(run) -> str:
    return str(run.get("date_display") or "") or str(run.get("observed_at") or "")[:10]


def _observed_at(run):
    value = run.get("observed_at")
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError:
        return None


def _days_between(earlier, later) -> int:
    start, end = _observed_at(earlier), _observed_at(later)
    if not start or not end:
        return 0
    return max(0, (end - start).days)


def _claim_text(value) -> str:
    """Generated labels carry markdown and [S3] citation markers; the ledger
    lists them as sentences, so both are stripped before display and before
    the labels are compared.

    Labels are already clipped to a fixed length when the Position Map is
    written, so a long one can end mid-word. Marking that is the difference
    between a sentence that looks broken and one that is visibly cut short.
    """
    text = markdown_to_plaintext(value, limit=200)
    if len(text) >= 130 and text[-1] not in ".!?…":
        return text.rstrip(",;:-— ") + "…"
    return text


def _is_claim_like(label: str) -> bool:
    """Whether a generated label is a statement the ledger can track."""
    tokens = [
        token.lower() for token in _WORD_RE.findall(label) if len(token) > 2
    ]
    if len(tokens) < MIN_CLAIM_TOKENS:
        return False
    return tokens[-1] not in _DANGLING_TAIL


def _appearance(run: dict, dimension: dict) -> dict:
    positions = [
        position for position in dimension.get("positions") or []
        if position.get("models")
    ]
    models = []
    for position in positions:
        for model in position.get("models") or []:
            if model not in models:
                models.append(model)
    return {
        "label": _claim_text(dimension.get("label")),
        "key": str(dimension.get("key") or ""),
        "type": str(dimension.get("type") or "emphasis"),
        "positions": positions,
        "models": models,
        "contested": len(positions) > 1,
        "run_models": len(run.get("models") or []),
    }


def _record_appearance(entry: dict, appearance: dict, index: int) -> None:
    # The newest appearance wins: the ledger shows a claim in its current
    # wording, and later matching compares against that same wording. An
    # identity once established is the exception -- a later run without a key
    # must not erase it.
    key = appearance["key"] or entry.get("key") or ""
    entry.update(appearance)
    entry["key"] = key
    entry["last_index"] = index
    entry["seen"].append(index)


def _chain_claims(runs) -> tuple[list[dict], list[bool]]:
    """Group the dimensions of every run into one entry per distinct claim.

    Also reports, per run, whether it enumerated the answer's claims at all.
    """
    entries: list[dict] = []
    enumerating: list[bool] = []
    for index, run in enumerate(runs):
        dimensions = (run.get("opinion_map") or {}).get("dimensions") or []
        appearances = [
            _appearance(run, dimension) for dimension in dimensions
            if dimension.get("label")
        ]
        appearances = [
            appearance for appearance in appearances
            if _is_claim_like(appearance["label"])
        ]
        enumerating.append(len(appearances) >= ENUMERATING_MIN_CLAIMS)
        matched_appearances, matched_entries = set(), set()
        # Runs published since the identity Judge exists carry a stable key,
        # and a key beats any comparison of wording. Older runs have none, so
        # both paths have to work side by side in the same history.
        by_key = {
            entry["key"]: index for index, entry in enumerate(entries)
            if entry.get("key")
        }
        for appearance_index, appearance in enumerate(appearances):
            entry_index = by_key.get(appearance["key"]) if appearance["key"] else None
            if entry_index is None or entry_index in matched_entries:
                continue
            matched_appearances.add(appearance_index)
            matched_entries.add(entry_index)
            _record_appearance(entries[entry_index], appearance, index)
        candidates = []
        for appearance_index, appearance in enumerate(appearances):
            if appearance_index in matched_appearances:
                continue
            for entry_index, entry in enumerate(entries):
                if entry_index in matched_entries:
                    continue
                if (
                    appearance["key"] and entry.get("key")
                    and appearance["key"] != entry["key"]
                ):
                    # Two identities the Judge kept apart stay apart, however
                    # similar the wording reads.
                    continue
                # Wording drifts run by run, so a claim is compared against
                # both its current and its original phrasing.
                score = max(
                    similarity(appearance["label"], entry["label"]),
                    similarity(appearance["label"], entry["first_label"]),
                )
                if score >= MATCH_THRESHOLD:
                    candidates.append((score, appearance_index, entry_index))
        # Best pair first, so a run with two similar claims cannot attach the
        # weaker one to the entry the stronger one belongs to.
        candidates.sort(key=lambda item: item[0], reverse=True)
        for _score, appearance_index, entry_index in candidates:
            if appearance_index in matched_appearances or entry_index in matched_entries:
                continue
            matched_appearances.add(appearance_index)
            matched_entries.add(entry_index)
            _record_appearance(entries[entry_index], appearances[appearance_index], index)
        for appearance_index, appearance in enumerate(appearances):
            if appearance_index in matched_appearances:
                continue
            entry = {
                "first_index": index,
                "first_label": appearance["label"],
                "seen": [],
            }
            entries.append(entry)
            _record_appearance(entry, appearance, index)
    return entries, enumerating


def _claim_view(entry: dict, dates, enumerating: list[bool]) -> dict:
    """One claim's life, measured only against the checks that listed claims.

    A check that produced no claim inventory is a gap in the record, not
    evidence that a claim was dropped, so it neither breaks a streak nor
    counts as an absence.
    """
    total = len(dates)
    seen = set(entry["seen"])
    counted = [index for index in range(total) if enumerating[index]]
    latest_counted = counted[-1] if counted else total - 1

    streak = 0
    for index in reversed(counted):
        if index > entry["last_index"]:
            continue
        if index not in seen:
            break
        streak += 1
    holding = entry["last_index"] >= latest_counted
    window = [index for index in counted if index >= entry["first_index"]]
    lifeline_start = max(0, total - MAX_LIFELINE_TICKS)
    return {
        "label": entry["label"],
        "type": entry["type"],
        "positions": entry["positions"],
        "models": entry["models"],
        "model_count": len(entry["models"]),
        "run_model_count": entry["run_models"],
        "contested": entry["contested"] or entry["type"] == "contradiction",
        # A claim only some of the answering models state is a weaker part of
        # the record than one all of them state, and the page has to say so.
        "partial": bool(
            entry["run_models"] and len(entry["models"]) < entry["run_models"]
        ),
        "appearances": len(entry["seen"]),
        "streak": streak,
        "holding": holding,
        # Only a record long enough to have a "before" can call a claim new.
        "is_new": (
            holding
            and len(window) <= NEW_WINDOW_CHECKS
            and len(counted) > NEW_WINDOW_CHECKS
        ),
        "first_display": dates[entry["first_index"]],
        "last_display": dates[entry["last_index"]],
        "lifeline": [
            {
                "state": (
                    "on" if index in seen
                    else ("gap" if not enumerating[index] else "off")
                ),
                "date": dates[index],
            }
            for index in range(lifeline_start, total)
        ],
    }


def build_claim_ledger(runs) -> dict | None:
    """The claim-by-claim record over ``runs`` (oldest first).

    Returns ``None`` when no run carries a usable Position Map, which is the
    case for manually seeded Topics and for the very first automatic runs.
    """
    runs = list(runs or [])
    if not runs:
        return None
    entries, enumerating = _chain_claims(runs)
    if not entries:
        return None
    dates = [_date_display(run) for run in runs]
    claims = [_claim_view(entry, dates, enumerating) for entry in entries]

    fresh = [claim for claim in claims if claim["holding"] and claim["is_new"]]
    contested = [
        claim for claim in claims
        if claim["holding"] and not claim["is_new"] and claim["contested"]
    ]
    holding = [
        claim for claim in claims
        if claim["holding"] and not claim["is_new"] and not claim["contested"]
    ]
    # A claim seen in exactly one check and never again is a one-off phrasing,
    # not a position the record lost. It is counted, not listed.
    dropped = [
        claim for claim in claims
        if not claim["holding"] and claim["appearances"] > 1
    ]
    one_off = sum(
        1 for claim in claims
        if not claim["holding"] and claim["appearances"] == 1
    )

    fresh.sort(key=lambda claim: (-claim["model_count"], claim["label"]))
    contested.sort(key=lambda claim: (-claim["streak"], claim["label"]))
    holding.sort(key=lambda claim: (-claim["streak"], -claim["appearances"]))
    dropped.sort(key=lambda claim: (-claim["appearances"], claim["label"]))
    enumerated = sum(1 for value in enumerating if value)
    return {
        "checks": len(runs),
        "enumerated": enumerated,
        "thin": len(runs) - enumerated,
        "tracked_since": dates[0],
        "new": fresh,
        "contested": contested,
        "holding": holding,
        "retired": dropped[:MAX_RETIRED_CLAIMS],
        "retired_count": len(dropped),
        "one_off_count": one_off,
        "holding_count": len(fresh) + len(contested) + len(holding),
        "longest_streak": max((claim["streak"] for claim in claims), default=0),
    }


def build_record_summary(runs) -> dict | None:
    """Headline facts about the run series itself (oldest first).

    The anchor is the last run the Change Judge graded as a material change.
    Score movement is deliberately *not* an anchor: the agreement score steps
    between a small set of grading levels, so a step without a material grade
    is a wording difference, not a shift in substance. A "minor" grade is not
    an anchor either — that is the Judge saying the answer was restated.
    """
    runs = list(runs or [])
    if not runs:
        return None
    latest = runs[-1]
    anchor_index = 0
    for index, run in enumerate(runs):
        if str(run.get("change_type") or "stable") in MATERIAL_CHANGE_TYPES:
            anchor_index = index
    anchor = runs[anchor_index]
    material = [
        {
            "display": _date_display(run),
            "iso": str(run.get("observed_at") or "")[:19],
            "summary": str(run.get("change_summary") or ""),
            "change_type": str(run.get("change_type") or "stable"),
            "run_id": run.get("id"),
            "agreement_score": run.get("agreement_score"),
        }
        for run in reversed(runs)
        if str(run.get("change_type") or "stable") in MATERIAL_CHANGE_TYPES
    ]
    return {
        "checks": len(runs),
        "first_display": _date_display(runs[0]),
        "latest_display": _date_display(latest),
        "span_days": _days_between(runs[0], latest),
        "material_events": material,
        "material_count": len(material),
        # Checks that found no material change since the anchor. Zero means the
        # newest check is itself the change.
        "steady_checks": len(runs) - 1 - anchor_index,
        "steady_days": _days_between(anchor, latest),
        "anchor_display": _date_display(anchor),
        "anchor_summary": str(anchor.get("change_summary") or ""),
        "anchor_run_id": anchor.get("id"),
        "anchor_is_first": anchor_index == 0 and not material,
        "changed_now": bool(material) and anchor_index == len(runs) - 1,
    }


def collapse_timeline(runs) -> list[dict]:
    """Fold stretches of unchanged checks in a newest-first run list.

    Fourteen consecutive "wording only" entries tell a reader nothing that the
    single line "14 checks, no material change" does not, and they hide the
    entries that do. The runs stay reachable inside the folded entry.
    """
    runs = list(runs or [])
    entries: list[dict] = []
    quiet: list[dict] = []

    def flush():
        if not quiet:
            return
        if len(quiet) == 1:
            entries.append({"kind": "run", "run": quiet[0]})
        else:
            entries.append({
                "kind": "quiet",
                "runs": list(quiet),
                "count": len(quiet),
                "from_display": _date_display(quiet[0]),
                "to_display": _date_display(quiet[-1]),
            })
        quiet.clear()

    for index, run in enumerate(runs):
        material = str(run.get("change_type") or "stable") in MATERIAL_CHANGE_TYPES
        # The newest and oldest check always stay visible: they are the ends of
        # the record, and the selected one has to stay findable.
        if material or run.get("is_selected") or index in {0, len(runs) - 1}:
            flush()
            entries.append({"kind": "run", "run": run})
        else:
            quiet.append(run)
    flush()
    return entries


def apply_source_chronicle(runs, selected) -> dict:
    """Date the selected snapshot's sources and list the ones that dropped out.

    Sources are the part of the archive a single answer cannot reproduce at
    all: when a link entered the record, how long it has been cited, and which
    links stopped being cited. Annotates ``selected["evidence"]`` in place.
    """
    runs = list(runs or [])
    selected = selected or {}
    first_seen: dict[str, int] = {}
    last_seen: dict[str, int] = {}
    counts: dict[str, int] = {}
    latest_item: dict[str, dict] = {}
    site_checks: dict[str, set] = {}
    for index, run in enumerate(runs):
        for item in run.get("evidence") or []:
            url = str(item.get("url") or "")
            if not url:
                continue
            first_seen.setdefault(url, index)
            last_seen[url] = index
            counts[url] = counts.get(url, 0) + 1
            latest_item[url] = item
            host = str(item.get("host") or "")
            # A search-grounding redirect host is transport, not a publisher.
            if host and not item.get("is_indirect"):
                site_checks.setdefault(host, set()).add(index)
    dates = [_date_display(run) for run in runs]
    selected_index = next(
        (
            index for index, run in enumerate(runs)
            if run.get("id") == selected.get("id")
        ),
        len(runs) - 1,
    )

    new_count = 0
    for item in selected.get("evidence") or []:
        url = str(item.get("url") or "")
        index = first_seen.get(url, selected_index)
        item["first_display"] = dates[index] if dates else ""
        item["appearances"] = counts.get(url, 1)
        item["is_new"] = index >= selected_index and selected_index > 0
        if item["is_new"]:
            new_count += 1

    retired = []
    for url, index in last_seen.items():
        if index >= selected_index:
            continue
        item = dict(latest_item[url])
        item["last_display"] = dates[index]
        item["appearances"] = counts.get(url, 1)
        item["last_index"] = index
        retired.append(item)
    retired.sort(key=lambda item: (-item["last_index"], -item["appearances"]))
    # Models re-run their own web search on every check, so the individual URL
    # list churns. The durable signal is which sites the record keeps coming
    # back to, which is counted in checks, not in links.
    sites = sorted(
        (
            {"host": host, "checks": len(indexes)}
            for host, indexes in site_checks.items() if len(indexes) > 1
        ),
        key=lambda item: (-item["checks"], item["host"]),
    )
    return {
        "retired": retired[:MAX_RETIRED_SOURCES],
        "retired_count": len(retired),
        "new_count": new_count,
        "tracked_count": len(first_seen),
        "checks": len(runs),
        "sites": sites[:MAX_RECORD_SITES],
        "site_count": len(sites),
    }
