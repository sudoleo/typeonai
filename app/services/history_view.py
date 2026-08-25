"""Shared rendering view for a sequence of consensus checks over time.

Consensus Watches and curated Topics both answer the same question repeatedly
and keep every result. The visible part of that — the agreement curve, the
change events and the Position Map — is identical for both, so both pages build
it here instead of growing two divergent versions of the same chart.

Input points are dictionaries with ``ts`` (datetime), ``agreement_score``,
``changed`` and optionally ``severity``, ``change_summary``, ``opinion_map``,
``run_id`` and ``has_snapshot``. ``trigger``, ``score_event`` and ``restated``
are derived from the series by :mod:`app.services.drift_signal`, so a stored
trigger written under an older rule is never rendered.
"""

from __future__ import annotations

from app.services import drift_signal
from app.services.public_markdown import markdown_to_plaintext


def _plain_dimensions(dimensions):
    """Position-Map-Labels kommen aus dem Antworttext und tragen dessen Markdown.

    Die Karten rendern kein Markdown, also stuenden dort sonst sichtbare
    Sternchen ("**Neither is universally cheaper.**") auf der oeffentlichen
    Seite.
    """
    plain = []
    for dimension in dimensions or []:
        if not isinstance(dimension, dict):
            continue
        positions = []
        for position in dimension.get("positions") or []:
            if not isinstance(position, dict):
                continue
            positions.append({
                **position,
                "stance": markdown_to_plaintext(position.get("stance") or ""),
            })
        plain.append({
            **dimension,
            "label": markdown_to_plaintext(dimension.get("label") or ""),
            "positions": positions,
        })
    return plain


def build_history_view(points):
    if not points:
        return None
    # One shared rule decides which checks moved the answer; the curve marks
    # exactly those, so a page cannot highlight more events than it alerts on.
    points = drift_signal.annotate_points(points)
    width, height = 640, 190
    left, right, top, bottom = 38, 18, 18, 30
    plot_w, plot_h = width - left - right, height - top - bottom
    count = len(points)
    coords = []
    for index, point in enumerate(points):
        x = left + (plot_w * index / (count - 1) if count > 1 else plot_w / 2)
        y = top + plot_h * (100 - point["agreement_score"]) / 100
        score_event = bool(point.get("score_event"))
        trigger = point.get("trigger") if point.get("trigger") in {"stable", "changed"} else "stable"
        coords.append({
            **point,
            # Stable within the rendered history and independent of Firestore
            # document IDs, so chart points can link to their visible run row.
            "anchor_id": f"check-{index + 1}",
            "trigger": trigger,
            "x": round(x, 1),
            "y": round(y, 1),
            "score_event": score_event,
        })
    path = " ".join(
        ("M" if index == 0 else "L") + f" {point['x']} {point['y']}"
        for index, point in enumerate(coords)
    )
    events = [point for point in reversed(coords) if point["trigger"] == "changed"]
    mapped_points = [point for point in coords if point.get("opinion_map")]
    position_view = None
    if mapped_points:
        latest_map = mapped_points[-1]["opinion_map"]
        providers = []
        for point in mapped_points:
            for model in point["opinion_map"].get("models") or []:
                provider = model.get("provider")
                if provider and provider not in providers:
                    providers.append(provider)
        trajectories = []
        for provider in providers:
            cells = []
            for point in mapped_points:
                model = next((
                    item for item in point["opinion_map"].get("models") or []
                    if item.get("provider") == provider
                ), None)
                cells.append({
                    "date": point["ts"].strftime("%Y-%m-%d"),
                    "score": model.get("movement_score") if model else None,
                    "moved": bool(model and model.get("moved")),
                    "summary": model.get("summary") if model else "",
                })
            trajectories.append({"provider": provider, "cells": cells})
        position_view = {
            "dates": [point["ts"].strftime("%b %d") for point in mapped_points],
            "trajectories": trajectories,
            "dimensions": _plain_dimensions(latest_map.get("dimensions")),
            "shift_score": latest_map.get("shift_score"),
            "shift_label": latest_map.get("shift_label") or "New baseline",
        }
    return {
        "width": width,
        "height": height,
        "path": path,
        "points": coords,
        "events": events,
        "start_date": points[0]["ts"].strftime("%Y-%m-%d"),
        "end_date": points[-1]["ts"].strftime("%Y-%m-%d"),
        "latest_score": points[-1]["agreement_score"],
        "position_map": position_view,
    }
