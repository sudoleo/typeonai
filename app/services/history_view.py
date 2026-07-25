"""Shared rendering view for a sequence of consensus checks over time.

Consensus Watches and curated Topics both answer the same question repeatedly
and keep every result. The visible part of that — the agreement curve, the
change events and the Position Map — is identical for both, so both pages build
it here instead of growing two divergent versions of the same chart.

Input points are dictionaries with ``ts`` (datetime), ``agreement_score``,
``changed`` and optionally ``change_summary``, ``opinion_map``, ``run_id``,
``has_snapshot`` and ``trigger``.
"""

from __future__ import annotations


def build_history_view(points):
    if not points:
        return None
    width, height = 640, 190
    left, right, top, bottom = 38, 18, 18, 30
    plot_w, plot_h = width - left - right, height - top - bottom
    count = len(points)
    coords = []
    for index, point in enumerate(points):
        x = left + (plot_w * index / (count - 1) if count > 1 else plot_w / 2)
        y = top + plot_h * (100 - point["agreement_score"]) / 100
        previous_score = points[index - 1]["agreement_score"] if index else None
        score_event = previous_score is not None and abs(point["agreement_score"] - previous_score) >= 15
        trigger = point.get("trigger")
        if trigger not in {"stable", "changed"}:
            trigger = "changed" if point.get("changed") or score_event else "stable"
        coords.append({
            **point,
            "trigger": trigger,
            "x": round(x, 1),
            "y": round(y, 1),
            "score_event": score_event,
        })
    path = " ".join(
        ("M" if index == 0 else "L") + f" {point['x']} {point['y']}"
        for index, point in enumerate(coords)
    )
    events = [point for point in reversed(coords) if point["changed"] or point["score_event"]]
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
            "dimensions": latest_map.get("dimensions") or [],
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
