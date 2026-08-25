"""One definition of what counts as movement between two consensus checks.

A tracked page is only worth reading if "Changed since last check" is rarer
than "we looked again". That stopped being true because every surface built its
own version of the question and the loosest one won on the page:

* the change Judge returns ``changed=True`` for a rewritten qualification as
  well, and only grades the substance in ``severity``. The page badge used the
  flag and ignored the grade, so a paraphrase was announced as movement.
* the agreement score is quantised by the caps in ``consensus_scoring`` (90/84
  /64/39). One contradiction graded "major" once moves the score a whole cap
  step, so ``abs(delta) >= 15`` fires on the labelling noise of a single
  difference and fires again when the next check labels it back.

The rule below keeps both arms, because both can carry real news, but asks each
one to survive that noise:

* the Judge arm requires ``severity == "major"`` -- the bar the change mail has
  always used, so the badge and the alert can no longer disagree with each
  other;
* the score arm requires the new score to sit ``SCORE_BAND_DELTA`` points away
  from *every* one of the last ``SCORE_BAND_WINDOW`` scores, so a value
  oscillating between two cap steps produces one event on the way out of its
  band instead of one on every swing.

Nothing is discarded: ``changed``/``severity`` stay on the history point, and a
stable check that the Judge still saw a difference in is marked ``restated`` so
a page can say "the wording moved, the conclusion held" in a quiet line instead
of in the badge.
"""

from __future__ import annotations

SCORE_BAND_DELTA = 15
SCORE_BAND_WINDOW = 3

MATERIAL_SEVERITY = "major"


def _numeric(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def recent_scores(points, limit: int = SCORE_BAND_WINDOW) -> list:
    """The last numeric agreement scores of an ascending history, oldest first."""
    scores = []
    for point in points or []:
        score = _numeric((point or {}).get("agreement_score"))
        if score is not None:
            scores.append(score)
    return scores[-limit:] if limit else scores


def score_left_band(score, previous_scores) -> bool:
    """True when the new score leaves the band the recent checks held.

    Compared against every score in the window, not just the predecessor: a
    score that keeps jumping between two cap steps never leaves its band and
    therefore stops reporting each jump as an event.
    """
    score = _numeric(score)
    if score is None:
        return False
    window = [
        value for value in (_numeric(item) for item in previous_scores or [])
        if value is not None
    ]
    if not window:
        return False
    return all(abs(score - value) >= SCORE_BAND_DELTA for value in window)


def is_material(changed, severity, score=None, previous_scores=()) -> bool:
    """Did this check move the answer, or only restate it?"""
    if bool(changed) and str(severity or "").lower() == MATERIAL_SEVERITY:
        return True
    return score_left_band(score, previous_scores)


def classify(changed, severity, score=None, previous_scores=()) -> str:
    """The persisted trigger for one check: ``changed`` or ``stable``."""
    return "changed" if is_material(changed, severity, score, previous_scores) else "stable"


def is_restated(changed, severity) -> bool:
    """The Judge saw a difference, but not one that carries the conclusion."""
    return bool(changed) and str(severity or "").lower() != MATERIAL_SEVERITY


def annotate_points(points) -> list[dict]:
    """Classify an ascending history series in one pass.

    Every read surface (share page, agreement curve, dashboard JSON, morning
    brief) runs its points through here instead of re-deriving the rule, and
    the stored ``trigger`` of older checks is recomputed rather than trusted:
    it was written under the loose rule and would otherwise keep pages showing
    a change badge on every row forever.
    """
    annotated = []
    window: list = []
    for index, point in enumerate(points or []):
        point = dict(point or {})
        score = _numeric(point.get("agreement_score"))
        previous = window[-1] if window else None
        material = is_material(
            point.get("changed"), point.get("severity"), score, window,
        )
        point["trigger"] = "changed" if material else "stable"
        point["score_event"] = score_left_band(score, window)
        point["restated"] = is_restated(point.get("changed"), point.get("severity"))
        point["score_delta"] = (
            int(score) - int(previous)
            if score is not None and previous is not None and index
            else None
        )
        annotated.append(point)
        if score is not None:
            window = (window + [score])[-SCORE_BAND_WINDOW:]
    return annotated


def steady_checks(points) -> int:
    """Completed checks since the last material one (0 = the newest is it)."""
    annotated = annotate_points(points)
    steady = 0
    for point in reversed(annotated):
        if point["trigger"] == "changed":
            break
        steady += 1
    return max(0, min(steady, max(0, len(annotated) - 1)))
