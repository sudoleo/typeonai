"""The line between "the answer moved" and "we checked again".

Every tracked page had drifted to marking each completed check as a change,
which makes the marking carry no information. These tests pin the two arms of
the rule and the noise each one has to survive.
"""

import os

os.environ.setdefault("UNIT_TEST_MODE", "1")

from app.services import drift_signal


def test_a_minor_grade_is_a_restatement_not_a_change():
    """The Judge sets `changed` for a rewritten qualification as well; only the
    grade says whether the substance moved."""
    assert drift_signal.classify(True, "minor", 68, [70]) == "stable"
    assert drift_signal.is_restated(True, "minor") is True
    assert drift_signal.classify(True, "major", 68, [70]) == "changed"
    assert drift_signal.is_restated(True, "major") is False


def test_a_score_that_leaves_the_band_of_the_recent_checks_is_movement():
    assert drift_signal.classify(False, "minor", 64, [84, 84, 84]) == "changed"
    assert drift_signal.classify(False, "minor", 78, [84, 84, 84]) == "stable"


def test_a_score_swinging_between_two_cap_steps_reports_the_first_step_only():
    """84 <-> 64 is one contradiction graded major once. Comparing only with the
    predecessor turns that grading noise into an event on every check."""
    series = [
        {"agreement_score": 84, "changed": False, "severity": "minor"},
        {"agreement_score": 64, "changed": False, "severity": "minor"},
        {"agreement_score": 84, "changed": False, "severity": "minor"},
        {"agreement_score": 64, "changed": False, "severity": "minor"},
        {"agreement_score": 84, "changed": False, "severity": "minor"},
    ]

    triggers = [point["trigger"] for point in drift_signal.annotate_points(series)]

    assert triggers == ["stable", "changed", "stable", "stable", "stable"]


def test_a_sustained_step_reports_once_and_then_settles():
    series = [
        {"agreement_score": 84, "changed": False, "severity": "minor"},
        {"agreement_score": 84, "changed": False, "severity": "minor"},
        {"agreement_score": 84, "changed": False, "severity": "minor"},
        {"agreement_score": 64, "changed": False, "severity": "minor"},
        {"agreement_score": 64, "changed": False, "severity": "minor"},
    ]

    triggers = [point["trigger"] for point in drift_signal.annotate_points(series)]

    assert triggers == ["stable", "stable", "stable", "changed", "stable"]


def test_a_stored_trigger_from_the_looser_rule_is_recomputed_not_trusted():
    series = [
        {"agreement_score": 84, "changed": False, "severity": "minor", "trigger": "stable"},
        {"agreement_score": 82, "changed": True, "severity": "minor", "trigger": "changed"},
    ]

    annotated = drift_signal.annotate_points(series)

    assert [point["trigger"] for point in annotated] == ["stable", "stable"]
    assert annotated[1]["restated"] is True
    assert annotated[1]["changed"] is True


def test_the_first_check_has_no_band_and_no_predecessor():
    annotated = drift_signal.annotate_points(
        [{"agreement_score": 90, "changed": False, "severity": ""}]
    )

    assert annotated[0]["trigger"] == "stable"
    assert annotated[0]["score_delta"] is None
    assert annotated[0]["score_event"] is False


def test_steady_checks_counts_back_to_the_last_material_check():
    series = [
        {"agreement_score": 84, "changed": False, "severity": "minor"},
        {"agreement_score": 84, "changed": True, "severity": "major"},
        {"agreement_score": 84, "changed": True, "severity": "minor"},
        {"agreement_score": 84, "changed": False, "severity": "minor"},
    ]

    assert drift_signal.steady_checks(series) == 2
    assert drift_signal.steady_checks(series[:2]) == 0
