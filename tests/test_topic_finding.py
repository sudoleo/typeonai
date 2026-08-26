"""The finding: the one sentence a Topic page states before anything else."""

from app.services import topic_finding


def claim(label, *, streak=5, models=2, run_models=2, contested=False,
          holding=True, is_new=False):
    return {
        "label": label,
        "streak": streak,
        "appearances": streak,
        "model_count": models,
        "run_model_count": run_models,
        "contested": contested,
        "holding": holding,
        "is_new": is_new,
    }


def ledger(*, holding=(), new=(), contested=()):
    return {
        "holding": list(holding),
        "new": list(new),
        "contested": list(contested),
        "retired": [],
    }


def record(*, checks=5, changed_now=False, steady=3):
    return {
        "checks": checks,
        "changed_now": changed_now,
        "steady_checks": steady,
        "steady_days": steady,
        "first_display": "Jul 23, 2026",
        "latest_display": "Aug 19, 2026",
    }


def test_the_finding_is_the_claim_the_record_puts_first():
    """Held longest wins; on a tie the shorter sentence does, because the
    finding is read at a glance and not parsed."""
    found = topic_finding.build_finding(
        ledger(holding=[
            claim("The GPT-5.6 family is the current frontier line, and it "
                  "covers the Sol, Terra and Luna variants", streak=5),
            claim("OpenAI has not announced a release date for GPT-6", streak=5),
            claim("Pricing has not been published", streak=2),
        ]),
        record(),
        {"consensus_md": "Something else entirely."},
    )

    assert found["line"] == "OpenAI has not announced a release date for GPT-6."
    assert found["source"] == "claim"
    assert found["state"] == "settled"
    assert found["voice"] == "All 2 models say the same"


def test_a_contested_claim_never_becomes_the_finding():
    """The page cannot state as settled fact something the models state
    differently -- but it has to say that the disagreement is there."""
    found = topic_finding.build_finding(
        ledger(
            holding=[claim("OpenAI has not announced a release date", streak=4)],
            contested=[claim("Ships in 2026", contested=True, streak=9)],
        ),
        record(),
        {},
    )

    assert found["line"] == "OpenAI has not announced a release date."
    assert found["state"] == "split"
    assert found["state_label"] == "The models disagree"
    assert found["split_count"] == 1


def test_a_check_that_moved_the_answer_outranks_every_other_state():
    found = topic_finding.build_finding(
        ledger(holding=[claim("A launch window is now on record", streak=2)]),
        record(changed_now=True),
        {},
    )

    assert found["state"] == "moved"


def test_a_claim_only_some_models_state_says_so_in_the_finding():
    found = topic_finding.build_finding(
        ledger(holding=[claim("No date is on record", models=2, run_models=3)]),
        record(),
        {},
    )

    assert found["voice"] == "2 of 3 models state this"


def test_supporting_lines_never_repeat_the_finding_or_run_long():
    found = topic_finding.build_finding(
        ledger(holding=[
            claim("OpenAI has not announced a release date for GPT-6", streak=6),
            # Same statement, different wording: nothing gained by printing it.
            claim("OpenAI has announced no release date for GPT-6 so far", streak=4),
            claim("GPT-5.6 is the current flagship", streak=4),
            # A paragraph, not a supporting sentence.
            claim("A" * (topic_finding.MAX_SUPPORT_LINE + 5), streak=4),
            claim("Pricing has not been published", streak=3),
        ]),
        record(),
        {},
    )

    assert found["support"] == [
        "GPT-5.6 is the current flagship.",
        "Pricing has not been published.",
    ]


def test_an_editorial_headline_wins_over_the_derived_one():
    found = topic_finding.build_finding(
        ledger(holding=[claim("OpenAI has not announced a release date")]),
        record(),
        {"headline": "still no gpt-6 date"},
    )

    assert found["line"] == "Still no gpt-6 date."
    assert found["source"] == "editorial"


def test_a_topic_without_a_position_map_still_states_an_answer():
    """Manually seeded Topics carry no claims. The first sentence of the
    consensus is a weaker finding than a tracked claim, and still beats a
    score as the first thing on the page."""
    found = topic_finding.build_finding(
        None,
        record(checks=1, steady=0),
        {"consensus_md": "**No confirmed release date exists.** Everything "
                         "circulating is speculation."},
    )

    assert found["line"] == "No confirmed release date exists."
    assert found["source"] == "consensus"
    assert found["state"] == "first"
    assert found["support"] == []


def test_a_run_with_nothing_to_state_produces_no_finding():
    assert topic_finding.build_finding(None, record(), {}) is None
    assert topic_finding.build_finding(None, None, None) is None


def test_a_long_claim_stays_one_sentence_and_is_marked_as_long():
    long_claim = (
        "As of the latest official information available, there is no confirmed "
        "OpenAI blog post, product page, API model entry or release note that "
        "says when GPT-6 will launch"
    )
    found = topic_finding.build_finding(
        ledger(holding=[claim(long_claim)]), record(), {}
    )

    assert found["line"] == long_claim + "."
    assert found["is_long"] is True


def test_a_clipped_claim_is_restated_from_the_consensus_it_came_from():
    """Position Map labels are stored clipped. The finding is the statement,
    not the truncation, so the wording comes back from the answer itself."""
    clipped = (
        "The current evidence supports a qualified yes: AI has solved some "
        "previously open mathematical problems and has materially advanced "
        "others, but the strongest claims sti…"
    )
    full = (
        "The current evidence supports a qualified yes: AI has solved some "
        "previously open mathematical problems and has materially advanced "
        "others, but the strongest claims still come from a small number of "
        "highly publicised cases."
    )
    found = topic_finding.build_finding(
        ledger(holding=[claim(clipped, streak=4)]),
        record(),
        {"consensus_md": full + " What is most convincing right now."},
    )

    assert found["line"] == full
    assert "…" not in found["line"]


def test_a_clipped_claim_with_no_match_falls_back_to_a_whole_sentence():
    found = topic_finding.build_finding(
        ledger(holding=[
            claim("Some entirely unrelated wording that was cut off here…", streak=4),
            claim("OpenAI has not announced a release date for GPT-6", streak=2),
        ]),
        record(),
        {"consensus_md": "A consensus about something else completely."},
    )

    assert found["line"] == "OpenAI has not announced a release date for GPT-6."


def test_a_mid_sentence_fragment_is_never_the_finding_or_a_supporting_line():
    """A label that starts lower-case is the tail of a quoted sentence."""
    found = topic_finding.build_finding(
        ledger(holding=[
            claim("it was a real research problem, not a benchmark;", streak=9),
            claim("AI has advanced several open problems", streak=3),
        ]),
        record(),
        {"consensus_md": "Nothing comparable here."},
    )

    assert found["line"] == "AI has advanced several open problems."
    assert found["support"] == []


def test_a_claim_that_held_through_a_fraction_of_the_record_is_not_settled():
    """Two restatements out of twenty checks is churn, not a settled answer."""
    churning = topic_finding.build_finding(
        {**ledger(holding=[claim("AI has advanced several open problems", streak=2)]),
         "enumerated": 18},
        record(checks=20),
        {},
    )
    steady = topic_finding.build_finding(
        {**ledger(holding=[claim("AI has advanced several open problems", streak=15)]),
         "enumerated": 18},
        record(checks=20),
        {},
    )

    assert churning["state"] == "forming"
    assert churning["state_label"] == "Still forming"
    assert steady["state"] == "settled"


def test_a_statement_that_ends_inside_a_quotation_keeps_one_full_stop():
    found = topic_finding.build_finding(
        ledger(holding=[claim(
            "Google referred to progress on \u201cour new models including "
            "Gemini 4.\u201d"
        )]),
        record(),
        {},
    )

    assert found["line"].endswith("Gemini 4.\u201d")
