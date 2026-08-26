"""The Claim Ledger: what a Topic's stored runs say when read as one record."""

from datetime import datetime, timedelta, timezone

from app.services import claim_ledger, topic_runner


START = datetime(2026, 7, 23, 12, 0, tzinfo=timezone.utc)


def dimension(label, *, key="", models=("OpenAI", "Gemini"), stance="", type_="claim"):
    return {
        "label": label,
        "type": type_,
        "key": key,
        "positions": [{"stance": stance or label, "models": list(models)}],
    }


def split_dimension(label, *, key=""):
    return {
        "label": label,
        "type": "contradiction",
        "key": key,
        "positions": [
            {"stance": "Ships in 2026", "models": ["OpenAI"]},
            {"stance": "Nothing before 2027", "models": ["Gemini"]},
        ],
    }


def run(index, dimensions, *, change_type="stable", summary="", evidence=(), score=90):
    observed = START + timedelta(days=index)
    return {
        "id": f"run-{index}",
        "version": index + 1,
        "observed_at": observed.isoformat(),
        "date_display": observed.strftime("%b %d, %Y"),
        "change_type": change_type,
        "change_summary": summary,
        "agreement_score": score,
        "models": ["OpenAI: gpt-5.6", "Gemini: gemini-3.5"],
        "evidence": [dict(item) for item in evidence],
        "opinion_map": {"dimensions": list(dimensions)},
    }


def source(url, title="Release notes", host="openai.com"):
    return {"url": url, "title": title, "type": "primary", "host": host}


def test_one_claim_stays_one_claim_through_rewording():
    """The wording of a claim is regenerated on every run. A reader has to see
    that the statement held, not a new entry per phrasing."""
    runs = [
        run(0, [
            dimension("OpenAI has not announced a release date for GPT-6"),
            dimension("The current frontier line is the GPT-5.6 family"),
        ]),
        run(1, [
            dimension("As of today, OpenAI has not announced a release date for GPT-6"),
            dimension("The current frontier line is still the GPT-5.6 family"),
        ]),
        run(2, [
            dimension(
                "As of the latest official information, OpenAI has not announced "
                "a release date for GPT-6"
            ),
            dimension("The current frontier line remains the GPT-5.6 family"),
        ]),
    ]

    ledger = claim_ledger.build_claim_ledger(runs)

    assert ledger["checks"] == 3
    assert ledger["enumerated"] == 3
    assert len(ledger["holding"]) == 2
    assert ledger["holding"][0]["streak"] == 3
    assert ledger["holding"][0]["appearances"] == 3
    assert ledger["retired"] == []
    # The newest phrasing is the one shown.
    assert ledger["holding"][0]["label"].startswith("As of the latest official")


def test_a_check_without_a_claim_list_is_a_gap_not_a_retirement():
    """Some runs produce a single 'emphasis' dimension instead of an inventory.
    Reading that as 'the claim was dropped' would invent movement that the run
    never reported."""
    held = "OpenAI has not announced a release date for GPT-6"
    runs = [
        run(0, [dimension(held), dimension("The frontier line is the GPT-5.6 family")]),
        run(1, [dimension("Only one dimension about prediction markets here")]),
        run(2, [dimension(held), dimension("The frontier line is the GPT-5.6 family")]),
    ]

    ledger = claim_ledger.build_claim_ledger(runs)

    assert ledger["thin"] == 1
    assert ledger["enumerated"] == 2
    claim = ledger["holding"][0]
    assert claim["streak"] == 2, "the gap must not break the streak"
    assert claim["appearances"] == 2
    assert [tick["state"] for tick in claim["lifeline"]] == ["on", "gap", "on"]
    assert ledger["retired"] == []


def test_half_sentences_from_the_differences_judge_never_become_claims():
    runs = [
        run(0, [
            dimension("OpenAI has not announced a release date for GPT-6"),
            dimension("recent releases have focused on the"),
            dimension("still no"),
        ]),
    ]

    ledger = claim_ledger.build_claim_ledger(runs)

    labels = [claim["label"] for group in ("new", "contested", "holding", "retired")
              for claim in ledger[group]]
    assert labels == ["OpenAI has not announced a release date for GPT-6"]
    # One usable claim is not an inventory of the answer.
    assert ledger["enumerated"] == 0


def test_claim_keys_decide_identity_where_wording_would_mislead():
    """The identity Judge stamps a key on every claim of a run. A key joins two
    claims no lexical comparison would join, and keeps two apart that a lexical
    comparison would merge."""
    runs = [
        run(0, [
            dimension("OpenAI has not announced a release date for GPT-6", key="k1"),
            dimension("Pricing for the API tier stayed flat this quarter", key="k2"),
        ]),
        run(1, [
            dimension("No launch window is on record for the next model", key="k1"),
            dimension("Pricing for the API tier stayed flat this month", key="k3"),
        ]),
    ]

    ledger = claim_ledger.build_claim_ledger(runs)
    by_label = {claim["label"]: claim for group in ("new", "contested", "holding", "retired")
                for claim in ledger[group]}

    joined = by_label["No launch window is on record for the next model"]
    assert joined["appearances"] == 2, "the key joins two unlike phrasings"
    assert by_label["Pricing for the API tier stayed flat this month"]["appearances"] == 1
    assert ledger["one_off_count"] == 1, "k2 and k3 stay separate claims"


def test_a_claim_absent_from_the_newest_check_is_reported_as_dropped():
    dropped = "Any date circulating online is speculation"
    runs = [
        run(0, [dimension("No official release date has been announced"), dimension(dropped)]),
        run(1, [dimension("No official release date has been announced"), dimension(dropped)]),
        run(2, [
            dimension("No official release date has been announced"),
            dimension("The GPT-5.6 family is the current frontier line"),
        ]),
    ]

    ledger = claim_ledger.build_claim_ledger(runs)

    assert [claim["label"] for claim in ledger["retired"]] == [dropped]
    assert ledger["retired"][0]["last_display"] == "Jul 24, 2026"
    assert ledger["retired"][0]["appearances"] == 2
    assert ledger["retired"][0]["holding"] is False


def test_contested_claims_are_separated_from_the_ones_all_models_state():
    runs = [
        run(0, [
            dimension("No official release date has been announced"),
            split_dimension("The likely launch window"),
        ]),
        run(1, [
            dimension("No official release date has been announced"),
            split_dimension("The likely launch window"),
        ]),
    ]

    ledger = claim_ledger.build_claim_ledger(runs)

    assert [claim["label"] for claim in ledger["contested"]] == ["The likely launch window"]
    assert ledger["contested"][0]["positions"][0]["models"] == ["OpenAI"]
    assert [claim["label"] for claim in ledger["holding"]] == ["No official release date has been announced"]


def test_record_summary_anchors_on_material_change_not_on_score_movement():
    """The agreement score steps between a few grading levels, so a step is not
    a change. Only the Change Judge's grade may anchor the headline."""
    runs = [
        run(0, [dimension("No official release date has been announced")], score=90),
        run(1, [dimension("No official release date has been announced")], score=64,
            change_type="major", summary="A rumoured window entered the answer."),
        run(2, [dimension("No official release date has been announced")], score=90),
        run(3, [dimension("No official release date has been announced")], score=64),
    ]

    record = claim_ledger.build_record_summary(runs)

    assert record["checks"] == 4
    assert record["material_count"] == 1
    assert record["anchor_display"] == "Jul 24, 2026"
    assert record["steady_checks"] == 2
    assert record["steady_days"] == 2
    assert record["changed_now"] is False
    assert record["material_events"][0]["summary"] == "A rumoured window entered the answer."


def test_a_minor_grade_restates_the_answer_and_does_not_anchor_the_record():
    """"minor" is the Judge saying the wording moved, not the answer. A record
    that anchors on it reports a change after nearly every check, which is the
    state the drift rule was tightened to end."""
    runs = [
        run(0, [dimension("No official release date has been announced")]),
        run(1, [dimension("No official release date has been announced")],
            change_type="minor", summary="A qualification was rephrased."),
        run(2, [dimension("No official release date has been announced")],
            change_type="minor", summary="A citation was swapped."),
    ]

    record = claim_ledger.build_record_summary(runs)

    assert record["material_count"] == 0
    assert record["changed_now"] is False
    assert record["steady_checks"] == 2


def test_record_summary_without_any_material_change_points_at_the_first_check():
    runs = [run(index, [dimension("No official release date has been announced")]) for index in range(5)]

    record = claim_ledger.build_record_summary(runs)

    assert record["material_count"] == 0
    assert record["steady_checks"] == 4
    assert record["first_display"] == "Jul 23, 2026"
    assert record["anchor_is_first"] is True


def test_unchanged_checks_fold_into_one_timeline_entry():
    """Fourteen 'wording only' rows hide the two rows that matter."""
    runs = [
        run(6, [], change_type="stable"),
        run(5, [], change_type="stable"),
        run(4, [], change_type="stable"),
        run(3, [], change_type="major", summary="The answer flipped."),
        run(2, [], change_type="stable"),
        run(1, [], change_type="stable"),
        run(0, [], change_type="stable"),
    ]
    runs[4]["is_selected"] = True

    entries = claim_ledger.collapse_timeline(runs)

    kinds = [entry["kind"] for entry in entries]
    assert kinds == ["run", "quiet", "run", "run", "run", "run"]
    folded = entries[1]
    assert folded["count"] == 2
    assert folded["from_display"] == "Jul 28, 2026"
    assert folded["to_display"] == "Jul 27, 2026"
    # Newest, oldest, the material change and the selected run stay visible.
    assert [entry["run"]["id"] for entry in entries if entry["kind"] == "run"] == [
        "run-6", "run-3", "run-2", "run-1", "run-0",
    ]


def test_sources_are_dated_and_the_ones_that_left_the_record_are_listed():
    old = source("https://openai.com/old-note", "Superseded note")
    kept = source("https://openai.com/release-notes")
    fresh = source("https://openai.com/newsroom", "Newsroom")
    runs = [
        run(0, [], evidence=[old, kept]),
        run(1, [], evidence=[kept]),
        run(2, [], evidence=[kept, fresh]),
    ]
    selected = runs[2]

    chronicle = claim_ledger.apply_source_chronicle(runs, selected)

    tracked = {item["url"]: item for item in selected["evidence"]}
    assert tracked["https://openai.com/release-notes"]["first_display"] == "Jul 23, 2026"
    assert tracked["https://openai.com/release-notes"]["appearances"] == 3
    assert tracked["https://openai.com/release-notes"]["is_new"] is False
    assert tracked["https://openai.com/newsroom"]["is_new"] is True
    assert chronicle["new_count"] == 1
    assert chronicle["tracked_count"] == 3
    assert [item["url"] for item in chronicle["retired"]] == ["https://openai.com/old-note"]
    assert chronicle["retired"][0]["last_display"] == "Jul 23, 2026"
    # Single URLs churn with every web search; the site count is the part of
    # the source record that survives that.
    assert chronicle["sites"] == [{"host": "openai.com", "checks": 3}]
    assert chronicle["checks"] == 3


def test_a_site_cited_in_a_single_check_is_not_called_part_of_the_record():
    redirect = dict(
        source("https://vertexaisearch.cloud.google.com/id/1",
               host="vertexaisearch.cloud.google.com"),
        is_indirect=True,
    )
    runs = [
        run(0, [], evidence=[source("https://openai.com/a"), redirect]),
        run(1, [], evidence=[
            source("https://openai.com/b"),
            source("https://rumors.example/x", host="rumors.example"),
            dict(redirect, url="https://vertexaisearch.cloud.google.com/id/2"),
        ]),
    ]

    chronicle = claim_ledger.apply_source_chronicle(runs, runs[1])

    # A single check is not a pattern, and a grounding redirect is not a site.
    assert chronicle["sites"] == [{"host": "openai.com", "checks": 2}]


def test_the_first_snapshot_of_a_topic_flags_no_source_as_new():
    """Everything is new in the first check, so the badge would say nothing."""
    runs = [run(0, [], evidence=[source("https://openai.com/release-notes")])]

    chronicle = claim_ledger.apply_source_chronicle(runs, runs[0])

    assert chronicle["new_count"] == 0
    assert runs[0]["evidence"][0]["is_new"] is False


def test_known_claims_carry_the_newest_wording_of_each_tracked_claim():
    runs = [
        run(0, [dimension("No official release date has been announced", key="k1")]),
        run(1, [
            dimension("No launch date has been announced so far", key="k1"),
            dimension("The GPT-5.6 family is the frontier line", key="k2"),
        ]),
    ]

    known = topic_runner.known_claims_from_runs(runs)

    assert known == [
        {"key": "k1", "label": "No launch date has been announced so far"},
        {"key": "k2", "label": "The GPT-5.6 family is the frontier line"},
    ]


def test_a_topic_without_any_position_map_has_no_ledger():
    assert claim_ledger.build_claim_ledger([run(0, [])]) is None
    assert claim_ledger.build_claim_ledger([]) is None
    assert claim_ledger.build_record_summary([]) is None


def test_a_claim_clipped_mid_citation_does_not_keep_half_a_marker():
    """Position Map labels are stored clipped, so a citation marker can be cut
    in half. A reader must never see "[S4" as part of the sentence."""
    stored = (
        "As of the latest official information available, there is no confirmed "
        "OpenAI blog post, product page, API model entry, or release note that "
        "says when GPT-6 will launch.[S4"
    )

    text = claim_ledger._claim_text(stored)

    assert text.endswith("will launch.")
    assert "[S" not in text


def test_a_claim_carries_the_sources_its_own_wording_cites():
    """Evidence is numbered per run, so a claim's markers only resolve against
    the run its current wording came from."""
    early = source("https://openai.com/early", title="Early note")
    late = source("https://openai.com/late", title="Release notes")
    runs = [
        run(0, [dimension("No release date has been announced[S1]")],
            evidence=[{**early, "id": "S1"}]),
        run(1, [dimension("No release date has been announced for GPT-6[S2]")],
            evidence=[{**late, "id": "S1"}, {**early, "id": "S2"}]),
    ]

    ledger = claim_ledger.build_claim_ledger(runs)
    claim_ledger.attach_claim_sources(ledger, runs)
    claim = ledger["holding"][0]

    assert claim["source_ids"] == ["2"]
    # S2 in the newest run, not S2 as it would have been numbered earlier.
    assert [item["url"] for item in claim["sources"]] == ["https://openai.com/early"]


def test_a_claim_without_markers_lists_no_sources_of_its_own():
    runs = [run(0, [dimension("No release date has been announced")],
                evidence=[{**source("https://openai.com/a"), "id": "S1"}])]

    ledger = claim_ledger.build_claim_ledger(runs)
    claim_ledger.attach_claim_sources(ledger, runs)

    assert ledger["holding"][0]["sources"] == []


def test_the_check_strip_gives_every_check_one_cell_and_a_reason():
    """One cell per check, oldest first, each saying what happened in it."""
    held = "No release date has been announced for GPT-6"
    runs = [
        run(0, [dimension(held), dimension("The GPT-5.6 family is current")]),
        run(1, [dimension(held), dimension("The GPT-5.6 family is current")]),
        run(2, [dimension(held), dimension("The GPT-5.6 family is current")],
            change_type="major", summary="A rumoured window entered the answer."),
        run(3, [dimension(held), dimension("Pricing has not been published")]),
    ]

    ledger = claim_ledger.build_claim_ledger(runs)
    strip = claim_ledger.build_check_strip(runs, ledger)

    assert [cell["kind"] for cell in strip] == ["first", "stable", "material", "event"]
    assert strip[0]["note"] == "First check. The record starts here."
    assert strip[2]["note"] == "A rumoured window entered the answer."
    # The last check both gained and lost a statement, and says so.
    assert "entered" in strip[3]["note"] and "dropped out" in strip[3]["note"]
    assert strip[-1]["is_latest"] is True
    assert [cell["is_latest"] for cell in strip[:-1]] == [False, False, False]
    assert strip[2]["run_id"] == "run-2"


def test_the_check_strip_stands_alone_without_a_ledger():
    """Manually seeded Topics carry no Position Map. The strip is still the
    record, so it has to render from the runs alone."""
    strip = claim_ledger.build_check_strip([run(0, []), run(1, [])])

    assert [cell["kind"] for cell in strip] == ["first", "stable"]
    assert claim_ledger.build_check_strip([]) == []


def test_the_check_strip_keeps_the_newest_checks_when_a_topic_runs_long():
    runs = [run(index, [dimension("No release date has been announced")])
            for index in range(claim_ledger.MAX_STRIP_CELLS + 12)]

    strip = claim_ledger.build_check_strip(runs)

    assert len(strip) == claim_ledger.MAX_STRIP_CELLS
    assert strip[-1]["run_id"] == runs[-1]["id"]
