"""The one sentence a Topic page has to say before anything else.

A Topic page holds more about its question than anything else on the web: a
claim-by-claim record, dated sources, every earlier answer. None of that helps
a reader who arrived from a search box and wants to know whether GPT-6 has a
release date. Until the page states the finding, the record is a filing
cabinet with no label on the drawer.

The finding is *derived*, never generated: it is the claim the record itself
puts first -- longest unbroken run of restatements, then the widest model
support, then the shortest wording, because a headline is read, not parsed.
That makes it true by construction, free of an extra model call, and correct
for every run already stored, including the ones published years before this
module existed.

A run may carry an editorial ``headline`` written by hand; when it does, it
wins. Nothing writes that field today.
"""

from __future__ import annotations

import re

from app.services.opinion_map import similarity
from app.services.public_markdown import markdown_to_plaintext


MAX_LINE = 320
MAX_SUPPORT = 2
# A supporting sentence has to be readable in one pass under the finding.
# Anything longer is a paragraph, and it belongs in the statement list.
MAX_SUPPORT_LINE = 130
# Two claims this close say the same thing twice under the headline.
SUPPORT_OVERLAP = 0.5
# Below this a claim reads as a headline; above it the page sets it smaller so
# a long sentence still fits the same block.
LONG_LINE = 108
SETTLED_STREAK = 2
# A claim is only "settled" if it held through a real share of the checks that
# listed claims at all. Two restatements out of twenty is churn, not a record.
SETTLED_SHARE = 3
MIN_HEADLINE_WORDS = 6
MIN_SUPPORT_WORDS = 5
# How close a consensus sentence has to be to a clipped label to count as the
# same statement, written out in full.
RESTATE_OVERLAP = 0.4
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

STATE_LABELS = {
    "moved": "The answer moved",
    "split": "The models disagree",
    "settled": "Settled",
    "forming": "Still forming",
    "first": "First check",
}


def _sentence(text: str) -> str:
    text = markdown_to_plaintext(text, limit=MAX_LINE).strip()
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    # A sentence that ends inside a quotation already has its full stop; adding
    # a second one after the closing quote reads as a typo.
    closed = text.rstrip("\"'”’)]")
    if not closed or closed[-1] not in ".!?…":
        text += "."
    return text


def _is_clipped(label) -> bool:
    """Position Map labels are stored clipped, so some end mid-sentence."""
    return str(label or "").rstrip().endswith("…")


def _reads_as_a_sentence(label, *, min_words: int) -> bool:
    """Whether a stored label can stand on its own as a statement.

    A label that starts lower-case is the tail of a sentence the judge quoted
    from the middle of an answer ("it was a real research problem"). It is a
    true fragment of the record and still unusable as the line a reader is
    supposed to take away.
    """
    text = str(label or "").strip()
    if not text or _is_clipped(text):
        return False
    if not (text[0].isupper() or text[0].isdigit()):
        return False
    return len(text.split()) >= min_words


def _consensus_sentences(selected) -> list[str]:
    plain = markdown_to_plaintext((selected or {}).get("consensus_md"), limit=1200)
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(plain) if part.strip()]


def _restated_in_full(label, selected) -> str:
    """The consensus sentence a clipped label was cut out of, if it is there.

    The claim record carries the identity of a statement; the consensus
    carries its full wording. When a label is clipped, taking the wording back
    from the answer it came from is the difference between a finding and a
    sentence that stops mid-word.
    """
    best, score = "", 0.0
    for sentence in _consensus_sentences(selected):
        if not _reads_as_a_sentence(sentence, min_words=MIN_HEADLINE_WORDS):
            continue
        current = similarity(sentence, label)
        if current > score:
            best, score = sentence, current
    return best if score >= RESTATE_OVERLAP else ""


def _headline_claim(ledger: dict):
    """The claim the record puts first.

    Held longest wins, then widest model support, then the shortest wording,
    because a finding is read at a glance and not parsed. Claims still stated
    today and claims that only just entered compete on the same terms: what
    decides is how much of the record stands behind the statement.

    A contested claim is never the headline: the page cannot state as the
    finding something the models do not state the same way.
    """
    candidates = list(ledger.get("holding") or []) + list(ledger.get("new") or [])
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda claim: (
            -int(claim.get("streak") or 0),
            -int(claim.get("model_count") or 0),
            len(str(claim.get("label") or "")),
        ),
    )[0]


def _headline_line(headline, ledger, selected) -> str:
    """The wording the finding is stated in.

    Preference order: the strongest claim as written; the consensus sentence
    it was clipped out of; the strongest claim that reads as a whole sentence.
    """
    if not headline:
        return ""
    label = str(headline.get("label") or "")
    if _reads_as_a_sentence(label, min_words=MIN_HEADLINE_WORDS):
        return _sentence(label)
    restated = _restated_in_full(label, selected)
    if restated:
        return _sentence(restated)
    whole = [
        claim for claim in
        list(ledger.get("holding") or []) + list(ledger.get("new") or [])
        if _reads_as_a_sentence(claim.get("label"), min_words=MIN_HEADLINE_WORDS)
    ]
    if whole:
        return _sentence(max(whole, key=lambda claim: int(claim.get("streak") or 0))["label"])
    return _sentence(label)


def _voice(claim) -> str:
    """How much of the panel is behind the headline, counted in models."""
    if not claim:
        return ""
    stated = int(claim.get("model_count") or 0)
    total = int(claim.get("run_model_count") or 0)
    if not stated:
        return ""
    if total and stated < total:
        return f"{stated} of {total} models state this"
    if stated == 1:
        return "One model states this"
    return f"All {stated} models say the same"


def _state(ledger, record, headline) -> str:
    if record and record.get("changed_now"):
        return "moved"
    if ledger and ledger.get("contested"):
        return "split"
    if record and int(record.get("checks") or 0) <= 1:
        return "first"
    enumerated = int((ledger or {}).get("enumerated") or 0)
    needed = max(SETTLED_STREAK, enumerated // SETTLED_SHARE)
    if headline and int(headline.get("streak") or 0) >= needed:
        return "settled"
    return "forming"


def build_finding(ledger, record, selected, *, lead_question: str = "") -> dict | None:
    """The finding for one snapshot, or ``None`` when there is nothing to say.

    ``selected`` is the run being shown; ``ledger`` and ``record`` are built
    from the runs up to and including it, so an older version states the
    finding as it stood then rather than today's.
    """
    selected = selected or {}
    headline = _headline_claim(ledger) if ledger else None
    line = _sentence(str(selected.get("headline") or ""))
    source = "editorial"
    if not line and headline:
        line = _headline_line(headline, ledger, selected)
        source = "claim"
    if not line:
        # Manually seeded Topics carry no Position Map at all. The first
        # sentence of the consensus is a weaker finding than a tracked claim,
        # but it is still an answer, and an answer beats a score.
        plain = markdown_to_plaintext(selected.get("consensus_md"), limit=400)
        first = plain.split(". ")[0] if plain else ""
        line = _sentence(first)
        source = "consensus"
    if not line:
        return None

    support = []
    supporting = (
        list(ledger.get("holding") or []) + list(ledger.get("new") or [])
        if ledger else []
    )
    for claim in supporting:
        if claim is headline or len(support) >= MAX_SUPPORT:
            continue
        if not _reads_as_a_sentence(claim.get("label"), min_words=MIN_SUPPORT_WORDS):
            continue
        text = _sentence(claim.get("label"))
        if not text or len(text) > MAX_SUPPORT_LINE:
            continue
        if similarity(text, line) >= SUPPORT_OVERLAP:
            continue
        if any(similarity(text, seen) >= SUPPORT_OVERLAP for seen in support):
            continue
        support.append(text)

    state = _state(ledger, record, headline)
    checks = int((record or {}).get("checks") or 0)
    return {
        "line": line,
        "source": source,
        "support": support,
        "state": state,
        "state_label": STATE_LABELS.get(state, "Tracked"),
        "voice": _voice(headline),
        "is_long": len(line) > LONG_LINE,
        "checks": checks,
        "steady_checks": int((record or {}).get("steady_checks") or 0),
        "steady_days": int((record or {}).get("steady_days") or 0),
        "tracked_since": (record or {}).get("first_display") or "",
        "latest_display": (record or {}).get("latest_display") or "",
        "split_count": len((ledger or {}).get("contested") or []),
        "new_count": len((ledger or {}).get("new") or []),
        "lead_question": str(lead_question or ""),
    }
