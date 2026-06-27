"""§9.2 – extract_letter, majority_vote (inkl. Ties -> no_majority), grade."""

import unittest

from benchmark.parse import NO_MAJORITY, extract_letter, grade, majority_vote


class ExtractLetterTests(unittest.TestCase):
    def test_answer_is_phrasings(self):
        self.assertEqual(extract_letter("The answer is (C)."), "C")
        self.assertEqual(extract_letter("The answer is (I)"), "I")
        self.assertEqual(extract_letter("the answer is C"), "C")
        self.assertEqual(extract_letter("Answer: B"), "B")
        self.assertEqual(extract_letter("Final answer: I"), "I")
        self.assertEqual(extract_letter("Some reasoning.\nThe answer is (D)."), "D")

    def test_clear_final_answer_signal_wins_over_non_final_reasoning(self):
        self.assertEqual(
            extract_letter("A is tempting because of the first clause.\nFinal answer: E"),
            "E",
        )

    def test_option_signal(self):
        self.assertEqual(extract_letter("After weighing the choices, Option I."), "I")
        self.assertEqual(extract_letter("option c"), "C")

    def test_line_starting_with_letter(self):
        self.assertEqual(extract_letter("Reasoning here.\nC) Option three"), "C")
        self.assertEqual(extract_letter("D."), "D")
        self.assertEqual(extract_letter("(I)"), "I")

    def test_bold_letter(self):
        self.assertEqual(extract_letter("**C**"), "C")

    def test_option_text_match(self):
        text = "After thinking, Paris is the capital."
        options = ["London", "Paris", "Berlin"]
        self.assertEqual(extract_letter(text, options=options), "B")

    def test_option_text_ambiguous_returns_none(self):
        text = "It could be London or Paris."
        options = ["London", "Paris", "Berlin"]
        self.assertIsNone(extract_letter(text, options=options))

    def test_option_components_match_unique_fill_in_answer(self):
        text = (
            "The blanks should be filled as unsafe practices, fear or distress, "
            "and serious offence."
        )
        options = [
            "Safe practices, Fear, Jealousy, Trivial",
            "Unsafe practices, Distress, Joy, Trivial",
            "Unsafe practices, Distress, Fear, Serious",
        ]
        self.assertEqual(extract_letter(text, options=options), "C")

    def test_option_components_ambiguous_returns_none(self):
        text = "The final wording uses red, blue, and green."
        options = [
            "Red, Blue, Green",
            "Red, Blue, Green",
            "Red, Blue, Yellow",
        ]
        self.assertIsNone(extract_letter(text, options=options))

    def test_smoke_consensus_text_matches_option_i_strictly(self):
        text = (
            "Typical advertising regulatory bodies, such as the Advertising Standards "
            "Authority (ASA) in the UK, have strict codes of conduct. Based on standard "
            "advertising regulations, the blanks can be filled in as follows:\n\n"
            "Adverts must not: encourage **unsafe practices**, cause unnecessary "
            "**fear** or **distress**, and must not cause **serious** (or widespread) offence."
        )
        options = [
            "Safe practices, Fear, Jealousy, Trivial",
            "Unsafe practices, Distress, Joy, Trivial",
            "Safe practices, Wants, Jealousy, Trivial",
            "Safe practices, Distress, Fear, Trivial",
            "Unsafe practices, Wants, Jealousy, Serious",
            "Safe practices, Distress, Jealousy, Serious",
            "Safe practices, Wants, Fear, Serious",
            "Unsafe practices, Wants, Fear, Trivial",
            "Unsafe practices, Distress, Fear, Serious",
        ]
        self.assertEqual(extract_letter(text, options=options), "I")

    def test_false_positive_answer_mentions_return_none(self):
        self.assertIsNone(extract_letter("Option I is discussed but not selected."))
        self.assertIsNone(extract_letter("Answer: Image classification is hard."))
        self.assertIsNone(extract_letter("I. think the evidence is insufficient."))

    def test_conflicting_final_signals_return_none(self):
        self.assertIsNone(extract_letter("Final answer: A\nAnswer: B"))
        self.assertIsNone(extract_letter("Final answer: A. Option B."))
        self.assertEqual(extract_letter("Final answer: B\nAnswer: B"), "B")

    def test_garbage_returns_none(self):
        self.assertIsNone(extract_letter("I cannot determine this."))
        self.assertIsNone(extract_letter(""))
        self.assertIsNone(extract_letter(None))


class MajorityVoteTests(unittest.TestCase):
    def test_clear_majority(self):
        self.assertEqual(majority_vote(["A", "A", "B", "A", "C", "B"]), "A")

    def test_tie_is_no_majority(self):
        self.assertEqual(majority_vote(["A", "A", "B", "B"]), NO_MAJORITY)
        self.assertEqual(majority_vote(["A", "B", "C"]), NO_MAJORITY)

    def test_abstain_votes_ignored(self):
        self.assertEqual(majority_vote(["A", None, "A", None, "B"]), "A")

    def test_all_abstain_is_no_majority(self):
        self.assertEqual(majority_vote([None, None]), NO_MAJORITY)
        self.assertEqual(majority_vote([]), NO_MAJORITY)


class GradeTests(unittest.TestCase):
    def test_correct(self):
        self.assertTrue(grade("C", "C"))
        self.assertTrue(grade("c", "C"))

    def test_incorrect(self):
        self.assertFalse(grade("A", "C"))

    def test_no_majority_and_none_never_correct(self):
        self.assertFalse(grade(NO_MAJORITY, "C"))
        self.assertFalse(grade(None, "C"))


if __name__ == "__main__":
    unittest.main()
