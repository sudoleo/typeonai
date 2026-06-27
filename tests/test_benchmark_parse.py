"""§9.2 – extract_letter, majority_vote (inkl. Ties -> no_majority), grade."""

import unittest

from benchmark.parse import NO_MAJORITY, extract_letter, grade, majority_vote


class ExtractLetterTests(unittest.TestCase):
    def test_answer_is_phrasings(self):
        self.assertEqual(extract_letter("The answer is (C)."), "C")
        self.assertEqual(extract_letter("the answer is C"), "C")
        self.assertEqual(extract_letter("Answer: B"), "B")
        self.assertEqual(extract_letter("Some reasoning.\nThe answer is (D)."), "D")

    def test_last_answer_wins(self):
        self.assertEqual(
            extract_letter("First the answer is (A) but on reflection the answer is (E)."),
            "E",
        )

    def test_line_starting_with_letter(self):
        self.assertEqual(extract_letter("Reasoning here.\nC) Option three"), "C")
        self.assertEqual(extract_letter("D."), "D")

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
