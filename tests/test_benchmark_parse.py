"""§9.2 - extract_letter, majority_vote, grade."""

import unittest

from benchmark.parse import NO_MAJORITY, extract_letter, grade, majority_vote


class ExtractLetterTests(unittest.TestCase):
    def test_final_answer_marker_on_last_line(self):
        self.assertEqual(
            extract_letter("Brief reason first.\nFINAL_ANSWER: C"),
            "C",
        )
        self.assertEqual(
            extract_letter("The supported calculation gives 2/3.\n\nFINAL_ANSWER: J"),
            "J",
        )

    def test_marker_must_be_last_non_empty_line(self):
        self.assertIsNone(extract_letter("FINAL_ANSWER: B\nAdditional text."))
        self.assertEqual(extract_letter("Reasoning.\nFINAL_ANSWER: B\n\n"), "B")

    def test_marker_format_is_strict(self):
        self.assertIsNone(extract_letter("Final answer: B"))
        self.assertIsNone(extract_letter("FINAL_ANSWER: (B)"))
        self.assertIsNone(extract_letter("FINAL_ANSWER: B."))
        self.assertIsNone(extract_letter("final_answer: B"))
        self.assertIsNone(extract_letter("FINAL_ANSWER:B"))
        self.assertIsNone(extract_letter("FINAL_ANSWER: b"))

    def test_no_semantic_option_text_or_letter_fallback(self):
        options = ["London", "Paris", "Berlin"]
        self.assertIsNone(extract_letter("Paris is the capital.", options=options))
        self.assertIsNone(extract_letter("The answer is (B)."))
        self.assertIsNone(extract_letter("Answer: B"))
        self.assertIsNone(extract_letter("Option B"))
        self.assertIsNone(extract_letter("B."))
        self.assertIsNone(extract_letter("(B)"))

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
