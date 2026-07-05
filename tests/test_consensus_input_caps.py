import unittest

import app.core.config as cfg
from app.api.routers.chat import cap_engine_text


class CapEngineTextTests(unittest.TestCase):
    def test_short_text_passes_unchanged(self):
        self.assertEqual(cap_engine_text("hello", 100), "hello")

    def test_none_and_non_strings_pass_through(self):
        self.assertIsNone(cap_engine_text(None, 100))
        self.assertEqual(cap_engine_text(123, 100), 123)
        self.assertEqual(cap_engine_text(["a"], 100), ["a"])

    def test_oversized_text_is_truncated(self):
        capped = cap_engine_text("a" * 500, 100)
        self.assertEqual(len(capped), 100)

    def test_truncation_strips_trailing_whitespace(self):
        text = "word " * 100  # 500 Zeichen, Schnitt landet auf einem Space
        capped = cap_engine_text(text, 103)
        self.assertFalse(capped.endswith(" "))

    def test_empty_string_stays_falsy(self):
        # Wichtig für die Missing-Parameter-Validierung in /consensus:
        # Kappen darf aus einem leeren Wert keinen truthy Wert machen.
        self.assertEqual(cap_engine_text("", 100), "")


class ConsensusInputLimitTests(unittest.TestCase):
    def test_default_limits_exist_and_are_generous(self):
        self.assertGreaterEqual(cfg.get_consensus_answer_char_limit(), 30_000)
        self.assertGreaterEqual(cfg.get_consensus_question_char_limit(), 4_000)

    def test_limits_are_admin_overridable(self):
        original = cfg.get_limits_config()
        try:
            overrides = dict(original)
            overrides["consensus_max_answer_chars"] = 12_345
            overrides["consensus_max_question_chars"] = 1_234
            cfg.apply_limits(overrides)
            self.assertEqual(cfg.get_consensus_answer_char_limit(), 12_345)
            self.assertEqual(cfg.get_consensus_question_char_limit(), 1_234)
        finally:
            cfg.apply_limits(original)

    def test_answer_limit_never_clips_legit_deep_search_answers(self):
        # 8192 Output-Tokens entsprechen grob 32k Zeichen; der Cap muss
        # darüber liegen, sonst kappt er echte Antworten statt Abuse.
        approx_max_answer_chars = cfg.LIMITS["pro_deep_search_max_tokens"] * 4
        self.assertGreaterEqual(
            cfg.get_consensus_answer_char_limit(), approx_max_answer_chars
        )


if __name__ == "__main__":
    unittest.main()
