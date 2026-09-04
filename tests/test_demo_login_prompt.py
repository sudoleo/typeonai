import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DemoLoginPromptContractTests(unittest.TestCase):
    def test_prompt_is_hidden_until_demo_finishes(self):
        template = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
        demo_module = (ROOT / "static" / "demo.js").read_text(encoding="utf-8")
        shell = (ROOT / "static" / "css" / "shell.css").read_text(encoding="utf-8")

        self.assertIn('id="postDemoLoginPrompt"', template)
        self.assertIn('aria-live="polite" hidden', template)
        self.assertIn("showPostDemoLoginPrompt();", demo_module)
        self.assertIn("if (!prompt || window.auth?.currentUser) return;", demo_module)
        self.assertIn(
            "body:has(#postDemoLoginPrompt:not([hidden])) .chat-input-container",
            shell,
        )

    def test_prompt_opens_login_and_is_removed_after_auth(self):
        template = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
        demo_module = (ROOT / "static" / "demo.js").read_text(encoding="utf-8")
        app_init = (ROOT / "static" / "js" / "app-init.js").read_text(encoding="utf-8")

        self.assertIn('id="postDemoLoginButton"', template)
        self.assertIn('document.getElementById("loginModal")', demo_module)
        self.assertIn("postDemoLoginPrompt.hidden = true;", app_init)

    def test_question_is_cleared_before_model_loading_starts(self):
        demo_module = (ROOT / "static" / "demo.js").read_text(encoding="utf-8")
        flow = demo_module.split("async function runDemoFlow()", 1)[1]
        typed = flow.index("await typeIntoInput")
        cleared = flow.index('qi.value = "";')
        loading = flow.index('window.setAgentModeStatus?.("running");')

        self.assertLess(typed, cleared)
        self.assertLess(cleared, loading)
        self.assertNotIn(
            'window.setAgentModeStatus?.("running");',
            flow[:cleared],
        )

    def test_demo_result_has_an_agreement_score(self):
        demo_module = (ROOT / "static" / "demo.js").read_text(encoding="utf-8")

        self.assertIn("agreement: {", demo_module)
        self.assertIn("score: 45,", demo_module)

    def test_every_checkable_demo_passage_has_a_coverage_claim(self):
        demo_module = (ROOT / "static" / "demo.js").read_text(encoding="utf-8")
        claims = demo_module.split("    claims: [", 1)[1].split(
            "    differences: [", 1
        )[0]
        anchors = (
            "Consensus: send it — after two fixes",
            "All six models read the draft as close to sendable",
            "Nothing in the draft is impolite",
            "The risk is in three sentences",
            "The new date belongs in the first line",
            "She is scanning for a date.",
            "Send it today, not on the 15th",
            "Say what Anna actually gets on the 15th",
            "Put it in writing, so she can forward it",
            "Name the day you will confirm the 29th",
            "The closing line splits the models down the middle",
            "A few things came up on our side is the weakest sentence in the draft",
            "The apology itself is not disputed",
            "Hi Anna, the launch moves to the 29th",
            "One clause on the cause.",
            "What you will have on the 15th is the checkout flow on staging",
            "I will confirm the 29th by the 22nd at the latest.",
            "Your closing line.",
            "Both bracketed parts are the ones the models could not settle for you",
        )
        for anchor in anchors:
            self.assertIn(f'anchor: "{anchor}"', claims)
        self.assertEqual(claims.count('coverage: "'), len(anchors))


if __name__ == "__main__":
    unittest.main()
