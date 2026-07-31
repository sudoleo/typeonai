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
        self.assertIn("score: 83,", demo_module)


if __name__ == "__main__":
    unittest.main()
