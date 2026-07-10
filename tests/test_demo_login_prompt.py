import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DemoLoginPromptContractTests(unittest.TestCase):
    def test_prompt_is_hidden_until_demo_finishes(self):
        template = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
        demo_module = (ROOT / "static" / "demo.js").read_text(encoding="utf-8")

        self.assertIn('id="postDemoLoginPrompt"', template)
        self.assertIn('aria-live="polite" hidden', template)
        self.assertIn("showPostDemoLoginPrompt();", demo_module)
        self.assertIn("if (!prompt || window.auth?.currentUser) return;", demo_module)

    def test_prompt_opens_login_and_is_removed_after_auth(self):
        template = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
        demo_module = (ROOT / "static" / "demo.js").read_text(encoding="utf-8")
        app_init = (ROOT / "static" / "js" / "app-init.js").read_text(encoding="utf-8")

        self.assertIn('id="postDemoLoginButton"', template)
        self.assertIn('document.getElementById("loginModal")', demo_module)
        self.assertIn("postDemoLoginPrompt.hidden = true;", app_init)


if __name__ == "__main__":
    unittest.main()
