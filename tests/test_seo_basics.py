import unittest
from pathlib import Path

from app.api.routers.pages import SITE_URL, robots_txt, sitemap_xml


ROOT = Path(__file__).resolve().parents[1]


class SeoBasicsTests(unittest.TestCase):
    def test_robots_txt_points_to_sitemap(self):
        content = robots_txt()

        self.assertIn("User-agent: *", content)
        self.assertIn("Allow: /", content)
        self.assertIn(f"Sitemap: {SITE_URL}/sitemap.xml", content)

    def test_sitemap_contains_public_indexable_pages(self):
        response = sitemap_xml()
        content = response.body.decode("utf-8")

        self.assertEqual(response.media_type, "application/xml")
        self.assertIn(f"<loc>{SITE_URL}/</loc>", content)
        self.assertIn(f"<loc>{SITE_URL}/ai-model-comparison</loc>", content)
        self.assertIn(f"<loc>{SITE_URL}/about</loc>", content)
        self.assertNotIn(f"<loc>{SITE_URL}/app</loc>", content)
        self.assertNotIn(f"<loc>{SITE_URL}/privacy</loc>", content)
        self.assertNotIn(f"<loc>{SITE_URL}/imprint</loc>", content)

    def test_landing_template_has_core_seo_metadata(self):
        template = (ROOT / "templates" / "landing.html").read_text(encoding="utf-8")

        self.assertIn('<link rel="canonical" href="https://www.consens.io/">', template)
        self.assertIn('property="og:title"', template)
        self.assertIn('name="twitter:card"', template)
        self.assertIn('type="application/ld+json"', template)
        self.assertIn('"@type": "WebApplication"', template)

    def test_ai_model_comparison_page_has_seo_metadata(self):
        template = (ROOT / "templates" / "ai-model-comparison.html").read_text(encoding="utf-8")

        self.assertIn('<link rel="canonical" href="https://www.consens.io/ai-model-comparison">', template)
        self.assertIn("Compare AI models", template)
        self.assertIn('"@type": "FAQPage"', template)

    def test_app_template_is_noindex(self):
        template = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")

        self.assertIn('<meta name="robots" content="noindex, follow">', template)


if __name__ == "__main__":
    unittest.main()
