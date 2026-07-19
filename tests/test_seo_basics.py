import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routers import pages as pages_router
from app.api.routers.pages import SITE_URL, robots_txt, sitemap_xml, sitemap_pages_xml


ROOT = Path(__file__).resolve().parents[1]


class SeoBasicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = FastAPI()
        app.include_router(pages_router.router)
        cls.client = TestClient(app)

    def test_landing_remains_accessible_with_session_cookie(self):
        response = self.client.get("/", headers={"Cookie": "session=valid-looking-session"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("consens.io", response.text)
        self.assertNotEqual(str(response.url.path), "/app")

    def test_app_logo_links_to_landing_page(self):
        template = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
        self.assertIn('<a href="/" id="logoLink"', template)
        self.assertIn('id="logoLink" aria-label="consens.io home"', template)

    def test_robots_txt_points_to_sitemap(self):
        content = robots_txt()

        self.assertIn("User-agent: *", content)
        self.assertIn("Allow: /", content)
        self.assertIn(f"Sitemap: {SITE_URL}/sitemap.xml", content)

    def test_sitemap_is_index_of_pages_and_shares(self):
        response = sitemap_xml()
        content = response.body.decode("utf-8")

        self.assertEqual(response.media_type, "application/xml")
        self.assertIn("<sitemapindex", content)
        self.assertIn(f"<loc>{SITE_URL}/sitemap-pages.xml</loc>", content)
        self.assertIn(f"<loc>{SITE_URL}/sitemap-shares.xml</loc>", content)

    def test_pages_sitemap_contains_public_indexable_pages(self):
        response = sitemap_pages_xml()
        content = response.body.decode("utf-8")

        self.assertEqual(response.media_type, "application/xml")
        self.assertIn(f"<loc>{SITE_URL}/</loc>", content)
        self.assertIn(f"<loc>{SITE_URL}/ai-model-comparison</loc>", content)
        self.assertIn(f"<loc>{SITE_URL}/consensus-engine</loc>", content)
        self.assertIn(f"<loc>{SITE_URL}/questions</loc>", content)
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

    def test_consensus_engine_page_has_seo_metadata(self):
        template = (ROOT / "templates" / "consensus-engine.html").read_text(encoding="utf-8")

        self.assertIn('<link rel="canonical" href="https://www.consens.io/consensus-engine">', template)
        self.assertIn("Consensus Engine", template)
        self.assertIn('"@type": "FAQPage"', template)

    def test_questions_hub_template_has_seo_metadata(self):
        template = (ROOT / "templates" / "questions.html").read_text(encoding="utf-8")

        self.assertIn('<link rel="canonical" href="https://www.consens.io/questions">', template)
        self.assertIn('<meta name="robots" content="index, follow">', template)
        self.assertIn('property="og:title"', template)
        self.assertIn('type="application/ld+json"', template)

    def test_public_nav_and_footer_link_to_questions_hub(self):
        # SEO-Kern des Hubs: indexierte Shares hängen nicht mehr nur in der
        # Sitemap, sondern sind aus Nav + Footer jeder Public-Seite erreichbar.
        nav = (ROOT / "templates" / "partials" / "public_nav.html").read_text(encoding="utf-8")
        footer = (ROOT / "templates" / "partials" / "public_footer.html").read_text(encoding="utf-8")

        self.assertIn('href="/questions"', nav)
        self.assertIn('href="/questions"', footer)

    def test_app_template_is_noindex(self):
        template = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")

        self.assertIn('<meta name="robots" content="noindex, follow">', template)


if __name__ == "__main__":
    unittest.main()
