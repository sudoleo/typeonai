"""Eine Entitaet, nicht viele Erwaehnungen.

Google baut ein Knowledge-Panel aus einer Entitaet, die auf vielen Seiten
identisch beschrieben ist. Sobald eine Seite wieder ihren eigenen
Organization-Block mit blossem Namensstring mitbringt, zerfaellt genau das --
und im gerenderten HTML sieht man das nicht.
"""
import json
import re
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routers import pages as pages_router
from app.core import seo_entity

JSONLD_BLOCK = re.compile(r'<script type="application/ld\+json">(.*?)</script>', re.S)

PUBLIC_PAGES = ["/", "/about", "/benchmark", "/ai-model-comparison", "/consensus-engine"]


def graphs(html: str) -> list[dict]:
    """Alle JSON-LD-Knoten einer Seite, egal ob @graph oder Einzelknoten."""
    nodes: list[dict] = []
    for block in JSONLD_BLOCK.findall(html):
        document = json.loads(block)
        nodes.extend(document.get("@graph", [document]))
    return nodes


class SeoEntityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = FastAPI()
        app.include_router(pages_router.router)
        cls.client = TestClient(app)

    def test_every_public_page_carries_the_same_organization(self):
        for path in PUBLIC_PAGES:
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)

                orgs = [n for n in graphs(response.text) if n.get("@type") == "Organization"]
                self.assertEqual(len(orgs), 1, "genau ein Organization-Knoten pro Seite")
                self.assertEqual(orgs[0]["@id"], seo_entity.ORGANIZATION_ID)
                self.assertEqual(orgs[0]["name"], seo_entity.ORGANIZATION_NAME)
                self.assertEqual(orgs[0]["description"], seo_entity.ORGANIZATION_DESCRIPTION)

    def test_no_page_invents_a_second_organization_by_name(self):
        # Ein {"@type": "Organization", "name": "consens.io"} ohne @id ist fuer
        # Google eine neue, unsichere Entitaet -- der Fehler, der hier zurueckkam.
        for path in PUBLIC_PAGES:
            with self.subTest(path=path):
                for node in graphs(self.client.get(path).text):
                    for value in node.values():
                        if isinstance(value, dict) and value.get("@type") == "Organization":
                            self.fail(f"{path}: Organization inline statt per @id referenziert")

    def test_id_references_resolve_inside_the_same_document(self):
        for path in PUBLIC_PAGES:
            with self.subTest(path=path):
                nodes = graphs(self.client.get(path).text)
                defined = {n["@id"] for n in nodes if "@id" in n}
                referenced = {
                    value["@id"]
                    for node in nodes
                    for value in node.values()
                    if isinstance(value, dict) and set(value) == {"@id"}
                }
                self.assertLessEqual(referenced, defined, "Verweis auf undefinierten Knoten")

    def test_same_as_holds_only_absolute_urls(self):
        # sameAs ist eine Identitaetsbehauptung. Ein Handle oder ein relativer
        # Pfad belegt nichts und schwaecht das Signal.
        for url in seo_entity.ORGANIZATION_SAME_AS:
            with self.subTest(url=url):
                self.assertRegex(url, r"^https://")

    def test_founder_stays_out_of_structured_data(self):
        # Impressum und Kontakt sind als Canvas gerendert, damit Crawler den
        # Klarnamen nicht abgreifen. JSON-LD darf das nicht unterlaufen.
        self.assertNotIn("founder", seo_entity.ORGANIZATION)

    def test_page_graph_embeds_the_entity_next_to_the_page_node(self):
        document = seo_entity.page_graph(
            {"@context": "https://schema.org", "@type": "Article", "headline": "x"}
        )
        types = [node["@type"] for node in document["@graph"]]

        self.assertEqual(types, ["Organization", "WebSite", "Article"])
        self.assertNotIn("@context", document["@graph"][-1])


if __name__ == "__main__":
    unittest.main()
