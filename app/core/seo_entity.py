"""Eine Quelle der Wahrheit fuer die Entitaet hinter der Domain.

Google baut ein Knowledge-Panel nicht aus einzelnen Seiten, sondern aus einer
Entitaet, die auf vielen Seiten identisch beschrieben wird. Jede Seite, die
"consens.io" nur als Namensstring nennt, ist fuer Google eine eigene, unsichere
Erwaehnung -- fuenf lose Erwaehnungen statt einer Entitaet mit fuenf Belegen.
Deshalb steht der Organization-Knoten hier genau einmal; Seiten binden ihn ein
und verweisen sonst nur per @id darauf.

Bewusst NICHT enthalten: `founder`. Impressum und Kontakt sind als Canvas-Bilder
gerendert, damit Crawler den Klarnamen nicht abgreifen. Ein founder-Feld im
JSON-LD wuerde genau das aushebeln.
"""
from __future__ import annotations

import json
from typing import Any

from app.core.site import SITE_URL

ORGANIZATION_ID = SITE_URL + "/#organization"
WEBSITE_ID = SITE_URL + "/#website"

# Der NAME muss ueberall zeichengleich stehen -- Website, Profile aus SAME_AS,
# jede Bio. "Consens.io" vs. "consens.io" vs. "Consens" macht die Zuordnung
# unschaerfer. Bei der Beschreibung geht es dagegen um die Aussage, nicht um den
# Wortlaut: dieselben tragenden Begriffe reichen, wenn ein Profilfeld kuerzer ist.
ORGANIZATION_NAME = "consens.io"
ORGANIZATION_DESCRIPTION = (
    "consens.io independently measures where leading AI answers agree, "
    "contradict one another, and change over time."
)

# sameAs ist eine Identitaetsbehauptung, keine Linksammlung: nur Profile, die
# wirklich zu dieser Entitaet gehoeren, erreichbar sind und denselben Namen und
# dieselbe Beschreibung tragen. Tote oder fremde Links schwaechen das Signal.
#
# Die Handles unterscheiden sich (askconsensio, consens_io) -- das ist egal,
# geprueft wird die URL. Der ANZEIGENAME auf beiden Profilen muss "consens.io"
# sein, sonst faellt der Abgleich auseinander.
ORGANIZATION_SAME_AS: list[str] = [
    "https://www.linkedin.com/company/askconsensio/",
    "https://x.com/consens_io",
]

ORGANIZATION: dict[str, Any] = {
    "@type": "Organization",
    "@id": ORGANIZATION_ID,
    "name": ORGANIZATION_NAME,
    "url": SITE_URL + "/",
    "description": ORGANIZATION_DESCRIPTION,
    "logo": {
        "@type": "ImageObject",
        "@id": SITE_URL + "/#logo",
        "url": SITE_URL + "/static/favicon-square.png",
    },
}
if ORGANIZATION_SAME_AS:
    ORGANIZATION["sameAs"] = ORGANIZATION_SAME_AS

WEBSITE: dict[str, Any] = {
    "@type": "WebSite",
    "@id": WEBSITE_ID,
    "name": ORGANIZATION_NAME,
    "url": SITE_URL + "/",
    "description": ORGANIZATION_DESCRIPTION,
    "publisher": {"@id": ORGANIZATION_ID},
    "inLanguage": "en",
}

# Verweis statt Kopie: ueberall dort, wo bisher ein eigener Organization-Block
# mit blossem Namen stand.
ORG_REF: dict[str, str] = {"@id": ORGANIZATION_ID}
SITE_REF: dict[str, str] = {"@id": WEBSITE_ID}


def dumps(payload: Any) -> str:
    """JSON fuer ein <script type="application/ld+json">-Element.

    "</" wird escaped, damit eingebettete Inhalte das script-Element nie
    schliessen koennen.
    """
    return json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")


def page_graph(*nodes: dict[str, Any]) -> dict[str, Any]:
    """Ein JSON-LD-Dokument, das die Entitaet mitliefert statt sie nur zu nennen.

    Ein blosser @id-Verweis auf einen Knoten, der im selben Dokument nirgends
    definiert ist, laesst sich nicht aufloesen. Organization und WebSite stehen
    deshalb in jedem Graph mit drin; gleiche @id heisst fuer JSON-LD "derselbe
    Knoten", die Wiederholung ueber Seiten hinweg ist genau der Beleg, den
    Google sammelt.
    """
    # Im @graph traegt nur das Dokument den Kontext, nicht der einzelne Knoten.
    inner = [{key: value for key, value in node.items() if key != "@context"} for node in nodes]
    return {
        "@context": "https://schema.org",
        "@graph": [ORGANIZATION, WEBSITE, *inner],
    }


def register_seo_globals(templates: Any) -> None:
    """Stellt die Entitaets-Knoten den statischen Templates zur Verfuegung."""
    templates.env.globals["organization_jsonld"] = dumps(ORGANIZATION)
    templates.env.globals["website_jsonld"] = dumps(WEBSITE)
    templates.env.globals["organization_id"] = ORGANIZATION_ID
    templates.env.globals["website_id"] = WEBSITE_ID
