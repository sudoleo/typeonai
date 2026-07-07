import os

from slowapi import Limiter
from slowapi.util import get_remote_address


def client_ip_key(request):
    """Client-IP hinter dem Render-Proxy für IP-basierte Rate-Limits.

    uvicorn läuft ohne --proxy-headers (Start-Kommando liegt im Render-
    Dashboard, nicht im Repo), daher wäre request.client.host immer die
    Proxy-IP und alle IP-Limits würden global statt pro Besucher greifen.

    Render hängt die tatsächliche Verbindungs-IP als letzten Eintrag an
    X-Forwarded-For an. Genau ein vertrauenswürdiger Proxy => der letzte
    Eintrag ist die echte Client-IP und kann – anders als der erste – nicht
    durch einen mitgeschickten Header gefälscht werden. Lokal (kein Proxy,
    kein Header) fällt es auf request.client.host zurück.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        last_hop = forwarded.split(",")[-1].strip()
        if last_hop:
            return last_hop
    return get_remote_address(request)


# DISABLE_RATE_LIMIT=1 nur fuer die lokale E2E-Suite: die Tests feuern
# mehrere /ask_*- und /consensus-Requests pro Minute von derselben IP und
# wuerden sonst in 429er laufen. In Produktion nie setzen.
limiter = Limiter(key_func=client_ip_key, enabled=os.environ.get("DISABLE_RATE_LIMIT") != "1")
