"""Drei Kontostufen statt eines Pro-Booleans.

Free  - Basis-Modelle, kleines Tageskontingent.
Plus  - Modellauswahl exakt wie Free (KEINE Frontier-/Premium-Modelle, KEIN
        Deep Think), dafuer Anhaenge, Resolve-Runden und das groesste
        Run-Kontingent. Gedacht fuer Tester, die die Funktionen ausprobieren
        sollen, ohne Frontier-Kosten ausloesen zu koennen.
Pro   - alles.

Zwei Regeln halten die Kosten fest:

1. ``is_pro`` behaelt UEBERALL seine alte Bedeutung "darf teure Modelle und
   Deep Think" und ist fuer Plus ``False``. Jeder Pfad, der noch nicht
   tier-bewusst ist, behandelt einen Plus-Account damit automatisch wie Free
   (fail-closed) - ein vergessener Aufrufer verschenkt hoechstens Komfort,
   nie Geld.
2. Alles, was Plus zusaetzlich bekommt, laeuft ueber ``tier`` bzw. ueber die
   Entitlements - nie ueber ``is_pro``.
"""

from __future__ import annotations

from dataclasses import dataclass

TIER_FREE = "free"
TIER_PLUS = "plus"
TIER_PRO = "pro"

#: Aufsteigend geordnet; der Index ist der Rang (siehe ``tier_at_least``).
TIERS = (TIER_FREE, TIER_PLUS, TIER_PRO)

# Historische Schreibweisen aus Firestore. "premium" ist der aelteste Tag und
# muss weiter Pro bedeuten, sonst verlieren bestehende Konten ihren Zugriff.
_TIER_ALIASES = {
    "premium": TIER_PRO,
    "pro": TIER_PRO,
    "plus": TIER_PLUS,
    "free": TIER_FREE,
}


def normalize_tier(value) -> str:
    """Beliebige Tier-Angabe auf ``free``/``plus``/``pro`` normalisieren.

    Booleans bleiben erlaubt, weil die Limit-Getter jahrelang ``is_pro: bool``
    genommen haben und Aufrufer/Tests diese Signatur weiter benutzen:
    ``True`` -> Pro, ``False`` -> Free. Alles Unbekannte faellt auf Free.
    """
    if isinstance(value, bool):
        return TIER_PRO if value else TIER_FREE
    if isinstance(value, Entitlements):
        return value.tier
    text = str(value or "").strip().lower()
    return _TIER_ALIASES.get(text, TIER_FREE)


def tier_rank(value) -> int:
    return TIERS.index(normalize_tier(value))


def tier_at_least(value, minimum: str) -> bool:
    return tier_rank(value) >= tier_rank(minimum)


@dataclass(frozen=True)
class Entitlements:
    """Was eine Stufe darf. Reine Funktionsschalter - Zahlen (Kontingente,
    Wort-/Token-Caps) stehen in der Admin-konfigurierbaren Limits-Tabelle."""

    tier: str

    @property
    def is_pro(self) -> bool:
        """Darf teure Modelle ausloesen. NUR Pro - siehe Regel 1 oben."""
        return self.tier == TIER_PRO

    @property
    def is_plus(self) -> bool:
        """Genau die Plus-Stufe (nicht Pro)."""
        return self.tier == TIER_PLUS

    @property
    def premium_models(self) -> bool:
        return self.is_pro

    @property
    def deep_think(self) -> bool:
        return self.is_pro

    @property
    def attachments(self) -> bool:
        return tier_at_least(self.tier, TIER_PLUS)

    @property
    def resolve(self) -> bool:
        return tier_at_least(self.tier, TIER_PLUS)


FREE = Entitlements(TIER_FREE)
PLUS = Entitlements(TIER_PLUS)
PRO = Entitlements(TIER_PRO)

_BY_TIER = {TIER_FREE: FREE, TIER_PLUS: PLUS, TIER_PRO: PRO}


def entitlements_for(value) -> Entitlements:
    return _BY_TIER[normalize_tier(value)]
