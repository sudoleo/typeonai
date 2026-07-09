"""Anonyme Differences-Telemetrie.

Jeder erfolgreiche Consensus-Lauf erzeugt strukturierte Widersprüche zwischen
Modellen (differences_data). Dieses Modul persistiert daraus ein anonymes
Statistik-Dokument in Firestore (`differences_stats`), damit die Daten später
aggregiert ausgewertet werden können (z. B. welche Modellpaare wie oft und wie
schwer widersprechen).

Datenschutz (bewusste Designentscheidung, DSGVO):
- KEINE Frage-, Antwort- oder Claim-Texte, keine Zitate, keine Anchors.
- KEINE UID, keine IP, kein User-Agent, keine Referenz auf Bookmarks/Shares.
- Nur Zähl- und Strukturdaten plus Modell-/Engine-Metadaten und grobe
  Kontextwerte (z. B. Wortzahl der Frage als Zahl, nie der Text selbst).
Damit sind die Dokumente anonym im Sinne von ErwGr. 26 DSGVO (kein Bezug zu
einer identifizierbaren Person herstellbar) und fallen nicht unter
personenbezogene Daten. Beim Erweitern des Schemas gilt: niemals Inhalte oder
nutzerbezogene Identifikatoren aufnehmen.

Metadaten-Vollständigkeit: schema_version + engine-/modellbezogene Felder sind
Pflicht, damit alte Datensätze bei späteren Analysen nicht entwertet werden.
Schema-Änderungen => schema_version erhöhen und hier dokumentieren.

Schema v2 (ein Dokument pro Consensus-Lauf mit Differences-Ergebnis).
Änderung gegenüber v1 (2026-07-09): + `judges` — Metadaten des
Differences-Judges (und künftig Adjudicators), seit die Judge-Familie immer
eine andere ist als die der Consensus-Engine. Nur Provider-/Modell-Metadaten,
keine Texte.
  schema_version        int    (2)
  created_at            server timestamp
  consensus_model       str    Engine-Key des Consensus-/Judge-Aufrufs
  judges                {differences: {provider, model, tier},
                         adjudicator?: {provider, model, tier}}
                         tatsächlich genutzter Judge (nach Fallbacks)
  models_compared       [str]  Provider-Labels (OpenAI, Mistral, ...)
  model_count           int
  model_ids             {provider: model-label}  konkrete Modellnamen, soweit
                                                 vom Client gemeldet (sanitized)
  excluded_count        int    vom Nutzer ausgeschlossene Antworten
  best_model            str    Provider-Label oder ""
  agreement             {score, level, model_count, major_contradictions,
                         minor_contradictions, emphases}
  claims                [{agree: int, dissent: int}]  nur Zähler, keine Texte
  differences           [{type, severity, position_count,
                          positions: [{models: [provider, ...]}, ...]}]
                          Positionen als Provider-Gruppen (Array von Maps —
                          Firestore verbietet direkt verschachtelte Arrays),
                          keine Stance-/Quote-Texte
  question_word_count   int    Länge der Frage (Zahl, nicht der Text)
  is_pro_user           bool
  used_own_keys         bool
  source                str    "app"
"""

import logging

from firebase_admin import firestore

from app.core.security import db_firestore
from app.services.share_snapshots import sanitize_model_labels

DIFFERENCES_STATS_SCHEMA_VERSION = 2
DIFFERENCES_STATS_COLLECTION = "differences_stats"


def build_differences_stats_doc(
    differences_data,
    *,
    consensus_model="",
    model_labels=None,
    excluded_count=0,
    is_pro_user=False,
    used_own_keys=False,
    question_word_count=0,
    source="app",
):
    """Baut das anonyme Statistik-Dokument aus differences_data.

    Gibt None zurück, wenn keine strukturierten Daten vorliegen. Nimmt
    ausschließlich Zähler/Metadaten auf — niemals Texte (siehe Modul-Docstring).
    """
    if not isinstance(differences_data, dict):
        return None

    models_compared = [
        str(m) for m in (differences_data.get("models_compared") or []) if m
    ]

    claims = []
    for claim in differences_data.get("claims") or []:
        if not isinstance(claim, dict):
            continue
        claims.append({
            "agree": len(claim.get("agree") or []),
            "dissent": len(claim.get("dissent") or []),
        })

    differences = []
    for diff in differences_data.get("differences") or []:
        if not isinstance(diff, dict):
            continue
        positions = [p for p in (diff.get("positions") or []) if isinstance(p, dict)]
        differences.append({
            "type": str(diff.get("type") or ""),
            "severity": str(diff.get("severity") or ""),
            "position_count": len(positions),
            # Array von Maps statt Array von Arrays: Firestore lehnt direkt
            # verschachtelte Arrays ab (400 "invalid nested entity").
            "positions": [
                {"models": sorted(str(m) for m in (p.get("models") or []) if m)}
                for p in positions
            ],
        })

    agreement = differences_data.get("agreement")
    if not isinstance(agreement, dict):
        agreement = {}

    # Judge-Metadaten (v2): welcher Provider/welches Modell die Analyse
    # tatsächlich geliefert hat — nur Metadaten, niemals Texte.
    judges = {}
    raw_judges = differences_data.get("judges")
    if isinstance(raw_judges, dict):
        for role in ("differences", "adjudicator"):
            entry = raw_judges.get(role)
            if isinstance(entry, dict) and entry.get("provider"):
                judges[role] = {
                    "provider": str(entry.get("provider") or "")[:40],
                    "model": str(entry.get("model") or "")[:80],
                    "tier": str(entry.get("tier") or "")[:20],
                }

    return {
        "schema_version": DIFFERENCES_STATS_SCHEMA_VERSION,
        "consensus_model": str(consensus_model or "")[:80],
        "judges": judges,
        "models_compared": models_compared,
        "model_count": len(models_compared),
        "model_ids": sanitize_model_labels(model_labels, models_compared),
        "excluded_count": int(excluded_count or 0),
        "best_model": str(differences_data.get("best_model") or "")[:40],
        "agreement": {
            "score": agreement.get("score"),
            "level": str(agreement.get("level") or ""),
            "model_count": agreement.get("model_count"),
            "major_contradictions": agreement.get("major_contradictions"),
            "minor_contradictions": agreement.get("minor_contradictions"),
            "emphases": agreement.get("emphases"),
        },
        "claims": claims,
        "differences": differences,
        "question_word_count": int(question_word_count or 0),
        "is_pro_user": bool(is_pro_user),
        "used_own_keys": bool(used_own_keys),
        "source": str(source or "app")[:20],
    }


def record_differences_stats(differences_data, *, db=None, **meta):
    """Persistiert das anonyme Statistik-Dokument (fire-and-forget).

    Fehler werden nur geloggt — die Telemetrie darf niemals einen
    Consensus-Lauf zum Scheitern bringen.
    """
    try:
        doc = build_differences_stats_doc(differences_data, **meta)
        if doc is None:
            return None
        db = db if db is not None else db_firestore
        if db is None:
            return None
        doc["created_at"] = firestore.SERVER_TIMESTAMP
        ref = db.collection(DIFFERENCES_STATS_COLLECTION).document()
        ref.set(doc)
        return ref.id
    except Exception:
        logging.warning("Recording differences stats failed", exc_info=True)
        return None
