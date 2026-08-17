"""Das nutzereigene Gedaechtnis: selbst geschriebener Kontext vor jedem Lauf.

Bewusst OHNE Ableitung aus Antworten. Was hier steht, hat der Nutzer selbst
getippt -- damit ist der Autor eindeutig, es kostet keinen LLM-Call, es kann
nichts halluzinieren und es braucht keine Review-Schleife. Ein aus Laeufen
destilliertes Profil waere eine andere Datenverarbeitung (Profilbildung) und
eine andere Rechtsgrundlage; das ist eine spaetere Stufe, nicht diese.

Drei Grenzen sind Absicht und keine Sparmassnahme:

1. **Hart gedeckelt.** Der Text geht ALLEN sechs Modellen identisch voran. Jeder
   gemeinsame Vorspann ist ein gemeinsamer Bias: je mehr davon, desto aehnlicher
   die Antworten, desto hoeher der Agreement-Score -- ohne dass die Modelle sich
   einiger waeren. Die vier Profilfelder bleiben deshalb kurz. Das separate
   Notizfeld ist die ausdrueckliche Langform fuer importierte Erinnerungen und
   wird ebenfalls auf ein dokumentiertes, technisch sicheres Maximum begrenzt.
2. **Form, nicht Inhalt.** Der Rahmen sagt den Modellen ausdruecklich, dass das
   Profil beeinflusst, WIE geantwortet wird, nie WAS wahr ist. Sonst uebernehmen
   sechs Modelle synchron eine Meinung des Nutzers und der Vergleich misst sie
   statt der Sache.
3. **Nur im interaktiven Lauf.** Injiziert wird ausschliesslich in ``handle_ask``.
   Watch-Reruns, Publisher- und Topic-Laeufe gehen direkt ueber ``engines.py``
   und sehen das Profil nie -- eine Watch-Baseline muss mit der Welt driften,
   nicht mit dem Profil ihres Besitzers.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

from app.core.observability import safe_exception
from app.services import persistence_guard


PROFILE_SCHEMA_VERSION = 2
MEMORY_COLLECTION = "memory"
PROFILE_DOCUMENT_ID = "profile"

# Reihenfolge ist die Lesereihenfolge im Prompt und in der UI.
SHORT_PROFILE_FIELDS = ("role", "focus", "style", "constraints")
NOTES_FIELD = "notes"
PROFILE_FIELDS = (*SHORT_PROFILE_FIELDS, NOTES_FIELD)

PROFILE_FIELD_LABELS = {
    "role": "Who they are",
    "focus": "What they work on",
    "style": "How they want answers written",
    "constraints": "Constraints that always apply",
}

MAX_FIELD_CHARS = 250
# Genug fuer eine vollstaendige exportierte ChatGPT-Erinnerungszusammenfassung,
# ohne das Firestore-Dokument oder jeden der sechs Provider-Prompts unbeschraenkt
# wachsen zu lassen. Anders als die vier Kurzfelder bleibt die Absatzstruktur
# erhalten: dieses Feld ist bewusst eine Notebox, keine abgeleitete Memory.
MAX_NOTES_CHARS = 12_000
# Gerenderter Inhalt ohne Rahmen: vier Kurzfelder plus Langnotiz und Labels.
MAX_PROFILE_CHARS = 13_200

# Marken, die einen Prompt-Rahmen schliessen. Ein Nutzer koennte sie sonst in ein
# Profilfeld tippen und damit den Chat-Kontext-Rahmen vorzeitig beenden. Das
# schadet nur seinem eigenen Lauf -- aber es waere ein stiller Defekt, der
# aussieht wie ein Modellfehler.
_FRAME_MARKER_RE = re.compile(
    r"(?:END\s+)?(?:AUTHORITATIVE\s+CHAT\s+CONTEXT|OF\s+USER\s+PROFILE|ABOUT\s+THE\s+USER)",
    re.IGNORECASE,
)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class UserMemoryError(Exception):
    pass


class UserMemoryUnavailable(UserMemoryError):
    pass


def empty_profile() -> dict:
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "enabled": True,
        **{field: "" for field in PROFILE_FIELDS},
    }


def _clean_field(value: object) -> str:
    """Ein Feld auf eine speicherbare, prompt-sichere Form bringen.

    Zeilenumbrueche bleiben erhalten (eine Randbedingung pro Zeile ist die
    natuerliche Schreibweise), alles andere wird eingeebnet.
    """
    if not isinstance(value, str):
        return ""
    text = _CONTROL_RE.sub(" ", value.replace("\r\n", "\n").replace("\r", "\n"))
    text = _FRAME_MARKER_RE.sub(" ", text)
    lines = [" ".join(line.split()) for line in text.split("\n")]
    text = "\n".join(line for line in lines if line).strip()
    if len(text) > MAX_FIELD_CHARS:
        text = text[:MAX_FIELD_CHARS].rstrip()
    return text


def _clean_notes(value: object) -> str:
    """Die grosse, manuell gepflegte Notiz prompt-sicher normalisieren.

    Absatz- und Listenstruktur bleibt erhalten. Es gibt weder Zusammenfassung
    noch LLM-Schritt; gekappt wird ausschliesslich an der dokumentierten Grenze.
    """
    if not isinstance(value, str):
        return ""
    text = _CONTROL_RE.sub(" ", value.replace("\r\n", "\n").replace("\r", "\n"))
    text = _FRAME_MARKER_RE.sub(" ", text).strip()
    if len(text) > MAX_NOTES_CHARS:
        text = text[:MAX_NOTES_CHARS].rstrip()
    return text


def sanitize_profile(value: object) -> dict:
    raw = value if isinstance(value, dict) else {}
    profile = empty_profile()
    # Kein Feld gesetzt heisst nicht "aus": ein leeres Profil rendert ohnehin
    # nichts. ``enabled`` ist die bewusste Pause bei gefuelltem Profil.
    profile["enabled"] = raw.get("enabled") is not False
    for field in SHORT_PROFILE_FIELDS:
        profile[field] = _clean_field(raw.get(field))
    profile[NOTES_FIELD] = _clean_notes(raw.get(NOTES_FIELD))
    return profile


def profile_is_empty(profile: dict) -> bool:
    return not any(str(profile.get(field) or "").strip() for field in PROFILE_FIELDS)


def render_profile(profile: object) -> str:
    """Der Profilblock fuer den System-Prompt -- oder "" , wenn es keinen gibt.

    "" ist der Normalfall fuer alle, die nichts eingetragen oder das Profil
    pausiert haben. Ein leerer Rahmen waere teurer Ballast in sechs Prompts.
    """
    clean = sanitize_profile(profile)
    if clean["enabled"] is not True or profile_is_empty(clean):
        return ""

    lines: list[str] = []
    used = 0
    for field in SHORT_PROFILE_FIELDS:
        text = clean[field]
        if not text:
            continue
        # Ein Feld ist im Prompt genau EINE Zeile. Gespeichert bleiben die
        # Umbrueche (eine Randbedingung pro Zeile liest sich im Formular
        # besser); im Prompt wuerde eine Folgezeile ohne Bindestrich die
        # Aufzaehlung zerreissen und wie ein neuer Abschnitt aussehen.
        inline = "; ".join(text.split("\n"))
        line = f"- {PROFILE_FIELD_LABELS[field]}: {inline}"
        if used + len(line) > MAX_PROFILE_CHARS:
            room = MAX_PROFILE_CHARS - used
            if room < 40:
                break
            line = line[:room].rstrip()
        lines.append(line)
        used += len(line) + 1
    notes = clean[NOTES_FIELD]
    if notes and used < MAX_PROFILE_CHARS:
        heading = "SAVED MEMORIES (a verbatim note the user maintains manually):"
        room = MAX_PROFILE_CHARS - used - len(heading) - 1
        if room >= 40:
            note_text = notes[:room].rstrip()
            lines.append(f"{heading}\n{note_text}")
            used += len(heading) + len(note_text) + 2
    if not lines:
        return ""

    body = "\n".join(lines)
    return (
        "ABOUT THE USER (a standing profile the user wrote in their own settings; "
        "it applies to every question they ask):\n"
        f"{body}\n"
        "Use it to shape how you answer: language, depth, framing, format and which "
        "trade-offs matter to this person. It never changes what is true. Where it "
        "conflicts with the question, with the evidence, or with your own assessment, "
        "the question and the evidence win -- say so plainly instead of bending the "
        "answer to the profile. Do not restate the profile and do not mention it "
        "unless the user asks about it.\n"
        "END OF USER PROFILE."
    )


def build_user_memory_system_prompt(base_prompt: str, memory_text: str) -> str:
    """Profil an die stehende Anweisung haengen, nicht davor.

    Der Chat-Kontext umschliesst spaeter das Ergebnis (Kontext zuerst, Anweisung
    zuletzt). Damit steht das Profil neben der Basisanweisung -- als stehende
    Praeferenz -- und nicht im Datenteil, wo es wie Gespraechsinhalt gelesen wuerde.
    """
    base = str(base_prompt or "").strip()
    memory = str(memory_text or "").strip()
    if not memory:
        return base
    if not base:
        return memory
    return f"{base}\n\n{memory}"


# Der Lauf wartet auf diesen Read. Firestores Default-Retry haelt einen Aufruf
# im Fehlerfall minutenlang; das Profil ist aber optional -- ohne es geht der
# Lauf ganz normal raus. Dieselbe Begruendung wie beim Modell-Config-Read im
# Startup (config.py: timeout=5.0, retry=None), nur enger, weil hier sechs
# parallele Requests eines einzigen Nutzerklicks daran haengen.
PROFILE_READ_TIMEOUT_SECONDS = 3.0


def _bounded_get(reference):
    """Einen Dokument-Read mit hartem Budget lesen.

    Test-Doubles und der Emulator-Pfad kennen die beiden Kwargs nicht; ein
    ``TypeError`` bedeutet hier nur "diese Referenz nimmt sie nicht entgegen"
    und niemals einen Lesefehler.
    """
    try:
        return reference.get(timeout=PROFILE_READ_TIMEOUT_SECONDS, retry=None)
    except TypeError:
        return reference.get()


class FirestoreUserMemoryRepository:
    def __init__(self, db, *, transaction_runner=None):
        self.db = db
        self._transaction_runner = transaction_runner

    def get(self, uid: str) -> dict:
        snapshot = _bounded_get(self._profile_ref(uid))
        if not snapshot.exists:
            return empty_profile()
        return sanitize_profile(snapshot.to_dict() or {})

    def save(self, uid: str, profile: object, *, now: datetime | None = None) -> dict:
        raw = dict(profile) if isinstance(profile, dict) else {}
        # Ein bereits offener Browser mit der v1-UI sendet ``notes`` gar nicht.
        # Das additive Feld darf dadurch nicht verschwinden. Die aktuelle UI
        # sendet beim bewussten Leeren den String "" und loescht es damit normal.
        preserve_existing_notes = (
            NOTES_FIELD not in raw or raw.get(NOTES_FIELD) is None
        )
        written = datetime.now(timezone.utc) if now is None else now
        profile_ref = self._profile_ref(uid)
        saved: dict[str, dict] = {}

        def operation(transaction):
            # Derselbe Zaun wie bei jedem anderen nutzergebundenen Write: ein
            # Profil darf nach der quittierten Kontoloeschung nicht neu entstehen.
            persistence_guard.ensure_account_write_allowed(
                uid=uid, db=self.db, transaction=transaction
            )
            if preserve_existing_notes:
                snapshot = profile_ref.get(transaction=transaction)
                previous = snapshot.to_dict() if snapshot.exists else {}
                raw[NOTES_FIELD] = (previous or {}).get(NOTES_FIELD, "")
            clean = sanitize_profile(raw)
            transaction.set(profile_ref, {**clean, "updated_at": written})
            saved["profile"] = clean

        self._transaction(operation)
        return saved["profile"]

    def delete(self, uid: str) -> None:
        self._profile_ref(uid).delete()

    def _transaction(self, operation):
        if self._transaction_runner is not None:
            return self._transaction_runner(operation)
        fake_runner = getattr(self.db, "run_transaction", None)
        if callable(fake_runner):
            return fake_runner(operation)
        from firebase_admin import firestore

        transaction = self.db.transaction(max_attempts=6)

        @firestore.transactional
        def run(tx):
            return operation(tx)

        return run(transaction)

    def _profile_ref(self, uid: str):
        uid = str(uid or "").strip()
        if not uid:
            raise UserMemoryError("uid must not be empty")
        return (
            self.db.collection("users")
            .document(uid)
            .collection(MEMORY_COLLECTION)
            .document(PROFILE_DOCUMENT_ID)
        )


def load_profile_text(repository: FirestoreUserMemoryRepository, uid: str) -> str:
    """Der Profilblock fuer einen Lauf -- fail-open.

    Ein nicht lesbares Profil darf einen Lauf nie aufhalten: der Nutzer hat eine
    Frage gestellt, nicht eine Einstellung geoeffnet. Der Lauf geht dann ohne
    Profil raus, so wie vor diesem Feature. Sichtbar bleibt es trotzdem, sonst
    faellt es still fuer alle aus, ohne dass irgendwo etwas kaputt aussieht.
    """
    if not uid:
        return ""
    try:
        return render_profile(repository.get(uid))
    except Exception as exc:
        logging.warning("user memory load failed category=%s", safe_exception(exc))
        return ""
