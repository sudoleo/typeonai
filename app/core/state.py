# In-Memory-Zustand — bewusst nicht persistent: der taegliche Render-Restart
# setzt alle Zaehler zurueck (siehe docs/codebase-map.md §4/§8).
import threading

usage_counter = {}  # { uid: anzahl_anfragen }
deep_search_usage = {}  # { uid: anzahl_deep_search_anfragen }
last_feedback_time = {} # { uid: timestamp }

# Die /ask_*-Endpoints sind sync-def und laufen beim Frage-Fan-out parallel in
# Threadpool-Workern. Ohne Lock lesen mehrere Worker denselben Zaehlerstand und
# ueberschreiben sich gegenseitig (Unterzaehlung = faktische Limit-Umgehung).
_usage_lock = threading.Lock()


def get_usage_snapshot(uid) -> tuple:
    """Konsistentes (usage, deep_usage)-Paar fuer Read-only-Stellen
    (/usage, /prepare, Remaining-Felder in Responses)."""
    with _usage_lock:
        return usage_counter.get(uid, 0), deep_search_usage.get(uid, 0)


def check_and_increment_usage(
    uid,
    *,
    limit_regular,
    limit_deep,
    increment=1.0,
    deep_search: bool = False,
) -> tuple:
    """Atomarer Limit-Check + Inkrement fuer /ask_*, /consensus und /resolve.

    Returns (status, usage, deep_usage):
      status: "ok" | "limit_regular" | "limit_deep"
      usage/deep_usage: Zaehlerstaende nach dem Inkrement (bei "ok") bzw.
      unveraendert zum Zeitpunkt der Ablehnung (fuer Fehler-Details).
    """
    with _usage_lock:
        current_usage = usage_counter.get(uid, 0)
        current_deep_usage = deep_search_usage.get(uid, 0)

        if current_usage + increment > limit_regular:
            return "limit_regular", current_usage, current_deep_usage
        if deep_search and current_deep_usage + increment > limit_deep:
            return "limit_deep", current_usage, current_deep_usage

        usage_counter[uid] = current_usage + increment
        if deep_search:
            deep_search_usage[uid] = current_deep_usage + increment
        return "ok", usage_counter[uid], deep_search_usage.get(uid, 0)


def reset_usage(uid) -> None:
    """Entfernt die Usage-Zaehler eines Nutzers (z.B. bei /delete_account)."""
    with _usage_lock:
        usage_counter.pop(uid, None)
        deep_search_usage.pop(uid, None)
