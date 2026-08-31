"""Die im UI sichtbare Build-Kennung.

Vorher stand dort eine handgepflegte Versionsnummer ("v1.11.1"), die niemand
aktualisiert hat und die deshalb nur noch behauptet hat, aktuell zu sein.
Jetzt zeigt die Fusszeile den Commit, der tatsaechlich laeuft.

Aufloesungsreihenfolge (erste nicht-leere Quelle gewinnt):

1. Die Umgebungsvariablen der Hosts. Render setzt ``RENDER_GIT_COMMIT``;
   die uebrigen Namen decken die ueblichen Build-Systeme ab.
2. Das ``.git``-Verzeichnis des Checkouts (lokale Entwicklung). Bewusst als
   Dateilesung statt als ``git``-Subprozess: der Aufruf haengt sonst am
   Startpfad und an einer installierten Binary, und ein Prozess-Spawn pro
   Deploy-Frage ist die Sache nicht wert.

Ohne beides bleibt der Wert leer; das Template blendet die Zeile dann aus,
statt eine Unwahrheit anzuzeigen.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

_ENV_KEYS = (
    "RENDER_GIT_COMMIT",
    "GIT_COMMIT",
    "SOURCE_VERSION",
    "COMMIT_SHA",
    "VERCEL_GIT_COMMIT_SHA",
)

_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")

# app/core/version.py -> app/core -> app -> Projektwurzel
_REPO_ROOT = Path(__file__).resolve().parents[2]

SHORT_SHA_LENGTH = 7


def _clean(value: str | None) -> str:
    value = (value or "").strip().lower()
    return value if _SHA_RE.match(value) else ""


def _from_env() -> str:
    for key in _ENV_KEYS:
        sha = _clean(os.environ.get(key))
        if sha:
            return sha
    return ""


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _from_git_dir() -> str:
    git_dir = _REPO_ROOT / ".git"
    # Worktrees und Submodule schreiben statt eines Verzeichnisses eine Datei
    # mit "gitdir: <pfad>". Ein Schritt reicht, tiefer wird es nicht.
    if git_dir.is_file():
        pointer = _read(git_dir)
        if not pointer.startswith("gitdir:"):
            return ""
        git_dir = Path(pointer.split(":", 1)[1].strip())
        if not git_dir.is_absolute():
            git_dir = (_REPO_ROOT / git_dir).resolve()

    head = _read(git_dir / "HEAD")
    if not head:
        return ""
    if not head.startswith("ref:"):
        return _clean(head)  # detached HEAD

    ref = head.split(":", 1)[1].strip()
    sha = _clean(_read(git_dir / ref))
    if sha:
        return sha

    # Frisch geklonte Checkouts haben lose Refs gepackt.
    for line in _read(git_dir / "packed-refs").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == ref:
            return _clean(parts[0])
    return ""


@lru_cache(maxsize=1)
def get_commit_sha() -> str:
    """Voller Commit-Hash des laufenden Builds, oder "" wenn unbekannt."""
    return _from_env() or _from_git_dir()


@lru_cache(maxsize=1)
def get_commit_short() -> str:
    """Der Hash in der Laenge, die im UI steht (leer, wenn unbekannt)."""
    return get_commit_sha()[:SHORT_SHA_LENGTH]


# Die Fusszeile verlinkt den laufenden Commit auf das oeffentliche Repository.
REPO_URL = "https://github.com/sudoleo/typeonai"
