"""Privacy-preserving favicon proxy for public source embeds.

Public Topic pages render self-hosted source cards. To give each card a real
site favicon *without* the visitor's browser ever talking to a third party, we
fetch the icon server-side once and cache it in memory (a daily Render restart
is an acceptable cache lifetime, consistent with the rest of the app). The
browser only ever requests ``/api/topics/favicon`` from consens.io.

We resolve icons through Google's public S2 favicon service. That call happens
server-to-server, so no visitor data is exposed; unknown domains still yield a
neutral globe glyph, which the card layer treats as "no icon" and falls back to
a self-hosted monogram.
"""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import requests

_HOST_RE = re.compile(
    r"^(?=.{1,253}$)([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$"
)
_SERVICE = "https://www.google.com/s2/favicons"
_TIMEOUT = 2
_MAX_BYTES = 100_000
_OK_TTL = 60 * 60 * 24 * 7      # 7 days for a resolved icon
_MISS_TTL = 60 * 60 * 24       # negative cache prevents retry amplification
_MAX_ENTRIES = 2000

_CACHE: OrderedDict[str, tuple[float, bytes | None, str]] = OrderedDict()
_INFLIGHT: dict[str, threading.Event] = {}
_LOCK = threading.Lock()
EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="favicon-proxy")


def normalize_host(value) -> str:
    """Reduce arbitrary input to a bare, validated registrable host or ""."""
    host = str(value or "").strip().lower()
    host = re.sub(r"^https?://", "", host).split("/")[0].split("?")[0]
    host = host.split(":")[0].strip(".")
    host = re.sub(r"^www\.", "", host)
    return host if _HOST_RE.match(host) else ""


def _fetch(host: str) -> tuple[bytes | None, str]:
    try:
        resp = requests.get(
            _SERVICE,
            params={"domain": host, "sz": "64"},
            timeout=_TIMEOUT,
            headers={"User-Agent": "consens.io-favicon/1.0"},
        )
    except requests.RequestException:
        return None, ""
    if resp.status_code != 200:
        return None, ""
    content_type = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
    content = resp.content or b""
    if not content_type.startswith("image/") or not content or len(content) > _MAX_BYTES:
        return None, ""
    return content, content_type


def get_favicon(value) -> tuple[bytes | None, str]:
    """Return (bytes, content_type) for a host, cached; (None, "") if unknown."""
    host = normalize_host(value)
    if not host:
        return None, ""
    now = time.time()
    with _LOCK:
        cached = _CACHE.get(host)
        if cached and cached[0] > now:
            _CACHE.move_to_end(host)
            return cached[1], cached[2]
        if cached:
            _CACHE.pop(host, None)
        pending = _INFLIGHT.get(host)
        if pending is None:
            pending = threading.Event()
            _INFLIGHT[host] = pending
            leader = True
        else:
            leader = False
    if not leader:
        pending.wait(timeout=_TIMEOUT + 0.5)
        with _LOCK:
            cached = _CACHE.get(host)
            if cached and cached[0] > time.time():
                _CACHE.move_to_end(host)
                return cached[1], cached[2]
        return None, ""
    data, content_type = _fetch(host)
    ttl = _OK_TTL if data else _MISS_TTL
    with _LOCK:
        _CACHE[host] = (now + ttl, data, content_type)
        _CACHE.move_to_end(host)
        while len(_CACHE) > _MAX_ENTRIES:
            _CACHE.popitem(last=False)
        _INFLIGHT.pop(host, None)
        pending.set()
    return data, content_type
