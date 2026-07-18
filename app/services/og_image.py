"""Server-rendered Open-Graph-Karten (1200x630 PNG) für Share-Seiten.

Statt des Favicons zeigt der Link-Preview das "Scoreboard" der Seite: Frage,
Agreement-Score, Modell-/Widerspruchszahl und (bei Watch-Seiten) die
Score-Sparkline. Rein aus Snapshot-Daten gerendert – keine LLM-Calls.
Monochromes Design analog zur App; Teal nur für die Sparkline.
"""

from __future__ import annotations

import io
import logging
import os

from cachetools import TTLCache

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except Exception:  # pragma: no cover - Pillow fehlt nur in Alt-Umgebungen
    _PIL_OK = False

WIDTH, HEIGHT = 1200, 630
MARGIN = 72
FONT_PATH = os.path.join("static", "fonts", "Inter-Variable.ttf")

BG = (23, 23, 23)
TEXT = (245, 245, 244)
MUTED = (163, 163, 163)
FAINT = (82, 82, 82)
ACCENT = (45, 212, 191)  # Teal – einzige Nicht-Grau-Farbe

_cache = TTLCache(maxsize=256, ttl=1800)


def is_available() -> bool:
    return _PIL_OK and os.path.isfile(FONT_PATH)


def _font(size: int, weight: int = 600):
    font = ImageFont.truetype(FONT_PATH, size)
    try:
        # Inter ist eine Variable Font mit Achsen (opsz, wght).
        font.set_variation_by_axes([size, weight])
    except Exception:
        pass
    return font


def _wrap(draw, text: str, font, max_width: int, max_lines: int) -> list[str]:
    words = str(text or "").split()
    lines, current = [], ""
    for word in words:
        candidate = (current + " " + word).strip()
        if draw.textlength(candidate, font=font) <= max_width or not current:
            current = candidate
            continue
        lines.append(current)
        current = word
        if len(lines) == max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    if len(lines) == max_lines and len(" ".join(lines)) < len(" ".join(words)):
        lines[-1] = lines[-1].rstrip(".,;:") + "…"
    return lines


def _draw_sparkline(draw, scores: list[int], box: tuple[int, int, int, int]):
    left, top, right, bottom = box
    if len(scores) < 2:
        return
    # Auf den tatsächlichen Score-Bereich (min. 20 Punkte Spanne) skalieren,
    # sonst wirkt jede Bewegung auf der 0-100-Achse flach.
    low, high = min(scores), max(scores)
    center = (low + high) / 2
    span_scores = max(20, high - low + 10)
    lo = max(0, min(100 - span_scores, center - span_scores / 2))
    step = (right - left) / (len(scores) - 1)
    span = bottom - top
    points = [
        (left + index * step, bottom - span * (score - lo) / span_scores)
        for index, score in enumerate(scores)
    ]
    draw.line([(left, bottom), (right, bottom)], fill=FAINT, width=2)
    draw.line(points, fill=ACCENT, width=5, joint="curve")
    x, y = points[-1]
    draw.ellipse([x - 8, y - 8, x + 8, y + 8], fill=ACCENT)


def render_share_card(*, question: str, score, model_count: int,
                      contradiction_count: int, history_scores: list[int],
                      checked_label: str = "") -> bytes:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)

    brand_font = _font(30, 650)
    draw.text((MARGIN, 56), "consens.io", font=brand_font, fill=TEXT)
    tagline = "AI CONSENSUS CHECK"
    tagline_font = _font(22, 550)
    draw.text(
        (WIDTH - MARGIN - draw.textlength(tagline, font=tagline_font), 62),
        tagline, font=tagline_font, fill=MUTED,
    )
    draw.line([(MARGIN, 112), (WIDTH - MARGIN, 112)], fill=FAINT, width=2)

    question_font = _font(56, 640)
    lines = _wrap(draw, question, question_font, WIDTH - 2 * MARGIN, 4)
    y = 156
    for line in lines:
        draw.text((MARGIN, y), line, font=question_font, fill=TEXT)
        y += 72

    # Untere Statistik-Zeile: Score groß links, Fakten + Sparkline rechts.
    base = HEIGHT - 150
    if isinstance(score, (int, float)):
        score_font = _font(96, 700)
        score_text = str(int(score))
        draw.text((MARGIN, base - 24), score_text, font=score_font, fill=TEXT)
        suffix_font = _font(34, 550)
        suffix_x = MARGIN + draw.textlength(score_text, font=score_font) + 14
        draw.text((suffix_x, base + 26), "/100 agreement", font=suffix_font, fill=MUTED)
        facts_x = suffix_x + draw.textlength("/100 agreement", font=suffix_font) + 56
    else:
        facts_x = MARGIN

    scores = [int(s) for s in (history_scores or []) if isinstance(s, (int, float))]
    has_spark = len(scores) >= 2
    spark_w = 240
    spark_left = WIDTH - MARGIN - spark_w

    facts_font = _font(30, 550)
    facts = [f"{model_count} AI models" if model_count else "Multiple AI models"]
    if contradiction_count:
        facts.append(f"{contradiction_count} contradiction{'s' if contradiction_count != 1 else ''}")
    if checked_label:
        facts.append(checked_label)
    # Fakten dürfen nie unter die Sparkline laufen: hinten kürzen, bis es passt.
    facts_max = (spark_left - 48 if has_spark else WIDTH - MARGIN) - facts_x
    while len(facts) > 1 and draw.textlength("  ·  ".join(facts), font=facts_font) > facts_max:
        facts.pop()
    draw.text((facts_x, base + 2), "  ·  ".join(facts), font=facts_font, fill=MUTED)

    if has_spark:
        _draw_sparkline(
            draw, scores[-16:],
            (spark_left, base - 30, WIDTH - MARGIN, base + 50),
        )

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def share_card_png(share_id: str, *, question: str, score, model_count: int,
                   contradiction_count: int, history_scores: list[int],
                   checked_label: str = "") -> bytes | None:
    """Gecachte Karte; None wenn Rendering nicht verfügbar/fehlgeschlagen."""
    if not is_available():
        return None
    key = (share_id, int(score) if isinstance(score, (int, float)) else -1,
           len(history_scores or []), checked_label)
    cached = _cache.get(key)
    if cached is not None:
        return cached
    try:
        png = render_share_card(
            question=question, score=score, model_count=model_count,
            contradiction_count=contradiction_count,
            history_scores=history_scores, checked_label=checked_label,
        )
    except Exception:
        logging.exception("og_image rendering failed for %s", share_id)
        return None
    _cache[key] = png
    return png
