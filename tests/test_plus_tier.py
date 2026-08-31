"""Die Plus-Stufe: mehr Kontingent und die Komfortfunktionen, aber exakt die
Free-Modellauswahl.

Der Kern ist die Kostengrenze. Plus existiert, damit Tester Anhaenge und
Resolve ausprobieren koennen, ohne einen Frontier-Lauf ausloesen zu koennen --
deshalb pruefen diese Tests vor allem, was Plus NICHT darf.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

import app.core.config as cfg
import app.core.security as security
from app.core.entitlements import (
    TIER_FREE,
    TIER_PLUS,
    TIER_PRO,
    entitlements_for,
    normalize_tier,
    tier_at_least,
)
from app.services.llm.attachments import parse_attachments
from app.services.llm.base import validate_model


# --- Normalisierung --------------------------------------------------------

@pytest.mark.parametrize(
    "value,expected",
    [
        ("plus", TIER_PLUS),
        ("Plus", TIER_PLUS),
        ("  PLUS ", TIER_PLUS),
        ("pro", TIER_PRO),
        # "premium" ist der aelteste Pro-Tag in Firestore und muss Pro bleiben.
        ("premium", TIER_PRO),
        ("free", TIER_FREE),
        ("", TIER_FREE),
        (None, TIER_FREE),
        ("early", TIER_FREE),
        ("plus ", TIER_PLUS),
        # Bool-Kompatibilitaet: die Limit-Getter nahmen jahrelang is_pro.
        (True, TIER_PRO),
        (False, TIER_FREE),
    ],
)
def test_normalize_tier(value, expected):
    assert normalize_tier(value) == expected


def test_unknown_tier_never_becomes_plus_or_pro():
    for value in ("plus-trial", "PRO+", "plus_beta", 3, object()):
        assert normalize_tier(value) == TIER_FREE


def test_tier_ordering():
    assert tier_at_least(TIER_PLUS, TIER_PLUS)
    assert tier_at_least(TIER_PRO, TIER_PLUS)
    assert not tier_at_least(TIER_FREE, TIER_PLUS)
    assert not tier_at_least(TIER_PLUS, TIER_PRO)


# --- Entitlements ----------------------------------------------------------

def test_plus_gets_the_features_but_not_the_expensive_models():
    plus = entitlements_for(TIER_PLUS)
    assert plus.attachments is True
    assert plus.resolve is True
    # Die eigentliche Kostengrenze.
    assert plus.is_pro is False
    assert plus.premium_models is False
    assert plus.deep_think is False


def test_free_gets_nothing_and_pro_gets_everything():
    free = entitlements_for(TIER_FREE)
    assert not any([free.attachments, free.resolve, free.deep_think, free.premium_models])
    pro = entitlements_for(TIER_PRO)
    assert all([pro.attachments, pro.resolve, pro.deep_think, pro.premium_models])


# --- Limits ----------------------------------------------------------------

def test_plus_has_the_largest_run_quota():
    # Plus faehrt die guenstigen Modelle und darf deshalb mehr laufen als Pro.
    assert cfg.get_consensus_run_limit(TIER_PLUS) > cfg.get_consensus_run_limit(TIER_PRO)
    assert cfg.get_consensus_run_limit(TIER_PLUS) > cfg.get_consensus_run_limit(TIER_FREE)


def test_plus_has_no_deep_think_quota():
    assert cfg.get_deep_think_run_limit(TIER_PLUS) == cfg.get_deep_think_run_limit(TIER_FREE)
    assert cfg.get_deep_think_run_limit(TIER_PLUS) == 0


def test_plus_deep_search_limits_fall_back_to_free():
    assert cfg.get_word_limit(TIER_PLUS, True) == cfg.get_word_limit(TIER_FREE, True)
    assert cfg.get_output_token_limit(TIER_PLUS, True) == cfg.get_output_token_limit(TIER_FREE, True)


def test_plus_sits_between_free_and_pro_for_memory():
    assert (
        cfg.get_memory_char_limit(TIER_FREE)
        <= cfg.get_memory_char_limit(TIER_PLUS)
        <= cfg.get_memory_char_limit(TIER_PRO)
    )


def test_plus_watch_limit_sits_between_free_and_pro():
    assert (
        cfg.get_watch_active_limit(TIER_FREE)
        <= cfg.get_watch_active_limit(TIER_PLUS)
        <= cfg.get_watch_active_limit(TIER_PRO)
    )


def test_plus_watches_run_on_the_free_models():
    # Mehr Watches, gleiche Kosten pro Lauf.
    assert cfg.get_watch_models(TIER_PLUS) == cfg.get_watch_models(TIER_FREE)
    assert cfg.get_watch_consensus_model(TIER_PLUS) == cfg.get_watch_consensus_model(TIER_FREE)


def test_plus_daily_watch_interval_follows_the_admin_switch():
    limits = dict(cfg.LIMITS)
    try:
        cfg.LIMITS["watch_daily_interval_requires_pro"] = 1
        cfg.LIMITS["watch_plus_daily_interval_allowed"] = 1
        assert cfg.is_watch_daily_allowed(TIER_PLUS) is True
        assert cfg.is_watch_daily_allowed(TIER_FREE) is False

        cfg.LIMITS["watch_plus_daily_interval_allowed"] = 0
        assert cfg.is_watch_daily_allowed(TIER_PLUS) is False
        # Pro bleibt davon unberuehrt.
        assert cfg.is_watch_daily_allowed(TIER_PRO) is True
    finally:
        cfg.LIMITS.clear()
        cfg.LIMITS.update(limits)


def test_a_plus_limit_missing_from_the_config_falls_back_to_free_not_pro():
    limits = dict(cfg.LIMITS)
    try:
        del cfg.LIMITS["plus_consensus_run_limit"]
        assert cfg.get_consensus_run_limit(TIER_PLUS) == cfg.get_consensus_run_limit(TIER_FREE)
    finally:
        cfg.LIMITS.clear()
        cfg.LIMITS.update(limits)


def test_plus_limits_are_admin_configurable():
    normalized = cfg.normalize_limits_config({"plus_consensus_run_limit": 1234})
    assert normalized["plus_consensus_run_limit"] == 1234
    # Und sie stehen im Admin-GET, sonst waere der Wert nicht bedienbar.
    assert "plus_consensus_run_limit" in cfg.get_limits_config()
    assert "memory_plus_chars" in cfg.get_memory_edit_config()


def test_memory_plus_chars_is_clamped_between_free_and_pro():
    config = cfg.normalize_memory_edit_config({
        "memory_free_chars": 5_000,
        "memory_plus_chars": 24_000,
        "memory_pro_chars": 10_000,
    })
    assert config["memory_free_chars"] <= config["memory_plus_chars"] <= config["memory_pro_chars"]


# --- Modell- und Feature-Gates ---------------------------------------------

def test_plus_cannot_pick_a_premium_model():
    premium = next(iter(cfg.PREMIUM_MODELS))
    config = cfg.get_model_config(premium)
    provider = config.provider if config else "openai"
    allowed = set(cfg.PROVIDERS[provider].models) | {premium}
    with pytest.raises(HTTPException) as excinfo:
        validate_model(
            premium, allowed, provider,
            is_pro=entitlements_for(TIER_PLUS).premium_models,
        )
    assert excinfo.value.status_code == 403


def test_attachments_open_at_plus():
    data = {"attachments": [{"name": "a.txt", "mime": "text/plain", "data": "aGk="}]}
    with pytest.raises(HTTPException) as excinfo:
        parse_attachments(data, attachments_allowed=entitlements_for(TIER_FREE).attachments)
    assert excinfo.value.status_code == 403
    # Plus kommt durch das Gate (der Inhalt wird danach normal validiert).
    assert parse_attachments(data, attachments_allowed=entitlements_for(TIER_PLUS).attachments)


# --- Firestore-Lookup ------------------------------------------------------

def make_firestore_mock(data):
    doc = MagicMock()
    doc.exists = True
    doc.to_dict.return_value = data
    db = MagicMock()
    db.collection.return_value.document.return_value.get.return_value = doc
    return db


@pytest.mark.parametrize(
    "stored,tier,is_pro,is_plus",
    [
        ({"tier": "plus"}, TIER_PLUS, False, True),
        ({"tier": "Plus"}, TIER_PLUS, False, True),
        ({"tier": "pro"}, TIER_PRO, True, True),
        ({"tier": "premium"}, TIER_PRO, True, True),
        ({"tier": "free"}, TIER_FREE, False, False),
        ({}, TIER_FREE, False, False),
    ],
)
def test_tier_lookup_derives_all_three_flags(stored, tier, is_pro, is_plus):
    security.invalidate_tier_cache("uid-tier")
    db = make_firestore_mock(stored)
    with patch.object(security, "db_firestore", db):
        assert security.get_user_tier("uid-tier") == tier
        assert security.is_user_pro("uid-tier") is is_pro
        assert security.is_user_plus("uid-tier") is is_plus
    security.invalidate_tier_cache("uid-tier")
