import unittest
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.core.config as cfg
from app.core.rate_limit import limiter
from app.api.routers import share as share_router
from app.api.routers import admin as admin_router
from app.services import share_snapshots as snapshots
from app.services.share_snapshots import ShareError
from app.services.public_markdown import render_public_markdown, markdown_to_plaintext


# --- Minimaler Firestore-Fake (nur was share_snapshots braucht) ---

class FakeSnap:
    def __init__(self, data):
        self._data = data

    @property
    def exists(self):
        return self._data is not None

    def to_dict(self):
        return dict(self._data) if self._data is not None else None


class FakeDocRef:
    def __init__(self, store, key):
        self._store = store
        self._key = key

    @property
    def id(self):
        return self._key

    def get(self, transaction=None):
        return FakeSnap(self._store.get(self._key))

    def set(self, data):
        self._store[self._key] = dict(data)

    def update(self, data):
        self._store[self._key].update(data)

    def delete(self):
        self._store.pop(self._key, None)


class FakeQueryDoc:
    def __init__(self, store, key):
        self.id = key
        self._store = store
        self.reference = FakeDocRef(store, key)

    def to_dict(self):
        return dict(self._store[self.id])


class FakeQuery:
    def __init__(self, store, field, op, value, max_items=None):
        self._store = store
        self._field = field
        self._op = op
        self._value = value
        self._max_items = max_items

    def limit(self, n):
        return FakeQuery(self._store, self._field, self._op, self._value, n)

    def _matches(self, data):
        if self._field is None:
            return True
        actual = data.get(self._field)
        if self._op == "<":
            try:
                return actual is not None and actual < self._value
            except TypeError:
                return False
        return actual == self._value

    def stream(self):
        emitted = 0
        for key, data in list(self._store.items()):
            if not self._matches(data):
                continue
            yield FakeQueryDoc(self._store, key)
            emitted += 1
            if self._max_items is not None and emitted >= self._max_items:
                break


class FakeCollection:
    def __init__(self, store):
        self._store = store

    def document(self, doc_id):
        return FakeDocRef(self._store, doc_id)

    def where(self, field, op, value):
        return FakeQuery(self._store, field, op, value)

    def limit(self, n):
        return FakeQuery(self._store, None, None, None, n)


class FakeDb:
    def __init__(self):
        self.stores = defaultdict(dict)

    def collection(self, name):
        return FakeCollection(self.stores[name])


def make_pending(uid="user-1", **overrides):
    pending = {
        "schema_version": 1,
        "owner_uid": uid,
        "question": "Wie funktioniert Photosynthese in Pflanzen?",
        "consensus_md": "Photosynthese wandelt Licht in Energie um. [S1]",
        "differences_data": {"claims": [], "differences": [], "best_model": "", "models_compared": []},
        "differences_text": "No differences found.",
        "sources": [{"id": "S1", "title": "Quelle", "url": "https://example.org/a", "provider": "web"}],
        "included_models": ["OpenAI: gpt-test", "Google Gemini"],
        "consensus_model": "Anthropic",
        "answered_at": "2026-06-11T10:00:00+00:00",
        "expires_at": datetime.now(timezone.utc) + timedelta(hours=12),
    }
    pending.update(overrides)
    return pending


class SlugAndIdTests(unittest.TestCase):
    def test_slugify_handles_umlauts_and_punctuation(self):
        self.assertEqual(
            snapshots.slugify_question("Wie funktioniert Photosynthese?"),
            "wie-funktioniert-photosynthese",
        )
        self.assertEqual(snapshots.slugify_question("Größe & Ökonomie!"), "groesse-oekonomie")

    def test_slugify_truncates_at_word_boundary(self):
        slug = snapshots.slugify_question("wort " * 40)
        self.assertLessEqual(len(slug), snapshots.MAX_SLUG_CHARS)
        self.assertFalse(slug.endswith("-"))

    def test_slugify_fallback_for_empty_input(self):
        self.assertEqual(snapshots.slugify_question("???"), "consensus")
        self.assertEqual(snapshots.slugify_question(None), "consensus")

    def test_generate_share_id_charset_and_uniqueness(self):
        ids = {snapshots.generate_share_id() for _ in range(500)}
        self.assertEqual(len(ids), 500)
        for share_id in ids:
            self.assertTrue(snapshots.is_valid_share_id(share_id))
            self.assertNotIn("-", share_id)

    def test_is_valid_share_id_rejects_bad_input(self):
        self.assertFalse(snapshots.is_valid_share_id("short"))
        self.assertFalse(snapshots.is_valid_share_id("x" * 17))
        self.assertFalse(snapshots.is_valid_share_id("abc_def-ghi*jkl0"))
        self.assertFalse(snapshots.is_valid_share_id(None))

    def test_split_slug_id(self):
        self.assertEqual(snapshots.split_slug_id("mein-slug-abc123"), ("mein-slug", "abc123"))
        self.assertEqual(snapshots.split_slug_id("abc123"), ("", "abc123"))

    def test_question_hash_normalizes(self):
        self.assertEqual(
            snapshots.question_hash("  Was ist KI? "),
            snapshots.question_hash("was ist ki"),
        )


class SanitizerTests(unittest.TestCase):
    def test_sources_drop_non_http_and_dedupe(self):
        result = snapshots.sanitize_sources({
            "OpenAI": [
                {"id": "S2", "title": "B", "url": "https://example.org/b"},
                {"id": "S9", "title": "Evil", "url": "javascript:alert(1)"},
                {"id": "S4", "title": "Dup", "url": "https://example.org/b/"},
            ],
            "Gemini": [{"id": "s1", "title": "A", "url": "https://example.org/a"}],
        })
        self.assertEqual([s["id"] for s in result], ["S1", "S2"])
        self.assertTrue(all(s["url"].startswith("https://") for s in result))

    def test_sources_cap(self):
        many = [{"id": "S%d" % i, "url": "https://example.org/%d" % i} for i in range(80)]
        self.assertEqual(len(snapshots.sanitize_sources(many)), snapshots.MAX_SOURCES)

    def test_differences_whitelist(self):
        data = {
            "claims": [{"anchor": "x" * 800, "agree": ["OpenAI", 42], "dissent": [{"model": "Grok", "quote": "q", "secret": "drop"}]}],
            "differences": [
                {"claim": "c", "type": "contradiction", "severity": "major", "positions": [{"stance": "s", "models": ["Grok"], "quote": "q"}], "verify": "v"},
                {"claim": "no positions", "positions": []},
            ],
            "best_model": "OpenAI",
            "models_compared": ["OpenAI", "Grok"],
            "agreement": {"score": 64, "level": "partially", "model_count": 2,
                          "major_contradictions": 1, "minor_contradictions": 0,
                          "emphases": 0, "internal_debug": "drop"},
            "internal_cost": 1.23,
        }
        result = snapshots.sanitize_differences_data(data)
        self.assertNotIn("internal_cost", result)
        self.assertEqual(len(result["claims"][0]["anchor"]), 500)
        self.assertEqual(result["claims"][0]["agree"], ["OpenAI"])
        self.assertNotIn("secret", result["claims"][0]["dissent"][0])
        self.assertEqual(len(result["differences"]), 1)
        self.assertEqual(result["differences"][0]["severity"], "major")
        self.assertEqual(result["agreement"], {
            "score": 64, "level": "partially", "model_count": 2,
            "major_contradictions": 1, "minor_contradictions": 0, "emphases": 0,
        })
        self.assertIsNone(snapshots.sanitize_differences_data("not a dict"))

    def test_judges_are_whitelisted(self):
        result = snapshots.sanitize_differences_data({
            "claims": [], "differences": [], "best_model": "", "models_compared": [],
            "judges": {
                "differences": {"provider": "Gemini", "model": "gemini-x", "tier": "pro", "cost": "drop"},
                "adjudicator": "not a dict",
                "unknown_role": {"provider": "X"},
            },
        })
        self.assertEqual(result["judges"], {
            "differences": {"provider": "Gemini", "model": "gemini-x", "tier": "pro"},
        })
        # Ohne verwertbare Judge-Einträge fehlt das Feld komplett.
        result = snapshots.sanitize_differences_data({
            "claims": [], "differences": [], "best_model": "", "models_compared": [],
            "judges": {"differences": {"model": "no provider"}},
        })
        self.assertNotIn("judges", result)

    def test_resolution_is_whitelisted(self):
        diff = {
            "claim": "c", "type": "contradiction", "severity": "major",
            "positions": [{"stance": "s", "models": ["Grok"], "quote": "q"}],
            "resolution": {
                "outcome": "resolved",
                "results": [
                    {"model": "Grok", "decision": "revise", "position": "p" * 900,
                     "reason": "r", "internal": "drop"},
                    {"model": "OpenAI", "decision": "shrug", "position": "p", "reason": "r"},
                    "not a dict",
                ],
                "extra": "drop",
            },
        }
        result = snapshots.sanitize_differences_data(
            {"claims": [], "differences": [diff], "best_model": "", "models_compared": []}
        )
        resolution = result["differences"][0]["resolution"]
        self.assertEqual(resolution["outcome"], "resolved")
        self.assertNotIn("extra", resolution)
        self.assertEqual(len(resolution["results"]), 2)
        self.assertEqual(len(resolution["results"][0]["position"]), 500)
        self.assertNotIn("internal", resolution["results"][0])
        # Unbekannte Decision wird konservativ zu "error".
        self.assertEqual(resolution["results"][1]["decision"], "error")

    def test_invalid_resolution_is_dropped(self):
        base = {
            "claim": "c", "type": "contradiction", "severity": "major",
            "positions": [{"stance": "s", "models": ["Grok"], "quote": "q"}],
        }
        for bad in ({"outcome": "nonsense", "results": [{"model": "Grok", "decision": "revise"}]},
                    {"outcome": "resolved", "results": []},
                    "not a dict", None):
            diff = dict(base, resolution=bad)
            result = snapshots.sanitize_differences_data(
                {"claims": [], "differences": [diff], "best_model": "", "models_compared": []}
            )
            self.assertNotIn("resolution", result["differences"][0])

    def test_agreement_score_is_clamped(self):
        result = snapshots.sanitize_differences_data({
            "claims": [], "differences": [], "best_model": "",
            "models_compared": [],
            "agreement": {"score": 999, "level": "x" * 80, "model_count": -3,
                          "major_contradictions": "NaN"},
        })
        self.assertEqual(result["agreement"]["score"], 100)
        self.assertEqual(result["agreement"]["model_count"], 0)
        self.assertEqual(result["agreement"]["major_contradictions"], 0)
        self.assertEqual(len(result["agreement"]["level"]), 20)

    def test_included_models_order_and_labels(self):
        result = snapshots.build_included_models(
            ["Grok", "OpenAI", "Anthropic"],
            {"OpenAI": "gpt-test", "Grok": "", "Unknown": "x", "Anthropic": 42},
        )
        self.assertEqual(result, ["OpenAI: gpt-test", "Anthropic Claude", "Grok"])

    def test_model_labels_only_for_included_providers(self):
        labels = snapshots.sanitize_model_labels(
            {"OpenAI": "gpt-test", "Gemini": "gemini-test", "Mistral": "mistral-test",
             "FooAI": "kein echter Provider"},
            included_providers=["OpenAI", "Gemini", "FooAI"],
        )
        self.assertEqual(labels, {"OpenAI": "gpt-test", "Gemini": "gemini-test"})

    def test_model_labels_charset_and_length_fallback(self):
        labels = snapshots.sanitize_model_labels({
            "OpenAI": "gpt-5, FakeModel: evil",   # Komma könnte Zitation fälschen
            "Gemini": "x" * 100,                  # zu lang
            "Mistral": "<script>x</script>",      # unzulässige Zeichen
            "Grok": "Grok 4.2 (Beta)",
        }, included_providers=snapshots.PROVIDER_ORDER)
        self.assertEqual(labels, {"Grok": "Grok 4.2 (Beta)"})
        # Provider-Fallback greift in build_included_models:
        result = snapshots.build_included_models(["OpenAI", "Grok"], {"OpenAI": "bad,label"})
        self.assertEqual(result, ["OpenAI", "Grok"])

    def test_model_labels_strip_badge_suffix(self):
        labels = snapshots.sanitize_model_labels({
            "OpenAI": "GPT-5.1 · New",
            "Gemini": "Gemini 2.5 Flash • Pro",
            "Grok": "Grok 4.2",
        }, included_providers=snapshots.PROVIDER_ORDER)
        self.assertEqual(labels, {
            "OpenAI": "GPT-5.1",
            "Gemini": "Gemini 2.5 Flash",
            "Grok": "Grok 4.2",
        })
        result = snapshots.build_included_models(
            ["OpenAI", "Gemini"],
            {"OpenAI": "GPT-5.1 · New", "Gemini": "Gemini 2.5 Flash • Pro"},
        )
        self.assertEqual(result, ["OpenAI: GPT-5.1", "Google Gemini: Gemini 2.5 Flash"])

    def test_consulted_models_view_maps_icon_and_model(self):
        view = snapshots.consulted_models_view(
            ["OpenAI: gpt-5.1", "Google Gemini", "Grok"]
        )
        self.assertEqual(view[0]["provider"], "OpenAI")
        self.assertEqual(view[0]["model"], "gpt-5.1")
        self.assertTrue(view[0]["icon"].endswith("chatgpt.png"))
        # Eintrag ohne Modellname behält Provider-Label, kein Icon-Verlust:
        self.assertEqual(view[1]["provider"], "Gemini")
        self.assertEqual(view[1]["model"], "")
        self.assertTrue(view[1]["icon"].endswith("gemini-icon.png"))
        # Unbekanntes Label bleibt erhalten, aber ohne Icon:
        unknown = snapshots.consulted_models_view(["FooAI: x"])
        self.assertEqual(unknown[0]["provider"], "")
        self.assertEqual(unknown[0]["icon"], "")

    def test_consensus_model_view_resolves_provider_and_pro(self):
        pro = snapshots.consensus_model_view("Anthropic-Pro")
        self.assertEqual(pro["provider"], "Anthropic")
        self.assertTrue(pro["pro"])
        self.assertTrue(pro["icon"].endswith("claude.png"))
        self.assertTrue(pro["model"])  # konkretes Modell aufgelöst
        # Frontier-ID per Substring -> Gemini:
        frontier = snapshots.consensus_model_view(
            snapshots.cfg.GEMINI_FRONTIER_LOW_MODEL
        )
        self.assertEqual(frontier["provider"], "Gemini")
        self.assertFalse(frontier["pro"])
        self.assertIsNone(snapshots.consensus_model_view(""))


class PendingResultTests(unittest.TestCase):
    def test_requires_uid_question_and_consensus(self):
        self.assertIsNone(snapshots.build_pending_result(None, "q", "c", None, "", {}, [], None, "m"))
        self.assertIsNone(snapshots.build_pending_result("u", "", "c", None, "", {}, [], None, "m"))
        self.assertIsNone(snapshots.build_pending_result("u", "q", "  ", None, "", {}, [], None, "m"))

    def test_caps_applied(self):
        payload = snapshots.build_pending_result(
            "u", "q" * 5000, "c" * (snapshots.MAX_CONSENSUS_CHARS + 50),
            None, "d" * (snapshots.MAX_DIFFERENCES_TEXT_CHARS + 50),
            {}, ["OpenAI"], None, "Anthropic",
        )
        self.assertEqual(len(payload["question"]), snapshots.MAX_QUESTION_CHARS)
        self.assertLessEqual(len(payload["consensus_md"]), snapshots.MAX_CONSENSUS_CHARS + 20)
        self.assertTrue(payload["consensus_md"].endswith("*[truncated]*"))
        self.assertEqual(len(payload["differences_text"]), snapshots.MAX_DIFFERENCES_TEXT_CHARS)
        self.assertEqual(payload["owner_uid"], "u")
        self.assertTrue(payload["answered_at"])

    def test_index_eligible(self):
        sources = [{"url": "https://a"}, {"url": "https://b"}]
        models = ["A", "B", "C"]
        good_question = "Wie funktioniert Photosynthese?"
        self.assertTrue(snapshots.compute_index_eligible(good_question, "x" * 600, sources, models))
        self.assertFalse(snapshots.compute_index_eligible(good_question, "x" * 100, sources, models))
        self.assertFalse(snapshots.compute_index_eligible(good_question, "x" * 600, sources[:1], models))
        self.assertFalse(snapshots.compute_index_eligible(good_question, "x" * 600, sources, models[:2]))
        self.assertFalse(snapshots.compute_index_eligible("kurz", "x" * 600, sources, models))

    def test_index_eligible_uses_limits_config(self):
        sources = [{"url": "https://a"}, {"url": "https://b"}]
        models = ["A", "B", "C"]
        good_question = "Wie funktioniert Photosynthese?"
        original = cfg.get_limits_config()
        try:
            cfg.apply_limits({**original, "share_min_consensus_chars": 50})
            self.assertTrue(snapshots.compute_index_eligible(good_question, "x" * 100, sources, models))
            cfg.apply_limits({**original, "share_min_sources": 5})
            self.assertFalse(snapshots.compute_index_eligible(good_question, "x" * 600, sources, models))
        finally:
            cfg.apply_limits(original)


class PublicPayloadTests(unittest.TestCase):
    def test_whitelist_excludes_internal_fields(self):
        data = make_pending()
        data.update({
            "owner_uid": "secret-uid",
            "reports_count": 3,
            "indexed": True,
            "index_eligible": True,
            "question_hash": "h",
            "source_result_id": "r",
            "status": "active",
            "slug": "slug",
            "created_at": datetime(2026, 6, 11, tzinfo=timezone.utc),
        })
        payload = snapshots.public_share_payload(data)
        for forbidden in ("owner_uid", "reports_count", "indexed", "index_eligible",
                          "question_hash", "source_result_id", "status", "expires_at"):
            self.assertNotIn(forbidden, payload)
        self.assertEqual(payload["question"], data["question"])
        self.assertEqual(payload["created_at"], "2026-06-11T00:00:00+00:00")

    def test_citation_contains_canonical_url(self):
        payload = snapshots.public_share_payload(make_pending())
        citation = snapshots.build_citation(payload, "https://www.consens.io/s/slug-abc")
        self.assertIn("consens.io. (2026-06-11).", citation)
        self.assertIn('Consensus answer to "Wie funktioniert Photosynthese in Pflanzen?".', citation)
        self.assertIn("Models consulted: OpenAI: gpt-test, Google Gemini.", citation)
        self.assertIn("Consensus model: Anthropic.", citation)
        self.assertIn("Sources: https://example.org/a", citation)
        self.assertIn("Retrieved from https://www.consens.io/s/slug-abc", citation)

    def test_citation_empty_without_models(self):
        payload = snapshots.public_share_payload(make_pending(included_models=[]))
        self.assertEqual(snapshots.build_citation(payload, "https://x"), "")


class PublicMarkdownTests(unittest.TestCase):
    SOURCES = [{"id": "S1", "url": "https://example.org/a"}, {"id": "S2", "url": "https://example.org/b"}]

    def test_script_and_raw_html_neutralized(self):
        html = render_public_markdown("Hallo <script>alert(1)</script> <img src=x onerror=alert(1)>")
        # Rohes HTML aus LLM-Antworten wird zu Text escaped, nie zu Markup.
        self.assertNotIn("<script", html)
        self.assertNotIn("<img", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("&lt;img", html)

    def test_javascript_links_stripped(self):
        html = render_public_markdown("[klick mich](javascript:alert(1))")
        self.assertNotIn('href="javascript', html)
        self.assertNotIn("href='javascript", html)

    def test_http_links_get_rel(self):
        html = render_public_markdown("[ok](https://example.org)")
        self.assertIn('href="https://example.org"', html)
        self.assertIn("nofollow", html)

    def test_source_tags_become_anchor_links(self):
        html = render_public_markdown("Fakt eins. [S1] Fakt zwei. [S1, S2]", self.SOURCES)
        self.assertIn('href="#src-1"', html)
        self.assertIn('href="#src-2"', html)
        # Link-Text ist der erkennbare Site-Name, nicht der technische Marker.
        self.assertIn(">example<", html)
        self.assertNotIn("[S1]", html)

    def test_source_tags_without_url_fall_back_to_id(self):
        html = render_public_markdown("Fakt. [S1]", [{"id": "S1", "url": ""}])
        self.assertIn('href="#src-1"', html)
        self.assertIn(">S1<", html)

    def test_unknown_source_tags_untouched(self):
        html = render_public_markdown("Fakt. [S9]", self.SOURCES)
        self.assertNotIn('href="#src-9"', html)
        self.assertIn("[S9]", html)

    def test_source_tags_in_code_untouched(self):
        html = render_public_markdown("```\nx = arr[1]\n```", self.SOURCES)
        self.assertNotIn("#src-1", html)

    def test_tables_render(self):
        html = render_public_markdown("| a | b |\n| --- | --- |\n| 1 | 2 |")
        self.assertIn("<table>", html)

    def test_plaintext_strips_and_clips(self):
        text = markdown_to_plaintext("# Titel\n\nEin **fetter** Satz.", limit=15)
        self.assertNotIn("<", text)
        self.assertNotIn("#", text)
        self.assertLessEqual(len(text), 15)

    def test_plaintext_strips_source_tags(self):
        text = markdown_to_plaintext("Fakt eins. [S1] Fakt zwei. [S1, S2]")
        self.assertNotIn("[S1]", text)
        self.assertNotIn("[S1, S2]", text)


class ShareFlowTests(unittest.TestCase):
    def setUp(self):
        self.db = FakeDb()
        self.uid = "user-1"

    def _store_pending(self, result_id=None, **overrides):
        result_id = result_id or snapshots.generate_share_id()
        self.db.stores[snapshots.PENDING_COLLECTION][result_id] = make_pending(self.uid, **overrides)
        return result_id

    def test_unknown_result_id_raises_not_found(self):
        with self.assertRaises(ShareError) as ctx:
            snapshots.create_share_from_pending(self.uid, snapshots.generate_share_id(),
                                                db=self.db, consume_quota=lambda: True)
        self.assertEqual(ctx.exception.code, "not_found")

    def test_invalid_result_id_raises_not_found(self):
        with self.assertRaises(ShareError) as ctx:
            snapshots.create_share_from_pending(self.uid, "../../etc/passwd",
                                                db=self.db, consume_quota=lambda: True)
        self.assertEqual(ctx.exception.code, "not_found")

    def test_foreign_result_raises_forbidden(self):
        result_id = self._store_pending()
        with self.assertRaises(ShareError) as ctx:
            snapshots.create_share_from_pending("other-user", result_id,
                                                db=self.db, consume_quota=lambda: True)
        self.assertEqual(ctx.exception.code, "forbidden")

    def test_expired_result_is_deleted_and_not_found(self):
        result_id = self._store_pending(
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1))
        with self.assertRaises(ShareError) as ctx:
            snapshots.create_share_from_pending(self.uid, result_id,
                                                db=self.db, consume_quota=lambda: True)
        self.assertEqual(ctx.exception.code, "not_found")
        self.assertNotIn(result_id, self.db.stores[snapshots.PENDING_COLLECTION])

    def test_quota_exceeded(self):
        result_id = self._store_pending()
        with self.assertRaises(ShareError) as ctx:
            snapshots.create_share_from_pending(self.uid, result_id,
                                                db=self.db, consume_quota=lambda: False)
        self.assertEqual(ctx.exception.code, "quota_exceeded")

    def test_create_share_snapshot(self):
        result_id = self._store_pending()
        result = snapshots.create_share_from_pending(self.uid, result_id,
                                                     db=self.db, consume_quota=lambda: True)
        self.assertTrue(result["created"])
        share = self.db.stores[snapshots.SHARES_COLLECTION][result["share_id"]]
        self.assertEqual(share["status"], "active")
        self.assertEqual(share["owner_uid"], self.uid)
        self.assertEqual(share["slug"], result["slug"])
        self.assertFalse(share["indexed"])
        self.assertEqual(share["reports_count"], 0)
        self.assertIn("question_hash", share)
        # Backlink für Idempotenz
        pending = self.db.stores[snapshots.PENDING_COLLECTION][result_id]
        self.assertEqual(pending["share_id"], result["share_id"])

    def test_create_share_is_idempotent(self):
        result_id = self._store_pending()
        quota_calls = []

        def quota():
            quota_calls.append(1)
            return True

        first = snapshots.create_share_from_pending(self.uid, result_id, db=self.db, consume_quota=quota)
        second = snapshots.create_share_from_pending(self.uid, result_id, db=self.db, consume_quota=quota)
        self.assertEqual(first["share_id"], second["share_id"])
        self.assertFalse(second["created"])
        self.assertEqual(len(quota_calls), 1)

    def test_private_and_public_snapshots_are_distinct_and_idempotent(self):
        result_id = self._store_pending()
        quota_calls = []

        def quota():
            quota_calls.append(1)
            return True

        private = snapshots.create_share_from_pending(
            self.uid, result_id, db=self.db, consume_quota=quota, visibility="private"
        )
        private_again = snapshots.create_share_from_pending(
            self.uid, result_id, db=self.db, consume_quota=quota, visibility="private"
        )
        public = snapshots.create_share_from_pending(
            self.uid, result_id, db=self.db, consume_quota=quota, visibility="public"
        )
        self.assertEqual(private["share_id"], private_again["share_id"])
        self.assertNotEqual(private["share_id"], public["share_id"])
        self.assertEqual(len(quota_calls), 2)
        self.assertEqual(
            self.db.stores[snapshots.SHARES_COLLECTION][private["share_id"]]["visibility"],
            "private",
        )

    def test_pending_result_reuse_requires_owner_and_unexpired_ttl(self):
        db = FakeDb()
        result_id = "P" * 16
        db.stores["pending_results"][result_id] = make_pending(uid="owner")
        self.assertTrue(snapshots.pending_result_is_available("owner", result_id, db=db))
        self.assertFalse(snapshots.pending_result_is_available("other", result_id, db=db))
        db.stores["pending_results"][result_id]["expires_at"] = (
            datetime.now(timezone.utc) - timedelta(seconds=1)
        )
        self.assertFalse(snapshots.pending_result_is_available("owner", result_id, db=db))

    def test_private_share_cannot_be_reported(self):
        result_id = self._store_pending()
        created = snapshots.create_share_from_pending(
            self.uid, result_id, db=self.db, consume_quota=lambda: True,
            visibility="private",
        )
        with self.assertRaises(ShareError):
            snapshots.report_share(created["share_id"], "spam", db=self.db)

    def test_revoke_by_owner(self):
        result_id = self._store_pending()
        created = snapshots.create_share_from_pending(self.uid, result_id,
                                                      db=self.db, consume_quota=lambda: True)
        snapshots.revoke_share(created["share_id"], self.uid, db=self.db)
        share = self.db.stores[snapshots.SHARES_COLLECTION][created["share_id"]]
        self.assertEqual(share["status"], "revoked")

    def test_revoke_foreign_share_forbidden_unless_admin(self):
        result_id = self._store_pending()
        created = snapshots.create_share_from_pending(self.uid, result_id,
                                                      db=self.db, consume_quota=lambda: True)
        with self.assertRaises(ShareError) as ctx:
            snapshots.revoke_share(created["share_id"], "other-user", db=self.db)
        self.assertEqual(ctx.exception.code, "forbidden")
        snapshots.revoke_share(created["share_id"], "other-user", is_admin=True, db=self.db)
        self.assertEqual(self.db.stores[snapshots.SHARES_COLLECTION][created["share_id"]]["status"], "revoked")

    def test_report_share_increments_and_aggregates(self):
        result_id = self._store_pending()
        created = snapshots.create_share_from_pending(self.uid, result_id,
                                                      db=self.db, consume_quota=lambda: True)
        count = snapshots.report_share(created["share_id"], "spam", db=self.db)
        self.assertEqual(count, 1)
        count = snapshots.report_share(created["share_id"], "kein-gueltiger-grund", db=self.db)
        self.assertEqual(count, 2)
        share = self.db.stores[snapshots.SHARES_COLLECTION][created["share_id"]]
        self.assertEqual(share["reports_count"], 2)
        self.assertEqual(share["report_reasons"], {"spam": 1, "other": 1})

    def test_report_share_rejects_inactive_or_unknown(self):
        with self.assertRaises(ShareError):
            snapshots.report_share(snapshots.generate_share_id(), "spam", db=self.db)
        result_id = self._store_pending()
        created = snapshots.create_share_from_pending(self.uid, result_id,
                                                      db=self.db, consume_quota=lambda: True)
        snapshots.revoke_share(created["share_id"], self.uid, db=self.db)
        with self.assertRaises(ShareError):
            snapshots.report_share(created["share_id"], "spam", db=self.db)

    def test_report_fields_stay_out_of_public_payload(self):
        payload = snapshots.public_share_payload({
            "question": "q", "report_reasons": {"spam": 1}, "last_reported_at": "x",
        })
        self.assertNotIn("report_reasons", payload)
        self.assertNotIn("last_reported_at", payload)

    def test_list_shares_for_owner_is_whitelisted(self):
        result_id = self._store_pending()
        created = snapshots.create_share_from_pending(self.uid, result_id,
                                                      db=self.db, consume_quota=lambda: True)
        other_id = snapshots.generate_share_id()
        self.db.stores[snapshots.SHARES_COLLECTION][other_id] = {"owner_uid": "someone-else", "question": "x"}

        shares = snapshots.list_shares_for_owner(self.uid, db=self.db)
        self.assertEqual(len(shares), 1)
        self.assertEqual(shares[0]["share_id"], created["share_id"])
        self.assertEqual(set(shares[0].keys()), {
            "share_id", "path", "question", "status", "visibility", "created_at",
            "indexed", "index_requested", "index_eligible",
        })


class ModerationAndCleanupTests(unittest.TestCase):
    def setUp(self):
        self.db = FakeDb()
        self.uid = "user-1"
        snapshots.invalidate_share_cache()

    def _make_share(self, **overrides):
        share_id = snapshots.generate_share_id()
        doc = make_pending(self.uid)
        doc.update({
            "status": "active", "slug": "test-slug", "indexed": False,
            "index_eligible": True, "reports_count": 0,
            "question_hash": snapshots.question_hash(doc["question"]),
            "created_at": datetime(2026, 6, 1, tzinfo=timezone.utc),
        })
        doc.update(overrides)
        self.db.stores[snapshots.SHARES_COLLECTION][share_id] = doc
        return share_id

    def _share(self, share_id):
        return self.db.stores[snapshots.SHARES_COLLECTION][share_id]

    def test_block_sets_status_and_clears_indexed(self):
        share_id = self._make_share(indexed=True, needs_review=True)
        result = snapshots.moderate_share(share_id, action="block", db=self.db)
        stored = self._share(share_id)
        self.assertEqual(stored["status"], "blocked")
        self.assertFalse(stored["indexed"])
        self.assertFalse(stored["needs_review"])
        self.assertEqual(result["status"], "blocked")

    def test_unblock_requires_blocked_status(self):
        share_id = self._make_share(status="blocked")
        snapshots.moderate_share(share_id, action="unblock", db=self.db)
        self.assertEqual(self._share(share_id)["status"], "active")
        with self.assertRaises(ShareError):
            snapshots.moderate_share(share_id, action="unblock", db=self.db)

    def test_indexed_only_for_active_shares(self):
        share_id = self._make_share()
        snapshots.moderate_share(share_id, indexed=True, db=self.db)
        self.assertTrue(self._share(share_id)["indexed"])
        snapshots.moderate_share(share_id, indexed=False, db=self.db)
        self.assertFalse(self._share(share_id)["indexed"])

        blocked_id = self._make_share(status="blocked")
        with self.assertRaises(ShareError):
            snapshots.moderate_share(blocked_id, indexed=True, db=self.db)
        # De-Indexieren bleibt auch für nicht-aktive erlaubt
        snapshots.moderate_share(blocked_id, indexed=False, db=self.db)

    def test_moderate_validates_input(self):
        share_id = self._make_share()
        with self.assertRaises(ShareError):
            snapshots.moderate_share(share_id, action="purge", db=self.db)
        with self.assertRaises(ShareError):
            snapshots.moderate_share(share_id, db=self.db)
        with self.assertRaises(ShareError):
            snapshots.moderate_share(snapshots.generate_share_id(), action="block", db=self.db)

    def test_revoke_cannot_be_blocked(self):
        share_id = self._make_share(status="revoked")
        with self.assertRaises(ShareError):
            snapshots.moderate_share(share_id, action="block", db=self.db)

    def test_auto_noindex_after_report_threshold(self):
        share_id = self._make_share(indexed=True,
                                    reports_count=snapshots.AUTO_NOINDEX_REPORTS - 1)
        snapshots.report_share(share_id, "spam", db=self.db)
        stored = self._share(share_id)
        self.assertEqual(stored["reports_count"], snapshots.AUTO_NOINDEX_REPORTS)
        self.assertFalse(stored["indexed"])
        self.assertTrue(stored["needs_review"])

    def test_reports_below_threshold_keep_indexed(self):
        share_id = self._make_share(indexed=True)
        snapshots.report_share(share_id, "spam", db=self.db)
        stored = self._share(share_id)
        self.assertTrue(stored["indexed"])
        self.assertNotIn("needs_review", stored)

    def test_admin_list_prioritizes_review_and_reports(self):
        plain = self._make_share()
        reported = self._make_share(reports_count=2)
        urgent = self._make_share(reports_count=7, needs_review=True)

        shares = snapshots.list_shares_for_admin(db=self.db)
        self.assertEqual([s["share_id"] for s in shares], [urgent, reported, plain])

        reported_only = snapshots.list_shares_for_admin(db=self.db, only_reported=True)
        self.assertEqual([s["share_id"] for s in reported_only], [urgent, reported])
        # Moderationsfelder vorhanden, Snapshot-Inhalte nicht
        self.assertNotIn("consensus_md", reported_only[0])
        self.assertEqual(reported_only[0]["reports_count"], 7)

    def test_cleanup_revoked_shares_after_30_days(self):
        old = self._make_share(status="revoked",
                               revoked_at=datetime.now(timezone.utc) - timedelta(days=31))
        recent = self._make_share(status="revoked",
                                  revoked_at=datetime.now(timezone.utc) - timedelta(days=5))
        active = self._make_share()

        deleted = snapshots.cleanup_revoked_shares(db=self.db)
        store = self.db.stores[snapshots.SHARES_COLLECTION]
        self.assertEqual(deleted, 1)
        self.assertNotIn(old, store)
        self.assertIn(recent, store)
        self.assertIn(active, store)

    def test_find_canonical_share_prefers_oldest_indexed(self):
        qh = snapshots.question_hash("Wie funktioniert Photosynthese in Pflanzen?")
        self._make_share(indexed=False)  # nicht indexiert: kein Canonical-Ziel
        newer = self._make_share(indexed=True,
                                 created_at=datetime(2026, 6, 10, tzinfo=timezone.utc))
        oldest = self._make_share(indexed=True,
                                  created_at=datetime(2026, 6, 2, tzinfo=timezone.utc))
        self._make_share(indexed=True, status="revoked",
                         created_at=datetime(2026, 6, 1, tzinfo=timezone.utc))

        best = snapshots.find_canonical_share(qh, db=self.db)
        self.assertEqual(best["share_id"], oldest)

        other_hash = snapshots.question_hash("ganz andere frage")
        self.assertIsNone(snapshots.find_canonical_share(other_hash, db=self.db))

    def test_list_indexed_share_urls_filters(self):
        indexed = self._make_share(indexed=True)
        self._make_share(indexed=False)
        self._make_share(indexed=True, status="blocked")

        urls = snapshots.list_indexed_share_urls(db=self.db)
        self.assertEqual(len(urls), 1)
        self.assertIn(indexed, urls[0]["path"])
        self.assertEqual(urls[0]["lastmod"], "2026-06-01")

    def test_list_related_shares_only_indexed_active_and_excludes_self(self):
        current = self._make_share(indexed=True,
                                   question="Wie funktioniert Photosynthese in Pflanzen?")
        related = self._make_share(indexed=True,
                                   question="Photosynthese und Chlorophyll erklaert",
                                   slug="photo-slug")
        self._make_share(indexed=False, question="Photosynthese fuer Kinder")  # noindex
        self._make_share(indexed=True, status="blocked",
                         question="Photosynthese im Detail")  # gesperrt

        items = snapshots.list_related_shares(current, "Photosynthese", db=self.db)
        paths = [item["path"] for item in items]
        self.assertEqual(len(items), 1)
        self.assertIn(related, paths[0])
        self.assertNotIn(current, paths[0])
        self.assertEqual(set(items[0].keys()), {"path", "question", "models_count"})

    def test_list_related_shares_ranks_by_token_overlap(self):
        current = self._make_share(indexed=True, question="How do solar panels work?")
        relevant = self._make_share(indexed=True,
                                    question="Are solar panels worth the cost?",
                                    slug="solar-slug",
                                    created_at=datetime(2026, 5, 1, tzinfo=timezone.utc))
        unrelated = self._make_share(indexed=True,
                                     question="What is the capital of France?",
                                     slug="france-slug",
                                     created_at=datetime(2026, 6, 30, tzinfo=timezone.utc))

        items = snapshots.list_related_shares(current, "How do solar panels work?",
                                              db=self.db)
        # Themen-Treffer schlaegt den neueren, aber unverwandten Share.
        self.assertIn(relevant, items[0]["path"])
        self.assertEqual(len(items), 2)

    def test_share_cache_returns_cached_until_invalidated(self):
        share_id = self._make_share()
        first = snapshots.get_share_cached(share_id, db=self.db)
        self.assertEqual(first["status"], "active")

        # Direkte Firestore-Änderung: Cache liefert weiter den alten Stand …
        self._share(share_id)["status"] = "blocked"
        cached = snapshots.get_share_cached(share_id, db=self.db)
        self.assertEqual(cached["status"], "active")

        # … bis invalidiert wird (wie bei revoke/moderate).
        snapshots.invalidate_share_cache(share_id)
        fresh = snapshots.get_share_cached(share_id, db=self.db)
        self.assertEqual(fresh["status"], "blocked")

    def test_revoke_invalidates_cache(self):
        share_id = self._make_share()
        snapshots.get_share_cached(share_id, db=self.db)
        snapshots.revoke_share(share_id, self.uid, db=self.db)
        self.assertEqual(snapshots.get_share_cached(share_id, db=self.db)["status"], "revoked")


class SharePageRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = FastAPI()
        app.state.limiter = limiter
        app.include_router(share_router.router)
        cls.client = TestClient(app)
        cls.share_id = snapshots.generate_share_id()

    def setUp(self):
        # get_share_cached cached sonst das Dokument des vorherigen Tests
        snapshots.invalidate_share_cache()
        self.related_shares_patch = patch.object(
            share_router.snapshots, "list_related_shares", return_value=[]
        )
        self.related_shares_patch.start()
        self.addCleanup(self.related_shares_patch.stop)
        self.watch_history_patch = patch.object(
            share_router.snapshots, "list_watch_history", return_value=[]
        )
        self.watch_history_patch.start()
        self.addCleanup(self.watch_history_patch.stop)
        self.watch_meta_patch = patch.object(
            share_router.watch_service, "get_public_watch_meta", return_value=None
        )
        self.watch_meta_patch.start()
        self.addCleanup(self.watch_meta_patch.stop)

    def _share_doc(self, **overrides):
        data = make_pending()
        data.update({"status": "active", "slug": "wie-funktioniert-photosynthese-in-pflanzen"})
        data.update(overrides)
        return data

    def test_invalid_id_renders_404_with_noindex(self):
        response = self.client.get("/s/irgendwas-kurz")
        self.assertEqual(response.status_code, 404)
        self.assertIn("noindex", response.headers.get("X-Robots-Tag", ""))

    def test_unknown_id_renders_404(self):
        with patch.object(share_router.snapshots, "get_share", return_value=None):
            response = self.client.get("/s/slug-%s" % self.share_id)
        self.assertEqual(response.status_code, 404)

    def test_revoked_share_renders_410(self):
        doc = self._share_doc(status="revoked")
        with patch.object(share_router.snapshots, "get_share", return_value=doc):
            response = self.client.get("/s/%s-%s" % (doc["slug"], self.share_id))
        self.assertEqual(response.status_code, 410)
        self.assertIn("noindex", response.headers.get("X-Robots-Tag", ""))

    def test_wrong_slug_redirects_to_canonical(self):
        doc = self._share_doc()
        with patch.object(share_router.snapshots, "get_share", return_value=doc):
            response = self.client.get("/s/falscher-slug-%s" % self.share_id, follow_redirects=False)
        self.assertEqual(response.status_code, 301)
        self.assertEqual(response.headers["location"], "/s/%s-%s" % (doc["slug"], self.share_id))

    def test_active_share_renders_content_with_noindex(self):
        doc = self._share_doc()
        with patch.object(share_router.snapshots, "get_share", return_value=doc):
            response = self.client.get("/s/%s-%s" % (doc["slug"], self.share_id))
        self.assertEqual(response.status_code, 200)
        self.assertIn("noindex", response.headers.get("X-Robots-Tag", ""))
        body = response.text
        self.assertIn("Wie funktioniert Photosynthese in Pflanzen?", body)
        self.assertIn("Photosynthese wandelt Licht in Energie um.", body)
        self.assertIn('content="noindex, follow"', body)
        self.assertIn("Ask your own question", body)
        self.assertNotIn("user-1", body)  # owner_uid darf nie im HTML landen

    def test_private_share_requires_owner_and_is_never_publicly_cached(self):
        doc = self._share_doc(visibility="private", owner_uid="owner-1")
        path = "/s/%s-%s" % (doc["slug"], self.share_id)
        with patch.object(share_router.snapshots, "get_share", return_value=doc), \
                patch.object(share_router, "verify_user_token", return_value=""):
            denied = self.client.get(path)
        self.assertEqual(denied.status_code, 403)
        self.assertNotIn("Photosynthese wandelt", denied.text)

        with patch.object(share_router.snapshots, "get_share", return_value=doc), \
                patch.object(share_router, "extract_id_token", return_value="token"), \
                patch.object(share_router, "verify_user_token", return_value="owner-1"):
            allowed = self.client.get(path)
        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(allowed.headers["Cache-Control"], "private, no-store")
        self.assertIn("· Private", allowed.text)
        self.assertNotIn("Report this page", allowed.text)
        self.assertIn("noindex, nofollow", allowed.headers["X-Robots-Tag"])

    def test_differences_cards_and_toggle_rendered(self):
        doc = self._share_doc(differences_data={
            "claims": [],
            "differences": [{
                "claim": "Die Modelle widersprechen sich bei der Hauptstadt.",
                "type": "contradiction",
                "positions": [
                    {"stance": "Paris ist die Hauptstadt.", "models": ["OpenAI", "Gemini"], "quote": "Paris"},
                    {"stance": "Lyon ist die Hauptstadt.", "models": ["Grok"], "quote": "Lyon"},
                ],
                "verify": "Offizielle Quelle prüfen.",
            }],
            "best_model": "OpenAI",
            "models_compared": ["OpenAI", "Gemini", "Grok"],
        })
        with patch.object(share_router.snapshots, "get_share", return_value=doc):
            response = self.client.get("/s/%s-%s" % (doc["slug"], self.share_id))
        body = response.text
        self.assertIn("toggleDifferencesView", body)
        self.assertIn("Die Modelle widersprechen sich bei der Hauptstadt.", body)
        self.assertIn("1 notable difference", body)
        self.assertIn("1 contradiction", body)
        self.assertIn("across 3 models", body)
        self.assertIn("How to verify: Offizielle Quelle prüfen.", body)
        self.assertIn("Report this page", body)

    def test_differences_fallback_text_rendered(self):
        doc = self._share_doc(differences_data=None, differences_text="**Unterschied:** nur Detailfragen.")
        with patch.object(share_router.snapshots, "get_share", return_value=doc):
            response = self.client.get("/s/%s-%s" % (doc["slug"], self.share_id))
        body = response.text
        self.assertIn("toggleDifferencesView", body)
        self.assertIn("<strong>Unterschied:</strong>", body)

    def test_related_questions_section_rendered(self):
        doc = self._share_doc()
        related = [
            {"path": "/s/solar-abc", "question": "Are solar panels worth it?", "models_count": 4},
        ]
        with patch.object(share_router.snapshots, "get_share", return_value=doc), \
                patch.object(share_router.snapshots, "list_related_shares", return_value=related):
            response = self.client.get("/s/%s-%s" % (doc["slug"], self.share_id))
        body = response.text
        self.assertIn("Related questions", body)
        self.assertIn("Are solar panels worth it?", body)
        self.assertIn("/s/solar-abc", body)
        self.assertIn("4 models compared", body)

    def test_related_questions_section_hidden_when_empty(self):
        doc = self._share_doc()
        with patch.object(share_router.snapshots, "get_share", return_value=doc), \
                patch.object(share_router.snapshots, "list_related_shares", return_value=[]):
            response = self.client.get("/s/%s-%s" % (doc["slug"], self.share_id))
        self.assertNotIn("Related questions", response.text)

    def test_watch_history_renders_inline_svg_and_events(self):
        doc = self._share_doc()
        points = [
            {"ts": datetime(2026, 7, 1, tzinfo=timezone.utc), "agreement_score": 54,
             "verdict": "partially", "changed": False, "severity": "minor", "change_summary": ""},
            {"ts": datetime(2026, 7, 8, tzinfo=timezone.utc), "agreement_score": 76,
             "verdict": "mostly", "changed": True, "severity": "major",
             "change_summary": "The central recommendation changed."},
        ]
        with patch.object(share_router.snapshots, "get_share", return_value=doc), \
                patch.object(share_router.snapshots, "list_watch_history", return_value=points):
            response = self.client.get("/s/%s-%s" % (doc["slug"], self.share_id))
        body = response.text
        self.assertIn("Agreement over time", body)
        self.assertIn('class="watch-chart"', body)
        self.assertIn("M 38.0", body)
        self.assertIn("The central recommendation changed.", body)
        self.assertIn("76<span>/100</span>", body)

    def test_watch_history_renders_position_map_and_direction_shift(self):
        doc = self._share_doc()
        position_map = {
            "schema_version": 1,
            "shift_score": 50,
            "shift_label": "Turning",
            "center": ["Adopt now"],
            "models": [
                {"provider": "OpenAI", "movement_score": 100, "moved": True,
                 "summary": "Recommended adoption timeline: Wait"},
                {"provider": "Gemini", "movement_score": 0, "moved": False, "summary": ""},
            ],
            "dimensions": [{
                "label": "Recommended adoption timeline", "type": "contradiction",
                "positions": [
                    {"stance": "Adopt now", "models": ["Gemini"]},
                    {"stance": "Wait", "models": ["OpenAI"]},
                ],
            }],
        }
        points = [{
            "ts": datetime(2026, 7, 8, tzinfo=timezone.utc), "agreement_score": 62,
            "verdict": "partially", "changed": True, "severity": "major",
            "change_summary": "Recommendation changed.", "opinion_map": position_map,
        }]
        with patch.object(share_router.snapshots, "get_share", return_value=doc), \
                patch.object(share_router.snapshots, "list_watch_history", return_value=points):
            response = self.client.get("/s/%s-%s" % (doc["slug"], self.share_id))
        body = response.text
        self.assertIn("Position Map", body)
        self.assertIn("Direction Shift", body)
        self.assertIn("50<span>/100</span>", body)
        self.assertIn("Recommended adoption timeline", body)
        self.assertIn("OpenAI", body)

    def test_active_watch_page_shows_run_metadata_before_history_exists(self):
        doc = self._share_doc()
        now = datetime(2026, 7, 12, 8, 30, tzinfo=timezone.utc)
        meta = {
            "status": "active", "interval": "weekly",
            "last_run_at": now, "next_run_at": now + timedelta(days=7),
            "created_at": now - timedelta(days=14),
        }
        with patch.object(share_router.snapshots, "get_share", return_value=doc), \
                patch.object(share_router.watch_service, "get_public_watch_meta", return_value=meta):
            response = self.client.get("/s/%s-%s" % (doc["slug"], self.share_id))
        body = response.text
        self.assertIn('class="watch-meta-compact is-active"', body)
        self.assertIn("<b>Last</b>", body)
        self.assertIn("2026-07-12 08:30 UTC", body)
        self.assertIn("<b>Next</b>", body)
        self.assertIn("Original consensus", body)

    def test_watch_page_shows_selected_local_run_time(self):
        doc = self._share_doc()
        next_run = datetime(2026, 7, 19, 7, 0, tzinfo=timezone.utc)
        meta = {
            "status": "active", "interval": "weekly", "run_time": "09:00",
            "timezone": "Europe/Berlin", "next_run_at": next_run,
        }
        with patch.object(share_router.snapshots, "get_share", return_value=doc), \
                patch.object(share_router.watch_service, "get_public_watch_meta", return_value=meta):
            response = self.client.get("/s/%s-%s" % (doc["slug"], self.share_id))
        self.assertIn("Weekly at 09:00 (Europe/Berlin)", response.text)
        self.assertIn("2026-07-19 09:00 Europe/Berlin", response.text)

    def test_rendered_citation_contains_canonical_url(self):
        doc = self._share_doc()
        with patch.object(share_router.snapshots, "get_share", return_value=doc):
            response = self.client.get("/s/%s-%s" % (doc["slug"], self.share_id))
        body = response.text
        canonical = "%s/s/%s-%s" % (share_router.SITE_URL, doc["slug"], self.share_id)
        self.assertIn("Retrieved from %s" % canonical, body)
        self.assertIn('rel="canonical" href="%s"' % canonical, body)
        self.assertIn("copyCitationBtn", body)

    def test_noindex_page_has_seo_tags_and_cache_control(self):
        doc = self._share_doc()
        with patch.object(share_router.snapshots, "get_share", return_value=doc):
            response = self.client.get("/s/%s-%s" % (doc["slug"], self.share_id))
        self.assertEqual(response.status_code, 200)
        self.assertIn("noindex", response.headers.get("X-Robots-Tag", ""))
        self.assertEqual(
            response.headers.get("Cache-Control"),
            "public, max-age=300, s-maxage=86400, stale-while-revalidate=86400",
        )
        body = response.text
        self.assertIn('property="og:title"', body)
        self.assertIn('name="twitter:card"', body)
        self.assertIn('name="description"', body)
        self.assertIn('"@type": "Article"', body)
        self.assertIn('"citation"', body)
        self.assertIn("https://example.org/a", body)

    def test_indexed_page_gets_index_follow(self):
        doc = self._share_doc(indexed=True)
        with patch.object(share_router.snapshots, "get_share", return_value=doc):
            response = self.client.get("/s/%s-%s" % (doc["slug"], self.share_id))
        self.assertEqual(response.headers.get("X-Robots-Tag"), "index, follow")
        self.assertIn('content="index, follow"', response.text)
        # selbst-kanonisch
        canonical = "%s/s/%s-%s" % (share_router.SITE_URL, doc["slug"], self.share_id)
        self.assertIn('rel="canonical" href="%s"' % canonical, response.text)

    def test_noindex_duplicate_points_canonical_to_indexed_share(self):
        doc = self._share_doc(question_hash="qh-1")
        target = {"share_id": snapshots.generate_share_id(), "slug": "anderer-slug", "created_key": ""}
        with patch.object(share_router.snapshots, "get_share", return_value=doc), \
                patch.object(share_router.snapshots, "find_canonical_share",
                             return_value=target) as mocked:
            response = self.client.get("/s/%s-%s" % (doc["slug"], self.share_id))
        mocked.assert_called_once_with("qh-1")
        body = response.text
        canonical = "%s/s/%s-%s" % (share_router.SITE_URL, target["slug"], target["share_id"])
        own_url = "%s/s/%s-%s" % (share_router.SITE_URL, doc["slug"], self.share_id)
        self.assertIn('rel="canonical" href="%s"' % canonical, body)
        # Zitation bleibt auf der eigenen URL, nur das rel=canonical dedupliziert
        self.assertIn("Retrieved from %s" % own_url, body)
        self.assertIn("noindex", response.headers.get("X-Robots-Tag", ""))

    def test_sitemap_shares_lists_only_indexed(self):
        urls = [{"path": "/s/frage-eins-abc", "lastmod": "2026-06-10"}]
        with patch.object(share_router.snapshots, "list_indexed_share_urls", return_value=urls):
            response = self.client.get("/sitemap-shares.xml")
        self.assertEqual(response.status_code, 200)
        self.assertIn("application/xml", response.headers.get("content-type", ""))
        body = response.text
        self.assertIn("<loc>%s/s/frage-eins-abc</loc>" % share_router.SITE_URL, body)
        self.assertIn("<lastmod>2026-06-10</lastmod>", body)

    def test_report_endpoint_maps_share_errors(self):
        with patch.object(share_router.snapshots, "report_share", return_value=1) as mocked:
            response = self.client.post(
                "/api/share/%s/report" % self.share_id, json={"reason": "spam"})
        self.assertEqual(response.status_code, 200)
        mocked.assert_called_once_with(self.share_id, "spam")

        with patch.object(share_router.snapshots, "report_share",
                          side_effect=ShareError("not_found", "Share not found.")):
            response = self.client.post(
                "/api/share/%s/report" % self.share_id, json={"reason": "spam"})
        self.assertEqual(response.status_code, 404)


class ShareApiRouteTests(unittest.TestCase):
    """Auth-gated Share-API-Routen mit gemockter Token-Verifikation."""

    @classmethod
    def setUpClass(cls):
        app = FastAPI()
        app.state.limiter = limiter
        app.include_router(share_router.router)
        cls.client = TestClient(app)
        cls.share_id = snapshots.generate_share_id()

    def _auth_patches(self, uid="user-1"):
        return (
            patch.object(share_router, "extract_id_token", return_value="tok"),
            patch.object(share_router, "verify_user_token", return_value=uid),
        )

    def test_my_shares_requires_auth(self):
        with patch.object(share_router, "extract_id_token", return_value=None):
            response = self.client.get("/api/my/shares")
        self.assertEqual(response.status_code, 401)

    def test_my_shares_returns_owner_list(self):
        shares = [{"share_id": self.share_id, "path": "/s/slug-%s" % self.share_id,
                   "question": "Q?", "status": "active", "created_at": ""}]
        token_patch, verify_patch = self._auth_patches()
        with token_patch, verify_patch, \
                patch.object(share_router.snapshots, "list_shares_for_owner",
                             return_value=shares) as mocked:
            response = self.client.get("/api/my/shares")
        self.assertEqual(response.status_code, 200)
        mocked.assert_called_once_with("user-1")
        data = response.json()
        self.assertEqual(data["shares"], shares)
        self.assertEqual(data["site_url"], share_router.SITE_URL)

    def test_delete_share_revokes_for_owner(self):
        token_patch, verify_patch = self._auth_patches()
        with token_patch, verify_patch, \
                patch.object(share_router, "is_user_admin", return_value=False), \
                patch.object(share_router.snapshots, "revoke_share") as mocked:
            response = self.client.delete("/api/share/%s" % self.share_id)
        self.assertEqual(response.status_code, 200)
        mocked.assert_called_once_with(self.share_id, "user-1", is_admin=False)

    def test_delete_share_maps_share_errors(self):
        token_patch, verify_patch = self._auth_patches()
        with token_patch, verify_patch, \
                patch.object(share_router, "is_user_admin", return_value=False), \
                patch.object(share_router.snapshots, "revoke_share",
                             side_effect=ShareError("forbidden", "You can only revoke your own shares.")):
            response = self.client.delete("/api/share/%s" % self.share_id)
        self.assertEqual(response.status_code, 403)


class AdminShareRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = FastAPI()
        app.state.limiter = limiter
        app.include_router(admin_router.router)
        cls.client = TestClient(app)
        cls.share_id = snapshots.generate_share_id()

    def _admin_patches(self, is_admin=True):
        return (
            patch.object(admin_router, "extract_id_token", return_value="tok"),
            patch.object(admin_router, "verify_user_token", return_value="admin-1"),
            patch.object(admin_router, "is_user_admin", return_value=is_admin),
        )

    def test_list_requires_admin(self):
        token_patch, verify_patch, admin_patch = self._admin_patches(is_admin=False)
        with token_patch, verify_patch, admin_patch:
            response = self.client.get("/api/admin/shares")
        self.assertEqual(response.status_code, 403)

    def test_list_passes_filter(self):
        token_patch, verify_patch, admin_patch = self._admin_patches()
        with token_patch, verify_patch, admin_patch, \
                patch.object(admin_router, "snapshots_site_url", return_value="https://x"), \
                patch.object(admin_router.snapshots, "list_shares_for_admin",
                             return_value=[]) as mocked:
            response = self.client.get("/api/admin/shares")
            self.assertEqual(response.status_code, 200)
            mocked.assert_called_with(only_reported=True)

            response = self.client.get("/api/admin/shares?filter=all")
            self.assertEqual(response.status_code, 200)
            mocked.assert_called_with(only_reported=False)

    def test_moderate_forwards_action_and_indexed(self):
        token_patch, verify_patch, admin_patch = self._admin_patches()
        result = {"status": "blocked", "indexed": False, "index_eligible": True}
        with token_patch, verify_patch, admin_patch, \
                patch.object(admin_router.snapshots, "moderate_share",
                             return_value=result) as mocked:
            response = self.client.post(
                "/api/admin/shares/%s/moderate" % self.share_id,
                json={"action": "block"})
        self.assertEqual(response.status_code, 200)
        mocked.assert_called_once_with(
            self.share_id,
            action="block",
            indexed=None,
            actor_uid="admin-1",
            source="admin_ui",
        )
        self.assertEqual(response.json()["share"]["share_status"], "blocked")

    def test_moderate_validates_indexed_type_and_maps_errors(self):
        token_patch, verify_patch, admin_patch = self._admin_patches()
        with token_patch, verify_patch, admin_patch:
            response = self.client.post(
                "/api/admin/shares/%s/moderate" % self.share_id,
                json={"indexed": "yes"})
            self.assertEqual(response.status_code, 400)

            with patch.object(admin_router.snapshots, "moderate_share",
                              side_effect=ShareError("not_found", "Share not found.")):
                response = self.client.post(
                    "/api/admin/shares/%s/moderate" % self.share_id,
                    json={"action": "block"})
            self.assertEqual(response.status_code, 404)


class IndexingRequestTests(unittest.TestCase):
    """Self-Service-Nominierung fuer den Google-Index (nur Flag, nie indexed)."""

    def setUp(self):
        self.db = FakeDb()
        self.uid = "user-1"
        snapshots.invalidate_share_cache()
        self.share_id = snapshots.generate_share_id()
        doc = make_pending(self.uid)
        doc.update({
            "status": "active", "slug": "test-slug", "visibility": "public",
            "indexed": False, "index_eligible": True, "reports_count": 0,
        })
        self.db.stores[snapshots.SHARES_COLLECTION][self.share_id] = doc

    def test_owner_can_request_and_withdraw(self):
        state = snapshots.request_share_indexing(self.share_id, self.uid, want=True, db=self.db)
        self.assertEqual(state, {"indexed": False, "index_requested": True, "index_eligible": True})
        stored = self.db.stores[snapshots.SHARES_COLLECTION][self.share_id]
        self.assertTrue(stored["index_requested"])
        self.assertTrue(stored["needs_review"])
        self.assertFalse(stored["indexed"])  # nie automatisch indexieren

        state = snapshots.request_share_indexing(self.share_id, self.uid, want=False, db=self.db)
        self.assertFalse(state["index_requested"])
        stored = self.db.stores[snapshots.SHARES_COLLECTION][self.share_id]
        self.assertFalse(stored["index_requested"])
        self.assertFalse(stored["needs_review"])

    def test_only_owner_public_active_can_request(self):
        with self.assertRaisesRegex(ShareError, "own pages"):
            snapshots.request_share_indexing(self.share_id, "intruder", db=self.db)
        self.db.stores[snapshots.SHARES_COLLECTION][self.share_id]["visibility"] = "private"
        with self.assertRaisesRegex(ShareError, "public pages"):
            snapshots.request_share_indexing(self.share_id, self.uid, db=self.db)
        self.db.stores[snapshots.SHARES_COLLECTION][self.share_id]["visibility"] = "public"
        self.db.stores[snapshots.SHARES_COLLECTION][self.share_id]["status"] = "blocked"
        with self.assertRaisesRegex(ShareError, "active pages"):
            snapshots.request_share_indexing(self.share_id, self.uid, db=self.db)

    def test_already_indexed_returns_state_without_new_request(self):
        self.db.stores[snapshots.SHARES_COLLECTION][self.share_id]["indexed"] = True
        state = snapshots.request_share_indexing(self.share_id, self.uid, want=True, db=self.db)
        self.assertTrue(state["indexed"])
        self.assertFalse(state["index_requested"])
        self.assertNotIn("index_requested", self.db.stores[snapshots.SHARES_COLLECTION][self.share_id])

    def test_moderation_clears_open_request(self):
        snapshots.request_share_indexing(self.share_id, self.uid, want=True, db=self.db)
        snapshots.moderate_share(self.share_id, indexed=True, db=self.db)
        stored = self.db.stores[snapshots.SHARES_COLLECTION][self.share_id]
        self.assertTrue(stored["indexed"])
        self.assertFalse(stored["index_requested"])

    def test_indexing_request_route_requires_auth_and_owner(self):
        app = FastAPI()
        app.state.limiter = limiter
        app.include_router(share_router.router)
        client = TestClient(app)
        response = client.post("/api/share/%s/indexing-request" % self.share_id, json={})
        self.assertEqual(response.status_code, 401)
        with patch.object(share_router, "extract_id_token", return_value="token"), \
                patch.object(share_router, "verify_user_token", return_value=self.uid), \
                patch.object(share_router.snapshots, "request_share_indexing",
                             return_value={"indexed": False, "index_requested": True, "index_eligible": True}) as mocked:
            response = client.post("/api/share/%s/indexing-request" % self.share_id, json={"want": True})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["index_requested"])
        mocked.assert_called_once_with(self.share_id, self.uid, want=True)

    def test_sitemap_lastmod_prefers_last_watch_run(self):
        created = datetime(2026, 6, 1, tzinfo=timezone.utc)
        run = datetime(2026, 7, 15, tzinfo=timezone.utc)
        self.db.stores[snapshots.SHARES_COLLECTION][self.share_id].update({
            "indexed": True, "created_at": created, "last_watch_run_at": run,
        })
        urls = snapshots.list_indexed_share_urls(db=self.db)
        self.assertEqual(urls[0]["lastmod"], "2026-07-15")


class ApiRunPublishingServiceTests(unittest.TestCase):
    def setUp(self):
        self.db = FakeDb()
        self.uid = "admin-1"
        snapshots.invalidate_share_cache()

    def _run(self, **overrides):
        run = {
            "run_id": "a" * 32,
            "uid": self.uid,
            "status": "succeeded",
            "request": {
                "question": "Which energy storage technologies are most viable for a renewable electricity grid?"
            },
            "model_plan": {"consensus_model": "OpenAI"},
            "succeeded_at": datetime(2026, 7, 18, 10, 0, tzinfo=timezone.utc),
            "result": {
                "consensus_response": "Evidence-backed consensus. [S1] " + "x" * 650,
                "differences": "Minor differences.",
                "differences_data": {"claims": [], "differences": []},
                "model_answers": [
                    {
                        "provider": "OpenAI",
                        "model": "gpt-test",
                        "sources": [{"id": "S1", "title": "A", "url": "https://example.org/a"}],
                    },
                    {
                        "provider": "Anthropic",
                        "model": "claude-test",
                        "sources": [{"id": "S2", "title": "B", "url": "https://example.org/b"}],
                    },
                    {"provider": "Gemini", "model": "gemini-test", "sources": []},
                ],
            },
        }
        run.update(overrides)
        return run

    def test_api_run_publication_is_idempotent_and_skips_pending_results(self):
        run = self._run()
        first = snapshots.create_share_from_api_run(
            self.uid, run, db=self.db, consume_quota=lambda: True
        )
        second = snapshots.create_share_from_api_run(
            self.uid, run, db=self.db, consume_quota=lambda: self.fail("quota consumed twice")
        )

        self.assertTrue(first["created"])
        self.assertFalse(second["created"])
        self.assertEqual(first["share_id"], second["share_id"])
        self.assertTrue(snapshots.is_valid_share_id(first["share_id"]))
        share = self.db.stores[snapshots.SHARES_COLLECTION][first["share_id"]]
        self.assertEqual(share["source_api_run_id"], run["run_id"])
        self.assertEqual(share["visibility"], "public")
        self.assertTrue(share["index_eligible"])
        self.assertFalse(self.db.stores[snapshots.PENDING_COLLECTION])

    def test_api_run_requires_ownership_and_success(self):
        with self.assertRaisesRegex(ShareError, "Run not found"):
            snapshots.create_share_from_api_run(
                "other", self._run(), db=self.db, consume_quota=lambda: True
            )
        with self.assertRaisesRegex(ShareError, "Only succeeded"):
            snapshots.create_share_from_api_run(
                self.uid,
                self._run(status="running"),
                db=self.db,
                consume_quota=lambda: True,
            )

    def test_api_indexing_enforces_quality_dedup_and_audit(self):
        published = snapshots.create_share_from_api_run(
            self.uid, self._run(), db=self.db, consume_quota=lambda: True
        )
        share_id = published["share_id"]
        indexed = snapshots.set_api_share_indexing(
            share_id,
            self.uid,
            indexed=True,
            actor_key_id="f" * 64,
            db=self.db,
        )
        self.assertTrue(indexed["indexed"])
        self.assertEqual(indexed["review_source"], "consensus_api")
        self.assertEqual(indexed["reviewed_by_api_key_id"], "f" * 64)

        snapshots.set_api_share_indexing(
            share_id,
            self.uid,
            indexed=False,
            actor_key_id="f" * 64,
            db=self.db,
        )
        self.db.stores[snapshots.SHARES_COLLECTION][share_id]["index_eligible"] = False
        with self.assertRaisesRegex(ShareError, "quality gate"):
            snapshots.set_api_share_indexing(
                share_id,
                self.uid,
                indexed=True,
                actor_key_id="f" * 64,
                db=self.db,
            )

        current = self.db.stores[snapshots.SHARES_COLLECTION][share_id]
        current["index_eligible"] = True
        canonical_id = snapshots.generate_share_id()
        self.db.stores[snapshots.SHARES_COLLECTION][canonical_id] = {
            **current,
            "slug": "canonical",
            "indexed": True,
            "created_at": datetime(2026, 7, 1, tzinfo=timezone.utc),
            "source_api_run_id": "b" * 32,
        }
        with self.assertRaises(ShareError) as raised:
            snapshots.set_api_share_indexing(
                share_id,
                self.uid,
                indexed=True,
                actor_key_id="f" * 64,
                db=self.db,
            )
        self.assertEqual(raised.exception.code, "duplicate")
        self.assertEqual(raised.exception.details["canonical_share_id"], canonical_id)


class ShareSeoEnhancementTests(unittest.TestCase):
    """Scoreboard, datengeführte Description, dateModified und OG-Karte."""

    @classmethod
    def setUpClass(cls):
        app = FastAPI()
        app.state.limiter = limiter
        app.include_router(share_router.router)
        cls.client = TestClient(app)
        cls.share_id = snapshots.generate_share_id()

    def setUp(self):
        snapshots.invalidate_share_cache()
        for name in ("list_related_shares", "list_watch_history"):
            patcher = patch.object(share_router.snapshots, name, return_value=[])
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(share_router.watch_service, "get_public_watch_meta", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _share_doc(self, **overrides):
        data = make_pending()
        data.update({
            "status": "active",
            "slug": "wie-funktioniert-photosynthese-in-pflanzen",
            "differences_data": {
                "claims": [],
                "differences": [{
                    "claim": "Widerspruch",
                    "type": "contradiction",
                    "positions": [
                        {"stance": "A", "models": ["OpenAI"], "quote": ""},
                        {"stance": "B", "models": ["Grok"], "quote": ""},
                    ],
                    "verify": "",
                }],
                "best_model": "",
                "models_compared": ["OpenAI", "Gemini", "Grok"],
                "agreement": {"score": 71, "level": "medium", "model_count": 3,
                              "major_contradictions": 1, "minor_contradictions": 0, "emphases": 0},
            },
        })
        data.update(overrides)
        return data

    def _history(self):
        return [
            {"ts": datetime(2026, 6, 20, tzinfo=timezone.utc), "agreement_score": 71,
             "verdict": "", "changed": False, "severity": "", "change_summary": "", "opinion_map": None},
            {"ts": datetime(2026, 7, 10, 8, 30, tzinfo=timezone.utc), "agreement_score": 55,
             "verdict": "", "changed": True, "severity": "major", "change_summary": "Flip.", "opinion_map": None},
        ]

    def test_scoreboard_teaser_and_data_led_description(self):
        doc = self._share_doc()
        with patch.object(share_router.snapshots, "get_share", return_value=doc):
            response = self.client.get("/s/%s-%s" % (doc["slug"], self.share_id))
        body = response.text
        self.assertIn("share-scoreboard", body)
        self.assertIn("diffTeaser", body)
        self.assertIn("3 AI models answered independently", body)
        self.assertIn("agreement 71/100", body)
        self.assertIn("How it works", body)

    def test_date_modified_uses_last_watch_run(self):
        doc = self._share_doc()
        with patch.object(share_router.snapshots, "get_share", return_value=doc), \
                patch.object(share_router.snapshots, "list_watch_history", return_value=self._history()):
            response = self.client.get("/s/%s-%s" % (doc["slug"], self.share_id))
        body = response.text
        self.assertIn('"dateModified": "2026-07-10T08:30:00+00:00"', body)
        self.assertIn('"datePublished": "2026-06-11T10:00:00+00:00"', body)

    def test_follow_form_only_on_active_public_watch_pages(self):
        doc = self._share_doc()
        meta = {
            "status": "active", "interval": "weekly", "run_weekday": "", "run_time": "",
            "timezone": "", "last_run_at": None, "next_run_at": None, "created_at": None,
        }
        with patch.object(share_router.snapshots, "get_share", return_value=doc), \
                patch.object(share_router.snapshots, "list_watch_history", return_value=self._history()), \
                patch.object(share_router.watch_service, "get_public_watch_meta", return_value=meta):
            response = self.client.get("/s/%s-%s" % (doc["slug"], self.share_id))
        self.assertIn("shareFollowForm", response.text)

        with patch.object(share_router.snapshots, "get_share", return_value=doc):
            response = self.client.get("/s/%s-%s" % (doc["slug"], self.share_id))
        self.assertNotIn("shareFollowForm", response.text)

    def test_og_card_route_and_meta(self):
        if not share_router.og_image.is_available():
            self.skipTest("Pillow/Font nicht verfuegbar")
        doc = self._share_doc()
        with patch.object(share_router.snapshots, "get_share", return_value=doc):
            page = self.client.get("/s/%s-%s" % (doc["slug"], self.share_id))
            og = self.client.get("/s/%s-%s/og.png" % (doc["slug"], self.share_id))
        self.assertIn("/og.png", page.text)
        self.assertIn("summary_large_image", page.text)
        self.assertEqual(og.status_code, 200)
        self.assertEqual(og.headers["content-type"], "image/png")
        self.assertTrue(og.content.startswith(b"\x89PNG"))

    def test_og_card_404_for_private_pages(self):
        doc = self._share_doc(visibility="private")
        with patch.object(share_router.snapshots, "get_share", return_value=doc):
            response = self.client.get("/s/%s-%s/og.png" % (doc["slug"], self.share_id))
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
