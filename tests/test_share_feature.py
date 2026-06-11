import unittest
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.rate_limit import limiter
from app.api.routers import share as share_router
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
    def __init__(self, store, field, op, value):
        self._store = store
        self._field = field
        self._value = value

    def stream(self):
        for key, data in list(self._store.items()):
            if data.get(self._field) == self._value:
                yield FakeQueryDoc(self._store, key)


class FakeCollection:
    def __init__(self, store):
        self._store = store

    def document(self, doc_id):
        return FakeDocRef(self._store, doc_id)

    def where(self, field, op, value):
        return FakeQuery(self._store, field, op, value)


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
                {"claim": "c", "type": "contradiction", "positions": [{"stance": "s", "models": ["Grok"], "quote": "q"}], "verify": "v"},
                {"claim": "no positions", "positions": []},
            ],
            "best_model": "OpenAI",
            "models_compared": ["OpenAI", "Grok"],
            "internal_cost": 1.23,
        }
        result = snapshots.sanitize_differences_data(data)
        self.assertNotIn("internal_cost", result)
        self.assertEqual(len(result["claims"][0]["anchor"]), 500)
        self.assertEqual(result["claims"][0]["agree"], ["OpenAI"])
        self.assertNotIn("secret", result["claims"][0]["dissent"][0])
        self.assertEqual(len(result["differences"]), 1)
        self.assertIsNone(snapshots.sanitize_differences_data("not a dict"))

    def test_included_models_order_and_labels(self):
        result = snapshots.build_included_models(
            ["Grok", "OpenAI", "Anthropic"],
            {"OpenAI": "gpt-test", "Grok": "", "Unknown": "x", "Anthropic": 42},
        )
        self.assertEqual(result, ["OpenAI: gpt-test", "Anthropic Claude", "Grok"])


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
        self.assertIn("[S1]", html)

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

    def test_list_shares_for_owner_is_whitelisted(self):
        result_id = self._store_pending()
        created = snapshots.create_share_from_pending(self.uid, result_id,
                                                      db=self.db, consume_quota=lambda: True)
        other_id = snapshots.generate_share_id()
        self.db.stores[snapshots.SHARES_COLLECTION][other_id] = {"owner_uid": "someone-else", "question": "x"}

        shares = snapshots.list_shares_for_owner(self.uid, db=self.db)
        self.assertEqual(len(shares), 1)
        self.assertEqual(shares[0]["share_id"], created["share_id"])
        self.assertEqual(set(shares[0].keys()), {"share_id", "path", "question", "status", "created_at"})


class SharePageRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = FastAPI()
        app.state.limiter = limiter
        app.include_router(share_router.router)
        cls.client = TestClient(app)
        cls.share_id = snapshots.generate_share_id()

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


if __name__ == "__main__":
    unittest.main()
