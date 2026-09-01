import base64
import io
import unittest
import zipfile
from unittest.mock import MagicMock, patch

from fastapi import HTTPException

from app.services.llm.attachments import (
    DOCX_MIME,
    IMAGE_MAX_EDGE,
    IMAGE_MAX_PIXELS,
    MAX_ATTACHMENT_BYTES,
    MAX_ATTACHMENT_BASE64_CHARS,
    MAX_ATTACHMENT_TOTAL_BYTES,
    MAX_ATTACHMENTS,
    MAX_IMAGE_BYTES,
    PDF_NATIVE_MAX_BYTES,
    TEXT_MIME,
    build_attachment_question_suffix,
    extract_docx_text,
    native_attachments_for_provider,
    parse_attachments,
)
from app.services.llm.engines import build_provider_payload


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 32
WEBP_BYTES = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 32
PDF_BYTES = b"%PDF-1.7\n%fake-pdf-for-tests\n" + b"\x00" * 32
TXT_BYTES = "Notes: hello wörld\nSecond line".encode("utf-8")


def make_docx(paragraphs):
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    xml = (
        '<?xml version="1.0"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body></w:document>"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", xml)
    return buffer.getvalue()


DOCX_BYTES = make_docx(["First paragraph.", "Second paragraph."])


def b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def make_attachment(name="file.png", raw=PNG_BYTES):
    return {"name": name, "data": b64(raw)}


class ParseAttachmentsTests(unittest.TestCase):
    def test_no_attachments_returns_empty_list(self):
        self.assertEqual(parse_attachments({}, attachments_allowed=False), [])
        self.assertEqual(parse_attachments({"attachments": []}, attachments_allowed=True), [])

    def test_attachments_are_refused_below_plus(self):
        data = {"attachments": [make_attachment()]}
        with self.assertRaises(HTTPException) as ctx:
            parse_attachments(data, attachments_allowed=False)
        self.assertEqual(ctx.exception.status_code, 403)

    def test_valid_types_are_sniffed_from_magic_bytes(self):
        cases = [
            ("doc.pdf", PDF_BYTES, "application/pdf"),
            ("doc.docx", DOCX_BYTES, DOCX_MIME),
            ("notes.txt", TXT_BYTES, TEXT_MIME),
            ("img.png", PNG_BYTES, "image/png"),
            ("img.jpg", JPEG_BYTES, "image/jpeg"),
            ("img.webp", WEBP_BYTES, "image/webp"),
        ]
        for name, raw, expected_mime in cases:
            with self.subTest(name=name):
                parsed = parse_attachments(
                    {"attachments": [make_attachment(name, raw)]}, attachments_allowed=True
                )
                self.assertEqual(parsed[0]["mime"], expected_mime)
                self.assertEqual(parsed[0]["raw"], raw)

    def test_unsupported_type_is_rejected_even_with_image_extension(self):
        data = {"attachments": [make_attachment("evil.png", b"MZ\x90\x00" + b"\x00" * 32)]}
        with self.assertRaises(HTTPException) as ctx:
            parse_attachments(data, attachments_allowed=True)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_generic_zip_is_not_accepted_as_docx(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("other.xml", "<x/>")
        data = {"attachments": [make_attachment("fake.docx", buffer.getvalue())]}
        with self.assertRaises(HTTPException) as ctx:
            parse_attachments(data, attachments_allowed=True)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_docx_text_extraction_joins_paragraphs(self):
        self.assertEqual(
            extract_docx_text(DOCX_BYTES),
            "First paragraph.\nSecond paragraph.",
        )

    def test_invalid_base64_is_rejected(self):
        data = {"attachments": [{"name": "x.png", "data": "not base64!!"}]}
        with self.assertRaises(HTTPException) as ctx:
            parse_attachments(data, attachments_allowed=True)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_size_limit_is_enforced(self):
        big = b"\x89PNG\r\n\x1a\n" + b"\x00" * MAX_ATTACHMENT_BYTES
        data = {"attachments": [make_attachment("big.png", big)]}
        with self.assertRaises(HTTPException) as ctx:
            parse_attachments(data, attachments_allowed=True)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_encoded_size_limit_is_checked_before_base64_decode(self):
        data = {
            "attachments": [
                {"name": "big.png", "data": "A" * (MAX_ATTACHMENT_BASE64_CHARS + 1)}
            ]
        }
        with patch("app.services.llm.attachments.base64.b64decode") as decode:
            with self.assertRaises(HTTPException) as ctx:
                parse_attachments(data, attachments_allowed=True)
        self.assertEqual(ctx.exception.status_code, 400)
        decode.assert_not_called()

    def test_docx_zip_bomb_ratio_is_rejected_during_parse(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("word/document.xml", b"A" * (2 * 1024 * 1024))
        data = {"attachments": [make_attachment("bomb.docx", buffer.getvalue())]}

        with self.assertRaises(HTTPException) as ctx:
            parse_attachments(data, attachments_allowed=True)

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("safe Word document", ctx.exception.detail)

    def test_docx_traversal_entry_is_rejected(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("word/document.xml", "<w:document/>")
            archive.writestr("../secret.txt", "secret")
        data = {"attachments": [make_attachment("unsafe.docx", buffer.getvalue())]}

        with self.assertRaises(HTTPException) as ctx:
            parse_attachments(data, attachments_allowed=True)

        self.assertEqual(ctx.exception.status_code, 400)

    def test_attachment_count_limit_is_enforced(self):
        data = {"attachments": [make_attachment() for _ in range(MAX_ATTACHMENTS + 1)]}
        with self.assertRaises(HTTPException) as ctx:
            parse_attachments(data, attachments_allowed=True)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_data_url_prefix_is_tolerated(self):
        data = {"attachments": [{"name": "x.png", "data": "data:image/png;base64," + b64(PNG_BYTES)}]}
        parsed = parse_attachments(data, attachments_allowed=True)
        self.assertEqual(parsed[0]["mime"], "image/png")


def real_png(width, height, color=(30, 90, 200), mode="RGB"):
    """Ein echtes Bild -- die Fixtures oben sind nur Dateikoepfe, an denen
    Pillow scheitert (und scheitern soll)."""
    from PIL import Image

    image = Image.new(mode, (width, height), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class ImageShrinkTests(unittest.TestCase):
    """Ein Anhang geht an bis zu sechs Familien gleichzeitig raus. Was hier
    durchrutscht, kostet also sechsmal."""

    def parsed(self, raw, name="photo.png"):
        return parse_attachments(
            {"attachments": [make_attachment(name, raw)]}, attachments_allowed=True
        )[0]

    def test_oversized_image_is_shrunk_to_provider_size(self):
        att = self.parsed(real_png(3000, 2000))

        from PIL import Image

        with Image.open(io.BytesIO(att["raw"])) as image:
            self.assertLessEqual(max(image.size), IMAGE_MAX_EDGE)
        self.assertEqual(att["mime"], "image/jpeg")
        self.assertLessEqual(len(att["raw"]), MAX_IMAGE_BYTES)

    def test_shrunk_image_data_matches_its_bytes(self):
        att = self.parsed(real_png(3000, 2000))
        self.assertEqual(base64.b64decode(att["data"]), att["raw"])

    def test_server_side_jpeg_conversion_updates_the_filename(self):
        att = self.parsed(real_png(3000, 2000), "photo.png")
        self.assertEqual(att["name"], "photo.jpg")

    def test_pixel_budget_is_checked_before_image_load(self):
        image = MagicMock()
        image.width = IMAGE_MAX_PIXELS + 1
        image.height = 1
        image.size = (image.width, image.height)

        with patch("PIL.Image.open", return_value=image):
            with self.assertRaises(HTTPException) as ctx:
                self.parsed(PNG_BYTES, "huge.png")

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("too many pixels", ctx.exception.detail)
        image.load.assert_not_called()
        image.close.assert_called_once()

    def test_small_image_is_left_untouched(self):
        raw = real_png(400, 300)
        att = self.parsed(raw, "screenshot.png")
        self.assertEqual(att["mime"], "image/png")
        self.assertEqual(att["raw"], raw)

    def test_transparency_is_flattened_onto_white(self):
        raw = real_png(3000, 2000, color=(0, 0, 0, 0), mode="RGBA")
        att = self.parsed(raw, "logo.png")

        from PIL import Image

        with Image.open(io.BytesIO(att["raw"])) as image:
            pixel = image.convert("RGB").getpixel((0, 0))
        self.assertTrue(all(channel > 240 for channel in pixel), pixel)

    def test_undecodable_image_stays_untouched(self):
        att = self.parsed(PNG_BYTES, "broken.png")
        self.assertEqual(att["raw"], PNG_BYTES)
        self.assertEqual(att["mime"], "image/png")

    def test_total_size_limit_is_enforced(self):
        half = MAX_ATTACHMENT_TOTAL_BYTES // 2 + 1024
        pdf = PDF_BYTES + b"\x00" * half
        data = {
            "attachments": [
                make_attachment("a.pdf", pdf),
                make_attachment("b.pdf", pdf),
            ]
        }
        with self.assertRaises(HTTPException) as ctx:
            parse_attachments(data, attachments_allowed=True)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("too large together", ctx.exception.detail)


class LargePdfRoutingTests(unittest.TestCase):
    """Ein grosses PDF nativ zu schicken heisst: die kompletten Bytes an jede
    PDF-faehige Familie. Der extrahierte Text tut es auch."""

    def big_pdf(self):
        return parse_attachments(
            {
                "attachments": [
                    make_attachment(
                        "contract.pdf", PDF_BYTES + b"\x00" * PDF_NATIVE_MAX_BYTES
                    )
                ]
            },
            attachments_allowed=True,
        )

    def test_large_pdf_goes_out_as_text_instead_of_file(self):
        attachments = self.big_pdf()
        with patch(
            "app.services.llm.attachments.extract_pdf_text", return_value="Clause 7 applies."
        ):
            self.assertEqual(native_attachments_for_provider(attachments, "openai"), [])
            suffix = build_attachment_question_suffix(attachments, "openai")
        self.assertIn("Clause 7 applies.", suffix)

    def test_large_pdf_without_extractable_text_stays_native(self):
        attachments = self.big_pdf()
        with patch("app.services.llm.attachments.extract_pdf_text", return_value=None):
            self.assertEqual(native_attachments_for_provider(attachments, "openai"), attachments)
            self.assertEqual(build_attachment_question_suffix(attachments, "openai"), "")

    def test_small_pdf_still_goes_out_natively(self):
        attachments = parse_attachments(
            {"attachments": [make_attachment("short.pdf", PDF_BYTES)]},
            attachments_allowed=True,
        )
        self.assertEqual(native_attachments_for_provider(attachments, "openai"), attachments)


class AttachmentPayloadTests(unittest.TestCase):
    def parsed(self, raw, name):
        return parse_attachments({"attachments": [make_attachment(name, raw)]}, attachments_allowed=True)

    def test_openai_image_becomes_openrouter_image_url_block(self):
        request = build_provider_payload(
            "openai",
            question="what is in this image?",
            system_prompt="system",
            max_output_tokens=128,
            attachments=self.parsed(PNG_BYTES, "img.png"),
        )
        content = request["payload"]["messages"][1]["content"]
        self.assertEqual([block["type"] for block in content], ["text", "image_url"])
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_openai_pdf_becomes_input_file_block(self):
        request = build_provider_payload(
            "openai",
            question="summarize",
            system_prompt="system",
            max_output_tokens=128,
            attachments=self.parsed(PDF_BYTES, "doc.pdf"),
        )
        blocks = request["payload"]["messages"][1]["content"]
        self.assertEqual([block["type"] for block in blocks], ["text", "file"])
        self.assertEqual(blocks[1]["file"]["filename"], "doc.pdf")

    def test_anthropic_gets_image_and_document_blocks(self):
        request = build_provider_payload(
            "anthropic",
            question="summarize",
            system_prompt="system",
            max_output_tokens=128,
            attachments=self.parsed(PDF_BYTES, "doc.pdf") + self.parsed(PNG_BYTES, "img.png"),
        )
        content = request["payload"]["messages"][1]["content"]
        types = [block["type"] for block in content]
        self.assertEqual(types, ["text", "file", "image_url"])

    def test_gemini_gets_openrouter_file_block(self):
        request = build_provider_payload(
            "gemini",
            question="summarize",
            system_prompt="system",
            max_output_tokens=128,
            attachments=self.parsed(PDF_BYTES, "doc.pdf"),
        )
        content = request["payload"]["messages"][1]["content"]
        self.assertEqual(content[1]["type"], "file")
        self.assertTrue(content[1]["file"]["file_data"].startswith("data:application/pdf;base64,"))

    def test_grok_image_native_but_pdf_falls_back_to_text(self):
        request = build_provider_payload(
            "grok",
            question="summarize",
            system_prompt="system",
            max_output_tokens=128,
            attachments=self.parsed(PNG_BYTES, "img.png") + self.parsed(PDF_BYTES, "doc.pdf"),
        )
        content = request["payload"]["messages"][1]["content"]
        types = [block["type"] for block in content]
        self.assertIn("image_url", types)
        self.assertNotIn("file", types)
        text_block = next(b for b in content if b["type"] == "text")
        self.assertIn("doc.pdf", text_block["text"])

    def test_text_only_providers_get_fallback_notes(self):
        for provider in ("mistral", "deepseek"):
            with self.subTest(provider=provider):
                request = build_provider_payload(
                    provider,
                    question="summarize",
                    system_prompt="system",
                    max_output_tokens=128,
                    attachments=self.parsed(PNG_BYTES, "img.png"),
                )
                payload_text = str(request["payload"])
                self.assertIn("img.png", payload_text)
                self.assertNotIn(b64(PNG_BYTES), payload_text)

    def test_docx_and_text_fall_back_to_extracted_text_for_all_providers(self):
        request = build_provider_payload(
            "openai",
            question="summarize",
            system_prompt="system",
            max_output_tokens=128,
            attachments=self.parsed(DOCX_BYTES, "doc.docx") + self.parsed(TXT_BYTES, "notes.txt"),
        )
        # Kein nativer Content-Block: alles landet als Text-Suffix in der Frage.
        payload_input = request["payload"]["messages"][1]["content"]
        self.assertIsInstance(payload_input, str)
        self.assertIn("First paragraph.", payload_input)
        self.assertIn("Second line", payload_input)

    def test_no_attachments_keeps_payload_shape_unchanged(self):
        request = build_provider_payload(
            "openai",
            question="plain question",
            system_prompt="system",
            max_output_tokens=128,
        )
        self.assertEqual(request["payload"]["messages"][1]["content"], "plain question")


class BookmarkAttachmentMetaTests(unittest.TestCase):
    def setUp(self):
        from app.api.routers.bookmarks import sanitize_attachment_meta
        self.sanitize = sanitize_attachment_meta

    def test_missing_field_returns_none_so_merge_keeps_existing(self):
        self.assertIsNone(self.sanitize(None))

    def test_file_data_is_never_stored(self):
        result = self.sanitize([
            {"name": "doc.pdf", "mime": "application/pdf", "size": 1234, "data": "JVBERi0xLjc="},
        ])
        self.assertEqual(result, [{"name": "doc.pdf", "mime": "application/pdf", "size": 1234}])
        self.assertNotIn("data", result[0])

    def test_invalid_entries_are_dropped(self):
        result = self.sanitize([
            {"name": "", "mime": "application/pdf"},
            {"name": "x.exe", "mime": "application/octet-stream"},
            "not-a-dict",
            {"name": "ok.png", "mime": "image/png", "size": "not-a-number"},
        ])
        self.assertEqual(result, [{"name": "ok.png", "mime": "image/png", "size": 0}])

    def test_list_is_capped_and_non_list_becomes_empty(self):
        many = [{"name": f"f{i}.png", "mime": "image/png", "size": 1} for i in range(5)]
        self.assertEqual(len(self.sanitize(many)), 2)
        self.assertEqual(self.sanitize("garbage"), [])

    def test_browser_type_variants_survive_as_their_canonical_type(self):
        """Der Lauf entscheidet nach den BYTES, die Metadaten kamen als Angabe.

        Chrome meldet dieselbe Textdatei je nach System als "text/markdown"
        oder "text/csv". Beides fiel aus der Allowlist heraus - die Datei ging
        also mit der Frage raus, tauchte im gespeicherten Chat aber nie auf.
        """
        result = self.sanitize([
            {"name": "rows.csv", "mime": "text/csv", "size": 900},
            {"name": "notes.md", "mime": "text/markdown; charset=utf-8", "size": 12},
        ])
        self.assertEqual(result, [
            {"name": "rows.csv", "mime": TEXT_MIME, "size": 900},
            {"name": "notes.md", "mime": TEXT_MIME, "size": 12},
        ])


if __name__ == "__main__":
    unittest.main()
