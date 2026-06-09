import base64
import unittest

from fastapi import HTTPException

from app.services.llm.attachments import (
    MAX_ATTACHMENT_BYTES,
    MAX_ATTACHMENTS,
    parse_attachments,
)
from app.services.llm.engines import build_provider_payload


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 32
WEBP_BYTES = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 32
PDF_BYTES = b"%PDF-1.7\n%fake-pdf-for-tests\n" + b"\x00" * 32


def b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def make_attachment(name="file.png", raw=PNG_BYTES):
    return {"name": name, "data": b64(raw)}


class ParseAttachmentsTests(unittest.TestCase):
    def test_no_attachments_returns_empty_list(self):
        self.assertEqual(parse_attachments({}, is_pro=False), [])
        self.assertEqual(parse_attachments({"attachments": []}, is_pro=True), [])

    def test_attachments_require_pro(self):
        data = {"attachments": [make_attachment()]}
        with self.assertRaises(HTTPException) as ctx:
            parse_attachments(data, is_pro=False)
        self.assertEqual(ctx.exception.status_code, 403)

    def test_valid_types_are_sniffed_from_magic_bytes(self):
        cases = [
            ("doc.pdf", PDF_BYTES, "application/pdf"),
            ("img.png", PNG_BYTES, "image/png"),
            ("img.jpg", JPEG_BYTES, "image/jpeg"),
            ("img.webp", WEBP_BYTES, "image/webp"),
        ]
        for name, raw, expected_mime in cases:
            with self.subTest(name=name):
                parsed = parse_attachments(
                    {"attachments": [make_attachment(name, raw)]}, is_pro=True
                )
                self.assertEqual(parsed[0]["mime"], expected_mime)
                self.assertEqual(parsed[0]["raw"], raw)

    def test_unsupported_type_is_rejected_even_with_image_extension(self):
        data = {"attachments": [make_attachment("evil.png", b"MZ\x90\x00" + b"\x00" * 32)]}
        with self.assertRaises(HTTPException) as ctx:
            parse_attachments(data, is_pro=True)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_invalid_base64_is_rejected(self):
        data = {"attachments": [{"name": "x.png", "data": "not base64!!"}]}
        with self.assertRaises(HTTPException) as ctx:
            parse_attachments(data, is_pro=True)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_size_limit_is_enforced(self):
        big = b"\x89PNG\r\n\x1a\n" + b"\x00" * MAX_ATTACHMENT_BYTES
        data = {"attachments": [make_attachment("big.png", big)]}
        with self.assertRaises(HTTPException) as ctx:
            parse_attachments(data, is_pro=True)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_attachment_count_limit_is_enforced(self):
        data = {"attachments": [make_attachment() for _ in range(MAX_ATTACHMENTS + 1)]}
        with self.assertRaises(HTTPException) as ctx:
            parse_attachments(data, is_pro=True)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_data_url_prefix_is_tolerated(self):
        data = {"attachments": [{"name": "x.png", "data": "data:image/png;base64," + b64(PNG_BYTES)}]}
        parsed = parse_attachments(data, is_pro=True)
        self.assertEqual(parsed[0]["mime"], "image/png")


class AttachmentPayloadTests(unittest.TestCase):
    def parsed(self, raw, name):
        return parse_attachments({"attachments": [make_attachment(name, raw)]}, is_pro=True)

    def test_openai_image_becomes_input_image_block(self):
        request = build_provider_payload(
            "openai",
            question="what is in this image?",
            system_prompt="system",
            max_output_tokens=128,
            attachments=self.parsed(PNG_BYTES, "img.png"),
        )
        message = request["payload"]["input"][0]
        types = [block["type"] for block in message["content"]]
        self.assertIn("input_image", types)
        self.assertIn("input_text", types)

    def test_openai_pdf_becomes_input_file_block(self):
        request = build_provider_payload(
            "openai",
            question="summarize",
            system_prompt="system",
            max_output_tokens=128,
            attachments=self.parsed(PDF_BYTES, "doc.pdf"),
        )
        blocks = request["payload"]["input"][0]["content"]
        self.assertEqual(blocks[0]["type"], "input_file")
        self.assertEqual(blocks[0]["filename"], "doc.pdf")

    def test_anthropic_gets_image_and_document_blocks(self):
        request = build_provider_payload(
            "anthropic",
            question="summarize",
            system_prompt="system",
            max_output_tokens=128,
            attachments=self.parsed(PDF_BYTES, "doc.pdf") + self.parsed(PNG_BYTES, "img.png"),
        )
        content = request["payload"]["messages"][0]["content"]
        types = [block["type"] for block in content]
        self.assertEqual(types, ["document", "image", "text"])

    def test_gemini_gets_inline_data_parts(self):
        request = build_provider_payload(
            "gemini",
            question="summarize",
            system_prompt="system",
            max_output_tokens=128,
            attachments=self.parsed(PDF_BYTES, "doc.pdf"),
        )
        parts = request["payload"]["contents"][0]["parts"]
        self.assertIn("inline_data", parts[0])
        self.assertEqual(parts[0]["inline_data"]["mime_type"], "application/pdf")
        self.assertIn("text", parts[-1])

    def test_grok_image_native_but_pdf_falls_back_to_text(self):
        request = build_provider_payload(
            "grok",
            question="summarize",
            system_prompt="system",
            max_output_tokens=128,
            attachments=self.parsed(PNG_BYTES, "img.png") + self.parsed(PDF_BYTES, "doc.pdf"),
        )
        message = request["payload"]["input"][0]
        types = [block["type"] for block in message["content"]]
        self.assertIn("input_image", types)
        self.assertNotIn("input_file", types)
        text_block = next(b for b in message["content"] if b["type"] == "input_text")
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

    def test_no_attachments_keeps_payload_shape_unchanged(self):
        request = build_provider_payload(
            "openai",
            question="plain question",
            system_prompt="system",
            max_output_tokens=128,
        )
        self.assertEqual(request["payload"]["input"], "plain question")


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


if __name__ == "__main__":
    unittest.main()
