import re
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.security import CustomSecurityMiddleware
from app.core.site import normalize_public_site_url
from app.services.consensus_pipeline import run_consensus_pipeline
from tests.frontend_order import loads_before


ROOT = Path(__file__).resolve().parents[1]


def source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_public_site_origin_is_neutral_validated_core_configuration():
    assert normalize_public_site_url(None) == "https://www.consens.io"
    assert normalize_public_site_url("https://preview.example/") == "https://preview.example"
    for invalid in ("preview.example", "ftp://preview.example", "https://user@preview.example", "https://preview.example/path"):
        with pytest.raises(RuntimeError):
            normalize_public_site_url(invalid)
    service_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "app/services").rglob("*.py")
    )
    assert "from app.api.routers.pages import SITE_URL" not in service_sources


def test_neutral_pipeline_owns_fanout_synthesis_parsing_and_scoring():
    calls = []

    def provider(provider, model, question, keys, is_pro, deep_think):
        calls.append((provider, model, deep_think))
        return {"text": f"{provider} answer", "sources": []}

    def synthesize(*args, **kwargs):
        assert args[0] == "Question"
        assert args[1] == "openai answer"
        assert args[2] == "mistral answer"
        return "Consensus"

    def judge(*args, **kwargs):
        return "Differences", {
            "models_compared": ["OpenAI", "Mistral"],
            "claims": [{"agree": ["OpenAI", "Mistral"], "dissent": []}],
            "differences": [],
        }

    result = run_consensus_pipeline(
        question="Question",
        provider_models={"openai": "o", "mistral": "m"},
        consensus_model="OpenAI",
        keys={},
        is_pro=False,
        provider_call=provider,
        synthesize=synthesize,
        judge=judge,
    )
    assert calls == [("openai", "o", False), ("mistral", "m", False)]
    assert result["consensus_response"] == "Consensus"
    assert result["differences_data"]["agreement"]["score"] == 75
    assert [item["provider"] for item in result["model_answers"]] == ["OpenAI", "Mistral"]


def test_neutral_pipeline_can_select_the_first_successful_provider_as_engine():
    def provider(provider, model, question, keys, is_pro, deep_think):
        if provider == "openai":
            raise RuntimeError("transport failed")
        return {"text": f"{provider} answer", "sources": []}

    def synthesize(*args, **kwargs):
        assert args[8] == "Mistral"
        return "Consensus"

    result = run_consensus_pipeline(
        question="Question",
        provider_models={"openai": "o", "mistral": "m", "gemini": "g"},
        consensus_model=lambda answers: next(
            {"mistral": "Mistral", "gemini": "Gemini"}[provider]
            for provider in ("mistral", "gemini") if provider in answers
        ),
        keys={},
        is_pro=False,
        provider_call=provider,
        synthesize=synthesize,
        judge=lambda *args, **kwargs: (
            "Differences",
            {"models_compared": ["Mistral", "Gemini"], "claims": [], "differences": []},
        ),
    )
    assert [item["provider"] for item in result["model_answers"]] == ["Mistral", "Gemini"]


def test_all_server_product_paths_delegate_to_neutral_pipeline():
    api = source("app/services/api_consensus_runner.py")
    watch = source("app/services/watch_scheduler.py")
    topic = source("app/services/topic_runner.py")
    topic_pipeline = source("app/services/topic_pipeline.py")
    chat = source("app/api/routers/chat.py")
    assert "run_consensus_pipeline(" in api
    assert "run_consensus_pipeline(" in watch
    assert "analyze_provider_answers(" in chat
    assert "topic_pipeline.execute_topic" in topic
    assert "watch_scheduler" not in topic
    assert "run_consensus_pipeline(" in topic_pipeline
    engine = source("app/services/llm/consensus_engine.py")
    assert "from app.services.llm.consensus_parsing import" in engine
    assert "from app.services.llm.consensus_scoring import" in engine


def test_cross_module_frontend_state_has_enforced_owners_and_no_direct_writers():
    state = source("static/js/app-state.js")
    for owner in ("run", "evidence", "consensus", "share", "userTier", "runUi"):
        assert f'owner: "{owner}"' in state
    assert "Direct write to window.${key} is forbidden" in state
    direct_write = re.compile(
        r"window\.(?:lastQuestion|currentEvidenceSources|consensusCitationMeta|"
        r"lastShareResultId|isUserPro|currentMaxLimit|currentDeepLimit|spinnerHTML)\s*=(?!=)"
    )
    offenders = []
    for path in (ROOT / "static").rglob("*.js"):
        if path.name == "app-state.js":
            continue
        if direct_write.search(path.read_text(encoding="utf-8")):
            offenders.append(path.relative_to(ROOT).as_posix())
    assert offenders == []
    assert loads_before("app-state.js", "firebase.js")
    assert loads_before("consensus-anchor.js", "consensus-insights.js")


def test_privileged_app_and_admin_templates_are_external_script_surfaces():
    inline_script = re.compile(r"<script(?![^>]*\bsrc=)[^>]*>", re.I)
    inline_handler = re.compile(r"\son[a-z]+\s*=", re.I)
    for name in (
        "templates/index.html",
        "templates/admin.html",
        "templates/admin_benchmark.html",
    ):
        html = source(name)
        assert not inline_script.search(html)
        assert not inline_handler.search(html)
        assert "<style" not in html
        assert not re.search(r"\sstyle\s*=", html, re.I)
    admin = source("templates/admin.html")
    assert len(admin.splitlines()) < 700
    assert "/static/css/admin.css?v=20260812-auditfix" in admin
    assert "/static/js/admin.js?v=20260817-memoryedit1" in admin
    assert "createAdminClient" in source("static/js/admin.js")
    benchmark = source("templates/admin_benchmark.html")
    assert 'id="adminBootstrapConfig"' in benchmark
    for attribute, value in (
        ("firebase-api-key", "firebase_api_key"),
        ("firebase-auth-domain", "firebase_auth_domain"),
        ("firebase-project-id", "firebase_project_id"),
        ("firebase-storage-bucket", "firebase_storage_bucket"),
        ("firebase-messaging-sender-id", "firebase_messaging_sender_id"),
        ("firebase-app-id", "firebase_app_id"),
    ):
        assert f'data-{attribute}="{{{{ {value} | e }}}}"' in benchmark
    assert "window.FIREBASE_CONFIG" not in benchmark
    assert "/static/css/admin-benchmark.css?v=20260812-auditfix" in benchmark
    assert "/static/js/admin-benchmark.js?v=20260812-auditfix" in benchmark
    assert benchmark.index("/static/js/admin-config.js") < benchmark.index(
        "/static/js/admin-benchmark.js"
    )
    benchmark_js = source("static/js/admin-benchmark.js")
    assert 'import { initializeApp } from "https://www.gstatic.com/firebasejs/' in benchmark_js
    assert "fill.style.width" in benchmark_js
    assert "{ style:" not in benchmark_js


def test_app_and_all_admin_pages_receive_strict_script_csp():
    app = FastAPI()
    app.add_middleware(CustomSecurityMiddleware)

    @app.get("/{path:path}")
    def page(path: str):
        return {"path": path}

    client = TestClient(app)
    for path in ("/app", "/app/watches", "/admin", "/admin/benchmark", "/admin/nested/page"):
        csp = client.get(path).headers["content-security-policy"]
        script_src = csp.split("script-src", 1)[1].split(";", 1)[0]
        assert "'unsafe-inline'" not in script_src
    for path in ("/", "/administrator"):
        public_script_src = client.get(path).headers["content-security-policy"].split(
            "script-src", 1
        )[1].split(";", 1)[0]
        assert "'unsafe-inline'" in public_script_src


def test_consensus_visuals_are_shared_and_dead_dom_contracts_are_gone():
    shared = source("static/css/components-consensus-visuals.css")
    assert ".consensus-verdict" in shared
    assert ".claim-badge" in shared
    assert ".cx-claim.is-major" in shared
    # Die Punkte neben den Marken sind seit 2026-08-15 ersatzlos weg
    # (User-Vorgabe); nichts darf sie unbemerkt wieder einfuehren.
    assert ".cx-marker" not in shared
    assert "components-consensus-visuals.css?v=20260817-graycount1" in source("static/css/landing.css")
    assert "components-consensus-visuals.css?v=20260817-graycount1" in source("static/css/components-consensus-insights.css")
    all_static = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "static").rglob("*")
        if path.is_file() and path.suffix in {".js", ".css"}
    )
    for dead_id in ('getElementById("consensusButton")', 'getElementById("toggleAllButton")',
                    'getElementById("toggleApiTest")', 'getElementById("apiTestArea")'):
        assert dead_id not in all_static
