"""Browser regressions for the Phase-4 frontend state races."""

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from playwright.sync_api import expect


FIREBASE_APP_STUB = """
export function initializeApp(config) { return { config }; }
"""

FIRESTORE_STUB = """
export function getFirestore() { return {}; }
export function doc() { return {}; }
export async function setDoc() {}
export async function getDoc() { return { exists: () => false, data: () => ({}) }; }
export function increment(value) { return value; }
export async function addDoc() { return {}; }
export async function deleteDoc() {}
"""

FIREBASE_AUTH_STUB = """
const callbacks = [];
const auth = { currentUser: null };

function makeUser(uid) {
  return {
    uid,
    email: uid + "@example.invalid",
    emailVerified: true,
    displayName: uid,
    getIdToken: async () => "token-" + uid,
    reload: async () => {},
  };
}

window.__switchE2EUser = async function (uid) {
  auth.currentUser = uid ? makeUser(uid) : null;
  for (const callback of callbacks) await callback(auth.currentUser);
};

export class GoogleAuthProvider { setCustomParameters() {} }
export const browserLocalPersistence = {};
export const browserSessionPersistence = {};
export function getAuth() { return auth; }
export async function setPersistence() {}
export function onIdTokenChanged(_auth, callback) {
  callbacks.push(callback);
  queueMicrotask(() => window.__switchE2EUser(window.__E2E_INITIAL_UID));
  return () => {};
}
export function onAuthStateChanged(_auth, callback) { return onIdTokenChanged(_auth, callback); }
export async function signOut() { await window.__switchE2EUser(null); }
export async function signInWithEmailAndPassword() { return { user: auth.currentUser }; }
export async function signInWithPopup() { return { user: auth.currentUser }; }
export async function signInWithRedirect() {}
export async function getRedirectResult() { return null; }
export async function sendPasswordResetEmail() {}
export async function sendEmailVerification() {}
"""


@pytest.fixture(scope="session")
def phase4_server():
    """Read-only frontend server for fully mocked Phase-4 browser races.

    No request in this module reaches Firestore. E2E mode still enforces the
    loopback emulator target and disables every lifespan writer, but Java is
    intentionally not a prerequisite for these browser-only regressions.
    """

    port = 8033
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env.update({
        "E2E_TEST_MODE": "1",
        "FIRESTORE_EMULATOR_HOST": "127.0.0.1:8085",
        "GOOGLE_CLOUD_PROJECT": "demo-consensio-e2e",
        "GCLOUD_PROJECT": "demo-consensio-e2e",
        "FIREBASE_PROJECT_ID": "demo-consensio-e2e",
        "MOCK_LLM": "1",
        "MOCK_AUTH": "1",
        "DISABLE_RATE_LIMIT": "1",
    })
    env.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--port", str(port), "--log-level", "warning"],
        cwd=str(Path(__file__).resolve().parents[2]),
        env=env,
    )
    deadline = time.monotonic() + 45
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"Phase-4 frontend server exited with {proc.returncode}")
            try:
                with urllib.request.urlopen(base_url + "/app", timeout=2) as response:
                    if response.status < 500:
                        break
            except (urllib.error.URLError, OSError):
                time.sleep(0.25)
        else:
            raise RuntimeError("Phase-4 frontend server did not become ready")
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _json(route, payload, status=200):
    route.fulfill(
        status=status,
        content_type="application/json",
        body=json.dumps(payload),
    )


def _real_firebase_page(
    browser, app_server, initial_uid="account-a", path="/app", init_script=None
):
    context = browser.new_context(viewport={"width": 1280, "height": 820})
    context.add_init_script(
        f"window.__E2E_INITIAL_UID = {json.dumps(initial_uid)};"
    )
    if init_script:
        context.add_init_script(init_script)
    context.route(
        "https://www.gstatic.com/firebasejs/9.22.0/firebase-app.js",
        lambda route: route.fulfill(content_type="application/javascript", body=FIREBASE_APP_STUB),
    )
    context.route(
        "https://www.gstatic.com/firebasejs/9.22.0/firebase-firestore.js",
        lambda route: route.fulfill(content_type="application/javascript", body=FIRESTORE_STUB),
    )
    context.route(
        "https://www.gstatic.com/firebasejs/9.22.0/firebase-auth.js",
        lambda route: route.fulfill(content_type="application/javascript", body=FIREBASE_AUTH_STUB),
    )
    context.route(
        "https://cloud.umami.is/**",
        lambda route: route.fulfill(content_type="application/javascript", body="/* e2e */"),
    )
    page = context.new_page()
    page.route("**/confirm-registration", lambda route: _json(route, {"status": "ok"}))
    page.route(
        "**/user_status",
        lambda route: _json(route, {"is_pro": False, "limit": 3, "deep_limit": 0}),
    )
    page.route(
        "**/usage",
        lambda route: _json(route, {
            "is_pro": False,
            "remaining": 3,
            "deep_remaining": 0,
            "total_limit": 3,
            "deep_total_limit": 0,
        }),
    )
    page.route(
        "**/bookmarks?*",
        lambda route: _json(route, {"bookmarks": [], "next_cursor": None}),
    )
    page.route("**/auth/session", lambda route: _json(route, {"status": "ok"}))
    page.goto(app_server + path, wait_until="domcontentloaded")
    page.wait_for_function(
        "() => window.__consensioAuthState?.known === true"
        " && typeof window.openShareDialog === 'function'"
    )
    return context, page


def test_auth_module_failure_exposes_usable_login_dialog(browser, phase4_server):
    context = browser.new_context(viewport={"width": 1280, "height": 820})
    context.add_init_script(
        "localStorage.setItem('id_token', 'stale-token');"
        "window.AUTH_BOOTSTRAP_TIMEOUT_MS = 30;"
    )
    context.route("**/static/firebase.js*", lambda route: route.abort())
    context.route("**/static/dist/firebase.*.js", lambda route: route.abort())
    context.route(
        "https://cloud.umami.is/**",
        lambda route: route.fulfill(content_type="application/javascript", body="/* e2e */"),
    )
    page = context.new_page()
    try:
        page.goto(phase4_server + "/app", wait_until="domcontentloaded")
        assert page.evaluate(
            "() => typeof window.App?.emailVerification"
        ) == "object"
        login = page.locator("#authTopLoginBtn")
        expect(login).to_be_visible(timeout=5000)
        expect(page.locator("#loginContainer .login-skeleton")).to_have_count(0)

        login.click()
        modal = page.get_by_role("dialog", name="Log in to consens.io")
        expect(modal).to_be_visible()
        expect(page.locator("#loginError")).to_contain_text("temporarily unavailable")
        expect(page.locator("#closeLoginModal")).to_be_focused()

        page.keyboard.press("Escape")
        expect(modal).to_be_hidden()
        expect(login).to_be_focused()
    finally:
        context.close()


def test_account_a_late_bookmark_save_cannot_mutate_account_b(browser, phase4_server):
    context, page = _real_firebase_page(browser, phase4_server)
    try:
        page.evaluate(
            """() => {
              const nativeFetch = window.fetch.bind(window);
              window.fetch = (input, options = {}) => {
                if (String(input) === "/bookmark" && options.method === "POST") {
                  return new Promise(resolve => {
                    window.__resolveAccountABookmark = () => resolve(new Response(JSON.stringify({
                      bookmark: { id: "account-a-bookmark", query: "A private question", responses: {} }
                    }), { status: 200, headers: { "Content-Type": "application/json" } }));
                  });
                }
                return nativeFetch(input, options);
              };
              window.__lateSave = window.saveBookmark(
                "A private question", "A private answer", "OpenAI", "normal"
              );
            }"""
        )
        page.wait_for_function("() => typeof window.__resolveAccountABookmark === 'function'")
        page.evaluate(
            """async () => {
              await window.__switchE2EUser(null);
              await window.__switchE2EUser("account-b");
              window.__resolveAccountABookmark();
              await window.__lateSave;
            }"""
        )

        assert page.evaluate("() => window.__consensioAuthState.uid") == "account-b"
        assert page.evaluate(
            "() => window.bookmarksData.some(item => item.id === 'account-a-bookmark')"
        ) is False
        expect(page.locator('[data-id="account-a-bookmark"]')).to_have_count(0)
    finally:
        context.close()


def test_account_a_share_response_cannot_overwrite_account_b_view(browser, phase4_server):
    context, page = _real_firebase_page(browser, phase4_server)
    try:
        page.evaluate(
            """() => {
              const nativeFetch = window.fetch.bind(window);
              window.fetch = (input, options = {}) => {
                if (String(input) === "/api/my/shares") {
                  const authHeader = options.headers?.Authorization || "";
                  if (authHeader.includes("account-a")) {
                    return new Promise(resolve => {
                      window.__resolveAccountAShares = () => resolve(new Response(JSON.stringify({
                        site_url: "https://example.invalid",
                        shares: [{ share_id: "share-a", status: "active", path: "/a", question: "A private share" }]
                      }), { status: 200, headers: { "Content-Type": "application/json" } }));
                    });
                  }
                  return Promise.resolve(new Response(JSON.stringify({
                    site_url: "https://example.invalid",
                    shares: [{ share_id: "share-b", status: "active", path: "/b", question: "B share" }]
                  }), { status: 200, headers: { "Content-Type": "application/json" } }));
                }
                return nativeFetch(input, options);
              };
              window.openShareDialog("list");
            }"""
        )
        page.wait_for_function("() => typeof window.__resolveAccountAShares === 'function'")
        page.evaluate(
            """async () => {
              await window.__switchE2EUser(null);
              await window.__switchE2EUser("account-b");
              window.openShareDialog("list");
            }"""
        )
        expect(page.locator("#shareModalBody")).to_contain_text("B share")

        page.evaluate("() => window.__resolveAccountAShares()")
        page.wait_for_timeout(100)
        expect(page.locator("#shareModalBody")).to_contain_text("B share")
        expect(page.locator("#shareModalBody")).not_to_contain_text("A private share")
    finally:
        context.close()


def test_last_bookmark_click_wins_when_detail_responses_arrive_out_of_order(browser, phase4_server):
    context, page = _real_firebase_page(browser, phase4_server)
    try:
        page.evaluate(
            """() => {
              const nativeFetch = window.fetch.bind(window);
              const responseFor = (id, question) => new Response(JSON.stringify({
                bookmark: {
                  id, query: question, title: question, mode: "normal",
                  responses: { OpenAI: "Model answer", consensus: "Consensus " + id },
                  sources: [], attachments: []
                }
              }), { status: 200, headers: { "Content-Type": "application/json" } });
              window.fetch = (input, options = {}) => {
                if (String(input) === "/bookmarks/bookmark_a") {
                  return new Promise(resolve => {
                    window.__resolveBookmarkA = () => resolve(responseFor("bookmark_a", "Question A"));
                  });
                }
                if (String(input) === "/bookmarks/bookmark_b") {
                  return Promise.resolve(responseFor("bookmark_b", "Question B"));
                }
                return nativeFetch(input, options);
              };
              window.__openA = window.openBookmark("bookmark_a");
            }"""
        )
        page.wait_for_function("() => typeof window.__resolveBookmarkA === 'function'")
        page.evaluate("() => { window.__openB = window.openBookmark('bookmark_b'); }")
        expect(page.locator("#threadAskText")).to_have_text("Question B")

        page.evaluate("() => window.__resolveBookmarkA()")
        page.wait_for_timeout(100)
        expect(page.locator("#threadAskText")).to_have_text("Question B")
        assert page.evaluate("() => window.App.bookmarkSession.currentId()") == "bookmark_b"
        assert page.evaluate("() => window.lastQuestion") == "Question B"
        assert page.evaluate("() => window.consensusCitationMeta?.question") == "Question B"
        expect(page.locator(".explanation-popup")).to_have_count(0)
    finally:
        context.close()


def test_logged_out_watch_deep_link_survives_and_late_login_renders(browser, phase4_server):
    context, page = _real_firebase_page(browser, phase4_server, initial_uid=None, path="/app/watches")
    try:
        page.route(
            "**/api/my/watches",
            lambda route: _json(route, {
                "watches": [],
                "limits": {"plan": "free", "active_count": 0, "active_limit": 1},
            }),
        )
        page.route("**/api/my/watch-brief", lambda route: _json(route, {"brief": {}}))
        page.route("**/api/my/telegram", lambda route: _json(route, {"telegram": {}}))

        expect(page.locator("#watchDashBody")).to_contain_text(
            "Please log in to see your Consensus Watch dashboard."
        )
        assert page.url.endswith("/app/watches")

        page.evaluate("() => window.__switchE2EUser('account-b')")
        expect(page.locator("#watchDashCreate")).to_be_visible(timeout=5000)
        expect(page.locator("#watchDashBody")).to_contain_text("Keep changing answers current")
        assert page.url.endswith("/app/watches")
    finally:
        context.close()


def test_all_model_failures_end_in_error_without_consensus(browser, phase4_server):
    context, page = _real_firebase_page(browser, phase4_server)
    try:
        consensus_requests = []
        page.on(
            "request",
            lambda request: consensus_requests.append(request.url)
            if request.url.endswith("/consensus") else None,
        )
        page.route(
            "**/prepare",
            lambda route: _json(route, {"system_prompt": "Prepared prompt"}),
        )
        page.route(
            "**/ask_*",
            lambda route: _json(route, {"error": "provider unavailable"}, status=500),
        )
        page.fill("#questionInput", "Why did every provider fail in this test?")
        page.evaluate("() => window.sendQuestion()")

        expect(page.locator("#agentModeStatus")).to_have_text(
            "All selected model requests failed.", timeout=30000
        )
        assert consensus_requests == []
    finally:
        context.close()


def test_disabled_agent_mode_is_six_answers_only(browser, phase4_server):
    context, page = _real_firebase_page(browser, phase4_server)
    try:
        consensus_requests = []
        page.on(
            "request",
            lambda request: consensus_requests.append(request.url)
            if request.url.endswith("/consensus") else None,
        )
        page.route(
            "**/prepare",
            lambda route: _json(route, {"system_prompt": "Prepared prompt"}),
        )
        page.route(
            "**/ask_*",
            lambda route: _json(route, {
                "response": "Direct answer from " + route.request.url.rsplit("_", 1)[-1],
                "sources": [],
            }),
        )
        page.click("#attachTrigger")
        menu_toggle = page.locator("#agentModeMenuSwitch")
        menu_toggle_row = page.locator('label[for="agentModeMenuSwitch"]')
        expect(menu_toggle_row).to_be_visible()
        expect(menu_toggle).to_be_enabled()
        expect(menu_toggle).to_be_checked()
        menu_toggle_row.click()
        expect(menu_toggle).not_to_be_checked()
        expect(page.locator("#agentModeSwitch")).not_to_be_checked()
        expect(page.locator("#autoConsensusToggle")).not_to_be_checked()
        page.fill("#questionInput", "Give me six direct answers only.")
        page.evaluate("() => window.sendQuestion()")

        for response_id in (
            "openaiResponse", "mistralResponse", "claudeResponse",
            "geminiResponse", "deepseekResponse", "grokResponse",
        ):
            expect(page.locator(f"#{response_id}")).to_contain_text(
                "Direct answer from", timeout=10000
            )
            expect(page.locator(f"#{response_id}")).to_be_visible()

        page.wait_for_timeout(500)
        assert consensus_requests == []
        assert page.locator("body").evaluate(
            "el => el.classList.contains('is-hero')"
            " && el.classList.contains('direct-comparison-active')"
            " && !el.classList.contains('agent-mode-enabled')"
        )
        expect(page.locator("#threadAsk")).to_be_hidden()
        expect(page.locator("#consensusRun")).to_be_hidden()
        expect(page.locator("#consensusOutput")).to_be_hidden()
        expect(page.locator("#consensusAnswerBody .cx-claim")).to_have_count(0)
        expect(page.locator("#differencesCards")).to_be_hidden()
        auto_toggle = page.locator("#autoConsensusToggle")
        expect(auto_toggle).not_to_be_checked()
        expect(auto_toggle).to_be_disabled()
    finally:
        context.close()


def test_attachment_filter_blocks_one_model_run_before_prepare(browser, phase4_server):
    context, page = _real_firebase_page(browser, phase4_server)
    try:
        requests = []
        page.on(
            "request",
            lambda request: requests.append(request.url)
            if any(part in request.url for part in ("/prepare", "/ask_")) else None,
        )
        page.evaluate(
            """() => {
              window.App.state.set("isUserPro", true, "userTier");
              for (const id of [
                "selectOpenAI", "selectMistral", "selectClaude",
                "selectGemini", "selectDeepSeek", "selectGrok"
              ]) document.getElementById(id).checked = false;
              document.getElementById("selectOpenAI").checked = true;
              document.getElementById("selectDeepSeek").checked = true;
              window.pendingAttachments = [{
                name: "brief.pdf", mime: "application/pdf", size: 9, data: "JVBERi0xLjcK"
              }];
              window.renderAttachmentChips();
              // Simulate stale external state after the attachment renderer disabled it.
              document.getElementById("selectDeepSeek").checked = true;
            }"""
        )
        page.fill("#questionInput", "Can this filtered run start?")
        page.evaluate("() => window.sendQuestion()")

        expect(page.locator(".explanation-popup")).to_contain_text(
            "Choose at least two compatible models"
        )
        assert requests == []
    finally:
        context.close()


def test_watched_navigation_closes_the_shared_modal(browser, phase4_server):
    context, page = _real_firebase_page(browser, phase4_server)
    try:
        page.route(
            "**/api/my/watches",
            lambda route: _json(route, {
                "watches": [],
                "limits": {"plan": "free", "active_count": 0, "active_limit": 1},
            }),
        )
        page.route("**/api/my/watch-brief", lambda route: _json(route, {"brief": {}}))
        page.route("**/api/my/telegram", lambda route: _json(route, {"telegram": {}}))

        page.evaluate(
            "() => { window.App.state.set('lastShareResultId', 'phase4-result', 'share'); window.openShareDialog('confirm'); }"
        )
        expect(page.locator("#shareModal")).to_be_visible()
        page.click("#watchListLink")

        expect(page.locator("#shareModal")).to_be_hidden()
        expect(page.locator("#watchDashboard")).to_be_visible()
        assert page.url.endswith("/app/watches")
    finally:
        context.close()


def test_cancel_during_token_resolution_keeps_followup_and_creates_no_usage_run(
    browser, phase4_server
):
    context, page = _real_firebase_page(browser, phase4_server)
    try:
        page.evaluate(
            """() => {
              document.getElementById("selectOpenAI").checked = true;
              document.getElementById("selectMistral").checked = true;
              document.getElementById("questionInput").value = "What changed since yesterday?";
              window.App.followup.offer("Previous question", "Previous consensus");
              let release;
              const pending = new Promise(resolve => { release = resolve; });
              window.auth.currentUser.getIdToken = () => pending;
              window.__releaseSlowToken = () => release("token-account-a");
              window.__slowSend = window.sendQuestion();
              window.__slowSend.then(() => { window.__slowSendDone = true; });
            }"""
        )
        page.wait_for_function("() => window.isQueryRequestRunning() === true")
        page.evaluate("() => window.cancelCurrentQuery()")
        page.evaluate("() => window.__releaseSlowToken()")
        page.wait_for_function("() => window.__slowSendDone === true")

        assert page.evaluate("() => window.App.followup.isArmed()") is True
        assert page.evaluate("() => !!window.App.followup.spentExchange") is False
        assert page.evaluate("() => window.App.usageRun?.current?.key || null") is None
        assert page.evaluate("() => window.isQueryRequestRunning()") is False
    finally:
        context.close()


def test_late_watch_create_cannot_overwrite_newer_share_modal(browser, phase4_server):
    context, page = _real_firebase_page(browser, phase4_server)
    try:
        page.route(
            "**/api/my/watches",
            lambda route: _json(route, {
                "watches": [],
                "limits": {"plan": "free", "active_count": 0, "active_limit": 5},
            }),
        )
        page.route("**/api/my/telegram", lambda route: _json(route, {"telegram": {}}))
        page.evaluate(
            """() => {
              const nativeFetch = window.fetch.bind(window);
              window.fetch = (input, options = {}) => {
                if (String(input) === "/api/watch" && options.method === "POST") {
                  return new Promise(resolve => {
                    window.__resolveWatchCreate = () => resolve(new Response(JSON.stringify({
                      watch: {
                        id: "watch-race", interval: "weekly", run_weekday: "monday",
                        run_time: "09:00", timezone: "Europe/Berlin", query_first: true,
                        share_path: "/s/watch-race", visibility: "private",
                        email_mode: "changes_only", email_enabled: true,
                        telegram_enabled: false
                      }
                    }), { status: 200, headers: { "Content-Type": "application/json" } }));
                  });
                }
                return nativeFetch(input, options);
              };
              window.openWatchDialog("create", {
                question: "Has this policy changed since last week?"
              });
            }"""
        )
        page.click("#watchQuestionNext")
        page.click("#watchConfirmBtn")
        page.wait_for_function("() => typeof window.__resolveWatchCreate === 'function'")

        page.evaluate("() => window.openShareDialog('confirm')")
        expect(page.locator("#shareModalTitle")).to_have_text(
            "Share this consensus publicly"
        )
        page.evaluate("() => window.__resolveWatchCreate()")
        page.wait_for_timeout(150)

        expect(page.locator("#shareModalTitle")).to_have_text(
            "Share this consensus publicly"
        )
        expect(page.locator("#shareModalBody")).not_to_contain_text(
            "Your Watch is active"
        )
    finally:
        context.close()


def test_watch_create_is_not_sent_after_modal_changes_during_token_wait(
    browser, phase4_server
):
    context, page = _real_firebase_page(browser, phase4_server)
    try:
        page.route(
            "**/api/my/watches",
            lambda route: _json(route, {
                "watches": [],
                "limits": {"plan": "free", "active_count": 0, "active_limit": 5},
            }),
        )
        page.route("**/api/my/telegram", lambda route: _json(route, {"telegram": {}}))
        page.evaluate(
            """() => window.openWatchDialog("create", {
              question: "Has this policy changed since last week?"
            })"""
        )
        page.click("#watchQuestionNext")
        page.evaluate(
            """() => {
              let release;
              const pending = new Promise(resolve => { release = resolve; });
              window.auth.currentUser.getIdToken = () => pending;
              window.__releaseWatchToken = () => release("token-account-a");
              window.__watchPostSent = false;
              const nativeFetch = window.fetch.bind(window);
              window.fetch = (input, options = {}) => {
                if (String(input) === "/api/watch" && options.method === "POST") {
                  window.__watchPostSent = true;
                }
                return nativeFetch(input, options);
              };
            }"""
        )
        page.click("#watchConfirmBtn")
        page.evaluate("() => window.openShareDialog('confirm')")
        page.evaluate("() => window.__releaseWatchToken()")
        page.wait_for_timeout(100)

        assert page.evaluate("() => window.__watchPostSent") is False
        expect(page.locator("#shareModalTitle")).to_have_text(
            "Share this consensus publicly"
        )
    finally:
        context.close()


def test_failed_utc_usage_refresh_retries_until_authoritative_success(
    browser, phase4_server
):
    context, page = _real_firebase_page(
        browser,
        phase4_server,
        init_script="window.USAGE_REFRESH_RETRY_MS = 25;",
    )
    try:
        page.evaluate(
            """() => {
              window.App.renderUsageDisplay({
                remaining: 0, deepRemaining: 0, totalLimit: 3, deepLimit: 0
              });
              window.__usageRefreshCalls = 0;
              window.refreshUsageData = async () => {
                window.__usageRefreshCalls += 1;
                if (window.__usageRefreshCalls === 1) return false;
                window.App.renderUsageDisplay({
                  remaining: 3, deepRemaining: 0, totalLimit: 3, deepLimit: 0
                });
                return true;
              };

              const OriginalDate = window.Date;
              const realStartedAt = OriginalDate.now();
              const today = new OriginalDate();
              const fakeStartedAt = OriginalDate.UTC(
                today.getUTCFullYear(), today.getUTCMonth(), today.getUTCDate() + 1,
                0, 0, 1
              );
              class NextUtcDayDate extends OriginalDate {
                constructor(...args) {
                  super(...(args.length
                    ? args
                    : [fakeStartedAt + (OriginalDate.now() - realStartedAt)]));
                }
                static now() {
                  return fakeStartedAt + (OriginalDate.now() - realStartedAt);
                }
                static UTC(...args) { return OriginalDate.UTC(...args); }
                static parse(value) { return OriginalDate.parse(value); }
              }
              window.Date = NextUtcDayDate;
            }"""
        )

        page.wait_for_function("() => window.__usageRefreshCalls >= 2", timeout=5000)
        assert page.evaluate(
            "() => window.App.usageLimit.preflight({ useOwnKeys: false })"
        ) is None
    finally:
        context.close()


def test_template_visibility_classes_remain_overridable_by_ui_controls(
    browser, phase4_server
):
    context, page = _real_firebase_page(
        browser, phase4_server, initial_uid=None
    )
    try:
        page.click("#authTopSignupBtn")
        expect(page.locator("#loginModal")).to_be_visible()
        expect(page.locator("#loginEmailConfirm")).to_be_visible()
        expect(page.locator("#loginPassword")).to_be_hidden()
        expect(page.locator("#confirmRegisterButton")).to_be_visible()

        page.click("#toggleRegister")
        expect(page.locator("#loginPassword")).to_be_visible()
        expect(page.locator("#confirmRegisterButton")).to_be_hidden()

        page.evaluate("() => document.getElementById('editSystemPromptBtn').click()")
        expect(page.locator("#systemPromptModal")).to_be_visible()
        page.click("#closeSystemPromptModal")
        expect(page.locator("#systemPromptModal")).to_be_hidden()

        page.evaluate("() => document.getElementById('editSystemPromptBtn').click()")
        page.click("#settingsTabConnections")
        expect(page.locator("#apiSettingsArea")).to_be_hidden()
        page.evaluate("() => document.getElementById('apiSettingsToggle').click()")
        expect(page.locator("#apiSettingsArea")).to_be_visible()
    finally:
        context.close()
