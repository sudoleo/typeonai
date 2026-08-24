from __future__ import annotations

import subprocess
from pathlib import Path
from tests.frontend_order import loads_before


ROOT = Path(__file__).resolve().parents[1]


def test_chat_session_state_fresh_followup_retry_and_reset_contract():
    script_path = ROOT / "static" / "js" / "chat-session.js"
    node_script = f"""
const assert = require("assert");
const fs = require("fs");
const vm = require("vm");
const {{ webcrypto }} = require("crypto");
global.window = global;
global.crypto = webcrypto;
global.App = {{}};
const calls = [];
let chatIds = ["c".repeat(32), "d".repeat(32), "e".repeat(32)];
let turnIds = ["1".repeat(32), "2".repeat(32), "3".repeat(32), "4".repeat(32)];
let failNextTurn = false;
let failNextChat = false;
global.fetch = async (url, options) => {{
  calls.push({{ url, options }});
  if (url === "/chats") {{
    if (failNextChat) {{
      failNextChat = false;
      return {{ ok: false, status: 503, json: async () => ({{}}) }};
    }}
    return {{ ok: true, status: 201, json: async () => ({{ chat: {{ id: chatIds.shift() }} }}) }};
  }}
  if (failNextTurn) {{
    failNextTurn = false;
    return {{ ok: false, status: 503, json: async () => ({{}}) }};
  }}
  return {{ ok: true, status: 201, json: async () => ({{ turn: {{ id: turnIds.shift() }} }}) }};
}};
vm.runInThisContext(fs.readFileSync({str(script_path)!r}, "utf8"), {{ filename: "chat-session.js" }});
const session = App.chatSession;
const run = (overrides = {{}}) => session.beginRun({{
  question: "Question one",
  mode: "Standard",
  deepSearch: false,
  selectedModels: ["model-a", "model-b"],
  consensusModel: "Gemini",
  isFollowup: false,
  prepareSucceeded: true,
  api_key: "must-not-survive",
  attachments: [{{ name: "chart.png", mime: "image/png", size: 2048, data: "must-not-survive" }}],
  ...overrides
}});

(async () => {{
  run();
  const stableId = session.pendingClientRequestId;
  const first = await session.ensurePendingTurn({{ idToken: "token", question: "Question one", consensusModel: "Gemini" }});
  assert.deepStrictEqual(first, {{ chatId: "c".repeat(32), turnId: "1".repeat(32) }});
  assert.strictEqual(session.activeChatId, null);
  assert.strictEqual(session.pendingChatId, first.chatId);
  assert.strictEqual(calls.length, 2);
  const firstTurnPayload = JSON.parse(calls[1].options.body);
  assert.deepStrictEqual(Object.keys(firstTurnPayload).sort(), [
    "attachments", "client_request_id", "consensus_model", "deep_search", "mode", "question",
    "selected_models"
  ].sort());
  assert.strictEqual(firstTurnPayload.client_request_id, stableId);
  // Der Anhang reist als reine Metadaten mit: der Turn muss erzaehlen koennen,
  // dass eine Datei an dieser Frage hing — die Datei selbst wird nie gespeichert.
  assert.deepStrictEqual(firstTurnPayload.attachments, [
    {{ name: "chart.png", mime: "image/png", size: 2048 }}
  ]);
  assert.ok(!JSON.stringify(firstTurnPayload).includes("must-not-survive"));
  assert.ok(!JSON.stringify(firstTurnPayload).includes("token"));

  const retryWithoutNetwork = await session.ensurePendingTurn({{ idToken: "token", question: "Question one", consensusModel: "Gemini" }});
  assert.deepStrictEqual(retryWithoutNetwork, first);
  assert.strictEqual(calls.length, 2);
  session.handleConsensusResult({{
    chatId: first.chatId,
    turnId: first.turnId,
    chatPersisted: true,
    chatTurnState: "completed"
  }});
  assert.strictEqual(session.activeChatId, first.chatId);
  assert.strictEqual(session.pendingChatId, null);
  assert.strictEqual(session.pendingTurnId, null);

  run({{ question: "Follow-up", isFollowup: true }});
  const beforeFollowup = calls.length;
  const followup = await session.ensurePendingTurn({{ idToken: "token", question: "Follow-up", consensusModel: "Gemini" }});
  assert.strictEqual(followup.chatId, first.chatId);
  assert.strictEqual(calls.length, beforeFollowup + 1);
  assert.strictEqual(calls.at(-1).url, `/chats/${{first.chatId}}/turns`);
  session.handleConsensusResult({{
    chatId: followup.chatId,
    turnId: followup.turnId,
    chatPersisted: false,
    chatTurnState: "pending"
  }});
  assert.strictEqual(session.activeChatId, first.chatId);
  assert.strictEqual(session.pendingChatId, first.chatId);

  run({{ question: "Follow-up behind pending", isFollowup: true }});
  const beforeUnsafeFollowup = calls.length;
  assert.strictEqual(await session.ensurePendingTurn({{
    idToken: "token",
    question: "Follow-up behind pending",
    consensusModel: "Gemini"
  }}), null);
  assert.strictEqual(calls.length, beforeUnsafeFollowup);
  assert.strictEqual(session.activeChatId, null);

  run({{ question: "Fresh again" }});
  assert.strictEqual(session.activeChatId, null);
  const freshAgain = await session.ensurePendingTurn({{ idToken: "token", question: "Fresh again", consensusModel: "Gemini" }});
  assert.strictEqual(freshAgain.chatId, "d".repeat(32));
  assert.strictEqual(session.activeChatId, null);
  session.handleConsensusResult({{
    chatId: freshAgain.chatId,
    turnId: freshAgain.turnId,
    chatPersisted: false,
    chatTurnState: "pending"
  }});
  assert.strictEqual(session.activeChatId, null);
  assert.strictEqual(session.pendingChatId, freshAgain.chatId);

  run({{ question: "Must not follow pending", isFollowup: true }});
  const beforePendingFollowup = calls.length;
  assert.strictEqual(await session.ensurePendingTurn({{
    idToken: "token",
    question: "Must not follow pending",
    consensusModel: "Gemini"
  }}), null);
  assert.strictEqual(calls.length, beforePendingFollowup);
  assert.strictEqual(session.activeChatId, null);

  session.reset();
  run({{ question: "Legacy follow-up", isFollowup: true }});
  const beforeLegacy = calls.length;
  const legacy = await session.ensurePendingTurn({{ idToken: "token", question: "Legacy follow-up", consensusModel: "Gemini" }});
  assert.strictEqual(legacy, null);
  assert.strictEqual(calls.length, beforeLegacy);

  session.reset();
  run({{ question: "Retry turn" }});
  failNextTurn = true;
  const requestId = session.pendingClientRequestId;
  const firstAttempt = await session.ensurePendingTurn({{ idToken: "token", question: "Retry turn", consensusModel: "Gemini" }});
  assert.strictEqual(firstAttempt, null);
  assert.strictEqual(session.activeChatId, null);
  const retryAttempt = await session.ensurePendingTurn({{ idToken: "token", question: "Retry turn", consensusModel: "Gemini" }});
  assert.strictEqual(retryAttempt.chatId, "e".repeat(32));
  const retryBodies = calls.filter(call => call.url === `/chats/${{"e".repeat(32)}}/turns`)
    .map(call => JSON.parse(call.options.body));
  assert.strictEqual(retryBodies.length, 2);
  assert.ok(retryBodies.every(body => body.client_request_id === requestId));

  const beforeLogicalRetry = calls.length;
  run({{ question: "Retry turn" }});
  assert.strictEqual(session.pendingClientRequestId, requestId);
  assert.deepStrictEqual(await session.ensurePendingTurn({{
    idToken: "token",
    question: "Retry turn",
    consensusModel: "Gemini"
  }}), retryAttempt);
  assert.strictEqual(calls.length, beforeLogicalRetry);

  session.handleConsensusResult({{}});
  assert.strictEqual(session.pendingChatId, retryAttempt.chatId);
  assert.strictEqual(session.pendingTurnId, retryAttempt.turnId);
  session.handleConsensusResult({{
    chatId: retryAttempt.chatId,
    turnId: retryAttempt.turnId,
    chatPersisted: false,
    chatTurnState: "pending"
  }});
  assert.strictEqual(session.pendingChatId, retryAttempt.chatId);
  assert.strictEqual(session.pendingTurnId, retryAttempt.turnId);
  session.handleConsensusResult({{
    chatId: retryAttempt.chatId,
    turnId: retryAttempt.turnId,
    chatPersisted: false,
    chatTurnState: "failed"
  }});
  assert.strictEqual(session.activeChatId, null);
  assert.strictEqual(session.pendingChatId, null);
  assert.strictEqual(session.pendingTurnId, null);

  session.reset();
  run({{ question: "Chat unavailable" }});
  failNextChat = true;
  const beforeFailure = calls.length;
  assert.strictEqual(await session.ensurePendingTurn({{ idToken: "token", question: "Chat unavailable", consensusModel: "Gemini" }}), null);
  assert.strictEqual(session.activeChatId, null);
  assert.strictEqual(await session.ensurePendingTurn({{ idToken: "token", question: "Chat unavailable", consensusModel: "Gemini" }}), null);
  assert.strictEqual(calls.length, beforeFailure + 1);

  session.reset();
  assert.strictEqual(session.activeChatId, null);
  assert.strictEqual(session.activeTurnId, null);
  assert.strictEqual(session.pendingChatId, null);
  assert.strictEqual(session.pendingTurnId, null);
  assert.strictEqual(session.pendingContextVersionId, null);
  assert.strictEqual(session.pendingUsageRunKey, null);
  assert.strictEqual(session.pendingClientRequestId, null);
  assert.strictEqual(session.logicalRun, null);
}})().catch(error => {{ console.error(error); process.exit(1); }});
"""
    result = subprocess.run(
        ["node", "-e", node_script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_chat_session_script_order_consensus_payload_and_legacy_bookmarks_remain():
    template = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    consensus = (ROOT / "static" / "js" / "consensus-run.js").read_text(
        encoding="utf-8"
    )
    query = (ROOT / "static" / "js" / "query-send.js").read_text(encoding="utf-8")
    firebase = (ROOT / "static" / "firebase.js").read_text(encoding="utf-8")
    app_init = (ROOT / "static" / "js" / "app-init.js").read_text(encoding="utf-8")

    assert loads_before("app-core.js", "chat-session.js")
    assert loads_before("chat-session.js", "consensus-run.js")
    assert loads_before("consensus-run.js", "query-send.js")
    assert loads_before("query-send.js", "app-init.js")
    assert 'data-engine-provider="{{ model.provider }}"' in template
    assert "consensusPayload.chat_id = chatTurnIds.chatId" in consensus
    assert "consensusPayload.turn_id = chatTurnIds.turnId" in consensus
    assert "consensusPayload.context_version_id = chatTurnIds.contextVersionId" in consensus
    assert "streamSSERequest(\"/consensus\", consensusPayload" in consensus
    assert consensus.index("ensurePendingTurn") < consensus.index(
        'streamSSERequest("/consensus", consensusPayload'
    )
    assert "window.saveBookmarkConsensus(" in consensus
    assert "window.saveBookmark?.(" in query
    assert "context.config.providers.map(provider => runProvider(" in query
    assert "window.App?.chatSession?.reset?.();" in firebase
    assert "window.App.chatSession?.reset?.();" in app_init
    assert "context.chatSession.beginRun({" in query
    assert "function conversationPayload(context, payload)" in query
    assert "context.previousExchange && !context.basis?.chatId" in query
    assert "Object.assign(payload, binding)" in query
    assert "memory_api_key" not in query
    offer_block = consensus.split("offer(question, consensusText", 1)[1].split(
        "arm()", 1
    )[0]
    assert "if (this.followupInFlight)" not in offer_block
    history_block = consensus.split("appendHistoryTurn(", 1)[1].split(
        "archiveCurrentExchange()", 1
    )[0]
    assert "injectMarkdown(answerBody, turnData.consensus, turnSources)" in history_block
    assert "window.currentEvidenceSources =" not in history_block
    assert "turnData.model_answers" in history_block
    assert "thread-history-sources" in history_block
    assert "terminalError" not in consensus
    assert "chatTurnState: chatDisposition.chat_turn_state" in consensus


def test_chat_turn_payload_contains_no_key_or_answer_fields():
    session_script = (ROOT / "static" / "js" / "chat-session.js").read_text(
        encoding="utf-8"
    )
    payload_block = session_script.split("const turnPayload = {", 1)[1].split("};", 1)[0]
    assert "question:" in payload_block
    assert "selected_models:" in payload_block
    assert "client_request_id:" in payload_block
    assert "api_key" not in payload_block
    assert "id_token" not in payload_block
    assert "answer" not in payload_block


def test_session6_frontend_replay_disposition_and_ui_ordering_contracts():
    consensus = (ROOT / "static" / "js" / "consensus-run.js").read_text(
        encoding="utf-8"
    )
    query = (ROOT / "static" / "js" / "query-send.js").read_text(
        encoding="utf-8"
    )

    assert 'const completedReplay = data.chat_replayed === true;' in consensus
    assert "if (!completedReplay && window.auth?.currentUser)" in consensus
    assert "if (!completedReplay && bestModelFromConsensus)" in consensus
    assert 'const dispositionOnly = trigger === "disposition";' in consensus
    assert "replayPendingTurn || dispositionOnly" in consensus
    assert 'trigger: "disposition", dispositionOnly: true' in query
    assert "context.keepConversationLock = true" in query

    # Own-key completeness is checked before /prepare, follow-up consumption,
    # pending-turn creation, and all provider calls.
    begin_context = query.split("function beginContext(", 1)[1].split(
        "function handComposerToRun", 1
    )[0]
    own_key_check = begin_context.index("const missing = validateOwnKeys")
    assert own_key_check < begin_context.index("registry.create({")
    send_flow = query.split("window.sendQuestion =", 1)[1]
    assert send_flow.index("beginContext(") < send_flow.index("executeRun(context)")

    # Context construction and every provider callback stay on the private
    # session/RunContext; query-send contains no response-box resets.
    context_build = query.index("await context.chatSession.ensureContext({")
    provider_fanout = query.index(
        "context.config.providers.map(provider => runProvider(context, provider, idToken"
    )
    assert context_build < provider_fanout
    assert "delete box.dataset.consensusAnswer" not in query

    # Fresh Turn 1 remains lazy; only a follow-up with a frozen authoritative
    # chat basis creates its pending turn before fanout. consensus-run retains
    # late creation for a fresh run.
    lifecycle = query.split("context.chatSession.beginRun({", 1)[1].split(
        'context.phase = "answers";', 1
    )[0]
    assert lifecycle.count("ensurePendingTurn") == 1
    assert lifecycle.index("if (context.previousExchange && context.basis?.chatId)") < lifecycle.index(
        "ensurePendingTurn"
    )
    assert "ensurePendingTurn" in consensus


def test_chat_context_version_retry_degraded_and_own_key_contract():
    script_path = ROOT / "static" / "js" / "chat-session.js"
    node_script = f"""
const assert = require("assert");
const fs = require("fs");
const vm = require("vm");
const {{ webcrypto }} = require("crypto");
global.window = global;
global.crypto = webcrypto;
global.App = {{}};
const calls = [];
const turnIds = ["1".repeat(32), "2".repeat(32), "3".repeat(32), "4".repeat(32)];
let contextResponses = [];
let inspectedState = "pending";
global.fetch = async (url, options = {{}}) => {{
  calls.push({{ url, options }});
  if (url.endsWith("/context")) return contextResponses.shift();
  if (options.method === "POST" && url.endsWith("/turns")) {{
    return {{ ok: true, status: 201, json: async () => ({{ turn: {{ id: turnIds.shift() }} }}) }};
  }}
  if (!options.method && url.includes("/turns/")) {{
    return {{ ok: true, status: 200, json: async () => ({{ turn: {{ status: inspectedState }} }}) }};
  }}
  throw new Error("unexpected fetch " + url);
}};
vm.runInThisContext(fs.readFileSync({str(script_path)!r}, "utf8"), {{ filename: "chat-session.js" }});
const session = App.chatSession;
const activeChat = "c".repeat(32);
session.activeChatId = activeChat;
session.activeTurnId = "a".repeat(32);
session.beginRun({{
  question: "Second", mode: "Standard", deepSearch: false,
  selectedModels: ["m1", "m2"], consensusModel: "Gemini",
  isFollowup: true, prepareSucceeded: true, useOwnKeys: false,
  usageRunKey: "usage-stable"
}});

(async () => {{
  const pending = await session.ensurePendingTurn({{
    idToken: "token", question: "Second", consensusModel: "Gemini"
  }});
  assert.strictEqual(pending.chatId, activeChat);
  const version = "b".repeat(32);
  contextResponses = [
    {{ ok: false, status: 202, headers: {{ get: () => "0" }}, json: async () => ({{ status: "building" }}) }},
    {{ ok: true, status: 200, headers: {{ get: () => null }}, json: async () => ({{
      context: {{ id: version, state: "ready", target_turn_id: pending.turnId }}
    }}) }}
  ];
  const ready = await session.ensureContext({{
    idToken: "token", useOwnKeys: false, usageRunKey: "usage-stable"
  }});
  assert.strictEqual(ready.contextVersionId, version);
  assert.deepStrictEqual(session.contextBinding(), {{
    chat_id: activeChat, turn_id: pending.turnId, context_version_id: version
  }});
  const contextCalls = calls.filter(call => call.url.endsWith("/context"));
  assert.strictEqual(contextCalls.length, 2);
  contextCalls.forEach(call => assert.deepStrictEqual(JSON.parse(call.options.body), {{
    useOwnKeys: false, usage_run_key: "usage-stable"
  }}));
  assert.strictEqual(session.canReuseUsageRun({{
    question: "Second", mode: "Standard", deepSearch: false,
    selectedModels: ["m1", "m2"], consensusModel: "Gemini",
    isFollowup: true, useOwnKeys: false
  }}), true);

  session.markPendingUncertain();
  inspectedState = "completed";
  const inspected = await session.inspectPendingTurn({{ idToken: "token" }});
  assert.strictEqual(inspected.status, "completed");
  assert.strictEqual(session.pendingTurnId, pending.turnId);
  session.handleConsensusResult({{
    chatId: activeChat, turnId: pending.turnId,
    chatPersisted: true, chatTurnState: "completed"
  }});
  assert.strictEqual(session.activeTurnId, pending.turnId);

  session.beginRun({{
    question: "Third", mode: "Standard", deepSearch: false,
    selectedModels: ["m1", "m2"], consensusModel: "Gemini",
    isFollowup: true, prepareSucceeded: true, useOwnKeys: true
  }});
  const third = await session.ensurePendingTurn({{
    idToken: "token", question: "Third", consensusModel: "Gemini"
  }});
  const degradedVersion = "d".repeat(32);
  contextResponses = [{{
    ok: true, status: 200, headers: {{ get: () => null }}, json: async () => ({{
      context: {{ id: degradedVersion, state: "degraded", target_turn_id: third.turnId }}
    }})
  }}];
  const degraded = await session.ensureContext({{
    idToken: "token", useOwnKeys: true, memoryApiKey: "only-memory-key"
  }});
  assert.strictEqual(degraded.contextState, "degraded");
  const ownBody = JSON.parse(calls.filter(call => call.url.endsWith("/context")).at(-1).options.body);
  assert.deepStrictEqual(ownBody, {{ useOwnKeys: true, memory_api_key: "only-memory-key" }});
  assert.ok(!JSON.stringify(ownBody).includes("usage-stable"));

  session.handleConsensusResult({{
    chatId: activeChat, turnId: third.turnId,
    chatPersisted: true, chatTurnState: "completed"
  }});
  session.beginRun({{
    question: "Fourth", mode: "Standard", deepSearch: false,
    selectedModels: ["m1", "m2"], consensusModel: "Gemini",
    isFollowup: true, prepareSucceeded: true, useOwnKeys: false,
    usageRunKey: "usage-fourth"
  }});
  const fourth = await session.ensurePendingTurn({{
    idToken: "token", question: "Fourth", consensusModel: "Gemini"
  }});
  const completedPredecessor = session.activeTurnId;
  contextResponses = [{{
    ok: false, status: 409, headers: {{ get: () => null }},
    json: async () => ({{ detail: "Chat context conflict" }})
  }}];
  await assert.rejects(session.ensureContext({{
    idToken: "token", useOwnKeys: false, usageRunKey: "usage-fourth"
  }}), /conflict/i);
  assert.strictEqual(session.activeTurnId, completedPredecessor);
  assert.strictEqual(session.pendingTurnId, fourth.turnId);

  const controller = new AbortController();
  contextResponses = [{{
    ok: false, status: 202, headers: {{ get: () => "1" }},
    json: async () => ({{ status: "building" }})
  }}];
  controller.abort();
  await assert.rejects(session.ensureContext({{
    idToken: "token", useOwnKeys: false, usageRunKey: "usage-fourth",
    signal: controller.signal
  }}), error => error.name === "AbortError");
  assert.strictEqual(session.pendingTurnId, fourth.turnId);
  session.handleConsensusResult({{
    chatId: activeChat, turnId: fourth.turnId,
    chatPersisted: false, chatTurnState: "failed"
  }});
  assert.strictEqual(session.activeTurnId, completedPredecessor);
  assert.strictEqual(session.pendingTurnId, null);
}})().catch(error => {{ console.error(error); process.exit(1); }});
"""
    result = subprocess.run(
        ["node", "-e", node_script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_replay_repaints_model_boxes_from_the_stored_turn():
    consensus = (ROOT / "static" / "js" / "consensus-run.js").read_text(
        encoding="utf-8"
    )

    # A replay never calls the providers, so the boxes must come from the
    # stored turn rather than from whatever the previous run left behind.
    assert "function restoreStoredModelAnswers(storedAnswers)" in consensus
    assert "restoreStoredModelAnswers(data.model_answers)" in consensus
    restore_call = consensus.index("restoreStoredModelAnswers(data.model_answers)")
    guard = consensus.index("const replayedAnswerCount = completedReplay")
    assert guard < restore_call

    body = consensus.split("function restoreStoredModelAnswers", 1)[1].split(
        "\n    }", 1
    )[0]
    # Providers without a stored answer are blanked, never left stale.
    assert 'box.dataset.responseState = "idle"' in body
    assert 'outputEl.innerHTML = ""' in body
    assert "delete box.dataset.responseError" in body
    assert 'box.dataset.responseState = "complete"' in body
    # Sources travel per stored answer so old [S…] links cannot leak across turns.
    assert "Array.isArray(stored?.sources)" in body

    # The agreement widget must count the stored models, not the stale DOM.
    assert "completedReplay ? replayedAnswerCount : includedAnswerCount" in consensus


def test_pending_turn_creation_does_not_rewrite_the_logical_run():
    session = (ROOT / "static" / "js" / "chat-session.js").read_text(encoding="utf-8")

    lifecycle = session.split("async ensurePendingTurn(", 1)[1].split(
        "async _createPendingTurn", 1
    )[0]
    # sameLogicalRun() compares a retry against what the user chose when the
    # run started; an unconditional overwrite would hide a changed picker.
    assert "run.consensusModel = cleanString(consensusModel) || run.consensusModel" not in lifecycle
    assert "if (!run.consensusModel) {" in lifecycle


def test_permanent_persistence_failures_are_reported_as_such():
    """Ein erreichtes Limit ist kein "Please retry".

    Chat- und Turn-Anlage koennen jetzt mit 403 (Limit, Tier-Gate) oder 429
    (UID-Rate-Limit) enden. Diese Ursachen behebt kein Wiederholen - die
    Session merkt sich den Grund, damit query-send.js ihn statt des
    generischen Retry-Hinweises zeigt.
    """
    script_path = ROOT / "static" / "js" / "chat-session.js"
    node_script = f"""
const assert = require("assert");
const fs = require("fs");
const vm = require("vm");
const {{ webcrypto }} = require("crypto");
global.window = global;
global.crypto = webcrypto;
global.App = {{}};
let nextResponse = null;
global.fetch = async (url) => {{
  if (nextResponse) {{
    const response = nextResponse;
    nextResponse = null;
    return response;
  }}
  if (url === "/chats") {{
    return {{ ok: true, status: 201, json: async () => ({{ chat: {{ id: "c".repeat(32) }} }}) }};
  }}
  return {{ ok: true, status: 201, json: async () => ({{ turn: {{ id: "1".repeat(32) }} }}) }};
}};
vm.runInThisContext(fs.readFileSync({str(script_path)!r}, "utf8"), {{ filename: "chat-session.js" }});
const session = App.chatSession;
const run = (overrides = {{}}) => session.beginRun({{
  question: "Question one",
  mode: "Standard",
  deepSearch: false,
  selectedModels: ["model-a"],
  consensusModel: "Gemini",
  isFollowup: false,
  prepareSucceeded: true,
  ...overrides
}});

(async () => {{
  // Chat-Limit: dauerhafter Grund, wortwoertlich weitergereicht.
  run();
  nextResponse = {{
    ok: false,
    status: 403,
    json: async () => ({{ detail: {{ error_code: "chat_limit_reached", error: "..." }} }})
  }};
  assert.strictEqual(await session.ensurePendingTurn({{
    idToken: "token", question: "Question one", consensusModel: "Gemini"
  }}), null);
  assert.ok(/maximum number of saved conversations/.test(session.lastPersistenceError));

  // Ein neuer Lauf startet ohne den alten Grund.
  run({{ question: "Question two" }});
  assert.strictEqual(session.lastPersistenceError, "");

  // Turn-Limit im bestehenden Chat.
  const ok = await session.ensurePendingTurn({{
    idToken: "token", question: "Question two", consensusModel: "Gemini"
  }});
  assert.ok(ok);
  session.handleConsensusResult({{
    chatId: ok.chatId, turnId: ok.turnId, chatPersisted: true, chatTurnState: "completed"
  }});
  run({{ question: "Follow-up", isFollowup: true }});
  nextResponse = {{
    ok: false,
    status: 403,
    json: async () => ({{ detail: {{ error_code: "turn_limit_reached", error: "..." }} }})
  }};
  assert.strictEqual(await session.ensurePendingTurn({{
    idToken: "token", question: "Follow-up", consensusModel: "Gemini"
  }}), null);
  assert.ok(/maximum length/.test(session.lastPersistenceError));

  // Ein unbekannter/voruebergehender Fehler bleibt ohne eigenen Text, damit
  // der Aufrufer auf den generischen Retry-Hinweis zurueckfaellt.
  run({{ question: "Question three", isFollowup: true }});
  nextResponse = {{ ok: false, status: 503, json: async () => ({{}}) }};
  await session.ensurePendingTurn({{
    idToken: "token", question: "Question three", consensusModel: "Gemini"
  }});
  assert.strictEqual(session.lastPersistenceError, "");
  assert.strictEqual(session.hasUncertainTurn(), true);

  // reset() raeumt den Grund ebenfalls ab.
  session.lastPersistenceError = "stale";
  session.reset();
  assert.strictEqual(session.lastPersistenceError, "");

  console.log("OK");
}})().catch(error => {{
  console.error(error);
  process.exit(1);
}});
"""
    result = subprocess.run(
        ["node", "-e", node_script], capture_output=True, text=True, cwd=ROOT
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_query_send_shows_the_real_reason_instead_of_a_dead_end_retry():
    send = (ROOT / "static" / "js" / "query-send.js").read_text(encoding="utf-8")

    assert "context.chatSession.lastPersistenceError" in send
    # Der generische Hinweis bleibt der Fallback, nicht die einzige Antwort.
    assert '|| "The conversation turn could not be prepared. Please retry."' in send


def test_archived_turns_render_the_same_difference_cards_as_the_live_run():
    consensus = (ROOT / "static" / "js" / "consensus-run.js").read_text(
        encoding="utf-8"
    )
    insights = (ROOT / "static" / "js" / "consensus-insights.js").read_text(
        encoding="utf-8"
    )

    # Ein archivierter Turn traegt dieselben strukturierten Daten wie der
    # Live-Lauf und muss sie auch benutzen — sonst liest sich derselbe Befund
    # im Verlauf als roher Judge-Freitext.
    assert "window.renderStoredDifferenceCards = renderStoredDifferenceCards" in insights
    assert "window.renderStoredDifferenceCards?.(" in consensus
    assert "hasStructuredDifferences || differencesText" in consensus

    stored = insights.split("function renderStoredDifferenceCards", 1)[1].split(
        "\n          }", 1
    )[0]
    assert "Array.isArray(differencesData.differences)" in stored
    assert "static: true" in stored

    # Der Freitext bleibt der Fallback fuer alte Turns ohne strukturierte
    # Daten — ohne die Buchhaltungszeile des Judges.
    assert "function stripBestModelLine(differencesText)" in consensus
    assert "BestModel:" in consensus.split("function stripBestModelLine", 1)[1][:300]
    assert "const differencesText = stripBestModelLine(turnData.differences)" in consensus


def test_archived_difference_cards_carry_no_live_run_controls():
    insights = (ROOT / "static" / "js" / "consensus-insights.js").read_text(
        encoding="utf-8"
    )

    cards = insights.split("function buildDifferenceCards", 1)[1].split(
        "function renderDifferenceCards", 1
    )[0]
    # Sprunglinks zeigten sonst auf die Antwortboxen des NEUESTEN Laufs, und
    # eine Resolve-Runde laeuft immer gegen die Modelle des aktiven Laufs.
    assert "if (!isStatic) {" in cards
    assert "(isStatic && !diff.resolution)" in cards
    assert 'resolveSection.querySelectorAll("button").forEach' in cards
    # Die Modellnamen kommen aus dem Turn, nicht aus den Live-Boxen.
    assert "pos.models.map(labelFor)" in cards
    assert "function storedModelLabeller(modelAnswers)" in (
        (ROOT / "static" / "js" / "consensus-run.js").read_text(encoding="utf-8")
    )
