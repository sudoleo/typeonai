import { initializeApp } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-app.js";
import { getAuth, onAuthStateChanged } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-auth.js";
import { createAdminClient } from "/static/js/admin-api.js?v=20260811-phase6";

const app = initializeApp(window.FIREBASE_CONFIG);
const auth = getAuth(app);
const shareAdminRequest = createAdminClient(auth);

let providers = [];
const limitGroups = [
    {
        title: 'Run Usage Limits (UTC day)',
        fields: [
            ['free_consensus_run_limit', 'Free consensus runs'],
            ['pro_consensus_run_limit', 'Pro consensus runs'],
            ['free_deep_think_run_limit', 'Free Deep Think runs'],
            ['pro_deep_think_run_limit', 'Pro Deep Think runs']
        ]
    },
    {
        title: 'Input / Context Limits',
        fields: [
            ['free_max_words', 'Free input words'],
            ['pro_max_words', 'Pro input words'],
            ['free_deep_search_max_words', 'Free Deep Think input words'],
            ['pro_deep_search_max_words', 'Pro Deep Think input words']
        ]
    },
    {
        title: 'Output Token Limits',
        fields: [
            ['free_max_tokens', 'Free output tokens'],
            ['pro_max_tokens', 'Pro output tokens'],
            ['free_deep_search_max_tokens', 'Free Deep Think output tokens'],
            ['pro_deep_search_max_tokens', 'Pro Deep Think output tokens']
        ]
    },
    {
        title: 'Consensus Limits',
        fields: [
            ['consensus_max_tokens', 'Consensus output tokens'],
            ['differences_max_tokens', 'Differences output tokens'],
            ['coverage_max_tokens', 'Coverage output tokens']
        ]
    },
    {
        title: 'Consensus Watch Limits',
        fields: [
            ['watch_free_active_limit', 'Free active watches'],
            ['watch_pro_active_limit', 'Pro active watches'],
            ['watch_max_runs_per_day', 'Global runs per day'],
            ['watch_daily_interval_requires_pro', 'Daily interval Pro-only (1 = yes, 0 = Free too)']
        ]
    },
    {
        title: 'Share Index Quality Filter',
        fields: [
            ['share_min_consensus_chars', 'Min consensus characters'],
            ['share_min_sources', 'Min sources'],
            ['share_min_models', 'Min models consulted'],
            ['share_question_min_chars', 'Min question characters'],
            ['share_question_max_chars', 'Max question characters']
        ]
    }
];
let globalModelsData = {};

// ==============================
// Tabs
// ==============================
const TAB_IDS = ['models', 'consensus', 'limits', 'api', 'shares', 'watches', 'topics', 'seo'];
function activateTab(tabId) {
    if (!TAB_IDS.includes(tabId)) tabId = 'models';
    if (tabId !== 'api') clearIssuedApiKey();
    TAB_IDS.forEach(id => {
        document.getElementById(`tab-${id}`).hidden = id !== tabId;
        const btn = document.querySelector(`.admin-tabs button[data-tab="${id}"]`);
        btn.classList.toggle('active', id === tabId);
        btn.setAttribute('aria-selected', id === tabId ? 'true' : 'false');
    });
    document.getElementById('adminSavebar').hidden = tabId === 'topics';
    history.replaceState(null, '', `#${tabId}`);
}
document.querySelectorAll('.admin-tabs button').forEach(btn => {
    btn.addEventListener('click', () => activateTab(btn.dataset.tab));
});
activateTab((location.hash || '').replace('#', ''));

// ==============================
// Dirty-Tracking
// ==============================
function markDirty() {
    document.getElementById('adminSavebar').classList.add('is-dirty');
}
function clearDirty() {
    document.getElementById('adminSavebar').classList.remove('is-dirty');
}
// Alle Eingaben in den Konfig-Tabs (nicht Shares) markieren als dirty.
['tab-models', 'tab-consensus', 'tab-limits'].forEach(id => {
    const el = document.getElementById(id);
    el.addEventListener('change', markDirty);
    el.addEventListener('input', markDirty);
});
window.addEventListener('beforeunload', (event) => {
    if (!document.getElementById('adminSavebar').classList.contains('is-dirty')) return;
    event.preventDefault();
    event.returnValue = '';
});
// Aenderungen an den Provider-Listen in die abhaengigen Dropdowns
// (Judges, Consensus-Add) spiegeln.
document.getElementById('tab-models').addEventListener('change', () => {
    renderJudgeSelects();
    renderConsensusAddSelect();
    renderPresetModels();
    renderWatchModelConfig();
});

// ==============================
// Meta-Helfer (Alias-Aufloesung, server-erzwungene Modelle)
// ==============================
function meta() { return globalModelsData.meta || {}; }
function providerLabel(provider) { return (meta().provider_labels || {})[provider] || provider; }
function dependencyReasons(provider, model) {
    return ((((meta().dependencies || {})[provider] || {})[model]) || []);
}
function labelFor(model) { return (meta().labels || {})[model] || ''; }
// Leer, solange der Server keine Auskunft geben konnte. Dann wird nichts
// behauptet, statt faelschlich "laeuft" oder "laeuft nicht" anzuzeigen.
function providerCredentials() { return meta().provider_credentials || {}; }
// Virtuelle IDs senden ein anderes API-Modell (z. B.
// grok-4.3-no-reasoning -> grok-4.3). Sichtbar machen, sonst sieht man
// zwei fast gleich aussehende Eintraege ohne erkennbaren Unterschied.
function apiModelFor(model) { return (meta().api_models || {})[model] || ''; }
function optionTextFor(model) {
    let text = labelFor(model) || model;
    if (text !== model) text += ` (${model})`;
    const apiModel = apiModelFor(model);
    if (apiModel) text += ` → ${apiModel}`;
    return text;
}
function consensusDescription(value) {
    const alias = (meta().aliases || {})[value];
    if (alias) return `alias → ${alias.provider} · ${alias.label} (${alias.model})`;
    const provider = providers.find(p => (globalModelsData[p] || []).includes(value));
    const label = labelFor(value);
    if (provider) return `${provider}${label && label !== value ? ' · ' + label : ''}`;
    return label && label !== value ? label : '';
}
function chip(kind, text, title) {
    const span = document.createElement('span');
    span.className = 'admin-chip' + (kind ? ` ${kind}` : '');
    span.textContent = text;
    if (title) span.title = title;
    return span;
}

function renderWatchModelConfig() {
    const container = document.getElementById('watchModelConfig');
    if (!container) return;
    const currentModels = { free: {}, pro: {} };
    container.querySelectorAll('[data-watch-tier][data-provider]').forEach(select => {
        currentModels[select.dataset.watchTier][select.dataset.provider] = select.value;
    });
    const currentConsensus = {};
    container.querySelectorAll('[data-watch-consensus-tier]').forEach(select => {
        currentConsensus[select.dataset.watchConsensusTier] = select.value;
    });
    container.innerHTML = '';
    ['Provider', 'Free Watch', 'Pro Watch'].forEach(text => {
        const head = document.createElement('div');
        head.className = 'watch-model-head';
        head.textContent = text;
        container.appendChild(head);
    });
    const premium = new Set(globalModelsData.premium || []);
    const savedModels = globalModelsData.watch_models || {};
    const configured = {
        free: { ...(savedModels.free || {}), ...currentModels.free },
        pro: { ...(savedModels.pro || {}), ...currentModels.pro },
    };
    providers.forEach(provider => {
        const label = document.createElement('div');
        label.className = 'watch-model-provider';
        label.textContent = providerLabel(provider);
        container.appendChild(label);
        ['free', 'pro'].forEach(tier => {
            const select = document.createElement('select');
            select.className = 'watch-model-select';
            select.dataset.watchTier = tier;
            select.dataset.provider = provider;
            const disabled = document.createElement('option');
            disabled.value = '';
            disabled.textContent = 'Disabled';
            select.appendChild(disabled);
            (globalModelsData[provider] || []).forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = optionTextFor(model);
                // Free-Watches duerfen keine Premium-Modelle fahren.
                // Die Eintraege bleiben trotzdem sichtbar, damit die
                // Liste vollstaendig ist und der Grund erkennbar wird.
                if (tier === 'free' && premium.has(model)) {
                    option.disabled = true;
                    option.textContent += ' — Pro only';
                }
                select.appendChild(option);
            });
            select.value = ((configured[tier] || {})[provider]) || '';
            select.addEventListener('change', markDirty);
            container.appendChild(select);
        });
    });

    const consensusLabel = document.createElement('div');
    consensusLabel.className = 'watch-model-provider';
    consensusLabel.textContent = 'Consensus engine';
    container.appendChild(consensusLabel);
    const configuredConsensus = {
        ...(globalModelsData.watch_consensus_models || {}),
        ...currentConsensus,
    };
    const consensusModels = consensusListValues();
    ['free', 'pro'].forEach(tier => {
        const select = document.createElement('select');
        select.className = 'watch-model-select';
        select.dataset.watchConsensusTier = tier;
        consensusModels.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model;
            const description = consensusDescription(model);
            if (description) option.textContent += ` — ${description}`;
            if (tier === 'free' && isLockedConsensusModel(model)) {
                option.disabled = true;
                option.textContent += ' — Pro only';
            }
            select.appendChild(option);
        });
        select.value = configuredConsensus[tier] || '';
        select.addEventListener('change', markDirty);
        container.appendChild(select);
    });

    renderWatchEffectiveRun(container, configured, premium);
}

// Ein Filter, der beim Lauf greift, muss dort sichtbar sein, wo man die
// Auswahl trifft. Sonst steht in der Konfiguration eine Modellzahl und im Lauf
// eine andere -- ohne dass irgendwo steht, welcher Provider warum fehlt.
function watchTierOutcome(configured, premium, tier) {
    const credentials = providerCredentials();
    const running = [];
    const skipped = [];
    providers.forEach(provider => {
        const model = (configured[tier] || {})[provider];
        if (!model) return;
        if (tier === 'free' && premium.has(model)) {
            skipped.push({ provider, model, reason: 'Pro only in the Free tier' });
            return;
        }
        if (credentials[provider] === false) {
            skipped.push({ provider, model, reason: 'no server credential' });
            return;
        }
        running.push({ provider, model });
    });
    return { running, skipped };
}

function renderWatchEffectiveRun(container, configured, premium) {
    const label = document.createElement('div');
    label.className = 'watch-model-provider';
    label.textContent = 'Actually runs';
    container.appendChild(label);
    ['free', 'pro'].forEach(tier => {
        const cell = document.createElement('div');
        cell.className = 'watch-effective-run';
        const outcome = watchTierOutcome(configured, premium, tier);
        const summary = document.createElement('div');
        summary.className = 'watch-effective-summary';
        summary.textContent = outcome.running.length
            ? `${outcome.running.length} providers: ` +
              outcome.running.map(item => item.provider).join(', ')
            : 'No provider left';
        cell.appendChild(summary);
        if (outcome.running.length < 2) {
            cell.appendChild(chip(
                'warn',
                'Needs at least 2',
                'A run with fewer than two answers cannot be compared.',
            ));
        }
        outcome.skipped.forEach(item => {
            cell.appendChild(chip(
                'warn',
                `${item.provider} skipped`,
                `${item.model} is configured but will not run: ${item.reason}.`,
            ));
        });
        container.appendChild(cell);
    });
}

function currentPresetModels() {
    const result = {};
    document.querySelectorAll('[data-preset-id]').forEach(select => {
        const presetId = select.dataset.presetId;
        if (!result[presetId]) result[presetId] = { answers: {} };
        if (select.dataset.presetSlot === 'consensus') {
            if (select.value) result[presetId].consensus = select.value;
            return;
        }
        if (select.dataset.presetAnswerIndex === undefined || !select.value) return;
        try {
            const [provider, model] = JSON.parse(select.value);
            if (provider && model) result[presetId].answers[provider] = model;
        } catch (_) {}
    });
    return result;
}

function isLockedConsensusModel(model) {
    const alias = (meta().aliases || {})[model];
    if (alias && model.endsWith('-Pro')) return true;
    return (globalModelsData.premium || []).includes(model);
}

function appendPresetOption(select, value, label, locked, showValue = true) {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = label || value;
    if (showValue && option.textContent !== value) option.textContent += ` (${value})`;
    const apiModel = showValue ? apiModelFor(value) : '';
    if (apiModel) option.textContent += ` → ${apiModel}`;
    // Daily/Balanced sind Free-faehig und duerfen keine Premium-Modelle
    // setzen. Sichtbar lassen statt ausblenden: sonst wirkt die Liste
    // unvollstaendig, ohne dass der Grund erkennbar waere.
    if (locked) {
        option.disabled = true;
        option.textContent += ' — Pro only';
    }
    select.appendChild(option);
}

function renderPresetModels() {
    const container = document.getElementById('presetModelsContainer');
    if (!container) return;
    const chosenNow = currentPresetModels();
    const saved = globalModelsData.preset_models || {};
    const premium = new Set(globalModelsData.premium || []);
    container.innerHTML = '';

    (meta().preset_definitions || []).forEach(definition => {
        const configured = chosenNow[definition.id] || saved[definition.id] || {};
        const configuredAnswers = configured.answers || Object.fromEntries(
            providers.filter(provider => configured[provider]).map(provider => [provider, configured[provider]])
        );
        const answerEntries = Object.entries(configuredAnswers).slice(0, 6);
        const card = document.createElement('div');
        card.className = 'preset-model-card';
        const title = document.createElement('h4');
        title.textContent = definition.label;
        if (definition.pro_only) title.appendChild(chip('', 'Pro', 'This preset is available to Pro users only.'));
        card.appendChild(title);

        for (let index = 0; index < 6; index += 1) {
            const field = document.createElement('div');
            field.className = 'preset-model-field';
            const label = document.createElement('label');
            label.textContent = `Answer ${index + 1}`;
            const select = document.createElement('select');
            select.dataset.presetId = definition.id;
            select.dataset.presetAnswerIndex = String(index);
            select.setAttribute('aria-label', `${definition.label} answer ${index + 1} model`);
            providers.forEach(provider => {
                currentProviderModels(provider).forEach(model => {
                    appendPresetOption(
                        select,
                        JSON.stringify([provider, model]),
                        `${providerLabel(provider)} · ${labelFor(model)}`,
                        !definition.pro_only && premium.has(model),
                        false,
                    );
                });
            });
            const selected = answerEntries[index];
            select.value = selected ? JSON.stringify(selected) : '';
            select.addEventListener('change', markDirty);
            field.appendChild(label);
            field.appendChild(select);
            card.appendChild(field);
        }

        const consensusField = document.createElement('div');
        consensusField.className = 'preset-model-field';
        const consensusLabel = document.createElement('label');
        consensusLabel.textContent = 'Consensus';
        const consensusSelect = document.createElement('select');
        consensusSelect.dataset.presetId = definition.id;
        consensusSelect.dataset.presetSlot = 'consensus';
        consensusSelect.setAttribute('aria-label', `${definition.label} consensus model`);
        consensusListValues().forEach(model => {
            appendPresetOption(consensusSelect, model, consensusDescription(model),
                !definition.pro_only && isLockedConsensusModel(model));
        });
        consensusSelect.value = configured.consensus || '';
        consensusSelect.addEventListener('change', markDirty);
        consensusField.append(consensusLabel, consensusSelect);
        card.appendChild(consensusField);
        container.appendChild(card);
    });
}

// ==============================
// Rendering
// ==============================
function renderUI() {
    renderLimits();

    const container = document.getElementById('providersContainer');
    container.innerHTML = '';

    const premiumSet = new Set(globalModelsData.premium || []);
    const consensusSet = new Set(globalModelsData.consensus || []);
    const defaults = globalModelsData.defaults || {};

    providers.forEach(p => {
        const section = document.createElement('div');
        section.className = 'admin-section';

        const title = document.createElement('h3');
        title.textContent = providerLabel(p);
        section.appendChild(title);

        const listContainer = document.createElement('div');
        listContainer.id = `list-${p}`;

        const models = globalModelsData[p] || [];
        models.forEach(m => {
            const row = createModelRow(p, m, premiumSet.has(m), consensusSet.has(m), defaults[p] === m);
            listContainer.appendChild(row);
        });

        section.appendChild(listContainer);

        const addBtn = document.createElement('button');
        addBtn.type = 'button';
        addBtn.className = 'add-btn';
        addBtn.textContent = '+ Add Model';
        addBtn.onclick = () => {
            listContainer.appendChild(createModelRow(p, '', false, false, false));
            markDirty();
        };
        section.appendChild(addBtn);

        container.appendChild(section);
    });

    // Nach den Provider-Listen rendern, damit Add-Dropdown und
    // Alias-Aufloesung den aktuellen DOM-Stand sehen.
    renderConsensusModels();
    renderPresetModels();
    renderDeepThinkSelect();
    renderJudgeSelects();
    renderWatchModelConfig();
}

function createModelRow(provider, modelName, isPremium, isConsensus, isDefault) {
    const row = document.createElement('div');
    row.className = 'model-row';

    const dependencies = modelName ? dependencyReasons(provider, modelName) : [];

    const upBtn = document.createElement('button');
    upBtn.type = 'button';
    upBtn.className = 'icon-btn';
    upBtn.textContent = '↑';
    upBtn.title = 'Move up (picker order)';
    upBtn.onclick = () => { moveRow(row, -1); markDirty(); };

    const downBtn = document.createElement('button');
    downBtn.type = 'button';
    downBtn.className = 'icon-btn';
    downBtn.textContent = '↓';
    downBtn.title = 'Move down (picker order)';
    downBtn.onclick = () => { moveRow(row, 1); markDirty(); };

    const input = document.createElement('input');
    input.type = 'text';
    input.value = modelName;
    input.placeholder = 'Model identifier (e.g. gpt-5.5)';
    const modelLabel = modelName ? labelFor(modelName) : '';
    if (modelLabel && modelLabel !== modelName) {
        input.title = (input.title ? input.title + ' · ' : '') + `Shown as “${modelLabel}”`;
    }
    input.addEventListener('focus', () => {
        row.dataset.previousModelName = input.value.trim();
    });
    input.addEventListener('change', () => {
        if (!consensusCheckbox.checked) return;
        const previous = row.dataset.previousModelName || '';
        if (previous && previous !== input.value.trim()) {
            removeConsensusListValue(previous);
        }
        addConsensusListValue(input.value.trim());
        row.dataset.previousModelName = input.value.trim();
    });

    const flags = document.createElement('div');
    flags.className = 'model-flags';

    if (dependencies.length) {
        flags.appendChild(chip(
            'required',
            'In use',
            `Referenced by: ${dependencies.join(', ')}. You can remove it after updating those selections.`
        ));
    }

    const defaultLabel = document.createElement('label');
    defaultLabel.title = 'Free default for this provider (shown before a manual pick). Must not be a Premium model.';
    const defaultRadio = document.createElement('input');
    defaultRadio.type = 'radio';
    defaultRadio.className = 'default-radio';
    defaultRadio.name = `default-${provider}`;
    defaultRadio.checked = !!isDefault;
    defaultLabel.appendChild(defaultRadio);
    defaultLabel.appendChild(document.createTextNode(' Default'));

    const premiumLabel = document.createElement('label');
    const premiumCheckbox = document.createElement('input');
    premiumCheckbox.type = 'checkbox';
    premiumCheckbox.className = 'premium-checkbox';
    premiumCheckbox.checked = isPremium;
    premiumLabel.appendChild(premiumCheckbox);
    premiumLabel.appendChild(document.createTextNode(' Premium'));

    const consensusLabel = document.createElement('label');
    consensusLabel.title = 'Offer this model in the Consensus picker (ordered in the Consensus tab).';
    const consensusCheckbox = document.createElement('input');
    consensusCheckbox.type = 'checkbox';
    consensusCheckbox.className = 'consensus-checkbox';
    consensusCheckbox.checked = isConsensus;
    consensusCheckbox.addEventListener('change', () => {
        if (consensusCheckbox.checked) {
            addConsensusListValue(input.value.trim());
        } else {
            removeConsensusListValue(input.value.trim());
        }
    });
    consensusLabel.appendChild(consensusCheckbox);
    consensusLabel.appendChild(document.createTextNode(' Consensus'));

    const removeBtn = document.createElement('button');
    removeBtn.type = 'button';
    removeBtn.className = 'icon-btn danger';
    removeBtn.textContent = '✕';
    removeBtn.title = dependencies.length
        ? `Remove model; then update: ${dependencies.join(', ')}`
        : 'Remove model';
    removeBtn.disabled = false;
    removeBtn.onclick = () => {
        if (consensusCheckbox.checked) removeConsensusListValue(input.value.trim());
        row.remove();
        renderJudgeSelects();
        renderConsensusAddSelect();
        renderPresetModels();
        renderDeepThinkSelect();
        renderWatchModelConfig();
        markDirty();
    };

    row.appendChild(upBtn);
    row.appendChild(downBtn);
    row.appendChild(input);
    row.appendChild(flags);
    row.appendChild(defaultLabel);
    row.appendChild(premiumLabel);
    row.appendChild(consensusLabel);
    row.appendChild(removeBtn);

    return row;
}

function moveRow(row, direction) {
    if (!row) return;
    if (direction < 0 && row.previousElementSibling) {
        row.parentNode.insertBefore(row, row.previousElementSibling);
    } else if (direction > 0 && row.nextElementSibling) {
        row.parentNode.insertBefore(row.nextElementSibling, row);
    }
}

// ==============================
// Consensus & Deep Think
// ==============================
function renderConsensusModels() {
    const listContainer = document.getElementById('consensusModelsList');
    listContainer.innerHTML = '';
    const models = globalModelsData.consensus || [];
    models.forEach(model => {
        listContainer.appendChild(createConsensusModelRow(model));
    });
    renderConsensusAddSelect();
}

function consensusListValues() {
    return Array.from(document.querySelectorAll('#consensusModelsList .consensus-row'))
        .map(row => (row.dataset.value || '').trim())
        .filter(Boolean);
}

function createConsensusModelRow(modelName) {
    const row = document.createElement('div');
    row.className = 'consensus-row';
    row.dataset.value = modelName;

    const forcedFirst = modelName === meta().consensus_forced_first;
    const isDeepThink = modelName === currentDeepThinkModel();

    const upBtn = document.createElement('button');
    upBtn.type = 'button';
    upBtn.className = 'icon-btn';
    upBtn.textContent = '↑';
    upBtn.title = 'Move up (picker order)';
    upBtn.onclick = () => { moveConsensusRow(row, -1); markDirty(); };

    const downBtn = document.createElement('button');
    downBtn.type = 'button';
    downBtn.className = 'icon-btn';
    downBtn.textContent = '↓';
    downBtn.title = 'Move down (picker order)';
    downBtn.onclick = () => { moveConsensusRow(row, 1); markDirty(); };

    const value = document.createElement('span');
    value.className = 'consensus-value';
    value.textContent = modelName;

    const desc = document.createElement('span');
    desc.className = 'consensus-desc';
    desc.textContent = consensusDescription(modelName);
    desc.title = desc.textContent;

    const removeBtn = document.createElement('button');
    removeBtn.type = 'button';
    removeBtn.className = 'icon-btn danger';
    removeBtn.textContent = '✕';
    if (forcedFirst) {
        removeBtn.disabled = true;
        removeBtn.title = 'Server-enforced: this engine is always available (re-inserted on save).';
    } else if (isDeepThink) {
        removeBtn.disabled = true;
        removeBtn.title = 'Currently the Deep Think model — pick a different Deep Think model first.';
    } else {
        removeBtn.title = 'Remove from Consensus picker';
    }
    removeBtn.onclick = () => {
        setProviderConsensusChecked(modelName, false);
        row.remove();
        renderConsensusAddSelect();
        renderDeepThinkSelect();
        markDirty();
    };

    row.appendChild(upBtn);
    row.appendChild(downBtn);
    row.appendChild(value);
    row.appendChild(desc);
    if (forcedFirst) row.appendChild(chip('required', 'Required', 'Always kept in the list by the server.'));
    if (isDeepThink) row.appendChild(chip('deepthink', 'Deep Think', 'Deep Think switches the Consensus engine to this model.'));
    row.appendChild(removeBtn);

    return row;
}

function moveConsensusRow(row, direction) {
    moveRow(row, direction);
}

function findProviderConsensusCheckbox(modelName) {
    if (!modelName) return null;
    for (const row of document.querySelectorAll('#providersContainer .model-row')) {
        const input = row.querySelector('input[type="text"]');
        if (input && input.value.trim() === modelName) {
            return row.querySelector('.consensus-checkbox');
        }
    }
    return null;
}

function setProviderConsensusChecked(modelName, checked) {
    const checkbox = findProviderConsensusCheckbox(modelName);
    if (checkbox) checkbox.checked = checked;
}

function addConsensusListValue(modelName) {
    const value = (modelName || '').trim();
    if (!value || consensusListValues().includes(value)) return;
    document.getElementById('consensusModelsList').appendChild(createConsensusModelRow(value));
    setProviderConsensusChecked(value, true);
    renderConsensusAddSelect();
    renderDeepThinkSelect();
    renderPresetModels();
    renderWatchModelConfig();
}

function removeConsensusListValue(modelName) {
    const value = (modelName || '').trim();
    if (!value) return;
    document.querySelectorAll('#consensusModelsList .consensus-row').forEach(row => {
        if ((row.dataset.value || '').trim() === value) row.remove();
    });
    setProviderConsensusChecked(value, false);
    renderConsensusAddSelect();
    renderDeepThinkSelect();
    renderPresetModels();
    renderWatchModelConfig();
}

// Kandidaten fuer das Add-Dropdown: Aliase + direkte Modell-IDs aus den
// Provider-Listen, die noch nicht in der Consensus-Liste stehen.
function renderConsensusAddSelect() {
    const select = document.getElementById('consensusAddSelect');
    if (!select) return;
    const existing = new Set(consensusListValues());
    select.innerHTML = '';

    const placeholder = document.createElement('option');
    placeholder.value = '';
    placeholder.textContent = 'Add engine…';
    select.appendChild(placeholder);

    const aliasGroup = document.createElement('optgroup');
    aliasGroup.label = 'Aliases (auto-track provider defaults)';
    Object.keys(meta().aliases || {}).forEach(alias => {
        if (existing.has(alias)) return;
        const opt = document.createElement('option');
        opt.value = alias;
        opt.textContent = `${alias} — ${consensusDescription(alias)}`;
        aliasGroup.appendChild(opt);
    });
    if (aliasGroup.children.length) select.appendChild(aliasGroup);

    providers.forEach(p => {
        const group = document.createElement('optgroup');
        group.label = providerLabel(p);
        (currentProviderModels(p) || []).forEach(model => {
            if (!model || existing.has(model)) return;
            const opt = document.createElement('option');
            opt.value = model;
            const label = labelFor(model);
            opt.textContent = label && label !== model ? `${model} — ${label}` : model;
            group.appendChild(opt);
        });
        if (group.children.length) select.appendChild(group);
    });
}

// Provider-Modelle aus dem aktuellen DOM (inkl. ungespeicherter Zeilen).
function currentProviderModels(provider) {
    const listContainer = document.getElementById(`list-${provider}`);
    if (!listContainer) return globalModelsData[provider] || [];
    return Array.from(listContainer.querySelectorAll('.model-row input[type="text"]'))
        .map(input => input.value.trim())
        .filter(Boolean);
}

function currentDeepThinkModel() {
    const select = document.getElementById('deepThinkModelSelect');
    if (select && select.value) return select.value;
    return globalModelsData.deep_think_model || (meta().deep_think_fallback || '');
}

function renderDeepThinkSelect() {
    const select = document.getElementById('deepThinkModelSelect');
    if (!select) return;
    const chosen = currentDeepThinkModel();
    select.innerHTML = '';
    consensusListValues().forEach(value => {
        const opt = document.createElement('option');
        opt.value = value;
        const descText = consensusDescription(value);
        opt.textContent = descText ? `${value} — ${descText}` : value;
        if (value === chosen) opt.selected = true;
        select.appendChild(opt);
    });
    // Deep-Think-Badges in der Liste aktualisieren, ohne alles neu zu bauen.
    document.querySelectorAll('#consensusModelsList .consensus-row').forEach(row => {
        const isDeepThink = (row.dataset.value || '') === select.value;
        const badge = row.querySelector('.admin-chip.deepthink');
        if (isDeepThink && !badge) {
            row.insertBefore(chip('deepthink', 'Deep Think', 'Deep Think switches the Consensus engine to this model.'), row.lastElementChild);
        } else if (!isDeepThink && badge) {
            badge.remove();
        }
        const removeBtn = row.lastElementChild;
        const forcedFirst = (row.dataset.value || '') === meta().consensus_forced_first;
        if (!forcedFirst) {
            removeBtn.disabled = isDeepThink;
            removeBtn.title = isDeepThink
                ? 'Currently the Deep Think model — pick a different Deep Think model first.'
                : 'Remove from Consensus picker';
        }
    });
}

// ==============================
// Differences Judges
// ==============================
function currentJudgeModels() {
    const result = {};
    document.querySelectorAll('[data-judge-provider]').forEach(select => {
        if (select.value) result[select.dataset.judgeProvider] = select.value;
    });
    return result;
}

function currentProJudgeModels() {
    const result = {};
    document.querySelectorAll('[data-projudge-provider]').forEach(select => {
        if (select.value) result[select.dataset.projudgeProvider] = select.value;
    });
    return result;
}

function currentChatMemoryModels() {
    const result = {};
    document.querySelectorAll('[data-chatmemory-provider]').forEach(select => {
        if (select.value) result[select.dataset.chatmemoryProvider] = select.value;
    });
    return result;
}

function currentJudgeFamilies() {
    const result = {};
    document.querySelectorAll('[data-judgefam-engine]').forEach(select => {
        if (select.value) result[select.dataset.judgefamEngine] = select.value;
    });
    return result;
}

function buildJudgeModelSelect(provider, chosen, defaultModel, datasetKey, ariaText) {
    const select = document.createElement('select');
    select.dataset[datasetKey] = provider;
    select.setAttribute('aria-label', ariaText);
    currentProviderModels(provider).forEach(model => {
        const opt = document.createElement('option');
        opt.value = model;
        const lbl = labelFor(model);
        opt.textContent = lbl && lbl !== model ? `${model} — ${lbl}` : model;
        const apiModel = apiModelFor(model);
        if (apiModel) opt.textContent += ` → ${apiModel}`;
        if (model === defaultModel) opt.textContent += ' (server default)';
        if (model === chosen) opt.selected = true;
        select.appendChild(opt);
    });
    return select;
}

function renderJudgeSelects() {
    const container = document.getElementById('judgeModelsContainer');
    if (!container) return;
    // Ungespeicherte Auswahl bei Re-Renders erhalten.
    const chosenNow = currentJudgeModels();
    const chosenProNow = currentProJudgeModels();
    container.innerHTML = '';
    const judgeDefaults = meta().judge_defaults || {};
    const proDefaults = meta().judge_pro_defaults || {};
    const saved = globalModelsData.judge_models || {};
    const savedPro = globalModelsData.judge_models_pro || {};

    const head = document.createElement('div');
    head.className = 'judge-row judge-head';
    ['Family', 'Standard judge', 'Pro judge (reduced effort)'].forEach(text => {
        const cell = document.createElement('span');
        cell.textContent = text;
        head.appendChild(cell);
    });
    container.appendChild(head);

    providers.forEach(p => {
        const row = document.createElement('div');
        row.className = 'judge-row';

        const label = document.createElement('label');
        label.textContent = providerLabel(p);

        const chosen = chosenNow[p] || saved[p] || judgeDefaults[p] || '';
        const chosenPro = chosenProNow[p] || savedPro[p] || proDefaults[p] || '';

        row.appendChild(label);
        row.appendChild(buildJudgeModelSelect(
            p, chosen, judgeDefaults[p], 'judgeProvider', `Standard differences judge for ${p}`));
        row.appendChild(buildJudgeModelSelect(
            p, chosenPro, proDefaults[p], 'projudgeProvider', `Pro differences judge for ${p}`));
        container.appendChild(row);
    });

    renderJudgeFamilies();
    renderChatMemorySelects();
}

// Chat-Memory je Provider-Familie. Die Familie selbst waehlt der Nutzer
// mit der Consensus-Engine — hier steht nur, welches Modell dieser
// Familie die Memory laengerer Chats fortschreibt.
function renderChatMemorySelects() {
    const container = document.getElementById('chatMemoryModelsContainer');
    if (!container) return;
    const chosenNow = currentChatMemoryModels();
    container.innerHTML = '';
    const defaults = meta().chat_memory_defaults || {};
    const saved = globalModelsData.chat_memory_models || {};

    const head = document.createElement('div');
    head.className = 'judge-fam-row judge-head';
    ['Family', 'Chat memory model'].forEach(text => {
        const cell = document.createElement('span');
        cell.textContent = text;
        head.appendChild(cell);
    });
    container.appendChild(head);

    providers.forEach(p => {
        const row = document.createElement('div');
        row.className = 'judge-fam-row';

        const label = document.createElement('label');
        label.textContent = providerLabel(p);

        const chosen = chosenNow[p] || saved[p] || defaults[p] || '';
        row.appendChild(label);
        row.appendChild(buildJudgeModelSelect(
            p, chosen, defaults[p], 'chatmemoryProvider', `Chat memory model for ${p}`));
        container.appendChild(row);
    });
}

function renderJudgeFamilies() {
    const container = document.getElementById('judgeFamiliesContainer');
    if (!container) return;
    const chosenNow = currentJudgeFamilies();
    container.innerHTML = '';
    const saved = globalModelsData.judge_families || {};
    const priority = meta().judge_priority || [];

    providers.forEach(engine => {
        const row = document.createElement('div');
        row.className = 'judge-fam-row';

        const label = document.createElement('label');
        label.textContent = `${providerLabel(engine)} engine`;

        const select = document.createElement('select');
        select.dataset.judgefamEngine = engine;
        select.setAttribute('aria-label', `Judge family for ${engine} engines`);

        const autoOrder = priority.filter(p => p !== engine).join(' → ');
        const autoOpt = document.createElement('option');
        autoOpt.value = '';
        autoOpt.textContent = `Auto — first available: ${autoOrder}`;
        select.appendChild(autoOpt);

        const chosen = chosenNow[engine] || saved[engine] || '';
        providers.forEach(judgeFamily => {
            if (judgeFamily === engine) return;
            const opt = document.createElement('option');
            opt.value = judgeFamily;
            opt.textContent = providerLabel(judgeFamily);
            if (judgeFamily === chosen) opt.selected = true;
            select.appendChild(opt);
        });

        row.appendChild(label);
        row.appendChild(select);
        container.appendChild(row);
    });
}

function renderLimits() {
    const container = document.getElementById('limitsContainer');
    container.innerHTML = '';
    const limits = globalModelsData.limits || {};

    limitGroups.forEach(group => {
        const section = document.createElement('div');
        section.className = 'admin-section';

        const title = document.createElement('h3');
        title.textContent = group.title;
        section.appendChild(title);

        group.fields.forEach(([key, labelText]) => {
            const row = document.createElement('div');
            row.className = 'limit-row';

            const label = document.createElement('label');
            label.htmlFor = `limit-${key}`;
            label.textContent = labelText;

            const input = document.createElement('input');
            input.type = 'number';
            input.min = '0';
            input.step = '1';
            input.id = `limit-${key}`;
            input.dataset.limitKey = key;
            input.value = Number.isFinite(Number(limits[key])) ? limits[key] : 0;

            row.appendChild(label);
            row.appendChild(input);
            section.appendChild(row);
        });

        container.appendChild(section);
    });

    const memory = globalModelsData.memory_edit || {};
    const section = document.createElement('div');
    section.className = 'admin-section';
    const title = document.createElement('h3');
    title.textContent = 'Edit Memory';
    section.appendChild(title);
    const hint = document.createElement('p');
    hint.className = 'section-hint';
    hint.textContent = 'Server-authoritative Luna patching, plan limits and persistent cost controls.';
    section.appendChild(hint);

    const enabledRow = document.createElement('div');
    enabledRow.className = 'limit-row';
    const enabledLabel = document.createElement('label');
    enabledLabel.htmlFor = 'memory-edit-enabled';
    enabledLabel.textContent = 'Feature enabled';
    const enabled = document.createElement('input');
    enabled.type = 'checkbox';
    enabled.id = 'memory-edit-enabled';
    enabled.dataset.memoryEditKey = 'memory_edit_enabled';
    enabled.checked = memory.memory_edit_enabled === true;
    enabledRow.append(enabledLabel, enabled);
    section.appendChild(enabledRow);

    const modelRow = document.createElement('div');
    modelRow.className = 'limit-row';
    const modelLabel = document.createElement('label');
    modelLabel.htmlFor = 'memory-edit-model';
    modelLabel.textContent = 'OpenAI model';
    const modelSelect = document.createElement('select');
    modelSelect.id = 'memory-edit-model';
    modelSelect.dataset.memoryEditKey = 'memory_edit_model';
    const memoryModels = [...(globalModelsData.openai || [])];
    if (memory.memory_edit_model && !memoryModels.includes(memory.memory_edit_model)) {
        memoryModels.unshift(memory.memory_edit_model);
    }
    memoryModels.forEach(model => {
        const option = document.createElement('option');
        option.value = model;
        option.textContent = labelFor(model);
        modelSelect.appendChild(option);
    });
    modelSelect.value = memory.memory_edit_model || '';
    modelRow.append(modelLabel, modelSelect);
    section.appendChild(modelRow);

    const fields = [
        ['memory_free_chars', 'Free Memory note characters'],
        ['memory_pro_chars', 'Pro Memory note characters'],
        ['memory_free_ai_edits_daily', 'Free AI edits / UTC day'],
        ['memory_pro_ai_edits_daily', 'Pro AI edits / UTC day'],
        ['memory_ai_edits_per_minute', 'AI edits / minute'],
        ['memory_global_calls_daily', 'Global calls / UTC day'],
        ['memory_edit_input_chars', 'Correction input characters'],
        ['memory_edit_output_tokens', 'Patch output tokens'],
        ['memory_edit_timeout_seconds', 'Provider timeout seconds']
    ];
    fields.forEach(([key, labelText]) => {
        const row = document.createElement('div');
        row.className = 'limit-row';
        const label = document.createElement('label');
        label.htmlFor = `memory-edit-${key}`;
        label.textContent = labelText;
        const input = document.createElement('input');
        input.type = 'number';
        input.min = '0';
        input.step = '1';
        input.id = `memory-edit-${key}`;
        input.dataset.memoryEditKey = key;
        input.value = Number.isFinite(Number(memory[key])) ? memory[key] : 0;
        row.append(label, input);
        section.appendChild(row);
    });
    container.appendChild(section);
}

// ==============================
// Laden & Speichern
// ==============================
function setStatus(message, isError) {
    const el = document.getElementById('statusMessage');
    el.textContent = message || '';
    el.className = isError ? 'error' : 'success';
}

async function fetchModels(idToken) {
    try {
        const response = await fetch('/api/admin/models', {
            headers: { 'Authorization': `Bearer ${idToken}` }
        });
        if (!response.ok) {
            throw new Error('Failed to fetch models. Admin access required.');
        }
        globalModelsData = await response.json();
        providers = Array.isArray(meta().provider_keys)
            ? meta().provider_keys.slice()
            : [];
        renderUI();
        clearDirty();
    } catch (err) {
        setStatus(err.message, true);
    }
}

async function reloadModels() {
    const user = auth.currentUser;
    if (!user) return;
    setStatus('Reloading…', false);
    await fetchModels(await user.getIdToken());
    setStatus('', false);
}

async function saveModels() {
    const user = auth.currentUser;
    if (!user) return;
    const idToken = await user.getIdToken();

    const data = {
        premium: [],
        consensus: consensusListValues(),
        preset_models: currentPresetModels(),
        deep_think_model: currentDeepThinkModel(),
        judge_models: currentJudgeModels(),
        judge_models_pro: currentProJudgeModels(),
        judge_families: currentJudgeFamilies(),
        chat_memory_models: currentChatMemoryModels(),
        watch_models: { free: {}, pro: {} },
        watch_consensus_models: { free: '', pro: '' },
        defaults: {},
        limits: {},
        memory_edit: {}
    };
    function addConsensusValue(modelName) {
        if (modelName && !data.consensus.includes(modelName)) {
            data.consensus.push(modelName);
        }
    }
    document.querySelectorAll('[data-limit-key]').forEach(input => {
        const value = parseInt(input.value, 10);
        data.limits[input.dataset.limitKey] = Number.isFinite(value) && value >= 0 ? value : 0;
    });
    document.querySelectorAll('[data-memory-edit-key]').forEach(input => {
        const key = input.dataset.memoryEditKey;
        if (input.type === 'checkbox') data.memory_edit[key] = input.checked;
        else if (input.tagName === 'SELECT') data.memory_edit[key] = input.value;
        else data.memory_edit[key] = Number.parseInt(input.value, 10);
    });

    providers.forEach(p => {
        data[p] = [];
        const listContainer = document.getElementById(`list-${p}`);
        const rows = listContainer.querySelectorAll('.model-row');
        rows.forEach(row => {
            const input = row.querySelector('input[type="text"]');
            const premiumCheckbox = row.querySelector('.premium-checkbox');
            const consensusCheckbox = row.querySelector('.consensus-checkbox');
            const defaultRadio = row.querySelector('.default-radio');
            const modelName = input.value.trim();

            if (modelName) {
                // Reihenfolge = DOM-Reihenfolge der Zeilen (per ↑/↓ sortierbar).
                data[p].push(modelName);
                if (premiumCheckbox.checked) {
                    data.premium.push(modelName);
                }
                if (consensusCheckbox.checked) {
                    addConsensusValue(modelName);
                }
                if (defaultRadio && defaultRadio.checked) {
                    data.defaults[p] = modelName;
                }
            }
        });
    });

    document.querySelectorAll('[data-watch-tier][data-provider]').forEach(select => {
        if (select.value) data.watch_models[select.dataset.watchTier][select.dataset.provider] = select.value;
    });
    document.querySelectorAll('[data-watch-consensus-tier]').forEach(select => {
        data.watch_consensus_models[select.dataset.watchConsensusTier] = select.value;
    });
    for (const tier of ['free', 'pro']) {
        if (Object.keys(data.watch_models[tier]).length < 2) {
            setStatus(`Select at least two ${tier} Watch models.`, true);
            return;
        }
        if (!data.watch_consensus_models[tier]) {
            setStatus(`Select a ${tier} Watch consensus engine.`, true);
            return;
        }
    }
    for (const definition of (meta().preset_definitions || [])) {
        const configured = data.preset_models[definition.id] || {};
        const answers = configured.answers || {};
        if (Object.keys(answers).length !== 6 || !configured.consensus) {
            setStatus(`${definition.label} must select six different model families and one consensus engine.`, true);
            return;
        }
    }

    try {
        const response = await fetch('/api/admin/models', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${idToken}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        if (response.ok) {
            setStatus('Configuration saved.', false);
            clearDirty();
            // Normalisierten Server-Stand nachladen (ensures/drops sichtbar machen).
            await fetchModels(idToken);
            setTimeout(() => setStatus('', false), 4000);
        } else {
            const resData = await response.json();
            throw new Error(resData.detail || 'Failed to update models');
        }
    } catch (err) {
        setStatus(err.message, true);
    }
}

document.getElementById('saveBtn').addEventListener('click', saveModels);
document.getElementById('reloadBtn').addEventListener('click', reloadModels);
document.getElementById('consensusAddBtn').addEventListener('click', () => {
    const select = document.getElementById('consensusAddSelect');
    if (select.value) {
        addConsensusListValue(select.value);
        markDirty();
    }
});
document.getElementById('deepThinkModelSelect').addEventListener('change', renderDeepThinkSelect);

// === Shared Pages Moderation ===
let currentSharesFilter = 'reported';

function sharesStatus(message, isError) {
    const el = document.getElementById('sharesStatus');
    el.textContent = message || '';
    el.className = isError ? 'error' : 'success';
}

async function loadShares(filter) {
    currentSharesFilter = filter;
    sharesStatus('Loading…', false);
    let data;
    try {
        data = await shareAdminRequest('GET', `/api/admin/shares?filter=${filter}`);
    } catch (err) {
        sharesStatus(err.message, true);
        return;
    }
    sharesStatus('', false);
    renderShares(data.shares || [], data.site_url || '');
}

async function moderateShare(shareId, payload, confirmText) {
    if (confirmText && !confirm(confirmText)) return;
    try {
        await shareAdminRequest('POST', `/api/admin/shares/${encodeURIComponent(shareId)}/moderate`, payload);
        await loadShares(currentSharesFilter);
    } catch (err) {
        sharesStatus(err.message, true);
    }
}

async function deleteShare(shareId, question, statusFn = sharesStatus) {
    const label = question || shareId;
    if (!confirm(`Permanently delete "${label}"? The page, its Watch schedule, history, and followers will be removed immediately. This cannot be undone.`)) return;
    statusFn('Deleting page…', false);
    try {
        await shareAdminRequest('DELETE', `/api/admin/shares/${encodeURIComponent(shareId)}`);
        await Promise.all([
            loadShares(currentSharesFilter),
            loadPublisherWatches(),
            loadAdminWatches()
        ]);
    } catch (err) {
        statusFn(err.message, true);
    }
}

function renderShares(shares, siteUrl) {
    const container = document.getElementById('sharesContainer');
    container.innerHTML = '';
    if (!shares.length) {
        container.textContent = currentSharesFilter === 'reported'
            ? 'No reported shares.' : 'No shares found.';
        return;
    }
    shares.forEach(share => {
        const row = document.createElement('div');
        row.className = 'share-mod-row';

        const link = document.createElement('a');
        link.className = 'share-mod-question';
        link.textContent = share.question || '(untitled)';
        link.href = siteUrl + share.path;
        link.target = '_blank';
        link.rel = 'noopener';
        row.appendChild(link);

        if (share.needs_review) row.appendChild(badge('review', 'needs review'));
        if (share.index_requested) row.appendChild(badge('review', 'listing requested', 'The owner asked for this page to be indexed on Google.'));
        if (share.reports_count > 0) {
            const reasons = Object.entries(share.report_reasons || {})
                .map(([k, v]) => `${k}: ${v}`).join(', ');
            row.appendChild(badge('reported', `${share.reports_count} report(s)`, reasons));
        }
        row.appendChild(badge('', share.status));
        if (share.visibility === 'private') row.appendChild(badge('', 'private'));
        if (share.indexed) row.appendChild(badge('indexed', 'indexed'));
        else if (share.visibility !== 'private' && share.index_eligible && share.status === 'active') row.appendChild(badge('', 'eligible'));

        const actions = document.createElement('div');
        actions.className = 'share-mod-actions';
        if (share.status === 'blocked') {
            actions.appendChild(actionBtn('Unblock', () =>
                moderateShare(share.share_id, { action: 'unblock' })));
        } else if (share.status === 'active') {
            actions.appendChild(actionBtn('Block', () =>
                moderateShare(share.share_id, { action: 'block' },
                    'Block this page? It will return 410 for all visitors.')));
            if (share.indexed) {
                actions.appendChild(actionBtn('De-index', () =>
                    moderateShare(share.share_id, { indexed: false })));
            } else if (share.visibility !== 'private') {
                actions.appendChild(actionBtn('Index', () =>
                    moderateShare(share.share_id, { indexed: true },
                        share.index_eligible ? '' :
                        'This page does NOT meet the quality filter. Index anyway?')));
            }
        }
        if (share.needs_review && share.status !== 'blocked') {
            actions.appendChild(actionBtn('Mark reviewed', () =>
                moderateShare(share.share_id, { indexed: !!share.indexed })));
        }
        const remove = actionBtn('Delete', () => deleteShare(share.share_id, share.question));
        remove.className = 'danger';
        actions.appendChild(remove);
        row.appendChild(actions);

        const meta = document.createElement('div');
        meta.className = 'share-mod-meta';
        meta.textContent = `${share.share_id} · created ${share.created_at || '–'}`
            + (share.last_reported_at ? ` · last report ${share.last_reported_at}` : '');
        row.appendChild(meta);

        container.appendChild(row);
    });
}

function badge(kind, text, title) {
    const span = document.createElement('span');
    span.className = 'share-mod-badge' + (kind ? ` ${kind}` : '');
    span.textContent = text;
    if (title) span.title = title;
    return span;
}

function actionBtn(label, onClick) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = label;
    btn.onclick = onClick;
    return btn;
}

document.getElementById('loadReportedSharesBtn').addEventListener('click', () => loadShares('reported'));
document.getElementById('loadAllSharesBtn').addEventListener('click', () => loadShares('all'));

// === Scheduled Consensus Publisher ===
function publisherStatus(message, isError) {
    const el = document.getElementById('publisherConfigStatus');
    el.textContent = message || '';
    el.className = isError ? 'error' : 'success';
}

function syncPublisherWatchFields() {
    const enabled = document.getElementById('publisherWeeklyWatch').checked;
    ['publisherWatchWeekday', 'publisherWatchTime', 'publisherWatchTimezone']
        .forEach(id => { document.getElementById(id).disabled = !enabled; });
}

async function loadPublisherConfig() {
    publisherStatus('Loading...', false);
    try {
        const data = await shareAdminRequest('GET', '/api/admin/publisher-config');
        const config = data.config || {};
        document.getElementById('publisherEnabled').checked = !!config.enabled;
        document.getElementById('publisherTopicBrief').value = config.topic_brief || '';
        document.getElementById('publisherAutoIndex').checked = !!config.auto_index;
        document.getElementById('publisherWeeklyWatch').checked = !!config.weekly_watch_enabled;
        document.getElementById('publisherWatchWeekday').value = config.watch_weekday || 'tuesday';
        document.getElementById('publisherWatchTime').value = config.watch_time || '09:00';
        document.getElementById('publisherWatchTimezone').value = config.watch_timezone || 'Europe/Berlin';
        document.getElementById('publisherWatchLimit').value = config.max_active_publisher_watches || 12;
        document.getElementById('publisherWatchProfile').textContent =
            `${config.watch_interval || 'weekly'} · ${config.watch_model_tier || 'free'} Watch providers · DeepSeek excluded`;
        syncPublisherWatchFields();
        publisherStatus('', false);
    } catch (err) {
        publisherStatus(err.message, true);
    }
}

async function savePublisherConfig() {
    const button = document.getElementById('savePublisherConfigBtn');
    button.disabled = true;
    publisherStatus('Saving...', false);
    try {
        const payload = {
            enabled: document.getElementById('publisherEnabled').checked,
            topic_brief: document.getElementById('publisherTopicBrief').value.trim(),
            auto_index: document.getElementById('publisherAutoIndex').checked,
            weekly_watch_enabled: document.getElementById('publisherWeeklyWatch').checked,
            watch_weekday: document.getElementById('publisherWatchWeekday').value,
            watch_time: document.getElementById('publisherWatchTime').value,
            watch_timezone: document.getElementById('publisherWatchTimezone').value.trim(),
            max_active_publisher_watches: Number(document.getElementById('publisherWatchLimit').value || 12)
        };
        await shareAdminRequest('PUT', '/api/admin/publisher-config', payload);
        publisherStatus('Publisher configuration saved.', false);
        await loadPublisherConfig();
    } catch (err) {
        publisherStatus(err.message, true);
    } finally {
        button.disabled = false;
    }
}

document.getElementById('publisherWeeklyWatch').addEventListener('change', syncPublisherWatchFields);
document.getElementById('reloadPublisherConfigBtn').addEventListener('click', loadPublisherConfig);
document.getElementById('savePublisherConfigBtn').addEventListener('click', savePublisherConfig);

// === Scheduled Publisher Watch pages ===
function publisherWatchesStatus(message, isError) {
    const el = document.getElementById('publisherWatchesStatus');
    el.textContent = message || '';
    el.className = isError ? 'error' : 'success';
}

function renderPublisherWatches(watches) {
    const container = document.getElementById('publisherWatchesContainer');
    container.innerHTML = '';
    const publisherWatches = watches.filter(watch => watch.model_tier === 'free');
    if (!publisherWatches.length) {
        container.textContent = 'No automation-created Watch pages found.';
        return;
    }
    publisherWatches.forEach(watch => {
        const row = document.createElement('div');
        row.className = 'watch-admin-row';

        const question = document.createElement('a');
        question.className = 'watch-admin-question';
        question.href = watch.share_path || '#';
        question.target = '_blank';
        question.rel = 'noopener';
        question.textContent = watch.question || '(untitled)';

        const actions = document.createElement('div');
        actions.className = 'share-mod-actions';
        const remove = actionBtn('Delete page', () =>
            deleteShare(watch.share_id, watch.question, publisherWatchesStatus));
        remove.className = 'danger';
        actions.appendChild(remove);

        const metaLine = document.createElement('div');
        metaLine.className = 'watch-admin-meta';
        metaLine.textContent = [
            `Watch: ${watch.status}`,
            `Listing: ${watch.indexed ? 'indexed' : watch.index_requested ? 'requested' : 'noindex'}`,
            `Schedule: ${watch.interval}${watch.run_weekday ? ` on ${watch.run_weekday}` : ''}${watch.run_time ? ` at ${watch.run_time} (${watch.timezone})` : ''}`,
            `Next: ${formatAdminTime(watch.next_run_at)}`,
            `Last: ${formatAdminTime(watch.last_run_at)}`,
            `Share ID: ${watch.share_id}`
        ].join(' | ');
        row.append(question, actions, metaLine);
        container.appendChild(row);
    });
}

async function loadPublisherWatches() {
    publisherWatchesStatus('Loading...', false);
    try {
        const data = await shareAdminRequest('GET', '/api/admin/watches');
        renderPublisherWatches(data.watches || []);
        publisherWatchesStatus('', false);
    } catch (err) {
        publisherWatchesStatus(err.message, true);
    }
}

document.getElementById('reloadPublisherWatchesBtn').addEventListener('click', loadPublisherWatches);

// === Consensus API keys ===
function apiKeysStatus(message, isError) {
    const el = document.getElementById('apiKeysStatus');
    el.textContent = message || '';
    el.className = isError ? 'error' : 'success';
}

function clearIssuedApiKey() {
    const panel = document.getElementById('issuedApiKeyPanel');
    const input = document.getElementById('issuedApiKeyValue');
    input.value = '';
    panel.hidden = true;
}

async function copyIssuedApiKey() {
    const input = document.getElementById('issuedApiKeyValue');
    if (!input.value) return;
    try {
        await navigator.clipboard.writeText(input.value);
    } catch (err) {
        input.focus();
        input.select();
        document.execCommand('copy');
    }
    apiKeysStatus('API key copied. Store it in your secret manager now.', false);
}

function renderApiKeys(keys) {
    const container = document.getElementById('apiKeysContainer');
    container.innerHTML = '';
    if (!keys.length) {
        container.textContent = 'No API keys found.';
        return;
    }
    keys.forEach(key => {
        const row = document.createElement('div');
        row.className = 'api-key-row';

        const main = document.createElement('div');
        main.className = 'api-key-main';
        const name = document.createElement('div');
        name.className = 'api-key-name';
        const label = document.createElement('span');
        label.textContent = key.label || 'Unlabelled key';
        const prefix = document.createElement('span');
        prefix.className = 'api-key-prefix';
        prefix.textContent = `${key.prefix || 'cns_live_'}...`;
        name.append(label, prefix, chip('', key.status || 'unknown'));

        const metadata = document.createElement('div');
        metadata.className = 'api-key-meta';
        metadata.textContent = [
            `UID: ${key.uid || 'unknown'}`,
            `Scopes: ${(key.scopes || []).join(', ') || 'legacy defaults'}`,
            `Created: ${formatAdminTime(key.created_at)}`,
            `Last used: ${formatAdminTime(key.last_used_at)}`,
            `Key ID: ${key.key_id || 'unknown'}`
        ].join(' | ');
        main.append(name, metadata);

        const actions = document.createElement('div');
        actions.className = 'api-key-actions';
        if (key.status === 'active') {
            const revoke = actionBtn('Revoke', () => revokeApiKey(key));
            revoke.className = 'admin-btn secondary';
            actions.appendChild(revoke);
        }
        row.append(main, actions);
        container.appendChild(row);
    });
}

async function loadApiKeys() {
    const filter = document.getElementById('apiKeyFilterUid').value.trim();
    apiKeysStatus('Loading...', false);
    try {
        const path = '/api/admin/api-keys' + (filter ? `?uid=${encodeURIComponent(filter)}` : '');
        const data = await shareAdminRequest('GET', path);
        renderApiKeys(data.keys || []);
        apiKeysStatus('', false);
    } catch (err) {
        apiKeysStatus(err.message, true);
    }
}

async function issueApiKey() {
    const uid = document.getElementById('apiKeyUid').value.trim();
    const label = document.getElementById('apiKeyLabel').value.trim();
    const scopes = ['consensus:run', 'share:write'];
    if (document.getElementById('apiKeyDirectIndex').checked) scopes.push('share:index');
    if (!uid) {
        apiKeysStatus('Enter the Firebase UID that should own this key.', true);
        document.getElementById('apiKeyUid').focus();
        return;
    }
    const button = document.getElementById('issueApiKeyBtn');
    button.disabled = true;
    clearIssuedApiKey();
    apiKeysStatus('Issuing key...', false);
    try {
        const key = await shareAdminRequest('POST', '/api/admin/api-keys', { uid, label, scopes });
        document.getElementById('issuedApiKeyValue').value = key.api_key;
        document.getElementById('issuedApiKeyPanel').hidden = false;
        document.getElementById('apiKeyFilterUid').value = uid;
        apiKeysStatus('API key issued. This is the only time the full key is available.', false);
        await loadApiKeys();
    } catch (err) {
        apiKeysStatus(err.message, true);
    } finally {
        button.disabled = false;
    }
}

async function revokeApiKey(key) {
    if (!confirm(`Revoke API key "${key.label || key.prefix}"? Existing clients will stop authenticating immediately.`)) return;
    apiKeysStatus('Revoking key...', false);
    try {
        await shareAdminRequest('DELETE', `/api/admin/api-keys/${encodeURIComponent(key.key_id)}`);
        apiKeysStatus('API key revoked.', false);
        await loadApiKeys();
    } catch (err) {
        apiKeysStatus(err.message, true);
    }
}

document.getElementById('issueApiKeyBtn').addEventListener('click', issueApiKey);
document.getElementById('reloadApiKeysBtn').addEventListener('click', loadApiKeys);
document.getElementById('copyIssuedApiKeyBtn').addEventListener('click', copyIssuedApiKey);
document.getElementById('dismissIssuedApiKeyBtn').addEventListener('click', clearIssuedApiKey);

// === Consensus Watch diagnostics ===
function watchesStatus(message, isError) {
    const el = document.getElementById('watchesStatus');
    el.textContent = message || '';
    el.className = isError ? 'error' : 'success';
}

function formatAdminTime(value) {
    if (!value) return 'never';
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function renderAdminWatches(watches) {
    const container = document.getElementById('watchesContainer');
    container.innerHTML = '';
    if (!watches.length) {
        container.textContent = 'No Consensus Watches exist.';
        return;
    }
    watches.forEach(watch => {
        const row = document.createElement('div');
        row.className = 'watch-admin-row';
        const question = document.createElement('a');
        question.className = 'watch-admin-question';
        question.href = watch.share_path || '#';
        question.target = '_blank';
        question.rel = 'noopener';
        question.textContent = watch.question || '(untitled)';

        const actions = document.createElement('div');
        actions.className = 'watch-admin-actions';
        const queue = actionBtn('Run now', async () => {
            if (!confirm('Run this watch now? This performs real LLM calls and applies its configured e-mail rule.')) return;
            queue.disabled = true;
            watchesStatus('Starting watch…', false);
            try {
                await shareAdminRequest('POST', `/api/admin/watches/${encodeURIComponent(watch.id)}/run`, {});
                watchesStatus('Watch queued and scheduler started. Reload shortly to see the result.', false);
                await loadAdminWatches();
            } catch (err) {
                watchesStatus(err.message, true);
            } finally {
                queue.disabled = false;
            }
        });
        queue.className = 'admin-btn secondary';
        const leaseActive = watch.claimed_until && new Date(watch.claimed_until).getTime() > Date.now();
        queue.disabled = watch.status !== 'active' || leaseActive;
        queue.title = watch.status !== 'active' ? 'Only active watches can run' : 'Start now through the normal leased scheduler path';
        actions.appendChild(queue);

        const metaLine = document.createElement('div');
        metaLine.className = 'watch-admin-meta';
        metaLine.textContent = [
            `Status: ${watch.status}`,
            `Mode: ${watch.email_mode}`,
            `Interval: ${watch.interval}${watch.run_weekday ? ` on ${watch.run_weekday}` : ''}${watch.run_time ? ` at ${watch.run_time} (${watch.timezone})` : ''}`,
            `Next: ${formatAdminTime(watch.next_run_at)}`,
            `Last: ${formatAdminTime(watch.last_run_at)}`,
            `Failures: ${watch.consecutive_failures}`,
            `Owner: ${watch.owner_uid}`
        ].join(' · ');
        row.append(question, actions, metaLine);
        container.appendChild(row);
    });
}

async function loadAdminWatches() {
    watchesStatus('Loading…', false);
    try {
        const data = await shareAdminRequest('GET', '/api/admin/watches');
        document.getElementById('smtpConfigState').textContent = data.smtp_configured ? 'SMTP configured' : 'SMTP not configured';
        renderAdminWatches(data.watches || []);
        watchesStatus('', false);
    } catch (err) {
        watchesStatus(err.message, true);
    }
}

document.getElementById('loadWatchesBtn').addEventListener('click', loadAdminWatches);
document.getElementById('sendWatchTestMailBtn').addEventListener('click', async function () {
    if (!confirm('Send a real SMTP test message to your verified admin e-mail address?')) return;
    this.disabled = true;
    watchesStatus('Sending test e-mail…', false);
    try {
        const data = await shareAdminRequest('POST', '/api/admin/watches/test-email', {});
        watchesStatus(`Test e-mail accepted for ${data.recipient}.`, false);
    } catch (err) {
        watchesStatus(err.message, true);
    } finally {
        this.disabled = false;
    }
});

// === Public Topic tickers ===
let adminTopics = [];
let selectedTopicId = '';
let selectedTopicDetail = null;
let topicSlugTouched = false;

function topicAdminStatus(message, isError) {
    const el = document.getElementById('topicAdminStatus');
    el.textContent = message || '';
    el.className = `topic-status-line ${isError ? 'error' : 'success'}`;
}

function topicSlug(value) {
    return String(value || '').toLowerCase().trim()
        .replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
}

function renderAdminTopicList() {
    const container = document.getElementById('adminTopicList');
    container.innerHTML = '';
    if (!adminTopics.length) {
        container.textContent = 'No Topics yet.';
        return;
    }
    adminTopics.forEach(topic => {
        const button = document.createElement('button');
        button.type = 'button';
        button.classList.toggle('is-active', topic.id === selectedTopicId);
        const title = document.createElement('strong');
        title.textContent = topic.title || '(untitled)';
        const metaLine = document.createElement('small');
        metaLine.textContent = `${topic.status} · ${topic.run_count || 0} runs · next ${formatAdminTime(topic.next_run_at)}`;
        button.append(title, metaLine);
        button.addEventListener('click', () => selectAdminTopic(topic.id));
        container.appendChild(button);
    });
}

function renderTopicModelPlan(providerModels) {
    const selected = providerModels || {};
    const defaults = (globalModelsData.watch_models || {}).free || {};
    const container = document.getElementById('topicModelPlan');
    container.innerHTML = '';
    providers.forEach(provider => {
        const models = globalModelsData[provider] || [];
        const chosen = selected[provider] || defaults[provider] || '';
        const row = document.createElement('div');
        row.className = 'topic-model-row';
        row.dataset.provider = provider;
        const toggleLabel = document.createElement('label');
        const toggle = document.createElement('input');
        toggle.type = 'checkbox';
        toggle.checked = !!chosen;
        toggle.setAttribute('aria-label', `Run ${provider}`);
        toggleLabel.append(toggle, document.createTextNode(providerLabel(provider)));
        const select = document.createElement('select');
        const options = models.length ? models : (chosen ? [chosen] : []);
        options.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = ((globalModelsData.meta || {}).labels || {})[model] || model;
            option.selected = model === chosen;
            select.appendChild(option);
        });
        select.disabled = !toggle.checked;
        toggle.addEventListener('change', () => {
            select.disabled = !toggle.checked;
            if (toggle.checked && !select.value && select.options.length) select.selectedIndex = 0;
        });
        row.append(toggleLabel, select);
        container.appendChild(row);
    });
}

function fillAdminTopic(topic, runs) {
    selectedTopicDetail = topic;
    document.getElementById('topicEditorEmpty').hidden = true;
    document.getElementById('topicAdminForm').hidden = false;
    document.getElementById('adminTopicTitle').value = topic.title || '';
    document.getElementById('adminTopicSlug').value = topic.slug || '';
    const slugHistory = document.getElementById('adminTopicSlugHistory');
    const retired = (topic.slug_history || []).filter(Boolean);
    slugHistory.hidden = retired.length === 0;
    slugHistory.textContent = retired.length
        ? `Redirecting (301): ${retired.map((item) => `/topics/${item}`).join(', ')}`
        : '';
    document.getElementById('adminTopicQuestion').value = topic.lead_question || '';
    document.getElementById('adminTopicCategory').value = topic.category || '';
    document.getElementById('adminTopicSummary').value = topic.summary || '';
    document.getElementById('adminTopicStatus').value = topic.status || 'active';
    document.getElementById('adminTopicInterval').value = topic.update_interval || 'weekly';
    document.getElementById('adminTopicDomains').value =
        ((topic.source_rules || {}).preferred_domains || []).join('\n');
    document.getElementById('adminTopicSourceNotes').value =
        (topic.source_rules || {}).notes || '';
    document.getElementById('adminTopicSeoTitle').value = (topic.seo || {}).title || '';
    document.getElementById('adminTopicSeoDescription').value = (topic.seo || {}).description || '';
    document.getElementById('adminTopicNoindex').checked = !!(topic.seo || {}).noindex;
    renderTopicModelPlan((topic.run_config || {}).provider_models || {});
    const schedule = document.getElementById('topicScheduleState');
    schedule.textContent = topic.id
        ? `Last: ${formatAdminTime(topic.latest_run_at)} · Next: ${formatAdminTime(topic.next_run_at)} · ${topic.last_run_status || 'never'}${topic.last_run_error ? ` · ${topic.last_run_error}` : ''}`
        : 'Save the Topic before its first run.';
    const history = document.getElementById('topicRunHistory');
    history.innerHTML = '';
    (runs || []).forEach(run => {
        const item = document.createElement('div');
        const headline = document.createElement('strong');
        headline.textContent = `Run ${run.version} · ${run.agreement_score}/100 · ${run.change_type}`;
        const details = document.createElement('small');
        details.textContent = `${formatAdminTime(run.observed_at)} · ${(run.models || []).length} models · ${(run.evidence || []).length} sources`;
        const summary = document.createElement('small');
        summary.textContent = run.change_summary || 'No material change.';
        item.append(headline, details, summary);
        history.appendChild(item);
    });
    if (!(runs || []).length) history.textContent = 'No runs yet.';
    const leaseActive = topic.claimed_until && new Date(topic.claimed_until).getTime() > Date.now();
    document.getElementById('runAdminTopicBtn').disabled =
        !topic.id || topic.status !== 'active' || leaseActive;
    document.getElementById('openAdminTopicBtn').hidden = !topic.latest_run_id;
    document.getElementById('openAdminTopicBtn').href = topic.slug ? `/topics/${encodeURIComponent(topic.slug)}` : '#';
    topicSlugTouched = !!topic.id;
    topicAdminStatus('', false);
}

async function selectAdminTopic(topicId) {
    selectedTopicId = topicId;
    renderAdminTopicList();
    topicAdminStatus('Loading Topic...', false);
    try {
        const data = await shareAdminRequest('GET', `/api/admin/topics/${encodeURIComponent(topicId)}`);
        fillAdminTopic(data.topic, data.runs || []);
    } catch (err) {
        topicAdminStatus(err.message, true);
    }
}

async function loadAdminTopics(selectId) {
    topicAdminStatus('Loading Topics...', false);
    try {
        const data = await shareAdminRequest('GET', '/api/admin/topics');
        adminTopics = data.topics || [];
        renderAdminTopicList();
        const target = selectId || selectedTopicId;
        if (target) await selectAdminTopic(target);
        topicAdminStatus('', false);
    } catch (err) {
        topicAdminStatus(err.message, true);
    }
}

function newAdminTopic() {
    selectedTopicId = '';
    renderAdminTopicList();
    fillAdminTopic({
        status: 'active',
        update_interval: 'weekly',
        source_rules: { allowed_types: ['primary', 'research', 'documentation', 'reporting', 'community', 'rumor'] },
        run_config: { provider_models: (globalModelsData.watch_models || {}).free || {} },
        seo: {}
    }, []);
    topicSlugTouched = false;
}

function topicProviderModels() {
    const result = {};
    document.querySelectorAll('#topicModelPlan .topic-model-row').forEach(row => {
        const toggle = row.querySelector('input[type="checkbox"]');
        const select = row.querySelector('select');
        if (toggle.checked && select.value) result[row.dataset.provider] = select.value;
    });
    return result;
}

function topicLines(value) {
    return String(value || '').split(/\r?\n/).map(item => item.trim()).filter(Boolean);
}

function adminTopicPayload() {
    return {
        title: document.getElementById('adminTopicTitle').value.trim(),
        slug: document.getElementById('adminTopicSlug').value.trim(),
        lead_question: document.getElementById('adminTopicQuestion').value.trim(),
        category: document.getElementById('adminTopicCategory').value.trim(),
        summary: document.getElementById('adminTopicSummary').value.trim(),
        status: document.getElementById('adminTopicStatus').value,
        update_interval: document.getElementById('adminTopicInterval').value,
        run_config: { provider_models: topicProviderModels(), collect_sources: true },
        source_rules: {
            allowed_types: ['primary', 'research', 'documentation', 'reporting', 'community', 'rumor'],
            preferred_domains: topicLines(document.getElementById('adminTopicDomains').value),
            notes: document.getElementById('adminTopicSourceNotes').value.trim()
        },
        seo: {
            title: document.getElementById('adminTopicSeoTitle').value.trim(),
            description: document.getElementById('adminTopicSeoDescription').value.trim(),
            noindex: document.getElementById('adminTopicNoindex').checked
        }
    };
}

async function saveAdminTopic() {
    const button = document.getElementById('saveAdminTopicBtn');
    button.disabled = true;
    topicAdminStatus('Saving Topic...', false);
    try {
        const data = await shareAdminRequest(
            selectedTopicId ? 'PUT' : 'POST',
            selectedTopicId ? `/api/admin/topics/${encodeURIComponent(selectedTopicId)}` : '/api/admin/topics',
            adminTopicPayload()
        );
        selectedTopicId = data.topic.id;
        await loadAdminTopics(selectedTopicId);
        topicAdminStatus('Topic configuration saved.', false);
    } catch (err) {
        topicAdminStatus(err.message, true);
    } finally {
        button.disabled = false;
    }
}

async function runAdminTopic() {
    if (!selectedTopicId) return;
    if (!confirm('Run this Topic now? The selected models will research current sources and create a new immutable timeline point.')) return;
    const button = document.getElementById('runAdminTopicBtn');
    button.disabled = true;
    topicAdminStatus('Researching sources and building Consensus. This can take a minute...', false);
    try {
        const data = await shareAdminRequest('POST', `/api/admin/topics/${encodeURIComponent(selectedTopicId)}/runs`, {});
        await loadAdminTopics(selectedTopicId);
        topicAdminStatus(`Run ${data.run.version} saved: ${data.run.agreement_score}/100 agreement and ${(data.run.evidence || []).length} sources.`, false);
    } catch (err) {
        topicAdminStatus(err.message, true);
    } finally {
        button.disabled = false;
    }
}

document.getElementById('newAdminTopicBtn').addEventListener('click', newAdminTopic);
document.getElementById('reloadAdminTopicsBtn').addEventListener('click', () => loadAdminTopics(selectedTopicId));
document.getElementById('saveAdminTopicBtn').addEventListener('click', saveAdminTopic);
document.getElementById('runAdminTopicBtn').addEventListener('click', runAdminTopic);
document.getElementById('topicAdminForm').addEventListener('submit', event => event.preventDefault());
document.getElementById('adminTopicTitle').addEventListener('input', event => {
    if (!topicSlugTouched) document.getElementById('adminTopicSlug').value = topicSlug(event.target.value);
});
document.getElementById('adminTopicSlug').addEventListener('input', () => { topicSlugTouched = true; });

// === Read-only SEO data foundation ===
function seoStatus(message, isError) {
    const el = document.getElementById('seoStatus');
    el.textContent = message || '';
    el.className = isError ? 'error' : 'success';
}

function appendSeoText(parent, tag, text, className) {
    const node = document.createElement(tag);
    node.textContent = text || '';
    if (className) node.className = className;
    parent.appendChild(node);
    return node;
}

function formatSeoNumber(value) {
    const number = Number(value || 0);
    return Number.isInteger(number)
        ? number.toLocaleString()
        : number.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function formatSeoMetric(metric, key) {
    if (key === 'ctr') return `${(Number(metric.ctr || 0) * 100).toFixed(2)}%`;
    if (key === 'position') return metric.position == null ? '–' : Number(metric.position).toFixed(1);
    return formatSeoNumber(metric[key]);
}

// Ordered loudest first: the operator should read trouble before routine.
const SEO_STATUS_ORDER = [
    'declining', 'opportunity', 'insufficient_data', 'invisible', 'winner', 'emerging'
];
const SEO_STATUS_LABELS = {
    declining: 'declining',
    opportunity: 'opportunity',
    insufficient_data: 'insufficient data',
    invisible: 'invisible',
    winner: 'winner',
    emerging: 'emerging'
};
// "emerging" is the resting state of a healthy young page: nothing to decide.
// Everything else is either good news or a problem and stays in the default view.
const SEO_QUIET_STATUSES = new Set(['emerging']);
const SEO_HEALTHY_RUN_STATUSES = new Set(['success', 'partial']);
const SEO_METRIC_KEYS = ['clicks', 'impressions', 'ctr', 'position'];

const seoState = {
    data: null,
    // null means "never checked in this session", which is not the same as failed.
    connection: null,
    reviewGroupByPage: new Map(),
    filters: { search: '', scope: 'attention', status: '', recommendation: '', origin: '' }
};

function seoRows() {
    return (seoState.data || {}).rows || [];
}

function seoLatestRecommendation(row) {
    return (row.recommendation_history || [])[0] || {};
}

function seoRowMatchesFilters(row) {
    const filters = seoState.filters;
    const status = row.status || 'insufficient_data';
    const inReview = seoState.reviewGroupByPage.has(row.page_id);
    if (filters.scope === 'attention' && SEO_QUIET_STATUSES.has(status) && !inReview) return false;
    if (filters.status && status !== filters.status) return false;
    if (filters.origin && (row.origin || '') !== filters.origin) return false;
    if (filters.recommendation) {
        const recommendation = seoLatestRecommendation(row).recommendation || '';
        if ((recommendation || 'not_generated') !== filters.recommendation) return false;
    }
    if (filters.search) {
        const haystack = [
            row.url || '', (row.dossier || {}).title || '', row.share_id || '', row.origin || ''
        ].join(' ').toLowerCase();
        if (!haystack.includes(filters.search)) return false;
    }
    return true;
}

function renderSeoOverview(data) {
    seoState.data = data;
    // The review runs first: it owns the map that tells the page list which
    // rows already carry a pending decision, so nothing is told twice.
    renderSeoWeeklyReview(data.weekly_review || {});
    renderSeoDiagnostics(data);
    renderSeoPortfolio(data);
    renderSeoFilterOptions(data);
    renderSeoTable();
    renderSeoAlerts();
}

function renderSeoDiagnostics(data) {
    const config = data.configuration || {};
    const configState = document.getElementById('seoConfigState');
    configState.textContent = config.configured
        ? 'configured'
        : config.status === 'not_configured' ? 'not configured' : (config.status || 'not configured');
    configState.title = config.message || '';
    document.getElementById('seoConfigMessage').textContent = config.message || '';
    document.getElementById('seoCapturedCount').textContent = `${data.captured_urls || 0} / ${data.eligible_urls || 0}`;
    const run = data.last_run || {};
    document.getElementById('seoLastRun').textContent = run.started_at
        ? `${run.status || 'unknown'} · ${new Date(run.started_at).toLocaleString()}`
            + (run.metrics_written != null ? ` · ${run.metrics_written} daily rows` : '')
        : 'No collection run yet';
    document.getElementById('seoLastRun').title = run.message || '';
    document.getElementById('seoLastRunMessage').textContent = run.message || '';
    document.getElementById('seoFinalDate').textContent = data.final_date || '–';
    document.getElementById('seoDisclaimer').textContent = data.disclaimer || '';
    const judge = data.content_judge || {};
    const judgeState = document.getElementById('seoContentJudgeState');
    judgeState.textContent = judge.configured ? 'configured' : 'not configured';
    judgeState.title = judge.message || '';
    document.getElementById('seoRules').textContent = Object.entries(data.status_rules || {})
        .map(([name, definition]) => `${name}: ${definition}`).join(' · ');
}

// --- Level 0: the alert strip -------------------------------------------
// Configuration and the collection run used to sit in a quiet card and stayed
// quiet for five weeks while the pipeline was dead. Every failure state that
// can silence the pipeline has to surface here, above everything else.
function addSeoAlert(container, tone, title, detail) {
    const box = document.createElement('div');
    box.className = `seo-alert is-${tone}`;
    appendSeoText(box, 'strong', title);
    if (detail) appendSeoText(box, 'span', detail);
    container.appendChild(box);
    return box;
}

function renderSeoAlerts() {
    const container = document.getElementById('seoAlerts');
    container.textContent = '';
    const data = seoState.data || {};
    const config = data.configuration || {};
    const run = data.last_run || {};
    const review = (data.weekly_review || {}).latest_review || {};

    if (!config.configured) {
        addSeoAlert(
            container, 'error', 'Search Console is not configured.',
            config.message || 'No usable credentials. No collection run can succeed.'
        );
    }

    const connection = seoState.connection;
    if (connection && !connection.connected) {
        addSeoAlert(
            container, 'error', 'Search Console connection failed.',
            connection.message || String(connection.status || 'connection_failed').replaceAll('_', ' ')
        );
    } else if (!connection && config.configured) {
        addSeoAlert(
            container, 'notice', 'Connection not verified in this session.',
            'Use “Check Search Console connection” to confirm the property still answers.'
        );
    }

    const runStatus = String(run.status || '');
    if (!run.started_at) {
        addSeoAlert(
            container, 'error', 'No Search Console collection has ever run.',
            'Without stored daily rows every status below is a data gap, not a traffic signal.'
        );
    } else if (runStatus === 'running') {
        addSeoAlert(
            container, 'notice', 'A collection run is still in progress.',
            [new Date(run.started_at).toLocaleString(), run.message || ''].filter(Boolean).join(' · ')
        );
    } else if (!SEO_HEALTHY_RUN_STATUSES.has(runStatus)) {
        addSeoAlert(
            container, 'error',
            `Latest collection run did not succeed (${runStatus || 'unknown'}).`,
            [new Date(run.started_at).toLocaleString(), run.message || ''].filter(Boolean).join(' · ')
        );
    }

    const reviewStatus = String(review.status || '');
    if (reviewStatus === 'collection_failed' || reviewStatus === 'error') {
        addSeoAlert(
            container, 'error', `Latest weekly review ended as ${reviewStatus.replaceAll('_', ' ')}.`,
            review.summary || (review.collection || {}).message || 'Check server diagnostics.'
        );
    }

    if (review.run_id && !review.judge_called && review.judge_error) {
        addSeoAlert(
            container, 'warning', 'The portfolio judge did not answer in the last review.',
            `${review.judge_error} The assessment was generated from the rules, not by the judge.`
        );
    }

    const eligible = Number(data.eligible_urls || 0);
    if (eligible > 0 && Number(data.captured_urls || 0) === 0) {
        addSeoAlert(
            container, 'error', `No page has any stored Search Console data (0 of ${eligible}).`,
            'Every status below reads insufficient_data because nothing was written, not because traffic is low.'
        );
    }

    container.hidden = container.children.length === 0;
}

// --- Level 1: the portfolio ---------------------------------------------
function renderSeoPortfolio(data) {
    const rows = data.rows || [];
    const counts = new Map();
    rows.forEach(row => {
        const status = row.status || 'insufficient_data';
        counts.set(status, (counts.get(status) || 0) + 1);
    });
    const ordered = SEO_STATUS_ORDER
        .concat([...counts.keys()].filter(status => !SEO_STATUS_ORDER.includes(status)))
        .filter(status => counts.get(status));

    const bar = document.getElementById('seoStatusBar');
    bar.textContent = '';
    ordered.forEach(status => {
        const segment = document.createElement('div');
        segment.className = `seo-status-segment is-${status}`;
        segment.style.flexGrow = String(counts.get(status));
        segment.title = `${SEO_STATUS_LABELS[status] || status}: ${counts.get(status)}`;
        bar.appendChild(segment);
    });

    const legend = document.getElementById('seoStatusLegend');
    legend.textContent = '';
    ordered.forEach(status => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = `seo-legend-item is-${status}`;
        if (seoState.filters.status === status) button.classList.add('is-active');
        appendSeoText(button, 'span', '', 'seo-legend-dot');
        appendSeoText(button, 'span', `${SEO_STATUS_LABELS[status] || status} ${counts.get(status)}`);
        button.title = (data.status_rules || {})[status] || '';
        button.addEventListener('click', () => {
            // A second click on the active class clears the filter again.
            const next = seoState.filters.status === status ? '' : status;
            seoState.filters.status = next;
            document.getElementById('seoStatusFilter').value = next;
            if (next) {
                seoState.filters.scope = 'all';
                document.getElementById('seoScopeFilter').value = 'all';
            }
            renderSeoPortfolio(seoState.data || {});
            renderSeoTable();
        });
        legend.appendChild(button);
    });

    const visible = rows.filter(row => !SEO_QUIET_STATUSES.has(row.status || 'insufficient_data')).length;
    document.getElementById('seoPortfolioLine').textContent = rows.length
        ? `${rows.length} tracked pages · ${visible} outside the quiet “emerging” state`
            + ` · latest finalized data ${data.final_date || 'unknown'}`
        : 'No pages tracked yet.';
}

// --- Level 3: the page inventory ----------------------------------------
function fillSeoSelect(select, values, labels) {
    const previous = select.value;
    const placeholder = select.options[0];
    select.textContent = '';
    select.appendChild(placeholder);
    values.forEach(value => {
        const option = document.createElement('option');
        option.value = value;
        option.textContent = (labels || {})[value] || value;
        select.appendChild(option);
    });
    select.value = [...select.options].some(option => option.value === previous) ? previous : '';
    return select.value;
}

function renderSeoFilterOptions(data) {
    const rows = data.rows || [];
    const statuses = SEO_STATUS_ORDER
        .concat([...new Set(rows.map(row => row.status || 'insufficient_data'))])
        .filter((status, index, all) => all.indexOf(status) === index)
        .filter(status => rows.some(row => (row.status || 'insufficient_data') === status));
    seoState.filters.status = fillSeoSelect(
        document.getElementById('seoStatusFilter'), statuses, SEO_STATUS_LABELS
    );
    const recommendations = [...new Set(
        rows.map(row => seoLatestRecommendation(row).recommendation || 'not_generated')
    )].sort();
    seoState.filters.recommendation = fillSeoSelect(
        document.getElementById('seoRecommendationFilter'), recommendations,
        { not_generated: 'not generated' }
    );
    const origins = [...new Set(rows.map(row => row.origin || '').filter(Boolean))].sort();
    seoState.filters.origin = fillSeoSelect(document.getElementById('seoOriginFilter'), origins);
}

function seoMetricCell(row, key) {
    const td = document.createElement('td');
    appendSeoText(td, 'div', formatSeoMetric(row.metrics_28d || {}, key));
    appendSeoText(td, 'div', formatSeoMetric(row.metrics_7d || {}, key), 'seo-metric-7d')
        .title = 'Last 7 days';
    return td;
}

function buildSeoPageCell(row) {
    const pageCell = document.createElement('td');
    const link = document.createElement('a');
    link.href = row.url;
    link.target = '_blank';
    link.rel = 'noopener';
    link.textContent = row.url;
    pageCell.appendChild(link);
    const reviewGroup = seoState.reviewGroupByPage.get(row.page_id);
    if (reviewGroup) {
        // The decision itself lives in the work list above; the list only says
        // that this page is already waiting there.
        appendSeoText(pageCell, 'span', `In this review: ${reviewGroup}`, 'seo-review-pill');
    }
    appendSeoText(
        pageCell, 'div', row.origin + (row.share_id ? ` · ${row.share_id}` : ''), 'section-hint'
    );
    const dossier = row.dossier || {};
    if (dossier.title) appendSeoText(pageCell, 'div', dossier.title, 'section-hint');
    const queryData = row.query_data || {};
    const queries = (queryData.top_queries || []).slice(0, 3);
    if (queries.length || queryData.period_end) {
        const partial = queryData.partial ? ' · partial' : '';
        appendSeoText(
            pageCell, 'div',
            queries.length
                ? `Top queries${partial}: ${queries.map(item => item.query).join(', ')}`
                : `Query snapshot ${queryData.period_end || ''}${partial}: no rows`,
            'seo-query-list'
        );
    }
    return pageCell;
}

function buildSeoStatusCell(row, statusRules) {
    const statusCell = document.createElement('td');
    const status = document.createElement('span');
    status.className = 'seo-status';
    status.textContent = SEO_STATUS_LABELS[row.status] || row.status || 'insufficient data';
    status.title = statusRules[row.status] || '';
    statusCell.appendChild(status);
    const window28 = row.metrics_28d || {};
    const days = Number(window28.days || 0);
    appendSeoText(
        statusCell, 'div', `${days}/28 daily rows`, 'section-hint'
    ).title = 'Stored finalized daily rows. The status depends on these rows, not on clicks.';
    if (row.status === 'insufficient_data') {
        // This exact confusion once sent the operator hunting for a traffic
        // problem that was really a missing-data problem.
        appendSeoText(
            statusCell, 'div',
            'Data gap, not low traffic: fewer than 7 stored rows.',
            'seo-status-note'
        );
    }
    return statusCell;
}

function buildSeoRecommendationCell(row) {
    const history = row.recommendation_history || [];
    const latest = history[0] || {};
    const cell = document.createElement('td');
    cell.className = 'seo-recommendation';
    const recommendation = document.createElement('span');
    recommendation.className = 'seo-status';
    recommendation.textContent = latest.recommendation || 'not generated';
    recommendation.title = (latest.evidence || []).join(' ');
    cell.appendChild(recommendation);
    if (latest.recommendation) {
        appendSeoText(
            cell, 'div',
            `${Math.round(Number(latest.confidence || 0) * 100)}% confidence`
                + (latest.llm_evaluation ? ` · content judge: ${latest.llm_evaluation.recommendation}` : ''),
            'section-hint'
        );
    }
    if (history.length) {
        const details = document.createElement('details');
        details.className = 'seo-history';
        appendSeoText(details, 'summary', `History (${history.length})`);
        history.forEach(item => {
            const timestamp = item.created_at ? new Date(item.created_at).toLocaleString() : 'unknown time';
            appendSeoText(
                details, 'div',
                `${timestamp} · ${item.recommendation}`
                    + (item.llm_evaluation ? ` / ${item.llm_evaluation.recommendation}` : '')
            );
        });
        cell.appendChild(details);
    }
    return cell;
}

function buildSeoActionCell(row, judgeConfigured) {
    const actionCell = document.createElement('td');
    const actions = document.createElement('div');
    actions.className = 'seo-action-stack';
    const generate = document.createElement('button');
    generate.type = 'button';
    generate.className = 'admin-btn secondary';
    generate.textContent = 'Generate recommendation';
    generate.addEventListener('click', () => runSeoRecommendation(row.page_id, false, generate));
    actions.appendChild(generate);
    const latest = seoLatestRecommendation(row);
    const judgeApplicable = ['opportunity', 'declining'].includes(row.status)
        || latest.recommendation === 'noindex_candidate';
    if (judgeConfigured && judgeApplicable) {
        const ask = document.createElement('button');
        ask.type = 'button';
        ask.className = 'admin-btn secondary';
        ask.textContent = 'Ask content judge';
        ask.addEventListener('click', () => runSeoRecommendation(row.page_id, true, ask));
        actions.appendChild(ask);
    }
    actionCell.appendChild(actions);
    return actionCell;
}

function renderSeoTable() {
    const data = seoState.data || {};
    const statusRules = data.status_rules || {};
    const judgeConfigured = !!(data.content_judge || {}).configured;
    const allRows = seoRows();
    const rows = allRows.filter(seoRowMatchesFilters);
    const body = document.getElementById('seoTableBody');
    body.textContent = '';

    const byStatus = new Map();
    rows.forEach(row => {
        const status = row.status || 'insufficient_data';
        if (!byStatus.has(status)) byStatus.set(status, []);
        byStatus.get(status).push(row);
    });
    const ordered = SEO_STATUS_ORDER
        .concat([...byStatus.keys()].filter(status => !SEO_STATUS_ORDER.includes(status)))
        .filter(status => byStatus.has(status));

    ordered.forEach(status => {
        const groupRow = document.createElement('tr');
        groupRow.className = 'seo-group-row';
        const cell = document.createElement('td');
        cell.colSpan = 9;
        appendSeoText(cell, 'strong', SEO_STATUS_LABELS[status] || status);
        appendSeoText(cell, 'span', ` ${byStatus.get(status).length}`, 'seo-group-count');
        appendSeoText(cell, 'span', statusRules[status] || '', 'seo-group-rule');
        groupRow.appendChild(cell);
        body.appendChild(groupRow);

        byStatus.get(status).forEach(row => {
            const tr = document.createElement('tr');
            tr.appendChild(buildSeoPageCell(row));
            const visibility = document.createElement('td');
            appendSeoText(
                visibility, 'div',
                String(Math.round(Number((row.metrics_28d || {}).visibility || 0)))
            ).title = 'Visibility over 28 days. One click weighs as much as 20 impressions.';
            tr.appendChild(visibility);
            SEO_METRIC_KEYS.forEach(key => tr.appendChild(seoMetricCell(row, key)));
            tr.appendChild(buildSeoStatusCell(row, statusRules));
            tr.appendChild(buildSeoRecommendationCell(row));
            tr.appendChild(buildSeoActionCell(row, judgeConfigured));
            body.appendChild(tr);
        });
    });

    document.getElementById('seoEmptyState').hidden = allRows.length > 0;
    document.getElementById('seoFilterEmptyState').hidden = !allRows.length || rows.length > 0;
    document.getElementById('seoInventoryCount').textContent = allRows.length
        ? `Showing ${rows.length} of ${allRows.length} pages, sorted by visibility inside each status.`
        : 'No pages loaded.';
}

function readSeoFilters() {
    seoState.filters.search = document.getElementById('seoSearch').value.trim().toLowerCase();
    seoState.filters.scope = document.getElementById('seoScopeFilter').value;
    seoState.filters.status = document.getElementById('seoStatusFilter').value;
    seoState.filters.recommendation = document.getElementById('seoRecommendationFilter').value;
    seoState.filters.origin = document.getElementById('seoOriginFilter').value;
    renderSeoPortfolio(seoState.data || {});
    renderSeoTable();
}

document.getElementById('seoSearch').addEventListener('input', readSeoFilters);
['seoScopeFilter', 'seoStatusFilter', 'seoRecommendationFilter', 'seoOriginFilter']
    .forEach(id => document.getElementById(id).addEventListener('change', readSeoFilters));

document.getElementById('resetSeoFiltersBtn').addEventListener('click', function () {
    document.getElementById('seoSearch').value = '';
    document.getElementById('seoScopeFilter').value = 'attention';
    document.getElementById('seoStatusFilter').value = '';
    document.getElementById('seoRecommendationFilter').value = '';
    document.getElementById('seoOriginFilter').value = '';
    readSeoFilters();
});

// --- Level 2: the work list ---------------------------------------------
const SEO_GROUP_LABELS = {
    keep_indexed: 'Keep indexed',
    pause_watch_only: 'Pause Watch only',
    resume_watch: 'Resume Watch',
    noindex_only: 'Noindex only',
    noindex_and_pause_watch: 'Noindex and pause Watch',
    delete_candidate: 'Delete candidate',
    manual_improvement: 'Editorial decision required'
};
const SEO_GROUP_BUTTONS = {
    keep_indexed: 'Mark selected reviewed',
    pause_watch_only: 'Pause selected Watches',
    resume_watch: 'Resume selected Watches',
    noindex_only: 'Set selected noindex',
    noindex_and_pause_watch: 'Noindex + pause selected',
    delete_candidate: 'Delete selected candidates',
    manual_improvement: 'Record editorial decisions below'
};
const SEO_EDITORIAL_DECISION_LABELS = {
    keep_as_is: 'Keep unchanged',
    create_successor: 'Create improved successor snapshot',
    investigate: 'Investigate before changing',
    noindex: 'Review for noindex',
    delete: 'Review for deletion',
    edit_static_page: 'Edit existing static page'
};
const SEO_SAFE_APPLY_GROUPS = new Set([
    'keep_indexed', 'pause_watch_only', 'resume_watch',
    'noindex_only', 'noindex_and_pause_watch'
]);

function formatSeoScheduleTime(value, timezone) {
    if (!value) return '';
    try {
        return `${new Date(value).toLocaleString(undefined, { timeZone: timezone })} (${timezone})`;
    } catch (_err) {
        return new Date(value).toLocaleString();
    }
}

function buildSeoEditorialControls(runId, pageId, template) {
    const controls = document.createElement('div');
    controls.className = 'seo-editorial-controls';
    const select = document.createElement('select');
    (template.options || ['keep_as_is', 'investigate']).forEach(value => {
        const option = document.createElement('option');
        option.value = value;
        option.textContent = SEO_EDITORIAL_DECISION_LABELS[value] || value;
        select.appendChild(option);
    });
    select.value = template.suggested_decision || 'keep_as_is';
    const note = document.createElement('input');
    note.type = 'text';
    note.maxLength = 500;
    note.placeholder = template.explanation || 'Optional note';
    const save = document.createElement('button');
    save.type = 'button';
    save.className = 'admin-btn secondary';
    save.textContent = 'Confirm decision';
    save.addEventListener('click', () => saveSeoEditorialDecision(
        runId, pageId, select.value, note.value, save
    ));
    controls.append(select, note, save);
    return controls;
}

function renderSeoReviewGroup(review, group, ids, pages) {
    const label = SEO_GROUP_LABELS[group] || group;
    const editorial = group === 'manual_improvement';
    const box = document.createElement('div');
    box.className = 'seo-review-group';
    const head = document.createElement('div');
    head.className = 'seo-review-group-head';
    appendSeoText(head, 'strong', `${label} (${ids.length})`);
    if (editorial) {
        // One button for the whole group instead of three interactions per row.
        const suggested = ids
            .map(id => ({ id, template: (pages.get(id) || {}).editorial_decision_template || {} }))
            .filter(item => item.template.suggested_decision);
        if (suggested.length) {
            const bulk = document.createElement('button');
            bulk.type = 'button';
            bulk.className = 'admin-btn secondary';
            bulk.textContent = `Confirm ${suggested.length} suggested decisions`;
            bulk.addEventListener('click', () => confirmSuggestedSeoDecisions(
                review.run_id,
                suggested.map(item => ({
                    pageId: item.id,
                    decision: item.template.suggested_decision,
                    title: (pages.get(item.id) || {}).title || (pages.get(item.id) || {}).url || item.id
                })),
                bulk
            ));
            head.appendChild(bulk);
        }
    } else {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = group === 'delete_candidate' ? 'admin-btn danger' : 'admin-btn secondary';
        button.textContent = SEO_GROUP_BUTTONS[group] || label;
        button.addEventListener('click', () => applySeoReviewGroup(review.run_id, group, box, button));
        head.appendChild(button);
    }
    box.appendChild(head);
    if (editorial) {
        appendSeoText(box, 'p', SEO_GROUP_BUTTONS.manual_improvement, 'section-hint u-m-0');
    }
    const list = document.createElement('div');
    list.className = 'seo-review-pages';
    ids.forEach(id => {
        const page = pages.get(id) || {};
        const row = document.createElement(editorial ? 'div' : 'label');
        row.className = 'seo-review-page';
        if (editorial) {
            appendSeoText(row, 'span', 'Decision');
        } else {
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.checked = true;
            checkbox.dataset.pageId = id;
            row.appendChild(checkbox);
        }
        appendSeoText(row, 'span', page.title || page.url || id);
        const template = page.editorial_decision_template || {};
        const judgeReason = ((page.portfolio_judge || {}).reason || '').trim();
        appendSeoText(
            row, 'small',
            `${page.indexed ? 'indexed' : 'noindex'} · Watch ${page.watch_status || 'none'} · ${page.recommendation || ''}`
                + (template.immutable_snapshot ? ' · immutable snapshot' : '')
                + (judgeReason ? ` · Terra: ${judgeReason}` : '')
        );
        if (editorial) row.appendChild(buildSeoEditorialControls(review.run_id, id, template));
        list.appendChild(row);
    });
    box.appendChild(list);
    return box;
}

function renderSeoWeeklyReview(status) {
    const config = status.config || {};
    const review = status.latest_review || {};
    const watches = status.publisher_watches || {};
    document.getElementById('seoReviewEnabled').checked = !!config.enabled;
    document.getElementById('seoReviewInterval').value = config.interval_days || 7;
    document.getElementById('seoReviewTime').value = config.run_time || '09:00';
    document.getElementById('seoReviewTimezone').value = config.timezone || 'Europe/Berlin';
    document.getElementById('seoReviewLast').textContent = config.last_run_at
        ? formatSeoScheduleTime(config.last_run_at, config.timezone || 'Europe/Berlin') : 'never';
    document.getElementById('seoReviewNext').textContent = config.next_run_at
        ? formatSeoScheduleTime(config.next_run_at, config.timezone || 'Europe/Berlin') : 'not scheduled';
    document.getElementById('seoPortfolioJudge').textContent = (status.judge || {}).configured
        ? `configured${status.judge.model ? ` (${status.judge.model})` : ''}` : 'not configured';
    document.getElementById('seoPublisherWatchCount').textContent =
        `${watches.active || 0} active / ${watches.paused || 0} paused / limit ${watches.limit || 12}`;
    document.getElementById('seoSearchRules').textContent = status.search_opportunity_rules || '';
    document.getElementById('runSeoReviewBtn').disabled = !!status.running;
    document.getElementById('seoReviewSummary').textContent = review.summary || 'No weekly review yet.';
    renderSeoReviewDiagnostics(review);

    const groupsContainer = document.getElementById('seoReviewGroups');
    groupsContainer.textContent = '';
    seoState.reviewGroupByPage = new Map();
    const pages = new Map((review.pages || []).map(page => [page.page_id, page]));
    const completed = new Map();
    (review.applied_actions || []).forEach(action => {
        (action.results || []).forEach(result => {
            if (result.status === 'success') {
                completed.set(result.page_id, { result, applied_at: action.applied_at, group: action.group });
            }
        });
    });
    Object.entries(review.editorial_decisions || {}).forEach(([id, decision]) => {
        completed.set(id, {
            editorialDecision: decision,
            applied_at: decision.decided_at,
            group: 'manual_improvement'
        });
    });
    let pendingSafeCount = 0;
    let pendingSafeOnlyReviews = true;
    Object.entries(SEO_GROUP_LABELS).forEach(([group, label]) => {
        const ids = ((review.groups || {})[group] || []).filter(id => !completed.has(id));
        if (!ids.length) return;
        if (SEO_SAFE_APPLY_GROUPS.has(group)) {
            pendingSafeCount += ids.length;
            if (group !== 'keep_indexed') pendingSafeOnlyReviews = false;
        }
        ids.forEach(id => seoState.reviewGroupByPage.set(id, label));
        groupsContainer.appendChild(renderSeoReviewGroup(review, group, ids, pages));
    });

    const applyAll = document.getElementById('applyAllSeoReviewBtn');
    applyAll.hidden = !review.run_id || pendingSafeCount === 0;
    applyAll.dataset.runId = review.run_id || '';
    applyAll.textContent = pendingSafeOnlyReviews
        ? `Mark ${pendingSafeCount} pages reviewed`
        : `Apply ${pendingSafeCount} safe recommendations`;
    const openEditorial = ((review.groups || {}).manual_improvement || [])
        .filter(id => !completed.has(id)).length;
    document.getElementById('seoReviewProgress').textContent = review.run_id
        ? `${pendingSafeCount} safe actions still open · ${openEditorial} editorial decisions open · ${completed.size} pages completed · Telegram ${(review.telegram_notification || {}).status || 'pending'}.`
        : 'No review progress yet.';

    const completedPanel = document.getElementById('seoReviewCompleted');
    const completedList = document.getElementById('seoReviewCompletedList');
    completedList.textContent = '';
    completedPanel.hidden = completed.size === 0;
    completed.forEach((entry, id) => {
        const page = pages.get(id) || {};
        const row = appendSeoText(
            completedList, 'div',
            `✓ ${page.title || page.url || id} · ${entry.editorialDecision
                ? (SEO_EDITORIAL_DECISION_LABELS[entry.editorialDecision.decision] || entry.editorialDecision.decision)
                : (SEO_GROUP_LABELS[page.group] || page.group || 'completed')}`,
            'seo-review-page'
        );
        row.title = entry.applied_at ? new Date(entry.applied_at).toLocaleString() : '';
    });
    renderSeoTopicBrief(review, pages);

    const briefOpen = !document.getElementById('seoTopicBriefPanel').hidden;
    const empty = document.getElementById('seoWorklistEmpty');
    empty.hidden = groupsContainer.children.length > 0 || briefOpen;
    empty.textContent = review.run_id
        ? 'Nothing needs a decision right now.'
        : 'No weekly review has run yet, so there is nothing to decide.';
}

function renderSeoFindingList(container, label, items) {
    container.textContent = '';
    if (!items.length) return;
    appendSeoText(container, 'strong', label);
    const list = document.createElement('ul');
    items.forEach(item => appendSeoText(list, 'li', item));
    container.appendChild(list);
}

function describeSeoDelta(delta) {
    if (!delta || !delta.comparable) return 'no comparable previous run';
    const changed = delta.changed || [];
    const added = delta.new_pages || [];
    if (!changed.length && !added.length) return 'nothing changed';
    const parts = [];
    if (changed.length) parts.push(changed.length + ' status change' + (changed.length === 1 ? '' : 's'));
    if (added.length) parts.push(added.length + ' new');
    if (delta.removed) parts.push(delta.removed + ' gone');
    return parts.join(' · ');
}

function describeSeoStatusCounts(counts) {
    return Object.keys(counts || {}).map(key => key + ' ' + counts[key]).join(', ');
}

function renderSeoReviewDiagnostics(review) {
    const judgeState = document.getElementById('seoReviewJudgeState');
    const judgeError = document.getElementById('seoReviewJudgeError');
    if (!review.run_id) {
        judgeState.textContent = 'no review yet';
        judgeError.textContent = '';
    } else if (review.judge_called) {
        judgeState.textContent = 'answered';
        judgeError.textContent = '';
    } else {
        judgeState.textContent = review.judge_error ? 'failed' : 'not called';
        judgeError.textContent = review.judge_error
            || 'The assessment below was generated from the rules, not by the judge.';
    }

    const collection = review.collection || {};
    document.getElementById('seoReviewCollection').textContent = review.run_id
        ? (collection.status || 'unknown') + ' · '
            + (collection.metrics_written != null ? collection.metrics_written : '?') + ' rows written'
        : 'no review yet';
    document.getElementById('seoReviewCollectionMessage').textContent = review.run_id
        ? [
            collection.days_collected != null
                ? collection.days_collected + '/'
                    + (collection.days_requested != null ? collection.days_requested : '?') + ' days collected'
                : '',
            collection.gsc_rows_matched != null ? collection.gsc_rows_matched + ' GSC rows matched' : '',
            collection.message || ''
        ].filter(Boolean).join(' · ')
        : '';

    const deltaCard = document.getElementById('seoReviewDelta');
    deltaCard.textContent = review.run_id ? describeSeoDelta(review.delta) : 'no review yet';
    deltaCard.title = describeSeoStatusCounts(review.status_counts);

    const findingsPanel = document.getElementById('seoReviewFindings');
    const positive = (review.findings || {}).positive || [];
    const negative = (review.findings || {}).negative || [];
    findingsPanel.hidden = !positive.length && !negative.length;
    renderSeoFindingList(document.getElementById('seoReviewFindingsPositive'), 'Working', positive);
    renderSeoFindingList(document.getElementById('seoReviewFindingsNegative'), 'Not working', negative);
}

function renderSeoReviewHistory(entries) {
    const container = document.getElementById('seoReviewHistory');
    container.textContent = '';
    if (!entries.length) {
        appendSeoText(container, 'p', 'No reviews recorded yet.', 'section-hint');
        return;
    }
    entries.forEach(entry => {
        const box = document.createElement('div');
        box.className = 'seo-review-group';
        const when = entry.started_at ? new Date(entry.started_at).toLocaleString() : 'unknown time';
        appendSeoText(box, 'strong', when + ' · ' + entry.status
            + (entry.trigger ? ' · ' + entry.trigger : ''));
        appendSeoText(
            box, 'div',
            entry.pages_reviewed + ' pages · '
                + (describeSeoStatusCounts(entry.status_counts) || 'no page data')
                + ' · ' + describeSeoDelta(entry.delta)
                + ' · judge ' + (entry.judge_called ? 'answered' : (entry.judge_error || 'not called'))
                + ' · Telegram ' + (entry.telegram_status || 'unknown'),
            'section-hint'
        );
        appendSeoText(box, 'div', entry.summary || 'No summary.');
        const findings = ((entry.findings || {}).positive || []).map(item => '+ ' + item)
            .concat(((entry.findings || {}).negative || []).map(item => '− ' + item));
        if (findings.length) {
            const list = document.createElement('ul');
            findings.forEach(item => appendSeoText(list, 'li', item));
            box.appendChild(list);
        }
        container.appendChild(box);
    });
}

let seoReviewHistoryLoaded = false;

async function loadSeoReviewHistory(force) {
    if (seoReviewHistoryLoaded && !force) return;
    const container = document.getElementById('seoReviewHistory');
    container.textContent = 'Loading…';
    try {
        const data = await shareAdminRequest('GET', '/api/admin/seo/reviews?limit=12');
        seoReviewHistoryLoaded = true;
        renderSeoReviewHistory(data.reviews || []);
    } catch (err) {
        container.textContent = '';
        appendSeoText(container, 'p', err.message, 'section-hint');
    }
}

function renderSeoTopicBrief(review, pages) {
    const panel = document.getElementById('seoTopicBriefPanel');
    const proposed = review.proposed_topic_brief || '';
    panel.hidden = !proposed;
    if (!proposed) return;
    document.getElementById('seoCurrentTopicBrief').textContent = review.current_topic_brief || '';
    document.getElementById('seoProposedTopicBrief').textContent = proposed;
    document.getElementById('seoTopicBriefStrength').textContent = review.topic_brief_evidence_strength === 'weak'
        ? 'Weak evidence: only ' + (review.mature_page_count || 0) + ' reviewed pages are mature '
            + '(28 days old with 28 finalized rows). Shown for information; it cannot be applied.'
        : '';
    document.getElementById('seoTopicBriefReason').textContent = review.topic_brief_reason || '';
    document.getElementById('seoTopicBriefEvidence').textContent = (review.topic_brief_evidence_page_ids || [])
        .map(id => (pages.get(id) || {}).title || (pages.get(id) || {}).url || id).join(' · ');
    const decision = review.topic_brief_decision
        || (review.topic_brief_accepted_at ? 'accepted' : 'pending');
    const pending = decision === 'pending';
    document.getElementById('seoTopicBriefDecision').textContent = `Decision: ${decision}`;
    document.getElementById('acceptSeoTopicBriefBtn').disabled = !pending;
    document.getElementById('rejectSeoTopicBriefBtn').disabled = !pending;
    document.getElementById('acceptSeoTopicBriefBtn').dataset.runId = review.run_id || '';
    document.getElementById('rejectSeoTopicBriefBtn').dataset.runId = review.run_id || '';
}

async function previewAndApplySeoReview(runId, payload, confirmDelete) {
    const preview = await shareAdminRequest('POST', `/api/admin/seo/reviews/${encodeURIComponent(runId)}/preview`, payload);
    const lines = (preview.pages || []).map(page =>
        `• ${page.title || page.url}\n  index: ${page.current_indexed ? 'indexed' : 'noindex'}; Watch: ${page.current_watch_status}; planned: ${page.planned_action}`
    );
    const warning = confirmDelete
        ? '\n\nDeleting affects the Share, Watch, history, and followers and cannot be undone.' : '';
    const reviewOnly = (preview.pages || []).every(page => page.planned_action === 'mark_reviewed');
    const explanation = reviewOnly
        ? '\n\nThis only records that you reviewed these pages. Indexing, Watches, and content remain unchanged.'
        : '';
    if (!lines.length || !confirm(`Apply these actions?\n\n${lines.join('\n')}${explanation}${warning}`)) return null;
    return shareAdminRequest('POST', `/api/admin/seo/reviews/${encodeURIComponent(runId)}/apply`, {
        ...payload,
        confirm_delete: !!confirmDelete
    });
}

async function applySeoReviewGroup(runId, group, box, button) {
    const pageIds = [...box.querySelectorAll('input[data-page-id]:checked')].map(input => input.dataset.pageId);
    if (!pageIds.length) return;
    button.disabled = true;
    try {
        const result = await previewAndApplySeoReview(runId, { group, page_ids: pageIds, apply_all: false }, group === 'delete_candidate');
        if (result) {
            const failed = (result.results || []).filter(item => item.status !== 'success');
            seoStatus(`Weekly review action finished: ${(result.results || []).length - failed.length} succeeded, ${failed.length} failed.`, failed.length > 0);
            await loadSeoOverview();
        }
    } catch (err) {
        seoStatus(err.message, true);
    } finally {
        button.disabled = false;
    }
}

async function saveSeoEditorialDecision(runId, pageId, decision, note, button) {
    if (!confirm(`Record the decision “${SEO_EDITORIAL_DECISION_LABELS[decision] || decision}”? This records the follow-up but does not mutate an immutable snapshot.`)) return;
    button.disabled = true;
    try {
        await shareAdminRequest('POST', `/api/admin/seo/reviews/${encodeURIComponent(runId)}/editorial-decision`, {
            page_id: pageId,
            decision,
            note: note || ''
        });
        seoStatus('Editorial decision recorded.', false);
        await loadSeoOverview();
    } catch (err) {
        seoStatus(err.message, true);
        button.disabled = false;
    }
}

// The endpoint is per page. Several calls are fine, but every one is
// acknowledged on its own and partial failures have to be named.
async function confirmSuggestedSeoDecisions(runId, entries, button) {
    if (!runId || !entries.length) return;
    const preview = entries.slice(0, 12).map(entry =>
        `• ${entry.title}\n  ${SEO_EDITORIAL_DECISION_LABELS[entry.decision] || entry.decision}`
    ).join('\n');
    const more = entries.length > 12 ? `\n… and ${entries.length - 12} more` : '';
    if (!confirm(
        `Record the suggested decision for ${entries.length} pages?\n\n${preview}${more}`
        + '\n\nThis records the follow-up for each page and does not mutate any immutable snapshot.'
    )) return;
    button.disabled = true;
    let succeeded = 0;
    const failures = [];
    for (const entry of entries) {
        try {
            await shareAdminRequest(
                'POST', `/api/admin/seo/reviews/${encodeURIComponent(runId)}/editorial-decision`,
                { page_id: entry.pageId, decision: entry.decision, note: '' }
            );
            succeeded += 1;
        } catch (err) {
            failures.push(`${entry.title}: ${err.message}`);
        }
    }
    seoStatus(
        failures.length
            ? `${succeeded} decisions recorded, ${failures.length} failed — ${failures.join(' · ')}`
            : `${succeeded} suggested decisions recorded.`,
        failures.length > 0
    );
    await loadSeoOverview();
}

document.getElementById('saveSeoReviewConfigBtn').addEventListener('click', async function () {
    this.disabled = true;
    try {
        await shareAdminRequest('PUT', '/api/admin/seo/review/config', {
            enabled: document.getElementById('seoReviewEnabled').checked,
            interval_days: Number(document.getElementById('seoReviewInterval').value || 7),
            run_time: document.getElementById('seoReviewTime').value || '09:00',
            timezone: document.getElementById('seoReviewTimezone').value.trim() || 'Europe/Berlin'
        });
        seoStatus('Weekly review configuration saved.', false);
        await loadSeoOverview();
    } catch (err) {
        seoStatus(err.message, true);
    } finally { this.disabled = false; }
});

document.getElementById('seoReviewHistoryPanel').addEventListener('toggle', function () {
    if (this.open) loadSeoReviewHistory(false);
});

document.getElementById('runSeoReviewBtn').addEventListener('click', async function () {
    if (!confirm('Run Search Console collection and the weekly portfolio review now? At most one portfolio Judge call may be made.')) return;
    this.disabled = true;
    seoStatus('Running weekly SEO review…', false);
    try {
        const result = await shareAdminRequest('POST', '/api/admin/seo/review/run', {});
        seoStatus(result.summary || result.status, result.status === 'error' || result.status === 'collection_failed');
        await loadSeoOverview();
        if (seoReviewHistoryLoaded) await loadSeoReviewHistory(true);
    } catch (err) {
        seoStatus(err.message, true);
    } finally { this.disabled = false; }
});

document.getElementById('applyAllSeoReviewBtn').addEventListener('click', async function () {
    const runId = this.dataset.runId || '';
    if (!runId) return;
    this.disabled = true;
    try {
        const result = await previewAndApplySeoReview(runId, { apply_all: true, page_ids: [] }, false);
        if (result) {
            const failed = (result.results || []).filter(item => item.status !== 'success');
            seoStatus(`Safe recommendations: ${(result.results || []).length - failed.length} succeeded, ${failed.length} failed.`, failed.length > 0);
            await loadSeoOverview();
        }
    } catch (err) { seoStatus(err.message, true); }
    finally { this.disabled = false; }
});

document.getElementById('acceptSeoTopicBriefBtn').addEventListener('click', async function () {
    const runId = this.dataset.runId;
    if (!runId || !confirm('Replace only the current Topic Brief with this suggestion? Other Publisher settings stay unchanged.')) return;
    this.disabled = true;
    try {
        await shareAdminRequest('POST', `/api/admin/seo/reviews/${encodeURIComponent(runId)}/topic-brief/accept`, {});
        seoStatus('Suggested Topic Brief accepted.', false);
        await loadPublisherConfig();
        await loadSeoOverview();
    } catch (err) { seoStatus(err.message, true); }
    finally { this.disabled = false; }
});

document.getElementById('rejectSeoTopicBriefBtn').addEventListener('click', async function () {
    const runId = this.dataset.runId;
    if (!runId || !confirm('Reject this Publisher Topic Brief suggestion and keep the current prompt unchanged?')) return;
    this.disabled = true;
    try {
        await shareAdminRequest('POST', `/api/admin/seo/reviews/${encodeURIComponent(runId)}/topic-brief/reject`, {});
        seoStatus('Suggested Topic Brief rejected.', false);
        await loadSeoOverview();
    } catch (err) { seoStatus(err.message, true); }
    finally { this.disabled = false; }
});

async function runSeoRecommendation(pageId, useContentJudge, button) {
    button.disabled = true;
    seoStatus(useContentJudge ? 'Asking the optional content judge…' : 'Generating recommendation…', false);
    const suffix = useContentJudge ? 'content-judge' : 'recommendation';
    try {
        const result = await shareAdminRequest('POST', `/api/admin/seo/pages/${encodeURIComponent(pageId)}/${suffix}`, {});
        const label = result.llm_evaluation
            ? `${result.recommendation} / content judge: ${result.llm_evaluation.recommendation}`
            : result.recommendation;
        seoStatus(`Saved read-only recommendation: ${label}.`, false);
        await loadSeoOverview();
    } catch (err) {
        seoStatus(err.message, true);
    } finally {
        button.disabled = false;
    }
}

function renderSeoConnection(result) {
    seoState.connection = result;
    const state = document.getElementById('seoConnectionState');
    state.textContent = result.connected ? 'connected' : (result.status || 'connection failed').replaceAll('_', ' ');
    state.title = result.message || '';
    document.getElementById('seoConnectionMessage').textContent = result.message || '';
    renderSeoAlerts();
}

async function loadSeoOverview() {
    seoStatus('Loading…', false);
    try {
        const data = await shareAdminRequest('GET', '/api/admin/seo');
        renderSeoOverview(data);
        // A successful admin-only request is the visibility gate for
        // controls that can make server-side Search Console requests.
        document.getElementById('collectSeoBtn').hidden = false;
        document.getElementById('checkSeoConnectionBtn').hidden = false;
        seoStatus('', false);
    } catch (err) {
        document.getElementById('collectSeoBtn').hidden = true;
        document.getElementById('checkSeoConnectionBtn').hidden = true;
        seoStatus(err.message, true);
    }
}

document.getElementById('reloadSeoBtn').addEventListener('click', loadSeoOverview);
document.getElementById('checkSeoConnectionBtn').addEventListener('click', async function () {
    this.disabled = true;
    seoStatus('Checking Search Console connection…', false);
    try {
        const result = await shareAdminRequest('POST', '/api/admin/seo/check', {});
        renderSeoConnection(result);
        seoStatus(result.message || result.status, !result.connected);
    } catch (err) {
        renderSeoConnection({ connected: false, status: 'connection_failed', message: err.message });
        seoStatus(err.message, true);
    } finally {
        this.disabled = false;
    }
});
document.getElementById('collectSeoBtn').addEventListener('click', async function () {
    this.disabled = true;
    seoStatus('Collecting finalized Search Console data…', false);
    try {
        const result = await shareAdminRequest('POST', '/api/admin/seo/collect', {});
        seoStatus(result.message || result.status || 'Collection finished.', result.status === 'error');
        await loadSeoOverview();
    } catch (err) {
        seoStatus(err.message, true);
    } finally {
        this.disabled = false;
    }
});

onAuthStateChanged(auth, async (user) => {
    if (user) {
        const idToken = await user.getIdToken();
        fetchModels(idToken);
        loadPublisherConfig();
        loadPublisherWatches();
        loadApiKeys();
        loadShares('reported');
        loadAdminWatches();
        loadAdminTopics();
        loadSeoOverview();
    } else {
        setStatus('Please log in to access the admin panel.', true);
        window.location.href = '/';
    }
});
