# consens.io — Consensus Benchmark Snapshot (MMLU‑Pro) — Plan

> Status: **Phase 2 umgesetzt.** Dieses Dokument ist die verbindliche Grundlage.
> Gebaut: `benchmark_mode` (default-off, 4.1), das isolierte `benchmark/`-Paket,
> disjunkte Pilot-/Final-Manifeste (5 / 98 = 7×14, committet), Tests aus §9 (1–5)
> grün, `--dry-run` mit Kostenprojektion + Tool-Audit. Es wurde **kein** API-Call
> an ein LLM ausgeführt. Stand: 2026-06-28.
>
> **Korrektur zu §2:** Die Annahme „keine neue Dependency zwingend" war falsch —
> `pandas.read_parquet` braucht ein Parquet-Engine (pyarrow/fastparquet), das im
> Repo nicht vorhanden ist. Lösung: pyarrow als **benchmark-only Dev-Dependency**
> (`benchmark/requirements-benchmark.txt`); die Produktions-`requirements.txt` und
> das Render-Image bleiben unverändert.

## 0. Zielbild

Eine spätere statische Unterseite auf consens.io soll den Mehrwert des Produkts
zeigen — über einen kleinen, reproduzierbaren Benchmark auf **98 Fragen aus
MMLU‑Pro** (Multiple-Choice, closed-book; 14 Kategorien × 7 Fragen, sauber
gleichverteilt):

1. Alle 6 aktuellen Consensus-Modelle beantworten dieselbe MC-Frage.
2. Die bestehende Consensus-Logik erzeugt daraus eine finale Antwort.
3. Auswertung gegen die bekannte Ground Truth des Datensatzes.
4. Verglichen werden: jedes Einzelmodell, **Majority Vote**, **Consensus** und
   optional der **Consensus-Synthesizer allein** (ohne Fremdantworten).
5. Besonders relevant: die Teilmenge der Fragen, bei denen die Modelle uneinig sind.

Kernaussage später **nicht** „consens.io schlägt jedes Modell immer", sondern:
„Bei objektiv bewertbaren Fragen zeigt Consensus, wie mehrere Modellperspektiven
verglichen und zu einer besseren bzw. stabileren Antwort zusammengeführt werden —
besonders wenn Modelle widersprechen."

### Rahmenbedingungen (verbindlich)

- MMLU‑Pro ist ein **closed-book**-Benchmark. Für diesen Lauf müssen Websuche,
  Search Mode, Retrieval, Tools und externe Quellen für **alle** Modelle
  deaktiviert sein.
- In der normalen App bleibt Webzugriff **unverändert** standardmäßig aktiv. Der
  Benchmark darf normale Nutzer **niemals** beeinflussen.
- Der interne Schalter (`benchmark_mode`) ist **kein** sichtbarer UI-Switch und
  **kein** HTTP-Request-Feld — er existiert nur als Funktionsparameter im
  in-process aufgerufenen Code.
- Der Lauf geht **nicht** über die GUI, sondern über einen separaten Runner, der
  möglichst viel bestehende produktive Modell- und Consensus-Logik wiederverwendet.
- Alle Rohantworten, Prompts, Modell-IDs, Einstellungen, Token-Usage, Kosten,
  Zeitstempel und Auswertungen werden gespeichert (nachvollziehbar + fortsetzbar).
- Erst **Pilot mit 5 Fragen**, danach fester Run mit **98 Fragen**.
- Kostenkontrolle: Dry-Run, Budget-Limit, Resume.
- **v1-Scope:** nur 6 Einzelantworten, Consensus und optional Synthesizer-allein.
  **Kein** Differences-Run (siehe E7).

---

## 1. Zentraler technischer Befund

Websuche ist nicht konfigurierbar, sondern in `build_provider_payload`
(`app/services/llm/engines.py:168`) **pro Provider fest verdrahtet**. Das ist der
**einzige** produktive Eingriffspunkt, den der Benchmark braucht.

Die Consensus-Synthese selbst (`app/services/llm/consensus_engine.py`) ist
**bereits tool-frei** (reine Chat/Messages-Calls; Gemini poppt `tools` in
`consensus_engine.py:63`) — dort ist nichts zu ändern.

| Provider | Stelle in engines.py | injiziertes Web-Tool |
|---|---|---|
| OpenAI | `:126‑129` (`_openai_responses_payload`) | `tools:[{type:web_search}]` + `include:[web_search_call…]` |
| Mistral | `:243` | `tools:[{type:web_search}]` |
| Anthropic | `:268‑272` | `web_search_20250305` |
| Gemini | `:302` | `tools:[{google_search:{}}]` |
| Grok | `:357` (via `_openai_responses_payload`) | `web_search` |
| DeepSeek | `:336‑344` | **kein Tool** → bereits closed-book |

Zusätzlich injiziert `/prepare` (`app/api/routers/chat.py:781`) Echtzeitkontext
via `get_intent_from_llm` + `get_realtime_context` (`tool_heuristics.py`). Der
Runner **umgeht `/prepare` komplett** und baut den System-Prompt selbst → diese
Quelle ist damit automatisch aus.

---

## 2. Analyse der wiederverwendbaren Komponenten

- **Modellkonfiguration:** `app/core/config.py` — `DEFAULT_MODEL_BY_PROVIDER`
  (`:75`), `get_model_config` / `resolve_api_model` (`:313‑332`).
  **Achtung:** `load_models_from_db()` (`:439`) überschreibt Defaults beim Startup
  aus Firestore. Der Benchmark **darf nicht** davon abhängen und friert die 6
  Modell-IDs explizit in seiner eigenen Config ein.
- **Provider-Adapter:** `build_provider_payload` (`engines.py:168`) +
  `query_openai/_mistral/_claude/_gemini/_deepseek/_grok` (`engines.py:416‑684`).
- **Response-Parser:** `parse_openai_response` / `parse_anthropic_response` /
  `parse_gemini_response` / `parse_mistral_content` (`citations.py:208‑360`) →
  liefern `{"text", "sources"}`.
- **Consensus-Synthese (wiederverwenden):** `query_consensus` / `stream_consensus`
  (`consensus_engine.py:248, 963`). In-process aufrufen, **nicht** über
  `/consensus`. `query_differences` / `stream_differences` (`:694, 1101`) werden in
  **v1 nicht** verwendet (E7).
- **Label-Verhalten (für den optionalen Pilot-Audit relevant):** Die Consensus-Engine
  schreibt **echte Klarnamen** in den Prompt (`_format_expert_opinion("OpenAI", …)`,
  `consensus_engine.py:175‑221`) — das ist der zu messende reale Zustand. Die
  Differences-Engine anonymisiert dagegen bereits (`Model A/B/…` via
  `random.shuffle`, `:439‑449`) und dient als Vorlage, **falls** Anonymisierung
  nach dem Pilot adoptiert wird (E5).
- **Persistenz-Lücke:** Token-Usage/Kosten werden **nirgends** erfasst —
  `parse_*_response` verwirft `usage`/`usageMetadata`; das einzige „usage" im Code
  ist der In-Memory-Quota-Zähler (`app/core/state.py`). Der Benchmark muss Usage
  selbst aus den Roh-JSONs ziehen.
- **Dependencies:** `huggingface-hub==0.29.1` + `pandas==2.3.3` sind vorhanden →
  MMLU‑Pro per `hf_hub_download` + `pandas.read_parquet` laden, **ohne** die
  schwere `datasets`-Lib. Input-Token-Schätzung im Dry-Run via `len/4`-Heuristik
  statt `tiktoken`. **Korrektur (Phase 2):** `pandas.read_parquet` benötigt
  zusätzlich ein Parquet-Engine — pyarrow ist als benchmark-only Dev-Dependency
  ergänzt (`benchmark/requirements-benchmark.txt`), Produktion unverändert.

---

## 3. Beschlossene Entscheidungen

### E1 — Usage-Erfassung: separate schlanke Transport-Schicht ✅
Es wird eine eigene `benchmark/transport.py` gebaut, die:
- `build_provider_payload(..., benchmark_mode=True)` für Payload + Modellauflösung
  wiederverwendet,
- die Response-Parser aus `citations.py` für Text + Quellen wiederverwendet,
- den **HTTP-Call und die Usage-Erfassung selbst** übernimmt (Usage aus demselben
  Roh-JSON, das auch der Parser bekommt).

Die produktiven `query_*`-Funktionen bleiben **unangetastet**.

**Absicherung gegen Drift** (Pflicht):
- Unit-Tests für den Transport je Provider gegen kanonische Provider-JSONs
  (gemockt), inkl. korrekter Usage-Extraktion.
- Ein späterer **Cross-Check** gegen die produktive Request-Logik: für denselben
  Prompt/dasselbe Modell wird der Benchmark-Transport-Text mit dem Ergebnis eines
  echten `query_*`-Calls verglichen (Toleranz für LLM-Nichtdeterminismus
  beachten; primär struktureller Abgleich von Payload/Parsing-Pfad).

### E2 — Majority-Tie: `no_majority` ✅
Bei Gleichstand der Mehrheit wird **`no_majority` gespeichert**. **Kein** zufälliger
Tie-Break. `no_majority` ist ein eigener Bucket und wird in der Auswertung separat
ausgewiesen (zählt nicht als korrekt).

### E3 — Pilot ist ein separates Sample ✅
Der 5-Fragen-Pilot ist ein **eigenständiges Development-/Pilot-Sample** und wird
**ausdrücklich nicht** Teil des finalen 98-Fragen-Samples. Beide Samples sind
**disjunkt** (keine überlappenden `question_id`s). `dataset.py` zieht zuerst das
Pilot-Sample, schließt dessen IDs aus und zieht danach das finale Sample aus den
verbleibenden Fragen. Beide werden als eingefrorene, committete Manifeste
abgelegt.

### E3b — Finales Sample: 98 Fragen, gleichverteilt ✅
MMLU‑Pro hat 14 Kategorien. Das finale Sample ist **7 Fragen × 14 Kategorien = 98**
(exakt gleichverteilt — wissenschaftlich sauberer als ein ungerades 100er-Sample).
`dataset.py` zieht pro Kategorie deterministisch (fester `final_seed`) 7 Fragen aus
dem um die Pilot-IDs bereinigten Pool. Sollte eine Kategorie wider Erwarten < 7
verfügbare Fragen haben, bricht das Sampling mit Fehler ab (kein stilles Auffüllen).

### E4 — Reihenfolge-Audits ✅
Zwei Audit-Ebenen verpflichtend:

1. **Tool-/Sicherheits-Audit:** `assert_no_web_tools(payload)` über **jeden**
   erzeugten Payload (im Dry-Run und vor jedem realen Call). Bricht ab, falls
   irgendwo ein Web-Tool-Key auftaucht.

2. **Optionen-Permutations-Audit** (Positions-Bias der Einzelmodelle): auf einem
   kleinen Subset werden die Antwortoptionen umsortiert; geprüft wird, dass die
   extrahierte Antwort auf denselben **Options-Inhalt** zeigt (nicht auf eine feste
   Buchstabenposition).

3. **Consensus-Reihenfolge-Audit** (Stabilität der Consensus-Synthese): Bei
   identischen, bereits gespeicherten Kandidatenantworten wird **nur der Consensus
   neu berechnet** — **keine** erneute Anfrage an die Kandidatenmodelle — in drei
   Reihenfolgen:
   - normale Reihenfolge,
   - umgekehrte Reihenfolge,
   - deterministisch zufällig gemischte Reihenfolge (fester Seed).
   Geprüft/protokolliert wird, ob die extrahierte Consensus-Antwort über die drei
   Reihenfolgen stabil bleibt (Reihenfolge-Invarianz der Synthese). Der Audit ist
   vom Label-Modus unabhängig; im Pilot kann er sowohl mit Modellnamen als auch im
   anonymisierten Audit-Modus (E5) laufen.

### E5 — Label-Modus: Hauptpfad mit Modellnamen, Anonymisierung nur als Pilot-Audit ✅
Der **primäre Benchmark misst zunächst die reale consens.io-Experience.** Dort sieht
der Consensus aktuell die Modellnamen (`_format_expert_opinion("OpenAI", …)`). Der
**Hauptpfad im Runner verwendet deshalb die bestehende Consensus-Logik unverändert
mit Modellnamen** — keine verpflichtende produktive Änderung an der Anonymisierung.

Für den **5-Fragen-Pilot** unterstützt der Runner **zusätzlich** einen *optionalen*
anonymisierten **Audit-Modus** zum Vergleich:
- nutzt **dieselben bereits gespeicherten** sechs Kandidatenantworten,
- berechnet den Consensus **einmal mit Modellnamen** und **einmal als
  `Response A–F`**,
- **ohne** die Kandidatenmodelle erneut anzufragen (nur Consensus neu berechnen).

Die Provider→Buchstabe-Zuordnung im Audit-Modus wird deterministisch (fester Seed)
vergeben und protokolliert.

**Entscheidung über den finalen Label-Modus fällt erst nach dem Pilot** — anhand von
Ergebnis, Stabilität und Zusatzkomplexität wird festgelegt, ob der finale
98-Fragen-Snapshot mit Modellnamen oder anonymisiert läuft. Bis dahin ist
Anonymisierung **kein** Standard und **keine** fest eingeplante Produktiv-Änderung
(siehe §4.2 und §10).

### E6 — Run-Einstellungen vor dem finalen Run einfrieren ✅
Vor den 98 Fragen müssen in `benchmark/config.py` **fest stehen** und ins Manifest
geschrieben werden:
- exakte Modell-IDs (`internal_id` + aufgelöstes `api_model`) je Provider,
- Reasoning-Stufe je Modell (z. B. `low_config` / kein Reasoning),
- Temperatur,
- Output-Token-Limit,
- Consensus-Modell (Pin).

**Nach dem Pilot** dürfen diese Werte noch verbessert werden. **Danach nicht mehr**
— der finale Run läuft auf eingefrorener Config; jede spätere Änderung erzwingt
einen neuen `run_id` und einen neuen Snapshot.

### E7 — Kein Differences-Run in v1 ✅
Für die Messung werden nur **6 Einzelantworten**, **Consensus** und optional der
**Synthesizer allein** gebraucht. `query_differences` / `stream_differences` werden
in v1 **nicht** aufgerufen — sie bringen keinen Messmehrwert, kosten aber Calls und
erhöhen die Runner-Komplexität. Die Rolle `differences` entfällt im Schema.

---

## 4. Produktiv-Änderungen

**Genau eine** verpflichtende, default-off Änderung (4.1). Die Consensus-
Anonymisierung (4.2) ist **nicht** fest eingeplant, sondern bedingt und
aufgeschoben (E5). Produktion bleibt in beiden Fällen byte-identisch, weil die
Defaults nichts ändern und die `/ask_*`- bzw. `/consensus`-Pfade die Flags nie
setzen.

### 4.1 `benchmark_mode` (Web-Tools deaktivieren) — verpflichtend
`build_provider_payload` (`engines.py:168`) erhält `benchmark_mode: bool = False`.
Bei `True`:
- OpenAI/Grok: `tools`, `tool_choice`, `include` weglassen
  (`_openai_responses_payload`, `:110`).
- Mistral: `tools` weglassen (`:243`).
- Anthropic: `tools`-Block weglassen (`:268`).
- Gemini: `tools` weglassen (`:302`).
- DeepSeek: no-op (Parameter konsistent durchreichen).

Default `False`; die `/ask_*`-Router setzen das Flag nie. Optionaler Name
`evaluation_mode` — kein UI-Switch, kein Request-Body-Feld. Da der Runner in-process
läuft, bedeutet „pro Request": pro Modell-Call wird `benchmark_mode=True` an
`build_provider_payload` übergeben.

### 4.2 Consensus-Anonymisierung — bedingt, nur falls nach Pilot adoptiert (E5)
**Nicht** Teil des verpflichtenden Scopes. Der Hauptpfad nutzt die Consensus-Logik
unverändert mit Modellnamen. Anonymisierung wird **nur** gebraucht für (a) den
optionalen Pilot-Audit und (b) — *falls nach dem Pilot so entschieden* — den finalen
Snapshot.

Wenn benötigt, ist die leichteste Variante eine optionale neutrale Label-Quelle in
`_build_consensus_prompt` / `query_consensus` / `stream_consensus`
(`consensus_engine.py:184, 248, 963`): Flag `anonymize_labels: bool = False` bzw.
eine `label_map`, sodass `_format_expert_opinion` `Response A`..`Response F` statt
der Klarnamen nutzt; Synthese-Mechanismus identisch, Default aus ⇒ Produktion
unverändert. Die konkrete Umsetzung (optionaler Flag vs. benchmark-lokaler
Prompt-Builder für den Audit) wird erst festgelegt, wenn der Pilot-Audit gebaut
wird — sie ist **kein** committeter Bestandteil von Phase 2.

---

## 5. Runner-Struktur

Neues, isoliertes Paket `benchmark/` (kein Import aus `app.api.*`, nur aus
`app.services.llm.*` + `app.core.config`):

```
benchmark/
  config.py        # 6 eingefrorene Modell-IDs, consensus_model-Pin, seeds,
                   #   sample sizes, FIXIERTER closed-book system prompt,
                   #   pricing-Tabelle, Pfade
  dataset.py       # hf_hub_download(TIGER-Lab/MMLU-Pro) + disjunktes,
                   #   reproduzierbares Pilot- und Final-Sampling + Manifeste
  prompt.py        # MC-Template: Optionen A–J + Instruktion "The answer is (X)"
  transport.py     # execute(request_data, api_key)
                   #   -> {text, sources, usage, raw, status, latency}
  parse.py         # extract_letter(), majority_vote() (mit no_majority), grade()
  cost.py          # usage -> USD aus pricing-Tabelle
  runner.py        # Orchestrierung: Zellen-Loop, JSONL-Writer, Resume, Budget
  audit.py         # assert_no_web_tools + Optionen-Permutations-Audit
                   #   + Consensus-Reihenfolge-Audit (Response A–F: normal/
                   #   umgekehrt/gemischt, nur Consensus neu berechnen)
  __main__.py      # CLI: --dry-run --pilot --limit N --budget USD --resume <run_id>
```

Zusätzliche Daten-/Doku-Artefakte:

```
data/benchmark/
  mmlu_pro_pilot_v1.json     # eingefrorene Pilot-IDs (committet)
  mmlu_pro_sample_v1.json    # eingefrorene finale IDs (committet, disjunkt zum Pilot)
  runs/<run_id>/manifest.json
  runs/<run_id>/calls.jsonl
  runs/<run_id>/results.json
tests/
  test_benchmark_mode.py
  test_benchmark_parse.py
  test_benchmark_dataset.py
  test_benchmark_runner.py
  test_benchmark_transport.py
  test_benchmark_anonymize.py
  fixtures/mmlu_pro_mini.parquet
docs/benchmark-plan.md   # dieses Dokument
```

**Datenfluss pro Frage (eine „Zelle" = ein API-Call):**
1. `prompt.build_mc_question(row)` → MC-Frage-Text.
2. Pro Provider: `build_provider_payload(provider, question=…,
   system_prompt=FIXED, model_override=<pin>, benchmark_mode=True)` →
   `transport.execute(...)`. Text via `parse_*`, Usage zusätzlich aus dem Roh-JSON.
3. `parse.extract_letter` pro Modellantwort.
4. **Hauptpfad:** `query_consensus(...)` (Produktionslogik, **unverändert, mit
   Modellnamen**) → Consensus-Text → `extract_letter`.
5. Optional „Synthesizer allein": einzelner Call des Consensus-Modells nur mit der
   MC-Frage (kein Fremdkontext).
6. `parse.majority_vote` über die 6 Letters (Tie → `no_majority`).
7. Alle Schritte als JSONL-Zeilen anhängen.

**Nur im 5-Fragen-Pilot (optionaler Audit-Modus, E5):** auf denselben gespeicherten
Antworten den Consensus zusätzlich anonymisiert (`Response A–F`) neu berechnen — kein
erneuter Kandidaten-Call — und beide Label-Modi gegenüberstellen.

**Kein Differences-Schritt** (E7).

---

## 6. MMLU‑Pro laden & reproduzierbar samplen

- **Quelle:** `hf_hub_download(repo_id="TIGER-Lab/MMLU-Pro",
  filename="data/test-*.parquet", repo_type="dataset")` → `pandas.read_parquet`.
  Felder u. a.: `question_id`, `question`, `options` (Liste, bis 10 → A–J),
  `answer` (Buchstabe), `answer_index`, `category`.
- **Disjunktes, reproduzierbares Sampling (E3):**
  1. Pilot-Sample (5 Fragen) mit festem `pilot_seed` ziehen →
     `data/benchmark/mmlu_pro_pilot_v1.json` committen.
  2. Pilot-IDs aus dem Pool ausschließen.
  3. Finales Sample mit festem `final_seed` ziehen: **7 Fragen pro Kategorie über
     alle 14 Kategorien = 98** (E3b) → `data/benchmark/mmlu_pro_sample_v1.json`
     committen.
  Ab dann sind beide Samples eingefroren und unabhängig von Seed/HF-Version exakt
  rekonstruierbar.
- **Reproduzierbarkeit vs. `get_system_prompt()`:** `get_system_prompt()`
  (`base.py:7`) bettet das **Tagesdatum** ein → nicht reproduzierbar. Der
  Benchmark nutzt einen **fixierten** closed-book System-Prompt aus dem Manifest,
  **nicht** `/prepare` und **nicht** `get_system_prompt()`.
- **Manifest pro Run** (`runs/<run_id>/manifest.json`) — schreibt die **eingefrorene
  Config** (E6): Sample-Version + IDs, Seeds, 6 Modell-Pins (`internal_id` +
  aufgelöstes `api_model`), Reasoning-Stufe je Modell, Temperatur,
  Output-Token-Limit, consensus_model-Pin, **Label-Modus des Runs
  (`names`/`anon`)**, fixierter System-Prompt, Pricing-Tabelle, Git-Commit-Hash,
  Lib-Versionen, Startzeit.

---

## 7. Parsing & Bewertung

- **`extract_letter(text)`** — Regex-Kaskade: (1) `the answer is \(?([A-J])\)?`,
  (2) letzte Zeile `^([A-J])[).]`, (3) Match auf Options-Text, (4) sonst `None` →
  `abstain` (nicht korrekt, aber separat gezählt).
- **`grade`** — Vergleich mit `answer` (Buchstabe). Pro Frage gespeichert: Letter
  je Modell, Majority-Letter (oder `no_majority`), Consensus-Letter, optional
  Synth-Alone-Letter, jeweils `correct` bool.
- **Vergleichsgrößen** (Post-Processing aus `calls.jsonl`): Accuracy je
  Einzelmodell, Majority Vote, Consensus, optional Synthesizer-allein.
- **Uneinigkeits-Teilmenge:** Frage gilt als „uneinig", wenn nicht alle 6 Letters
  identisch sind. Accuracy aller Aggregationen wird zusätzlich **nur** auf dieser
  Teilmenge berechnet — das ist die kommunikativ relevante Aussage.

### Pro API-Call gespeicherte Felder (`calls.jsonl`)
`run_id, ts, question_id, category, role (model|consensus|synth_alone),
provider, internal_model, api_model, benchmark_mode, label_mode (names|anon),
system_prompt, user_prompt, request_payload (Keys redigiert), raw_response (oder
text+usage), parsed_text, extracted_letter, ground_truth, correct, abstain,
usage{prompt,completion,total}, est_cost_usd, latency_ms, http_status, error,
error_code`.

Der Hauptpfad schreibt `label_mode = names`. Nur die optionalen Pilot-Audit-Zeilen
schreiben `label_mode = anon` und zusätzlich `anon_map` (Provider→Buchstabe,
deterministisch). Rolle `differences` entfällt (E7).

---

## 8. Dry-Run, Budget-Cap, Resume

- **Dry-Run (`--dry-run`):** baut alle Payloads, **kein HTTP**. Gibt Zellzahl,
  geschätzte Input-Tokens (`len/4`), Output-Cap (aus `max_tokens`) × Pricing →
  projizierte Maximalkosten. Läuft `audit.assert_no_web_tools(payload)` über jeden
  Payload.
- **Budget-Cap (`--budget USD`):** laufender Ist-Kosten-Zähler aus realer Usage;
  vor jeder Zelle: wenn `ist + nächste_schätzung > cap` → sauberer, resumebarer
  Stopp.
- **Resume (`--resume <run_id>`):** Zelle idempotent über Key
  `(question_id, role, provider)`. Beim Start `calls.jsonl` lesen, erfolgreiche
  Zellen überspringen, fehlerhafte optional erneut versuchen. Append-only.

---

## 9. Tests vor dem ersten echten API-Run

Alle **ohne** echte API-Calls (Mocks/Fixtures); bestehende Pytest-Baseline
(zuletzt 145 passed) muss grün bleiben:

1. `test_benchmark_mode.py`: `benchmark_mode=True` entfernt Web-Tools für **alle 6**
   Provider; `False` ⇒ Payload identisch zum Status quo (Regression).
2. `test_benchmark_parse.py`: `extract_letter` auf Fixtures („The answer is (C).",
   „C)", „**C**", deutsch, Müll→None); `majority_vote` inkl. Ties → `no_majority`;
   `grade`.
3. `test_benchmark_dataset.py`: Sampling deterministisch (gleiche Seeds → gleiche
   IDs) gegen committete Mini-Fixture (`tests/fixtures/mmlu_pro_mini.parquet`),
   offline; Pilot- und Final-Sample **disjunkt**; finales Sample = 7×Kategorie.
4. `test_benchmark_runner.py`: Resume-Skip-Logik bei partieller JSONL;
   Budget-Estimator stoppt bei Cap; Dry-Run-Audit schlägt an, wenn Tool injiziert.
5. `test_benchmark_transport.py`: Usage + Text-Extraktion aus kanonischen
   Provider-JSONs (gemockt) je Provider (E1-Drift-Absicherung).
6. `test_benchmark_anonymize.py` *(nur falls der anonymisierte Pilot-Audit gebaut
   wird, E5)*: anonymisierter Modus erzeugt einen Consensus-Prompt **ohne**
   Provider-Klarnamen (nur `Response A–F`); der Hauptpfad-/Namens-Modus bleibt
   byte-identisch zum Status quo (Regression).

---

## 10. Offene spätere Entscheidungen

- **Pricing-Tabelle** existiert nicht im Repo → muss in `benchmark/config.py`
  manuell gepflegt werden (USD/1M Tokens je `api_model`); Kosten sind Schätzungen.
- **Consensus-Output-Format:** Der Consensus-Prompt ist auf Prosa ausgelegt, nicht
  auf MC. Da die MC-Instruktion in der eingebetteten Frage steckt, gibt Consensus
  i. d. R. einen Letter aus; der Parser fängt Abweichungen ab. Falls die
  Extraktions-Trefferquote zu niedrig ist → später entscheiden, ob ein
  benchmark-eigener Consensus-Hinweis nötig ist (würde „bestehende Logik
  unverändert" leicht aufweichen).
- **LLM-Nichtdeterminismus:** identische Inputs ≠ identische Outputs (temp 0.2 bei
  Mistral/Gemini, Reasoning-Modelle). Der Snapshot ist nachvollziehbar (Roh-Antworten
  gespeichert), aber **nicht** bit-reproduzierbar — auf der Website transparent machen.
- **Modell-Pins ≠ Firestore-Produktion:** der Benchmark misst die eingefrorenen
  IDs; Abweichungen zu Firestore sind im Manifest dokumentiert (gewollt).
- **Finaler Label-Modus (names vs. anon)** — **wird nach dem Pilot entschieden**
  (E5), anhand von Ergebnis, Reihenfolge-/Label-Stabilität und Zusatzkomplexität.
  Default-Annahme bis dahin: Hauptpfad mit Modellnamen (reale Experience).
- **MMLU‑Pro-Lizenz/Attribution** auf der späteren Website korrekt angeben.
- **Synthesizer-allein optional:** endgültig entscheiden, ob diese vierte Größe in
  v1 mitläuft (zusätzliche Calls) oder erst in v2.

---

## 11. Phasen

**Phase 1 — Analyse/Architektur** *(erledigt; dieses Dokument).* Befunde, E1–E4
beschlossen, `benchmark/`-Layout + Manifest-Schema festgezurrt.

**Phase 2 — Infrastruktur ohne API-Calls:** *(umgesetzt, Stand 2026-06-28.)*
`benchmark_mode` (4.1) als verpflichtende default-off Änderung in
`build_provider_payload` gebaut; Consensus-Anonymisierung (4.2) **nicht** Teil von
Phase 2. `benchmark/`-Paket mit Hauptpfad (Consensus mit Modellnamen) angelegt;
disjunkte Pilot-/Final-Samples gezogen und committet (5 / Final = 7×14 = 98);
Tests aus §9 (1–5) grün — Gesamt-Suite 186 passed (Baseline 148 + 38 neu, davon
1 nur mit Parquet-Engine aktiv); Test 6 (anonymize) noch nicht gebaut (Audit-Modus
ist Phase 3). `--dry-run` liefert Kostenprojektion + Tool-Audit grün
(Pilot ≈ \$2.23, Final ≈ \$43.69, Schätzungen). **Kein echter Call ausgeführt.**

**Phase 3 — 5-Fragen-Pilot (separates Sample):** `--pilot --budget <klein>` über
das eigenständige Pilot-Sample; Hauptpfad (Modellnamen) plus **optionaler
anonymisierter Audit-Modus** (E5: gleiche gespeicherte Antworten, einmal Namen /
einmal `Response A–F`, ohne erneuten Kandidaten-Call) sowie alle drei Audits aus E4.
Roh-Antworten, Usage, Kosten, Parsing-Trefferquote prüfen; ggf.
Parser/Prompt/Settings nachschärfen. **Danach zwei Entscheidungen:** (a) finaler
Label-Modus (E5/§10), (b) Einfrieren der Run-Config (E6) — danach keine Änderungen
mehr an Modell-IDs, Reasoning, Temperatur, Output-Token-Limit oder Consensus-Modell.

**Phase 4 — finaler 98-Fragen-Snapshot:** `--limit 98 --budget <cap> --resume`
über das finale (disjunkte) Sample auf eingefrorener Config und im **nach dem Pilot
festgelegten Label-Modus**; vollständige
`calls.jsonl` + `manifest.json`; Aggregation `results.json` (Einzelmodelle, Majority
inkl. `no_majority`, Consensus, optional Synth-allein; gesamt +
Uneinigkeits-Teilmenge). **Kein Differences-Run** (E7).

**Phase 5 — statische Website-Auswertung:** aus `results.json` eine neue statische
Unterseite rendern (eigenes Template, analog `templates/ai-model-comparison.html`);
CSP/`?v=`-Cache-Busting beachten; Methodik + MMLU‑Pro-Attribution + „closed-book,
Websuche deaktiviert" transparent ausweisen.
