"""Isoliertes Benchmark-Paket (MMLU-Pro Consensus-Snapshot).

Importiert ausschliesslich aus ``app.services.llm.*`` und ``app.core.config`` –
niemals aus ``app.api.*``. Macht selbst keine API-Calls beim Import.

Siehe ``docs/benchmark-plan.md`` fuer den verbindlichen Plan (Phase 2).
"""
