"""MECHANISMS.md tests: new neuron job categories on Qwen3-0.6B (CPU) and 8B weights / ledger. Was src/mechanisms/.

Each script imports common first (CPU only, local model copies, outputs under <AC_DATA_DIR>/mechanisms). build_pack.py
turns the outputs into the ledger contribution packs in modelmapper/ledger/contrib/mechanisms*/. The data files
(parallel_sentences.json, claims.jsonl) live here, next to the scripts that read them.
"""
