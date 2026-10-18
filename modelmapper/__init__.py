"""Model Mapper: map what each MLP neuron and attention head of a language model does (README.md).

Layout (SCRIPTS.md maps every old src/ path to its module here):
  core/         model loading, neuron reads and steering, the labeller and judge clients, the main pipelines, settings
  loop/         the mapping loop: prepare, map_core, enrich (v3), frame, report, and the run checks
  experiments/  one module per study; experiments/registry.py lists them and their stages
  ledger/       the evidence ledger (LEDGER.md)
  dist/         the rented-GPU runner
  archive/      the 16-17 Oct one-off analyses, unchanged, kept for the record
The `modelmapper` command (modelmapper/cli.py) runs all of them. Importing this package imports nothing heavy.
"""
__version__ = "2026.10.18"
