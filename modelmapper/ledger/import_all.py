"""Run every importer (each is a no-op when its inputs and code are unchanged), then build, verify and export.

  modelmapper ledger import [--force] [--no-export]      (was: python src/ledger/import_all.py)
"""
import importlib
import sys
import time
from pathlib import Path

IMPORTERS = ["run1", "earlier_maps", "facts", "era", "numeric", "shared_neurons", "forced_readout", "fanout", "specificity",
             "concept_search", "search_verify", "concept_atlas", "labels", "web_firings", "curated_claims", "job_names", "links", "jlens", "loop_v3", "scales", "attrib_graphs"]
CONTRIB = Path(__file__).resolve().parent / "contrib"


def main():
    force = "--force" in sys.argv
    for name in IMPORTERS:
        t = time.time(); print(f"== {name}", flush=True)
        importlib.import_module(f"modelmapper.ledger.importers.{name}").main(force=force)
        print(f"   {time.time() - t:.0f}s", flush=True)
    from modelmapper.ledger import pack
    for d in sorted(p for p in CONTRIB.iterdir() if (p / "pack.json").exists()):
        print(f"== contrib/{d.name}", flush=True); pack.from_jsonl(d, force=force)
    from modelmapper.ledger import build, export, verify
    build.build(check=True)
    try:
        verify.main()
    except SystemExit as e:
        if e.code:
            print("VERIFY FAILED - export skipped"); sys.exit(1)
    if "--no-export" not in sys.argv:
        export.export()


if __name__ == "__main__":
    main()
