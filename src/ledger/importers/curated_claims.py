"""Hand-written findings from the written-up syntheses -> ledger claims. Ported from the map visualiser's
explainer/gen/map/sources/curated.py + its findings/*.jsonl.

Every line of every findings/*.jsonl becomes one claim: claim_id = its 'id' (or a stable hash of the line), kind =
'prediction' for pre-registered tests (preregistered.jsonl or kind 'pre-registered test'), else 'finding'; scope = the
'unit' field ('layers:28-35', 'model', 'L34H28', 'concept:<slug>'); unit = it when it is a neuron / head id. The map's
own finding 'kind' (layer band, head role, fact job, ...) goes in the claims 'category' column.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ledger.core import PackWriter, parse_unit, read_jsonl  # noqa: E402

FINDINGS = Path(os.environ.get("MAP_FINDINGS", "C:/dev/already-clean-map/explainer/gen/map/findings"))
EXP = "curated_syntheses"
KNOWN = {"id", "unit", "kind", "title", "detail", "strength", "status", "metric", "docs", "concept", "source"}


def main(force=False):
    files = sorted(FINDINGS.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no findings/*.jsonl under {FINDINGS}")
    with PackWriter("curated_claims", inputs=files, code=[__file__], force=force) as pw:
        if pw.skip:
            print("curated_claims: unchanged, skipped"); return
        pw.experiment(EXP, title="Written-up syntheses (LAYERS_AND_JOBS, PREREG_MAP, attention study)",
                      method=("Claims transcribed from the project's own write-ups, each with the strength the write-up gives it: "
                              "'strong' needs several independent methods agreeing, 'moderate' fewer or narrower evidence; "
                              "pre-registered tests were frozen before their data existed. Each finding links the section it comes from."),
                      status="as marked on each claim", date="2026-10-18", script=None,
                      reproduce="See the linked document section; each one names the script and results file behind it.",
                      data_files=[f.as_posix() for f in files], caveats=["Layer bands are bins over a gradient, not hard boundaries."],
                      docs=["LAYERS_AND_JOBS.md", "PREREG_MAP.md", "RESEARCH_STATE.md"],
                      params={"source": "explainer/gen/map/findings/*.jsonl (map visualiser worktree)"},
                      description="Curated claims (findings and pre-registered predictions) from the syntheses; no evidence rows.",
                      code_commit=None)
        seen = set()
        for f in files:
            for i, r in enumerate(read_jsonl(f)):
                scope = r.get("unit") or "model"
                cid = r.get("id") or "c_" + hashlib.sha256(json.dumps(r, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]
                if cid in seen:
                    raise ValueError(f"duplicate claim id {cid} ({f.name}:{i + 1})")
                seen.add(cid)
                try:
                    unit = parse_unit(scope)["unit_key"]
                except ValueError:
                    unit = None
                concept = None
                if scope.startswith("concept:"):
                    concept = "thing:" + scope.split(":", 1)[1]
                elif r.get("concept"):
                    concept = "thing:" + r["concept"]
                prediction = f.stem == "preregistered" or r.get("kind") == "pre-registered test"
                metric = dict(r.get("metric") or {})
                extra = {k: v for k, v in r.items() if k not in KNOWN}
                if extra:
                    metric["extra_fields"] = extra
                pw.claim(cid, kind="prediction" if prediction else "finding", text=r.get("title"), scope=scope, unit=unit,
                         concept=concept, detail=r.get("detail"), grade=r.get("strength"), status=r.get("status", "exploratory"),
                         metric=metric or None, docs=r.get("docs") or (), experiment=EXP, category=r.get("kind"))


if __name__ == "__main__":
    main(force="--force" in sys.argv)
