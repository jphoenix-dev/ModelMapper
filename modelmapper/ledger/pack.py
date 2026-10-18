"""Third-party contributions: turn a folder of JSON lines into a validated ledger pack (LEDGER.md "Contributing").

  python src/ledger/pack.py from-jsonl <dir> [--force]
  python src/ledger/pack.py validate <dir>

<dir> contains:
  pack.json          {"importer": "<short name, unique to you>", "contributor": "...", "model": "qwen3-8b",
                      "inputs": ["<paths of the raw files this came from, optional>"], "supersedes": []}
  experiments.jsonl  one object per experiment: experiment_id, title, method, status, date, script, reproduce, data_files,
                     caveats, docs, params, description  (experiment_id must be new to the ledger, e.g. prefixed with your name)
  concepts.jsonl     (optional) concept_id, label, kind, aliases, qid, attributes   - see the id conventions in core.py
  evidence.jsonl     (optional for a claims-only pack) one object per measured fact: experiment, evidence_type, metric, and any of unit, unit2, concept, concept2,
                     context, value, value_text, uom, n, noise_floor, threshold, passes, rank, details, source, key
  claims.jsonl       (optional) claim_id, kind, text, scope, grade, status, detail, unit, concept, metric, docs
Units may be written 'L31:8930', 'L23H10' or 'qwen3-8b/L31/N8930'. Unknown evidence types / concept kinds are accepted but
reported: add them to vocab.yaml (with a definition and a claim template) in the same contribution.
"""
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ledger.core import PackWriter, read_jsonl  # noqa: E402

HERE = Path(__file__).resolve().parent
EV_KEYS = {"experiment", "evidence_type", "metric", "unit", "unit2", "concept", "concept2", "context", "value", "value_text", "uom", "n",
           "noise_floor", "threshold", "passes", "rank", "details", "source", "key"}


def validate(d):
    d = Path(d); V = yaml.safe_load((HERE / "vocab.yaml").read_text(encoding="utf-8")); problems, warnings = [], []
    meta = json.loads((d / "pack.json").read_text(encoding="utf-8"))
    if not meta.get("importer"):
        problems.append("pack.json: 'importer' is required")
    exps = {e["experiment_id"] for e in read_jsonl(d / "experiments.jsonl")} if (d / "experiments.jsonl").exists() else set()
    if not exps:
        problems.append("experiments.jsonl: at least one experiment is required")
    for i, e in enumerate(read_jsonl(d / "evidence.jsonl") if (d / "evidence.jsonl").exists() else []):
        if extra := set(e) - EV_KEYS:
            problems.append(f"evidence.jsonl:{i + 1}: unknown fields {sorted(extra)} (put extra data in 'details')")
        for k in ("experiment", "evidence_type", "metric"):
            if not e.get(k):
                problems.append(f"evidence.jsonl:{i + 1}: '{k}' is required")
        if e.get("experiment") not in exps:
            problems.append(f"evidence.jsonl:{i + 1}: experiment {e.get('experiment')!r} not in experiments.jsonl")
        if e.get("evidence_type") and e["evidence_type"] not in V["evidence_types"]:
            warnings.append(f"new evidence_type {e['evidence_type']!r} - add it to vocab.yaml")
        if len(problems) > 50:
            break
    if (d / "concepts.jsonl").exists():
        for c in read_jsonl(d / "concepts.jsonl"):
            if c.get("kind") not in V["concept_kinds"]:
                warnings.append(f"new concept kind {c.get('kind')!r} - add it to vocab.yaml")
    return problems, sorted(set(warnings))


def from_jsonl(d, force=False):
    d = Path(d); problems, warnings = validate(d)
    for w in warnings:
        print("warning:", w)
    if problems:
        print("\n".join(problems)); sys.exit(1)
    meta = json.loads((d / "pack.json").read_text(encoding="utf-8"))
    files = [p for p in d.glob("*.jsonl")] + [d / "pack.json"] + [Path(p) for p in meta.get("inputs", []) if Path(p).exists()]
    with PackWriter(meta["importer"], inputs=files, code=[__file__], supersedes=meta.get("supersedes", []), force=force,
                    model=meta.get("model", "qwen3-8b"), contributor=meta.get("contributor", "unknown")) as pw:
        if pw.skip:
            print("unchanged, skipped"); return
        for e in read_jsonl(d / "experiments.jsonl"):
            eid = e.pop("experiment_id"); pw.experiment(eid, **{k: e.get(k) for k in ("title", "method", "status", "date")},
                                                         **{k: e[k] for k in ("script", "reproduce", "data_files", "caveats", "docs", "params", "description", "code_commit") if k in e})
        if (d / "concepts.jsonl").exists():
            for c in read_jsonl(d / "concepts.jsonl"):
                pw.concept(c["concept_id"], c["label"], c["kind"], aliases=c.get("aliases", []), qid=c.get("qid"), attributes=c.get("attributes"))
        for e in (read_jsonl(d / "evidence.jsonl") if (d / "evidence.jsonl").exists() else []):
            pw.evidence(**e)
        if (d / "claims.jsonl").exists():
            for c in read_jsonl(d / "claims.jsonl"):
                cid = c.pop("claim_id"); pw.claim(cid, **c)


if __name__ == "__main__":
    cmd, d = sys.argv[1], sys.argv[2]
    if cmd == "validate":
        p, w = validate(d); print("\n".join(p + ["warning: " + x for x in w]) or "ok")
    else:
        from_jsonl(d, force="--force" in sys.argv)
