"""Export the ledger for the map visualiser (interface agreed 18 Oct; LEDGER.md "Interface").

  modelmapper ledger export        -> <LEDGER_DIR>/export/ (written to export.tmp, then swapped in atomically)
                                     (was: python src/ledger/export.py)

Files: manifest.json, experiments.jsonl, concepts.jsonl, units.jsonl, evidence/<experiment_id>/part-NNNN.jsonl.gz, claims.jsonl,
artifacts.jsonl, views/{unit_summary,concept_summary,layer_summary,head_summary}.jsonl, scripts.jsonl.
The ledger's experiment rows name their scripts by the paths used before the 18 Oct 2026 package refactor (`script`,
`reproduce`); the packs are left as they are, so their ids stay valid. The export adds, per experiment, `script_files`
(the new file of each script named), `command` (the `modelmapper` command for `script`) and `reproduce_command`
(`reproduce` with each old `python src/...` call replaced by its command), and scripts.jsonl holds the whole
old path -> module -> command table (SCRIPTS.md).
"""
import gzip
import hashlib
import json
import math
import re
import shutil
import sys
import time
from pathlib import Path

import duckdb
import yaml

from modelmapper.cli import new_command, new_files, rewrite_commands, scripts_rows
from modelmapper.ledger.core import LEDGER_DIR, SCHEMA_VERSION  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = LEDGER_DIR / "export"
PART = 100_000


class _Fmt(dict):
    def __missing__(self, k):
        return "?"


def _f(x):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return "n/a" if math.isnan(x) else f"{x:.2f}"
    return str(x)


def render(tpl, row, names):
    if not tpl:
        return None
    d = _Fmt({k: _f(v) for k, v in row.items()})
    d["unit"] = row.get("unit_key") or "?"; d["unit2"] = row.get("unit2_key") or "?"
    d["concept"] = names.get(row.get("concept_id"), row.get("concept_id") or "")
    d["concept2"] = names.get(row.get("concept2_id"), row.get("concept2_id") or "")
    d["context"] = row.get("context") or ""
    try:
        s = tpl.format_map(d)
    except Exception:  # noqa: BLE001
        return None
    s = re.sub(r"\(\s*;\s*", "(", s); s = re.sub(r";\s*\)", ")", s); s = re.sub(r"\s*\(\s*\)", "", s)   # empty fields leave no '()' / '(; x)'
    return re.sub(r"\s{2,}", " ", s).strip()


def clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return str(v)
    return v


def jdump(f, obj):
    f.write(json.dumps({k: clean(v) for k, v in obj.items()}, ensure_ascii=False) + "\n")


def export():
    t0 = time.time()
    V = yaml.safe_load((HERE / "vocab.yaml").read_text(encoding="utf-8"))
    tpl = {k: v.get("claim") for k, v in V["evidence_types"].items()}
    con = duckdb.connect(str(LEDGER_DIR / "ledger.duckdb"), read_only=True)
    meta = dict(zip(["schema_version", "salience_version", "built_utc", "packs", "superseded"], con.execute("SELECT * FROM ledger_meta").fetchone()))
    tmp = LEDGER_DIR / "export.tmp"
    shutil.rmtree(tmp, ignore_errors=True); (tmp / "evidence").mkdir(parents=True); (tmp / "views").mkdir()
    counts = {}

    def table(name, sql, transform=lambda r: r):
        cur = con.execute(sql); cols = [c[0] for c in cur.description]; n = 0
        with open(tmp / name, "w", encoding="utf-8") as f:
            while batch := cur.fetchmany(50_000):
                for r in batch:
                    jdump(f, transform(dict(zip(cols, r)))); n += 1
        counts[name] = n

    J = lambda s: json.loads(s) if isinstance(s, str) and s[:1] in "[{" else s
    status = dict(con.execute("SELECT experiment_id, status FROM experiments").fetchall())
    table("experiments.jsonl", "SELECT * FROM experiments ORDER BY date, experiment_id",
          lambda r: {**r, **{k: J(r[k]) for k in ("data_files", "caveats", "docs", "params")},
                     "script_files": new_files(r["script"]),
                     "command": rewrite_commands(r["script"]) if r["script"] and " " in r["script"].strip() else new_command(r["script"]),
                     "reproduce_command": rewrite_commands(r["reproduce"])})
    names = dict(con.execute("SELECT concept_id, label FROM concepts").fetchall())
    table("concepts.jsonl", "SELECT * FROM concepts ORDER BY concept_id",
          lambda r: {"concept_id": r["concept_id"], "name": r["label"], "slug": r["slug"], "qid": r["qid"], "kind": r["kind"],
                     "aliases": J(r["aliases"]), "attributes": J(r["attributes"]), "packs": J(r["packs"])})
    table("units.jsonl", "SELECT * FROM units ORDER BY unit_kind, layer, unit_key",
          lambda r: {"model": r["model_id"], "unit": r["unit_key"], "unit_id": r["unit_id"], "kind": r["unit_kind"], "layer": r["layer"],
                     "index": int(r["unit_key"].split(":")[1]) if ":" in r["unit_key"] else int(r["unit_key"].split("H")[1])})
    slugs = dict(con.execute("SELECT concept_id, slug FROM concepts").fetchall()); qids = dict(con.execute("SELECT concept_id, qid FROM concepts").fetchall())

    def ev(r):
        strength = r["value"] / r["noise_floor"] if r["value"] is not None and r["noise_floor"] and r["noise_floor"] > 0 else None
        return {"evidence_id": r["evidence_id"], "model": r["model_id"], "unit": r["unit_key"], "unit_id": r["unit_id"], "kind": r["unit_kind"],
                "layer": r["layer"], "unit2": r["unit2_key"], "concept": names.get(r["concept_id"]), "concept_id": r["concept_id"],
                "slug": slugs.get(r["concept_id"]), "qid": qids.get(r["concept_id"]), "concept2": names.get(r["concept2_id"]),
                "concept2_id": r["concept2_id"], "experiment_id": r["experiment_id"], "evidence_type": r["evidence_type"], "measure": r["metric"],
                "value": r["value"], "value_text": r["value_text"], "uom": r["uom"], "n": r["n"], "noise_floor": r["noise_floor"],
                "threshold": r["threshold"], "passes_threshold": r["passes"], "rank": r["rank"],
                "claim": render(tpl.get(r["evidence_type"]), r, names), "status": status.get(r["experiment_id"]), "strength": strength,
                "context": r["context"], "details": J(r["details"]), "source": r["source"], "pack_id": r["pack_id"]}
    for (exp,) in con.execute("SELECT DISTINCT experiment_id FROM evidence ORDER BY 1").fetchall():
        # evidence/<experiment>/part-NNNN.jsonl.gz, <= PART rows each, ordered by concept then unit (scales to any size)
        cur = con.execute(f"SELECT * FROM evidence WHERE experiment_id = '{exp}' ORDER BY concept_id, unit_id, evidence_type, metric")
        cols = [c[0] for c in cur.description]; part, n, f = 0, 0, None
        (tmp / "evidence" / exp).mkdir()
        while batch := cur.fetchmany(20_000):
            for r in batch:
                if f is None or n % PART == 0:
                    if f:
                        f.close()
                    part += 1; name = f"evidence/{exp}/part-{part:04d}.jsonl.gz"; counts[name] = 0
                    f = gzip.open(tmp / name, "wt", encoding="utf-8", compresslevel=5)
                jdump(f, ev(dict(zip(cols, r)))); n += 1; counts[name] += 1
        if f:
            f.close()
    table("claims.jsonl", "SELECT * FROM claims ORDER BY claim_id", lambda r: {**r, **{k: J(r[k]) for k in ("metric", "evidence_ids", "docs")}})
    table("artifacts.jsonl", "SELECT * FROM artifacts ORDER BY path", lambda r: {**r, "keys": J(r["keys"])})
    with open(tmp / "scripts.jsonl", "w", encoding="utf-8") as f:
        rows = scripts_rows()
        for old, new, kind, cmd in rows:
            jdump(f, {"old_path": old, "new_file": new, "kind": kind, "command": cmd})
        counts["scripts.jsonl"] = len(rows)
    for v in ("unit_summary", "concept_summary", "layer_summary", "head_summary"):
        table(f"views/{v}.jsonl", f"SELECT * FROM {v}", lambda r: {k: (dict(x) if hasattr(x, "items") else x) for k, x in r.items()})
    files = {}
    for p in sorted(list(tmp.rglob("*.jsonl")) + list(tmp.rglob("*.jsonl.gz"))):
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        files[p.relative_to(tmp).as_posix()] = {"rows": counts[p.relative_to(tmp).as_posix()], "bytes": p.stat().st_size, "sha256": h}
    (tmp / "manifest.json").write_text(json.dumps({"exported_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "schema_version": SCHEMA_VERSION,
                                                   **meta, "packs": json.loads(meta["packs"]), "superseded": json.loads(meta["superseded"]),
                                                   "files": files}, indent=1), encoding="utf-8")
    con.close()
    old = LEDGER_DIR / "export.old"
    shutil.rmtree(old, ignore_errors=True)
    if OUT.exists():
        OUT.rename(old)
    tmp.rename(OUT); shutil.rmtree(old, ignore_errors=True)
    print(f"exported to {OUT} in {time.time() - t0:.0f}s: " + ", ".join(f"{k} {v:,}" for k, v in counts.items() if not k.startswith("evidence/")) +
          f"; evidence files {sum(1 for k in counts if k.startswith('evidence/'))} ({sum(v for k, v in counts.items() if k.startswith('evidence/')):,} rows)")


if __name__ == "__main__":
    export()
