"""Build ledger.duckdb from all packs (LEDGER.md). Never edits packs; safe to delete ledger.duckdb and rebuild.

  python src/ledger/build.py            build (views over pack Parquet + merged concepts + derived salience tables)
  python src/ledger/build.py --check    also report unknown vocabulary, dangling ids, duplicate evidence ids
"""
import json
import sys
import time
from pathlib import Path

import duckdb
import yaml

from modelmapper.ledger.core import LEDGER_DIR, PACKS, SCHEMA_VERSION, TABLES  # noqa: E402

HERE = Path(__file__).resolve().parent
SALIENCE_VERSION = "salience-1"


def active_packs():
    """Each importer's CURRENT pack = its most recently created or re-activated pack (older ones stay on disk, inactive).
    A pack may also explicitly supersede packs of OTHER importers (pack.json 'supersedes')."""
    packs = []
    for pj in sorted(PACKS.glob("*/pack.json")):
        p = json.loads(pj.read_text(encoding="utf-8"))
        act = pj.parent / "activated_utc"
        p["_when"] = max(p["created_utc"], act.read_text().strip() if act.exists() else "")
        packs.append(p)
    latest = {}
    for p in packs:
        if p["importer"] not in latest or p["_when"] > latest[p["importer"]]["_when"]:
            latest[p["importer"]] = p
    current = list(latest.values())
    cross = {s for p in current for s in p.get("supersedes", []) if not s.startswith(p["importer"] + "__")}
    active = [p for p in current if p["pack_id"] not in cross]
    inactive = sorted(p["pack_id"] for p in packs if p not in active)
    return sorted(active, key=lambda p: p["pack_id"]), inactive


def build(check=False):
    t0 = time.time()
    packs, superseded = active_packs()
    db_path = LEDGER_DIR / "ledger.duckdb"; tmp = LEDGER_DIR / "ledger.duckdb.tmp"
    for p in (tmp, Path(str(tmp) + ".wal")):
        p.unlink(missing_ok=True)
    con = duckdb.connect(str(tmp))
    for t, schema in TABLES.items():
        files = [str(PACKS / p["pack_id"] / f"{t}.parquet").replace("\\", "/") for p in packs if t in p.get("tables", {})]
        if files:
            con.execute(f"CREATE VIEW {t}_raw AS SELECT * FROM read_parquet({files!r}, union_by_name=true)")
        else:
            cols = ", ".join(f"{f.name} {'DOUBLE' if str(f.type) == 'double' else 'BIGINT' if str(f.type) == 'int64' else 'BOOLEAN' if str(f.type) == 'bool' else 'VARCHAR'}" for f in schema)
            con.execute(f"CREATE TABLE {t}_raw ({cols})")
    con.execute("CREATE VIEW evidence AS SELECT * FROM evidence_raw")
    con.execute("CREATE VIEW experiments AS SELECT * FROM experiments_raw")
    con.execute("CREATE VIEW artifacts AS SELECT * FROM artifacts_raw")
    con.execute("CREATE VIEW claims AS SELECT * FROM claims_raw")
    # concepts merged across packs: first non-null label/kind/qid, aliases unioned, attributes merged (later packs win)
    rows = con.execute("SELECT concept_id, label, kind, slug, qid, aliases, attributes, pack_id FROM concepts_raw ORDER BY pack_id").fetchall()
    merged = {}
    for cid, label, kind, slug, qid, aliases, attrs, pid in rows:
        m = merged.setdefault(cid, {"concept_id": cid, "label": label, "kind": kind, "slug": slug, "qid": qid, "aliases": set(), "attributes": {}, "packs": []})
        m["qid"] = m["qid"] or qid
        if label and m["label"] and m["label"].lower().startswith(("the ", "a ", "an ")) and not label.lower().startswith(("the ", "a ", "an ")):
            m["aliases"].add(m["label"]); m["label"] = label            # prefer the article-free spelling when sources differ
        m["aliases"].update(json.loads(aliases or "[]")); m["attributes"].update(json.loads(attrs or "{}")); m["packs"].append(pid)
    con.execute("CREATE TABLE concepts (concept_id VARCHAR, label VARCHAR, kind VARCHAR, slug VARCHAR, qid VARCHAR, aliases VARCHAR, attributes VARCHAR, packs VARCHAR)")
    if merged:
        con.executemany("INSERT INTO concepts VALUES (?,?,?,?,?,?,?,?)",
                        [(m["concept_id"], m["label"], m["kind"], m["slug"], m["qid"], json.dumps(sorted(m["aliases"]), ensure_ascii=False),
                          json.dumps(m["attributes"], ensure_ascii=False), json.dumps(m["packs"])) for m in merged.values()])
    con.execute("""CREATE TABLE units AS
                   SELECT unit_id, any_value(unit_key) AS unit_key, any_value(model_id) AS model_id, any_value(unit_kind) AS unit_kind,
                          any_value(layer) AS layer FROM evidence WHERE unit_id IS NOT NULL GROUP BY unit_id""")
    con.execute((HERE / "salience.sql").read_text(encoding="utf-8"))
    con.execute("CREATE TABLE ledger_meta AS SELECT ? AS schema_version, ? AS salience_version, ? AS built_utc, ? AS packs, ? AS superseded",
                [SCHEMA_VERSION, SALIENCE_VERSION, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 json.dumps([p["pack_id"] for p in packs]), json.dumps(superseded)])
    report = [f"built from {len(packs)} packs ({len(superseded)} superseded) in {time.time() - t0:.1f}s"]
    for t in ("evidence", "concepts", "units", "experiments", "artifacts", "claims", "unit_concept", "unit_summary", "concept_summary"):
        report.append(f"  {t}: {con.execute(f'SELECT count(*) FROM {t}').fetchone()[0]:,}")
    if check:
        report += checks(con)
    con.close()
    db_path.unlink(missing_ok=True); tmp.rename(db_path)
    print("\n".join(report))


def checks(con):
    V = yaml.safe_load((HERE / "vocab.yaml").read_text(encoding="utf-8"))
    out = ["checks:"]
    for col, key in (("evidence_type", "evidence_types"), ("unit_kind", "unit_kinds")):
        bad = [r[0] for r in con.execute(f"SELECT DISTINCT {col} FROM evidence WHERE {col} IS NOT NULL").fetchall() if r[0] not in V[key]]
        out.append(f"  unknown {col}: {bad or 'none'}")
    bad = [r[0] for r in con.execute("SELECT DISTINCT kind FROM concepts").fetchall() if r[0] not in V["concept_kinds"]]
    out.append(f"  unknown concept kinds: {bad or 'none'}")
    d = con.execute("SELECT count(*) FROM (SELECT concept_id FROM evidence WHERE concept_id IS NOT NULL EXCEPT SELECT concept_id FROM concepts)").fetchone()[0]
    d2 = con.execute("SELECT count(*) FROM (SELECT concept2_id FROM evidence WHERE concept2_id IS NOT NULL EXCEPT SELECT concept_id FROM concepts)").fetchone()[0]
    out.append(f"  evidence concept ids missing from concepts: {d} (concept2: {d2})")
    e = con.execute("SELECT count(*) FROM (SELECT DISTINCT experiment_id FROM evidence EXCEPT SELECT experiment_id FROM experiments)").fetchone()[0]
    out.append(f"  evidence experiment ids missing from experiments: {e}")
    dup = con.execute("SELECT count(*) - count(DISTINCT evidence_id) FROM evidence").fetchone()[0]
    out.append(f"  duplicate evidence ids: {dup}")
    return out


if __name__ == "__main__":
    build(check="--check" in sys.argv)
