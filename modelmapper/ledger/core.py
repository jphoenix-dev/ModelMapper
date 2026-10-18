"""Ledger core: identifiers, table schemas, and PackWriter (the one way data enters the ledger). Spec: LEDGER.md.

A pack is an immutable folder packs/<pack_id>/ with pack.json + Parquet tables. Pack ids are deterministic
(<importer>__<12 hex of inputs + code>), so re-running an importer on unchanged inputs is a no-op.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import time
import unicodedata
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION = "ledger-1.0"
LEDGER_DIR = Path(os.environ.get("LEDGER_DIR", "D:/already-clean-data/ledger"))
DATA_ROOT = Path(os.environ.get("AC_DATA_DIR", "D:/already-clean-data"))          # research outputs
REPO_RESULTS = Path(os.environ.get("AC_REPO_RESULTS", "C:/dev/already-clean/results"))  # earlier results (main checkout)
REPO = Path(__file__).resolve().parents[2]
PACKS = LEDGER_DIR / "packs"
MODELS = {"qwen3-8b": {"hf_name": "Qwen/Qwen3-8B", "n_layers": 36, "d_mlp": 12288, "n_heads": 32}}
KIND_LETTER = {"neuron": "N", "head": "H"}
LETTER_KIND = {v: k for k, v in KIND_LETTER.items()}

# ---------------------------------------------------------------- identifiers
_SHORT_N = re.compile(r"^L(\d+)[:_]N?(\d+)$")       # 'L31:8930', 'L0_N492'
_SHORT_H = re.compile(r"^L(\d+)H(\d+)$")            # 'L23H10'
_FULL = re.compile(r"^([^/]+)/L(\d+)/([A-Z])(\d+)$")  # 'qwen3-8b/L31/N8930'


def parse_unit(u, model="qwen3-8b"):
    """Any unit spelling -> dict(model, kind, layer, index, unit_id, unit_key). Accepts 'L31:8930', 'L23H10',
    'qwen3-8b/L31/N8930', or a tuple (layer, index[, kind])."""
    if isinstance(u, (tuple, list)):
        layer, index = int(u[0]), int(u[1]); kind = u[2] if len(u) > 2 else "neuron"
    else:
        s = str(u).strip()
        if m := _FULL.match(s):
            model, layer, kind, index = m.group(1), int(m.group(2)), LETTER_KIND[m.group(3)], int(m.group(4))
        elif m := _SHORT_N.match(s):
            layer, index, kind = int(m.group(1)), int(m.group(2)), "neuron"
        elif m := _SHORT_H.match(s):
            layer, index, kind = int(m.group(1)), int(m.group(2)), "head"
        else:
            raise ValueError(f"not a unit id: {u!r}")
    spec = MODELS.get(model)
    if spec:
        if not 0 <= layer < spec["n_layers"]:
            raise ValueError(f"layer out of range: {u!r}")
        if not 0 <= index < (spec["d_mlp"] if kind == "neuron" else spec["n_heads"]):
            raise ValueError(f"index out of range: {u!r}")
    key = f"L{layer}:{index}" if kind == "neuron" else f"L{layer}H{index}"
    return {"model": model, "kind": kind, "layer": layer, "index": index,
            "unit_id": f"{model}/L{layer}/{KIND_LETTER[kind]}{index}", "unit_key": key}


def map_slug(name):
    """The map visualiser's slug (explainer/gen/map/common.py) - emitted verbatim as the join key."""
    s = re.sub(r"[^a-z0-9]+", "-", str(name).lower().replace("&", " and ")).strip("-")
    return s[:80] or "concept"


def ascii_slug(name):
    return map_slug(unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode())


_ARTICLE = re.compile(r"^(the|a|an)\s+", re.I)


def concept_id(kind, label):
    """things drop a leading article ('the elephant' == 'elephant'; the label keeps it), matching the map's own rule."""
    lab = _ARTICLE.sub("", str(label).strip()) if kind == "thing" else label
    return f"{kind}:{ascii_slug(lab)}"


def status_class(status):
    s = (status or "").lower()
    return "invalid" if "invalid" in s else "pre-registered" if ("pre-registered" in s or "preregistered" in s) else "exploratory"


def ordinal(c):
    return f"{c}{'th' if 10 <= c % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(c % 10, 'th')}"


# Shared concept conventions - every importer uses these so the same thing gets the same id:
#   things (entities, common nouns, articles):  thing:<ascii slug of the article / entity name>   e.g. thing:eiffel-tower
#   centuries:  era:century-<n> (label '19th century'); years written NNxx: era:c<NN>xx (label '18xx')
#   relations / question types: relation:<slug>  (relation:century, relation:year-completed, relation:height)
#   digit values: digit:<d>;  other shared answer values: value:<slug>
def thing(pw, label, **attrs):
    return pw.concept(concept_id("thing", label), label, "thing", attributes=attrs)


def era_century(pw, c):
    return pw.concept(f"era:century-{c}", f"{ordinal(c)} century", "era", aliases=[f"{ordinal(c)}", f"{(c - 1)}xx" if c > 1 else None],
                      attributes={"start_year": (c - 1) * 100 + 1, "end_year": c * 100})


def relation(pw, name):
    return pw.concept(concept_id("relation", name), name, "relation")


# ---------------------------------------------------------------- schemas
S, F, I, B = pa.string(), pa.float64(), pa.int64(), pa.bool_()
EVIDENCE = pa.schema([
    ("evidence_id", S), ("pack_id", S), ("experiment_id", S), ("model_id", S),
    ("unit_id", S), ("unit_key", S), ("unit_kind", S), ("layer", I),
    ("unit2_id", S), ("unit2_key", S),
    ("concept_id", S), ("concept2_id", S), ("context", S),
    ("evidence_type", S), ("metric", S), ("value", F), ("value_text", S), ("uom", S), ("n", I),
    ("noise_floor", F), ("threshold", F), ("passes", B), ("rank", I),
    ("details", S), ("source", S)])
CONCEPTS = pa.schema([("concept_id", S), ("label", S), ("kind", S), ("slug", S), ("qid", S), ("aliases", S),
                      ("attributes", S), ("pack_id", S)])
EXPERIMENTS = pa.schema([("experiment_id", S), ("pack_id", S), ("title", S), ("method", S), ("status", S), ("date", S),
                         ("script", S), ("reproduce", S), ("code_commit", S), ("data_files", S), ("caveats", S),
                         ("docs", S), ("params", S), ("description", S), ("status_class", S)])
ARTIFACTS = pa.schema([("artifact_id", S), ("pack_id", S), ("experiment_id", S), ("path", S), ("bytes", I), ("sha256", S),
                       ("format", S), ("keys", S), ("concept_id", S), ("unit_id", S), ("description", S)])
CLAIMS = pa.schema([("claim_id", S), ("pack_id", S), ("kind", S), ("scope", S), ("unit_id", S), ("unit_key", S),
                    ("concept_id", S), ("text", S), ("detail", S), ("grade", S), ("status", S), ("metric", S), ("category", S),
                    ("evidence_sql", S), ("evidence_ids", S), ("docs", S), ("experiment_id", S)])
TABLES = {"evidence": EVIDENCE, "concepts": CONCEPTS, "experiments": EXPERIMENTS, "artifacts": ARTIFACTS, "claims": CLAIMS}


def _j(x):
    if x is None:
        return None
    return json.dumps(x, ensure_ascii=False, default=_jdefault, allow_nan=True)


def _jdefault(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def _num(x):
    if x is None:
        return None
    x = float(x)
    return x                       # NaN / inf are kept (Parquet stores them); never silently dropped


def _sha(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def git_commit():
    import subprocess
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# ---------------------------------------------------------------- pack writer
class PackWriter:
    """with PackWriter('era', inputs=[...], code=[__file__]) as pw: pw.experiment(...); pw.evidence(...)

    inputs: source files (fingerprinted by size + mtime; sha256 recorded for files under SHA_LIMIT bytes).
    code: importer source files (sha256). Re-running with identical inputs + code is a no-op (pw.skip is True)."""
    SHA_LIMIT = 256 << 20
    FLUSH = 100_000

    def __init__(self, importer, inputs, code, supersedes=(), force=False, model="qwen3-8b", contributor="already-clean research (James + Claude)"):
        self.importer, self.model, self.contributor = importer, model, contributor
        inputs = [Path(p) for p in inputs]
        missing = [str(p) for p in inputs if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{importer}: missing inputs {missing}")
        h = hashlib.sha256(f"{SCHEMA_VERSION}|{importer}".encode())
        self.inputs = []
        for p in sorted(inputs, key=str):
            st = p.stat()
            h.update(f"{p.as_posix()}|{st.st_size}|{int(st.st_mtime)}".encode())
            self.inputs.append({"path": _rel(p), "bytes": st.st_size, "mtime": int(st.st_mtime),
                                "sha256": _sha(p) if st.st_size <= self.SHA_LIMIT else None})
        self.code = []
        for c in list(code) + [__file__]:
            c = Path(c); h.update(c.read_bytes())
            self.code.append({"path": _rel(c), "sha256": _sha(c)})
        self.pack_id = f"{importer}__{h.hexdigest()[:12]}"
        self.dir = PACKS / self.pack_id
        self.skip = (self.dir / "pack.json").exists() and not force
        if self.skip:                       # re-activating an existing pack makes it its importer's current pack again
            (self.dir / "activated_utc").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        # a new import of the same importer supersedes its earlier packs (kept on disk, ignored by the build)
        earlier = [p.parent.name for p in PACKS.glob(f"{importer}__*/pack.json") if p.parent.name != self.pack_id]
        self.supersedes = sorted(set(supersedes) | set(earlier))
        self._tmp = PACKS / f".tmp_{self.pack_id}_{os.getpid()}"
        self._rows = {t: [] for t in TABLES}; self._writers = {}; self._counts = {t: 0 for t in TABLES}
        self._concepts = {}; self._ids = set(); self._experiments = set()

    # -- context
    def __enter__(self):
        if not self.skip:
            if self._tmp.exists():
                shutil.rmtree(self._tmp)
            self._tmp.mkdir(parents=True)
        return self

    def __exit__(self, et, ev, tb):
        if self.skip:
            return False
        if et is not None:
            for w in self._writers.values():
                w.close()
            shutil.rmtree(self._tmp, ignore_errors=True)
            return False
        self._finish()
        return False

    # -- rows
    def experiment(self, experiment_id, *, title, method, status, date, script=None, reproduce=None, data_files=(),
                   caveats=(), docs=(), params=None, description=None, code_commit=None, status_class=None):
        self._experiments.add(experiment_id)
        self._add("experiments", {"experiment_id": experiment_id, "pack_id": self.pack_id, "title": title, "method": method,
                                  "status": status, "date": date, "script": script, "reproduce": reproduce,
                                  "code_commit": code_commit or git_commit(), "data_files": _j(list(data_files)),
                                  "caveats": _j(list(caveats)), "docs": _j(list(docs)), "params": _j(params), "description": description,
                                  "status_class": status_class or globals()["status_class"](status)})

    def concept(self, cid, label, kind, *, aliases=(), qid=None, attributes=None):
        """Register a concept (merged by id within the pack: aliases unioned, attributes updated)."""
        c = self._concepts.get(cid)
        if c is None:
            c = self._concepts[cid] = {"concept_id": cid, "label": label, "kind": kind, "slug": map_slug(label), "qid": qid,
                                       "aliases": set(), "attributes": {}}
        c["aliases"].update(a for a in aliases if a and a != c["label"])
        c["qid"] = c["qid"] or qid
        c["attributes"].update({k: v for k, v in (attributes or {}).items() if v is not None})
        return cid

    def evidence(self, *, experiment, evidence_type, metric, unit=None, unit2=None, concept=None, concept2=None,
                 context=None, value=None, value_text=None, uom=None, n=None, noise_floor=None, threshold=None,
                 passes=None, rank=None, details=None, source=None, key=None):
        if experiment not in self._experiments:
            raise ValueError(f"declare experiment {experiment!r} before its evidence")
        u = parse_unit(unit, self.model) if unit is not None else None
        u2 = parse_unit(unit2, self.model) if unit2 is not None else None
        rk = key or "|".join(str(x) for x in (experiment, u and u["unit_id"], u2 and u2["unit_id"], concept, concept2, context,
                                              evidence_type, metric, source))   # source (file#record) keeps rows of merged aliases apart
        eid = hashlib.sha256(f"{self.pack_id}|{rk}".encode()).hexdigest()[:16]
        if eid in self._ids:
            raise ValueError(f"duplicate evidence key in {self.pack_id}: {rk} (pass key= to disambiguate)")
        self._ids.add(eid)
        self._add("evidence", {
            "evidence_id": eid, "pack_id": self.pack_id, "experiment_id": experiment, "model_id": self.model,
            "unit_id": u and u["unit_id"], "unit_key": u and u["unit_key"], "unit_kind": u and u["kind"], "layer": u and u["layer"],
            "unit2_id": u2 and u2["unit_id"], "unit2_key": u2 and u2["unit_key"],
            "concept_id": concept, "concept2_id": concept2, "context": context, "evidence_type": evidence_type, "metric": metric,
            "value": _num(value), "value_text": None if value_text is None else str(value_text), "uom": uom,
            "n": None if n is None else int(n), "noise_floor": _num(noise_floor), "threshold": _num(threshold),
            "passes": None if passes is None else bool(passes), "rank": None if rank is None else int(rank),
            "details": _j(details), "source": source})
        return eid

    def artifact(self, path, *, experiment=None, fmt=None, keys=None, concept=None, unit=None, description=None, sha=False):
        p = Path(path); st = p.stat()
        aid = hashlib.sha256(f"{self.pack_id}|{_rel(p)}".encode()).hexdigest()[:16]
        self._add("artifacts", {"artifact_id": aid, "pack_id": self.pack_id, "experiment_id": experiment, "path": _rel(p),
                                "bytes": st.st_size, "sha256": _sha(p) if sha else None, "format": fmt or p.suffix.lstrip("."),
                                "keys": _j(keys), "concept_id": concept, "unit_id": unit and parse_unit(unit, self.model)["unit_id"],
                                "description": description})

    def claim(self, claim_id, *, kind, text, scope, grade=None, status=None, detail=None, unit=None, concept=None,
              metric=None, evidence_sql=None, evidence_ids=None, docs=(), experiment=None, category=None):
        u = parse_unit(unit, self.model) if unit is not None else None
        self._add("claims", {"claim_id": claim_id, "pack_id": self.pack_id, "kind": kind, "scope": scope,
                             "unit_id": u and u["unit_id"], "unit_key": u and u["unit_key"], "concept_id": concept, "text": text,
                             "detail": detail, "grade": grade, "status": status, "metric": _j(metric), "category": category, "evidence_sql": evidence_sql,
                             "evidence_ids": _j(evidence_ids), "docs": _j(list(docs)), "experiment_id": experiment})

    # -- internals
    def _add(self, table, row):
        self._rows[table].append(row); self._counts[table] += 1
        if len(self._rows[table]) >= self.FLUSH:
            self._flush(table)

    def _flush(self, table):
        rows = self._rows[table]
        if not rows:
            return
        t = pa.Table.from_pylist(rows, schema=TABLES[table])
        if table not in self._writers:
            self._writers[table] = pq.ParquetWriter(self._tmp / f"{table}.parquet", TABLES[table], compression="zstd")
        self._writers[table].write_table(t); self._rows[table] = []

    def _finish(self):
        for c in self._concepts.values():
            self._add("concepts", {**c, "aliases": _j(sorted(c["aliases"])), "attributes": _j(c["attributes"]), "pack_id": self.pack_id})
        for t in TABLES:
            self._flush(t)
        for w in self._writers.values():
            w.close()
        meta = {"pack_id": self.pack_id, "importer": self.importer, "schema_version": SCHEMA_VERSION, "model": self.model,
                "contributor": self.contributor, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "code_commit": git_commit(), "inputs": self.inputs, "code": self.code, "supersedes": self.supersedes,
                "experiments": sorted(self._experiments),
                "tables": {t: {"rows": self._counts[t], "sha256": _sha(self._tmp / f"{t}.parquet")} for t in TABLES if self._counts[t]}}
        (self._tmp / "pack.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        if self.dir.exists():
            shutil.rmtree(self.dir)
        self._tmp.rename(self.dir)
        print(f"pack {self.pack_id}: " + ", ".join(f"{t} {n:,}" for t, n in self._counts.items() if n))


def _rel(p):
    p = Path(p).resolve()
    for root, tag in ((DATA_ROOT.resolve(), "data"), (REPO_RESULTS.resolve(), "repo_results"), (REPO.resolve(), "repo")):
        try:
            return f"{tag}:{p.relative_to(root).as_posix()}"
        except ValueError:
            pass
    return p.as_posix()


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)
