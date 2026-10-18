"""Set up a distributed run in a store: config, corpus shards, promptset and task manifest.

Corpus shards hold the same documents, in the same order, as build_atlas.py reads them (scan.docs: up to 1,024
tokens per document, documents under 64 tokens skipped), so global doc numbers match data/atlas_<m>_docs.npz.

Usage (smoke test, Qwen3-0.6B, 2 shards of 20k tokens, one ablate range):
  python src/dist/make_run.py --store C:/tmp/run1 --model Qwen/Qwen3-0.6B --shard-tokens 20000 --n-shards 2 \
      --ablate 10:0-63
Writes: config.json, corpus/shard_*.npz, promptsets/<ps>.json, manifest.json. Refuses to overwrite an existing run.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CORPUS, model_path  # noqa: E402
from dist.storage import open_store  # noqa: E402

REVISIONS = {"Qwen/Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218",   # hub main on 2026-10-17
             "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca"}
CTX, MIN_LEN = 1024, 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--revision", help="Hugging Face commit hash (default: the pinned one for known models)")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--shard-tokens", type=int, default=2_000_000)
    ap.add_argument("--n-shards", type=int, default=5)
    ap.add_argument("--k", type=int, default=40, help="top-K firings kept per neuron pole")
    ap.add_argument("--batch-tokens", type=int, default=4096, help="tokens per forward pass in observe tasks")
    ap.add_argument("--thresholds", default="0.1,0.3,1.0", help="count |act| > t per neuron (sparsity)")
    ap.add_argument("--ablate", action="append", default=[], help="L:a-b, e.g. 10:0-63 (repeatable)")
    ap.add_argument("--promptset", default="ps0")
    ap.add_argument("--n-pos", type=int, default=10)
    ap.add_argument("--n-neg", type=int, default=5)
    ap.add_argument("--n-ctrl", type=int, default=10)
    ap.add_argument("--window", type=int, default=128, help="context tokens ending at the firing position")
    ap.add_argument("--batch-rows", type=int, default=64)
    ap.add_argument("--n-null", type=int, default=16, help="firing contexts rerun alone per ablate task (noise floor)")
    ap.add_argument("--kind", default="neuron", choices=["neuron", "map", "links", "jlens", "attrib"],
                    help="neuron: observe/merge/ablate (default). map: loop v2 concept batches from specs")
    ap.add_argument("--specs", help="map: local folder of spec .json files, each with a 'concept' key")
    ap.add_argument("--records", help="links: glob of loop v2 run records.jsonl files (status ok concepts are used)")
    ap.add_argument("--jl-windows", type=int, default=256, help="jlens: total windows of the draw (seed 301)")
    ap.add_argument("--jl-shards", default="", help="jlens: comma list of a-b[:rows], e.g. 0-2:4,0-2:8 or 32-64:4")
    ap.add_argument("--jl-docs", default="C:/dev/already-clean/data/atlas_Qwen3-8B_docs.npz", help="jlens: atlas docs npz")
    ap.add_argument("--ag-dir", default="D:/already-clean-data/attrib_graphs", help="attrib: local attrib_graphs folder")
    ap.add_argument("--ag-step", default="graphs", choices=["graphs", "ablate"], help="attrib: which step")
    ap.add_argument("--ag-rows", type=int, default=32, help="attrib: AG_ROWS (32 on 48 GB cards, 16 if out of memory)")
    ap.add_argument("--limit", type=int, default=0, help="links: only the first N concepts (gate tests)")
    ap.add_argument("--specs-in-store", action="store_true", help="map: specs already uploaded under specs/<batch>/")
    ap.add_argument("--batch-size", type=int, default=20, help="map: concepts per task")
    ap.add_argument("--append", action="store_true", help="map: add new specs to a run (created if missing), run stays open")
    ap.add_argument("--watch", type=int, default=0, help="map --append: repeat every N s until <specs>/_PREPARE_DONE exists")
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE set on every worker (map default RECALL_VARIANT=V3)")
    a = ap.parse_args()

    store = open_store(a.store)
    if store.exists("config.json") and not ((a.kind == "map" and a.append) or a.kind == "attrib"):
        raise SystemExit(f"{store} already holds a run (config.json exists)")
    rev = a.revision or REVISIONS.get(a.model)
    if not rev:
        raise SystemExit("pass --revision (a Hugging Face commit hash)")

    if a.kind == "map":
        return make_map_run(a, store, rev)
    if a.kind == "links":
        return make_links_run(a, store, rev)
    if a.kind == "jlens":
        return make_jlens_run(a, store, rev)
    if a.kind == "attrib":
        return make_attrib_run(a, store, rev)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path(a.model))
    from scan import docs
    gen = docs(tok, CTX, MIN_LEN, corpus=a.corpus)
    shards, doc0 = [], 0
    for s in range(a.n_shards):
        buf, n = [], 0
        while n < a.shard_tokens:
            try:
                x = next(gen)
            except StopIteration:
                break
            buf.append(x); n += len(x)
        if not buf:
            break
        name = f"shard_{s:05d}"
        store.put_npz(f"corpus/{name}.npz", tokens=np.concatenate(buf).astype(np.int32),
                      lengths=np.array([len(x) for x in buf], dtype=np.int32), doc0=np.array(doc0))
        shards.append(name); doc0 += len(buf)
        print(f"{name}: {len(buf)} docs, {n:,} tokens", flush=True)

    cfg = {"model": a.model, "revision": rev, "dtype": a.dtype, "corpus": Path(a.corpus).name, "ctx": CTX,
           "min_len": MIN_LEN, "shards": shards,
           "observe": {"k": a.k, "batch_tokens": a.batch_tokens, "thresholds": [float(x) for x in a.thresholds.split(",")]}}
    store.put_json("config.json", cfg)
    store.put_json(f"promptsets/{a.promptset}.json",
                   {"n_pos": a.n_pos, "n_neg": a.n_neg, "n_ctrl": a.n_ctrl, "window": a.window, "seed": 0,
                    "batch_rows": a.batch_rows, "n_tokens": 20, "modes": [0, 1], "n_null": a.n_null})
    tasks = [{"id": f"observe:{s}"} for s in shards]
    tasks.append({"id": "merge:observe", "after": ["observe:*"]})
    for spec in a.ablate:
        layer, rng = spec.split(":")
        tasks.append({"id": f"ablate:{int(layer)}:{rng}:{a.promptset}", "after": ["merge:observe"]})
    store.put_json("manifest.json", {"tasks": tasks})
    from dist import quota
    ps = store.get_json(f"promptsets/{a.promptset}.json")
    est = sum(quota.task_output_bytes(cfg, t["id"], promptset=ps) for t in tasks)
    quota.save_ops(store, f"make_run-{int(__import__('time').time())}")
    print(f"estimated results if nothing is drained: {est/1e9:.2f} GB (bucket cap {quota.MAX_GB} GB; use launcher --drain-to)")
    print(f"{store}: {len(tasks)} tasks, model {a.model}@{rev[:10]}")


def make_attrib_run(a, store, rev):
    """Attribution graphs: uploads pilot_prompts.json (and for ablate every <slug>/graph.json + groups.json); one task."""
    src = Path(a.ag_dir)
    files = [src / "pilot_prompts.json"]
    if a.ag_step == "ablate":
        files += [p for p in src.glob("*/groups.json")] + [p.with_name("graph.json") for p in src.glob("*/groups.json")]
    for f in files:
        store.put_file(f"data/attrib_graphs/{f.relative_to(src).as_posix()}", f)
    tid = f"attrib:{a.ag_step}" + (f":{a.limit}" if a.limit else "")
    cfg = store.get_json("config.json") or {"kind": "attrib", "model": a.model, "revision": rev, "dtype": a.dtype,
                                            "env": {}, "shards": [], "open": False}
    cfg["attrib"] = {"rows": a.ag_rows}
    store.put_json("config.json", cfg)
    man = store.get_json("manifest.json") or {"tasks": []}
    if tid not in [t["id"] for t in man["tasks"]]:
        man["tasks"].append({"id": tid})
    store.put_json("manifest.json", man)
    print(f"{store}: attrib run, task {tid}, {len(files)} input files, AG_ROWS={a.ag_rows}, model {a.model}@{rev[:10]}")


def make_jlens_run(a, store, rev):
    """J-lens shards (src/loop_v2/jlens.py estimate): uploads the atlas docs file once; one task per shard."""
    docs_key = "data/" + Path(a.jl_docs).name
    if not store.exists(docs_key):
        store.put_file(docs_key, a.jl_docs)
    tasks = []
    for s in a.jl_shards.split(","):
        rng, _, rows = s.partition(":")
        tasks.append({"id": f"jlens:{rng}:r{int(rows or 4)}"})
    store.put_json("config.json", {"kind": "jlens", "model": a.model, "revision": rev, "dtype": a.dtype, "env": {},
                                   "jlens": {"windows": a.jl_windows, "docs_key": docs_key}, "shards": [], "open": False})
    store.put_json("manifest.json", {"tasks": tasks})
    print(f"{store}: jlens run, tasks {[t['id'] for t in tasks]}, docs {docs_key}, model {a.model}@{rev[:10]}")


def make_links_run(a, store, rev):
    """links run (LINKS.md): one input per mapped concept from the run records, batches of --batch-size."""
    import glob
    import json
    specs = []
    for f in sorted(glob.glob(a.records)):
        batch = Path(f).parent.name
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if (r.get("row") or {}).get("status") != "ok":
                continue
            rk = r["rank"]
            specs.append({"concept": r["concept"], "units": [e["neuron"] for e in rk["top"][:30]],
                          "types": [x.get("role") for x in r["v2"].get("roles_top30", [])][:30],
                          "prompts": rk["prompts"], "answer_id": rk["answer_ids"][0], "base": rk["base_per_prompt"],
                          "source": {"records": f.replace("\\", "/"), "run_batch": batch}})
            if a.limit and len(specs) >= a.limit:
                break
        if a.limit and len(specs) >= a.limit:
            break
    if not specs:
        raise SystemExit("no ok concepts found in the records")
    bad = [s["concept"] for s in specs if len(s["prompts"]) != len(s["base"])]
    if bad:
        raise SystemExit(f"prompts/base length mismatch for {len(bad)} concepts, e.g. {bad[:3]}")
    batches = []
    for i in range(0, len(specs), a.batch_size):
        b = f"b{i // a.batch_size + 1:04d}"; batches.append(b)
        for s in specs[i:i + a.batch_size]:
            store.put_json(f"inputs/{b}/{_slug(s['concept'])}.json", s)
    store.put_json("config.json", {"kind": "links", "model": a.model, "revision": rev, "dtype": a.dtype, "env": {},
                                   "links": {"batch_size": a.batch_size, "batches": len(batches), "concepts": len(specs),
                                             "method": "LINKS.md (18 Oct 2026)"}, "shards": [], "open": False})
    store.put_json("manifest.json", {"tasks": [{"id": f"links:{b}"} for b in batches]})
    print(f"{store}: links run, {len(specs)} concepts in {len(batches)} batches of up to {a.batch_size}, "
          f"{sum(len(s['prompts']) for s in specs)} prompts, model {a.model}@{rev[:10]}")


def _slug(concept):
    import re
    return re.sub(r"[^a-z0-9]+", "_", concept.lower()).strip("_")[:60] or "concept"


def _ready_specs(folder, skip):
    """(slug, spec) for every complete spec file not yet uploaded. A file still being written (modified in the last
    5 s, or not valid JSON yet) is left for the next round."""
    import json
    out = []
    for f in sorted(Path(folder).glob("*.json")):
        if time.time() - f.stat().st_mtime < 5:
            continue
        try:
            sp = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if "concept" not in sp:
            print(f"skipping {f.name}: no 'concept' key", flush=True); continue
        slug = _slug(sp["concept"])
        if slug not in skip:
            out.append((slug, sp))
    return out


def make_map_run(a, store, rev):
    """Loop v2 run: specs/<batch>/<slug>.json, one map:<batch> task each.
    One-shot (default): upload every spec in --specs, write config + manifest.
    Incremental (--append, optionally --watch N): add new specs as further batches of --batch-size to an existing or
    new run while the run stays "open" (workers and the launcher wait rather than exit). With --watch, repeat every N
    seconds until <specs>/_PREPARE_DONE exists; then the last partial batch is uploaded and the run is closed."""
    env = dict(e.split("=", 1) for e in a.env) or {"RECALL_VARIANT": "V3"}
    if a.specs_in_store:
        batches = sorted({k.split("/")[1] for k in store.list("specs/") if k.endswith(".json")})
        if not batches:
            raise SystemExit("no specs found: nothing to map")
        store.put_json("config.json", {"kind": "map", "model": a.model, "revision": rev, "dtype": a.dtype, "env": env,
                                       "map": {"batch_size": a.batch_size, "batches": len(batches)}, "shards": [], "open": False})
        store.put_json("manifest.json", {"tasks": [{"id": f"map:{b}"} for b in batches]})
        print(f"{store}: map run from specs in the store, {len(batches)} batches"); return
    cfg = store.get_json("config.json")
    if cfg is None:
        cfg = {"kind": "map", "model": a.model, "revision": rev, "dtype": a.dtype, "env": env,
               "map": {"batch_size": a.batch_size, "batches": 0}, "shards": [], "open": bool(a.append)}
        store.put_json("config.json", cfg); store.put_json("manifest.json", {"tasks": []})
    elif cfg.get("kind") != "map":
        raise SystemExit(f"{store} holds a non-map run")
    bs = cfg["map"]["batch_size"]
    index = store.get_json("map_index.json") or {"uploaded": [], "next_batch": 1}
    done_flag = Path(a.specs) / "_PREPARE_DONE"
    while True:
        finishing = (not a.append) or (done_flag.exists() if a.watch else False)
        new = _ready_specs(a.specs, set(index["uploaded"]))
        n_full = len(new) // bs * bs
        take = new if finishing else new[:n_full]                # partial batches wait for more specs unless finishing
        added = []
        for i in range(0, len(take), bs):
            bname = f"b{index['next_batch']:04d}"; index["next_batch"] += 1
            for slug, sp in take[i:i + bs]:
                store.put_json(f"specs/{bname}/{slug}.json", sp); index["uploaded"].append(slug)
            added.append(bname)
        if added:
            man = store.get_json("manifest.json") or {"tasks": []}
            man["tasks"] += [{"id": f"map:{x}"} for x in added]
            store.put_json("map_index.json", index)              # index first: a crash here re-uploads, never skips
            store.put_json("manifest.json", man)
            cfg["map"]["batches"] = len(man["tasks"])
        if finishing and cfg.get("open"):
            cfg["open"] = False
        store.put_json("config.json", cfg)
        print(f"[{time.strftime('%H:%M:%S')}] {store}: +{sum(1 for _ in take)} concepts in {len(added)} new batches; "
              f"{len(index['uploaded'])} concepts / {index['next_batch'] - 1} batches in total; run "
              f"{'open' if cfg.get('open') else 'closed'}", flush=True)
        if finishing or not a.watch:
            break
        time.sleep(a.watch)


if __name__ == "__main__":
    main()
