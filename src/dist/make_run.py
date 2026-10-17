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
    ap.add_argument("--kind", default="neuron", choices=["neuron", "map"],
                    help="neuron: observe/merge/ablate (default). map: loop v2 concept batches from specs")
    ap.add_argument("--specs", help="map: local folder of spec .json files, each with a 'concept' key")
    ap.add_argument("--specs-in-store", action="store_true", help="map: specs already uploaded under specs/<batch>/")
    ap.add_argument("--batch-size", type=int, default=20, help="map: concepts per task")
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE set on every worker (map default RECALL_VARIANT=V3)")
    a = ap.parse_args()

    store = open_store(a.store)
    if store.exists("config.json"):
        raise SystemExit(f"{store} already holds a run (config.json exists)")
    rev = a.revision or REVISIONS.get(a.model)
    if not rev:
        raise SystemExit("pass --revision (a Hugging Face commit hash)")

    if a.kind == "map":
        return make_map_run(a, store, rev)
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


def make_map_run(a, store, rev):
    """Loop v2 run: specs/<batch>/<slug>.json (uploaded here or already in the store), one map:<batch> task each."""
    import json
    import re
    env = dict(e.split("=", 1) for e in a.env) or {"RECALL_VARIANT": "V3"}
    if a.specs_in_store:
        batches = sorted({k.split("/")[1] for k in store.list("specs/") if k.endswith(".json")})
    else:
        files = sorted(Path(a.specs).glob("*.json"))
        specs = [json.loads(f.read_text(encoding="utf-8")) for f in files]
        missing = [f.name for f, s in zip(files, specs) if "concept" not in s]
        if missing:
            raise SystemExit(f"specs without a 'concept' key: {missing[:5]}")
        batches = []
        for i in range(0, len(specs), a.batch_size):
            b = f"b{i // a.batch_size + 1:04d}"; batches.append(b)
            for s in specs[i:i + a.batch_size]:
                slug = re.sub(r"[^a-z0-9]+", "_", s["concept"].lower()).strip("_")[:60] or "concept"
                store.put_json(f"specs/{b}/{slug}.json", s)
    if not batches:
        raise SystemExit("no specs found: nothing to map")
    cfg = {"kind": "map", "model": a.model, "revision": rev, "dtype": a.dtype, "env": env,
           "map": {"batch_size": a.batch_size, "batches": len(batches)}, "shards": []}
    store.put_json("config.json", cfg)
    store.put_json("manifest.json", {"tasks": [{"id": f"map:{b}"} for b in batches]})
    n = sum(len(store.list(f"specs/{b}/")) for b in batches)
    print(f"{store}: map run, {n} concepts in {len(batches)} batches of up to {a.batch_size}, env {env}, model {a.model}@{rev[:10]}")


if __name__ == "__main__":
    main()
