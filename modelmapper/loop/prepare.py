"""Loop v2 PREPARE (desktop only: Luna + a light GPU step). Writes one spec per concept for the MAP tasks.

The same Luna steps and rules as map_loop.py (imported unchanged): L1 spec (questions + answer), L2 V2/V3 rewrites,
G1 Qwen's free answers to the V3 questions (answer discovery), L3 Luna judges which free answers name the concept.
Efficiency (the loop's lessons): Luna runs in a thread pool while the GPU generates G1 answers for G1_BATCH concepts per
batch as their L1/L2 finish; L3 judges in a separate pool; everything saved as it arrives (spec cache), resumable.
(G1 is batched across concepts here, so a borderline greedy reply can differ from the per-concept v1 generation -
these replies only propose extra answer forms, which L3 then judges.)
Output: <run>/specs/<slug>.json (the spec + "concept"), <run>/prepare_log.jsonl (one line per concept: status, timing,
Luna tokens), <run>/spec_cache/ (map_loop's spec files).
Usage: python src/loop_v2/prepare.py <frame.json> <run_dir> [luna_budget_usd=5] [luna_workers=16]
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

run = Path(sys.argv[2]); (run / "spec_cache").mkdir(parents=True, exist_ok=True); (run / "specs").mkdir(exist_ok=True)
os.environ["CONCEPT_OUT_DIR"] = str(run / "spec_cache")            # map_loop's spec files go here (read at import)
os.environ.setdefault("RECALL_VARIANT", "V3")
import torch  # noqa: E402

from modelmapper.core import concept_pipeline as cp  # noqa: E402
from modelmapper.loop import map_loop as ml  # noqa: E402
from modelmapper.core.config import DEFAULT_MODEL  # noqa: E402
from modelmapper.core.neurons import DEVICE, load  # noqa: E402

G1_BATCH = 16


def main():
    frame = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    concepts = [x["concept"] for x in frame["draw"]]
    budget = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0; workers = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    log_p = run / "prepare_log.jsonl"
    done = {json.loads(l)["concept"] for l in log_p.read_text(encoding="utf-8").splitlines()} if log_p.exists() else set()
    todo = [c for c in concepts if c not in done]
    print(f"{len(concepts)} concepts in the frame; {len(todo)} to prepare", flush=True)
    model, tok = load(DEFAULT_MODEL); tok.padding_side = "left"
    timing = {c: {} for c in todo}
    pool = ThreadPoolExecutor(workers); jpool = ThreadPoolExecutor(workers)
    prep = {c: pool.submit(ml.luna_prepare, c, timing[c], budget) for c in todo}
    judge, buf = {}, []
    logf = open(log_p, "a", encoding="utf-8"); t0 = time.time(); gpu = 0.0; counts = {}

    def finish(c, status, spec=None, detail=None):
        if spec is not None and status == "ready":
            (run / "specs" / f"{cp.slug(c)}.json").write_text(json.dumps({**spec, "concept": c}, indent=1, ensure_ascii=False), encoding="utf-8")
        u = ml.USAGE.by.get(c, {})
        logf.write(json.dumps({"concept": c, "status": status, "detail": detail, "timing_s": timing[c],
                               "luna": {k: {"in": x[0], "out": x[1], "calls": x[2]} for k, x in u.items()}}, ensure_ascii=False) + "\n"); logf.flush()
        counts[status] = counts.get(status, 0) + 1
        if sum(counts.values()) % 50 == 0:
            print(f"{time.strftime('%H:%M:%S')} {sum(counts.values())}/{len(todo)} done {counts}; GPU {gpu / 60:.1f} min; "
                  f"Luna ${ml.USAGE.usd():.2f}", flush=True)

    def g1(batch):
        nonlocal gpu
        t = time.time(); need = [(c, s) for c, s in batch if "answer_forms_extra" not in s and "discovery_replies" not in s]
        if need:
            chats = [(c, tok.apply_chat_template([{"role": "user", "content": q + " Answer with just the name."}], tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False)) for c, s in need for q in s["recall_v3"]]
            outs = []
            for i in range(0, len(chats), 160):
                ids = tok([x for _, x in chats[i:i + 160]], return_tensors="pt", padding=True).to(DEVICE)
                with torch.no_grad():
                    g = model.generate(**ids, max_new_tokens=16, do_sample=False, pad_token_id=tok.pad_token_id)
                outs += [tok.decode(o[ids.input_ids.shape[1]:], skip_special_tokens=True).strip().split("\n")[0].strip().rstrip(".") for o in g]
            k = 0
            for c, s in need:
                s["discovery_replies"] = outs[k:k + len(s["recall_v3"])]; k += len(s["recall_v3"]); ml.save_spec(c, s)
        dt = time.time() - t; gpu += dt
        for c, s in batch:
            timing[c]["G1_discovery_gen"] = round(dt / len(batch), 2)
            judge[c] = jpool.submit(ml.luna_judge, c, s, timing[c], budget)
    while prep or buf or judge:
        progressed = False
        for c, f in list(prep.items()):
            if f.done():
                del prep[c]; progressed = True
                try:
                    buf.append((c, f.result()))
                except Exception as e:  # noqa: BLE001  recorded, never dropped
                    finish(c, "prompts_failed", detail=str(e)[:300])
        if buf and (len(buf) >= G1_BATCH or not prep):
            g1(buf); buf = []; progressed = True
        for c, f in list(judge.items()):
            if f.done():
                del judge[c]; progressed = True
                try:
                    finish(c, "ready", spec=f.result())
                except Exception as e:  # noqa: BLE001
                    finish(c, "judge_failed", detail=str(e)[:300])
        if not progressed:
            time.sleep(0.2)
    el = time.time() - t0
    msg = (f"prepared {len(todo)} concepts in {el / 60:.1f} min; GPU (G1) busy {gpu / 60:.1f} min; Luna {ml.USAGE.total[2]} calls, "
           f"${ml.USAGE.usd():.3f}; ready specs {len(list((run / 'specs').glob('*.json')))}")
    (run / "prepare_meta.json").write_text(json.dumps({"elapsed_s": round(el, 1), "gpu_s": round(gpu, 1), "luna_calls": ml.USAGE.total[2],
                                                      "luna_in": ml.USAGE.total[0], "luna_out": ml.USAGE.total[1], "luna_usd": round(ml.USAGE.usd(), 4)}, indent=1))
    print(msg)
    pool.shutdown(); jpool.shutdown()


if __name__ == "__main__":
    main()
