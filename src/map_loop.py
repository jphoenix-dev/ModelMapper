"""Faster mapping loop: Luna and the GPU run as separate queues that meet only through the cached spec files.

Replaces wave_free.py (without steering) + rank_save.py for coverage runs (research branch; built 17 Oct after the
coverage-run timing showed: ~0.64 min per NOT-recalled concept, mostly waiting on Luna with the GPU idle; ranking done
twice; steering costly and ~0 for Wikipedia titles; cross-activation recomputed whenever the neuron union grows).

Per concept, the same steps and the same prompts / rules as pipeline_fast.make_prompts (V3) + wave_free.py + rank_save.py:
  L1  Luna writes the spec (concept_pipeline.PROMPT_SYSTEM)                    -> <slug>_spec.json   [Luna pool]
  L2  Luna rewrites the recall prompts to V2/V3 (recall_variants.REWRITE)       -> same spec file     [Luna pool]
  G1  Qwen answers each V3 question freely (answer discovery generation)        -> spec 'discovery_replies' [GPU]
  L3  Luna judges which replies name the concept; extra answer forms (same rules) -> spec              [Luna pool]
  G2  answer-form scoring + recall; ranking (attribution top-200, each neuron off alone); k_mean / k_median / k_flip;
      random-neuron control curve; every neuron's activation maps                                        [GPU]
The GPU loop always takes what is ready (G2 first, then G1) and only waits when nothing is; that idle time is logged.
Cached specs (any of L1/L2/L3 already present) are reused, so Luna is skipped for them.
Steering is NOT run (coverage mode).

Saved (everything, per James's rule): per concept a row with status, answer form, every answer form's per-prompt P,
recall prompts, per-prompt baseline log-odds, k values and drop curves, random curves, all 200 candidates with their
per-prompt single-neuron drops, the top-30 (rank_save format), per-step wall time and per-step Luna tokens / cost;
arrays <OUT>/rank_save_arrays/<slug>.npz: attr, last_act (answer-position mean activation from a prompt-only pass = cross-activation 'ans'; the attribution pass's
version is kept as last_act_attr_pass),
pmax / pmin (mean over prompts of each neuron's max / min activation over prompt positions 1..end, for 'peak'),
top200; and for ATTENTION HEADS: head_attr (per-head attribution, L x H), head_out_answer_pos (each head's output at
the answer position, mean over prompts, L x H*D), att<i> (per prompt: where every head looks from the answer position,
L x H x T); in the rank row, the 50 most-attributed heads each switched off alone with per-prompt recall drops.
Per-position attribution maps are NOT saved (~800 MB per concept); their per-neuron totals are ('attr').
Outputs: <OUT>/map_loop_<tag>.json (rows), <OUT>/rank_save_<tag>.json (rank_save-compatible), console log.

KNOCK_BATCH=<n> (env): test n single knockouts per forward pass (default 0 = one per pass, the validated path).
Usage: RECALL_VARIANT=V3 CONCEPT_OUT_DIR=<dir> python src/map_loop.py <wave json> <tag> [luna_budget_usd=2.5] [luna_workers=8]
"""
import json
import math
import os
import random
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

import concept_pipeline as cp
import oai
import recall_variants as rv
from config import DEFAULT_MODEL
from neurons import DEVICE, mlps, steer
from pipeline_fast import FastPipeline, leaks
from wave_free import KS, p_of

ROOT = Path(__file__).resolve().parents[1]
PRICE_IN, PRICE_OUT = 0.10e-6, 0.50e-6          # Luna's real price (James)
TOPN = 30
_tls = threading.local()


class Usage:
    def __init__(self):
        self.lock = threading.Lock(); self.by = {}; self.total = [0, 0, 0]

    def add(self, concept, step, u):
        with self.lock:
            d = self.by.setdefault(concept, {}).setdefault(step, [0, 0, 0])
            for i, k in enumerate(("prompt_tokens", "completion_tokens")):
                d[i] += u.get(k, 0); self.total[i] += u.get(k, 0)
            d[2] += 1; self.total[2] += 1

    def usd(self, x=None):
        x = x or self.total
        return x[0] * PRICE_IN + x[1] * PRICE_OUT


USAGE = Usage()


def luna_json(prompt, system, concept, step):
    """oai.chat_json, but returning exact per-call usage (no shared counter races)."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    data = oai._request("POST", "/chat/completions", {"model": oai.MODEL, "messages": msgs, "response_format": {"type": "json_object"}})
    USAGE.add(concept, step, data.get("usage") or {})
    return json.loads(data["choices"][0]["message"]["content"])


def spec_path(concept):
    return cp.OUT / f"{cp.slug(concept)}_spec.json"


def load_spec(concept):
    p = spec_path(concept)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def save_spec(concept, spec):
    spec_path(concept).write_text(json.dumps(spec, indent=1, ensure_ascii=False), encoding="utf-8")


def luna_prepare(concept, timing, budget):
    """L1 + L2 (Luna only). Returns spec or raises PromptsFailed-like ValueError."""
    spec = load_spec(concept)
    if spec is None:
        if USAGE.usd() >= budget:
            raise ValueError("Luna budget reached before this concept's spec")
        t = time.time()
        spec = luna_json(f"Concept: {concept}", cp.PROMPT_SYSTEM, concept, "L1_spec")
        timing["L1_spec"] = round(time.time() - t, 2); save_spec(concept, spec)
    if "answer" not in spec or "recall" not in spec:
        raise ValueError(f"Luna returned no answer/recall for this concept: {str(spec)[:160]}")
    if any(isinstance(q, dict) for q in spec.get("recall", [])):         # ISSUES #33 normalisation
        spec["recall"] = [q.get("prompt") or q.get("text") or next(iter(q.values())) if isinstance(q, dict) else q for q in spec["recall"]]
        save_spec(concept, spec)
    if "recall_v2" not in spec:
        t = time.time(); listing = "\n".join(f"{i+1}. {p}" for i, p in enumerate(spec["recall"])); got = []
        for _ in range(3):
            if USAGE.usd() >= budget:
                raise ValueError("Luna budget reached during rewrites")
            out = luna_json(f"Answer (for your reference only, never write it): {spec['answer']}\nFragments:\n{listing}", rv.REWRITE, concept, "L2_rewrite")
            items = out.get("items", []); got.append(len(items))
            if len(items) == len(spec["recall"]) and all("v2" in it and "v3" in it for it in items):
                spec["recall_v2"] = [it["v2"] for it in items]; spec["recall_v3"] = [it["v3"] for it in items]
                save_spec(concept, spec); break
        timing["L2_rewrite"] = round(time.time() - t, 2)
        if "recall_v2" not in spec:
            raise ValueError(f"labeller returned {got} rewrite items for {len(spec['recall'])} recall prompts in 3 attempts")
    return spec


def luna_judge(concept, spec, timing, budget):
    """L3: same prompt and rules as FastPipeline.discover_answers, on replies generated by the GPU (G1)."""
    if "answer_forms_extra" in spec:
        return spec
    t = time.time(); name = spec["answer"].strip(); replies = spec["discovery_replies"]; questions = spec["recall_v3"]
    try:
        if USAGE.usd() >= budget:
            raise ValueError("budget")
        ok = luna_json(
            f"Target name: {name}\nFor each numbered reply, is it this specific thing's own name (allowing "
            f"singular/plural, capitalisation, a well-known synonym or short name)? Say false for a different "
            f"thing, and false for a generic descriptive name made from place or category words (e.g. 'River X "
            f"Bridge', 'Town Y Museum') unless that IS the target's established name. "
            f"Reply JSON {{\"ok\": [true/false, ...]}} in order.\n" + "\n".join(f"{i+1}. {r}" for i, r in enumerate(replies)),
            None, concept, "L3_discovery_judge")["ok"]
    except Exception:  # noqa: BLE001  (as discover_answers: a failed judge accepts nothing)
        ok = [False] * len(replies)
    good = [r for r, v in zip(replies, ok) if v and r and sum(leaks(q, r) for q in questions) < 0.3 * len(questions)]
    spec["answer_forms_extra"] = [f for f, n in Counter(good).most_common(3) if n >= 2]
    spec["discovered_answers"] = [{"reply": r, "names_concept": bool(v)} for r, v in zip(replies, ok)]
    save_spec(concept, spec); timing["L3_discovery_judge"] = round(time.time() - t, 2)
    return spec


KNOCK_BATCH = int(os.environ.get("KNOCK_BATCH", "0"))   # >0: test this many single knockouts per forward pass


def multi_knockout_logodds(pipe, pairs, groups, kind):
    """Exact log-odds of every pair under each of len(groups) single knockouts, several knockouts per forward pass.

    groups: list of (layer, index); kind 'neuron' (activation set to 0 at every position, as neurons.steer clamp) or
    'head' (the head's output slice zeroed at every position, as pipe.ablate_heads). Rows = pairs x groups; each row
    has only its own group's unit knocked out. Logits are taken only at the answer positions (lm_head on gathered
    hidden states); log-odds as pipeline_fast.batch_logodds (float64, expm1). Returns {group: [log-odds per pair]}."""
    m = pipe.model; H = m.config.num_attention_heads; D = m.config.head_dim
    seqs, spans = [], []
    for prompt, ans in pairs:
        p_ids = pipe.tok.encode(prompt); seqs.append(p_ids + ans[:-1]); spans.append((len(p_ids) - 1, ans))
    T = max(map(len, seqs)); pad = pipe.tok.pad_token_id or 0
    base_ids = torch.full((len(seqs), T), pad, device=DEVICE); base_mask = torch.zeros_like(base_ids)
    for i, s in enumerate(seqs):
        base_ids[i, :len(s)] = torch.tensor(s, device=DEVICE); base_mask[i, :len(s)] = 1
    out = {}
    for c0 in range(0, len(groups), KNOCK_BATCH):
        chunk = groups[c0:c0 + KNOCK_BATCH]; G = len(chunk); P = len(seqs)
        ids = base_ids.repeat(G, 1); mask = base_mask.repeat(G, 1)
        by_layer = {}
        for gi, (l, k) in enumerate(chunk):
            by_layer.setdefault(l, []).append((gi, k))
        hooks = []
        for l, items in by_layer.items():
            rows = torch.tensor([gi * P + p for gi, _ in items for p in range(P)], device=DEVICE)
            idx = torch.tensor([k for _, k in items for _ in range(P)], device=DEVICE)
            if kind == "neuron":
                def h(mod, a, rows=rows, idx=idx):
                    x = a[0].clone(); x[rows, :, idx] = 0; return (x,)
                hooks.append(mlps(m)[l].down_proj.register_forward_pre_hook(h))
            else:
                def h(mod, a, rows=rows, idx=idx):
                    x = a[0].clone().view(a[0].shape[0], a[0].shape[1], H, D)
                    x[rows, :, idx, :] = 0; return (x.view(a[0].shape),)
                hooks.append(m.model.layers[l].self_attn.o_proj.register_forward_pre_hook(h))
        try:
            with torch.no_grad():
                hs = m.model(input_ids=ids, attention_mask=mask).last_hidden_state
        finally:
            for x in hooks:
                x.remove()
        for gi, g in enumerate(chunk):
            vals = []
            for p, (start, ans) in enumerate(spans):
                pos = torch.arange(start, start + len(ans), device=DEVICE)
                lp = m.lm_head(hs[gi * P + p, pos]).double().log_softmax(-1)
                s = lp[torch.arange(len(ans), device=DEVICE), torch.tensor(ans, device=DEVICE)].sum().item()
                if s >= 0.0:
                    raise ValueError("log P(answer) rounds to 0 in float64 (log-odds > ~36)")
                vals.append(s - math.log(-math.expm1(s)))
            out[g] = vals
    return out


def main():
    wave_p, tag = sys.argv[1], sys.argv[2]
    budget = float(sys.argv[3]) if len(sys.argv) > 3 else 2.5
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    assert os.environ.get("RECALL_VARIANT", "V0") == "V3", "map_loop implements the V3 (question) format only"
    wave = json.loads((ROOT / wave_p).read_text(encoding="utf-8"))
    concepts = [(cat, c) for cat, cs in wave["concepts"].items() for c in cs]
    out_p = cp.OUT / f"map_loop_{tag}.json"; rank_p = cp.OUT / f"rank_save_{tag}.json"
    out = json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists() else {}
    ranks = json.loads(rank_p.read_text(encoding="utf-8")) if rank_p.exists() else {}
    todo = [(cat, c) for cat, c in concepts if c not in out]
    pipe = FastPipeline(os.environ.get("MODEL", DEFAULT_MODEL), mode="add", reply_only=True); m = pipe.model
    arr_dir = cp.OUT / "rank_save_arrays"; arr_dir.mkdir(exist_ok=True)
    timing = {c: {} for _, c in todo}; cat_of = dict((c, cat) for cat, c in todo)
    pool = ThreadPoolExecutor(workers)
    jpool = ThreadPoolExecutor(workers)       # separate pool for L3: a shared FIFO pool starved the judges behind every
    #                                           concept's L1+L2 (first coverage-3 run: no concept finished for ~15 min)
    prep = {c: pool.submit(luna_prepare, c, timing[c], budget) for _, c in todo}      # L1+L2 for all, in parallel
    judge, ready_g1, done = {}, {}, set()
    t_start = time.time(); gpu_idle = 0.0

    def save():
        out_p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        rank_p.write_text(json.dumps(ranks, indent=1, ensure_ascii=False), encoding="utf-8")

    def finish(c, row):
        row["category"] = cat_of[c]; row["timing_s"] = timing[c]
        u = USAGE.by.get(c, {}); row["luna"] = {k: {"in": v[0], "out": v[1], "calls": v[2], "usd": round(USAGE.usd(v), 5)} for k, v in u.items()}
        out[c] = row; done.add(c); save()
        print(f"{c[:30]:<30} {row['status']:<14} " + (f"recall {row['recall']} k_mean {row.get('k_mean')} k_flip {row.get('k_flip')} | " if row['status'] == 'ok' else "")
              + "time " + " ".join(f"{k}={v}" for k, v in timing[c].items()) + f" | Luna so far ${USAGE.usd():.3f}", flush=True)

    def g1(c, spec):
        """GPU: free answers to the V3 questions (as discover_answers), then hand the judging to Luna."""
        t = time.time()
        chat = [pipe.tok.apply_chat_template([{"role": "user", "content": q + " Answer with just the name."}],
                                             tokenize=False, add_generation_prompt=True, enable_thinking=False) for q in spec["recall_v3"]]
        if "answer_forms_extra" not in spec and "discovery_replies" not in spec:
            pipe.tok.padding_side = "left"
            ids = pipe.tok(chat, return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad():
                g = m.generate(**ids, max_new_tokens=16, do_sample=False, pad_token_id=pipe.tok.pad_token_id)
            spec["discovery_replies"] = [pipe.tok.decode(o[ids.input_ids.shape[1]:], skip_special_tokens=True).strip().split("\n")[0].strip().rstrip(".") for o in g]
            save_spec(c, spec)
        timing[c]["G1_discovery_gen"] = round(time.time() - t, 2)
        judge[c] = jpool.submit(luna_judge, c, spec, timing[c], budget)

    def g2(c, spec):
        """GPU: recall (identical to the V3 path), then ranking and curves (identical to wave_free + rank_save, exact log-odds)."""
        T = timing[c]; t = time.time()
        chat = [pipe.tok.apply_chat_template([{"role": "user", "content": q + " Answer with just the name."}],
                                             tokenize=False, add_generation_prompt=True, enable_thinking=False) for q in spec["recall_v3"]]
        a = spec["answer"].strip(); base_ = a[4:].strip() if a.lower().startswith("the ") else a
        forms = list(dict.fromkeys([a, a[:1].upper() + a[1:], a.title(), base_, base_[:1].upper() + base_[1:]] + spec.get("answer_forms_extra", [])))
        best = None; form_scores = {}
        for f in forms:
            t_ = pipe.tok.encode(f); ps_ = pipe.batch_probs([(q, t_) for q in chat])
            form_scores[f] = [round(p, 4) for p in ps_]
            score = sum(p >= cp.MIN_P for p in ps_)
            if best is None or score > best[0]:
                best = (score, t_, ps_, f)
        _, tid, ps, form = best
        recall = [(q, round(p, 3)) for q, p, raw in zip(chat, ps, spec["recall_v3"]) if p >= cp.MIN_P and not leaks(raw, spec["answer"])]
        T["G2_recall"] = round(time.time() - t, 2)
        row = {"answer": a, "answer_form": form, "recall": len(recall), "discovered": spec.get("answer_forms_extra"),
               "answer_form_scores": form_scores, "recall_prompts": [q for q, _ in recall]}
        if len(recall) < 5:
            row["status"] = "not_recalled"; return row
        rp = [q for q, _ in recall]; pairs = [(q, tid) for q in rp]
        t = time.time(); base = pipe.batch_logodds(pairs); right = [x > 0 for x in base]
        A = pipe.attribute(rp, tid); top = A["top"][:200]; T["G2_attribution"] = round(time.time() - t, 2)

        def per_prompt(units):
            los = pipe.batch_logodds(pairs, steer(m, [(l, n, 0.0) for l, n in units]) if units else None)
            return [b - x for b, x in zip(base, los)], [p_of(x) for x in los]
        t = time.time()
        if KNOCK_BATCH > 0:
            mk = multi_knockout_logodds(pipe, pairs, top, "neuron"); pp = {u: [b - x for b, x in zip(base, mk[u])] for u in top}
        else:
            pp = {u: per_prompt([u])[0] for u in top}
        T["G2_single_knockouts"] = round(time.time() - t, 2)
        drop1 = {u: sum(pp[u]) / len(rp) for u in top}; single = sorted(top, key=lambda u: -sum(pp[u]))

        def kvals(order):
            km = kd = kf = None; drops = []
            for k in KS:
                d, ps2 = per_prompt(order[:k]); mean = sum(d) / len(d); med = sorted(d)[len(d) // 2]
                flip = sum(p < .5 for p, r in zip(ps2, right) if r) / max(1, sum(right))
                drops.append(round(mean, 2))
                km = km or (k if mean >= 3 else None); kd = kd or (k if med >= 3 else None); kf = kf or (k if flip >= .5 else None)
            return km, kd, kf, drops
        t = time.time(); km, kd, kf, drops = kvals(single); T["G2_k_curve"] = round(time.time() - t, 2)
        rng = random.Random(2)
        t = time.time(); _, _, _, rdrops = kvals([(l, rng.randrange(pipe.W)) for l, _ in top]); T["G2_random_curve"] = round(time.time() - t, 2)
        # activation maps for cross-activation: answer-position mean (= A['last_act']) and per-neuron max/min over prompt positions
        # + attention heads (James, 17 Oct): each head's output at the answer position (mean over prompts) and, per
        # prompt, where every head looks from the answer position (eager attention for this one pass)
        t = time.time(); L_, W_ = pipe.L, pipe.W; Hh = m.config.num_attention_heads
        pmax = torch.zeros(L_, W_, device=DEVICE); pmin = torch.zeros(L_, W_, device=DEVICE)
        zmean = torch.zeros(L_, m.config.num_attention_heads * m.config.head_dim, device=DEVICE); att_rows = {}
        ans = torch.zeros(L_, W_, device=DEVICE)
        for qi, q in enumerate(rp):                      # default (sdpa) kernel, prompt only - exactly as cross_activation.py
            ids = pipe.tok(q, return_tensors="pt").to(DEVICE).input_ids; st = {}; zs = {}
            hk = [mlps(m)[l].down_proj.register_forward_pre_hook(lambda mod, a_, l=l: st.__setitem__(l, a_[0][0, 1:].float())) for l in range(L_)]
            hk += [m.model.layers[l].self_attn.o_proj.register_forward_pre_hook(lambda mod, a_, l=l: zs.__setitem__(l, a_[0][0, -1].float())) for l in range(L_)]
            try:
                with torch.no_grad():
                    m(input_ids=ids)
            finally:
                for h in hk:
                    h.remove()
            for l in range(L_):
                ans[l] += st[l][-1] / len(rp); pmax[l] += st[l].max(0).values / len(rp); pmin[l] += st[l].min(0).values / len(rp)
                zmean[l] += zs[l] / len(rp)
        m.set_attn_implementation("eager")               # attention patterns need the eager kernel: separate pass
        try:
            for qi, q in enumerate(rp):
                ids = pipe.tok(q, return_tensors="pt").to(DEVICE).input_ids
                with torch.no_grad():
                    att = m(input_ids=ids, output_attentions=True).attentions
                att_rows[f"att{qi}"] = torch.stack([a_[0, :, -1, :] for a_ in att]).half().cpu().numpy()   # [L, H, T]
        finally:
            m.set_attn_implementation("sdpa")
        T["G2_maps_heads"] = round(time.time() - t, 2)
        # causal check per head: each of the 50 most-attributed heads switched off alone (all positions), per-prompt drops
        t = time.time(); ha = A["head_attr"].detach()
        top_heads = [divmod(i.item(), Hh) for i in ha.flatten().argsort(descending=True)[:50]]
        head_pp = {}
        if KNOCK_BATCH > 0:
            mh = multi_knockout_logodds(pipe, pairs, top_heads, "head")
            head_pp = {f"L{l}H{h}": [round(b - x, 3) for b, x in zip(base, mh[(l, h)])] for l, h in top_heads}
        else:
            for (l, h) in top_heads:
                los = pipe.batch_logodds(pairs, pipe.ablate_heads([(l, h)]))
                head_pp[f"L{l}H{h}"] = [round(b - x, 3) for b, x in zip(base, los)]
        T["G2_head_knockouts"] = round(time.time() - t, 2)
        np.savez_compressed(arr_dir / f"{cp.slug(c)}.npz", attr=A["attr"].detach().half().cpu().numpy(), last_act=ans.half().cpu().numpy(),
                            last_act_attr_pass=A["last_act"].detach().half().cpu().numpy(),
                            pmax=pmax.half().cpu().numpy(), pmin=pmin.half().cpu().numpy(), top200=np.array(top, dtype=np.int32),
                            head_attr=ha.half().cpu().numpy(), head_out_answer_pos=zmean.half().cpu().numpy(), **att_rows)
        row.update(status="ok", baseline_mean_logodds=round(sum(base) / len(base), 2), base_per_prompt=[round(x, 3) for x in base],
                   k_mean=km, k_median=kd, k_flip=kf, drops=drops, random_drops=rdrops)
        cand = [{"neuron": f"L{l}:{n}", "single_drop": round(drop1[(l, n)], 3), "attr_rank": top.index((l, n)),
                 "act_answer_pos": round(A["last_act"][l, n].item(), 3)} for l, n in single]
        ranks[c] = {"answer_token": pipe.tok.decode([tid[0]]), "recall": len(rp), "k_mean": km, "k_flip": kf,
                    "baseline_mean_logodds": round(sum(base) / len(base), 3), "base_per_prompt": [round(x, 3) for x in base],
                    "prompts": rp, "candidates_all200": cand, "top": cand[:TOPN],
                    "per_prompt_drop_all200": {f"L{l}:{n}": [round(x, 3) for x in pp[(l, n)]] for l, n in single},
                    "matched_original": False, "mismatch": None,
                    "heads_top50_by_attr": [f"L{l}H{h}" for l, h in top_heads], "head_knockout_per_prompt": head_pp,
                    "head_knockout_mean": {k: round(sum(v) / len(v), 3) for k, v in head_pp.items()}}
        return row

    while len(done) < len(todo):
        did = False
        for c, f in list(judge.items()):                              # G2 first: finish concepts
            if f.done():
                del judge[c]
                try:
                    spec = f.result(); finish(c, g2(c, spec))
                except Exception as e:  # noqa: BLE001
                    finish(c, {"status": "failed", "detail": repr(e)})
                did = True; break
        if did:
            continue
        for c, f in list(prep.items()):                               # then G1 for any concept whose Luna prep is done
            if f.done():
                del prep[c]
                try:
                    spec = f.result(); g1(c, spec)
                except Exception as e:  # noqa: BLE001
                    finish(c, {"status": "prompts_failed", "detail": str(e)[:300]})
                did = True; break
        if not did:
            t = time.time(); time.sleep(0.2); gpu_idle += time.time() - t
    pool.shutdown(); jpool.shutdown()
    el = time.time() - t_start
    print(f"done {len(todo)} concepts in {el / 60:.1f} min; GPU idle (waiting on Luna) {gpu_idle / 60:.1f} min "
          f"({gpu_idle / max(el, 1e-9):.0%}); Luna {USAGE.total[2]} calls, ${USAGE.usd():.3f}", flush=True)
    meta_p = cp.OUT / f"map_loop_{tag}_meta.json"
    meta_p.write_text(json.dumps({"elapsed_s": round(el, 1), "gpu_idle_s": round(gpu_idle, 1), "luna_calls": USAGE.total[2],
                                  "luna_in": USAGE.total[0], "luna_out": USAGE.total[1], "luna_usd": round(USAGE.usd(), 4),
                                  "concepts": len(todo), "workers": workers}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
