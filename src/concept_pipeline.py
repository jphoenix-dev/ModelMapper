"""Automated concept pipeline: concept in -> recall neurons, ablation curve, labels, nudgeability out.

For one concept (e.g. "Mount Everest"):
  1. Luna writes recall prompts that end right before the concept's name (without naming it), plus
     control prompts with other answers (some nearby in topic, some unrelated).
  2. Keep prompts where the model actually gives the answer (P(whole answer) >= MIN_P).
  3. Attribution: activation x gradient of log P(whole answer | prompt) (teacher-forced, so multi-piece
     names like " Mount Everest" count every piece), summed over positions, averaged over prompts.
  4. Ablation curve: zero the top-k (k = 1..50) vs k random neurons from the same layers; record
     P(answer) on recall prompts and on each control. k50 = smallest k that halves P(answer).
  5. Labels: top-N neurons via the atlas (blind Luna + Jev, as in label.py) plus what each writes to
     the vocabulary (direct output readout).
  6. Steering sweep: top late-layer neurons (layer >= L/2), n in N_STEER, strengths in STRENGTHS,
     clamped to strength x own peak in the sign they take on the recall prompts; Jev judges
     "about the concept?" and "fluent?" on 20 neutral prompts. Same for random neurons at the best
     setting. Nudgeability = best (on-topic rate x fluent rate).
Writes results/concepts/<slug>.json.

Usage: python src/concept_pipeline.py <model> "<concept>" [--no-labels] [--reply-only] [--mode clamp|add]
"""
import argparse
import contextlib
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path

import torch

import jev
import oai
from atlas import Atlas, output_words
from neurons import load, mlps, steer
from steer_test import PROMPTS as NEUTRAL, generate

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("CONCEPT_OUT_DIR", ROOT / "results" / "concepts"))  # override for test runs
MIN_P = float(os.environ.get("CONCEPT_MIN_P", 0.2))
KS = [1, 2, 3, 5, 10, 20, 50]
N_LABEL = 10
N_STEER = [2, 4, 6]
STRENGTHS = [0.5, 0.75, 1.0, 1.5, 2.0]

PROMPT_SYSTEM = """You write test prompts for probing whether a language model knows a concept.
Given a concept, return JSON:
{"answer": the concept's name exactly as it would continue a sentence, starting with a space (e.g. " Golden Gate Bridge", " Mount Everest" -> " Everest" if the prompt already ends with "Mount"),
 "recall": [20 prompts],
 "controls": [{"prompt": str, "answer": str}, ... 8 items]}
Rules for "recall": each is a plain sentence fragment (not a question) whose single most natural continuation is the answer.
Vary the angle: history, geography, people, facts, uses, appearance, culture. NEVER include the answer's words or obvious
giveaways like its distinctive name parts. All prompts must end so the answer comes next (e.g. "... is called the").
Rules for "controls": 8 fragments with OTHER single-word-ish answers, each starting with a space: 4 closely related to the
concept's category (similar things the model must still get right) and 4 unrelated well-known facts."""


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


class Pipeline:
    def __init__(self, model_name, mode="clamp", reply_only=False):
        self.model_name = model_name
        self.model, self.tok = load(model_name)
        self.model.requires_grad_(False)
        self.L, self.W = self.model.config.num_hidden_layers, self.model.config.intermediate_size
        self.stats = torch.load(ROOT / f"data/stats_{model_name.split('/')[-1]}.pt")
        self.atlas = None
        self.mode, self.reply_only = mode, reply_only

    # ---- helpers -----------------------------------------------------------
    def answer_ids(self, answer):
        return self.tok.encode(answer)

    def seq_logprob(self, prompt, ans):
        """log P(answer tokens | prompt), teacher-forced: sum over the answer's pieces."""
        p_ids = self.tok.encode(prompt)
        ids = torch.tensor([p_ids + ans[:-1]], device="cuda")
        lp = self.model(input_ids=ids).logits[0, len(p_ids) - 1:].float().log_softmax(-1)
        return lp[torch.arange(len(ans)), torch.tensor(ans, device="cuda")].sum()

    def probs(self, prompts, ans, edits=()):
        out = []
        ctx = steer(self.model, list(edits)) if edits else torch.no_grad()
        with torch.no_grad(), ctx:
            for p in prompts:
                out.append(self.seq_logprob(p, ans).exp().item())
        return out

    # ---- 1-2. prompts ------------------------------------------------------
    def make_prompts(self, concept):
        # cached per concept so every method variant is tested on identical prompts
        cache = OUT / f"{slug(concept)}_spec.json"
        if cache.exists():
            spec = json.loads(cache.read_text(encoding="utf-8"))
        else:
            spec = oai.chat_json(f"Concept: {concept}", PROMPT_SYSTEM)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(spec, indent=1, ensure_ascii=False), encoding="utf-8")
        tid = self.answer_ids(spec["answer"])
        p = self.probs(spec["recall"], tid)
        recall = [(q, round(x, 3)) for q, x in zip(spec["recall"], p) if x >= MIN_P]
        # diagnostics (logging only): every prompt's P(answer) and the model's actual top-5 next tokens
        self.recall_diag = []
        with torch.no_grad():
            for q, x in zip(spec["recall"], p):
                ids = self.tok(q, return_tensors="pt").to("cuda")
                top = self.model(**ids).logits[0, -1].float().softmax(-1).topk(5)
                self.recall_diag.append({"prompt": q, "p_answer": round(x, 4),
                                         "top5": [(self.tok.decode([i]), round(v, 3)) for v, i in zip(top.values.tolist(), top.indices.tolist())]})
        controls = []
        for c in spec["controls"]:
            ct = self.answer_ids(c["answer"])
            x = self.probs([c["prompt"]], ct)[0]
            if x >= 0.5:
                controls.append({"prompt": c["prompt"], "answer": c["answer"], "tid": ct, "p": round(x, 3)})
        return spec, tid, recall, controls

    # ---- 3. attribution ----------------------------------------------------
    def attribute(self, prompts, tid):
        """activation x gradient for every MLP neuron and every attention head (head = its slice of
        o_proj's input), per prompt and per token position. Big per-position maps are kept on CPU."""
        H = self.model.config.num_attention_heads
        acts, heads, hooks = {}, {}, []
        hooks.append(self.model.model.embed_tokens.register_forward_hook(lambda m, a, o: o.requires_grad_(True)))
        for l in range(self.L):
            def keep(m, args, l=l):
                args[0].retain_grad(); acts[l] = args[0]

            def keep_h(m, args, l=l):
                args[0].retain_grad(); heads[l] = args[0]
            hooks.append(mlps(self.model)[l].down_proj.register_forward_pre_hook(keep))
            hooks.append(self.model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(keep_h))
        attr = torch.zeros(self.L, self.W, device="cuda")
        head_attr = torch.zeros(self.L, H, device="cuda")
        last_act = torch.zeros(self.L, self.W, device="cuda")
        per_prompt, pos_maps, head_pos, tokens = [], [], [], []
        try:
            for p in prompts:
                with torch.enable_grad():
                    lp = self.seq_logprob(p, tid)
                    lp.backward()
                n_p = len(self.tok.encode(p))
                pa = torch.stack([(acts[l].float() * acts[l].grad.float())[0] for l in range(self.L)])   # [L, T, W]
                ha = torch.stack([(heads[l].float() * heads[l].grad.float())[0].view(heads[l].shape[1], H, -1).sum(-1)
                                  for l in range(self.L)])                                                # [L, T, H]
                attr += pa.sum(1) / len(prompts)
                head_attr += ha.sum(1) / len(prompts)
                last_act += torch.stack([acts[l][0, n_p - 1].float() for l in range(self.L)]) / len(prompts)
                per_prompt.append(pa.sum(1).cpu())
                pos_maps.append(pa.half().cpu())
                head_pos.append(ha.cpu())
                ids = self.tok.encode(p) + tid[:-1]
                tokens.append([self.tok.decode([t]) for t in ids])
                acts.clear(); heads.clear()
        finally:
            for h in hooks:
                h.remove()
        top = [divmod(i.item(), self.W) for i in attr.flatten().argsort(descending=True)[:200]]
        bottom = [divmod(i.item(), self.W) for i in attr.flatten().argsort()[:50]]
        top_heads = [divmod(i.item(), H) for i in head_attr.flatten().argsort(descending=True)[:50]]
        return {"top": top, "bottom": bottom, "attr": attr, "last_act": last_act, "per_prompt": per_prompt,
                "pos_maps": pos_maps, "tokens": tokens, "head_attr": head_attr, "head_pos": head_pos,
                "top_heads": top_heads, "H": H}

    @contextlib.contextmanager
    def ablate_heads(self, hs):
        """Zero chosen heads' outputs (their slice of o_proj's input) at every position."""
        H = self.model.config.num_attention_heads
        by_layer, handles = {}, []
        for l, h in hs:
            by_layer.setdefault(l, []).append(h)
        for l, hl in by_layer.items():
            def hook(m, args, hl=hl):
                x = args[0].clone()
                d = x.shape[-1] // H
                for h in hl:
                    x[..., h * d:(h + 1) * d] = 0
                return (x,)
            handles.append(self.model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(hook))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def head_looks(self, prompts, heads, n_src=3):
        """Where the top heads look from the last prompt token (needs eager attention to return weights)."""
        try:
            self.model.set_attn_implementation("eager")
        except Exception as e:  # noqa: BLE001
            return {"error": f"could not switch to eager attention: {e!r}"}
        out = {}
        try:
            for p in prompts[:8]:
                ids = self.tok(p, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    att = self.model(**ids, output_attentions=True).attentions      # L x [1, H, T, T]
                toks = [self.tok.decode([t]) for t in ids.input_ids[0].tolist()]
                for l, h in heads:
                    w = att[l][0, h, -1].float()
                    tv = w.topk(min(n_src, len(toks)))
                    out.setdefault(f"L{l}H{h}", []).append(
                        [(toks[i], round(v, 3)) for v, i in zip(tv.values.tolist(), tv.indices.tolist())])
        except Exception as e:  # noqa: BLE001
            out["error"] = repr(e)
        finally:
            try:
                self.model.set_attn_implementation("sdpa")
            except Exception:  # noqa: BLE001
                pass
        return out

    # ---- 4. ablation -------------------------------------------------------
    def ablation(self, top, recall, tid, controls):
        rp = [q for q, _ in recall]
        base = sum(self.probs(rp, tid)) / len(rp)
        curve, k50 = [], None
        for k in KS:
            edits = [(l, n, 0.0) for l, n in top[:k]]
            pt = sum(self.probs(rp, tid, edits)) / len(rp)
            rng = random.Random(k)
            pr = sum(self.probs(rp, tid, [(l, rng.randrange(self.W), 0.0) for l, _ in top[:k]])) / len(rp)
            ctrl = {c["answer"]: round(self.probs([c["prompt"]], c["tid"], edits)[0], 3) for c in controls}
            curve.append({"k": k, "p_answer": round(pt, 3), "p_answer_random_ablation": round(pr, 3), "controls": ctrl})
            if k50 is None and pt <= base / 2:
                k50 = k
        return round(base, 3), curve, k50

    # ---- 5. labels ---------------------------------------------------------
    def labels(self, top, concept_slug):
        import label as lab
        if self.atlas is None:
            self.atlas = Atlas(self.model_name)
        units = top[:N_LABEL]
        d = OUT / f"dossiers_{concept_slug}"
        key = self.atlas.write_dossiers(units, d)
        out = {}
        from concurrent.futures import ThreadPoolExecutor
        lab.random_fit("warm-up")  # load shared random snippets before threads start
        with ThreadPoolExecutor(len(key)) as ex:     # all units labelled at once (API-bound)
            labelled = dict(zip(key, ex.map(lambda c: lab.label_unit(d / f"{c}.md"), key)))
        for code, k in key.items():
            r = labelled[code]
            l, n = map(int, re.findall(r"\d+", k["unit"]))
            up, down = output_words(self.model, self.tok, l, n)
            out[k["unit"]] = {"pos": r["luna"].get("pos"), "neg": r["luna"].get("neg"),
                              "clean_pos": r.get("clean_pos"), "clean_neg": r.get("clean_neg"),
                              "jev_covered_pos": r.get("jev_covered_pos"), "jev_covered_neg": r.get("jev_covered_neg"),
                              "random_fit_pos": r.get("random_fit_pos"), "random_fit_neg": r.get("random_fit_neg"),
                              "writes_up": up, "writes_down": down}
        return out

    # ---- 6. steering -------------------------------------------------------
    def judge_detail(self, replies, concept):
        from concurrent.futures import ThreadPoolExecutor
        q = {"on": {"type": "noul", "instructions": f"Does this text mention, describe, or seem preoccupied with {concept}?"},
             "fluent": {"type": "noul", "instructions": "Is this text fluent, grammatical English that makes sense?"}}
        with ThreadPoolExecutor(40) as ex:
            ans = list(ex.map(lambda r: jev.ask(r or "(empty)", q), replies))
        return [round(a["on"]["noul"], 3) for a in ans], [round(a["fluent"]["noul"], 3) for a in ans]

    def judge(self, replies, concept):
        on, fl = self.judge_detail(replies, concept)
        return sum(x >= 0.5 for x in on) / len(on), sum(x >= 0.5 for x in fl) / len(fl)

    def steer_edits(self, units, strength, last_act):
        edits = []
        for l, n in units:
            neg = last_act[l, n].item() < 0          # the sign this neuron takes when the model recalls the concept
            peak = (self.stats["min"] if neg else self.stats["max"])[l, n].item()
            edits.append((l, n, strength * peak))
        return edits

    def run_steer(self, units, strength, last_act, concept, detail=False):
        edits = self.steer_edits(units, strength, last_act)
        with steer(self.model, edits, mode=self.mode, reply_only=self.reply_only):
            replies = generate(self.model, self.tok, NEUTRAL, 60)
        on_p, fl_p = self.judge_detail(replies, concept)
        on, fl = sum(x >= 0.5 for x in on_p) / len(on_p), sum(x >= 0.5 for x in fl_p) / len(fl_p)
        if detail:
            return on, fl, replies, on_p, fl_p
        return on, fl, replies

    def steering(self, top, last_act, concept):
        late = [(l, n) for l, n in top if l >= self.L // 2]
        grid, best = [], None
        for n_units in N_STEER:
            units = late[:n_units]
            for s in STRENGTHS:
                on, fl, replies, on_p, fl_p = self.run_steer(units, s, last_act, concept, detail=True)
                row = {"n": n_units, "strength": s, "units": units, "on_topic": on, "fluent": fl,
                       "score": round(on * fl, 3), "sample": replies[:3],
                       "replies": replies, "jev_on_topic": on_p, "jev_fluent": fl_p}
                grid.append(row)
                if best is None or row["score"] > best["score"]:
                    best = row
        # random control at the best setting: same count, same layers, same strength x own peak
        rng = random.Random(0)
        rand_units = [(l, rng.randrange(self.W)) for l, _ in late[:best["n"]]]
        on, fl, replies, on_p, fl_p = self.run_steer(rand_units, best["strength"], last_act, concept, detail=True)
        return grid, best, {"units": rand_units, "on_topic": on, "fluent": fl, "score": round(on * fl, 3),
                            "replies": replies, "jev_on_topic": on_p, "jev_fluent": fl_p}

    # ---- all ---------------------------------------------------------------
    def run(self, concept, do_labels=True):
        t0 = time.time()
        s = slug(concept)
        spec, tid, recall, controls = self.make_prompts(concept)
        row = {"concept": concept, "model": self.model_name, "answer": spec["answer"],
               "answer_tokens": [self.tok.decode([t]) for t in tid], "recall_kept": len(recall),
               "recall_total": len(spec["recall"]), "recall": recall, "controls": controls,
               "steer_mode": self.mode, "reply_only": self.reply_only,
               "recall_diagnostics": getattr(self, "recall_diag", None)}
        if len(recall) < 5:
            row["status"] = "model_does_not_recall"
            return self.save(s, row, t0)
        rp = [q for q, _ in recall]
        A = self.attribute(rp, tid)
        top, attr, last_act = A["top"], A["attr"], A["last_act"]
        la = last_act.cpu()
        peak = torch.where(la < 0, self.stats["min"], self.stats["max"]).abs().clamp(min=1e-3)
        frac = la.abs() / peak

        def nrec(l, n, i):
            return {"rank": i + 1, "layer": l, "neuron": n, "attribution": round(attr[l, n].item(), 4),
                    "sign_on_recall": "neg" if la[l, n] < 0 else "pos",
                    "frac_of_peak_on_recall": round(frac[l, n].item(), 3),
                    "per_prompt": [round(pp[l, n].item(), 4) for pp in A["per_prompt"]]}
        row["top_neurons"] = [nrec(l, n, i) for i, (l, n) in enumerate(top)]            # top 200
        row["bottom_neurons"] = [nrec(l, n, i) for i, (l, n) in enumerate(A["bottom"])]  # pushing AGAINST the answer

        def profile(maps, idx):
            per = []
            for pm, toks in zip(maps, A["tokens"]):
                v = pm[idx].float()
                mx = v.abs().max().item() or 1.0
                per.append([(toks[t], round(v[t].item(), 4)) for t in range(len(toks)) if abs(v[t].item()) >= 0.1 * mx])
            return per
        # which token each top-20 neuron / top-10 head does its work at, per prompt
        row["neuron_positions"] = {f"L{l}_N{n}": profile(A["pos_maps"], (l, slice(None), n)) for l, n in top[:20]}
        H = A["H"]
        row["top_heads"] = [{"rank": i + 1, "layer": l, "head": h, "attribution": round(A["head_attr"][l, h].item(), 4)}
                            for i, (l, h) in enumerate(A["top_heads"])]
        row["head_positions"] = {f"L{l}H{h}": profile(A["head_pos"], (l, slice(None), h)) for l, h in A["top_heads"][:10]}
        row["baseline_p"], row["ablation"], row["k50"] = self.ablation(top, recall, tid, controls)
        row["per_prompt_baseline"] = [round(x, 4) for x in self.probs(rp, tid)]
        # each top-20 neuron ablated alone (checks the straight-line attribution estimate)
        row["single_ablation"] = [{"layer": l, "neuron": n,
                                   "p_answer": round(sum(self.probs(rp, tid, [(l, n, 0.0)])) / len(rp), 4)}
                                  for l, n in top[:20]]
        # head ablation curve vs random heads
        hcurve, hk50 = [], None
        for k in [1, 2, 5, 10, 20, 50]:
            with self.ablate_heads(A["top_heads"][:k]):
                pt = sum(self.probs(rp, tid)) / len(rp)
                hctrl = {c["answer"]: round(self.probs([c["prompt"]], c["tid"])[0], 3) for c in controls}
            rng = random.Random(100 + k)
            with self.ablate_heads([(l, rng.randrange(H)) for l, _ in A["top_heads"][:k]]):
                pr = sum(self.probs(rp, tid)) / len(rp)
            hcurve.append({"k": k, "p_answer": round(pt, 3), "p_answer_random_ablation": round(pr, 3), "controls": hctrl})
            if hk50 is None and pt <= row["baseline_p"] / 2:
                hk50 = k
        row["head_ablation"], row["head_k50"] = hcurve, hk50
        row["head_looks"] = self.head_looks(rp, A["top_heads"][:5])
        del A
        if do_labels:
            row["labels"] = self.labels(top, s)
        row["steering_grid"], row["steering_best"], row["steering_random_control"] = self.steering(top, last_act, concept)
        row["nudgeability"] = row["steering_best"]["score"]
        row["status"] = "ok"
        return self.save(s, row, t0)

    def save(self, s, row, t0):
        row["seconds"] = round(time.time() - t0)
        import transformers
        src = ROOT / "src"
        head = ROOT / ".git" / "HEAD"
        row["provenance"] = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "code_sha256": {f: hashlib.sha256((src / f).read_bytes()).hexdigest()[:16]
                            for f in ("concept_pipeline.py", "neurons.py", "label.py", "atlas.py", "jev.py")},
            "git_head": head.read_text().strip() if head.exists() else None,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "atlas_tokens": self.atlas.a["tokens"] if self.atlas else None}
        row["cost"] = {"jev_usd": round(jev.cost(), 4), "luna_tokens": dict(oai.usage)}
        tag = f"{self.mode}{'_reply' if self.reply_only else ''}"
        out = OUT / f"{s}__{tag}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(row, indent=1, ensure_ascii=False), encoding="utf-8")
        return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("concepts", nargs="+")
    ap.add_argument("--no-labels", action="store_true"); ap.add_argument("--reply-only", action="store_true")
    ap.add_argument("--mode", default="clamp", choices=["clamp", "add"])
    ap.add_argument("--force", action="store_true", help="redo concepts that already have a result")
    a = ap.parse_args()
    ap_force = getattr(a, "force", False)
    pipe = Pipeline(a.model, mode=a.mode, reply_only=a.reply_only)
    tag = f"{a.mode}{'_reply' if a.reply_only else ''}"

    def done(c):
        f = OUT / f"{slug(c)}__{tag}.json"
        if not f.exists():
            return False
        st = json.loads(f.read_text(encoding="utf-8")).get("status")
        return st in ("ok", "model_does_not_recall")

    failed = []
    for c in a.concepts:
        if done(c) and not ap_force:
            print(f"=== {c}: already done, skipping (use --force to redo)", flush=True)
            continue
        try:
            r = pipe.run(c, do_labels=not a.no_labels)
        except Exception as e:  # noqa: BLE001  one concept failing must not stop the rest
            print(f"=== {c}: FAILED {e!r} (will retry once at the end)", flush=True)
            failed.append(c)
            continue
        print(f"\n=== {c}: {r['status']} | recall prompts kept {r['recall_kept']}/{r['recall_total']} | "
              f"baseline P {r.get('baseline_p')} | k50 {r.get('k50')} | nudgeability {r.get('nudgeability')} "
              f"(random control {r.get('steering_random_control', {}).get('score')}) | {r['seconds']}s", flush=True)
        if r["status"] == "ok":
            b = r["steering_best"]
            print(f"    best steer: n={b['n']} at {b['strength']}x -> on-topic {b['on_topic']:.0%}, fluent {b['fluent']:.0%}")
            for x in b["sample"][:2]:
                print("     ", repr(x[:160]))
    for c in failed:
        try:
            r = pipe.run(c, do_labels=not a.no_labels)
            print(f"=== {c}: retry {r['status']} | k50 {r.get('k50')} | nudgeability {r.get('nudgeability')}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"=== {c}: FAILED AGAIN {e!r}; re-run the same command later to pick it up", flush=True)


if __name__ == "__main__":
    main()
