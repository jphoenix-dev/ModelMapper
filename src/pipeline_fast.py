"""Faster concept pipeline (v2): same measurements as concept_pipeline.py, far better GPU use.

What changes (speed and one metric fix; judgements unchanged):
  * Batched scoring: recall checks, controls and every ablation condition run as one padded batch
    instead of one prompt at a time.
  * Whole steering grid in one go: a per-row clamp hook lets all (n, strength) settings x 20 prompts
    generate together in a few large batches.
  * Overlap: Luna prompt specs for all concepts are prefetched in parallel; neuron labelling runs in a
    background thread while the GPU does ablation and steering; Jev judges all replies concurrently.
  * Metric fix (ISSUES #13): nudgeability = share of replies that are BOTH on-topic and fluent.

Usage: python src/pipeline_fast.py <model> "<concept>" ["<concept>" ...] [--mode clamp] [--reply-only] [--force]
"""
import argparse
import contextlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor

import torch

import concept_pipeline as cp
from concept_pipeline import OUT, Pipeline, slug
from neurons import mlps
from steer_test import PROMPTS as NEUTRAL

GEN_TOKENS = 60
GEN_BATCH = 100        # rows per generate() call (settings x prompts)


import os
import re

RECALL_VARIANT = os.environ.get("RECALL_VARIANT", "V0")   # V0 as frozen run, V1 bridge, V2 rewritten, V3 question
BRIDGES = ["", " called", " known as", " named", " the"]


GENERIC_ANSWER_WORDS = {"tree", "bridge", "tower", "mountain", "river", "desert", "city", "lake", "island", "plant",
                        "animal", "game", "machine", "engine", "table", "wall", "game", "instrument", "building"}


def leaks(prompt, answer):
    """Does the prompt give away a DISTINCTIVE answer word? (ISSUES #22)
    Capitalised answer words (names: Golden, Eiffel, Shakespeare) count only when capitalised in the prompt, so an
    ordinary "the bridge" is not a leak for "Golden Gate Bridge"; lowercase answers (coffee, chess) count in any case."""
    for w in re.findall(r"[A-Za-z]{4,}", answer):
        if w.lower() in GENERIC_ANSWER_WORDS and not w[0].isupper():
            continue                       # "tree" in "oak tree" is the question's category, not a give-away (ISSUES #32)
        if w[0].isupper():
            if re.search(r"\b" + re.escape(w) + r"\b", prompt):
                return True
        elif re.search(r"\b" + re.escape(w) + r"\b", prompt.lower()):
            return True
    return False


class FastPipeline(Pipeline):
    # ---- recall prompt variants + leak filter --------------------------------
    def make_prompts(self, concept):
        """Same as v1, plus: leaky prompts dropped; RECALL_VARIANT picks the prompt format (ISSUES #9/#10)."""
        spec, tid, recall, controls = super().make_prompts(concept)
        if RECALL_VARIANT == "V0":
            recall = [(q, p) for q, p in recall if not leaks(q, spec["answer"])]
            return spec, tid, recall, controls
        if "recall_v2" not in spec:                             # generate the V2/V3 rewrites if missing
            import recall_variants as rv
            listing = chr(10).join(f"{i+1}. {p}" for i, p in enumerate(spec["recall"]))
            out = cp.oai.chat_json(f"Answer (for your reference only, never write it): {spec['answer']}{chr(10)}"
                                   f"Fragments:{chr(10)}{listing}", rv.REWRITE)
            items = out.get("items", [])
            if len(items) == len(spec["recall"]):
                spec["recall_v2"] = [it["v2"] for it in items]; spec["recall_v3"] = [it["v3"] for it in items]
                (OUT / f"{slug(concept)}_spec.json").write_text(json.dumps(spec, indent=1, ensure_ascii=False), encoding="utf-8")
        if RECALL_VARIANT == "V1":
            cands = []
            for q in spec["recall"]:
                best = max(((q + b, self.batch_probs([(q + b, tid)])[0]) for b in BRIDGES), key=lambda x: x[1])
                cands.append(best)
        elif RECALL_VARIANT == "V2":
            ps = self.batch_probs([(q, tid) for q in spec["recall_v2"]])
            cands = list(zip(spec["recall_v2"], ps))
        else:  # V3: chat question, answer scored at the start of the reply (no leading space)
            chat = [self.tok.apply_chat_template([{"role": "user", "content": q + " Answer with just the name."}],
                                                 tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    for q in spec["recall_v3"]]
            # the answer opens the reply, so Qwen writes it Capitalised and without a leading "the" ("Coffee",
            # "French Revolution"): score each natural form and keep whichever the model actually produces
            if "answer_forms_extra" not in spec and os.environ.get("ANSWER_DISCOVERY", "1") == "1":
                spec["answer_forms_extra"], spec["discovered_answers"] = self.discover_answers(spec["answer"].strip(), chat, spec["recall_v3"])
                (OUT / f"{slug(concept)}_spec.json").write_text(json.dumps(spec, indent=1, ensure_ascii=False), encoding="utf-8")
            a = spec["answer"].strip()
            base = a[4:].strip() if a.lower().startswith("the ") else a
            forms = list(dict.fromkeys([a, a[:1].upper() + a[1:], a.title(), base, base[:1].upper() + base[1:]]
                                       + spec.get("answer_forms_extra", [])))      # model's own wordings (ISSUES #29)
            best = None
            for f in forms:
                t_ = self.tok.encode(f)
                ps_ = self.batch_probs([(c, t_) for c in chat])
                score = sum(p >= cp.MIN_P for p in ps_)
                if best is None or score > best[0]:
                    best = (score, t_, ps_, f)
            _, tid, ps, self.answer_form = best
            cands = list(zip(chat, ps))
        src = {"V1": spec["recall"], "V2": spec["recall_v2"], "V3": spec["recall_v3"]}[RECALL_VARIANT]
        recall = [(q, round(p, 3)) for (q, p), raw in zip(cands, src) if p >= cp.MIN_P and not leaks(raw, spec["answer"])]
        self.recall_diag = [{"prompt": q, "p_answer": round(p, 4)} for q, p in cands]
        return spec, tid, recall, controls


    def discover_answers(self, name, chat, questions):
        """ISSUES #29: let the model answer each recall prompt (greedy, unsteered), keep replies Luna says name the
        concept, and return their most common wordings as extra answer forms (plus every reply and verdict, for audit).
        A made-up concept should yield no accepted replies, so this cannot manufacture recall by itself: the forms are
        still scored, and recall still needs P(answer) >= MIN_P on the prompts."""
        from collections import Counter
        self.tok.padding_side = "left"
        ids = self.tok(chat, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            g = self.model.generate(**ids, max_new_tokens=16, do_sample=False, pad_token_id=self.tok.pad_token_id)
        replies = [self.tok.decode(o[ids.input_ids.shape[1]:], skip_special_tokens=True).strip().split(chr(10))[0].strip().rstrip(".")
                   for o in g]
        try:
            ok = cp.oai.chat_json(
                f"Target name: {name}{chr(10)}For each numbered reply, is it this specific thing's own name (allowing "
                f"singular/plural, capitalisation, a well-known synonym or short name)? Say false for a different "
                f"thing, and false for a generic descriptive name made from place or category words (e.g. 'River X "
                f"Bridge', 'Town Y Museum') unless that IS the target's established name. "
                f"Reply JSON {{\"ok\": [true/false, ...]}} in order.{chr(10)}"
                + chr(10).join(f"{i+1}. {r}" for i, r in enumerate(replies)))["ok"]
        except Exception:  # noqa: BLE001
            ok = [False] * len(replies)
        # a wording whose distinctive words already appear in the questions is echoing the clues, not recall (ISSUES #31)
        good = [r for r, v in zip(replies, ok) if v and r and sum(leaks(q, r) for q in questions) < 0.3 * len(questions)]
        # drop one-off wordings (>= 2 replies). Recall itself still needs high P(answer) on >= 5 prompts, which is the
        # real recurrence test; a stricter cut here (>= 3) dropped "Sahara" and lost a real concept (ISSUES #31)
        return [f for f, n in Counter(good).most_common(3) if n >= 2], [{"reply": r, "names_concept": bool(v)} for r, v in zip(replies, ok)]

    # ---- batched teacher-forced scoring -------------------------------------
    def batch_probs(self, pairs, ctx=None):
        """pairs: [(prompt, answer_ids)] -> [P(answer | prompt)], one padded forward pass for all."""
        seqs, spans = [], []
        for prompt, ans in pairs:
            p_ids = self.tok.encode(prompt)
            seqs.append(p_ids + ans[:-1])
            spans.append((len(p_ids) - 1, ans))
        T = max(map(len, seqs))
        pad = self.tok.pad_token_id or 0
        ids = torch.full((len(seqs), T), pad, device="cuda")
        mask = torch.zeros_like(ids)
        for i, s in enumerate(seqs):                       # right padding: causal attention leaves real tokens untouched
            ids[i, :len(s)] = torch.tensor(s, device="cuda"); mask[i, :len(s)] = 1
        with torch.no_grad(), (ctx or contextlib.nullcontext()):
            lp = self.model(input_ids=ids, attention_mask=mask).logits.float().log_softmax(-1)
        out = []
        for i, (start, ans) in enumerate(spans):
            pos = torch.arange(start, start + len(ans), device="cuda")
            out.append(lp[i, pos, torch.tensor(ans, device="cuda")].sum().exp().item())
        return out

    def probs(self, prompts, ans, edits=()):
        from neurons import steer
        return self.batch_probs([(p, ans) for p in prompts], steer(self.model, list(edits)) if edits else None)

    # ---- batched ablation: recall prompts + controls in one pass per condition ----
    def ablation(self, top, recall, tid, controls):
        rp = [q for q, _ in recall]
        pairs = [(q, tid) for q in rp] + [(c["prompt"], c["tid"]) for c in controls]
        n = len(rp)
        from neurons import steer

        def run(edits):
            ps = self.batch_probs(pairs, steer(self.model, edits) if edits else None)
            return sum(ps[:n]) / n, {c["answer"]: round(p, 3) for c, p in zip(controls, ps[n:])}
        base, _ = run([])
        curve, k50 = [], None
        for k in cp.KS:
            pt, ctrl = run([(l, nn, 0.0) for l, nn in top[:k]])
            rng = random.Random(k)
            pr, _ = run([(l, rng.randrange(self.W), 0.0) for l, _ in top[:k]])
            curve.append({"k": k, "p_answer": round(pt, 3), "p_answer_random_ablation": round(pr, 3), "controls": ctrl})
            if k50 is None and pt <= base / 2:
                k50 = k
        return round(base, 3), curve, k50

    # ---- whole steering grid in a few batches --------------------------------
    @contextlib.contextmanager
    def steer_rows(self, units, values, active):
        """Per-row clamp: row r sets unit j to values[r, j] where active[r, j] (reply-only / clamp mode as configured)."""
        by_layer = {}
        for j, (l, n) in enumerate(units):
            by_layer.setdefault(l, []).append((j, n))
        handles = []
        for l, items in by_layer.items():
            js = torch.tensor([j for j, _ in items], device="cuda")
            idx = torch.tensor([n for _, n in items], device="cuda")

            def hook(m, args, js=js, idx=idx):
                x = args[0]
                if self.reply_only and x.shape[-2] > 1:
                    return None
                x = x.clone()
                rows = self._rows                             # rows in this generate() call
                v = values[rows][:, js].to(x.dtype)[:, None, :]   # [B, 1, U_l]
                a = active[rows][:, js][:, None, :]
                cur = x[..., idx]
                new = cur + v if self.mode == "add" else v.expand_as(cur)
                x[..., idx] = torch.where(a, new, cur)
                return (x,)
            handles.append(mlps(self.model)[l].down_proj.register_forward_pre_hook(hook))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def gen_rows(self, prompts, units, values, active):
        out = []
        texts = [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False) for p in prompts]
        self.tok.padding_side = "left"
        with self.steer_rows(units, values, active):
            for s in range(0, len(texts), GEN_BATCH):
                self._rows = torch.arange(s, min(s + GEN_BATCH, len(texts)), device="cuda")
                ids = self.tok(texts[s:s + GEN_BATCH], return_tensors="pt", padding=True).to("cuda")
                with torch.no_grad():
                    g = self.model.generate(**ids, max_new_tokens=GEN_TOKENS, do_sample=False,
                                            pad_token_id=self.tok.pad_token_id)
                out += [self.tok.decode(o[ids.input_ids.shape[1]:], skip_special_tokens=True).strip() for o in g]
        return out

    def steering(self, top, last_act, concept):
        late = [(l, n) for l, n in top if l >= self.L // 2][:max(cp.N_STEER)]
        peaks = torch.tensor([self.steer_edits([u], 1.0, last_act)[0][2] for u in late], device="cuda")
        settings = [(n, s) for n in cp.N_STEER for s in cp.STRENGTHS]
        rows_prompts, vals, act = [], [], []
        for n, s in settings:
            for p in NEUTRAL:
                rows_prompts.append(p)
                vals.append(peaks * s)
                act.append(torch.arange(len(late), device="cuda") < n)
        replies = self.gen_rows(rows_prompts, late, torch.stack(vals), torch.stack(act))
        on_p, fl_p = self.judge_detail(replies, concept)
        grid, best = [], None
        k = len(NEUTRAL)
        for i, (n, s) in enumerate(settings):
            o, f, r = on_p[i*k:(i+1)*k], fl_p[i*k:(i+1)*k], replies[i*k:(i+1)*k]
            joint = sum(a >= 0.5 and b >= 0.5 for a, b in zip(o, f)) / k
            row = {"n": n, "strength": s, "units": late[:n], "on_topic": sum(x >= 0.5 for x in o) / k,
                   "fluent": sum(x >= 0.5 for x in f) / k, "score": round(joint, 3),
                   "score_product_old": round((sum(x >= 0.5 for x in o) / k) * (sum(x >= 0.5 for x in f) / k), 3),
                   "sample": r[:3], "replies": r, "jev_on_topic": o, "jev_fluent": f}
            grid.append(row)
            if best is None or row["score"] > best["score"]:
                best = row
        rng = random.Random(0)
        rand_units = [(l, rng.randrange(self.W)) for l, _ in late[:best["n"]]]
        rpk = torch.tensor([self.steer_edits([u], 1.0, last_act)[0][2] for u in rand_units], device="cuda")
        rreplies = self.gen_rows(NEUTRAL, rand_units, (rpk * best["strength"]).expand(k, -1),
                                 torch.ones(k, len(rand_units), dtype=torch.bool, device="cuda"))
        ro, rf = self.judge_detail(rreplies, concept)
        rj = sum(a >= 0.5 and b >= 0.5 for a, b in zip(ro, rf)) / k
        return grid, best, {"units": rand_units, "on_topic": sum(x >= 0.5 for x in ro) / k,
                            "fluent": sum(x >= 0.5 for x in rf) / k, "score": round(rj, 3),
                            "replies": rreplies, "jev_on_topic": ro, "jev_fluent": rf}

    def judge_detail(self, replies, concept):
        q = {"on": {"type": "noul", "instructions": f"Does this text mention, describe, or seem preoccupied with {concept}?"},
             "fluent": {"type": "noul", "instructions": "Is this text fluent, grammatical English that makes sense?"}}
        with ThreadPoolExecutor(64) as ex:
            ans = list(ex.map(lambda r: cp.jev.ask(r or "(empty)", q), replies))
        return [round(a["on"]["noul"], 3) for a in ans], [round(a["fluent"]["noul"], 3) for a in ans]

    # ---- labels overlap with GPU work ----------------------------------------
    def labels(self, top, concept_slug):
        # run the API-bound labelling in a background thread; the GPU continues with steering meanwhile
        if not hasattr(self, "_bg"):
            self._bg = ThreadPoolExecutor(2)
        self.atlas = self.atlas or __import__("atlas").Atlas(self.model_name)
        return self._bg.submit(Pipeline.labels, self, top, concept_slug)

    def run(self, concept, do_labels=True):
        row = super().run(concept, do_labels=do_labels)
        return row

    def save(self, s, row, t0):
        if hasattr(row.get("labels"), "result"):         # resolve the background labelling before saving
            row["labels"] = row["labels"].result()
        row["pipeline"] = "fast_v2"
        row["recall_variant"] = RECALL_VARIANT
        row["answer_form_used"] = getattr(self, "answer_form", None)
        return super().save(s, row, t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("concepts", nargs="+")
    ap.add_argument("--no-labels", action="store_true"); ap.add_argument("--reply-only", action="store_true")
    ap.add_argument("--mode", default="clamp", choices=["clamp", "add"]); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    pipe = FastPipeline(a.model, mode=a.mode, reply_only=a.reply_only)
    # prefetch every concept's prompt spec from Luna in parallel (cached to disk)
    def spec(c):
        f = OUT / f"{slug(c)}_spec.json"
        if not f.exists():
            s = cp.oai.chat_json(f"Concept: {c}", cp.PROMPT_SYSTEM)
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(s, indent=1, ensure_ascii=False), encoding="utf-8")
    with ThreadPoolExecutor(32) as ex:
        list(ex.map(spec, a.concepts))
    tag = f"{a.mode}{'_reply' if a.reply_only else ''}"
    for c in a.concepts:
        f = OUT / f"{slug(c)}__{tag}.json"
        if f.exists() and not a.force and json.loads(f.read_text(encoding="utf-8")).get("status") in ("ok", "model_does_not_recall"):
            print(f"=== {c}: already done, skipping", flush=True)
            continue
        t0 = time.time()
        try:
            r = pipe.run(c, do_labels=not a.no_labels)
        except Exception as e:  # noqa: BLE001
            print(f"=== {c}: FAILED {e!r}", flush=True)
            continue
        print(f"=== {c}: {r['status']} | recall {r['recall_kept']}/{r['recall_total']} | k50 {r.get('k50')} | "
              f"nudgeability (joint) {r.get('nudgeability')} | random {r.get('steering_random_control', {}).get('score')} | "
              f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
