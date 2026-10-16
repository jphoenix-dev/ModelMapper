"""The dial: pick a concept, find its neurons, then talk to Qwen3-8B with the concept turned up.

Interactive (type at the prompt):
  concept <name>     find neurons for a concept (reuses saved prompts in results/concepts_dial; a new concept costs
                     ~2 Luna calls to write recall questions + answer discovery; no Jev)
  rule <R1|R2|R4>    which neurons to steer with: R1 late-layer attribution (default), R2 any layer, R4 "says the answer"
  n <k>              how many neurons (default 4)
  strength <x>       multiple of each neuron's own peak (default 1.0; 0 = off). Fluency usually breaks above ~2-3
  mode <add|clamp>   add = nudge on top of natural activity (default, gentler); clamp = force the value
  show               list the chosen neurons (layer:index, attribution rank)
  auto               try R1/R2/R4 x 2/4 neurons x 0.5-2x on 6 practice prompts; keep the setting whose replies most
                     often name the concept AND pass a free local fluency check (src/fluency_proxy.py, ~89% agreement
                     with the Jev judge); takes ~15 s (6 practice prompts, so the winning score is flattered)
  <anything else>    is sent as a chat message; you get the unsteered reply and the steered reply side by side
  quit

Steering is reply-only (the question is read normally). Greedy decoding, so the same input gives the same output.
Usage: python src/dial.py
"""
import os
import sys

os.environ.setdefault("RECALL_VARIANT", "V3")
os.environ.setdefault("CONCEPT_OUT_DIR", os.path.join(os.path.dirname(__file__), "..", "results", "concepts_dial"))

import torch  # noqa: E402

from pipeline_fast import FastPipeline  # noqa: E402


def main():
    print("loading Qwen3-8B ...", flush=True)
    pipe = FastPipeline("Qwen/Qwen3-8B", mode="add", reply_only=True)
    W_U = pipe.model.lm_head.weight.float(); g = pipe.model.model.norm.weight.float()
    st = {"rule": "R1", "n": 4, "strength": 1.0, "concept": None, "cands": {}, "last_act": None, "tokens": 120}

    def find(concept):
        spec, tid, recall, _ = pipe.make_prompts(concept)
        if len(recall) < 5:
            print(f"  the model doesn't reliably recall '{concept}' ({len(recall)}/20 prompts); steering may not work. "
                  "Try a more specific name.")
            if not recall:
                return
        A = pipe.attribute([q for q, _ in recall], tid)
        top = A["top"][:200]; first = tid[0]

        def says(u):
            l, n = u
            col = pipe.model.model.layers[l].mlp.down_proj.weight[:, n].float()
            sign = -1.0 if A["last_act"][l, n] < 0 else 1.0
            return sign * (W_U[first] @ (col * g)).item()
        st.update(concept=concept, last_act=A["last_act"], cands={
            "R1": [u for u in top if u[0] >= pipe.L // 2][:12], "R2": top[:12], "R4": sorted(top, key=lambda u: -says(u))[:12]})
        rank = {u: i + 1 for i, u in enumerate(top)}
        st["rank"] = rank
        st["answer"] = spec["answer"]
        print(f"  '{concept}': recalled on {len(recall)}/20 questions (answer '{spec['answer'].strip()}'). Ready.")

    def reply(text, strength):
        units = st["cands"][st["rule"]][:st["n"]]
        peaks = torch.tensor([pipe.steer_edits([u], 1.0, st["last_act"])[0][2] for u in units], device="cuda")
        vals = (peaks * strength)[None]
        act = torch.ones(1, len(units), dtype=torch.bool, device="cuda") if strength else torch.zeros(1, len(units), dtype=torch.bool, device="cuda")
        import pipeline_fast
        old = pipeline_fast.GEN_TOKENS; pipeline_fast.GEN_TOKENS = st["tokens"]
        try:
            return pipe.gen_rows([text], units, vals, act)[0]
        finally:
            pipeline_fast.GEN_TOKENS = old

    PRACTICE = ["What should I do this weekend?", "Tell me a fun fact.", "Describe a perfect afternoon.",
                "Write a two-sentence story.", "What's a good hobby to pick up?", "Tell me about someone you admire."]

    def fluent(text):
        """Free fluency proxy validated against Jev (RESULTS: 88.7% held-out agreement)."""
        ids = pipe.tok(text, return_tensors="pt").input_ids.to("cuda")
        if ids.shape[1] < 3:
            return False
        with torch.no_grad():
            lp = pipe.model(ids).logits.float().log_softmax(-1)[0, :-1].gather(-1, ids[0, 1:, None]).mean().item()
        toks = ids[0].tolist()
        return len(set(toks)) / len(toks) >= 0.64 and lp >= -2.3 and sum(ord(c) > 127 for c in text) / len(text) < 0.03

    def auto():
        import re
        words = [w.lower() for w in re.findall(r"[A-Za-z]{4,}", st["answer"])] or [st["answer"].strip().lower()]
        key = max(words, key=len)
        import time
        import pipeline_fast
        t0 = time.time()
        results = []
        strengths = (0.5, 1.0, 1.5, 2.0)
        old_tokens = pipeline_fast.GEN_TOKENS; pipeline_fast.GEN_TOKENS = 60      # practice replies: 60 tokens
        try:
            for rule in ("R1", "R2", "R4"):
                units = st["cands"][rule][:4]
                peaks = torch.tensor([pipe.steer_edits([u], 1.0, st["last_act"])[0][2] for u in units], device="cuda")
                rows, vals, act, settings = [], [], [], []
                for n in (2, 4):
                    for s_ in strengths:
                        settings.append((n, s_))
                        for p in PRACTICE:              # one batched generate per rule: all n x strengths x prompts
                            rows.append(p); vals.append(peaks * s_); act.append(torch.arange(len(units), device="cuda") < n)
                reps = pipe.gen_rows(rows, units, torch.stack(vals), torch.stack(act))
                k = len(PRACTICE)
                for i, (n, s_) in enumerate(settings):
                    rr = reps[i*k:(i+1)*k]
                    results.append((sum(key in r.lower() and fluent(r) for r in rr) / k, rule, n, s_))
        finally:
            pipeline_fast.GEN_TOKENS = old_tokens
        results.sort(reverse=True)
        score, rule, n, s_ = results[0]
        st.update(rule=rule, n=n, strength=s_)
        print(f"  best: {rule}, {n} neurons, x{s_}: {score:.0%} of practice replies name '{key}' and read fluently"
              f" ({time.time() - t0:.0f}s)")
        print("  runners-up: " + ", ".join(f"{r} n={k} x{v} {sc:.0%}" for sc, r, k, v in results[1:4]))
    while True:
        try:
            line = input(f"[{st['concept'] or 'no concept'} | {st['rule']} n={st['n']} x{st['strength']} {pipe.mode}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        cmd, _, arg = line.partition(" ")
        try:
            if cmd == "quit":
                break
            elif cmd == "concept":
                find(arg.strip())
            elif cmd == "rule" and arg.strip().upper() in ("R1", "R2", "R4"):
                st["rule"] = arg.strip().upper()
            elif cmd == "n":
                st["n"] = max(1, int(arg))
            elif cmd == "strength":
                st["strength"] = float(arg)
            elif cmd == "mode" and arg.strip() in ("add", "clamp"):
                pipe.mode = arg.strip()
            elif cmd == "auto":
                if st["concept"] is None:
                    print("  set a concept first:  concept coffee")
                else:
                    print("  searching (~15 s) ...", flush=True); auto()
            elif cmd == "show":
                for u in st["cands"].get(st["rule"], [])[:st["n"]]:
                    print(f"  L{u[0]}:{u[1]}  (attribution rank {st['rank'].get(u)})")
            elif st["concept"] is None:
                print("  set a concept first:  concept coffee")
            else:
                print("--- unsteered ---\n" + reply(line, 0.0))
                print(f"--- steered ({st['concept']}, x{st['strength']}) ---\n" + reply(line, st["strength"]))
        except Exception as e:  # noqa: BLE001
            print(f"  error: {e}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
