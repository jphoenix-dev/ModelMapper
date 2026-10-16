"""The dial: pick a concept, find its neurons, then talk to Qwen3-8B with the concept turned up.

Interactive (type at the prompt):
  concept <name>     find neurons for a concept (reuses saved prompts in results/concepts_dial; a new concept costs
                     ~2 Luna calls to write recall questions + answer discovery; no Jev)
  rule <R1|R2|R4>    which neurons to steer with: R1 late-layer attribution (default), R2 any layer, R4 "says the answer"
  n <k>              how many neurons (default 4)
  strength <x>       multiple of each neuron's own peak (default 1.0; 0 = off). Fluency usually breaks above ~2-3
  mode <add|clamp>   add = nudge on top of natural activity (default, gentler); clamp = force the value
  show               list the chosen neurons (layer:index, attribution rank)
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

    print(__doc__.split("Steering is")[0])
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
