"""Indirect path: do 'say it' neurons write the answer themselves, and do attribute neurons work THROUGH them?
(GPU; no API)

After job_overlap.py: most concepts have one late 'say it' neuron (the answer in the top-10 of its direct push) plus
1-3 attribute neurons that raise the answer more weakly. bowtie.py knocked out a concept's top-5 as a GROUP; this
splits it per neuron and asks whether the parts feed each other.
Per concept (every mapped concept), every recall prompt, answer position = the last prompt token:
  pass 1 (batch 1): base activations of every layer that holds a top-5 neuron (to pick the nulls)
  pass 2 (one batch, identical rows): row 0 as is; rows 1-5 one top-5 neuron zeroed at EVERY position (as in the
    maps); rows 6-10 for each of those, an ACTIVATION-MATCHED null: the neuron of the same layer, outside the concept's
    top-30, whose |activation| at the answer position is closest to it
  per row: answer score (answer logit - mean logit) and answer log-odds at the answer position; final rms; the
    activations of all top-5 neurons at the answer position
  direct_i = what neuron i wrote at the answer position, read through the final norm (row 0's rms) into the answer
    score; total_i = score(row 0) - score(row i); indirect_i = total_i - direct_i
Neuron types (from job_overlap.json / job_overlap_z.json): 'say' = late and answer in its top-10; 'attribute' = late,
not 'say', answer z >= 2; 'late-other' = late, z < 2; 'early' = layer < 20.
Predictions (stated before running; exploratory):
  (a) 'say' neurons act mostly directly (median direct/total >= 0.7) and more directly than attribute neurons.
  (b) zeroing an attribute neuron lowers a LATER 'say' neuron's activation (same concept) more than zeroing its
      activation-matched null: median relative drop >= 2x the null's and sign test p < 0.05.
Output: results/indirect_path.jsonl (per concept and prompt, everything above), results/indirect_path_gate.txt.
Usage: python src/indirect_path.py [max_concepts=all]
"""
import glob
import json
import sys
import time
from pathlib import Path

import torch

from config import DEFAULT_MODEL
from neurons import DEVICE, load, mlps

ROOT = Path(__file__).resolve().parents[1]
LATE = 20
OUT = ROOT / "results/indirect_path.jsonl"


def concepts():
    """Concepts whose map saved its recall prompts (rank_save fame3 / fame3m / fame5 = 555 of the 914; the 359 older
    maps did not store prompts - a stated limit, recorded in the gate file, not a silent cap)."""
    out, seen = {}, set()
    for f in sorted(glob.glob(str(ROOT / "results/concepts_*/rank_save_*.json"))):
        if "rank_save_test" in f or f.endswith("_exact.json"):
            continue
        for c, v in json.loads(Path(f).read_text(encoding="utf-8")).items():
            if "top" in v and c not in seen:
                seen.add(c)
                if "prompts" not in v:
                    continue
                out[c] = {"answer": v["answer_token"], "prompts": v["prompts"], "base": v["base_per_prompt"],
                          "top5": [e["neuron"] for e in v["top"][:5]], "top30": {e["neuron"] for e in v["top"][:30]}}
    return out


def types(C):
    jo = json.loads((ROOT / "results/job_overlap.json").read_text(encoding="utf-8"))
    jz = json.loads((ROOT / "results/job_overlap_z.json").read_text(encoding="utf-8"))["per_concept_z"]
    T = {}
    for c, v in C.items():
        t = {}
        late = [u for u in v["top5"] if int(u.split(":")[0][1:]) >= LATE]
        ranks = dict(zip(jo[c].get("neurons", []), jo[c].get("ranks", [])))
        zs = dict(zip(late, jz.get(c, [])[:-1])) if c in jz else {}
        for u in v["top5"]:
            if u not in late:
                t[u] = "early"
            elif ranks.get(u, 10 ** 9) <= 10:
                t[u] = "say"
            elif zs.get(u, 0) >= 2:
                t[u] = "attribute"
            else:
                t[u] = "late-other"
        T[c] = t
    return T


def main():
    maxc = int(sys.argv[1]) if len(sys.argv) > 1 else None
    C = concepts(); TY = types(C)
    model, tok = load(DEFAULT_MODEL)
    WU = model.lm_head.weight.detach(); mean_u = WU.float().mean(0); g = model.model.norm.weight.detach().float()
    eps = model.config.rms_norm_eps
    done = set()
    if OUT.exists():
        done = {json.loads(l)["concept"] for l in OUT.read_text(encoding="utf-8").splitlines()}
    keys = [c for c in C if c not in done][:maxc] if maxc else [c for c in C if c not in done]
    f = open(OUT, "a", encoding="utf-8"); t0 = time.time(); gate_bad = 0; nprom = 0
    for ci, c in enumerate(keys):
        v = C[c]; aid = tok.encode(v["answer"]); assert len(aid) == 1 and tok.decode(aid) == v["answer"], c
        aid = aid[0]; units = [(int(u.split(":")[0][1:]), int(u.split(":")[1])) for u in v["top5"]]
        layers = sorted({L for L, _ in units})
        top30 = {(int(u.split(":")[0][1:]), int(u.split(":")[1])) for u in v["top30"]}
        u_dir = WU[aid].float() - mean_u
        rows_out = []
        for pi, prompt in enumerate(v["prompts"]):
            ids = torch.tensor(tok.encode(prompt), device=DEVICE)[None]; P = ids.shape[1]
            # pass 1: base activations at the answer position for the layers involved
            acts = {}
            hks = [mlps(model)[L].down_proj.register_forward_pre_hook(lambda m, a, L=L: acts.__setitem__(L, a[0][0, -1].float().clone())) for L in layers]
            try:
                with torch.no_grad():
                    model(input_ids=ids)
            finally:
                for h in hks:
                    h.remove()
            nulls = []
            for L, n in units:
                a = acts[L].abs(); a[n] = float("inf")
                for (L2, n2) in top30:
                    if L2 == L:
                        a[n2] = float("inf")
                nulls.append((L, int((a - acts[L][n].abs()).abs().argmin())))
            kills = units + nulls; R = 1 + len(kills)
            rec = {}; fin = {}

            def mk(L):
                def h(m, a):
                    x = a[0].clone()
                    for r, (Lk, nk) in enumerate(kills, start=1):
                        if Lk == L:
                            x[r, :, nk] = 0.0
                    rec[L] = x[:, -1].float().clone()
                    return (x,)
                return h
            hks = [mlps(model)[L].down_proj.register_forward_pre_hook(mk(L)) for L in layers]
            hks.append(model.model.norm.register_forward_pre_hook(lambda m, a: fin.__setitem__("x", a[0][:, -1].float().clone())))
            try:
                with torch.no_grad():
                    lg = model(input_ids=ids.repeat(R, 1)).logits[:, -1].double()
            finally:
                for h in hks:
                    h.remove()
            lp = lg.log_softmax(-1)[:, aid]
            logodds = (lp - torch.log(-torch.expm1(lp))).tolist()
            score = (lg[:, aid] - lg.mean(-1)).tolist()
            rms = (fin["x"].pow(2).mean(-1) + eps).sqrt()
            unit_acts = [[rec[L][r, n].item() for L, n in units] for r in range(R)]
            direct = []
            for (L, n) in units:
                w = rec[L][0, n] * mlps(model)[L].down_proj.weight[:, n].detach().float()   # detach: same value, no autograd graph
                direct.append(float((w * g / rms[0]) @ u_dir))
            # saved baselines are WHOLE-answer log-odds (all answer tokens); this test scores the FIRST answer token, whose
            # probability is always >= the whole answer's, so the check is an inequality (0.5 = bf16 batch-shape tolerance)
            gate_bad += any(abs(rec[L][1 + k, n].item()) > 0 for k, (L, n) in enumerate(units)) or logodds[0] < v["base"][pi] - 0.5
            nprom += 1
            rows_out.append({"prompt_i": pi, "score": score, "logodds": logodds, "unit_acts": unit_acts, "direct": direct,
                             "nulls": [f"L{L}:{n}" for L, n in nulls], "null_base_act": [acts[L][n].item() for L, n in nulls]})
        f.write(json.dumps({"concept": c, "units": v["top5"], "types": [TY[c][u] for u in v["top5"]], "prompts": rows_out}) + "\n")
        f.flush()
        if ci % 50 == 0:
            el = time.time() - t0
            print(f"{ci + 1}/{len(keys)} concepts, {nprom} prompts, {el / 60:.1f} min; gate failures so far {gate_bad}", flush=True)
    msg = (f"scope: {len(C)} concepts whose maps saved their recall prompts (the 359 older maps did not)\n"
           f"gate - a zeroed unit not 0 at the answer position, or as-is FIRST-token log-odds below the saved WHOLE-answer log-odds - 0.5 "
           f"(bf16 batch-shape tolerance): {gate_bad} of {nprom} prompts")
    (ROOT / "results/indirect_path_gate.txt").write_text(msg + "\n", encoding="utf-8")
    print(msg)


if __name__ == "__main__":
    main()
