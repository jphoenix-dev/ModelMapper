"""H6: contrastive 'say' neurons - do say neurons push their word UP and its nearest rivals DOWN? (MECHANISMS.md H6)
Weights only, Qwen3-8B (no forward pass; CPU; reads only lm_head, the final norm and late down_proj columns).

Say neurons: units with role 'say' in loop v2 run 1 (ledger). Random late neurons: uniform over layers 28-35 among
units with no run-1 role. Up to N_EACH of each (seeded sample; the cap is stated, not silent: the full say list is
saved with which were sampled).
Per neuron: push on every token v = W_U (g * w) (w = its down_proj column, g = final norm weights); z = standardised
over the vocabulary. Its top token t = argmax v. Rivals = the 20 tokens nearest to t by cosine of unembedding rows,
excluding spelling variants of t (same text after strip + lowercase). Score = mean z of the rivals.
Contrastive = rival score < 0 (t pushed up, its nearest rivals pushed below the vocabulary average).
Pass rule (before running): H6 holds if say neurons are contrastive >= 3x as often as random late neurons and >= 10%.
Predictions (MECHANISMS.md, before running):
  P6a most say neurons push rivals UP: median rival z > 0
  P6b >= 10% of say neurons are contrastive, >= 3x the random late neurons' rate
Output: D:/already-clean-data/mechanisms/rival_push/{summary.json, per_neuron.jsonl}
Usage: python -X utf8 src/mechanisms/rival_push.py [N_EACH=400]
"""
import json
import random
import statistics as stx
import sys
import time
from pathlib import Path

from modelmapper.experiments.mechanisms.common import OUT, save  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from modelmapper.ledger.query import connect  # noqa: E402
from modelmapper.core.settings import MODELS_DIR  # noqa: E402  (after common, which defaults MODEL_MAPPER_MODELS)

M8 = Path(MODELS_DIR) / "Qwen3-8B"
N_EACH = int(sys.argv[1]) if len(sys.argv) > 1 else 400
NR = 20


def tensor(name, idx):
    f = idx["weight_map"][name]
    with safe_open(str(M8 / f), framework="pt") as fh:
        return fh.get_tensor(name)


def main():
    t0 = time.time()
    D = OUT / "rival_push"; D.mkdir(parents=True, exist_ok=True)
    idx = json.loads((M8 / "model.safetensors.index.json").read_text())
    tok = AutoTokenizer.from_pretrained(str(M8))
    con = connect()
    say = sorted({(r[0], r[1]) for r in con.execute(
        "SELECT layer, CAST(split_part(unit_key, ':', 2) AS INT) FROM evidence WHERE experiment_id='loop_v2_run1' "
        "AND evidence_type='role' AND value_text='say' AND unit_kind='neuron'").fetchall()})
    roled = {(r[0], r[1]) for r in con.execute(
        "SELECT DISTINCT layer, CAST(split_part(unit_key, ':', 2) AS INT) FROM evidence WHERE experiment_id='loop_v2_run1' "
        "AND evidence_type='role' AND unit_kind='neuron'").fetchall()}
    rng = random.Random(0)
    say_s = rng.sample(say, min(N_EACH, len(say)))
    rand = set()
    while len(rand) < N_EACH:
        u = (rng.randrange(28, 36), rng.randrange(12288))
        if u not in roled:
            rand.add(u)
    rand = sorted(rand)
    print(f"say neurons {len(say)} (sampled {len(say_s)}), random late {len(rand)} ({time.time() - t0:.0f}s)", flush=True)
    WU = tensor("lm_head.weight", idx)                     # bf16 [V, 4096]
    g = tensor("model.norm.weight", idx).float()
    wn = torch.cat([WU[s:s + 16384].float().norm(dim=1) for s in range(0, WU.shape[0], 16384)])
    norm_txt = {}

    def txt(t):
        if t not in norm_txt:
            norm_txt[t] = tok.decode([t]).strip().lower()
        return norm_txt[t]

    def push(cols):
        """cols [4096, k] float -> [V, k] push on every token."""
        gw = (g[:, None] * cols).to(torch.bfloat16)
        return torch.cat([(WU[s:s + 16384] @ gw).float() for s in range(0, WU.shape[0], 16384)])

    def rivals(t):
        cos = torch.cat([(WU[s:s + 16384].float() @ WU[t].float()) for s in range(0, WU.shape[0], 16384)]) / (wn * wn[t] + 1e-6)
        out = []
        for c in cos.argsort(descending=True)[1:400].tolist():
            if txt(c) != txt(t) and txt(c):
                out.append(c)
                if len(out) == NR:
                    break
        return out

    fo = (D / "per_neuron.jsonl").open("w", encoding="utf-8")
    res = {"say": [], "random": []}
    for grp, units in (("say", say_s), ("random", rand)):
        by = {}
        for l, n in units:
            by.setdefault(l, []).append(n)
        for l, ns in sorted(by.items()):
            Wd = tensor(f"model.layers.{l}.mlp.down_proj.weight", idx)   # [4096, 12288]
            P = push(Wd[:, ns].float())
            del Wd
            Z = (P - P.mean(0)) / P.std(0)
            for j, n in enumerate(ns):
                z = Z[:, j]; t = int(z.argmax()); rv = rivals(t)
                rec = {"group": grp, "unit": f"L{l}:{n}", "top_token": tok.decode([t]), "top_z": z[t].item(),
                       "rival_tokens": [tok.decode([r]) for r in rv], "rival_z": [round(z[r].item(), 3) for r in rv],
                       "rival_mean_z": stx.mean(z[r].item() for r in rv),
                       "bottom10": [tok.decode([b]) for b in z.argsort()[:10].tolist()]}
                rec["contrastive"] = rec["rival_mean_z"] < 0
                res[grp].append(rec); fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(grp, "layer", l, len(ns), f"({time.time() - t0:.0f}s)", flush=True)
    fo.close()
    S = {"n_say_total": len(say), "n_say_sampled": len(say_s), "n_random": len(rand), "n_rivals": NR}
    for grp in res:
        v = [r["rival_mean_z"] for r in res[grp]]
        S[grp] = {"median_rival_z": stx.median(v), "frac_contrastive": sum(r["contrastive"] for r in res[grp]) / len(v),
                  "quartiles_rival_z": [sorted(v)[int(q * len(v))] for q in (0.25, 0.5, 0.75)]}
    fs, fr = S["say"]["frac_contrastive"], S["random"]["frac_contrastive"]
    S["P6a_pass"] = S["say"]["median_rival_z"] > 0
    S["P6b_ratio"] = fs / fr if fr > 0 else None
    S["P6b_pass"] = fs >= 0.10 and fs >= 3 * fr
    S["examples_contrastive_say"] = [{k: r[k] for k in ("unit", "top_token", "rival_tokens", "rival_mean_z")}
                                     for r in sorted(res["say"], key=lambda r: r["rival_mean_z"])[:15]]
    S["say_sampled_units"] = [f"L{l}:{n}" for l, n in say_s]
    save("rival_push/summary.json", S)
    print({k: v for k, v in S.items() if k not in ("say_sampled_units",)})


if __name__ == "__main__":
    main()
