"""What each mapped neuron pushes the model towards THROUGH THE LATER LAYERS (the J-lens), when it fires positively and
negatively - the companion of push_words.py (which reads the weights directly, the logit lens).

    python -I -X utf8 explainer/gen/map/push_words_jlens.py [--top 10] [--jlens D:/already-clean-data/jlens256/jlens]

The J-lens of layer l (src/loop_v2/jlens.py, after Anthropic's J-space work) is the average Jacobian of the final
residual stream with respect to layer l's, estimated here from 256 FineWeb windows. A neuron's output direction read
through it gives the tokens its push ends up raising (positive side) or lowering (negative side) once the later layers
have acted on it. jlens.py `neurons` already saved every neuron's top / bottom 30 J-lens tokens; this only selects the
mapped neurons and decodes them. CPU only.

Writes data/push_j/L<layer>.json ({index: [[positive words], [negative words]]}, same format as data/push) and
data/push_j/layers.json: the layers where the map shows J-lens words. Rule, fixed before the 256-window results were
seen (18 Oct): a layer qualifies if, in jlens_validate.json, the J-lens's median held-out next-16-token rank is better
(lower) than the logit lens's, AND the split-half top-20 overlap (sampled every 5 layers; nearest sampled layer) is
>= 0.5.
"""
from __future__ import annotations

import argparse
import json
import statistics as stx
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODELS, clean_text  # noqa: E402

DATA = Path(__file__).resolve().parents[1].parent / "site" / "map" / "data"      # explainer/gen/map -> explainer/site


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--jlens", default="D:/already-clean-data/jlens256/jlens")
    args = ap.parse_args()
    J = Path(args.jlens)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODELS / "Qwen3-8B"))
    val = json.loads((J / "jlens_validate.json").read_text(encoding="utf-8"))
    ranks, sh = val["ranks"], {int(k): v for k, v in val["split_half"].items()}
    layers = []
    for l in range(36):
        jr, lr = stx.median(ranks[str(l)]["J"]), stx.median(ranks[str(l)]["logit"])
        near = min(sh, key=lambda s: (abs(s - l), s))
        layers.append({"layer": l, "j_rank": jr, "logit_rank": lr, "split_half": sh[near], "split_half_from": near,
                       "show": jr < lr and sh[near] >= 0.5})
    nm = json.loads((DATA / "neuronmap.json").read_text(encoding="utf-8"))["neurons"]
    want = {}
    for L, i in zip(nm["L"], nm["i"]):
        want.setdefault(L, set()).add(i)
    (DATA / "push_j").mkdir(exist_ok=True)
    dec = lambda ids: [clean_text(tok.decode([int(t)])) for t in ids[:args.top]]
    for L in range(36):
        out = {}
        if want.get(L):
            Z = np.load(J / f"jlens_neurons_L{L}.npz")
            top, bot = Z["j_top"], Z["j_bot"]
            for n in sorted(want[L]):
                out[str(n)] = [dec(top[n]), dec(bot[n])]
        (DATA / "push_j" / f"L{L}.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print(f"layer {L}: {len(out)} neurons; show={layers[L]['show']}", flush=True)
    (DATA / "push_j" / "layers.json").write_text(json.dumps({"source": str(J), "windows": 256, "rule": __doc__.split("Rule, ")[1].strip(),
                                                             "layers": layers}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("layers shown:", [x["layer"] for x in layers if x["show"]])


if __name__ == "__main__":
    main()
