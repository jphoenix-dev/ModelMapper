"""Where is the answer staged before the answer position? Logit lens over positions and layers (attention study; no API).

The late heads L34H28 / L35H22 copy mostly from the '<|im_start|>' that opens the assistant turn (writer_sources.py).
If the answer is staged there, the residual stream at that position should already 'read out' as the answer.
For each recall prompt, at positions: the last question word, '?', and each chat-template token after it up to the
answer position, at every layer output, we apply the model's final norm + unembedding (logit lens; crude for middle
layers, so only ranks are reported) and record the answer's first token's RANK among the ~152k words.
Reports per position: earliest layer where the answer is top-1 / top-10 (median over prompts and concepts), and the
answer's median rank at layers 20, 25, 30, 33, 35.  Control: the same for a random OTHER concept's answer token.
Usage: RECALL_VARIANT=V3 ANSWER_DISCOVERY=0 CONCEPT_OUT_DIR=<dir> python src/staging.py <tag> "<concept>" [...]
"""
import json
import random
import sys

import torch

from config import DEFAULT_MODEL
import concept_pipeline as cp
from neurons import DEVICE
from pipeline_fast import FastPipeline

CHECK = (20, 25, 30, 33, 35)


def main():
    tag = sys.argv[1]; concepts = [c.strip() for c in sys.argv[2:]]
    pipe = FastPipeline(DEFAULT_MODEL); m = pipe.model; L = m.config.num_hidden_layers
    out_p = cp.OUT / f"staging_{tag}.json"
    out = json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists() else {}
    firsts = {c: pipe.make_prompts(c)[1][0] for c in concepts}
    for concept in concepts:
        if concept in out:
            continue
        spec, tid, recall, _ = pipe.make_prompts(concept); a = tid[0]
        rng = random.Random(sum(map(ord, concept)))
        ctrl = rng.choice([firsts[c] for c in concepts if firsts[c] != a])
        rows = {}
        for q, _ in recall:
            ids = pipe.tok(q, return_tensors="pt").to(DEVICE).input_ids
            toks = [pipe.tok.decode([t]) for t in ids[0].tolist()]
            qm = max(i for i, t in enumerate(toks) if "?" in t)
            pos = {"last_word": qm - 1, "?": qm, **{f"[{j - len(toks)}]{toks[j]!r}": j for j in range(qm + 1, len(toks))}}
            # raw (un-normalised) residual after each layer, via hooks: HF's hidden_states[-1] is already final-normed
            res_ = {}
            hk = [m.model.layers[l].register_forward_hook(lambda mod, i, o, l=l: res_.__setitem__(l, (o[0] if isinstance(o, tuple) else o)[0]))
                  for l in range(L)]
            try:
                with torch.no_grad():
                    final = m(input_ids=ids).logits[0].float()
            finally:
                for h_ in hk:
                    h_.remove()
            with torch.no_grad():
                for name, j in pos.items():
                    H_ = torch.stack([res_[l][j] for l in range(L)])                # [L, d] layer outputs
                    lg = m.lm_head(m.model.norm(H_)).float()
                    assert (lg[-1].argmax() == final[j].argmax()).item(), "lens at the last layer must match the model's own prediction"
                    ra = (lg > lg[:, a:a + 1]).sum(-1).tolist(); rc = (lg > lg[:, ctrl:ctrl + 1]).sum(-1).tolist()
                    rows.setdefault(name, []).append((ra, rc))
        res = {}
        for name, rr in rows.items():
            def first(r, k):
                return next((l for l in range(L) if r[l] < k), None)
            res[name] = {"first_top1": [first(ra, 1) for ra, _ in rr], "first_top10": [first(ra, 10) for ra, _ in rr],
                         "rank_at": {l: sorted(ra[l] for ra, _ in rr)[len(rr) // 2] for l in CHECK},
                         "ctrl_rank_at": {l: sorted(rc[l] for _, rc in rr)[len(rr) // 2] for l in CHECK}}
        out[concept] = res
        print(f"{concept:<28} " + " | ".join(f"{k}: top10 from L{sorted(x for x in v['first_top10'] if x is not None)[len([x for x in v['first_top10'] if x is not None]) // 2] if any(x is not None for x in v['first_top10']) else '-'}"
                                             for k, v in res.items()), flush=True)
        out_p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
