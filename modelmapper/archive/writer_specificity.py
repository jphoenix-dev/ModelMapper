"""Content writers vs format writers among the neurons that write the answer word (static weights, CPU; no API).

For every neuron in a concept's top-10 direct writers (answer_readout json), take its static push on each concept's
answer first token (output weights -> final norm scale -> unembedding, minus the mean over the vocabulary; per unit
activation, times the sign of the neuron's contribution). Then
  specificity = (push on THIS concept's answer - median push on the OTHER concepts' answers) / std of its push over vocab
A content writer pushes its own answer far more than other answers (high specificity); a format writer ("the next word
starts with a capital", "not a digit or a space") pushes every answer start about equally (specificity ~ 0, but
the generic push > 0).  Threshold for labels (stated before looking at the results): specificity >= 3 -> content;
specificity < 1 and generic push > 0 -> format; else mixed.
Usage: python src/writer_specificity.py <answer_readout json> [...]  -> results/writer_specificity.json/.txt
"""
import json
import statistics as stx
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer

from config import DEFAULT_MODEL, model_path
from writer_lens import tensor

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = {}
    for f in sys.argv[1:]:
        rows.update(json.loads(Path(f).read_text(encoding="utf-8")))
    path = Path(model_path(DEFAULT_MODEL)); tok = AutoTokenizer.from_pretrained(str(path))
    WU = tensor(path, "lm_head.weight").float(); gf = tensor(path, "model.norm.weight").float()
    WUm = WU.mean(0)
    ans = {}
    for c, r in rows.items():
        ids = tok.encode(r["answer_token"], add_special_tokens=False)
        assert len(ids) == 1 and tok.decode(ids) == r["answer_token"], f"{c}: answer token {r['answer_token']!r} does not round-trip"
        ans[c] = ids[0]
    A = torch.tensor([ans[c] for c in rows]); names = list(rows)
    downs, out, lab = {}, {}, Counter()
    for c, r in rows.items():
        out[c] = []
        for nn, v, isown in r["top_neurons"]:
            L, n = (int(x) for x in nn.lstrip("L").split(":"))
            if L not in downs:
                downs[L] = tensor(path, f"model.layers.{L}.mlp.down_proj.weight")
            x_ = gf * downs[L][:, n].float(); s = WU @ x_ - WUm @ x_
            a_ = ans[c]
            # orient by the neuron's ACTIVATION sign (contrib = act * push[a] * positive, so sign(act) = sign(contrib) * sign(push[a]))
            s = s * ((1 if v > 0 else -1) * (1 if s[a_].item() > 0 else -1))
            sd = s.std().item(); onans = s[A]
            own = onans[names.index(c)].item(); others = [onans[i].item() for i in range(len(names)) if names[i] != c]
            gen = stx.median(others); spec = (own - gen) / sd
            frac_up = sum(o > 0 for o in others) / len(others)
            # same token shape: single-character answer starts vs longer ones (a "capital letter" writer is a format writer)
            short = len(tok.decode([a_]).strip()) == 1
            same = [onans[i].item() for i in range(len(names)) if names[i] != c and (len(tok.decode([ans[names[i]]]).strip()) == 1) == short]
            spec_same = (own - stx.median(same)) / sd if same else None
            # labels v2 (set after v1 showed the token-shape confound): format = pushes >= 75% of the other answers up;
            # content = own answer >= 3 sd above same-shape others and pushes <= 60% of the others up; else mixed
            label = ("format" if frac_up >= 0.75 else "content" if spec_same is not None and spec_same >= 3 and frac_up <= 0.6 else "mixed")
            lab[label] += 1
            out[c].append({"neuron": nn, "contrib": v, "own_concept_neuron": isown, "specificity": round(spec, 2),
                           "generic_push_sd": round(gen / sd, 2), "frac_other_answers_up": round(frac_up, 2),
                           "specificity_same_shape": None if spec_same is None else round(spec_same, 2), "short_answer": short, "label": label})
    L_ = []
    tot = sum(lab.values())
    L_.append(f"Top-10 direct answer writers x {len(rows)} concepts = {tot} entries: " + ", ".join(f"{k} {v} ({v / tot:.0%})" for k, v in lab.most_common()))
    contrib = Counter()
    for c in out:
        for e in out[c]:
            contrib[e["label"]] += e["contrib"]
    L_.append("summed contribution to the answer score by label (top-10 writers only): " + ", ".join(f"{k} {v:+.1f}" for k, v in contrib.most_common()))
    own_lab = Counter(e["label"] for c in out for e in out[c] if e["own_concept_neuron"])
    L_.append(f"labels of writers that are ALSO the concept's own top-10 (recall) neurons: {dict(own_lab)}")
    rec = Counter(e["neuron"] for c in out for e in out[c])
    L_.append("\nneurons in 3+ concepts' top-10 writers: label counts, median specificity, median generic push (sd units)")
    for nn, k in rec.most_common():
        if k < 3:
            break
        es = [e for c in out for e in out[c] if e["neuron"] == nn]
        L_.append(f"  {nn:<10} {k:>2}x  {dict(Counter(e['label'] for e in es))}  spec {stx.median(e['specificity'] for e in es):+.2f}  "
                  f"generic {stx.median(e['generic_push_sd'] for e in es):+.2f}  others up {stx.median(e['frac_other_answers_up'] for e in es):.2f}  "
                  f"in {sum(e['contrib'] > 0 for e in es)}/{k} it helps")
    L_.append("\nper concept: content / format / mixed among top-10 writers")
    for c in out:
        k = Counter(e["label"] for e in out[c])
        L_.append(f"  {c:<28} {k['content']}/{k['format']}/{k['mixed']}   top content writer: "
                  + next((f"{e['neuron']} ({e['contrib']:+.1f}, spec {e['specificity']:.1f})" for e in out[c] if e["label"] == "content"), "-"))
    txt = "\n".join(L_)
    (ROOT / "results/writer_specificity.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    (ROOT / "results/writer_specificity.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
