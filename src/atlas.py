"""Look up any neuron in the prebuilt atlas (see build_atlas.py) and render its top firings.

    from atlas import Atlas
    at = Atlas("Qwen/Qwen3-8B")
    at.examples(28, 6711, "pos", n=5)          # -> [{"value", "regime", "doc", "pos", "text"}, ...]
    at.write_dossiers([(28, 6711), (30, 2600)], "results/dossiers_x")   # blind dossiers for label.py

The firing token is marked [[like this]]. Without a model, other tokens aren't highlighted (that
needs a forward pass); pass model=... to examples()/write_dossiers() to add «highlights».
Positive-pole double-negative firings (gate < 0) are dropped by default, per PREREG.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from config import model_path
from neurons import capture, DEVICE

ROOT = Path(__file__).resolve().parents[1]


class Atlas:
    def __init__(self, model_name: str):
        short = model_name.split("/")[-1]
        self.model_name = model_name
        self.a = torch.load(ROOT / "data" / f"atlas_{short}.pt")
        d = np.load(ROOT / "data" / f"atlas_{short}_docs.npz")
        self.tokens, lengths = d["tokens"], d["lengths"]
        self.starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        self.lengths = lengths
        self.ctx = self.a["ctx"]
        self.tok = AutoTokenizer.from_pretrained(model_path(model_name))

    def doc(self, i: int) -> np.ndarray:
        return self.tokens[self.starts[i]:self.starts[i] + self.lengths[i]]

    def entries(self, layer, neuron, pole="pos", drop_double_negative=True):
        p = "top" if pole == "pos" else "bot"
        v = self.a[f"{p}_v"][layer, neuron].float().tolist()
        c = self.a[f"{p}_c"][layer, neuron].tolist()
        g = self.a[f"{p}_g"][layer, neuron].tolist()
        out = []
        for val, code, gate_pos in zip(v, c, g):
            regime = ("+" if gate_pos else "-") + ("+" if (val >= 0) == gate_pos else "-")
            if pole == "pos" and drop_double_negative and regime == "--":
                continue
            out.append({"value": round(val, 4), "doc": code // self.ctx, "pos": code % self.ctx, "regime": regime})
        return out

    def _render(self, e, before=200, after=12, acts=None, peak=None):
        ids = [int(t) for t in self.doc(e["doc"])]
        lo, hi = max(0, e["pos"] - before), min(len(ids), e["pos"] + after)
        pieces, prev = [], ""
        for p in range(lo, hi):
            cur = self.tok.decode(ids[lo:p + 1])      # incremental: tokens can split multi-byte chars
            t, prev = cur[len(prev):], cur
            if p == e["pos"]:
                pieces.append(f"[[{t}]]")
            elif acts is not None and peak and acts[p] / peak > 0.3:
                pieces.append(f"«{t}»")
            else:
                pieces.append(t)
        return "".join(pieces)

    def examples(self, layer, neuron, pole="pos", n=20, model=None, before=200, after=12, **kw):
        """before/after: tokens of context rendered around the firing token (CAPS.md #4)."""
        es = self.entries(layer, neuron, pole, **kw)[:n]
        for e in es:
            acts = None
            if model is not None:  # one forward pass over the doc to highlight other strong tokens
                ids = torch.as_tensor(self.doc(e["doc"]), dtype=torch.long)[None].to(DEVICE)
                with torch.no_grad(), capture(model, [layer]) as s:
                    model.model(input_ids=ids)
                sign = 1 if pole == "pos" else -1
                acts = (sign * s["act"][layer][0, :, neuron]).tolist()
            e["text"] = self._render(e, before=before, after=after, acts=acts, peak=abs(e["value"]))
        return es

    def write_dossiers(self, units, out_dir, n=20, model=None, seed=42, roles=None):
        """Same blind format as make_dossiers.py, so label.py works unchanged. Returns the key."""
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        rng = random.Random(seed)
        codes = rng.sample(range(100, 1000), len(units))
        key = {}
        for (layer, neuron), code in zip(units, codes):
            c = f"U{code}"
            uid = f"L{layer}_N{neuron}"
            all_pos = self.entries(layer, neuron, "pos", drop_double_negative=False)
            key[c] = {"unit": uid, "role": (roles or {}).get(uid, "?"),
                      "double_negative_removed_from_top40": sum(e["regime"] == "--" for e in all_pos)}
            lines = [f"# Unit {c}", "",
                     "Below are the text positions where this unit fires most strongly in a large sample of web text.",
                     "The firing token is marked [[like this]]. Other nearby tokens where it also fires noticeably are marked «like this».",
                     "Strength is shown relative to the unit's strongest firing (1.00).", ""]
            for title, pole in (("Positive pole: strongest positive firings", "pos"),
                                ("Negative pole: strongest negative firings", "neg")):
                exs = self.examples(layer, neuron, pole, n=n, model=model)
                lines += [f"## {title}", ""]
                peak = abs(exs[0]["value"]) if exs else 1.0
                for i, x in enumerate(exs, 1):
                    text = x["text"].replace("\n", " ⏎ ")
                    lines += [f"**{i}.** ({abs(x['value'])/peak:.2f}) …{text[-900:]}", ""]
            (out / f"{c}.md").write_text("\n".join(lines), encoding="utf-8")
        (out.parent / f"{out.name}_KEY.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
        return key


def output_words(model, tok, layer, neuron, k=8):
    """What a neuron writes directly to the vocabulary: its down_proj column through the final norm scale
    and the unembedding. Direct effect only (ignores downstream layers); most meaningful for late layers."""
    col = model.model.layers[layer].mlp.down_proj.weight[:, neuron].float()
    logits = model.lm_head.weight.float() @ (col * model.model.norm.weight.float())
    up = [tok.decode([i]) for i in logits.topk(k).indices.tolist()]
    down = [tok.decode([i]) for i in (-logits).topk(k).indices.tolist()]
    return up, down
