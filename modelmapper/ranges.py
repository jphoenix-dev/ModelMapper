"""Each neuron's normal range on web text (its max and min), for "x peak" steering.

Tables for Qwen3-0.6B and Qwen3-8B ship with the package (modelmapper/data/ranges/; 2,006,568 FineWeb tokens each,
the tables every published steering result used). For another model, build one:
    modelmapper ranges <model> [--tokens 2e6]     # streams MODEL_MAPPER_CORPUS (a FineWeb parquet shard)
which writes <cache>/ranges/ranges_<model>.pt. The running max / min is updated inside each layer's hook, so all
layers' activations are never held at once; token 0 of each document (an attention sink) is left out, documents are
cut at 1,024 tokens and those under 64 skipped (as every published table was built).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from modelmapper.settings import CACHE_DIR, CORPUS, short_name

SHIPPED = Path(__file__).resolve().parent / "data" / "ranges"


def docs(tok, ctx, min_len=64, corpus=CORPUS, centre=None):
    """Token arrays of up to ctx tokens per document. corpus: parquet (default) or .jsonl of {"text": ...}.
    centre: optional regex; the window then starts ~3,000 chars before its first match so it's included."""
    if str(corpus).endswith(".jsonl"):
        texts = (json.loads(l)["text"] for l in open(corpus, encoding="utf-8"))
    else:
        texts = (t for b in pq.ParquetFile(corpus).iter_batches(batch_size=256, columns=["text"])
                 for t in b.column(0).to_pylist())
    for text in texts:
        if centre and (m := re.search(centre, text, re.I)):
            text = text[max(0, m.start() - 3000):]
        ids = tok(text, add_special_tokens=False).input_ids[:ctx]
        if len(ids) >= min_len:
            yield np.array(ids, dtype=np.int32)


def ranges_path(model_name):
    """The table for a model: shipped if there is one, else the one built in the cache (may not exist yet)."""
    shipped = SHIPPED / f"ranges_{short_name(model_name)}.pt"
    return shipped if shipped.exists() else CACHE_DIR / "ranges" / f"ranges_{short_name(model_name)}.pt"


def load_ranges(model_name):
    p = ranges_path(model_name)
    if not p.exists():
        raise FileNotFoundError(f"no range table for {model_name}: build it with `modelmapper ranges {model_name}` "
                                "(needs a FineWeb shard, MODEL_MAPPER_CORPUS), or give absolute values (L23:2392=12.5)")
    return torch.load(p)


def build_ranges(model_name, n_tokens=2_000_000, corpus=None, out=None, log=print):
    """Stream web text through the model and keep every neuron's max and min (the published tables' procedure)."""
    from modelmapper.model import DEVICE, load, mlps
    corpus = corpus or CORPUS
    if not Path(corpus).exists():
        raise FileNotFoundError(f"no corpus at {corpus}: set MODEL_MAPPER_CORPUS to a FineWeb parquet shard")
    model, tok = load(model_name)
    L, W = model.config.num_hidden_layers, model.config.intermediate_size
    mx = torch.full((L, W), -1e9, device=DEVICE)
    mn = torch.full((L, W), 1e9, device=DEVICE)
    state = {}  # current batch's valid-token mask, read by the hooks

    def make_hook(l):
        def hook(m, args):
            a = args[0][state["valid"]].float()        # [n_valid, W]
            mx[l] = torch.maximum(mx[l], a.max(0).values)
            mn[l] = torch.minimum(mn[l], a.min(0).values)
        return hook

    handles = [mlps(model)[l].down_proj.register_forward_pre_hook(make_hook(l)) for l in range(L)]
    seen, buf, t0 = 0, [], time.time()
    gen = docs(tok, 1024, corpus=corpus)
    try:
        while seen < n_tokens:
            buf.append(next(gen))
            if sum(map(len, buf)) < 8192:
                continue
            T = max(map(len, buf))
            ids = torch.zeros((len(buf), T), dtype=torch.long)
            mask = torch.zeros_like(ids)
            for i, x in enumerate(buf):
                ids[i, :len(x)] = torch.as_tensor(x, dtype=torch.long); mask[i, :len(x)] = 1
            valid = mask.bool().to(DEVICE); valid[:, 0] = False  # token 0 is an attention sink
            state["valid"] = valid
            with torch.no_grad():
                model.model(input_ids=ids.to(DEVICE), attention_mask=mask.to(DEVICE))
            seen += int(valid.sum()); buf = []
            if seen // 8192 % 50 == 0:
                log(f"{seen/1e6:.2f}M tokens, {seen/(time.time()-t0):,.0f} tok/s")
    finally:
        for h in handles:
            h.remove()
    log(f"{seen:,} tokens in {time.time()-t0:.0f}s")
    out = Path(out) if out else CACHE_DIR / "ranges" / f"ranges_{short_name(model_name)}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"max": mx.cpu(), "min": mn.cpu(), "tokens": seen}, out)
    return out
