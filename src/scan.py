"""Look at units 'in the wild': stream web text through the model and keep each unit's
strongest positive and negative firings, with context.

A *unit* is a neuron or a fixed mix of neurons in one layer's MLP neuron space:
    {"id": "L18_N4012", "layer": 18, "neuron": 4012}                    # a real neuron
    {"id": "mix_a", "layer": 18, "weights": {"4012": 0.7, "77": 0.7}}    # planted mixture
    {"id": "rand_3", "layer": 18, "dense_seed": 3}                       # random unit direction
Unit activation = neuron activations (input to down_proj) dotted with the unit's weights.

Pass 1 streams N tokens and keeps top/bottom-k (value, doc, position) per unit, plus a random
sample of activations for percentiles. Pass 2 re-runs only the winning documents to record the
unit's activation on every token of the window and the gate/value signs at the peak.

Usage: python src/scan.py <model> <units.json> <out.json> [n_tokens=20e6] [ctx=1024] [corpus] [centre_regex]
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from config import CORPUS  # $ALREADY_CLEAN_CORPUS
from neurons import capture, load, mlps

K = 40              # examples kept per pole
SAMPLE_RATE = 0.002  # share of tokens kept for the activation-distribution sample


def unit_matrix(units, layer, width):
    us = [u for u in units if u["layer"] == layer]
    W = torch.zeros(width, len(us))
    for j, u in enumerate(us):
        if "neuron" in u:
            W[u["neuron"], j] = 1.0
        elif "weights" in u:
            for n, w in u["weights"].items():
                W[int(n), j] = w
        else:
            g = torch.Generator().manual_seed(u["dense_seed"])
            v = torch.randn(width, generator=g)
            W[:, j] = v / v.norm()
    return [u["id"] for u in us], W.cuda()


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


def forward_units(model, layers, mats, batch_ids, pad_id, gate_value=False):
    T = max(len(x) for x in batch_ids)
    ids = torch.full((len(batch_ids), T), pad_id)
    mask = torch.zeros_like(ids)
    for i, x in enumerate(batch_ids):
        ids[i, :len(x)] = torch.as_tensor(x, dtype=torch.long)
        mask[i, :len(x)] = 1
    ids, mask = ids.cuda(), mask.cuda()
    if gate_value:  # pass 2: few docs, one layer, keep full activations for gate/value lookup
        with torch.no_grad(), capture(model, layers, gate_value=True) as s:
            model.model(input_ids=ids, attention_mask=mask)  # skip the vocab-sized output layer
        acts = torch.cat([s["act"][l] @ mats[l][1] for l in layers], -1)
    else:  # pass 1: project to unit values inside the hook so we never hold whole layers
        proj, handles = {}, []
        for l in layers:
            handles.append(mlps(model)[l].down_proj.register_forward_pre_hook(
                lambda m, args, l=l: proj.__setitem__(l, args[0].float() @ mats[l][1])))
        try:
            with torch.no_grad():
                model.model(input_ids=ids, attention_mask=mask)
        finally:
            for h in handles:
                h.remove()
        acts, s = torch.cat([proj[l] for l in layers], -1), None  # [B, T, U]
    valid = mask.bool().clone()
    valid[:, 0] = False  # token 0 is an attention sink with huge activations
    return acts, valid, s


def main():
    model_name, units_path, out_path = sys.argv[1:4]
    n_tokens = int(float(sys.argv[4])) if len(sys.argv) > 4 else 20_000_000
    ctx = int(sys.argv[5]) if len(sys.argv) > 5 else 1024
    corpus = sys.argv[6] if len(sys.argv) > 6 else CORPUS
    centre = sys.argv[7] if len(sys.argv) > 7 else None
    units = json.loads(Path(units_path).read_text(encoding="utf-8"))
    torch.cuda.set_per_process_memory_fraction(0.92)  # fail loudly instead of spilling to system RAM
    model, tok = load(model_name)
    width = model.config.intermediate_size
    layers = sorted({u["layer"] for u in units})
    mats = {l: unit_matrix(units, l, width) for l in layers}
    ids_order = [i for l in layers for i in mats[l][0]]
    U = len(ids_order)
    pad = tok.pad_token_id or 0

    top_v = torch.full((U, K), -1e9, device="cuda"); top_i = torch.zeros((U, K), dtype=torch.long, device="cuda")
    bot_v = torch.full((U, K), 1e9, device="cuda");  bot_i = torch.zeros((U, K), dtype=torch.long, device="cuda")
    samples, all_docs = [], []
    seen, t0 = 0, time.time()
    batch_tokens = 6_144  # bigger batches overflowed the 32 GB card on 8B and silently spilled to system RAM
    buf = []
    gen = docs(tok, ctx, corpus=corpus, centre=centre)

    def flush(buf):
        nonlocal top_v, top_i, bot_v, bot_i
        acts, valid, _ = forward_units(model, layers, mats, buf, pad)
        base = len(all_docs)
        all_docs.extend(buf)
        T = acts.shape[1]
        flat = acts.reshape(-1, U).T                                   # [U, B*T]
        v = valid.reshape(-1)
        code = (torch.arange(len(buf), device="cuda")[:, None] + base) * ctx + torch.arange(T, device="cuda")[None]
        code = code.reshape(-1)
        hi = flat.masked_fill(~v, -1e9); lo = flat.masked_fill(~v, 1e9)
        top_v, idx = torch.cat([top_v, hi], 1).topk(K, 1); top_i = torch.cat([top_i, code.expand(U, -1)], 1).gather(1, idx)
        bot_v, idx = torch.cat([bot_v, lo], 1).topk(K, 1, largest=False); bot_i = torch.cat([bot_i, code.expand(U, -1)], 1).gather(1, idx)
        keep = v & (torch.rand_like(v, dtype=torch.float) < SAMPLE_RATE)
        samples.append(flat[:, keep].cpu())
        return int(v.sum())

    while seen < n_tokens:
        try:
            d = next(gen)
        except StopIteration:
            break
        buf.append(d)
        if sum(map(len, buf)) >= batch_tokens:
            seen += flush(buf); buf = []
            if len(all_docs) % 200 < 16:
                rate = seen / (time.time() - t0)
                print(f"{seen/1e6:.1f}M tokens, {rate:,.0f} tok/s, eta {(n_tokens-seen)/rate/60:.0f} min", flush=True)
    if buf:
        seen += flush(buf)

    # Pass 2: re-run winning docs to annotate the whole window and get gate/value signs at the peak.
    sample = torch.cat(samples, 1)  # [U, S]
    report = {"model": model_name, "tokens_scanned": seen, "ctx": ctx, "units": {}}
    gv_layers = layers
    for j, uid in enumerate(ids_order):
        unit = next(u for u in units if u["id"] == uid)
        dist = sample[j]
        qs = torch.quantile(dist, torch.tensor([0.001, 0.01, 0.5, 0.99, 0.999, 0.9999]))
        entry = {"unit": unit, "quantiles": dict(zip(["p0.1", "p1", "p50", "p99", "p99.9", "p99.99"],
                                                      [round(q, 4) for q in qs.tolist()])),
                 "max": round(top_v[j, 0].item(), 4), "min": round(bot_v[j, 0].item(), 4)}
        for pole, vals, codes in (("pos", top_v[j], top_i[j]), ("neg", bot_v[j], bot_i[j])):
            exs = []
            codes = codes.tolist()
            doc_ids = [c // ctx for c in codes]
            for start in range(0, len(doc_ids), 8):
                chunk = doc_ids[start:start + 8]
                acts, valid, s = forward_units(model, [unit["layer"]], {unit["layer"]: mats[unit["layer"]]},
                                               [all_docs[d] for d in chunk], pad, gate_value="neuron" in unit)
                col = mats[unit["layer"]][0].index(uid)
                for b, d in enumerate(chunk):
                    k = start + b
                    pos = codes[k] % ctx
                    a = acts[b, :len(all_docs[d]), col]
                    lo, hi = max(0, pos - 200), min(len(all_docs[d]), pos + 12)
                    peak = abs(vals[k].item()) or 1.0
                    pieces = []
                    window = [int(x) for x in all_docs[d][lo:hi]]
                    prev = ""
                    for p in range(lo, hi):
                        # decode incrementally: one token can hold half of a multi-byte character
                        cur = tok.decode(window[:p - lo + 1])
                        t, prev = cur[len(prev):], cur
                        r = a[p].item() / peak if pole == "pos" else -a[p].item() / peak
                        pieces.append(f"[[{t}]]" if p == pos else (f"«{t}»" if r > 0.3 else t))
                    ex = {"value": round(vals[k].item(), 4), "doc": d, "pos": pos, "text": "".join(pieces)}
                    if "neuron" in unit:
                        g = s["gate"][unit["layer"]][b, pos, unit["neuron"]].item()
                        v = s["value"][unit["layer"]][b, pos, unit["neuron"]].item()
                        ex["regime"] = ("+" if g >= 0 else "-") + ("+" if v >= 0 else "-")
                    exs.append(ex)
            entry[pole] = exs
        report["units"][uid] = entry
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"done: {seen:,} tokens, {U} units, {(time.time()-t0)/60:.1f} min -> {out_path}")


if __name__ == "__main__":
    main()
