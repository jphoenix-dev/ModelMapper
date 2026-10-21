"""The vocabulary sweep (v3's sweep, research vocab_sweep2): every token of the vocabulary in the slot of each context
of a fixed panel, read for EVERY MLP neuron at once. One forward pass gives every neuron's gate and up value at every
layer, so the panel is paid for once and each neuron's vocabulary favourites fall out of it.

Method. Each panel context is a sentence with one slot (PANEL_V1: twelve hand-written English sentences that vary
syntax and topic). Every allowed token (the whole vocabulary minus special tokens) is put in the slot, and each
layer's gate_proj output z_g and up_proj output z_u are read at the slot position and, if the sentence continues, at
its last position ("end"). A neuron's activation is a = SiLU(z_g) * z_u. Per context and read position r, kept for
all n_layers x n_neurons neurons:
  {r}_top_i / _top_v / _top_u       the top k tokens by z_g, with their z_g and z_u (float16; ids int32)
  {r}_mean / _std                   z_g mean and sd over all tokens
  {r}_apos_i / _apos_g / _apos_u    the top k tokens by a (the neuron's strongest positive writes)
  {r}_aneg_i / _aneg_g / _aneg_u    the top k tokens by -a (its strongest negative writes)
  {r}_amean / _astd                 a mean and sd over all tokens
  {r}_sample_z / _sample_u          full z_g / z_u profiles (every token) for the `sample` neurons
  {r}_pseudo_g / _pseudo_u          full profiles for the pseudo-neurons, then the `check` rows
plus tokens (the token ids, in profile row order), sample, pseudo (names) and pseudo_layer. A pseudo-neuron's values
are v . gate_row and v . up_row, with v the post_attention_layernorm output: exactly what gate_proj and up_proj read.
The `check` rows copy real neurons' gate and up rows into the pseudo path, so their pseudo profiles must reproduce
those neurons' sample profiles (an exactness check of the pseudo path).

How it runs. The context's prefix (the text before the slot) is run once and its attention cache is copied for each
batch (prefix reuse: `copy.deepcopy` of the DynamicCache, then `batch_repeat_interleave`), so a batch only runs the
slot and the suffix. This works on the CPU with the pinned transformers (5.20); tests/test_v3_sweep.py checks the
results against a full forward pass of the whole sentence without a cache. The favourites are merged layer by layer
inside the forward pass (hooks on gate_proj, up_proj and post_attention_layernorm): a layer's values are folded into
the running top-k lists as soon as its up_proj runs, then dropped, so peak memory holds one layer, not all of them.
Each batch writes into spare buffers that are swapped in only when the whole batch succeeded, so an out-of-memory
retry (batching.run_batched) cannot count a token twice. Batches are token-sized: tok_budget tokens per batch x read
position. Values are computed in the model's own dtype on the model's own device (v3: bfloat16 on a GPU).

Output: <out_dir>/vocab_sweep_<tag>_c<k>.npz (one per context) and vocab_sweep_<tag>.json (panel, timing, batching),
the same files and keys as the research run, so either can be read by the same code. hub_tokens() then counts, for
every token, how many neurons have it among their favourites (hub_tokens_<tag>.npz / .txt).
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from modelmapper.batching import run_batched

SLOT = "\x00SLOT\x00"
PANEL_V1 = [                                               # (text with the slot) - varied syntax and topic, English
    f"The{SLOT}",
    f"I just bought a new{SLOT} and I love it.",
    f"She spent the whole afternoon talking about{SLOT} with her friends.",
    f"Q: What do you know about{SLOT}?\nA:",
    f'The word "{SLOT}" means',
    f"Here is a list of things: apples,{SLOT}, chairs, and",
    f"Breaking news today:{SLOT}",
    f"He was born in{SLOT} and later moved to",
    f"My favourite{SLOT} is the one that",
    f"import{SLOT}\n",
    f"The history of{SLOT} goes back to",
    f"They went to the{SLOT} together on Sunday.",
]
CHECK_V3 = ["L31:10454", "L25:7850", "L7:11335", "L0:1951"]   # the check rows of the v3 run (Qwen3-8B sample neurons)
LISTS = ("top", "apos", "aneg")


def parse_neuron(spec: str) -> tuple[int, int]:
    """'L31:10454' (or '31:10454') -> (31, 10454)."""
    L, i = str(spec).removeprefix("L").split(":")
    return int(L), int(i)


def default_check(sample) -> list[str]:
    """The v3 run's check rows if they are all in the sample (Qwen3-8B), else the sample's first four neurons."""
    names = [f"L{l}:{i}" for l, i in map(parse_neuron, sample or [])]
    return list(CHECK_V3) if all(c in names for c in CHECK_V3) else names[:4]


def allowed_tokens(tok, vocab_size: int, max_tokens: int = 0) -> np.ndarray:
    """Every token id below min(vocab_size, len(tok)) except special tokens and '<|...' tokens. max_tokens > 0 keeps a
    fixed random subset of that many (seed 2026, sorted): a smoke run."""
    special = set(tok.all_special_ids) | {j for t, j in tok.get_vocab().items() if t.startswith("<|")}
    allowed = np.array([j for j in range(min(vocab_size, len(tok))) if j not in special])
    if max_tokens:
        allowed = np.sort(np.random.default_rng(2026).choice(allowed, max_tokens, replace=False))
    return allowed


def panel_contexts(tok, panel=PANEL_V1) -> list[dict]:
    """Each context: display text ([X] for the slot), prefix and suffix token ids, and its read positions."""
    out = []
    for text in panel:
        a, b = text.split(SLOT)
        out.append({"text": text.replace(SLOT, "[X]"), "prefix": tok.encode(a), "suffix": tok.encode(b),
                    "read": ["slot", "end"] if b else ["slot"]})
    return out


@torch.inference_mode()
def run_sweep(model, tok, out_dir, tag, *, sample=None, pseudo=None, check=(), real=True, k=100, max_tokens=0,
              contexts=None, batch=1024, tok_budget=8192, layers=None, log=print) -> dict:
    """Run the sweep (module docstring) and write its files to out_dir; returns the meta dict (the .json).

    sample: "L:i" neurons that get full profiles. pseudo: a pseudo-neuron set (pseudo.make_gate_set / make_up_set /
    pseudo.load). check: "L:i" real neurons copied as pseudo rows (default_check gives v3's). real=False scores only
    the pseudo-neurons (no favourites, stats or sample profiles). k: favourites kept per neuron, list and reading.
    max_tokens: a random subset of the vocabulary (smoke runs). contexts: panel indices to run (default all); the
    other contexts' entries stay in an existing .json of the same tag if it was made with the same k, tokens and
    pseudo-neurons. batch: the largest batch; tok_budget: tokens per batch x read position.
    layers: keep only these layers (plus the sample's), and stop each forward pass after the deepest layer needed;
    the arrays then have one row per kept layer and meta["layers"] lists them (layer_rows maps a layer to its row).
    The stored values are the same as a full sweep's."""
    D = Path(out_dir); D.mkdir(parents=True, exist_ok=True)
    dev = model.get_input_embeddings().weight.device
    lay = model.model.layers; NL, NI = len(lay), lay[0].mlp.gate_proj.weight.shape[0]
    wdt = lay[0].mlp.gate_proj.weight.dtype
    V = model.get_input_embeddings().weight.shape[0]
    allowed = allowed_tokens(tok, V, max_tokens)
    sample = [parse_neuron(u) for u in (sample or [])]
    LS = list(range(NL)) if layers is None else sorted({int(x) for x in layers} | {l for l, _ in sample})
    LR = {L: j for j, L in enumerate(LS)}; NS = len(LS)
    # pseudo-neurons + check rows, grouped by layer
    p_layer, p_gate, p_up, p_names = ([], [], [], []) if pseudo is None else (
        list(pseudo["layer"]), list(pseudo["gate"]), list(pseudo["up"]), list(pseudo["names"]))
    for spec in check:
        L, i = parse_neuron(spec)
        p_layer.append(L); p_names.append(f"CHECK:L{L}:{i}")
        p_gate.append(lay[L].mlp.gate_proj.weight[i].float().cpu().numpy()); p_up.append(lay[L].mlp.up_proj.weight[i].float().cpu().numpy())
    NP = len(p_layer); p_layer = np.array(p_layer, dtype=np.int64); p_names = np.array([str(x) for x in p_names], dtype=str)
    PG = {}
    for L in sorted(set(p_layer.tolist())):
        js = np.where(p_layer == L)[0]
        PG[L] = (torch.tensor(js, device=dev), torch.tensor(np.stack([p_gate[j] for j in js]), device=dev, dtype=wdt),
                 torch.tensor(np.stack([p_up[j] for j in js]), device=dev, dtype=wdt))
    ctxs = panel_contexts(tok)
    only = list(range(len(ctxs))) if contexts is None else [int(x) for x in contexts]
    meta, t_all = {"panel": "v1", "version": 2, "K": k, "n_tokens": int(len(allowed)), "pseudo": p_names.tolist(), "real": bool(real),
                   "intermediate_size": int(NI), "contexts": []}, time.time()     # the exam's per-layer hub shares need the width
    if layers is not None:
        meta["layers"] = LS
    mp = D / f"vocab_sweep_{tag}.json"
    if contexts is not None and mp.exists():
        old = json.loads(mp.read_text(encoding="utf-8"))
        if all(old.get(f) == meta[f] for f in ("K", "n_tokens", "pseudo")):
            meta["contexts"] = [x for x in old["contexts"] if x["k"] not in only]
    s_by_layer = {}
    for col, (l, i) in enumerate(sample):
        s_by_layer.setdefault(l, ([], []))[0].append(col); s_by_layer[l][1].append(i)
    s_by_layer = {l: (torch.tensor(cs, device=dev), torch.tensor(ix, device=dev)) for l, (cs, ix) in s_by_layer.items()}
    RP, on, job = [], [True], {}

    def on_gate(mod, a, o, l):
        if on[0] and real:
            job["g"] = o[:, RP]

    def on_up(mod, a, o, l):
        if not on[0] or not real:
            return
        zg, zu = job.pop("g"), o[:, RP]; n = zg.shape[0]; cand = job["cand"]
        for q, r in enumerate(job["reads"]):
            G, U = zg[:, q], zu[:, q]                                           # [n, NI] in the model's dtype
            Gf = G.float(); A = torch.nn.functional.silu(Gf) * U.float()
            job["sums"][r][:, LR[l]] = torch.stack([Gf.sum(0), (Gf * Gf).sum(0), A.sum(0), (A * A).sum(0)]).double()
            for ls, key, largest in (("top", Gf, True), ("apos", A, True), ("aneg", A, False)):
                bv, bi = key.T.topk(min(k, n), dim=1, largest=largest)          # [NI, min(k, n)]
                if not largest:
                    bv = -bv                                                    # aneg keeps -a as its ranking score
                bg = G.T.gather(1, bi).half(); bu = U.T.gather(1, bi).half()
                o_ = job["st"][r][ls]; nu = job["nu"][r][ls]
                j = LR[l]; allv = torch.cat([o_["v"][j], bv], 1); tv, ti = allv.topk(k, dim=1)
                nu["v"][j] = tv; nu["i"][j] = torch.cat([o_["i"][j], cand.int()[bi]], 1).gather(1, ti)
                nu["g"][j] = torch.cat([o_["g"][j], bg], 1).gather(1, ti); nu["u"][j] = torch.cat([o_["u"][j], bu], 1).gather(1, ti)
            if l in s_by_layer:
                cs, ix = s_by_layer[l]; job["sz"][r][:, cs] = G[:, ix].float(); job["su"][r][:, cs] = U[:, ix].float()

    def on_norm(mod, a, o, l):
        if on[0] and l in PG:
            ix, Gp, Up = PG[l]; vv = o[:, RP]
            for q, r in enumerate(job["reads"]):
                job["pg"][r][:, ix] = (vv[:, q] @ Gp.T).float(); job["pu"][r][:, ix] = (vv[:, q] @ Up.T).float()

    def lists():
        return {ls: {"v": torch.full((NS, NI, k), -1e4, device=dev), "i": torch.zeros((NS, NI, k), dtype=torch.int32, device=dev),
                     "g": torch.zeros((NS, NI, k), dtype=torch.float16, device=dev), "u": torch.zeros((NS, NI, k), dtype=torch.float16, device=dev)}
                for ls in LISTS}

    hs = [lay[l].mlp.gate_proj.register_forward_hook(lambda mod, a, o, l=l: on_gate(mod, a, o, l)) for l in LS]
    hs += [lay[l].mlp.up_proj.register_forward_hook(lambda mod, a, o, l=l: on_up(mod, a, o, l)) for l in LS]
    deepest = max(LS + [int(x) for x in PG])                # nothing after this layer is read: don't run it
    full_layers = model.model.layers
    if deepest < NL - 1:
        model.model.layers = torch.nn.ModuleList(list(full_layers)[:deepest + 1])
    hs += [lay[l].post_attention_layernorm.register_forward_hook(lambda mod, a, o, l=l: on_norm(mod, a, o, l)) for l in PG]
    try:
        for ki, c in enumerate(ctxs):
            if ki not in only:
                continue
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            t0 = time.time(); P, S = c["prefix"], c["suffix"]; T = len(P) + 1 + len(S)
            pos = {"slot": len(P), "end": T - 1}; off = len(P)
            RP[:] = [pos[r] - off for r in c["read"]]
            pkv = None
            if P:                                             # the shared prefix, run once (hooks off)
                on[0] = False; pkv = model.model(input_ids=torch.tensor([P], device=dev), use_cache=True).past_key_values; on[0] = True
            st = {r: lists() for r in c["read"]} if real else None     # state, and the next batch's buffers
            nub = {r: lists() for r in c["read"]} if real else None
            sums = {r: torch.zeros((4, NS, NI), dtype=torch.float64, device=dev) for r in c["read"]}   # g, g^2, a, a^2
            prof = {r: {nm: np.zeros((len(allowed), w), dtype=np.float16) for nm, w in (("sz", len(sample)), ("su", len(sample)), ("pg", NP), ("pu", NP))}
                    for r in c["read"]}
            base = torch.tensor(P + [0] + S, device=dev)

            def step(a0, a1):
                cand = torch.from_numpy(allowed[a0:a1]).to(dev); n = len(cand)
                ids = base.repeat(n, 1); ids[:, len(P)] = cand
                job.clear(); job.update({"cand": cand, "reads": c["read"], "st": st, "nu": nub,
                                         "sums": {r: torch.zeros((4, NS, NI), dtype=torch.float64, device=dev) for r in c["read"]},
                                         "sz": {r: torch.zeros((n, len(sample)), device=dev) for r in c["read"]},
                                         "su": {r: torch.zeros((n, len(sample)), device=dev) for r in c["read"]},
                                         "pg": {r: torch.zeros((n, NP), device=dev) for r in c["read"]},
                                         "pu": {r: torch.zeros((n, NP), device=dev) for r in c["read"]}})
                if pkv is None:
                    model.model(input_ids=ids, use_cache=False)
                else:
                    c_ = copy.deepcopy(pkv); c_.batch_repeat_interleave(n)
                    model.model(input_ids=ids[:, off:], past_key_values=c_, use_cache=True); del c_
                for r in c["read"]:                                           # commit: swap state and buffers
                    if real:
                        st[r], nub[r] = nub[r], st[r]; sums[r] += job["sums"][r]
                    prof[r]["sz"][a0:a1] = job["sz"][r].half().cpu().numpy(); prof[r]["su"][a0:a1] = job["su"][r].half().cpu().numpy()
                    prof[r]["pg"][a0:a1] = job["pg"][r].half().cpu().numpy(); prof[r]["pu"][a0:a1] = job["pu"][r].half().cpu().numpy()
                job.clear()
                if (a0 // max(1, a1 - a0)) % 50 == 0:
                    log(f"  context {ki}: {a1:,}/{len(allowed):,} tokens, {time.time() - t0:.0f} s")
            Bc = max(16, min(batch, tok_budget // (max(1, T - off) * len(c["read"]))))
            guard = run_batched(len(allowed), Bc, step, label=f"context {ki}", device=str(dev), log=log)
            out = {}
            for r in c["read"]:
                if not real:
                    out[f"{r}_pseudo_g"] = prof[r]["pg"]; out[f"{r}_pseudo_u"] = prof[r]["pu"]; continue
                N = len(allowed); s_ = sums[r]
                gm = s_[0] / N; am = s_[2] / N
                out[f"{r}_mean"] = gm.float().cpu().numpy(); out[f"{r}_std"] = (s_[1] / N - gm ** 2).clamp_min(0).sqrt().float().cpu().numpy()
                out[f"{r}_amean"] = am.float().cpu().numpy(); out[f"{r}_astd"] = (s_[3] / N - am ** 2).clamp_min(0).sqrt().float().cpu().numpy()
                o = st[r]
                out[f"{r}_top_i"] = o["top"]["i"].cpu().numpy(); out[f"{r}_top_v"] = o["top"]["g"].cpu().numpy(); out[f"{r}_top_u"] = o["top"]["u"].cpu().numpy()
                for ls in ("apos", "aneg"):
                    out[f"{r}_{ls}_i"] = o[ls]["i"].cpu().numpy(); out[f"{r}_{ls}_g"] = o[ls]["g"].cpu().numpy(); out[f"{r}_{ls}_u"] = o[ls]["u"].cpu().numpy()
                out[f"{r}_sample_z"] = prof[r]["sz"]; out[f"{r}_sample_u"] = prof[r]["su"]
                out[f"{r}_pseudo_g"] = prof[r]["pg"]; out[f"{r}_pseudo_u"] = prof[r]["pu"]
            np.savez(D / f"vocab_sweep_{tag}_c{ki}.npz", **out, tokens=allowed, sample=np.array([f"L{l}:{i}" for l, i in sample], dtype=str),
                     pseudo=p_names, pseudo_layer=p_layer)
            dt = time.time() - t0
            meta["contexts"].append({"k": ki, "text": c["text"], "read": c["read"], "positions": pos, "batch_start": Bc, "gpu_guard": guard, "seconds": round(dt, 1)})
            meta["contexts"].sort(key=lambda x: x["k"]); meta["seconds"] = round(time.time() - t_all, 1)
            meta["gpu"] = torch.cuda.get_device_name(dev) if dev.type == "cuda" else dev.type
            mp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
            log(f"context {ki} ({c['text'][:50]!r}): {dt:.0f} s ({len(allowed) / max(dt, 1e-9):,.0f} tokens/s)")
    finally:
        for h in hs:
            h.remove()
        model.model.layers = full_layers
    log(f"sweep {tag} done in {time.time() - t_all:.0f} s")
    return meta


def layer_rows(meta: dict) -> dict:
    """layer -> row of the sweep's arrays (identity for a full sweep; meta["layers"] for a layer-scoped one)."""
    ls = meta.get("layers")
    return (lambda L: int(L)) if ls is None else {int(L): j for j, L in enumerate(ls)}.__getitem__


def copy_hub_tokens(out_dir, from_tag: str, to_tag: str) -> Path:
    """A pseudo-only sweep (real=False) has no favourites to count hub tokens from; use the real sweep's list for it,
    as the research run did (hub_tokens_v2q.npz was v2's)."""
    import shutil
    D = Path(out_dir)
    for ext in (".npz", ".txt"):
        src = D / f"hub_tokens_{from_tag}{ext}"
        if src.exists():
            shutil.copyfile(src, D / f"hub_tokens_{to_tag}{ext}")
    return D / f"hub_tokens_{to_tag}.npz"


def hub_tokens(out_dir, tag, vocab_size=None, *, tok=None, log=print) -> dict:
    """Hub (glitch) tokens: tokens that are favourites of very many unrelated neurons (research hub_tokens.py).

    Rare, under-trained vocabulary entries can produce extreme gate values and so turn up among the favourites of
    many unrelated neurons. Counts, for every token, how many neurons have it in their top-k favourites (the
    {r}_top_i lists), per reading (context x read position) and per layer. Writes hub_tokens_<tag>.npz (per_read
    [readings, vocab], per_layer [layers, vocab], readings) and hub_tokens_<tag>.txt (the top hubs, and how
    concentrated favourites are). vocab_size: the count arrays' width (the research run used the model's embedding
    size, 151,936 for Qwen3); default the largest swept token id + 1. tok: decodes tokens in the .txt (else ids)."""
    D = Path(out_dir)
    meta = json.loads((D / f"vocab_sweep_{tag}.json").read_text(encoding="utf-8"))
    paths = [(c, D / f"vocab_sweep_{tag}_c{c['k']}.npz") for c in meta["contexts"]]
    if not vocab_size:
        vocab_size = 0
        for _, p in paths:
            with np.load(p) as f:
                vocab_size = max(vocab_size, int(f["tokens"].max()) + 1)
    V = vocab_size
    names, per_read, per_layer, n_neur = [], [], None, 0
    for c, p in paths:
        with np.load(p) as f:
            for r in c["read"]:
                if f"{r}_top_i" not in f:
                    raise ValueError(f"{p.name} has no favourites (a pseudo-only sweep): no hub tokens")
                ti = f[f"{r}_top_i"]                               # [layers, neurons, k]
                if per_layer is None:
                    per_layer = np.zeros((ti.shape[0], V), np.int64); n_neur = ti.shape[0] * ti.shape[1]
                per_read.append(np.bincount(ti.reshape(-1), minlength=V)); names.append(f"c{c['k']}:{r}")
                for l in range(ti.shape[0]):
                    per_layer[l] += np.bincount(ti[l].reshape(-1), minlength=V)
    per_read = np.stack(per_read); tot = per_read.sum(0)
    nread = (per_read > 0.01 * n_neur).sum(0)                     # readings in which the token is a favourite of > 1% of neurons
    order = np.argsort(-tot)
    if tok is not None:
        dec = lambda j: tok.decode([int(j)]).replace("\n", "⏎").replace("\t", "⇥").replace("\r", "␍")   # noqa: E731
    else:
        dec = lambda j: f"#{int(j)}"                                                                     # noqa: E731
    share = tot / (len(names) * n_neur)
    lines = [f"{len(names)} readings x {n_neur:,} neurons x K={meta['K']}; a token in every neuron's list = share 1.0",
             f"tokens that are a favourite of > 1% of all neurons in at least one reading: {(per_read > 0.01 * n_neur).any(0).sum():,}",
             f"share of all favourite slots taken by the top 100 / 1,000 tokens: {tot[order[:100]].sum() / tot.sum():.1%} / {tot[order[:1000]].sum() / tot.sum():.1%}",
             "top 60 hub tokens (share of neurons per reading, readings where > 1%, layers where most common):"]
    for j in order[:60]:
        top_layers = [int(meta["layers"][x]) if meta.get("layers") else int(x) for x in np.argsort(-per_layer[:, j])[:3]]
        lines.append(f"  {dec(j)!r:>16} id {j:>6}: {share[j]:.2%}  in {nread[j]} readings  layers {top_layers}")
    out = {"per_read": per_read.astype(np.int32), "per_layer": per_layer.astype(np.int32), "readings": np.array(names)}
    if meta.get("layers") is not None:                    # a layer-scoped sweep: per_layer rows are these layers
        out["layers"] = np.array(meta["layers"], dtype=np.int64)
    np.savez_compressed(D / f"hub_tokens_{tag}.npz", **out)
    (D / f"hub_tokens_{tag}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("\n".join(lines))
    return out
