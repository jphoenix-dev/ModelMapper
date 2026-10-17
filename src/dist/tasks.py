"""run_task(store, task_id, ctx): one idempotent unit of work. Writes results/<task>/... and a _DONE marker last.

If _DONE exists the task is skipped. A task killed halfway leaves no _DONE, so a rerun recomputes it from the start
and overwrites any partial files (tasks are 10-30 min, so no mid-task checkpoints).

Task kinds
  observe:<shard>        forward pass over corpus/<shard>.npz. Per MLP neuron: running top-K and bottom-K firings
                         (value, doc, position, gate sign; as build_atlas.py), plus sum, sum of squares, count > 0 and
                         count |act| > t for each threshold t. Per attention head (its slice of o_proj's input): top-K
                         positions by output norm, plus sum and sum of squares of the norm. Raw activations are never
                         stored. Token 0 of each window is skipped (attention sink), as build_atlas.py.
                         -> L<ll>.npz per layer, heads.npz
  merge:observe          after every observe task: merge per-shard top-K lists and stats into per-neuron results
                         (mean, variance, share > 0, share |act| > t). -> L<ll>.npz per layer, heads.npz
  ablate:<L>:<a>-<b>:<ps>  neurons a..b (inclusive) of layer L, promptset promptsets/<ps>.json. The neuron-indexed
                         knockout of meaning_jobs.py, on contexts where the neuron FIRES: for each neuron its top n_pos
                         and bottom n_neg firings from merge:observe, plus n_ctrl random corpus positions shared by all
                         neurons in the task (control). Context = the `window` tokens ending at the firing position.
                         The neuron is set to 0 at that last position only. Every knockout row sits in the same forward
                         pass as an unablated copy of its context (bf16 results depend on batch shape, so a baseline
                         from another pass would add noise; map_loop batched-knockout v1 failed for this reason).
                         Per (neuron, context): its activation, KL(base || ablated) of the next token, the 20 tokens
                         raised and lowered most, and the change in the final residual (pre-final-norm, fp16; direct
                         part = act x down_proj column, recomputable). Per neuron: the static write-side readout
                         (down_proj column -> final norm -> unembedding; writer_lens.py), free.
                         -> ablate.npz, writer.npz

TODO (shortcuts not yet built):
  - start the knockout rows from the cached residual entering layer L (the rows match the base row up to there);
  - attention-head knockouts (ablate_heads:<L>:<ps>), as map_loop's o_proj slice zeroing;
  - attribution-patching triage to choose which neurons get the full knockout.
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import torch

from dist.manifest import task_key
from neurons import DEVICE, mlps


class Ctx:
    """What tasks need besides the store: the run config, and the model (loaded on first use)."""

    def __init__(self, store, worker_id="local"):
        from dist import runtime
        self.store, self.worker_id = store, worker_id
        self.cfg = store.get_json("config.json")
        self.env = runtime.environment()
        self._model = None
        self._corpus = None

    @property
    def model(self):
        if self._model is None:
            from dist import runtime
            t0 = time.time()
            self._model, self.tok, self.model_meta = runtime.load_model(self.cfg)
            self.model_meta["load_seconds"] = round(time.time() - t0, 1)
        return self._model

    def corpus(self):
        """All shards' tokens, indexed by global doc number: (tokens, starts, lengths)."""
        if self._corpus is None:
            toks, lens, doc0s = [], [], []
            for name in self.cfg["shards"]:
                z = self.store.get_npz(f"corpus/{name}.npz")
                toks.append(z["tokens"]); lens.append(z["lengths"]); doc0s.append(int(z["doc0"]))
            assert doc0s == sorted(doc0s)
            lens = np.concatenate(lens); toks = np.concatenate(toks)
            self._corpus = toks, np.concatenate([[0], np.cumsum(lens)[:-1]]), lens
        return self._corpus


def run_task(store, task_id: str, ctx: Ctx) -> dict | None:
    """Run one task unless its _DONE marker exists. Returns its meta, or None if it was already done."""
    key = task_key(task_id)
    if store.exists(key + "/_DONE"):
        return None
    kind = task_id.split(":")[0]
    t0 = time.time()
    info = RUNNERS[kind](store, task_id, ctx, key)
    sec = time.time() - t0
    meta = {"task": task_id, "worker": ctx.worker_id, "seconds": round(sec, 2),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **info,
            "tokens_per_s": round(info.get("tokens", 0) / sec, 1) if sec else None,
            "config_sha256": hashlib.sha256(json.dumps(ctx.cfg, sort_keys=True).encode()).hexdigest()[:16],
            "model": getattr(ctx, "model_meta", None), "env": ctx.env}
    store.put_json(key + "/meta.json", meta)
    store.put(key + "/_DONE", b"")          # last: a task is done only once everything above is written
    return meta


# ---- observe ----------------------------------------------------------------------------------------------------
def _batches(toks, lens, batch_tokens):
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    i = 0
    while i < len(lens):
        j, n = i, 0
        while j < len(lens) and (n < batch_tokens or j == i):
            n += lens[j]; j += 1
        yield i, [toks[starts[k]:starts[k] + lens[k]] for k in range(i, j)]
        i = j


def _topk_update(vals, docs, poss, gates, new_v, new_d, new_p, new_g, largest):
    """Running top-K along dim 1: vals [R, K] merged with new_v [R, N]."""
    K = vals.shape[1]
    v, idx = torch.cat([vals, new_v], 1).topk(K, dim=1, largest=largest)
    docs.copy_(torch.cat([docs, new_d], 1).gather(1, idx))
    poss.copy_(torch.cat([poss, new_p], 1).gather(1, idx))
    if gates is not None:
        gates.copy_(torch.cat([gates, new_g], 1).gather(1, idx))
    vals.copy_(v)


def observe(store, task_id, ctx, key):
    shard = task_id.split(":")[1]
    oc = ctx.cfg["observe"]
    K, BT, TH = oc["k"], oc["batch_tokens"], torch.tensor(oc["thresholds"], device=DEVICE)
    z = store.get_npz(f"corpus/{shard}.npz")
    toks, lens, doc0 = z["tokens"], z["lengths"], int(z["doc0"])
    model = ctx.model
    c = model.config
    L, W, H = c.num_hidden_layers, c.intermediate_size, c.num_attention_heads
    D = getattr(c, "head_dim", c.hidden_size // H)
    f32, i32, i16 = torch.float32, torch.int32, torch.int16
    S = {p: {"v": torch.full((L, W, K), -float("inf") if p == "top" else float("inf"), device=DEVICE),
             "d": torch.full((L, W, K), -1, dtype=i32, device=DEVICE), "p": torch.zeros((L, W, K), dtype=i16, device=DEVICE),
             "g": torch.zeros((L, W, K), dtype=torch.bool, device=DEVICE)} for p in ("top", "bot")}
    s1 = torch.zeros((L, W), dtype=torch.float64, device=DEVICE); s2 = torch.zeros_like(s1)
    npos = torch.zeros((L, W), dtype=torch.int64, device=DEVICE)
    nabs = torch.zeros((len(TH), L, W), dtype=torch.int64, device=DEVICE)
    hv = torch.full((L, H, K), -float("inf"), device=DEVICE)
    hd = torch.full((L, H, K), -1, dtype=i32, device=DEVICE); hp = torch.zeros((L, H, K), dtype=i16, device=DEVICE)
    h1 = torch.zeros((L, H), dtype=torch.float64, device=DEVICE); h2 = torch.zeros_like(h1)
    st = {}

    def gate_hook(l):
        def h(m, args, out):
            st["gate"] = out[st["valid"]] >= 0                                   # [N, W]
        return h

    def act_hook(l):
        def h(m, args):
            a = args[0][st["valid"]].to(f32)                                     # [N, W]
            s1[l] += a.sum(0, dtype=torch.float64); s2[l] += (a * a).sum(0, dtype=torch.float64)
            npos[l] += (a > 0).sum(0)
            nabs[:, l] += (a.abs()[None] > TH[:, None, None]).sum(1)
            at, g = a.T, st.pop("gate").T                                        # [W, N]
            d, p = st["doc"].expand(W, -1), st["pos"].expand(W, -1)
            for pole, largest in (("top", True), ("bot", False)):
                b = S[pole]
                vv, dd, pp, gg = b["v"][l], b["d"][l], b["p"][l], b["g"][l]
                _topk_update(vv, dd, pp, gg, at, d, p, g, largest)
        return h

    def head_hook(l):
        def h(m, args):
            x = args[0][st["valid"]].to(f32)
            nrm = x.view(x.shape[0], H, D).norm(dim=-1)                          # [N, H]
            h1[l] += nrm.sum(0, dtype=torch.float64); h2[l] += (nrm * nrm).sum(0, dtype=torch.float64)
            _topk_update(hv[l], hd[l], hp[l], None, nrm.T, st["doc"].expand(H, -1), st["pos"].expand(H, -1), None, True)
        return h

    hooks = []
    for l in range(L):
        hooks.append(mlps(model)[l].gate_proj.register_forward_hook(gate_hook(l)))
        hooks.append(mlps(model)[l].down_proj.register_forward_pre_hook(act_hook(l)))
        hooks.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(head_hook(l)))
    n_tok = 0
    try:
        for i0, docs in _batches(toks, lens, BT):
            T = max(map(len, docs))
            ids = torch.zeros((len(docs), T), dtype=torch.long); mask = torch.zeros_like(ids)
            for i, x in enumerate(docs):
                ids[i, :len(x)] = torch.as_tensor(x.astype(np.int64)); mask[i, :len(x)] = 1
            valid = mask.bool(); valid[:, 0] = False                              # token 0 is an attention sink
            dgrid = (torch.arange(len(docs))[:, None] + doc0 + i0).expand(-1, T)
            pgrid = torch.arange(T)[None].expand(len(docs), -1)
            st["valid"] = valid.to(DEVICE)
            st["doc"] = dgrid[valid].to(DEVICE, i32)[None]
            st["pos"] = pgrid[valid].to(DEVICE, i16)[None]
            with torch.no_grad():
                model.model(input_ids=ids.to(DEVICE), attention_mask=mask.to(DEVICE))
            n_tok += int(valid.sum())
    finally:
        for h in hooks:
            h.remove()
    cpu = lambda t: t.cpu().numpy()  # noqa: E731
    for l in range(L):
        store.put_npz(f"{key}/L{l:02d}.npz",
                      top_v=cpu(S["top"]["v"][l]), top_d=cpu(S["top"]["d"][l]), top_p=cpu(S["top"]["p"][l]), top_g=cpu(S["top"]["g"][l]),
                      bot_v=cpu(S["bot"]["v"][l]), bot_d=cpu(S["bot"]["d"][l]), bot_p=cpu(S["bot"]["p"][l]), bot_g=cpu(S["bot"]["g"][l]),
                      sum=cpu(s1[l]), sumsq=cpu(s2[l]), n_pos=cpu(npos[l]), n_abs_gt=cpu(nabs[:, l]),
                      thresholds=np.array(oc["thresholds"], dtype=np.float32), n_tokens=np.array(n_tok))
    store.put_npz(f"{key}/heads.npz", top_v=cpu(hv), top_d=cpu(hd), top_p=cpu(hp), sum=cpu(h1), sumsq=cpu(h2),
                  n_tokens=np.array(n_tok))
    return {"tokens": n_tok, "docs": int(len(lens)), "shard": shard}


# ---- merge:observe ----------------------------------------------------------------------------------------------
def _merge_topk(vs, ds, ps, gs, K, largest):
    v = np.concatenate(vs, 1); d = np.concatenate(ds, 1); p = np.concatenate(ps, 1)
    g = np.concatenate(gs, 1) if gs is not None else None
    order = np.argsort(-v if largest else v, axis=1, kind="stable")[:, :K]
    take = lambda a: np.take_along_axis(a, order, 1)  # noqa: E731
    return take(v), take(d), take(p), (take(g) if g is not None else None)


def merge_observe(store, task_id, ctx, key):
    tasks = [t for t in store.get_json("manifest.json")["tasks"] if t["id"].startswith("observe:")]
    keys = [task_key(t["id"]) for t in tasks]
    missing = [t["id"] for t, k in zip(tasks, keys) if not store.exists(k + "/_DONE")]
    if missing:
        raise RuntimeError(f"merge:observe run before {len(missing)} observe tasks finished: {missing[:5]}")
    K = ctx.cfg["observe"]["k"]
    n_tok = 0; layer = 0
    while True:
        parts = [store.get_npz(f"{k}/L{layer:02d}.npz") for k in keys]
        if parts[0] is None:
            break
        out = {}
        for pole, largest in (("top", True), ("bot", False)):
            v, d, p, g = _merge_topk([x[f"{pole}_v"] for x in parts], [x[f"{pole}_d"] for x in parts],
                                     [x[f"{pole}_p"] for x in parts], [x[f"{pole}_g"] for x in parts], K, largest)
            out.update({f"{pole}_v": v, f"{pole}_d": d, f"{pole}_p": p, f"{pole}_g": g})
        n = sum(int(x["n_tokens"]) for x in parts); n_tok = n
        s1 = sum(x["sum"] for x in parts); s2 = sum(x["sumsq"] for x in parts)
        mean = s1 / n
        out.update(mean=mean, var=np.maximum(s2 / n - mean * mean, 0.0),
                   frac_pos=sum(x["n_pos"] for x in parts) / n, frac_abs_gt=sum(x["n_abs_gt"] for x in parts) / n,
                   thresholds=parts[0]["thresholds"], n_tokens=np.array(n))
        store.put_npz(f"{key}/L{layer:02d}.npz", **out)
        layer += 1
    hp = [store.get_npz(f"{k}/heads.npz") for k in keys]
    v, d, p, _ = _merge_topk([x["top_v"].reshape(-1, K) for x in hp], [x["top_d"].reshape(-1, K) for x in hp],
                             [x["top_p"].reshape(-1, K) for x in hp], None, K, True)
    L, H = hp[0]["sum"].shape
    n = sum(int(x["n_tokens"]) for x in hp); m = sum(x["sum"] for x in hp) / n
    store.put_npz(f"{key}/heads.npz", top_v=v.reshape(L, H, K), top_d=d.reshape(L, H, K), top_p=p.reshape(L, H, K),
                  mean=m, var=np.maximum(sum(x["sumsq"] for x in hp) / n - m * m, 0.0), n_tokens=np.array(n))
    return {"tokens": 0, "layers": layer, "shards": len(keys), "corpus_tokens": n_tok}


# ---- ablate -----------------------------------------------------------------------------------------------------
def _run_rows(model, layer, rows, nb):
    """rows: [(token ids, neuron to zero at the last position or -1)]. One forward pass.
    Returns next-token logits [B, V], final residual before the final norm [B, d], and every neuron's activation in
    `nb` (a slice of layer `layer`) at the last position [B, len(nb)] - before any knockout."""
    B = len(rows); T = max(len(x) for x, _ in rows)
    ids = torch.zeros((B, T), dtype=torch.long); mask = torch.zeros_like(ids)
    for i, (x, _) in enumerate(rows):                       # right padding: causal attention leaves real tokens untouched
        ids[i, :len(x)] = torch.as_tensor(np.asarray(x, dtype=np.int64)); mask[i, :len(x)] = 1
    last = torch.tensor([len(x) - 1 for x, _ in rows], device=DEVICE)
    ar = torch.arange(B, device=DEVICE)
    kr = torch.tensor([i for i, (_, n) in enumerate(rows) if n >= 0], device=DEVICE, dtype=torch.long)
    kn = torch.tensor([n for _, n in rows if n >= 0], device=DEVICE, dtype=torch.long)
    st = {}

    def knock(m, args):
        a = args[0]
        st["act"] = a[ar, last][:, nb].float()
        if len(kr):
            a = a.clone(); a[kr, last[kr], kn] = 0
            return (a,)

    def resid(m, args):
        st["resid"] = args[0][ar, last].float()

    hk = [mlps(model)[layer].down_proj.register_forward_pre_hook(knock),
          model.model.norm.register_forward_pre_hook(resid)]
    try:
        with torch.no_grad():
            hs = model.model(input_ids=ids.to(DEVICE), attention_mask=mask.to(DEVICE)).last_hidden_state[ar, last]
            logits = model.lm_head(hs).float()
    finally:
        for h in hk:
            h.remove()
    return logits, st["resid"], st["act"], int(mask.sum())


def ablate(store, task_id, ctx, key):
    _, L, rng_, ps = task_id.split(":")
    L = int(L); a, b = (int(x) for x in rng_.split("-"))
    P = store.get_json(f"promptsets/{ps}.json")
    win, ntop = P["window"], P["n_tokens"]
    merged = store.get_npz(f"{task_key('merge:observe')}/L{L:02d}.npz")
    if merged is None:
        raise RuntimeError("merge:observe has not run")
    toks, starts, lens = ctx.corpus()
    model = ctx.model
    neurons = np.arange(a, b + 1)
    nb = torch.as_tensor(neurons, device=DEVICE)

    def window(d, p):
        lo = max(0, p - win + 1)
        return toks[starts[d] + lo: starts[d] + p + 1]

    # contexts: (neuron offset or -1 for 'all neurons' (control), kind, doc, pos, value seen in observe)
    groups = []
    for i, n in enumerate(neurons):
        for pole, kind, cnt in (("top", 0, P["n_pos"]), ("bot", 1, P["n_neg"])):
            for j in range(cnt):
                d, p = int(merged[f"{pole}_d"][n, j]), int(merged[f"{pole}_p"][n, j])
                if d >= 0:
                    groups.append((i, kind, d, p, float(merged[f"{pole}_v"][n, j])))
    r = np.random.default_rng([P["seed"], L, a])
    for _ in range(P["n_ctrl"]):
        d = int(r.integers(len(lens)))
        groups.append((-1, 2, d, int(r.integers(1, lens[d])), float("nan")))

    # rows: each chunk = one base row + knockout rows of that same context, all in one forward pass
    chunks = []
    for gi, (i, kind, d, p, _) in enumerate(groups):
        targets = [i] if i >= 0 else list(range(len(neurons)))
        step = P["batch_rows"] - 1
        for c0 in range(0, len(targets), step):
            chunks.append((gi, targets[c0:c0 + step]))
    rec = {k: [] for k in ("neuron", "kind", "doc", "pos", "obs_act", "act", "kl", "base_top1", "window",
                           "up_ids", "up_d", "dn_ids", "dn_d", "total")}
    n_tok = 0
    batch, owners = [], []

    def flush():
        nonlocal n_tok, batch, owners
        if not batch:
            return
        logits, res, acts, nt = _run_rows(model, L, batch, nb)
        n_tok += nt
        lp = logits.log_softmax(-1)
        for base_r, gi, kr in owners:
            i, kind, d, p, ov = groups[gi]
            for k, ti in enumerate(kr):
                ar_ = base_r + 1 + k
                dlp = lp[base_r] - lp[ar_]
                up, dn = dlp.topk(ntop), (-dlp).topk(ntop)
                rec["neuron"].append(int(neurons[ti])); rec["kind"].append(kind); rec["doc"].append(d); rec["pos"].append(p)
                rec["obs_act"].append(ov); rec["act"].append(acts[base_r, ti].item())
                rec["kl"].append((lp[base_r].exp() * dlp).sum().item())
                rec["base_top1"].append(int(lp[base_r].argmax())); rec["window"].append(len(batch[base_r][0]))
                rec["up_ids"].append(up.indices.cpu().numpy()); rec["up_d"].append(up.values.cpu().numpy())
                rec["dn_ids"].append(dn.indices.cpu().numpy()); rec["dn_d"].append(-dn.values.cpu().numpy())
                rec["total"].append((res[base_r] - res[ar_]).half().cpu().numpy())
        batch, owners = [], []

    for gi, kr in chunks:
        i, kind, d, p, _ = groups[gi]
        if len(batch) + 1 + len(kr) > P["batch_rows"]:
            flush()
        x = window(d, p)
        owners.append((len(batch), gi, [neurons_i for neurons_i in kr]))
        batch.append((x, -1))
        batch += [(x, int(neurons[ti])) for ti in kr]
    flush()
    out = {k: np.array(v) for k, v in rec.items()}
    out["kind"] = out["kind"].astype(np.int8); out["neuron"] = out["neuron"].astype(np.int32)
    out["kl"] = out["kl"].astype(np.float32); out["act"] = out["act"].astype(np.float32)
    store.put_npz(f"{key}/ablate.npz", **out)
    # static write-side readout (writer_lens.py): down_proj column -> final-norm gain -> unembedding, minus the mean
    with torch.no_grad():
        dcol = mlps(model)[L].down_proj.weight[:, nb].float() * model.model.norm.weight.float()[:, None]
        s = model.lm_head.weight.float() @ dcol                              # [V, n]
        s = (s - s.mean(0, keepdim=True)).T
        up, dn = s.topk(ntop, dim=1), (-s).topk(ntop, dim=1)
    store.put_npz(f"{key}/writer.npz", neuron=neurons.astype(np.int32), up_ids=up.indices.cpu().numpy(),
                  up_s=up.values.cpu().numpy(), dn_ids=dn.indices.cpu().numpy(), dn_s=-dn.values.cpu().numpy(),
                  std=s.std(1).cpu().numpy())
    return {"tokens": n_tok, "rows": len(rec["kl"]), "contexts": len(groups), "neurons": len(neurons)}


RUNNERS = {"observe": observe, "merge": merge_observe, "ablate": ablate}
