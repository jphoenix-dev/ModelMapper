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
                         Two knockout modes (promptset "modes"): LAST = the firing token only; PASSAGE = every token in
                         the window where the neuron fires (>= half its median top firing, or <= half its median bottom
                         firing; the same thresholds on control contexts).
                         Noise floor (promptset "n_null"): n_null firing contexts are rerun as (base, knockout) pairs
                         alone; KL between the batched and the lone base = what batch shape alone does (bf16 rounding).
                         neuron_summary.npz: per neuron, median KL per mode and context kind, and the share above this
                         task's floor (single-token knockouts on 8B are mostly near it: results/smoke_test_report.md).
                         Per (neuron, context): its activation, KL(base || ablated) of the next token, the 20 tokens
                         raised and lowered most, and the change in the final residual (pre-final-norm, fp16; direct
                         part = act x down_proj column, recomputable). Per neuron: the static write-side readout
                         (down_proj column -> final norm -> unembedding; writer_lens.py), free.
                         -> ablate.npz, neuron_summary.npz, writer.npz

  map:<batch>            concept-indexed mapping, loop v2 (src/loop_v2/map_core.py, by the research branch): every
                         concept spec in specs/<batch>/ (each with a "concept" key) goes through map_batch on the pinned
                         model. Output: records.jsonl (one record per concept, provenance inside), arrays/<slug>.npz,
                         meta with status counts and failed concepts. Per-concept failures are records, not task
                         failures, unless more than half the batch fails (then raise: no _DONE, another worker retries).
                         Needs RECALL_VARIANT=V3 (config.json "env", applied before pipeline_fast is imported).

TODO (shortcuts not yet built):
  - start the knockout rows from the cached residual entering layer L (the rows match the base row up to there);
  - attention-head knockouts (ablate_heads:<L>:<ps>), as map_loop's o_proj slice zeroing;
  - attribution-patching triage to choose which neurons get the full knockout.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from modelmapper.dist.manifest import task_key
from modelmapper.core.neurons import DEVICE, mlps


class Ctx:
    """What tasks need besides the store: the run config, and the model (loaded on first use)."""

    def __init__(self, store, worker_id="local"):
        from modelmapper.dist import runtime
        self.store, self.worker_id = store, worker_id
        self.cfg = store.get_json("config.json")
        self.env = runtime.environment()
        for k, v in (self.cfg.get("env") or {}).items():       # e.g. RECALL_VARIANT=V3, read by modules at import
            os.environ.setdefault(k, str(v))
        self._model = None
        self._pipe = None
        self._corpus = None

    @property
    def model(self):
        if self._model is None:
            from modelmapper.dist import runtime
            t0 = time.time()
            self._model, self.tok, self.model_meta = runtime.load_model(self.cfg)
            self.model_meta["load_seconds"] = round(time.time() - t0, 1)
        return self._model

    def pipe(self):
        """loop v2's FastPipeline around the pinned model (built once). Uses model=/tok= if Pipeline accepts them, else
        sets the attributes Pipeline.__init__ would, without loading a second copy of the model."""
        if self._pipe is None:
            from modelmapper.core.pipeline_fast import FastPipeline
            model = self.model
            try:
                self._pipe = FastPipeline(self.cfg["model"], mode="add", reply_only=True, model=model, tok=self.tok)
            except TypeError:
                p = FastPipeline.__new__(FastPipeline)
                p.model_name, p.model, p.tok = self.cfg["model"], model, self.tok
                p.model.requires_grad_(False)
                p.L, p.W = model.config.num_hidden_layers, model.config.intermediate_size
                p.stats, p.atlas, p.mode, p.reply_only = None, None, "add", True
                self._pipe = p
        return self._pipe

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
    if DEVICE.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    io0 = dict(store.io or {"get_s": 0.0, "get_bytes": 0, "put_s": 0.0, "put_bytes": 0})
    t0 = time.time()
    info = RUNNERS[kind](store, task_id, ctx, key)
    sec = time.time() - t0
    io1 = store.io or io0
    io = {k: io1[k] - io0[k] for k in io0}
    info.update(start_epoch=round(t0, 2), end_epoch=round(time.time(), 2), io_get_s=round(io["get_s"], 2),
                io_get_mb=round(io["get_bytes"] / 1e6, 1), io_put_s=round(io["put_s"], 2), io_put_mb=round(io["put_bytes"] / 1e6, 1),
                compute_s=round(sec - io["get_s"] - io["put_s"], 2))
    if DEVICE.startswith("cuda"):     # what a 24 GB card must hold: allocated is the real need, reserved includes cache
        info["gpu_peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        info["gpu_peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
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
LAST, PASSAGE = 0, 1           # knockout modes: the firing token only / every token in the window where the neuron fires


def _run_rows(model, layer, rows, nb, thr=None):
    """rows: [(token ids, neuron or -1, mode)] (mode optional, default LAST). One forward pass.
    mode LAST zeroes the neuron at the last token. mode PASSAGE zeroes it at every token of the window where it fires:
    activation >= thr[n][0] or <= thr[n][1] (the row's own pre-knockout activations, which equal the base row's).
    Returns next-token logits [B, V], final residual before the final norm [B, d], every neuron in `nb` (a slice of
    layer `layer`) at the last position before any knockout [B, len(nb)], tokens processed, positions zeroed per row."""
    rows = [r if len(r) == 3 else (r[0], r[1], LAST) for r in rows]
    B = len(rows); T = max(len(x) for x, _, _ in rows)
    ids = torch.zeros((B, T), dtype=torch.long); mask = torch.zeros_like(ids)
    for i, (x, _, _) in enumerate(rows):                    # right padding: causal attention leaves real tokens untouched
        ids[i, :len(x)] = torch.as_tensor(np.asarray(x, dtype=np.int64)); mask[i, :len(x)] = 1
    lens = torch.tensor([len(x) for x, _, _ in rows], device=DEVICE)
    last = lens - 1
    ar = torch.arange(B, device=DEVICE)
    tpos = torch.arange(T, device=DEVICE)
    k0 = [(i, n) for i, (_, n, md) in enumerate(rows) if n >= 0 and md == LAST]
    k1 = [(i, n) for i, (_, n, md) in enumerate(rows) if n >= 0 and md == PASSAGE]
    r0 = torch.tensor([i for i, _ in k0], device=DEVICE, dtype=torch.long)
    n0 = torch.tensor([n for _, n in k0], device=DEVICE, dtype=torch.long)
    r1 = torch.tensor([i for i, _ in k1], device=DEVICE, dtype=torch.long)
    n1 = torch.tensor([n for _, n in k1], device=DEVICE, dtype=torch.long)
    if k1:
        hi = torch.tensor([thr[n][0] for _, n in k1], device=DEVICE)
        lo = torch.tensor([thr[n][1] for _, n in k1], device=DEVICE)
    zeroed = torch.zeros(B, dtype=torch.long, device=DEVICE)
    st = {}

    def knock(m, args):
        a = args[0]
        st["act"] = a[ar, last][:, nb].float()
        if len(r0) or len(r1):
            a = a.clone()
            if len(r0):
                a[r0, last[r0], n0] = 0
                zeroed[r0] = 1
            if len(r1):
                v = a[r1[:, None], tpos[None], n1[:, None]].float()                          # [R, T]
                hit = ((v >= hi[:, None]) | (v <= lo[:, None])) & (tpos[None] < lens[r1][:, None])
                rr, tt = hit.nonzero(as_tuple=True)
                a[r1[rr], tt, n1[rr]] = 0
                zeroed[r1] = hit.sum(1)
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
    return logits, st["resid"], st["act"], int(mask.sum()), zeroed.cpu().numpy()


def _kl(lp_base, lp_other):
    return (lp_base.exp() * (lp_base - lp_other)).sum(-1)


def ablate(store, task_id, ctx, key):
    _, L, rng_, ps = task_id.split(":")
    L = int(L); a, b = (int(x) for x in rng_.split("-"))
    P = store.get_json(f"promptsets/{ps}.json")
    win, ntop = P["window"], P["n_tokens"]
    modes = P.get("modes", [LAST]); n_null = P.get("n_null", 0)
    merged = store.get_npz(f"{task_key('merge:observe')}/L{L:02d}.npz")
    if merged is None:
        raise RuntimeError("merge:observe has not run")
    toks, starts, lens = ctx.corpus()
    model = ctx.model
    neurons = np.arange(a, b + 1)
    nb = torch.as_tensor(neurons, device=DEVICE)
    # PASSAGE threshold per neuron: half the median of its top (bottom) firings; the same on firing and control contexts
    thr = {}
    for n in neurons:
        hi = 0.5 * float(np.median(merged["top_v"][n, :max(P["n_pos"], 1)]))
        lo = 0.5 * float(np.median(merged["bot_v"][n, :max(P["n_neg"], 1)]))
        thr[int(n)] = (hi if hi > 0 else float("inf"), lo if lo < 0 else -float("inf"))

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
    firing = [gi for gi, g in enumerate(groups) if g[1] == 0]
    null_g = set(firing[::max(1, len(firing) // n_null)][:n_null]) if n_null else set()

    # rows: each chunk = one base row + knockout rows of that same context, all in one forward pass
    chunks = []
    for gi, (i, kind, d, p, _) in enumerate(groups):
        targets = [(ti, md) for ti in ([i] if i >= 0 else range(len(neurons))) for md in modes]
        step = P["batch_rows"] - 1
        for c0 in range(0, len(targets), step):
            chunks.append((gi, targets[c0:c0 + step]))
    rec = {k: [] for k in ("neuron", "kind", "mode", "n_zeroed", "doc", "pos", "obs_act", "act", "kl", "base_top1",
                           "window", "up_ids", "up_d", "dn_ids", "dn_d", "total")}
    null_base = {}
    n_tok = 0
    batch, owners = [], []

    def flush():
        nonlocal n_tok, batch, owners
        if not batch:
            return
        logits, res, acts, nt, zeroed = _run_rows(model, L, batch, nb, thr)
        n_tok += nt
        lp = logits.log_softmax(-1)
        for base_r, gi, kr in owners:
            i, kind, d, p, ov = groups[gi]
            if gi in null_g and gi not in null_base:
                null_base[gi] = lp[base_r].clone()
            for k, (ti, md) in enumerate(kr):
                ar_ = base_r + 1 + k
                dlp = lp[base_r] - lp[ar_]
                up, dn = dlp.topk(ntop), (-dlp).topk(ntop)
                rec["neuron"].append(int(neurons[ti])); rec["kind"].append(kind); rec["mode"].append(md)
                rec["n_zeroed"].append(int(zeroed[ar_])); rec["doc"].append(d); rec["pos"].append(p)
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
        owners.append((len(batch), gi, kr))
        batch.append((x, -1, LAST))
        batch += [(x, int(neurons[ti]), md) for ti, md in kr]
    flush()
    out = {k: np.array(v) for k, v in rec.items()}
    for k, t in (("kind", np.int8), ("mode", np.int8), ("neuron", np.int32), ("n_zeroed", np.int16),
                 ("kl", np.float32), ("act", np.float32)):
        out[k] = out[k].astype(t)

    # noise floor: some firing contexts again, each (base, LAST knockout) pair alone in its own pass (batch 2).
    #   null_kl          KL(base in the task's batch || base alone): what batch shape alone does (bf16 rounding)
    #   null_ko_batched / null_ko_alone   the same LAST knockout's KL in the batch and alone: stability of one estimate
    nn_, nd, npos, nkl, nkb, nka = [], [], [], [], [], []
    for gi in sorted(null_base):
        i, kind, d, p, _ = groups[gi]
        n = int(neurons[i]); x = window(d, p)
        logits, _, _, nt, _ = _run_rows(model, L, [(x, -1, LAST), (x, n, LAST)], nb, thr)
        n_tok += nt
        lp = logits.log_softmax(-1)
        sel = np.where((out["neuron"] == n) & (out["doc"] == d) & (out["pos"] == p) & (out["mode"] == LAST)
                       & (out["kind"] == kind))[0]
        nn_.append(n); nd.append(d); npos.append(p)
        nkl.append(_kl(null_base[gi], lp[0]).item()); nka.append(_kl(lp[0], lp[1]).item())
        nkb.append(float(out["kl"][sel[0]]))
    out.update(null_neuron=np.array(nn_, dtype=np.int32), null_doc=np.array(nd, dtype=np.int32),
               null_pos=np.array(npos, dtype=np.int32), null_kl=np.array(nkl, dtype=np.float32),
               null_ko_batched=np.array(nkb, dtype=np.float32), null_ko_alone=np.array(nka, dtype=np.float32))
    store.put_npz(f"{key}/ablate.npz", **out)

    # per neuron: median KL by mode and context kind, and the share above this task's noise floor (null_kl p90)
    floor = float(np.quantile(nkl, 0.9)) if nkl else float("nan")
    summ = {"neuron": neurons.astype(np.int32), "floor_p90": np.array(floor),
            "floor_median": np.array(float(np.median(nkl)) if nkl else np.nan)}
    for md, mname in ((LAST, "last"), (PASSAGE, "passage")):
        if md not in modes:
            continue
        for kind, kname in ((0, "top"), (1, "bot"), (2, "ctrl")):
            med, share = [], []
            for n in neurons:
                kl_ = out["kl"][(out["neuron"] == n) & (out["mode"] == md) & (out["kind"] == kind)]
                med.append(np.median(kl_) if len(kl_) else np.nan)
                share.append(np.mean(kl_ > floor) if len(kl_) else np.nan)
            summ[f"{mname}_{kname}_kl_median"] = np.array(med, dtype=np.float32)
            summ[f"{mname}_{kname}_share_above_floor"] = np.array(share, dtype=np.float32)
    store.put_npz(f"{key}/neuron_summary.npz", **summ)

    # static write-side readout (writer_lens.py): down_proj column -> final-norm gain -> unembedding, minus the mean
    with torch.no_grad():
        dcol = mlps(model)[L].down_proj.weight[:, nb].float() * model.model.norm.weight.float()[:, None]
        s_ = model.lm_head.weight.float() @ dcol                              # [V, n]
        s_ = (s_ - s_.mean(0, keepdim=True)).T
        up, dn = s_.topk(ntop, dim=1), (-s_).topk(ntop, dim=1)
    store.put_npz(f"{key}/writer.npz", neuron=neurons.astype(np.int32), up_ids=up.indices.cpu().numpy(),
                  up_s=up.values.cpu().numpy(), dn_ids=dn.indices.cpu().numpy(), dn_s=-dn.values.cpu().numpy(),
                  std=s_.std(1).cpu().numpy())
    info = {"tokens": n_tok, "rows": len(rec["kl"]), "contexts": len(groups), "neurons": len(neurons), "modes": modes}
    if nkl:
        lt = out["kl"][(out["mode"] == LAST) & (out["kind"] == 0)]
        info.update(noise_floor_median=float(np.median(nkl)), noise_floor_p90=floor,
                    last_top_kl_median=float(np.median(lt)), last_top_share_above_floor=float(np.mean(lt > floor)))
        if PASSAGE in modes:
            pt = out["kl"][(out["mode"] == PASSAGE) & (out["kind"] == 0)]
            info.update(passage_top_kl_median=float(np.median(pt)), passage_top_share_above_floor=float(np.mean(pt > floor)))
    return info


# ---- map (loop v2) ----------------------------------------------------------------------------------------------
def map_task(store, task_id, ctx, key):
    batch = task_id.split(":")[1]
    specs = {}
    for k in sorted(store.list(f"specs/{batch}/")):
        s = store.get_json(k)
        specs[s["concept"]] = s
    if not specs:
        raise RuntimeError(f"no specs under specs/{batch}/")
    from modelmapper.loop.map_core import map_batch
    pipe = ctx.pipe()
    with tempfile.TemporaryDirectory() as td:          # a fresh start every time: no partial resume
        recs = list(map_batch(pipe, specs, td, code_commit=ctx.env.get("git_commit") or "unknown"))
        status = collections.Counter((r.get("row") or {}).get("status", "?") for r in recs)
        failed = [r["concept"] for r in recs if (r.get("row") or {}).get("status") == "failed"]
        if recs and len(failed) > len(recs) / 2:
            raise RuntimeError(f"{len(failed)}/{len(recs)} concepts failed (systemic?): "
                               f"{[(r['concept'], r['row'].get('detail')) for r in recs if r['concept'] in failed[:3]]}")
        n_bytes = 0
        for f in sorted(Path(td).rglob("*")):
            if f.is_file():
                b = f.read_bytes(); n_bytes += len(b)
                store.put(f"{key}/{f.relative_to(td).as_posix()}", b)
    return {"tokens": 0, "concepts": len(recs), "status_counts": dict(status), "failed": failed,
            "output_mb": round(n_bytes / 1e6, 1), "env_vars": {k: os.environ.get(k) for k in (ctx.cfg.get("env") or {})}}


from modelmapper.dist.attrib_task import attrib_task  # noqa: E402
from modelmapper.dist.jlens_task import jlens_task  # noqa: E402
from modelmapper.dist.links import links_task  # noqa: E402

RUNNERS = {"observe": observe, "merge": merge_observe, "ablate": ablate, "map": map_task, "links": links_task, "jlens": jlens_task, "attrib": attrib_task}
