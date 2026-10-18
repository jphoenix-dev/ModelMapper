"""Attribution graphs on Qwen3's own MLP neurons (map session's ATTRIB_GRAPHS.md, written before data; James asked 18 Oct).

Why not circuit-tracer: its transcoder features are activation_fn(encoder . x); a SwiGLU neuron is SiLU(gate . x) * (up . x),
and with the gate frozen its slope differs per position - not expressible as a fixed encoder. So the same algorithm is
implemented directly (the method's stated fallback), on the model's own weights.

Linearised model (per question): our own forward pass over the HF weights in which the attention patterns, every RMSNorm
scale and each neuron's SiLU(gate) are FROZEN at their values on this question. The residual stream is then linear in
the token embeddings and in each neuron's value part, so contributions add up exactly.
  target   = logit(answer's first token) - mean logit, at the last position
  nodes    = (layer, pos, neuron) ranked by |activation x gradient of the target| (total effect, through everything);
             keep the largest covering 80% of the total |attribution|, cap 300; + one embedding node per position
  edges    = DIRECT contributions: every MLP's activations become fixed leaves, so a backward pass from a kept node's
             activation (or the target) reaches each source only through residual + frozen attention. Batched: B copies
             of the question, one node objective per row. Edges kept per destination: largest |w| covering 98% of inflow
  gates    (1) our clean forward reproduces the HF logits; (2) direct contributions of ALL neurons + embeddings sum to the
             real target (difference < 1%); (3) real single-node knockouts (HF model, zero at that position) track the
             attributions: Spearman over the top-30 nodes >= 0.7 (also reported with 30 random active nodes)
  step 3   (ablate) for groups.json: zero each group's neurons at their positions; answer probability vs 3 random sets of
             the same size matched on layers and positions
Output: D:/already-clean-data/attrib_graphs/<slug>/graph.json, ablations.json
Usage: python src/loop_v2/attrib_graph.py graphs [--model Qwen/Qwen3-0.6B --out <dir> --limit N] | ablate
"""
import json
import math
import os
import random
import re
import statistics as stx
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as Fn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
DATA = Path(os.environ.get("AC_DATA_DIR", "D:/already-clean-data")) / "attrib_graphs"
KEEP_FRAC, CAP, EDGE_FRAC, ROWS = 0.80, 300, 0.98, int(os.environ.get("AG_ROWS", "16"))


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:80]


class Lin:
    """Our own Qwen3 forward over the HF weights. mode: 'clean' (records the frozen quantities), 'total' (frozen;
    activations a = sg * up, gradients flow everywhere), 'direct' (frozen; each layer's activations are given leaves)."""

    def __init__(self, model):
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv
        self.m, self.rot, self.rep = model, apply_rotary_pos_emb, repeat_kv
        c = model.config; self.L, self.H, self.Hkv, self.dh, self.eps = c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.rms_norm_eps
        self.layers = model.model.layers

    @staticmethod
    def lin(x, w):
        return Fn.linear(x.to(w.dtype), w).float()

    def norm(self, x, w, inv=None):
        if inv is None:
            inv = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * inv * w.float(), inv

    def forward(self, ids, mode, fz=None, leaves=None, ans=None, objective=None):
        """ids [B,T]. Returns dict: target [B], logits_last [B,V] (clean), fz (clean), a (per layer [B,T,W]), h2 (direct),
        emb (leaf). objective(out) -> scalar is optional (direct mode node objectives)."""
        B, T = ids.shape; dev = ids.device
        emb = self.m.model.embed_tokens(ids).float().detach().requires_grad_(mode != "clean")
        x = emb; pos = torch.arange(T, device=dev)[None].expand(B, T)
        cos, sin = self.m.model.rotary_emb(x, pos)
        mask = torch.full((T, T), float("-inf"), device=dev).triu(1)
        rec = {"inv_in": [], "inv_post": [], "probs": [], "sg": []} if mode == "clean" else None
        A, H2 = [], []
        for l, ly in enumerate(self.layers):
            at = ly.self_attn
            h, inv = self.norm(x, ly.input_layernorm.weight, None if mode == "clean" else fz["inv_in"][l])
            v = self.lin(h, at.v_proj.weight).view(B, T, self.Hkv, self.dh).transpose(1, 2)
            if mode == "clean":
                q = at.q_norm(self.lin(h, at.q_proj.weight).view(B, T, self.H, self.dh).to(at.q_proj.weight.dtype)).float().transpose(1, 2)
                k = at.k_norm(self.lin(h, at.k_proj.weight).view(B, T, self.Hkv, self.dh).to(at.k_proj.weight.dtype)).float().transpose(1, 2)
                q, k = self.rot(q, k, cos.float(), sin.float())
                k = self.rep(k, self.H // self.Hkv)
                pr = torch.softmax(q @ k.transpose(-1, -2) * at.scaling + mask, -1)
                rec["inv_in"].append(inv); rec["probs"].append(pr)
            else:
                pr = fz["probs"][l]
            o = (pr @ self.rep(v, self.H // self.Hkv)).transpose(1, 2).reshape(B, T, -1)
            x = x + self.lin(o, at.o_proj.weight)
            h2, inv2 = self.norm(x, ly.post_attention_layernorm.weight, None if mode == "clean" else fz["inv_post"][l])
            mlp = ly.mlp
            if mode == "clean":
                sg = Fn.silu(self.lin(h2, mlp.gate_proj.weight)); rec["inv_post"].append(inv2); rec["sg"].append(sg)
                a = sg * self.lin(h2, mlp.up_proj.weight)
            elif mode == "total":
                a = fz["sg"][l] * self.lin(h2, mlp.up_proj.weight); a.retain_grad()
            else:
                a = leaves[l]; H2.append(h2)
            A.append(a)
            x = x + self.lin(a, mlp.down_proj.weight)
        hf, invf = self.norm(x[:, -1], self.m.model.norm.weight, None if mode == "clean" else fz["inv_f"])
        logits = self.lin(hf, self.m.lm_head.weight)
        out = {"a": A, "h2": H2, "emb": emb, "logits_last": logits}
        if ans is not None:
            out["target"] = logits[:, ans] - logits.mean(-1)
        if mode == "clean":
            rec["inv_f"] = invf; out["fz"] = {k: (v if isinstance(v, torch.Tensor) else [t.detach() for t in v]) for k, v in rec.items()}
            out["fz"]["inv_f"] = invf.detach()
        return out


def build_graph(model, tok, item, gate_knockouts=True):
    lin = Lin(model); dev = next(model.parameters()).device
    p_ids = tok.encode(item["prompt"]); ans = item["answer_ids"][0]
    ids1 = torch.tensor([p_ids], device=dev); T = len(p_ids); W = model.config.intermediate_size; L = lin.L
    t0 = time.time()
    with torch.no_grad():
        cl = lin.forward(ids1, "clean", ans=ans)
        hf = model(input_ids=ids1).logits[0, -1].float()
    real_target = float(hf[ans] - hf.mean()); ours = float(cl["target"][0])
    g1 = {"hf_vs_ours_target": [real_target, ours], "hf_argmax_equal": int(hf.argmax()) == int(cl["logits_last"][0].argmax()),
          "max_abs_logit_diff": float((hf - cl["logits_last"][0]).abs().max())}
    fz = cl["fz"]; a_clean = [a.detach() for a in cl["a"]]
    # total effects: act x grad of the target (frozen model, gradients through everything)
    tot = lin.forward(ids1, "total", fz=fz, ans=ans); tot["target"].sum().backward()
    attr = torch.stack([(a * a.grad)[0] for a in tot["a"]])                              # [L, T, W]
    emb_attr_total = (tot["emb"] * tot["emb"].grad)[0].sum(-1)                           # [T]
    del tot
    attr = attr.clone(); attr[:, 0, :] = 0            # amendment 2 (18 Oct): position 0, the attention sink, is not a node
    flat = attr.abs().flatten(); order = flat.argsort(descending=True); csum = flat[order].cumsum(0) / flat.sum()
    k = int(min(CAP, int((csum < KEEP_FRAC).sum()) + 1))
    kept = [tuple(int(v) for v in (i // (T * W), (i // W) % T, i % W)) for i in order[:k].tolist()]
    kept_share_total = float(csum[k - 1])
    # direct contributions: every layer's activations are leaves (clean values); objectives batched over rows
    def direct(objs):
        """objs: list of callables(out, r) -> scalar for row r. Returns per objective: (neuron contributions [L,T,W], emb [T])."""
        res = []
        for c0 in range(0, len(objs), ROWS):
            ch = objs[c0:c0 + ROWS]; Bn = len(ch)
            leaves = [a.expand(Bn, -1, -1).clone().requires_grad_(True) for a in a_clean]
            fzb = {"inv_in": [t.expand(Bn, -1, -1) for t in fz["inv_in"]], "inv_post": [t.expand(Bn, -1, -1) for t in fz["inv_post"]],
                   "probs": [t.expand(Bn, -1, -1, -1) for t in fz["probs"]], "sg": fz["sg"], "inv_f": fz["inv_f"].expand(Bn, -1)}
            out = lin.forward(ids1.expand(Bn, -1), "direct", fz=fzb, leaves=leaves, ans=ans)
            sum(f(out, r) for r, f in enumerate(ch)).backward()
            for r in range(Bn):
                res.append((torch.stack([(lv.grad[r] * a_clean[l][0]) if lv.grad is not None else torch.zeros_like(a_clean[l][0])
                                         for l, lv in enumerate(leaves)]),           # no grad = no path from that layer to the objective
                            (out["emb"].grad[r] * out["emb"][r]).sum(-1).detach()))
            del out, leaves
        return res
    tgt_obj = lambda out, r: out["target"][r]
    def node_obj(l, p, n):
        wu = model.model.layers[l].mlp.up_proj.weight[n].float()
        return lambda out, r: fz["sg"][l][0, p, n] * (out["h2"][l][r, p] @ wu)
    objs = [tgt_obj] + [node_obj(*u) for u in kept]
    D = direct(objs)
    tgt_contrib, tgt_emb = D[0]
    g2_sum = float(tgt_contrib.sum() + tgt_emb.sum())
    g2 = {"sum_direct_all_neurons_plus_emb": g2_sum, "real_target": real_target,
          "rel_diff": abs(g2_sum - real_target) / max(abs(real_target), 1e-9)}
    # nodes
    toks = [tok.decode([t]) for t in p_ids]
    nodes = [{"id": "target", "type": "target", "token": tok.decode([ans]), "pos": T - 1, "value": real_target}]
    nodes += [{"id": f"emb|{p}", "type": "embedding", "layer": -1, "pos": p, "token": toks[p], "attr": float(emb_attr_total[p])} for p in range(T)]
    kid = {}
    for (l, p, n) in kept:
        nid = f"{l}|{p}|{n}"; kid[(l, p, n)] = nid
        nodes.append({"id": nid, "type": "neuron", "layer": l, "pos": p, "token": toks[p], "neuron": f"L{l}:{n}",
                      "act": float(a_clean[l][0, p, n]), "attr": float(attr[l, p, n]), "direct_to_target": float(tgt_contrib[l, p, n])})
    # edges (direct), pruned per destination to 98% of |inflow| from kept sources + embeddings
    edges = []
    dsts = [("target", D[0])] + [(kid[u], D[i + 1]) for i, u in enumerate(kept)]
    for dst, (cn, ce) in dsts:
        cand = [(kid[u], float(cn[u])) for u in kept if kid[u] != dst and float(cn[u]) != 0.0]
        cand += [(f"emb|{p}", float(ce[p])) for p in range(T) if float(ce[p]) != 0.0]
        cand.sort(key=lambda x: -abs(x[1])); tot_w = sum(abs(w) for _, w in cand) or 1.0; acc = 0.0
        for src, w in cand:
            if acc / tot_w >= EDGE_FRAC:
                break
            edges.append({"src": src, "dst": dst, "w": w}); acc += abs(w)
    kept_direct = sum(float(tgt_contrib[u]) for u in kept); emb_direct = float(tgt_emb.sum())
    completeness = {"kept_share_of_total_abs_attribution": kept_share_total, "n_kept": len(kept),
                    "target_direct_from_kept_neurons": kept_direct / real_target, "target_direct_from_embeddings": emb_direct / real_target,
                    "target_direct_from_unkept_neurons": (g2_sum - kept_direct - emb_direct) / real_target}
    # gate 3: real knockouts (HF model, bf16), zero the neuron at its position
    g3 = None
    if gate_knockouts:
        from neurons import mlps
        rng = random.Random(f"ag|{item['concept']}")
        top30 = kept[:30]
        active = [(l, p, n) for l in range(L) for (p, n) in [(rng.randrange(T), rng.randrange(W)) for _ in range(40)]
                  if abs(float(a_clean[l][0, p, n])) > 1e-3 and (l, p, n) not in set(kept)]
        rnd30 = rng.sample(active, min(30, len(active)))
        def ko(u):
            l, p, n = u
            def h(mod, a):
                y = a[0].clone(); y[0, p, n] = 0; return (y,)
            hk = mlps(model)[l].down_proj.register_forward_pre_hook(h)
            try:
                with torch.no_grad():
                    lg = model(input_ids=ids1).logits[0, -1].float()
            finally:
                hk.remove()
            return real_target - float(lg[ans] - lg.mean())
        top_eff = [ko(u) for u in top30]; rnd_eff = [ko(u) for u in rnd30]
        top_att = [float(attr[u]) for u in top30]; rnd_att = [float(attr[u]) for u in rnd30]
        g3 = {"spearman_top30": spearman(top_att, top_eff), "spearman_top30_plus_random30": spearman(top_att + rnd_att, top_eff + rnd_eff),
              "median_abs_effect_top30": stx.median(abs(x) for x in top_eff), "median_abs_effect_random30": stx.median(abs(x) for x in rnd_eff),
              "top30": [{"neuron": f"L{u[0]}:{u[2]}", "pos": u[1], "attr": a_, "knockout": e} for u, a_, e in zip(top30, top_att, top_eff)],
              "random30": [{"neuron": f"L{u[0]}:{u[2]}", "pos": u[1], "attr": a_, "knockout": e} for u, a_, e in zip(rnd30, rnd_att, rnd_eff)]}
    # amendment 3: joint knockout - all kept neurons off together vs 3 random same-size sets matched on (layer, position)
    from neurons import mlps as _mlps

    def p_off(units):
        by = {}
        for (l, p, n) in units:
            by.setdefault(l, []).append((p, n))
        hks = []
        for l, pn in by.items():
            pi = torch.tensor([p for p, _ in pn], device=dev); ni = torch.tensor([n for _, n in pn], device=dev)
            def h(mod, a, pi=pi, ni=ni):
                y = a[0].clone(); y[0, pi, ni] = 0; return (y,)
            hks.append(_mlps(model)[l].down_proj.register_forward_pre_hook(h))
        try:
            with torch.no_grad():
                lg = model(input_ids=ids1).logits[0, -1].float()
        finally:
            for hk in hks:
                hk.remove()
        return float(torch.softmax(lg, -1)[ans])
    rj = random.Random(f"joint|{item['concept']}"); ks = set(kept); rand_sets = []
    for _ in range(3):
        rs = []
        for (l, p, _) in kept:
            while True:
                n = rj.randrange(W)
                if (l, p, n) not in ks:
                    rs.append((l, p, n)); break
        rand_sets.append(rs)
    joint = {"p_base": float(torch.softmax(hf, -1)[ans]), "p": p_off(kept), "p_random": [p_off(rs) for rs in rand_sets]}
    coverage = {"attr_share": kept_share_total, "joint_knockout": joint}
    gates = {"g1_forward": g1, "g2_linear_sum": g2, "g3_knockouts": g3,
             "pass": bool(g1["hf_argmax_equal"] and g2["rel_diff"] < 0.01 and (g3 is None or g3["spearman_top30"] >= 0.7))}
    qm = re.search(r"<\|im_start\|>user\n(.*?)(?: Answer with just the name\.)?<\|im_end\|>", item["prompt"], re.S)
    return {"concept": item["concept"], "question": qm.group(1).strip() if qm else item["prompt"], "coverage": coverage, "prompt": item["prompt"], "answer": item.get("answer"), "answer_token": tok.decode([ans]), "answer_id": ans,
            "tokens": toks, "target": real_target, "nodes": nodes, "edges": edges, "completeness": completeness, "gates": gates,
            "method": {"keep_frac": KEEP_FRAC, "cap": CAP, "edge_frac": EDGE_FRAC, "model": model.config._name_or_path, "position0_excluded": True,
                       "amendments": "ATTRIB_GRAPHS.md 'Amendments' (18 Oct): circuit-tracer not used; position 0 not a node; cap 300 + joint knockout; all edges stored",
                       "frozen": ["attention patterns", "RMSNorm scales", "SiLU(gate)"]}, "seconds": round(time.time() - t0, 1)}


def spearman(x, y):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v)
        for j, i in enumerate(o):
            r[i] = j
        return r
    a, b = rk(x), rk(y); ma, mb = stx.mean(a), stx.mean(b)
    num = sum((p - ma) * (q - mb) for p, q in zip(a, b)); den = math.sqrt(sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b))
    return num / den if den else 0.0


def load(name):
    from config import model_path
    from neurons import load as nload
    if not os.path.isdir(model_path(name)):
        raise SystemExit("set MODEL_MAPPER_MODELS (no local model)")
    model, tok = nload(name)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def graphs():
    from config import DEFAULT_MODEL
    arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    name = arg("--model", DEFAULT_MODEL); out = Path(arg("--out", str(DATA))); limit = int(arg("--limit", "0"))
    items = json.loads((DATA / "pilot_prompts.json").read_text(encoding="utf-8"))
    if limit:
        items = items[:limit]
    model, tok = load(name)
    for it in items:
        d = out / slug(it["concept"]); d.mkdir(parents=True, exist_ok=True)
        if name != DEFAULT_MODEL:                     # a different model: score its own greedy-free answer token = same answer string
            it = {**it, "answer_ids": tok.encode(it["answer"])}
        g = build_graph(model, tok, it)
        (d / "graph.json").write_text(json.dumps(g, ensure_ascii=False), encoding="utf-8")
        G = g["gates"]
        print(f"{it['concept']:<26} nodes {g['completeness']['n_kept']:>3} edges {len(g['edges']):>5} | g1 argmax {G['g1_forward']['hf_argmax_equal']} "
              f"maxdiff {G['g1_forward']['max_abs_logit_diff']:.3f} | g2 rel {G['g2_linear_sum']['rel_diff']:.4f} | g3 rho {G['g3_knockouts']['spearman_top30']:.2f} "
              f"(+rnd {G['g3_knockouts']['spearman_top30_plus_random30']:.2f}) | kept share {g['completeness']['kept_share_of_total_abs_attribution']:.2f} | "
              f"joint P {g['coverage']['joint_knockout']['p_base']:.3f}->{g['coverage']['joint_knockout']['p']:.3f} (random {[round(x, 3) for x in g['coverage']['joint_knockout']['p_random']]}) | "
              f"{g['seconds']} s | pass {G['pass']}", flush=True)


def ablate():
    """Step 3: for each graph with groups.json, zero each group's neurons at their positions; vs 3 random same-size sets
    matched on (layer, position)."""
    from neurons import mlps
    from config import DEFAULT_MODEL
    model, tok = load(DEFAULT_MODEL); dev = next(model.parameters()).device; W = model.config.intermediate_size
    for gd in sorted(DATA.iterdir()):
        gp, grp = gd / "graph.json", gd / "groups.json"
        if not (gp.exists() and grp.exists()):
            continue
        g = json.loads(gp.read_text(encoding="utf-8")); groups = json.loads(grp.read_text(encoding="utf-8"))
        groups = groups.get("groups", groups) if isinstance(groups, dict) else groups
        ids = torch.tensor([tok.encode(g["prompt"])], device=dev); ans = g["answer_id"]
        node = {n["id"]: n for n in g["nodes"] if n["type"] == "neuron"}

        def run(units):
            by = {}
            for (l, p, n) in units:
                by.setdefault(l, []).append((p, n))
            hks = []
            for l, pn in by.items():
                def h(mod, a, pn=pn):
                    y = a[0].clone()
                    for p, n in pn:
                        y[0, p, n] = 0
                    return (y,)
                hks.append(mlps(model)[l].down_proj.register_forward_pre_hook(h))
            try:
                with torch.no_grad():
                    lg = model(input_ids=ids).logits[0, -1].float()
            finally:
                for hk in hks:
                    hk.remove()
            pr = torch.softmax(lg, -1)
            return {"p_answer": float(pr[ans]), "target": float(lg[ans] - lg.mean())}
        base = run([]); res = {"p_base": base["p_answer"], "target_base": base["target"], "groups": {}}
        for gi, grpd in enumerate(groups):
            members = [node[m] for m in grpd.get("members", grpd.get("nodes", [])) if m in node]
            units = [(m["layer"], m["pos"], int(m["neuron"].split(":")[1])) for m in members]
            eff = run(units); rng = random.Random(f"abl|{gd.name}|{gi}"); rnd = []
            for _ in range(3):
                ru = [(l, p, rng.randrange(W)) for (l, p, _) in units]
                rnd.append(run(ru))
            drop = base["p_answer"] - eff["p_answer"]; rdrops = [base["p_answer"] - r["p_answer"] for r in rnd]
            res["groups"][str(grpd.get("label", gi))] = {"p": eff["p_answer"], "p_random": [r["p_answer"] for r in rnd], "n": len(units), "row": grpd.get("row"),
                                                        "target": eff["target"], "target_random": [r["target"] for r in rnd],
                                                        "beats_all_random": all(abs(drop) > abs(x) for x in rdrops), "sum_attr": sum(m["attr"] for m in members)}
        res["joint"] = {k: g["coverage"]["joint_knockout"][k] for k in ("p", "p_random")}           # from step 1 (same definition)
        (gd / "ablations.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(gd.name, [(k, round(res["p_base"] - v["p"], 3), v["beats_all_random"]) for k, v in res["groups"].items()], flush=True)


if __name__ == "__main__":
    {"graphs": graphs, "ablate": ablate}[sys.argv[1]]()
