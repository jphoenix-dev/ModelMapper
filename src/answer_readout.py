"""Who writes the answer word? Exact decomposition of the answer token's score (attention study, last step; no API).

At the answer position the model's score for token a is  logit_a = W_U[a] . (g_f * h_f) / rms(h_f),  where h_f is the
final residual stream. h_f is exactly the sum of the embedding, every head's output and every neuron's output, so
each source c contributes (W_U[a] - mean_v W_U[v]) . (g_f * c) / rms(h_f) to the answer's score RELATIVE TO THE MEAN
WORD (so a push that raises every word equally counts as zero). The final rms is held fixed (as elsewhere).
Sanity check: the pieces rebuild h_f (< 2% relative error) and the answer score (vs the model's logits).

Reports per concept: top direct contributors; shares from heads vs neurons by layer band; how much comes DIRECTLY from
the concept's own top-10 neurons; and "says the answer" neurons (top direct neuron writers).
Usage: RECALL_VARIANT=V3 ANSWER_DISCOVERY=0 CONCEPT_OUT_DIR=<dir> python src/answer_readout.py <tag> "<concept>" [...]
"""
import json
import sys
from collections import defaultdict

import torch

from config import DEFAULT_MODEL
import concept_pipeline as cp
from neurons import DEVICE, mlps
from pipeline_fast import FastPipeline


def band(l):
    return "0-15" if l < 16 else "16-23" if l < 24 else "24-29" if l < 30 else "30-35"


def main():
    tag = sys.argv[1]
    pipe = FastPipeline(DEFAULT_MODEL); m = pipe.model; cfg = m.config
    H, D, L, eps = cfg.num_attention_heads, cfg.head_dim, cfg.num_hidden_layers, cfg.rms_norm_eps
    WU = m.lm_head.weight.float(); WUm = WU.mean(0); gf = m.model.norm.weight.float()
    out_p = cp.OUT / f"answer_readout_{tag}.json"
    out = json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists() else {}
    for concept in [c.strip() for c in sys.argv[2:]]:
        if concept in out:
            continue
        x = json.loads((cp.OUT / f"{cp.slug(concept)}__clamp_reply.json").read_text(encoding="utf-8"))
        own = {(q["layer"], q["neuron"]) for q in x["top_neurons"][:10]}
        spec, tid, recall, _ = pipe.make_prompts(concept)
        a = tid[0]
        w = (WU[a] - WUm) * gf                                   # direction in the final residual that raises a vs the mean word
        VH = [(m.model.layers[l].self_attn.o_proj.weight.float().T @ w).view(H, D) for l in range(L)]
        VN = [mlps(m)[l].down_proj.weight.float().T @ w for l in range(L)]
        heads = defaultdict(float); neur = defaultdict(float); emb = 0.0; worst = 0.0; worst_logit = 0.0
        heads_c = defaultdict(float); neur_c_tot = [0.0]; comp_score = 0.0
        allpnc = torch.zeros(L, cfg.intermediate_size, device=WU.device)
        allpn = torch.zeros(L, cfg.intermediate_size, device=WU.device); allph = torch.zeros(L, H, device=WU.device)  # every source, for spread
        for q, _ in recall:
            st = {"z": {}, "a": {}}
            hk = [m.model.embed_tokens.register_forward_hook(lambda mod, i, o: st.__setitem__("e", o[0, -1].float())),
                  m.model.norm.register_forward_pre_hook(lambda mod, i: st.__setitem__("hf", i[0][0, -1].float()))]
            for l in range(L):
                hk.append(m.model.layers[l].self_attn.o_proj.register_forward_pre_hook(lambda mod, i, l=l: st["z"].__setitem__(l, i[0][0, -1].float())))
                hk.append(mlps(m)[l].down_proj.register_forward_pre_hook(lambda mod, i, l=l: st["a"].__setitem__(l, i[0][0, -1].float())))
            ids = pipe.tok(q, return_tensors="pt").to(DEVICE).input_ids
            try:
                with torch.no_grad():
                    logits = m(input_ids=ids).logits[0, -1].float()
            finally:
                for h_ in hk:
                    h_.remove()
            hf = st["hf"]; rms = torch.sqrt((hf * hf).mean() + eps)
            recon = st["e"].clone()
            for l in range(L):
                recon += m.model.layers[l].self_attn.o_proj.weight.float() @ st["z"][l] + mlps(m)[l].down_proj.weight.float() @ st["a"][l]
            err = ((recon - hf).norm() / hf.norm()).item(); worst = max(worst, err)
            assert err < 0.02, f"{concept}: final residual reconstruction error {err:.3f}"
            target = (logits[a] - logits.mean()).item(); mine = (w @ hf / rms).item()
            worst_logit = max(worst_logit, abs(mine - target) / (abs(target) + 1e-3))
            assert abs(mine - target) <= 0.05 * abs(target) + 0.2, f"{concept}: answer score {mine:.2f} vs model {target:.2f}"
            emb += (w @ st["e"] / rms).item() / len(recall)
            # rivals: the next-word distribution WITHOUT the answer, renormalised. log-odds(answer) = logit_a - log sum_{v!=a} e^logit_v,
            # whose gradient w.r.t. the logits is e_a - q: the direction that raises the answer against its likely rivals.
            # (v1 used p including the answer; that gradient vanishes when p(answer) ~ 1, which these facts are.)
            q = torch.softmax(logits, -1); q[a] = 0.0; q = q / q.sum()
            wc = (WU[a] - q @ WU) * gf
            tgt_c = (logits[a] - (q * logits).sum()).item(); mine_c = (wc @ hf / rms).item()
            assert abs(mine_c - tgt_c) <= 0.05 * abs(tgt_c) + 0.2, f"{concept}: competitor score {mine_c:.2f} vs model {tgt_c:.2f}"
            comp_score += tgt_c / len(recall)
            for l in range(L):
                phc = ((m.model.layers[l].self_attn.o_proj.weight.float().T @ wc).view(H, D) * st["z"][l].view(H, D)).sum(-1) / rms
                for hh, v in enumerate(phc.tolist()):
                    heads_c[(l, hh)] += v / len(recall)
                pnc = st["a"][l] * (mlps(m)[l].down_proj.weight.float().T @ wc) / rms
                neur_c_tot[0] += pnc.sum().item() / len(recall)
                allpnc[l] += pnc / len(recall)
            for l in range(L):
                ph = (VH[l] * st["z"][l].view(H, D)).sum(-1) / rms
                for hh, v in enumerate(ph.tolist()):
                    heads[(l, hh)] += v / len(recall)
                pn = st["a"][l] * VN[l] / rms
                allpn[l] += pn / len(recall); allph[l] += ph / len(recall)
                top = pn.abs().topk(15).indices.tolist()
                for n_ in top:
                    neur[(l, n_)] += pn[n_].item() / len(recall)
                neur[(l, "rest")] += (pn.sum().item() - pn[top].sum().item()) / len(recall)
        tot_h = sum(heads.values()); tot_n = sum(neur.values())
        assert abs(allpn.sum().item() - tot_n) <= 0.01 * abs(tot_n) + 0.05 and abs(allph.sum().item() - tot_h) <= 0.01 * abs(tot_h) + 0.05,             f"{concept}: full source vectors disagree with the itemised totals"

        def k_for(vec, frac):
            # fewest top positive writers whose sum reaches frac of the NET total of this source type (None if never)
            v = vec.flatten().sort(descending=True).values; tot = v.sum().item(); cs = v.cumsum(0)
            if tot <= 0:
                return None
            hit = (cs >= frac * tot).nonzero()
            return int(hit[0].item()) + 1 if len(hit) else None
        bands = defaultdict(float)
        for (l, hh), v in heads.items():
            bands["heads " + band(l)] += v
        for (l, n_), v in neur.items():
            bands["neurons " + band(l)] += v
        own_direct = sum(v for (l, n_), v in neur.items() if n_ != "rest" and (l, n_) in own)
        th = sorted(heads.items(), key=lambda kv: -abs(kv[1]))[:10]
        tn = sorted(((k, v) for k, v in neur.items() if k[1] != "rest"), key=lambda kv: -abs(kv[1]))[:10]
        out[concept] = {"answer_token": pipe.tok.decode([a]), "score_vs_mean": round(tot_h + tot_n + emb, 2), "from_heads": round(tot_h, 2),
                        "from_neurons": round(tot_n, 2), "from_embedding": round(emb, 2), "own_concept_neurons_direct": round(own_direct, 2),
                        "by_band": {k: round(v, 2) for k, v in sorted(bands.items())},
                        "top_heads": [(f"L{l}H{hh}", round(v, 2)) for (l, hh), v in th],
                        "top_neurons": [(f"L{l}:{n_}", round(v, 2), (l, n_) in own) for (l, n_), v in tn],
                        "neurons_for_half": k_for(allpn, 0.5), "neurons_for_90pct": k_for(allpn, 0.9),
                        "heads_for_half": k_for(allph, 0.5), "heads_for_90pct": k_for(allph, 0.9),
                        "vs_rivals": {"score": round(comp_score, 2), "from_heads": round(sum(heads_c.values()), 2),
                                           "from_neurons": round(neur_c_tot[0], 2),
                                           "own_concept_neurons_direct": round(sum(allpnc[l, n_].item() for l, n_ in own), 2),
                                           "top_heads": [(f"L{l}H{hh}", round(v, 2)) for (l, hh), v in sorted(heads_c.items(), key=lambda kv: -abs(kv[1]))[:10]],
                                           "named_heads": {f"L{l}H{hh}": round(heads_c[(l, hh)], 2) for l, hh in
                                                           ((35, 25), (34, 28), (35, 22), (35, 27), (34, 29), (35, 24))},
                                           "top_neurons": [(f"L{l}:{n_}", round(allpnc[l, n_].item(), 2)) for l, n_ in
                                                           [divmod(i, cfg.intermediate_size) for i in allpnc.flatten().abs().topk(10).indices.tolist()]]},
                        "max_recon_err": round(worst, 4), "max_score_err": round(worst_logit, 4)}
        o = out[concept]
        print(f"{concept:<26} '{o['answer_token']}' score {o['score_vs_mean']:+.1f} = heads {o['from_heads']:+.1f} + neurons {o['from_neurons']:+.1f} "
              f"(own concept neurons directly {o['own_concept_neurons_direct']:+.1f}) | top heads {[h for h, _ in o['top_heads'][:4]]} | top neurons "
              f"{[(n_, v) for n_, v, _ in o['top_neurons'][:3]]} | vs rivals {o['vs_rivals']['score']:+.1f}: anti-writers L35H27 "
              f"{o['vs_rivals']['named_heads']['L35H27']:+.2f} L34H29 {o['vs_rivals']['named_heads']['L34H29']:+.2f} | half the neuron write needs {o['neurons_for_half']} neurons | err {o['max_recon_err']:.3f}", flush=True)
        out_p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
