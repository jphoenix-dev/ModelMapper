"""Noise floor for single-neuron knockouts in bf16: is a knockout's KL above what batch shape alone produces?
Usage: python src/dist/noise_floor.py <store> <ablate task id>   (e.g. ablate:18:0-63:ps0; GPU)
null_shape: KL between the SAME unablated context run alone vs inside a batch of 32 (pure batch-shape noise).
ko_alone / ko_b32 / ko_b16: knockout KL measured with base+ablated alone (batch 2) vs inside batches of 32 / 16.
"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from dist.manifest import task_key
from dist.tasks import Ctx, _run_rows
from neurons import DEVICE
from dist.storage import open_store
st = open_store(sys.argv[1]); ctx = Ctx(st); m = ctx.model
TASK = sys.argv[2]; LAYER = int(TASK.split(":")[1]); A0, B0 = (int(x) for x in TASK.split(":")[2].split("-"))
z = st.get_npz(task_key(TASK) + "/ablate.npz")
toks, starts, lens = ctx.corpus(); win = st.get_json(f"promptsets/{TASK.split(':')[-1]}.json")["window"]
nb = torch.arange(A0, B0 + 1, device=DEVICE)
sel = np.where(z["kind"] == 0)[0][::5][:128]                 # top-firing rows
rows = [(toks[starts[int(z["doc"][r])] + max(0, int(z["pos"][r]) - win + 1): starts[int(z["doc"][r])] + int(z["pos"][r]) + 1], int(z["neuron"][r])) for r in sel]
def kl(a, b): return (a.exp() * (a - b)).sum(-1)
# base alone
base1 = torch.stack([_run_rows(m, LAYER, [(x, -1)], nb)[0][0].log_softmax(-1) for x, _ in rows])
# base inside batches of 32 (other contexts as neighbours)
baseB = torch.cat([_run_rows(m, LAYER, [(x, -1) for x, _ in rows[i:i + 32]], nb)[0].log_softmax(-1) for i in range(0, len(rows), 32)])
null = kl(base1, baseB).cpu().numpy()
def ko(group):   # knockout KL with `group` (base, ablated) pairs per pass
    out = []
    for i in range(0, len(rows), group):
        chunk = []
        for x, n in rows[i:i + group]: chunk += [(x, -1), (x, n)]
        lp = _run_rows(m, LAYER, chunk, nb)[0].log_softmax(-1)
        out += kl(lp[0::2], lp[1::2]).cpu().tolist()
    return np.array(out)
a1, a16, a8 = ko(1), ko(16), ko(8)
rk = lambda x: np.argsort(np.argsort(x))
sp = lambda x, y: np.corrcoef(rk(x), rk(y))[0, 1]
print(f"n={len(rows)} contexts (top firings, layer {LAYER})")
print(f"null KL (same context, batch 1 vs batch 32): median {np.median(null):.2e}, p90 {np.quantile(null, .9):.2e}")
print(f"knockout KL, pair alone:  median {np.median(a1):.2e}, p90 {np.quantile(a1, .9):.2e}")
print(f"knockout KL, 16 pairs/pass: median {np.median(a16):.2e};  8 pairs/pass: median {np.median(a8):.2e}")
print(f"spearman alone vs 16/pass {sp(a1, a16):.3f}; alone vs 8/pass {sp(a1, a8):.3f}; 16 vs 8 {sp(a16, a8):.3f}")
print(f"share of knockouts above the null p90: {np.mean(a1 > np.quantile(null, .9)):.2f}; above 10x null median: {np.mean(a1 > 10*np.median(null)):.2f}")
big = a1 > 10 * np.median(null)
if big.sum() > 5:
    print(f"on the {big.sum()} clearly-above-noise knockouts: spearman alone vs 16/pass {sp(a1[big], a16[big]):.3f}, median rel diff {np.median(np.abs(a1[big]-a16[big])/a1[big]):.3f}")
