"""J-lens pilot (src/loop_v2/jlens.py) -> ledger pack, experiment `jlens_qwen3_8b`.

Every MLP neuron (442,368): its output vector read through the J-lens of its layer and through the plain logit lens -
evidence_type readout_tokens, metrics j_lens_top30 / logit_lens_top30 (value = z of the top token's logit over the
vocabulary; value_text = top 10; details = top / bottom 30 tokens with logits). Informal readings, never labels.
Validation (J1) and prediction results (J2-J4) go in the experiment params; the J matrices are artifacts.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ledger.core import DATA_ROOT, PackWriter  # noqa: E402

D = DATA_ROOT / "jlens"


def main(force=False):
    npz = [D / f"jlens_neurons_L{l}.npz" for l in range(36)]
    txts = [D / "jlens_validate.txt", D / "jlens_tests.txt"]
    if not all(p.exists() for p in npz + txts):
        print("jlens: results not complete yet, skipped"); return
    from transformers import AutoTokenizer
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from config import DEFAULT_MODEL, model_path
    tok = AutoTokenizer.from_pretrained(model_path(DEFAULT_MODEL))
    with PackWriter("jlens", inputs=npz + txts + [D / "jlens_log.jsonl", D / "jlens_validate.json"], code=[__file__], force=force) as pw:
        if pw.skip:
            print("jlens: unchanged, skipped"); return
        log = [json.loads(x) for x in (D / "jlens_log.jsonl").read_text().splitlines() if x.strip() and '"bench"' not in x]
        pw.experiment("jlens_qwen3_8b", title="J-lens for Qwen3-8B, and every neuron read through it (pilot)",
                      method=("After Gurnee, Sofroniew, Pearce, Lindsey et al. 2026 (transformer-circuits.pub/2026/workspace). J_l = Jacobian of "
                              "the final pre-norm residual stream w.r.t. the residual stream after layer l, averaged over source positions "
                              "t >= 1, later positions t' >= t and FineWeb windows (one backward pass per output dimension gives all layers). "
                              "The lens reads a vector v at layer l as softmax(W_U (g * J_l v)). Each neuron's output vector (down_proj "
                              "column) is read through the J-lens of its layer and through the plain logit lens."),
                      status="pre-registered predictions J1-J4 (see jlens_tests.txt / validate)", status_class="pre-registered", date="2026-10-18",
                      script="src/loop_v2/jlens.py", reproduce="MODEL_MAPPER_MODELS=C:/dev/models/hf JL_ROWS=8 python src/loop_v2/jlens.py estimate --windows 32; ... validate; neurons; tests",
                      data_files=["data:jlens/j_sums.pt", "data:jlens/jlens_neurons_L<l>.npz", "data:jlens/jlens_validate.txt", "data:jlens/jlens_tests.txt"],
                      caveats=["Pilot: J estimated from 32 windows x 128 tokens (Anthropic used ~1,000 prompts); split-half reliability in params.",
                               "Single-token concepts only (a lens limitation).", "Readout tokens are informal readings, never labels."],
                      docs=["RESULTS.md (J-lens pilot)"],
                      params={"windows": len(log), "tokens_per_window": 128, "validate": (D / "jlens_validate.txt").read_text(encoding="utf-8"),
                              "tests": (D / "jlens_tests.txt").read_text(encoding="utf-8"), "timing": log})
        for a in ("j_sums.pt", "jlens_validate.json"):
            if (D / a).exists():
                pw.artifact(D / a, experiment="jlens_qwen3_8b", description="J-lens sums per split half (A/B) and pair counts" if a.endswith(".pt") else "per-layer future-token ranks")
        V = tok.convert_ids_to_tokens(list(range(len(tok))))
        dec = [tok.convert_tokens_to_string([t]) if t is not None else "" for t in V]
        for l in range(36):
            Z = np.load(D / f"jlens_neurons_L{l}.npz")
            for pre, metric in (("j", "j_lens_top30"), ("l", "logit_lens_top30")):
                top, tl, bot, bl, z = Z[f"{pre}_top"], Z[f"{pre}_top_lg"], Z[f"{pre}_bot"], Z[f"{pre}_bot_lg"], Z[f"{pre}_z"]
                for n in range(top.shape[0]):
                    tt = [dec[i] for i in top[n]]
                    pw.evidence(experiment="jlens_qwen3_8b", unit=(l, n), evidence_type="readout_tokens", metric=metric, value=float(z[n]),
                                value_text=" | ".join(t.strip() or repr(t) for t in tt[:10]),
                                details={"top": tt, "top_logit": [round(float(x), 3) for x in tl[n]], "bottom": [dec[i] for i in bot[n]],
                                         "bottom_logit": [round(float(x), 3) for x in bl[n]]},
                                source=f"jlens/jlens_neurons_L{l}.npz#{pre}:{n}")
            print(f"jlens: layer {l} rows written", flush=True)


if __name__ == "__main__":
    main(force="--force" in sys.argv)
