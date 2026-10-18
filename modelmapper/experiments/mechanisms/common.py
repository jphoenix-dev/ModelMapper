"""Shared helpers for the MECHANISMS.md tests: CPU-only Qwen3-0.6B, web-text windows, exact log-odds, output on D.

Every script in src/mechanisms imports this FIRST: it hides the GPUs (the 5090 runs another job), points the model
loader at the local copies, uses float32 on the CPU, and keeps temp files and outputs off drive C (nearly full).
"""
import json
import math
import os
import random
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MODEL_MAPPER_MODELS", "C:/dev/models/hf")
os.environ["MODEL_MAPPER_DEVICE"] = "cpu"
os.environ.setdefault("MODEL_MAPPER_DTYPE", "float32")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
# settings resolves MODEL_MAPPER_MODELS when first imported, so it comes after the defaults above (as config did before)
from modelmapper.core.settings import DATA_DIR, REPO_ROOT  # noqa: E402
for _k in ("TMP", "TEMP", "TMPDIR"):
    os.environ[_k] = (DATA_DIR.as_posix() + "/mechanisms/tmp")
sys.dont_write_bytecode = True

ROOT = REPO_ROOT
OUT = DATA_DIR / "mechanisms"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "tmp").mkdir(exist_ok=True)
CORPUS = ROOT / "data" / "fineweb_001_00000.parquet"
RUN1 = DATA_DIR / "loop_v2" / "run1" / "results" / "results" / "map"

import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("MECH_THREADS", "12")))
torch.set_grad_enabled(False)


def load06():
    from modelmapper.core.neurons import load
    model, tok = load("Qwen/Qwen3-0.6B")
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def web_windows(tok, n, T, seed=0, skip_docs=0):
    """n windows of exactly T tokens, one per document, from the first FineWeb row group (docs with >= T tokens)."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(str(CORPUS))
    texts = pf.read_row_group(0, columns=["text"]).column("text").to_pylist()
    rng = random.Random(seed); order = list(range(len(texts))); rng.shuffle(order)
    out = []
    for i in order[skip_docs:]:
        ids = tok.encode(texts[i])
        if len(ids) >= T:
            out.append(ids[:T])
            if len(out) == n:
                break
    return torch.tensor(out)


def exact_logodds(logits_rows, ans):
    """logits_rows: [len(ans), V] at the answer positions -> exact log-odds of the whole answer (ISSUES #35 form)."""
    lp = logits_rows.double().log_softmax(-1)
    s = lp[torch.arange(len(ans)), torch.tensor(ans)].sum().item()
    s = min(s, -1e-15)
    return s - math.log(-math.expm1(s))


def save(name, obj):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    print("saved", p)
    return p


def mlp_hook(model, layer, fn):
    """Register fn(args) -> new args as a forward pre-hook on layer's down_proj (the neuron activations)."""
    return model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(fn)
