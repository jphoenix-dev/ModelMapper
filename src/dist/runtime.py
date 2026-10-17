"""Model loading at a pinned revision, and the metadata recorded with every result.

The run's config.json names the model repo and the exact Hugging Face commit (revision). Workers download that
commit at boot. On the local 5090 the existing copy in $MODEL_MAPPER_MODELS is used instead, but only after its
safetensors files are checked against the pinned commit's LFS sha256 (cached, so the 16 GB hash runs once).
Weights are never quantised: dtype is bfloat16 or wider (config "dtype").
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MODELS_DIR, env, short_name  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ALLOWED_DTYPES = ("bfloat16", "float32")
WEIGHT_PATTERNS = ["*.json", "*.safetensors", "merges.txt", "vocab.json", "tokenizer*"]
_FP_CACHE = Path.home() / ".cache" / "model_mapper_weight_sha256.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def hub_sha256(repo: str, revision: str) -> dict[str, str]:
    """LFS sha256 of each safetensors file at the pinned commit."""
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    return {s.rfilename: s.lfs.sha256 for s in info.siblings if s.rfilename.endswith(".safetensors") and s.lfs}


def local_matches(local: Path, repo: str, revision: str) -> bool:
    """True if every safetensors file in `local` has the pinned commit's sha256. Hashes are cached by size+mtime."""
    cache = json.loads(_FP_CACHE.read_text()) if _FP_CACHE.exists() else {}
    want = hub_sha256(repo, revision)
    for name, sha in want.items():
        p = local / name
        if not p.exists():
            return False
        st = p.stat(); ck = f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns}"
        if ck not in cache:
            print(f"[runtime] hashing {p} ...", flush=True)
            cache[ck] = _sha256(p)
            _FP_CACHE.parent.mkdir(parents=True, exist_ok=True); _FP_CACHE.write_text(json.dumps(cache, indent=1))
        if cache[ck] != sha:
            return False
    return True


def model_dir(repo: str, revision: str) -> tuple[str, str, dict]:
    """(folder to load from, where it came from, download timing). Local copy if it matches the pin, else download."""
    local = Path(MODELS_DIR) / short_name(repo)
    if (local / "config.json").exists() and env("DIST_TRUST_LOCAL") != "0":
        t = time.time()
        if local_matches(local, repo, revision):
            return str(local), "local copy, sha256-verified against the pinned revision", {"verify_s": round(time.time() - t, 1)}
        print(f"[runtime] local copy {local} does not match {repo}@{revision}; downloading", flush=True)
    from huggingface_hub import snapshot_download
    t = time.time()
    path = snapshot_download(repo, revision=revision, allow_patterns=WEIGHT_PATTERNS)
    sec = time.time() - t
    gb = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e9
    return path, "downloaded from the hub at the pinned revision", {
        "download_s": round(sec, 1), "download_gb": round(gb, 2), "download_MBps": round(gb * 1e3 / sec, 1) if sec else None}


def load_model(cfg: dict):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from neurons import DEVICE
    dtype = env("DIST_DTYPE", cfg.get("dtype", "bfloat16"))
    if dtype not in ALLOWED_DTYPES:
        raise SystemExit(f"dtype {dtype} not allowed: exact weights only ({ALLOWED_DTYPES})")
    path, source, fetch = model_dir(cfg["model"], cfg["revision"])
    t = time.time()
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, dtype), device_map=DEVICE)
    model.eval()
    return model, tok, {"model": cfg["model"], "revision": cfg["revision"], "weights_source": source, "dtype": dtype,
                        **fetch, "to_gpu_s": round(time.time() - t, 1)}


def _cmd(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return None


def git_commit() -> str | None:
    """$GIT_COMMIT (baked into the image at build time), else the checkout's HEAD."""
    return env("GIT_COMMIT") or _cmd(["git", "-C", str(ROOT), "rev-parse", "HEAD"])


def environment() -> dict:
    """Hardware and software facts recorded with every result."""
    import transformers
    from neurons import DEVICE
    gpu = driver = None
    if DEVICE.startswith("cuda"):
        gpu = torch.cuda.get_device_name(torch.device(DEVICE))
        driver = _cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
        driver = driver.splitlines()[0] if driver else None
    return {"gpu": gpu, "driver": driver, "cuda": torch.version.cuda, "torch": torch.__version__,
            "transformers": transformers.__version__, "python": platform.python_version(), "device": DEVICE,
            "git_commit": git_commit(), "host": socket.gethostname(), "pod_id": os.environ.get("RUNPOD_POD_ID"),
            "gpu_type_env": os.environ.get("RUNPOD_GPU_TYPE")}
