"""Measurement guards: full provenance on every record, and an in / uncertain / out call for each neuron's top-30 place.

Provenance. Every record carries what it was measured on (GPU and its memory, CUDA, cuDNN, torch, transformers, numpy,
dtype, device, Python, OS, package version, model revision), because the same model on another card or software
version gives slightly different drops. A field that cannot be found is written "unknown", never left out; "none"
means it does not apply (no CUDA in a CPU build).

Calls. A concept's map keeps the 30 candidates with the largest single-knockout drop. Every knockout of one run is in
the same pass, so an offset that moves a whole concept between configurations cancels; the only question is whether a
neuron would still be in or out of the top 30 on another card or software version. For each of the 200 candidates:
    margin = its drop - the drop of the neuron ranked 30th in this run
    call   = "in" if margin > band, "out" if margin < -band, else "uncertain"   (the 30th itself is always uncertain)
The ranking and every drop stay exactly as measured; the calls are added next to them.

The band is per model, because a smaller model has smaller effects. For Qwen3-8B it is 0.13 log-odds: the 99th percentile (0.1285) of per-neuron |RTX 3090 - RTX 5090| mean-drop differences on
10 calibration concepts (395 neurons; torch 2.12.1+cu129 vs 2.11.0+cu128; research RESULTS, 19 Oct 2026). On 10 other
concepts the calls agreed across the two cards 98.3% of the time (predicted >= 99%); all 6 disagreements came from one
concept whose whole set moved by ~0.15. Comparing against the 30th neuron of the same pass is designed to remove such
shifts, but that is still to be tested on new data. Comparing a drop with a stored number from another run is not
covered by the band (METHOD.md).

Other models have no calibrated band yet (Qwen3-0.6B included: no paired-hardware runs), so their calls are
"uncalibrated", never a silent "in" or "out"; the margins are still recorded. A band can be given (MODEL_MAPPER_BAND,
or --band, or a calibration added to BANDS); the value used and where it came from are always recorded.
"""
from __future__ import annotations

import platform

from modelmapper.settings import env

# Calibrated bands, by model (short name). Add one only from paired runs of the same concepts on two configurations.
BANDS = {"Qwen3-8B": {"value": 0.13, "source": "Qwen3-8B, 3090 vs 5090, 10 concepts, p99, 19 Oct 2026"}}   # CAPS.md
TOPN = 30
UNKNOWN = "unknown"


def band(model: str | None, override: float | None = None) -> dict:
    """The band for `model`, with where it came from: an explicit override, else MODEL_MAPPER_BAND, else the model's
    calibrated band in BANDS, else {"value": None} (calls are then "uncalibrated")."""
    if override is not None:
        return {"value": float(override), "source": "override (argument / --band)"}
    if env("MODEL_MAPPER_BAND"):
        return {"value": float(env("MODEL_MAPPER_BAND")), "source": "override (MODEL_MAPPER_BAND)"}
    name = (model or "").split("/")[-1]
    if name in BANDS:
        return dict(BANDS[name])
    return {"value": None, "source": f"no calibrated band for {name or 'this model'}: calls are uncalibrated"}


def gpu_name() -> str:
    """The GPU's name, or 'cpu'; never fails (torch can report CUDA available with no device, e.g. CUDA_VISIBLE_DEVICES='')."""
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            from modelmapper.model import DEVICE
            idx = int(DEVICE.split(":")[1]) if DEVICE.startswith("cuda:") else 0
            return torch.cuda.get_device_name(idx)
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "Apple MPS"
    except Exception:  # noqa: BLE001
        return UNKNOWN
    return "cpu"


def os_name() -> str:
    """'Windows 11 (build 26100)', 'Linux 6.8.0-...', 'Darwin 24.1.0'. platform.release() says '10' on Windows 11, so
    Windows is named from its build number (22000 and later is Windows 11)."""
    try:
        if platform.system() == "Windows":
            import sys
            build = sys.getwindowsversion().build
            return f"Windows {11 if build >= 22000 else platform.release()} (build {build})"
        return f"{platform.system()} {platform.release()}"
    except Exception:  # noqa: BLE001
        return UNKNOWN


def _try(fn, absent="none"):
    try:
        v = fn()
    except Exception:  # noqa: BLE001
        return UNKNOWN
    return absent if v is None else v


def provenance_extra(pipe=None) -> dict:
    """Fields added to run 1's provenance block (strings or numbers; nothing secret; "unknown" when not found)."""
    import numpy
    import torch
    import modelmapper
    from modelmapper.model import DEVICE
    cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    idx = int(DEVICE.split(":")[1]) if DEVICE.startswith("cuda:") else 0
    cfg = getattr(getattr(pipe, "model", None), "config", None)
    return {
        "package": f"modelmapper {modelmapper.__version__}", "device": str(DEVICE),
        "cuda": _try(lambda: torch.version.cuda),
        "cudnn": _try(lambda: torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None),
        "numpy": numpy.__version__, "os": os_name(), "machine": platform.machine(),
        "gpu_memory_gb": _try(lambda: round(torch.cuda.get_device_properties(idx).total_memory / 2**30, 1) if cuda else None),
        "model_revision": _try(lambda: getattr(cfg, "_commit_hash", None), UNKNOWN),
        "prompt_format": _try(lambda: _prompt_format(pipe), UNKNOWN),   # "chat" (every published map) or "plain"
    }


def _prompt_format(pipe):
    from modelmapper.measure import prompt_format
    return prompt_format(pipe.tok)


def call(margin: float, band_value: float | None) -> str:
    """'in' (clearly above the 30th), 'out' (clearly below), 'uncertain' (within the band of it), or 'uncalibrated'
    when the model has no band."""
    if band_value is None:
        return "uncalibrated"
    if margin > band_value:
        return "in"
    if margin < -band_value:
        return "out"
    return "uncertain"


def annotate(record: dict, band_override: float | None = None, topn: int = TOPN) -> dict:
    """Give each candidate its margin and call, and add record["guards"] with the band and the counts.
    The candidates' order and drops are not changed."""
    rank = record.get("rank") or {}
    cands = rank.get("candidates_all200") or []
    b = band(((record.get("v2") or {}).get("provenance") or {}).get("model"), band_override)
    g = {"band": b, "bar_rank": topn, "bar": None, "n_in": None, "n_uncertain": None, "n_out_but_close": None}
    if len(cands) >= topn:
        bar = cands[topn - 1]["single_drop"]
        for c in cands:
            c["margin"] = round(c["single_drop"] - bar, 3)
            c["call"] = call(c["single_drop"] - bar, b["value"])
        for t, c in zip(rank.get("top") or [], cands):       # "top" is the first 30 of the candidates
            if t is not c:
                t["margin"], t["call"] = c["margin"], c["call"]
        g["bar"] = bar
        if b["value"] is not None:
            g.update(n_in=sum(c["call"] == "in" for c in cands[:topn]),
                     n_uncertain=sum(c["call"] == "uncertain" for c in cands[:topn]),
                     n_out_but_close=sum(c["call"] == "uncertain" for c in cands[topn:]))
    record["guards"] = g
    return record
