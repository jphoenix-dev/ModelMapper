"""Core tooling: load a Qwen3 model and read / steer its MLP neurons.

In Qwen3 each layer's MLP computes
    neuron = SiLU(gate_proj(x)) * up_proj(x)        # one number per neuron, per token
    out    = down_proj(neuron)                        # added to the residual stream
so "the neuron" is the input to down_proj. We also keep the raw gate and value
(up_proj output) so every firing can be tagged with its sign regime.
"""
from __future__ import annotations

import contextlib
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
sys.stdout.reconfigure(encoding="utf-8")  # Windows console defaults to cp1252 and chokes on token text


LOCAL_MODELS = "C:/dev/models/hf"  # filled by scripts/download_models.sh


def load(name: str = "Qwen/Qwen3-0.6B", dtype=torch.bfloat16):
    local = os.path.join(LOCAL_MODELS, name.split("/")[-1])
    if os.path.exists(os.path.join(local, "config.json")):
        name = local
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, device_map="cuda")
    model.eval()
    return model, tok


def mlps(model):
    return [layer.mlp for layer in model.model.layers]


@contextlib.contextmanager
def capture(model, layers=None, gate_value=False):
    """Record neuron activations (and optionally gate/value) for the listed layers.

    Yields a dict: store["act"][layer] -> tensor [batch, seq, n_neurons] (float32, on GPU).
    """
    layers = range(len(mlps(model))) if layers is None else layers
    store = {"act": {}, "gate": {}, "value": {}}
    handles = []
    for i in layers:
        mlp = mlps(model)[i]
        handles.append(mlp.down_proj.register_forward_pre_hook(
            lambda m, args, i=i: store["act"].__setitem__(i, args[0].float())))
        if gate_value:
            handles.append(mlp.gate_proj.register_forward_hook(
                lambda m, a, out, i=i: store["gate"].__setitem__(i, out.float())))
            handles.append(mlp.up_proj.register_forward_hook(
                lambda m, a, out, i=i: store["value"].__setitem__(i, out.float())))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


@contextlib.contextmanager
def steer(model, edits, mode="clamp", reply_only=False):
    """Overwrite (or nudge) neurons during the forward pass.

    edits: list of (layer, neuron, value). value=0 ablates; a large value steers.
    mode: "clamp" sets the neuron to value; "add" adds value on top of its natural activation.
    reply_only: during generate(), skip the first pass that reads the whole prompt and only act on the
    one-token passes that write the reply, so the user's question is read normally.
    """
    by_layer = {}
    for layer, neuron, value in edits:
        by_layer.setdefault(layer, []).append((neuron, value))
    handles = []
    for layer, items in by_layer.items():
        idx = torch.tensor([n for n, _ in items], device="cuda")
        val = torch.tensor([v for _, v in items], device="cuda")

        def hook(m, args, idx=idx, val=val):
            if reply_only and args[0].shape[-2] > 1:
                return None
            x = args[0].clone()
            if mode == "add":
                x[..., idx] += val.to(x.dtype)
            else:
                x[..., idx] = val.to(x.dtype)
            return (x,)
        handles.append(mlps(model)[layer].down_proj.register_forward_pre_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def regime(gate: torch.Tensor, value: torch.Tensor) -> list[str]:
    """Sign regime of each firing: '++', '+-', '-+', '--' (gate sign, value sign)."""
    g = ["+" if x >= 0 else "-" for x in gate.flatten().tolist()]
    v = ["+" if x >= 0 else "-" for x in value.flatten().tolist()]
    return [a + b for a, b in zip(g, v)]
