"""Read a model's MLP weights straight from its safetensors files, on the CPU, without loading the model.

The weights-only stages (pseudo-neurons, job type, write routes) need a few matrices per layer, not a forward pass:

    W = Weights("Qwen/Qwen3-8B")
    W.n_layers, W.hidden, W.inter                 # 36, 4096, 12288
    G = W.get("model.layers.5.mlp.gate_proj.weight")   # float32 tensor [inter, hidden]
    W.mlp(5)                                      # {"gate": G, "up": U, "down": D, "norm": post-attention norm gain}

The local copy comes from MODEL_MAPPER_MODELS (settings.model_path); otherwise only the json and safetensors files are
fetched from the Hugging Face hub (no tokenizer, no code).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

from modelmapper.settings import DEFAULT_MODEL, model_path

CHECKED_MODELS = {"qwen3"}              # the weights-only readings were checked on these model types
POST_NORM_MODELS = {"olmo2", "olmo3"}   # their post_attention_layernorm sits after attention, not before the MLP


def model_dir(name: str) -> Path:
    path = model_path(name)
    if os.path.isdir(path):
        return Path(path)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(path, allow_patterns=["*.json", "*.safetensors"]))


class Weights:
    def __init__(self, name: str | None = None, folder: str | Path | None = None):
        self.name = name or DEFAULT_MODEL
        self.dir = Path(folder) if folder else model_dir(self.name)
        cfg = json.loads((self.dir / "config.json").read_text(encoding="utf-8"))
        self.n_layers, self.hidden, self.inter = cfg["num_hidden_layers"], cfg["hidden_size"], cfg["intermediate_size"]
        self.model_type = cfg.get("model_type", "")
        index = self.dir / "model.safetensors.index.json"
        if index.exists():
            self.files = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        else:                                            # a small model: one file holds everything
            with safe_open(str(self.dir / "model.safetensors"), framework="pt") as f:
                self.files = {k: "model.safetensors" for k in f.keys()}

    def has(self, key: str) -> bool:
        return key in self.files

    def get(self, key: str) -> torch.Tensor:
        with safe_open(str(self.dir / self.files[key]), framework="pt") as f:
            return f.get_tensor(key).float()

    def mlp(self, layer: int) -> dict[str, torch.Tensor]:
        """gate [inter, hidden], up [inter, hidden], down [hidden, inter], and the gain of the norm the MLP reads
        through. In Qwen3 / Llama-style models that is post_attention_layernorm. OLMo-2 normalises after its MLP
        instead, so its MLP reads the residual directly (gain: ones); `approximate` flags any model we haven't
        checked, as roles do in mapping.py."""
        p = f"model.layers.{layer}."
        norm = p + "post_attention_layernorm.weight"
        reads_through_norm = self.model_type not in POST_NORM_MODELS and self.has(norm)
        return {"gate": self.get(p + "mlp.gate_proj.weight"), "up": self.get(p + "mlp.up_proj.weight"),
                "down": self.get(p + "mlp.down_proj.weight"),
                "norm": self.get(norm) if reads_through_norm else torch.ones(self.hidden),
                "approximate": self.model_type not in CHECKED_MODELS}
