import json
import os

import pytest


def pytest_configure(config):
    """As `modelmapper check` does: with no GPU and nothing setting a dtype, use float32 on the CPU (much faster)."""
    from modelmapper import settings
    if settings.env("MODEL_MAPPER_DTYPE"):
        return
    try:
        import torch
        gpu = (torch.cuda.is_available() and torch.cuda.device_count() > 0) or torch.backends.mps.is_available()
    except Exception:  # noqa: BLE001
        gpu = False
    if not gpu:
        os.environ["MODEL_MAPPER_DTYPE"] = "float32"
        if not settings.env("MODEL_MAPPER_DEVICE"):
            os.environ["MODEL_MAPPER_DEVICE"] = "cpu"


def make_record(concept="Bitcoin", n=40, status="ok", model="Qwen/Qwen3-0.6B"):
    """A small record in the loop_v2.0 format (FORMAT.md): n candidates with drops n/10, (n-1)/10, ..."""
    cands = [{"neuron": f"L{i % 28}:{100 + i}", "single_drop": round((n - i) / 10, 3), "attr_rank": i,
              "act_answer_pos": 1.0} for i in range(n)]
    rec = {"concept": concept,
           "row": {"status": status, "answer": concept, "answer_form": concept, "recall": 9, "k_mean": 5, "k_median": 5,
                   "k_flip": 1, "drops": [1.4, 2.2, 2.8, 3.6, 6.6, 7.2, 11.7], "random_drops": [0, 0, 0.01, 0, -0.01, -0.04, -0.12]},
           "rank": {"baseline_mean_logodds": 0.26, "candidates_all200": cands, "top": cands[:30],
                    "head_knockout_mean": {"L19H29": 0.4, "L3H1": 0.9}},
           "v2": {"provenance": {"schema": "loop_v2.0", "model": model},
                  "null_single_drops": {"L23:198": [0.0, 0.02], "L27:1271": [0.0, 0.0]},
                  "roles_top30": [{"neuron": cands[0]["neuron"], "role": "say", "job_tokens": [" Bitcoin", " BTC"]}]}}
    if status != "ok":
        rec["rank"] = None
    return rec


@pytest.fixture
def record():
    return make_record()


@pytest.fixture
def records_dir(tmp_path):
    d = tmp_path / "Qwen3-0.6B"
    d.mkdir()
    for c in ("Bitcoin", "YouTube"):
        (d / f"{c.lower()}.json").write_text(json.dumps(make_record(c)), encoding="utf-8")
    return d
