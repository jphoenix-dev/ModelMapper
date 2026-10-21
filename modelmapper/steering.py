"""Steering: hold chosen neurons at a value while the model answers, and read (or judge) what it says.

    steer("Qwen/Qwen3-0.6B", ["L23:2392=x8"])                      # 8 x the neuron's own peak on web text
    steer("Qwen/Qwen3-0.6B", ["L23:2392=12.5", "L4:2581=-3"])      # absolute activation values
    steer(..., prompts=["Tell me a fun fact."], judge="Does the text mention cryptocurrency?")

Strength: "=x<k>" means k times the neuron's normal peak (its maximum on 2M tokens of web text; a negative k uses its
minimum), read from the range table shipped for Qwen3-0.6B and Qwen3-8B, or built with `modelmapper ranges <model>`
for any other model; "=<v>" is an absolute activation, which needs no table. Neurons are clamped at every token of the
reply only, so the question is read normally (reply_only=False also steers the reading), greedy decoding, so a run is reproducible. With `judge=`, each reply
is also scored by a judge model: P(yes) to the question, plus P(fluent English). The published results used TypeSafe's
Jev as the judge (JUDGE_BACKEND=jev, needs JEV_API_KEY); JUDGE_BACKEND=llm uses any OpenAI-compatible model that
returns token log-probabilities (JUDGE_BASE_URL, JUDGE_MODEL), not calibrated like Jev (modelmapper/judge.py).
"""
from __future__ import annotations

import re

import torch

from modelmapper.measure import as_prompt
from modelmapper.model import DEVICE

PROMPTS = [
    "What should I do this weekend?",
    "Tell me about yourself.",
    "Write a short poem about autumn.",
    "Give me a recipe idea for dinner tonight.",
    "What is your favorite place in the world?",
    "Describe a typical morning.",
    "Suggest a name for a new coffee shop.",
    "What's a good hobby to pick up?",
    "Write the opening line of a novel.",
    "How do I stay motivated when studying?",
    "Plan a day trip for me.",
    "What is the most beautiful thing you can think of?",
    "Tell me a fun fact.",
    "Describe your dream house.",
    "What should I paint for my art class?",
    "Write a two-sentence story.",
    "Recommend a vacation destination.",
    "What's something people often misunderstand?",
    "Describe the view from a window.",
    "If you could build anything, what would it be?",
]


def generate(model, tok, prompts, max_new):
    texts = [as_prompt(tok, p) for p in prompts]                  # 0.1.1: as_prompt (the chat turn, or plain)
    tok.padding_side = "left"
    ids = tok(texts, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
    return [tok.decode(o[ids.input_ids.shape[1]:], skip_special_tokens=True).strip() for o in out]


FLUENT = "Is this text fluent, grammatical English that makes sense?"


def parse_unit(text):
    """'L23:2392=x8' -> (23, 2392, 'x', 8.0); 'L23:2392=12.5' -> (23, 2392, 'abs', 12.5)."""
    m = re.fullmatch(r"\s*L?(\d+)[:_]N?(\d+)\s*=\s*(x?)\s*(-?[0-9.]+(?:e-?\d+)?)\s*", text)
    if not m:
        raise ValueError(f"not a neuron setting: {text!r}; write L<layer>:<neuron>=<value> or =x<times its peak>, "
                         "e.g. L23:2392=x8")
    return int(m.group(1)), int(m.group(2)), "x" if m.group(3) else "abs", float(m.group(4))


def resolve(units, model_name, n_layers, width):
    """[(layer, neuron, mode, k)] -> [(layer, neuron, value)], reading the range table only if some unit needs it."""
    from modelmapper.ranges import load_ranges
    table = load_ranges(model_name) if any(mode == "x" for _, _, mode, _ in units) else None
    out = []
    for l, n, mode, k in units:
        if not (0 <= l < n_layers and 0 <= n < width):
            raise ValueError(f"L{l}:{n} does not exist in {model_name} ({n_layers} layers x {width} neurons)")
        if mode == "x":     # x8: 8 x its maximum; x-8: 8 x its minimum (the negative pole)
            out.append((l, n, abs(k) * (table["max"] if k >= 0 else table["min"])[l, n].item()))
        else:
            out.append((l, n, k))
    return out


def steer(model_name, units, prompts=None, max_new=60, judge=None, measurer=None, baseline=True, reply_only=True):
    """Generate replies with the neurons held at their values (and, if baseline, without), optionally judged.

    units: ["L23:2392=x8", ...] or [(layer, neuron, mode, k)]. Returns {"model", "units": [(l, n, value)], "prompts",
    "baseline": [...] or None, "steered": [...], "judge": None or {"question", "steered": [P], "baseline": [P],
    "fluent_steered": [P], "fluent_baseline": [P], "backend"}}."""
    from modelmapper.measure import Measurer
    from modelmapper.model import steer as hold
    m = measurer or Measurer(model_name)
    parsed = [parse_unit(u) if isinstance(u, str) else u for u in units]
    edits = resolve(parsed, model_name, m.L, m.W)
    prompts = list(prompts or PROMPTS[:5])
    base = generate(m.model, m.tok, prompts, max_new) if baseline else None
    with hold(m.model, edits, mode="clamp", reply_only=reply_only):
        steered = generate(m.model, m.tok, prompts, max_new)
    out = {"model": model_name, "units": [(l, n, round(v, 4)) for l, n, v in edits], "reply_only": reply_only, "prompts": prompts,
           "baseline": base, "steered": steered, "judge": None}
    if judge:
        from modelmapper import judge as J
        def ask(replies):
            res = [J.ask(r or "(empty)", {"q": {"type": "noul", "instructions": judge},
                                          "fluent": {"type": "noul", "instructions": FLUENT}}) for r in replies]
            return [x["q"]["noul"] for x in res], [x["fluent"]["noul"] for x in res]
        s_q, s_f = ask(steered)
        b_q, b_f = ask(base) if base else (None, None)
        out["judge"] = {"question": judge, "backend": J.BACKEND, "steered": s_q, "fluent_steered": s_f,
                        "baseline": b_q, "fluent_baseline": b_f}
    return out
