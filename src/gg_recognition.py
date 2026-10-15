"""Does Qwen recognise the 'unnamed' descriptions as the Golden Gate Bridge?

If it doesn't, a neuron failing to fire on them tells us nothing. Behavioural check: ask the model
directly, greedy decoding, no thinking.

Usage: python src/gg_recognition.py <model>
"""
import json
import sys
from pathlib import Path

import torch

from neurons import load

ROOT = Path(__file__).resolve().parents[1]
texts = json.loads((ROOT / "prompts/golden_gate.json").read_text(encoding="utf-8"))["gg_unnamed"]
model, tok = load(sys.argv[1])
hits = 0
for t in texts:
    q = f'Which specific bridge is this text about? Answer with just the bridge\'s name.\n\n"{t}"'
    text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=12, do_sample=False)
    ans = tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True).strip()
    hit = "golden gate" in ans.lower()
    hits += hit
    print(f"{'OK ' if hit else 'NO '} {ans!r:<32} <- {t[:70]}")
print(f"\nrecognised {hits}/{len(texts)}")
