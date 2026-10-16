"""Step 5 of Experiment GG: clamp neurons high and see what the model talks about.

Each neuron is clamped to `scale` x its own peak on web text (from data/stats_<model>.pt) at every
token while the model answers 20 neutral prompts (batched). Positive-pole units are clamped to
scale x max; units written with a trailing '-' are clamped to scale x min (their negative pole).
Replies are judged by Jev: does it bring up the Golden Gate Bridge? Is it still fluent English?

Usage:
  python src/steer_test.py <model> <units> <scales> [--no-jev] [--n 20] [--tokens 60] [--out f.json]
  units:  "26:1311,25:6245"   layer:neuron, comma separated; "12:9831-" = negative pole
          "random:10:7+12+26" = 10 random neurons drawn from layers 7, 12 and 26 (seeded)
  scales: "0,2,5,10,20"
"""
import argparse
import json
import random
from pathlib import Path

import torch

from neurons import load, steer, DEVICE

ROOT = Path(__file__).resolve().parents[1]
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


def parse_units(spec, stats, width):
    units = []
    for part in spec.split(","):
        if part.startswith("random:"):
            _, n, layers = part.split(":", 2)
            rng = random.Random(0)
            ls = [int(x) for x in layers.split("+")]
            for _ in range(int(n)):
                l, nn = rng.choice(ls), rng.randrange(width)
                units.append((l, nn, stats["max"][l, nn].item()))
            continue
        neg = part.endswith("-")
        layer, neuron = map(int, part.rstrip("-").split(":"))
        ref = (stats["min"] if neg else stats["max"])[layer, neuron].item()
        units.append((layer, neuron, ref))
    return units


def generate(model, tok, prompts, max_new):
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False) for p in prompts]
    tok.padding_side = "left"
    ids = tok(texts, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
    return [tok.decode(o[ids.input_ids.shape[1]:], skip_special_tokens=True).strip() for o in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("units"); ap.add_argument("scales")
    ap.add_argument("--no-jev", action="store_true")
    ap.add_argument("--n", type=int, default=20); ap.add_argument("--tokens", type=int, default=60)
    ap.add_argument("--out")
    a = ap.parse_args()
    model, tok = load(a.model)
    stats = torch.load(ROOT / f"data/stats_{a.model.split('/')[-1]}.pt")
    units = parse_units(a.units, stats, model.config.intermediate_size)
    print("clamping:", [(l, n, round(r, 2)) for l, n, r in units])
    if not a.no_jev:
        import jev
    results = []
    for scale in [float(x) for x in a.scales.split(",")]:
        edits = [(l, n, scale * ref) for l, n, ref in units]
        if scale > 0:
            with steer(model, edits):
                replies = generate(model, tok, PROMPTS[:a.n], a.tokens)
        else:
            replies = generate(model, tok, PROMPTS[:a.n], a.tokens)
        row = {"scale": scale, "replies": replies,
               "keyword_rate": sum("golden gate" in r.lower() for r in replies) / len(replies)}
        if not a.no_jev:
            gg, fl = [], []
            for r in replies:
                ans = jev.ask(r or "(empty)", {
                    "gg": {"type": "noul", "instructions": "Does this text mention, describe, or seem preoccupied with the Golden Gate Bridge?"},
                    "fluent": {"type": "noul", "instructions": "Is this text fluent, grammatical English that makes sense?"},
                })
                gg.append(ans["gg"]["noul"]); fl.append(ans["fluent"]["noul"])
            row["gg_rate"] = sum(x >= 0.5 for x in gg) / len(gg)
            row["fluent_rate"] = sum(x >= 0.5 for x in fl) / len(fl)
        results.append(row)
        print(f"\n### scale {scale:g}x peak: 'golden gate' in {row['keyword_rate']:.0%} of replies"
              + (f" | Jev: GG {row['gg_rate']:.0%}, fluent {row['fluent_rate']:.0%}" if "gg_rate" in row else ""))
        for p, r in list(zip(PROMPTS, replies))[:3]:
            print(f"  Q: {p}\n  A: {r[:200]!r}")
    if a.out:
        Path(a.out).write_text(json.dumps({"units": units, "results": results}, indent=1, ensure_ascii=False),
                               encoding="utf-8")
    if not a.no_jev:
        print(f"\nJev: {jev.usage}, ~${jev.cost():.4f}")


if __name__ == "__main__":
    main()
