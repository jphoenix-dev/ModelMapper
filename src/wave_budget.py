"""Run wave_free.py unchanged on a wave file, round-robin across its categories, under a Luna budget (research).

- Round-robin: the wave is rewritten so categories are interleaved (one concept from each tier in turn; category keys
  become '<tier>#<k>'), so a budget stop leaves the tiers balanced instead of cutting off the last tiers.
- Budget: every Luna call (oai.chat) first checks the ESTIMATED spend so far - at the same deliberately pessimistic
  price as luna_firing_judge.py v2 (Luna's real $0.10 / M input, $0.50 / M output tokens) - and stops the run (logged in
  results/wave_budget_<tag>.json and the console, not silent) once it reaches BUDGET_USD. The concept in progress at
  that moment is not saved; everything before it is (wave_free.py saves after every concept).
Usage: RECALL_VARIANT=V3 CONCEPT_OUT_DIR=<dir> python src/wave_budget.py <wave json> <tag> <budget_usd>
"""
import json
import sys
from pathlib import Path

import oai

ROOT = Path(__file__).resolve().parents[1]
PRICE_IN, PRICE_OUT = 0.10e-6, 0.50e-6          # Luna's real price (James, 16 Oct)


def main():
    wave_p, tag, budget = sys.argv[1], sys.argv[2], float(sys.argv[3])
    wave = json.loads((ROOT / wave_p).read_text(encoding="utf-8"))
    cats = list(wave["concepts"].items()); rr = {}
    for k in range(max(len(v) for _, v in cats)):
        for name, cs in cats:
            if k < len(cs):
                rr[f"{name}#{k + 1}"] = [cs[k]]
    rr_p = ROOT / f"prompts/_rr_{tag}.json"
    rr_p.write_text(json.dumps({"concepts": rr}, indent=1, ensure_ascii=False), encoding="utf-8")
    state_p = ROOT / f"results/wave_budget_{tag}.json"
    state = json.loads(state_p.read_text(encoding="utf-8")) if state_p.exists() else {"estimated_usd_before": 0.0}
    before = state.get("estimated_usd", 0.0)
    orig = oai.chat

    def est():
        return before + oai.usage["input_tokens"] * PRICE_IN + oai.usage["output_tokens"] * PRICE_OUT

    def chat(*a, **k):
        if est() >= budget:
            state.update(stopped=True, estimated_usd=round(est(), 4), usage=dict(oai.usage))
            state_p.write_text(json.dumps(state, indent=1), encoding="utf-8")
            print(f"BUDGET STOP: estimated Luna spend ${est():.3f} >= ${budget} (assumed price); stopping before the next call", flush=True)
            raise SystemExit(3)
        r = orig(*a, **k)
        state.update(estimated_usd=round(est(), 4), usage=dict(oai.usage), price_assumed=[PRICE_IN, PRICE_OUT])
        state_p.write_text(json.dumps(state, indent=1), encoding="utf-8")
        return r
    oai.chat = chat                          # chat_json looks chat up at call time, so it is covered too
    import wave_free
    sys.argv = ["wave_free.py", str(rr_p.relative_to(ROOT))]
    wave_free.main()
    print(f"wave finished; estimated Luna spend ${est():.3f}", flush=True)


if __name__ == "__main__":
    main()
