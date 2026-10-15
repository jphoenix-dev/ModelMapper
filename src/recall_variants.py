"""Which recall-prompt format lets the model show what it knows? (ISSUES #9/#10)

For every concept spec already generated (results/concepts/*_spec.json), score four variants:
  V0  as now: P(answer) right after the prompt fragment ("... is")
  V1  bridge: max over a short bridge ("", " called", " known as", " named", " the") of P(bridge + answer)
  V2  rewritten: Luna rewrites each fragment to end where only the name can follow ("... is called")
  V3  question: chat format, "<question>? Answer with just the name." -> P(answer at the start of the reply)
A prompt passes if P >= 0.2 (same bar as the pipeline). Real concepts should pass often; the 3 made-up concepts must
stay near 0 (otherwise the variant leaks the answer). Rewrites are cached in each spec file under "recall_v2"/"recall_v3".

Usage: python src/recall_variants.py rewrite        (API only: Luna writes V2/V3 prompts)
       python src/recall_variants.py score <model>  (GPU: scores all variants, writes results/recall_variants.json)
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPECS = sorted((ROOT / "results/concepts").glob("*_spec.json"))
BRIDGES = ["", " called", " known as", " named", " the"]
REWRITE = """You are given sentence fragments that are meant to make a language model say a specific name next.
Many end in "is" or "was", where a model naturally continues with "called", "a", "the", "known as" instead of the name.
For each fragment, write two new versions (keep the meaning and the clues; never include the answer itself):
  "v2": a fragment that ends so that ONLY the name can come next (e.g. end with "is called", "is named",
        "is known as", "was the playwright", "is the city of", "is the game of").
  "v3": a direct question that asks for the name (ending with "?").
Reply with JSON only: {"items": [{"v2": str, "v3": str}, ...]} in the same order as the input."""


def rewrite():
    import oai

    def one(f):
        s = json.loads(f.read_text(encoding="utf-8"))
        if "recall_v2" in s:
            return f.name, "cached"
        out = oai.chat_json(f"Answer (for your reference only, never write it): {s['answer']}\nFragments:\n" +
                            "\n".join(f"{i+1}. {p}" for i, p in enumerate(s["recall"])), REWRITE)
        items = out.get("items", [])
        if len(items) != len(s["recall"]):
            return f.name, f"length mismatch {len(items)}"
        s["recall_v2"] = [it["v2"] for it in items]
        s["recall_v3"] = [it["v3"] for it in items]
        f.write_text(json.dumps(s, indent=1, ensure_ascii=False), encoding="utf-8")
        return f.name, "ok"
    with ThreadPoolExecutor(32) as ex:
        for name, st in ex.map(one, SPECS):
            print(f"{name}: {st}")
    print(oai.usage)


def score(model_name):
    import torch
    from pipeline_fast import FastPipeline
    pipe = FastPipeline(model_name)
    tok = pipe.tok
    res = {}
    for f in SPECS:
        s = json.loads(f.read_text(encoding="utf-8"))
        if "recall_v2" not in s:
            continue
        ans = tok.encode(s["answer"])
        v0 = pipe.batch_probs([(p, ans) for p in s["recall"]])
        v1 = [0.0] * len(s["recall"])
        for b in BRIDGES:
            pb = pipe.batch_probs([(p, tok.encode(b + s["answer"]) if b else ans) for p in s["recall"]])
            v1 = [max(x, y) for x, y in zip(v1, pb)]
        v2 = pipe.batch_probs([(p, ans) for p in s["recall_v2"]])
        chat = [tok.apply_chat_template([{"role": "user", "content": q + " Answer with just the name."}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False) for q in s["recall_v3"]]
        ans3 = tok.encode(s["answer"].strip())
        v3 = pipe.batch_probs([(c, ans3) for c in chat])
        rate = lambda xs: round(sum(x >= 0.2 for x in xs) / len(xs), 3)   # noqa: E731
        res[f.stem.replace("_spec", "")] = {"answer": s["answer"], "V0": rate(v0), "V1": rate(v1), "V2": rate(v2), "V3": rate(v3),
                                            "mean_p": {k: round(sum(v) / len(v), 3) for k, v in (("V0", v0), ("V1", v1), ("V2", v2), ("V3", v3))}}
        r = res[f.stem.replace("_spec", "")]
        print(f"{f.stem.replace('_spec',''):<45} pass rate V0 {r['V0']:.0%} | V1 {r['V1']:.0%} | V2 {r['V2']:.0%} | V3 {r['V3']:.0%}", flush=True)
    (ROOT / "results/recall_variants.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    fake = [k for k in res if "fictional" in k]
    real = [k for k in res if k not in fake]
    for v in ("V0", "V1", "V2", "V3"):
        ok = sum(res[k][v] >= 0.25 for k in real)       # >= 5 of 20 prompts pass = pipeline would proceed
        print(f"{v}: real concepts recalled {ok}/{len(real)} | mean pass {sum(res[k][v] for k in real)/len(real):.0%} | "
              f"made-up mean pass {sum(res[k][v] for k in fake)/max(len(fake),1):.0%} (must stay ~0)")


if __name__ == "__main__":
    rewrite() if sys.argv[1] == "rewrite" else score(sys.argv[2])
