"""Answer discovery (ISSUES #29): let the model say its own answer, instead of guessing the wording.

For each concept: greedy-generate the model's reply (no steering) to every V3 recall prompt, take the first line, and
ask Luna whether it names the concept. The accepted replies' most common wordings are saved into the concept's spec as
"answer_forms_extra", which pipeline_fast's V3 scoring then tries alongside the usual forms. Re-run the pipeline with
--force afterwards. Reports every generated answer (no truncation of what the model said beyond its first line).

Usage: CONCEPT_OUT_DIR=results/concepts_w2 python src/answer_discovery.py "<concept>" [...]
"""
import json
import sys
from collections import Counter

import torch

from modelmapper.core.config import DEFAULT_MODEL
from modelmapper.core import concept_pipeline as cp
from modelmapper.core.neurons import load, DEVICE


def main():
    model, tok = load(DEFAULT_MODEL)
    tok.padding_side = "left"
    for concept in [c.strip() for c in sys.argv[1:]]:
        p = cp.OUT / f"{cp.slug(concept)}_spec.json"
        spec = json.loads(p.read_text(encoding="utf-8"))
        qs = spec["recall_v3"]
        chat = [tok.apply_chat_template([{"role": "user", "content": q + " Answer with just the name."}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False) for q in qs]
        ids = tok(chat, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            g = model.generate(**ids, max_new_tokens=16, do_sample=False, pad_token_id=tok.pad_token_id)
        answers = [tok.decode(o[ids.input_ids.shape[1]:], skip_special_tokens=True).strip().split("\n")[0].strip().rstrip(".")
                   for o in g]
        verdict = cp.oai.chat_json(
            f"Concept: {concept}\nFor each numbered reply, does it name this concept (allowing synonyms, singular/plural, "
            f"common short names)? Reply JSON {{\"ok\": [true/false, ...]}} in order.\n"
            + "\n".join(f"{i+1}. {a}" for i, a in enumerate(answers)))["ok"]
        ok = [a for a, v in zip(answers, verdict) if v]
        forms = [f for f, _ in Counter(ok).most_common(3)]
        print(f"{concept}: {len(ok)}/{len(answers)} replies name it; top forms {forms}", flush=True)
        for a, v in zip(answers, verdict):
            print(f"    {'OK ' if v else '-- '}{a}", flush=True)
        spec["answer_forms_extra"] = forms
        p.write_text(json.dumps(spec, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
