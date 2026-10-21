"""The questions for one concept: the labeller writes them, the model answers, the labeller judges the answers.

    spec = write_questions("Bitcoin", cache, timing={}, budget=0.5)   # recall fragments -> V2 / V3 rewrites (labeller)
    spec = model_answers(measurer, spec, "Bitcoin", cache)              # the model's own answers (no labeller)
    spec = judge_answers("Bitcoin", spec, cache, timing={}, budget=0.5) # which of them name the concept (labeller)

`cache` is a folder (one per model): each concept's spec is saved there after every step and reused, so a concept is
asked about only once. The prompts and rules are the ones run 1 of the map used (loop v2: map_loop.luna_prepare /
luna_judge, concept_pipeline.PROMPT_SYSTEM, recall_variants.REWRITE), copied unchanged; `luna_` in their names is
the labeller model the map used (OpenAI's gpt-6-luna). Spend: USAGE counts tokens; usd() prices them at Luna's prices.
"""
from __future__ import annotations

import json
import threading
import time
from collections import Counter
from pathlib import Path

import torch

from modelmapper import labeller as oai
from modelmapper.measure import as_prompt, leaks, slug
from modelmapper.model import DEVICE

PROMPT_SYSTEM = """You write test prompts for probing whether a language model knows a concept.
Given a concept, return JSON:
{"answer": the concept's name exactly as it would continue a sentence, starting with a space (e.g. " Golden Gate Bridge", " Mount Everest" -> " Everest" if the prompt already ends with "Mount"),
 "recall": [20 prompts],
 "controls": [{"prompt": str, "answer": str}, ... 8 items]}
Rules for "recall": each is a plain sentence fragment (not a question) whose single most natural continuation is the answer.
Vary the angle: history, geography, people, facts, uses, appearance, culture. NEVER include the answer's words or obvious
giveaways like its distinctive name parts. All prompts must end so the answer comes next (e.g. "... is called the").
Rules for "controls": 8 fragments with OTHER single-word-ish answers, each starting with a space: 4 closely related to the
concept's category (similar things the model must still get right) and 4 unrelated well-known facts."""

REWRITE = """You are given sentence fragments that are meant to make a language model say a specific name next.
Many end in "is" or "was", where a model naturally continues with "called", "a", "the", "known as" instead of the name.
For each fragment, write two new versions (keep the meaning and the clues; never include the answer itself):
  "v2": a fragment that ends so that ONLY the name can come next (e.g. end with "is called", "is named",
        "is known as", "was the playwright", "is the city of", "is the game of").
  "v3": a direct question that asks for the name (ending with "?").
Reply with JSON only: {"items": [{"v2": str, "v3": str}, ...]} in the same order as the input."""

PRICE_IN, PRICE_OUT = 0.10e-6, 0.50e-6          # Luna's real price (James)
_tls = threading.local()


class Usage:
    def __init__(self):
        self.lock = threading.Lock(); self.by = {}; self.total = [0, 0, 0]

    def add(self, concept, step, u):
        with self.lock:
            d = self.by.setdefault(concept, {}).setdefault(step, [0, 0, 0])
            for i, k in enumerate(("prompt_tokens", "completion_tokens")):
                d[i] += u.get(k, 0); self.total[i] += u.get(k, 0)
            d[2] += 1; self.total[2] += 1

    def usd(self, x=None):
        x = x or self.total
        return x[0] * PRICE_IN + x[1] * PRICE_OUT


USAGE = Usage()


def luna_json(prompt, system, concept, step):
    """oai.chat_json, but returning exact per-call usage (no shared counter races)."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    data = oai._request("POST", "/chat/completions", {"model": oai.MODEL, "messages": msgs, "response_format": {"type": "json_object"}})
    USAGE.add(concept, step, data.get("usage") or {})
    return json.loads(data["choices"][0]["message"]["content"])


_CACHE = {"dir": None}


def use_cache(folder):
    """Set the spec folder for the steps below (one folder per model)."""
    Path(folder).mkdir(parents=True, exist_ok=True)
    _CACHE["dir"] = Path(folder)


def spec_path(concept):
    if _CACHE["dir"] is None:
        raise RuntimeError("questions.use_cache(<folder>) first")
    return _CACHE["dir"] / f"{slug(concept)}_spec.json"


def load_spec(concept):
    p = spec_path(concept)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def save_spec(concept, spec):
    spec_path(concept).write_text(json.dumps(spec, indent=1, ensure_ascii=False), encoding="utf-8")


def luna_prepare(concept, timing, budget):
    """L1 + L2 (Luna only). Returns spec or raises PromptsFailed-like ValueError."""
    spec = load_spec(concept)
    if spec is None:
        if USAGE.usd() >= budget:
            raise ValueError("Luna budget reached before this concept's spec")
        t = time.time()
        spec = luna_json(f"Concept: {concept}", PROMPT_SYSTEM, concept, "L1_spec")
        timing["L1_spec"] = round(time.time() - t, 2); save_spec(concept, spec)
    if "answer" not in spec or "recall" not in spec:
        raise ValueError(f"Luna returned no answer/recall for this concept: {str(spec)[:160]}")
    if any(isinstance(q, dict) for q in spec.get("recall", [])):         # ISSUES #33 normalisation
        spec["recall"] = [q.get("prompt") or q.get("text") or next(iter(q.values())) if isinstance(q, dict) else q for q in spec["recall"]]
        save_spec(concept, spec)
    if "recall_v2" not in spec:
        t = time.time(); listing = "\n".join(f"{i+1}. {p}" for i, p in enumerate(spec["recall"])); got = []
        for _ in range(3):
            if USAGE.usd() >= budget:
                raise ValueError("Luna budget reached during rewrites")
            out = luna_json(f"Answer (for your reference only, never write it): {spec['answer']}\nFragments:\n{listing}", REWRITE, concept, "L2_rewrite")
            items = out.get("items", []); got.append(len(items))
            if len(items) == len(spec["recall"]) and all("v2" in it and "v3" in it for it in items):
                spec["recall_v2"] = [it["v2"] for it in items]; spec["recall_v3"] = [it["v3"] for it in items]
                save_spec(concept, spec); break
        timing["L2_rewrite"] = round(time.time() - t, 2)
        if "recall_v2" not in spec:
            raise ValueError(f"labeller returned {got} rewrite items for {len(spec['recall'])} recall prompts in 3 attempts")
    return spec


def luna_judge(concept, spec, timing, budget):
    """L3: same prompt and rules as FastPipeline.discover_answers, on replies generated by the GPU (G1)."""
    if "answer_forms_extra" in spec:
        return spec
    t = time.time(); name = spec["answer"].strip(); replies = spec["discovery_replies"]; questions = spec["recall_v3"]
    try:
        if USAGE.usd() >= budget:
            raise ValueError("budget")
        ok = luna_json(
            f"Target name: {name}\nFor each numbered reply, is it this specific thing's own name (allowing "
            f"singular/plural, capitalisation, a well-known synonym or short name)? Say false for a different "
            f"thing, and false for a generic descriptive name made from place or category words (e.g. 'River X "
            f"Bridge', 'Town Y Museum') unless that IS the target's established name. "
            f"Reply JSON {{\"ok\": [true/false, ...]}} in order.\n" + "\n".join(f"{i+1}. {r}" for i, r in enumerate(replies)),
            None, concept, "L3_discovery_judge")["ok"]
    except Exception:  # noqa: BLE001  (as discover_answers: a failed judge accepts nothing)
        ok = [False] * len(replies)
    good = [r for r, v in zip(replies, ok) if v and r and sum(leaks(q, r) for q in questions) < 0.3 * len(questions)]
    spec["answer_forms_extra"] = [f for f, n in Counter(good).most_common(3) if n >= 2]
    spec["discovered_answers"] = [{"reply": r, "names_concept": bool(v)} for r, v in zip(replies, ok)]
    save_spec(concept, spec); timing["L3_discovery_judge"] = round(time.time() - t, 2)
    return spec


def write_questions(concept, cache, timing=None, budget=0.5):
    """Steps L1 + L2: the labeller's recall fragments for the concept, rewritten as V2 fragments and V3 questions."""
    use_cache(cache)
    return luna_prepare(concept, {} if timing is None else timing, budget)


def model_answers(measurer, spec, concept, cache):
    """Step G1: the model answers each V3 question in its own words (greedy, 16 tokens), as run 1's prepare step did."""
    use_cache(cache)
    if "discovery_replies" in spec:
        return spec
    tok = measurer.tok
    chats = [as_prompt(tok, q + " Answer with just the name.") for q in spec["recall_v3"]]   # 0.1.1: as_prompt
    side = tok.padding_side; tok.padding_side = "left"
    try:
        ids = tok(chats, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            g = measurer.model.generate(**ids, max_new_tokens=16, do_sample=False, pad_token_id=tok.pad_token_id)
    finally:
        tok.padding_side = side
    spec["discovery_replies"] = [tok.decode(o[ids.input_ids.shape[1]:], skip_special_tokens=True).strip().split("\n")[0].strip().rstrip(".")
                                 for o in g]
    save_spec(concept, spec)
    return spec


def judge_answers(concept, spec, cache, timing=None, budget=0.5):
    """Step L3: which of the model's own answers name the concept (extra answer forms). Returns (spec, note): note is
    None, or why no extra forms were used (the check did not run: spend cap reached or a labeller error)."""
    use_cache(cache)
    before = "answer_forms_extra" in spec
    spec = luna_judge(concept, spec, {} if timing is None else timing, budget)
    note = None
    if not before and not USAGE.by.get(concept, {}).get("L3_discovery_judge"):
        note = ("the labeller's check of the model's own answers did not run (spend cap reached or a labeller error), "
                "so no extra answer forms were used")
    return spec, note
