"""Measurements on a loaded model: P(answer), exact log-odds, attribution, neuron and head knockouts.

    m = Measurer("Qwen/Qwen3-0.6B")
    m.batch_logodds([(prompt, m.tok.encode(" Paris"))])                 # exact log-odds, float64
    with steer(m.model, [(23, 2392, 0.0)]): ...                          # a neuron switched off (modelmapper.model)

The method bodies are the ones every published map used (concept_pipeline.Pipeline and pipeline_fast.FastPipeline
in the research code), copied unchanged; only the class around them is new, holding just what mapping needs.
"""
from __future__ import annotations

import contextlib
import hashlib
import math
import re

import torch

from modelmapper.model import DEVICE, load, mlps
from modelmapper.settings import env

MIN_P = float(env("CONCEPT_MIN_P", "0.2"))   # a question counts as recalled when P(answer) >= MIN_P (METHOD.md)
SLUG_MAX = 60

# How a question is put to the model. "chat" (default; every published map): one user turn through the tokenizer's chat
# template, thinking off. "plain" (MODEL_MAPPER_PROMPT=plain; experimental, uncalibrated): PLAIN below, for base models
# without a chat template. The format used is recorded in every record's provenance.
PROMPT_FORMAT = (env("MODEL_MAPPER_PROMPT", "chat") or "chat").strip().lower()
PLAIN = "Question: {text}\nAnswer:\n"


class PromptFormatError(ValueError):
    """The model cannot be asked questions in the chosen format (e.g. a base model with no chat template)."""


def prompt_format(tok) -> str:
    """'chat' or 'plain' for this tokenizer; raises PromptFormatError (before any labeller call) if neither applies."""
    if PROMPT_FORMAT == "plain":
        return "plain"
    if PROMPT_FORMAT != "chat":
        raise PromptFormatError(f"MODEL_MAPPER_PROMPT={PROMPT_FORMAT!r}: use 'chat' (the default) or 'plain'")
    if getattr(tok, "chat_template", None):
        return "chat"
    raise PromptFormatError(
        "this model's tokenizer has no chat template (a base model?). map-concept asks each question as a chat turn, "
        "as every published map did. Use the model's instruct / chat version, or set MODEL_MAPPER_PROMPT=plain to ask "
        "'Question: ...' / 'Answer:' instead (experimental and uncalibrated; recorded in each record).")


def as_prompt(tok, text: str) -> str:
    """`text` as the model sees it: a chat user turn (thinking off), or PLAIN with MODEL_MAPPER_PROMPT=plain."""
    if prompt_format(tok) == "chat":
        return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    return PLAIN.format(text=text)


def slug(s):
    """File-name stem for a concept. Names longer than SLUG_MAX (the made-up controls are whole descriptions) are cut
    to 50 characters plus a hash of the full name, so result paths stay well under Windows' 260-character limit."""
    out = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    if len(out) > SLUG_MAX:
        out = out[:50].rstrip("_") + "_" + hashlib.sha1(out.encode()).hexdigest()[:8]
    return out


GENERIC_ANSWER_WORDS = {"tree", "bridge", "tower", "mountain", "river", "desert", "city", "lake", "island", "plant",
                        "animal", "game", "machine", "engine", "table", "wall", "game", "instrument", "building"}


def f64(x):
    """x as float64. Apple MPS has no float64, so there x is moved to the CPU first (as float32, which holds every bf16 /
    fp16 / fp32 value exactly); on CUDA and the CPU this is x.double(), unchanged."""
    return (x.float().cpu() if x.device.type == "mps" else x).double()


def leaks(prompt, answer):
    """Does the prompt give away a DISTINCTIVE answer word? (ISSUES #22)
    Capitalised answer words (names: Golden, Eiffel, Shakespeare) count only when capitalised in the prompt, so an
    ordinary "the bridge" is not a leak for "Golden Gate Bridge"; lowercase answers (coffee, chess) count in any case."""
    for w in re.findall(r"[A-Za-z]{4,}", answer):
        if w.lower() in GENERIC_ANSWER_WORDS and not w[0].isupper():
            continue                       # "tree" in "oak tree" is the question's category, not a give-away (ISSUES #32)
        if w[0].isupper():
            if re.search(r"\b" + re.escape(w) + r"\b", prompt):
                return True
        elif re.search(r"\b" + re.escape(w) + r"\b", prompt.lower()):
            return True
    return False


class Measurer:
    """A loaded model plus the measurements mapping uses. model / tok: pass an already-loaded pair to reuse it."""

    def __init__(self, model_name, model=None, tok=None):
        self.model_name = model_name
        self.model, self.tok = (model, tok) if model is not None else load(model_name)
        self.model.requires_grad_(False)
        self.L, self.W = self.model.config.num_hidden_layers, self.model.config.intermediate_size

    def answer_ids(self, answer):
        return self.tok.encode(answer)

    def seq_logprob(self, prompt, ans):
        """log P(answer tokens | prompt), teacher-forced: sum over the answer's pieces."""
        p_ids = self.tok.encode(prompt)
        ids = torch.tensor([p_ids + ans[:-1]], device=DEVICE)
        lp = self.model(input_ids=ids).logits[0, len(p_ids) - 1:].float().log_softmax(-1)
        return lp[torch.arange(len(ans)), torch.tensor(ans, device=DEVICE)].sum()

    def probs(self, prompts, ans, edits=()):
        from modelmapper.model import steer
        return self.batch_probs([(p, ans) for p in prompts], steer(self.model, list(edits)) if edits else None)

    def batch_probs(self, pairs, ctx=None):
        """pairs: [(prompt, answer_ids)] -> [P(answer | prompt)], one padded forward pass for all."""
        seqs, spans = [], []
        for prompt, ans in pairs:
            p_ids = self.tok.encode(prompt)
            seqs.append(p_ids + ans[:-1])
            spans.append((len(p_ids) - 1, ans))
        T = max(map(len, seqs))
        pad = self.tok.pad_token_id or 0
        ids = torch.full((len(seqs), T), pad, device=DEVICE)
        mask = torch.zeros_like(ids)
        for i, s in enumerate(seqs):                       # right padding: causal attention leaves real tokens untouched
            ids[i, :len(s)] = torch.tensor(s, device=DEVICE); mask[i, :len(s)] = 1
        with torch.no_grad(), (ctx or contextlib.nullcontext()):
            lp = self.model(input_ids=ids, attention_mask=mask).logits.float().log_softmax(-1)
        out = []
        for i, (start, ans) in enumerate(spans):
            pos = torch.arange(start, start + len(ans), device=DEVICE)
            out.append(lp[i, pos, torch.tensor(ans, device=DEVICE)].sum().exp().item())
        return out

    def batch_logodds(self, pairs, ctx=None):
        """pairs: [(prompt, answer_ids)] -> [exact log-odds of the whole answer], log P - log(1 - P), no clamp.

        ISSUES #35: log(p / (1 - p)) on batch_probs' float32 output, with p clamped to 1 - 1e-9, capped log-odds at
        20.72 and lost precision above ~14 (float32 rounds log P to 0 near certainty). Here the log-softmax is taken in
        float64 at the answer positions and log(1 - P) via expm1, so it is exact up to ~36; beyond that it raises
        instead of capping."""
        seqs, spans = [], []
        for prompt, ans in pairs:
            p_ids = self.tok.encode(prompt)
            seqs.append(p_ids + ans[:-1])
            spans.append((len(p_ids) - 1, ans))
        T = max(map(len, seqs))
        pad = self.tok.pad_token_id or 0
        ids = torch.full((len(seqs), T), pad, device=DEVICE)
        mask = torch.zeros_like(ids)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s, device=DEVICE); mask[i, :len(s)] = 1
        with torch.no_grad(), (ctx or contextlib.nullcontext()):
            logits = self.model(input_ids=ids, attention_mask=mask).logits
        out = []
        for i, (start, ans) in enumerate(spans):
            pos = torch.arange(start, start + len(ans), device=DEVICE)
            lp = f64(logits[i, pos]).log_softmax(-1)
            s = lp[torch.arange(len(ans), device=lp.device), torch.tensor(ans, device=lp.device)].sum().item()
            if s >= 0.0:
                raise ValueError(f"log P(answer) rounds to 0 even in float64 (log-odds > ~36): {pairs[i][0][:60]!r}")
            out.append(s - math.log(-math.expm1(s)))
        return out

    def attribute(self, prompts, tid):
        """activation x gradient for every MLP neuron and every attention head (head = its slice of
        o_proj's input), per prompt and per token position. Big per-position maps are kept on CPU."""
        H = self.model.config.num_attention_heads
        acts, heads, hooks = {}, {}, []
        hooks.append(self.model.model.embed_tokens.register_forward_hook(lambda m, a, o: o.requires_grad_(True)))
        for l in range(self.L):
            def keep(m, args, l=l):
                args[0].retain_grad(); acts[l] = args[0]

            def keep_h(m, args, l=l):
                args[0].retain_grad(); heads[l] = args[0]
            hooks.append(mlps(self.model)[l].down_proj.register_forward_pre_hook(keep))
            hooks.append(self.model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(keep_h))
        attr = torch.zeros(self.L, self.W, device=DEVICE)
        head_attr = torch.zeros(self.L, H, device=DEVICE)
        last_act = torch.zeros(self.L, self.W, device=DEVICE)
        per_prompt, pos_maps, head_pos, tokens = [], [], [], []
        try:
            for p in prompts:
                with torch.enable_grad():
                    lp = self.seq_logprob(p, tid)
                    lp.backward()
                n_p = len(self.tok.encode(p))
                pa = torch.stack([(acts[l].float() * acts[l].grad.float())[0] for l in range(self.L)])   # [L, T, W]
                ha = torch.stack([(heads[l].float() * heads[l].grad.float())[0].view(heads[l].shape[1], H, -1).sum(-1)
                                  for l in range(self.L)])                                                # [L, T, H]
                attr += pa.sum(1) / len(prompts)
                head_attr += ha.sum(1) / len(prompts)
                last_act += torch.stack([acts[l][0, n_p - 1].float() for l in range(self.L)]) / len(prompts)
                per_prompt.append(pa.sum(1).cpu())
                pos_maps.append(pa.half().cpu())
                head_pos.append(ha.cpu())
                ids = self.tok.encode(p) + tid[:-1]
                tokens.append([self.tok.decode([t]) for t in ids])
                acts.clear(); heads.clear()
        finally:
            for h in hooks:
                h.remove()
        top = [divmod(i.item(), self.W) for i in attr.flatten().argsort(descending=True)[:200]]
        bottom = [divmod(i.item(), self.W) for i in attr.flatten().argsort()[:50]]
        top_heads = [divmod(i.item(), H) for i in head_attr.flatten().argsort(descending=True)[:50]]
        return {"top": top, "bottom": bottom, "attr": attr, "last_act": last_act, "per_prompt": per_prompt,
                "pos_maps": pos_maps, "tokens": tokens, "head_attr": head_attr, "head_pos": head_pos,
                "top_heads": top_heads, "H": H}

    @contextlib.contextmanager
    def ablate_heads(self, hs):
        """Zero chosen heads' outputs (their slice of o_proj's input) at every position."""
        H = self.model.config.num_attention_heads
        by_layer, handles = {}, []
        for l, h in hs:
            by_layer.setdefault(l, []).append(h)
        for l, hl in by_layer.items():
            def hook(m, args, hl=hl):
                x = args[0].clone()
                d = x.shape[-1] // H
                for h in hl:
                    x[..., h * d:(h + 1) * d] = 0
                return (x,)
            handles.append(self.model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(hook))
        try:
            yield
        finally:
            for h in handles:
                h.remove()
