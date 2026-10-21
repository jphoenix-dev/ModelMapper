"""Client for the *judge*: calibrated probabilities, no text. Default backend: TypeSafe's Jev ("System One").

One call = one `state` + any number of typed questions, all answered independently and in
parallel against that state. Billing is per input token only (~$0.042 / M); output is free.
Limits (jev-1.13): state + all questions <= 64k tokens, state + any one question <= 32k.

    jev.noul("The Golden Gate Bridge is in SF.", "Does the text mention a bridge?")  # -> 0.99
    jev.intruder(["bridge ...", "bridge ...", "bread ..."])                          # -> {2: 0.97, ...}

The API key is read lazily from $JEV_API_KEY (or $TYPESAFE_API_KEY), else from ./.env,
else from ./jevkey.env (the working folder).

Alternative backend, JUDGE_BACKEND=llm: the same three question types answered by any OpenAI-compatible
chat-completions endpoint that returns token log-probabilities (vLLM, llama.cpp server, OpenAI non-reasoning
models, ...). Each question becomes one call with max_tokens=1; the answer distribution is read from the
first token's top_logprobs (Yes/No for noul, option letters for choice/score), renormalised over the valid
answers. All results in this repo used Jev; the llm backend is untested against them and is not calibrated
the same way, so re-run the controls (spliced dossiers, random snippets) before comparing numbers.
    JUDGE_BASE_URL   default https://api.openai.com/v1
    JUDGE_MODEL      required for the llm backend
    JUDGE_API_KEY    optional (falls back to OPENAI_API_KEY; omit for a self-hosted endpoint without auth)
    JUDGE_EXTRA_BODY optional JSON merged into each request, e.g.
                     {"chat_template_kwargs": {"enable_thinking": false}} for a local Qwen3
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from modelmapper.settings import WORK_DIR, env

BACKEND = env("JUDGE_BACKEND", "jev")   # "jev" or "llm"
URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
USD_PER_MTOK = 0.042
KEY_FILE = Path(env("JEV_KEY_FILE") or WORK_DIR / "jevkey.env")    # JEV_KEY_FILE: a key file kept elsewhere
RETRY_STATUS = {408, 429, 500, 502, 503, 504, 529}

usage = {"calls": 0, "input_tokens": 0}   # running tally for this process
_client: httpx.Client | None = None


class JevError(RuntimeError):
    pass


def cost() -> float:
    """Estimated $ spent so far in this process (Jev pricing; $JUDGE_USD_PER_MTOK for the llm backend, default 0)."""
    rate = float(env("JUDGE_USD_PER_MTOK", "0")) if BACKEND == "llm" else USD_PER_MTOK
    return usage["input_tokens"] * rate / 1e6


def _load_key() -> str:
    """Env var / .env first; else jevkey.env: `TYPESAFE_API_KEY=...`, any `NAME=...`, or a bare key on one line."""
    for var in ("JEV_API_KEY", "TYPESAFE_API_KEY"):
        if env(var):
            return env(var)
    if not KEY_FILE.exists():
        raise JevError(f"no API key found in $JEV_API_KEY, $TYPESAFE_API_KEY, .env or {KEY_FILE}")
    lines = [l.strip() for l in KEY_FILE.read_text(encoding="utf-8-sig").splitlines()
             if l.strip() and not l.strip().startswith("#")]
    pick = next((l for l in lines if l.removeprefix("export ").startswith("TYPESAFE_API_KEY=")),
                lines[0] if lines else "")
    key = pick.split("=", 1)[1] if "=" in pick else pick
    key = key.strip().strip("'\"").strip()
    if not key:
        raise JevError(f"no API key found in $JEV_API_KEY, $TYPESAFE_API_KEY, .env or {KEY_FILE}")
    os.environ["TYPESAFE_API_KEY"] = key
    return key


def _http() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(timeout=60.0, headers={"Authorization": f"Bearer {_load_key()}"})
    return _client


def ask(state, questions: dict, model: str = MODEL, max_retries: int = 5) -> dict:
    """Raw call. questions: {id: {"type": "noul"|"choice"|"score", "instructions": ..., "criteria": ...}}.

    Returns the `answers` dict keyed by your ids. Retries 408/429/5xx/529 and network errors
    with exponential backoff (honouring Retry-After); raises JevError otherwise.
    """
    if BACKEND == "llm":
        return _ask_llm(state, questions, max_retries)
    body = {"state": state, "model": model, "questions": questions}
    for attempt in range(max_retries + 1):
        try:
            r = _http().post(URL, json=body)
        except (httpx.TransportError, httpx.TimeoutException) as e:
            if attempt == max_retries:
                raise JevError(f"network error after {attempt + 1} attempts: {e!r}") from e
        else:
            if r.status_code == 200:
                data = r.json()
                usage["calls"] += 1
                usage["input_tokens"] += (data.get("usage") or {}).get("input_tokens") or 0
                return data["answers"]
            if r.status_code not in RETRY_STATUS or attempt == max_retries:
                raise JevError(f"HTTP {r.status_code}: {r.text[:500]}")
            if r.headers.get("retry-after", "").replace(".", "", 1).isdigit():
                time.sleep(min(float(r.headers["retry-after"]), 60))
                continue
        time.sleep(min(0.5 * 2 ** attempt, 20) * random.uniform(0.75, 1.0))


# ---- llm backend (JUDGE_BACKEND=llm) --------------------------------------------------------------------
_llm_client: httpx.Client | None = None
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LLM_SYSTEM = ("You are a careful, impartial judge. Read the material and answer the question with exactly one "
              "token: the answer word or option letter requested, nothing else.")


def _llm_http() -> httpx.Client:
    global _llm_client
    if _llm_client is None:
        key = env("JUDGE_API_KEY") or env("OPENAI_API_KEY") or ""
        _llm_client = httpx.Client(timeout=120.0, headers={"Authorization": f"Bearer {key}"} if key else {})
    return _llm_client


def _render_state(state) -> str:
    if isinstance(state, dict):
        return "\n\n".join(f"[{k}]\n{v}" for k, v in state.items())
    return str(state)


def _llm_prompt(state_text: str, q: dict) -> tuple[str, dict[str, str]]:
    """User message, and {answer token: answer key} for reading the distribution."""
    head = f"{state_text}\n\n---\nQuestion: {q['instructions']}\n"
    if q["type"] == "noul":
        c = q.get("criteria") or {}
        crit = "".join(f"Answer {w} if: {c[k]}\n" for k, w in (("true", "Yes"), ("false", "No")) if c.get(k))
        return head + crit + "Answer with exactly one word: Yes or No.", {"YES": "yes", "NO": "no"}
    crit = q["criteria"]
    items = [(str(i), d) for i, d in enumerate(crit)] if q["type"] == "score" else list(crit.items())
    marks = list(LETTERS) if len(items) <= len(LETTERS) else [str(i + 1) for i in range(len(items))]
    lines = []
    for m, (k, d) in zip(marks, items):
        lines.append(f"{m}) {d}" if q["type"] == "score" else (f"{m}) {k}: {d}" if d else f"{m}) {k}"))
    word = "letter" if marks[0] == "A" else "number"
    user = head + "Options:\n" + "\n".join(lines) + f"\nAnswer with exactly one option {word}."
    return user, {m: k for m, (k, _) in zip(marks, items)}


def _llm_one(state_text: str, q: dict, max_retries: int) -> dict:
    user, answers = _llm_prompt(state_text, q)
    body = {"model": env("JUDGE_MODEL"), "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20,
            "messages": [{"role": "system", "content": LLM_SYSTEM}, {"role": "user", "content": user}]}
    if not body["model"]:
        raise JevError("JUDGE_BACKEND=llm needs JUDGE_MODEL")
    body.update(json.loads(env("JUDGE_EXTRA_BODY", "{}")))
    url = env("JUDGE_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
    for attempt in range(max_retries + 1):
        try:
            r = _llm_http().post(url, json=body)
        except (httpx.TransportError, httpx.TimeoutException) as e:
            if attempt == max_retries:
                raise JevError(f"network error after {attempt + 1} attempts: {e!r}") from e
        else:
            if r.status_code == 200:
                break
            if r.status_code not in RETRY_STATUS or attempt == max_retries:
                raise JevError(f"HTTP {r.status_code}: {r.text[:500]}")
        time.sleep(min(0.5 * 2 ** attempt, 20) * random.uniform(0.75, 1.0))
    data = r.json()
    usage["calls"] += 1
    usage["input_tokens"] += (data.get("usage") or {}).get("prompt_tokens") or 0
    top = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    mass = dict.fromkeys(answers.values(), 0.0)
    for t in top:
        tok = t["token"].strip().strip(".):").upper()
        if tok in answers:
            mass[answers[tok]] += math.exp(t["logprob"])
    total = sum(mass.values())
    if total == 0:   # none of the valid answers among the top tokens: no information
        mass, total = dict.fromkeys(mass, 1.0), float(len(mass))
    probs = {k: v / total for k, v in mass.items()}
    return {"noul": probs["yes"]} if q["type"] == "noul" else {"probabilities": probs}


def _ask_llm(state, questions: dict, max_retries: int) -> dict:
    """Same contract as ask(): {id: answer}. Questions are independent, so they run in parallel."""
    text = _render_state(state)
    with ThreadPoolExecutor(min(8, len(questions)) or 1) as ex:
        futs = {k: ex.submit(_llm_one, text, q, max_retries) for k, q in questions.items()}
        return {k: f.result() for k, f in futs.items()}


def noul(state, instructions, criteria: dict | None = None) -> float:
    """P(yes). criteria: optional {"true": ..., "false": ...} descriptions."""
    q = {"type": "noul", "instructions": instructions}
    if criteria:
        q["criteria"] = criteria
    return ask(state, {"q": q})["q"]["noul"]


def choice(state, instructions, options: dict) -> dict[str, float]:
    """options: {label: description or None} (<= 255). Returns {label: prob}, sums to 1.

    Jev has a mild first-option position bias; shuffle/permute options if it matters.
    """
    q = {"type": "choice", "instructions": instructions, "criteria": options}
    return ask(state, {"q": q})["q"]["probabilities"]


# ---- token exams (the v3 label exam, modelmapper/exam.py) ------------------------------------------------
# One state per description (or per shown token list), one yes/no question per candidate token. The framing,
# notation and rubric live in the state, which Jev bills once per call, so each question is only the token
# (about 12 tokens instead of about 68). Questions are answered independently, so any number of candidates can
# share one call (the exam merges up to 1,000 per call). Same prompts as the research run (judge_jev_iterate
# "terse" and judge_jev_examples).
WS = "Whitespace is shown as ⏎ newline, ⇥ tab, ␍ carriage return, ␣ leading space."
RATE_FRAME = ("A unit (neuron) inside a language model is described below. For each token you are asked about, judge from the "
              f"description alone whether the unit would switch on strongly for that token. {WS} A token qualifies if the "
              "description names it or clearly covers the kind of token it is (same topic, category, form or script); a token "
              "the description does not cover, or covers only vaguely, does not.")
EXAMPLES_FRAME = ("Below are the tokens (words or word pieces) that most strongly switch on one unit (neuron) inside a language "
                  f"model, strongest first. {WS} For each token you are asked about, judge whether it is likely to switch on the "
                  "same unit strongly too: it qualifies if it is one of these tokens or clearly the same kind of token (same "
                  "topic, category, form, script or function); otherwise it does not.")


def visible(t: str) -> str:
    """A token as shown to the labeller and judge: leading space as ␣, newline ⏎, tab ⇥, carriage return ␍."""
    return (("␣" + t[1:]) if t.startswith(" ") else t).replace("\n", "⏎").replace("\t", "⇥").replace("\r", "␍")


def _token_questions(tokens: list[str]) -> dict:
    return {f"t{i}": {"type": "noul", "instructions": f"Would the unit switch on strongly for the token «{visible(c)}»?"}
            for i, c in enumerate(tokens)}


def rate(description: str, tokens: list[str]) -> list[float]:
    """0-100 per token: how likely a unit with this description switches on strongly for it (100 x P(yes))."""
    ans = ask(f"{RATE_FRAME}\n\nUnit description: {description}", _token_questions(tokens))
    return [100.0 * float(ans[f"t{i}"]["noul"]) for i in range(len(tokens))]


def rate_examples(shown: list[str], tokens: list[str]) -> list[float]:
    """0-100 per token: the description-free version; the state is the unit's shown favourites (already visible())."""
    ans = ask(f"{EXAMPLES_FRAME}\n\nTokens that switch on the unit: " + "  ".join(shown), _token_questions(tokens))
    return [100.0 * float(ans[f"t{i}"]["noul"]) for i in range(len(tokens))]


INTRUDER_INSTRUCTIONS = ("All but one of these numbered text snippets share a common theme or "
                         "feature. Which snippet is the odd one out?")


def intruder(examples: list[str], instructions: str | None = None,
             rotations: int = 1) -> dict[int, float]:
    """P(snippet i is the odd one out), keyed by 0-based index into `examples`.

    Snippets go in the state numbered 1..N; the question is a Choice over "1".."N".
    rotations > 1 re-asks with the list cyclically shifted (one call each) and averages,
    to wash out position bias.
    """
    n = len(examples)
    probs = [0.0] * n
    for r in range(rotations):
        order = [(i + r * n // rotations) % n for i in range(n)]   # order[slot] = original index
        state = {str(s + 1): examples[i] for s, i in enumerate(order)}
        p = choice(state, instructions or INTRUDER_INSTRUCTIONS,
                   {str(s + 1): f"Snippet {s + 1}" for s in range(n)})
        for s, i in enumerate(order):
            probs[i] += p.get(str(s + 1), 0.0) / rotations
    return dict(enumerate(probs))
