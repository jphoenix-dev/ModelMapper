"""Thin client for TypeSafe's Jev ("System One") model: calibrated probabilities, no text.

One call = one `state` + any number of typed questions, all answered independently and in
parallel against that state. Billing is per input token only (~$0.042 / M); output is free.
Limits (jev-1.13): state + all questions <= 64k tokens, state + any one question <= 32k.

    jev.noul("The Golden Gate Bridge is in SF.", "Does the text mention a bridge?")  # -> 0.99
    jev.intruder(["bridge ...", "bridge ...", "bread ..."])                          # -> {2: 0.97, ...}

The API key is read lazily from $JEV_API_KEY (or $TYPESAFE_API_KEY), else from the repo-root .env,
else from jevkey.env at the repo root.
"""
from __future__ import annotations

import os
import random
import time
from pathlib import Path

import httpx

from config import env

URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
USD_PER_MTOK = 0.042
KEY_FILE = Path(__file__).resolve().parent.parent / "jevkey.env"
RETRY_STATUS = {408, 429, 500, 502, 503, 504, 529}

usage = {"calls": 0, "input_tokens": 0}   # running tally for this process
_client: httpx.Client | None = None


class JevError(RuntimeError):
    pass


def cost() -> float:
    """Estimated $ spent so far in this process."""
    return usage["input_tokens"] * USD_PER_MTOK / 1e6


def _load_key() -> str:
    """Env var / .env first; else jevkey.env: `TYPESAFE_API_KEY=...`, any `NAME=...`, or a bare key on one line."""
    for var in ("JEV_API_KEY", "TYPESAFE_API_KEY"):
        if env(var):
            return env(var)
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
