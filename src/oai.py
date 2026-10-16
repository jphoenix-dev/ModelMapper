"""Thin OpenAI client for cheap, fast text generation (neuron explanations).

    oai.chat("Explain ...", system="You are ...")  # -> str

The API key is read lazily from $OPENAI_API_KEY, else from the repo-root .env, else from OAI.env
at the repo root.
Never print it.
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import httpx

from config import env

BASE = "https://api.openai.com/v1"
MODEL = env("OAI_MODEL", "gpt-6-luna")
KEY_FILE = Path(__file__).resolve().parent.parent / "OAI.env"
RETRY_STATUS = {408, 429, 500, 502, 503, 504}

usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}  # running tally for this process
_client: httpx.Client | None = None


class OAIError(RuntimeError):
    pass


def _load_key() -> str:
    """Accepts `OPENAI_API_KEY=...`, any `NAME=...`, or a bare key on one line."""
    if env("OPENAI_API_KEY"):
        return env("OPENAI_API_KEY")
    lines = [l.strip() for l in KEY_FILE.read_text(encoding="utf-8-sig").splitlines()
             if l.strip() and not l.strip().startswith("#")]
    pick = next((l for l in lines if l.removeprefix("export ").startswith("OPENAI_API_KEY=")),
                lines[0] if lines else "")
    key = pick.split("=", 1)[1] if "=" in pick else pick
    key = key.strip().strip("'\"").strip()
    if not key:
        raise OAIError(f"no API key found in $OPENAI_API_KEY, .env or {KEY_FILE}")
    os.environ["OPENAI_API_KEY"] = key
    return key


def _http() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(timeout=120.0, headers={"Authorization": f"Bearer {_load_key()}"})
    return _client


def _wait_hint(r):
    """Seconds the server asks us to wait (Retry-After, or the rate-limit reset headers like '182ms' / '1.5s')."""
    for h in ("retry-after", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
        v = r.headers.get(h, "")
        try:
            if v.endswith("ms"):
                return float(v[:-2]) / 1000
            if v.endswith("s"):
                return float(v[:-1])
            if v:
                return float(v)
        except ValueError:
            pass
    return None


def _request(method, path, body=None, max_retries=8):
    for attempt in range(max_retries + 1):
        hint = None
        try:
            r = _http().request(method, BASE + path, json=body)
        except (httpx.TransportError, httpx.TimeoutException) as e:
            if attempt == max_retries:
                raise OAIError(f"network error after {attempt + 1} attempts: {e!r}") from e
        else:
            if r.status_code == 200:
                return r.json()
            if r.status_code not in RETRY_STATUS or attempt == max_retries:
                raise OAIError(f"HTTP {r.status_code}: {r.text[:500]}")
            hint = _wait_hint(r)
        backoff = min(0.5 * 2 ** attempt, 30) * random.uniform(0.75, 1.0)
        time.sleep(min(max(backoff, hint or 0), 60))


def models() -> list[str]:
    return sorted(m["id"] for m in _request("GET", "/models")["data"])


def chat(prompt: str, system: str | None = None, model: str = MODEL, json_mode: bool = False, **kw) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    body = {"model": model, "messages": msgs, **kw}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    data = _request("POST", "/chat/completions", body)
    u = data.get("usage") or {}
    usage["calls"] += 1
    usage["input_tokens"] += u.get("prompt_tokens", 0)
    usage["output_tokens"] += u.get("completion_tokens", 0)
    return data["choices"][0]["message"]["content"]


def chat_json(prompt: str, system: str | None = None, **kw) -> dict:
    return json.loads(chat(prompt, system, json_mode=True, **kw))
