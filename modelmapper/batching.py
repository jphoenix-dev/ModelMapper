"""Run work in batches without running out of GPU memory (the research run's gpu_guard; on a CPU, a plain loop).

On Linux an over-full GPU raises an out-of-memory error; on Windows the driver silently spills into system RAM and
the run slows 5-10x with no error. So on CUDA:
  1. a pre-flight probe runs two small batches, splits peak memory into a fixed part and a per-item part, and starts
     at the largest batch that fits the free memory (other processes on the card included);
  2. an out-of-memory error halves the batch and retries the same items;
  3. after each batch, if reserved memory is within HEADROOM of the card's total, the batch is halved for the rest;
  4. a batch that takes > 4x the median time per item is reported (the visible sign of spilling).
Every change is printed and returned: no silent caps.

    log = run_batched(n_items, batch, lambda a, b: work(a, b), label="context 3")   # work(a, b) does items a..b-1
"""
from __future__ import annotations

import time

import numpy as np
import torch

HEADROOM = 0.08          # keep 8% of the card free (about 2.6 GB on 32 GB, 1.9 GB on 24 GB)


def _probe(step, n, B, min_B, label, headroom, log):
    free, _ = torch.cuda.mem_get_info(); avail = (free + torch.cuda.memory_reserved()) * (1 - headroom)
    peaks, a = [], 0
    for sz in (min(n, min_B), min(n, 4 * min_B)):
        b = min(n, a + sz)
        if b <= a:
            break
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        step(a, b); torch.cuda.synchronize(); peaks.append((b - a, torch.cuda.max_memory_allocated())); a = b
    if len(peaks) < 2 or peaks[1][0] == peaks[0][0]:
        return a, B, "probe skipped (too few items)"
    per = max(0.0, (peaks[1][1] - peaks[0][1]) / (peaks[1][0] - peaks[0][0])); fixed = peaks[0][1] - per * peaks[0][0]
    if fixed > avail:
        raise RuntimeError(f"{label}: fixed memory {fixed / 2**30:.1f} GB exceeds the {avail / 2**30:.1f} GB available - "
                           "a smaller batch cannot fix this; reduce what is held per run")
    B2 = max(min_B, min(B, int((avail - fixed) / per) if per > 0 else B))
    msg = f"probe: fixed {fixed / 2**30:.1f} GB + {per / 2**20:.1f} MB per item; {avail / 2**30:.1f} GB available -> batch {B2} (asked {B})"
    log(f"  [batching] {label}: {msg}")
    return a, B2, msg


def run_batched(n, B, step, min_B=16, label="", headroom=HEADROOM, preflight=True, device="cuda", log=print):
    if not str(device).startswith("cuda"):
        for a in range(0, n, B):
            step(a, min(n, a + B))
        return {"final_batch": B, "events": []}
    total = torch.cuda.get_device_properties(torch.device(device)).total_memory
    a, per_item, events = 0, [], []
    if preflight:
        a, B, msg = _probe(step, n, B, min_B, label, headroom, log); events.append(msg)
    while a < n:
        b = min(n, a + B); t0 = time.time()
        try:
            step(a, b); torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if B <= min_B:
                raise
            B = max(min_B, B // 2); events.append(f"OOM at item {a}: batch -> {B}")
            log(f"  [batching] {label}: out of memory, batch halved to {B}")
            continue
        dt = (time.time() - t0) / (b - a); per_item.append(dt)
        if len(per_item) > 5 and dt > 4 * np.median(per_item) and (b - a) >= B // 2:   # a small last batch is always slow per item
            events.append(f"slow batch at item {a}: {dt * (b - a):.1f} s")
            log(f"  [batching] {label}: batch at item {a} took {dt / np.median(per_item):.0f}x the median per item - memory spill?")
        reserved = torch.cuda.memory_reserved()
        if reserved > (1 - headroom) * total and B > min_B:
            B = max(min_B, B // 2); torch.cuda.empty_cache(); events.append(f"near-full ({reserved / 2**30:.1f} GB) at item {b}: batch -> {B}")
            log(f"  [batching] {label}: {reserved / 2**30:.1f} of {total / 2**30:.1f} GB reserved - batch halved to {B}")
        a = b
    return {"final_batch": B, "events": events}
