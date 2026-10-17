"""Minimal RunPod REST client (https://rest.runpod.io/v1, OpenAPI spec checked 2026-10-17).

  create:    POST   /v1/pods            (cloudType COMMUNITY, one gpuTypeId, volumeInGb 0; on-demand by default:
                                         on 2026-10-17 interruptible creates failed with "Spot pods are no longer offered")
  list:      GET    /v1/pods
  terminate: DELETE /v1/pods/{id}       (204; a pod that is already gone returns 404 and counts as terminated)
Notes from the spec / docs: the REST API has no bid field for spot pods (only GraphQL has bidPerGpu); volumeInGb
defaults to 20 and containerDiskInGb to 50, so both are always passed; env is a JSON object; billing is per second.
The key comes from $RUNPOD_API_KEY (env or .env), never from the image.
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import env  # noqa: E402

BASE = "https://rest.runpod.io/v1"


def _req(method, path, body=None, key=None, ok_404=False):
    key = key or env("RUNPOD_API_KEY")
    if not key:
        raise SystemExit("RUNPOD_API_KEY is not set")
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE + path, data=data, method=method,
                               headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                        "User-Agent": "model-mapper-launcher/1.0"})   # Cloudflare blocks urllib's default (1010)
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            txt = resp.read().decode()
            return json.loads(txt) if txt.strip() else None
    except urllib.error.HTTPError as e:
        if ok_404 and e.code == 404:
            return None
        raise RuntimeError(f"RunPod {method} {path} -> {e.code}: {e.read().decode()[:500]}") from None


def list_pods(name_prefix: str | None = None, key=None) -> list[dict]:
    pods = _req("GET", "/pods", key=key) or []
    return [p for p in pods if not name_prefix or (p.get("name") or "").startswith(name_prefix)]


class NoCapacity(RuntimeError):
    """RunPod had no machine matching the request (GPU type, cloud, CUDA versions) right now."""


def create_pod(name, image, gpu_type, env_vars: dict, container_disk_gb=40, registry_auth_id=None, start_cmd=None,
               interruptible=False, cuda_versions=("12.8", "12.9", "13.0"), key=None) -> dict:
    types = [gpu_type] if isinstance(gpu_type, str) else list(gpu_type)
    body = {"name": name, "imageName": image, "gpuTypeIds": types, "gpuCount": 1,
            "cloudType": "COMMUNITY", "interruptible": interruptible,
            "containerDiskInGb": container_disk_gb, "volumeInGb": 0, "env": env_vars,
            "ports": [], "supportPublicIp": False,
            "allowedCudaVersions": list(cuda_versions)}      # the image is CUDA 12.8: hosts need a new enough driver
    if len(types) > 1:
        body["gpuTypePriority"] = "custom"      # try the types in the order given (cheapest per token first)
    if start_cmd:
        body["dockerStartCmd"] = start_cmd
    if registry_auth_id:
        body["containerRegistryAuthId"] = registry_auth_id
    try:
        return _req("POST", "/pods", body, key=key)
    except RuntimeError as e:
        msg = str(e).lower()       # RunPod words this several ways ("no instances currently available",
        if "instances currently available" in msg or "no longer any instances available" in msg                 or "no instances available" in msg:   # "no longer any instances available with the requested ...")
            raise NoCapacity(str(e)) from None
        raise


def balance(key=None) -> dict:
    """Account balance and current spend rate (GraphQL; the REST API has no balance endpoint). Needs the key's GraphQL
    permission. Returns {"clientBalance": USD, "currentSpendPerHr": USD/h, "spendLimit": USD/h}."""
    key = key or env("RUNPOD_API_KEY")
    r = urllib.request.Request("https://api.runpod.io/graphql",
                               data=json.dumps({"query": "{ myself { clientBalance currentSpendPerHr spendLimit } }"}).encode(),
                               headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                        "User-Agent": "model-mapper-launcher/1.0"})
    with urllib.request.urlopen(r, timeout=60) as resp:
        return json.loads(resp.read().decode())["data"]["myself"]


def terminate(pod_id: str, key=None) -> None:
    _req("DELETE", f"/pods/{pod_id}", key=key, ok_404=True)
