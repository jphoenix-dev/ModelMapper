"""Minimal RunPod REST client (https://rest.runpod.io/v1, OpenAPI spec checked 2026-10-17).

  create:    POST   /v1/pods            (cloudType COMMUNITY, interruptible true, one gpuTypeId, volumeInGb 0)
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


def create_pod(name, image, gpu_type, env_vars: dict, container_disk_gb=40, registry_auth_id=None, start_cmd=None,
               key=None) -> dict:
    body = {"name": name, "imageName": image, "gpuTypeIds": [gpu_type], "gpuCount": 1,
            "cloudType": "COMMUNITY", "interruptible": True,
            "containerDiskInGb": container_disk_gb, "volumeInGb": 0, "env": env_vars,
            "ports": [], "supportPublicIp": False,
            "allowedCudaVersions": ["12.8", "12.9", "13.0"]}      # the image is CUDA 12.8: hosts need a new enough driver
    if start_cmd:
        body["dockerStartCmd"] = start_cmd
    if registry_auth_id:
        body["containerRegistryAuthId"] = registry_auth_id
    return _req("POST", "/pods", body, key=key)


def terminate(pod_id: str, key=None) -> None:
    _req("DELETE", f"/pods/{pod_id}", key=key, ok_404=True)
