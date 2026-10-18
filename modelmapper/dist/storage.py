"""Where a distributed run keeps everything: a local folder (testing) or an S3-compatible bucket (R2 etc.).

One run = one store root. Layout under the root:
    config.json              model repo + pinned revision, dtype, task parameters (written by make_run.py)
    manifest.json            every task id, with its dependencies
    corpus/shard_00000.npz   token ids of the corpus shards (tokens, lengths, doc0)
    results/<task>/...       one folder per task (task id with ':' -> '/'), finished by a _DONE marker
    claims/<task>            soft claims (worker id + time); not locks, only to steer workers apart
    logs/<worker>/...        boot timings, per-task timings, heartbeats
    merged/...               merged per-neuron results

URLs: a plain path or file:///C:/x -> LocalStore; s3://bucket/prefix -> S3Store. S3 settings come from env / .env:
    S3_ENDPOINT_URL   e.g. https://<account>.r2.cloudflarestorage.com (omit for AWS)
    S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY   (scoped to this one bucket)
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from modelmapper.core.config import env  # noqa: E402


class Store:
    """Counts requests by R2 billing class (A: put, list page; B: get, head; delete is free) for quota.py, and the
    time and bytes of every get / put (io) so tasks can split their time into transfer and compute."""
    ops = None
    io = None

    def _io(self, kind, seconds, nbytes):
        if self.io is None:
            self.io = {"get_s": 0.0, "get_bytes": 0, "put_s": 0.0, "put_bytes": 0}
        self.io[f"{kind}_s"] += seconds; self.io[f"{kind}_bytes"] += nbytes

    def _count(self, cls, n=1):
        if self.ops is None:
            self.ops = {"A": 0, "B": 0}
        self.ops[cls] += n

    def get(self, key: str) -> bytes | None: raise NotImplementedError
    def put(self, key: str, data: bytes) -> None: raise NotImplementedError
    def list(self, prefix: str = "") -> list[str]: raise NotImplementedError
    def list_sizes(self, prefix: str = "") -> dict[str, int]: raise NotImplementedError
    def delete(self, key: str) -> None: raise NotImplementedError

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def put_file(self, key: str, path) -> None:
        self.put(key, Path(path).read_bytes())

    def get_file(self, key: str, path) -> bool:
        b = self.get(key)
        if b is None:
            return False
        Path(path).parent.mkdir(parents=True, exist_ok=True); Path(path).write_bytes(b)
        return True

    def get_json(self, key: str):
        b = self.get(key)
        return None if b is None else json.loads(b.decode("utf-8"))

    def put_json(self, key: str, obj) -> None:
        self.put(key, json.dumps(obj, indent=1, ensure_ascii=False).encode("utf-8"))

    def get_npz(self, key: str) -> dict | None:
        b = self.get(key)
        if b is None:
            return None
        with np.load(io.BytesIO(b), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def put_npz(self, key: str, compress: bool = True, **arrays) -> None:
        buf = io.BytesIO()
        (np.savez_compressed if compress else np.savez)(buf, **arrays)
        self.put(key, buf.getvalue())


class LocalStore(Store):
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _p(self, key):
        return self.root / key

    def get(self, key):
        self._count("B")
        t = time.perf_counter(); p = self._p(key)
        b = p.read_bytes() if p.is_file() else None
        self._io("get", time.perf_counter() - t, len(b or b""))
        return b

    def exists(self, key):
        self._count("B")
        return self._p(key).is_file()

    def put(self, key, data):
        self._count("A")
        t = time.perf_counter(); p = self._p(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".tmp_")   # atomic: readers never see half a file
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, p)
        self._io("put", time.perf_counter() - t, len(data))

    def list(self, prefix=""):
        return sorted(self.list_sizes(prefix))

    def list_sizes(self, prefix=""):
        base = self._p(prefix)
        top = base if base.is_dir() else base.parent
        out = {}
        if top.exists():
            for p in top.rglob("*"):
                if p.is_file() and not p.name.startswith(".tmp_"):
                    k = p.relative_to(self.root).as_posix()
                    if k.startswith(prefix):
                        out[k] = p.stat().st_size
        self._count("A", max(1, -(-len(out) // 1000)))
        return out

    def delete(self, key):
        self._p(key).unlink(missing_ok=True)

    def bucket_root(self):
        return self

    def put_file(self, key, path):
        import shutil
        self._count("A"); t = time.perf_counter(); p = self._p(key); p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(".tmp_" + p.name); shutil.copyfile(path, tmp); os.replace(tmp, p)
        self._io("put", time.perf_counter() - t, p.stat().st_size)

    def get_file(self, key, path):
        import shutil
        self._count("B"); p = self._p(key)
        if not p.is_file():
            return False
        t = time.perf_counter(); Path(path).parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, path)
        self._io("get", time.perf_counter() - t, p.stat().st_size)
        return True

    def __repr__(self):
        return f"LocalStore({self.root})"


class S3Store(Store):
    def __init__(self, bucket, prefix=""):
        import boto3
        from botocore.config import Config
        self.bucket, self.prefix = bucket, prefix.strip("/") + "/" if prefix.strip("/") else ""
        self.s3 = boto3.client(
            "s3", endpoint_url=env("S3_ENDPOINT_URL"), region_name=env("S3_REGION", "auto"),
            aws_access_key_id=env("S3_ACCESS_KEY_ID"), aws_secret_access_key=env("S3_SECRET_ACCESS_KEY"),
            config=Config(retries={"max_attempts": 10, "mode": "adaptive"}))

    def get(self, key):
        self._count("B")
        t = time.perf_counter()
        try:
            b = self.s3.get_object(Bucket=self.bucket, Key=self.prefix + key)["Body"].read()
        except self.s3.exceptions.NoSuchKey:
            b = None
        self._io("get", time.perf_counter() - t, len(b or b""))
        return b

    def exists(self, key):
        from botocore.exceptions import ClientError
        self._count("B")
        try:
            self.s3.head_object(Bucket=self.bucket, Key=self.prefix + key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def put(self, key, data):
        self._count("A")
        t = time.perf_counter()
        self.s3.put_object(Bucket=self.bucket, Key=self.prefix + key, Body=data)
        self._io("put", time.perf_counter() - t, len(data))

    def list(self, prefix=""):
        return sorted(self.list_sizes(prefix))

    def list_sizes(self, prefix=""):
        out, pag = {}, self.s3.get_paginator("list_objects_v2")
        for page in pag.paginate(Bucket=self.bucket, Prefix=self.prefix + prefix):
            self._count("A")
            out.update({o["Key"][len(self.prefix):]: o["Size"] for o in page.get("Contents", [])})
        return out

    def delete(self, key):
        self.s3.delete_object(Bucket=self.bucket, Key=self.prefix + key)

    def put_file(self, key, path):
        """Multipart upload (64 MB parts): single PUTs cap at 5 GB and would hold the whole file in memory."""
        from boto3.s3.transfer import TransferConfig
        size = Path(path).stat().st_size
        cfg = TransferConfig(multipart_threshold=64 * 2**20, multipart_chunksize=64 * 2**20, max_concurrency=8)
        t = time.perf_counter()
        self.s3.upload_file(str(path), self.bucket, self.prefix + key, Config=cfg)
        self._count("A", 2 + size // (64 * 2**20)); self._io("put", time.perf_counter() - t, size)

    def get_file(self, key, path):
        from boto3.s3.transfer import TransferConfig
        from botocore.exceptions import ClientError
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        t = time.perf_counter()
        try:
            self.s3.download_file(self.bucket, self.prefix + key, str(path),
                                  Config=TransferConfig(multipart_chunksize=64 * 2**20, max_concurrency=8))
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise
        size = Path(path).stat().st_size
        self._count("B", 1 + size // (64 * 2**20)); self._io("get", time.perf_counter() - t, size)
        return True

    def bucket_root(self):
        """The whole bucket (no run prefix), sharing this client and its counters: the free tier is per account."""
        r = object.__new__(S3Store)
        r.bucket, r.prefix, r.s3 = self.bucket, "", self.s3
        r.ops, r.io = self.ops, self.io
        return r

    def __repr__(self):
        return f"S3Store(s3://{self.bucket}/{self.prefix})"


def open_store(url: str | None = None) -> Store:
    """url (or $DIST_STORE): s3://bucket/prefix, file:///path, or a plain local path."""
    url = url or env("DIST_STORE")
    if not url:
        raise SystemExit("no store given: pass --store or set DIST_STORE (s3://bucket/prefix or a local folder)")
    if url.startswith("s3://"):
        bucket, _, prefix = url[5:].partition("/")
        return S3Store(bucket, prefix)
    if url.startswith("file://"):
        url = url[7:].lstrip("/") if os.name == "nt" else url[7:]
    return LocalStore(url)
