"""Download a large locked NVIDIA wheel with resumable HTTP ranges and SHA256 verification."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import http.client
import json
from pathlib import Path
import shutil
import time
import tomllib
import urllib.request
import urllib.error

ENV = Path("/home/ubuntu/Dev/Envs/isaaclab")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", choices=("isaacsim-extscache-kit", "isaacsim-extscache-kit-sdk", "open3d",
                                           "numpy", "onnxruntime-gpu"))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    lock = tomllib.loads((ENV / "src/IsaacLab/uv.lock").read_text())
    if args.package in ("numpy", "onnxruntime-gpu"):
        versions = dict(line.split("==") for line in
                        (Path(__file__).parent / "requirements-compat.txt").read_text().splitlines()
                        if line and not line.startswith("#"))
        metadata_url = f"https://pypi.org/pypi/{args.package}/{versions[args.package]}/json"
        with urllib.request.urlopen(metadata_url, timeout=60) as response:
            metadata = json.load(response)
        artifact = next(w for w in metadata["urls"] if "cp312" in w["filename"]
                        and "x86_64" in w["filename"] and "manylinux" in w["filename"])
        wheel = {"url": artifact["url"], "hash": "sha256:" + artifact["digests"]["sha256"]}
    else:
        package = next(p for p in lock["package"] if p["name"] == args.package)
        wheel = next(w for w in package["wheels"] if "x86_64" in w["url"] and "cp312" in w["url"])
    url, expected = wheel["url"], wheel["hash"].removeprefix("sha256:")
    target = ENV / "cache/wheels" / url.rsplit("/", 1)[1]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if digest(target) != expected:
            raise RuntimeError(f"existing wheel has incorrect SHA256: {target}")
        print(f"Verified existing {target}", flush=True)
        return
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as response:
        size = int(response.headers["Content-Length"])
    parts = target.parent / (target.name + ".parts")
    parts.mkdir(exist_ok=True)
    metadata = {"url": url, "bytes": size, "sha256": expected, "chunk_bytes": 16 * 1024**2}
    manifest = parts / "source.json"
    if manifest.exists() and json.loads(manifest.read_text()) != metadata:
        raise RuntimeError(f"partial download belongs to different input: {manifest}")
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    chunk = metadata["chunk_bytes"]
    count = (size + chunk - 1) // chunk

    def fetch(index):
        start, end = index * chunk, min((index + 1) * chunk, size) - 1
        path = parts / f"{index:05}.part"
        length = end - start + 1
        if path.exists() and path.stat().st_size == length:
            return index, length
        req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        temporary = path.with_suffix(".tmp")
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=180) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise RuntimeError(f"invalid range response for chunk {index}")
                    with temporary.open("wb") as output:
                        shutil.copyfileobj(response, output, length=1024**2)
                if temporary.stat().st_size != length:
                    raise http.client.IncompleteRead(b"", length - temporary.stat().st_size)
                break
            except (urllib.error.URLError, http.client.HTTPException, TimeoutError, ConnectionError) as exc:
                print(json.dumps({"chunk": index, "attempt": attempt + 1, "error": str(exc)}), flush=True)
                if attempt == 2:
                    raise RuntimeError(f"chunk {index} failed after 3 attempts") from exc
                time.sleep(attempt + 1)
        temporary.replace(path)
        return index, length

    started = time.monotonic()
    complete = 0
    with ThreadPoolExecutor(args.workers) as pool:
        for future in as_completed([pool.submit(fetch, i) for i in range(count)]):
            _, length = future.result()
            complete += length
            print(json.dumps({"package": args.package, "complete_bytes": complete, "total_bytes": size,
                              "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
    combined = target.with_suffix(".assembling")
    with combined.open("wb") as output:
        for i in range(count):
            with (parts / f"{i:05}.part").open("rb") as source:
                shutil.copyfileobj(source, output, length=8 * 1024**2)
    actual = digest(combined)
    if actual != expected:
        raise RuntimeError(f"wheel SHA256 mismatch: expected {expected}, got {actual}; partials retained")
    combined.replace(target)
    receipt = metadata | {"verified": True, "elapsed_s": time.monotonic() - started, "path": str(target)}
    target.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    # Remove only the chunk files listed by this verified download's manifest.
    for i in range(count):
        (parts / f"{i:05}.part").unlink()
    manifest.unlink()
    parts.rmdir()
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
