"""Exercise the local UI API, real geometry and optional locked GPU grasp job."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8766")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run", action="store_true", help="Submit one full hard/box/0-degree GPU task")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def api(path, body=None, expected=200, origin=None):
        headers = {"Content-Type": "application/json"}
        if origin: headers["Origin"] = origin
        request = urllib.request.Request(args.url + path,
            data=json.dumps(body).encode() if body is not None else None, headers=headers)
        try: response = opener.open(request, timeout=15)
        except urllib.error.HTTPError as error: response = error
        with response:
            assert response.status == expected, (path, response.status, response.read())
            return json.load(response)

    def image(job):
        with opener.open(args.url + job["image"], timeout=15) as response: payload = response.read()
        assert payload[:8] == b"\x89PNG\r\n\x1a\n"
        assert struct.unpack(">II", payload[16:24]) == (1280, 480)
        return dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())

    def wait(job, deadline=900):
        started, last_phase = time.monotonic(), None
        observed = []
        while time.monotonic() - started < deadline:
            job = api("/api/jobs/" + job["id"])
            phase = job.get("live", {}).get("sample", {}).get("phase")
            if phase and phase != last_phase:
                print(json.dumps(dict(id=job["id"], phase=phase, status=job["status"])), flush=True)
                last_phase = phase
            if job.get("image") and len(observed) < 4:
                digest = image(job)
                if not observed or digest != observed[-1]: observed.append(digest)
            if job["status"] in ("done", "error"):
                assert job["status"] == "done", job
                return job, observed
            time.sleep(1)
        raise TimeoutError(f"UI job still running after {deadline}s: {job['id']}")

    fixture = dict(meta=api("/api/meta"), previews={})
    for shape in ("box", "rounded", "pillow", "tapered"):
        job = api("/api/jobs", dict(mode="preview", settings=dict(shape=shape)), expected=202)
        result, pictures = wait(job, 60)
        assert result["done"]["outcome"] == "geometry_preview"
        assert result["live"]["sample"]["t"] == 0
        assert "llvmpipe" in result["live"]["renderer"].lower()
        assert pictures
        fixture["previews"][shape] = result
    dimensions = [r["done"]["dimensions"] for r in fixture["previews"].values()]
    assert len({d["topology_sha256"] for d in dimensions}) == 1
    assert len({round(d["mass_kg"], 6) for d in dimensions}) == 4
    rejected = []
    for settings in ({"young": -1}, {"roundness": 2}, {"shape": "unknown"}, {"density": True},
                     {"size": [.2, 0, .05]}, {"yield_strain": float("nan")}, {"solver": "CG"}):
        result = api("/api/jobs", dict(mode="preview", settings=settings), expected=400)
        rejected.append(result)
    api("/api/jobs", dict(mode="preview", settings={}), expected=400, origin="https://other.invalid")
    fixture["rejected"] = rejected
    (args.output / "ui-fixtures.json").write_text(json.dumps(fixture, indent=2, ensure_ascii=False))
    checks = dict(four_real_shapes=True, fixed_topology=True, cpu_geometry_only=True,
                  png_frames=True, rejected_inputs=len(rejected), same_origin=True, browser_test=False)
    if args.run:
        job = api("/api/jobs", dict(mode="run", settings=dict(material="hard", shape="box", yaw=0)), expected=202)
        api("/api/jobs", dict(mode="preview", settings={}), expected=409)
        result, pictures = wait(job)
        assert result["done"]["backend"] == "continuum.grasp_adapter", result
        assert result["done"]["dimensions"]["software"]["dynamics"] == "GPU"
        assert result["done"]["dimensions"]["software"]["solver"] == "Newton"
        assert result["done"]["config"]["pinch_coeff"] == 0
        assert len(pictures) >= 2 and result["video"]
        with opener.open(args.url + result["video"], timeout=30) as response: video = response.read()
        assert video[4:8] == b"ftyp" and len(video) > 10000
        (args.output / "gpu-job.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
        checks["http_to_gpu_to_video"] = True
        checks["gpu_job_id"] = result["id"]
        checks["physical_outcome"] = result["done"]["outcome"]
        checks["different_live_frames"] = pictures
    (args.output / "checks.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks), flush=True)


if __name__ == "__main__": main()
