"""Settle the parcel on its support before introducing the open YAM gripper."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time

import mujoco
import numpy as np

import physical_parcel_yam as backend
from physical_parcel_checks import film_box_geometry, film_payload_geometry


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-options", required=True)
    parser.add_argument("--seconds", type=float, default=.5)
    args = parser.parse_args()
    cfg = backend.Config(**json.loads(args.model_options))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    sources = {"prepare": Path(__file__), "adapter": Path(backend.__file__), **backend.source_files()}
    hashes = {}
    for name, path in sources.items():
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        (args.output / (name + "_source.py")).write_bytes(path.read_bytes())
    p, xml = backend.parcel_world(cfg)
    p.xml = xml
    p.m = p.compile_model(xml)
    p.d = mujoco.MjData(p.m)
    p.film_dofs = p.node_dofs(p.m)
    backend.initialize_state(p, p.m, p.d, cfg)
    trace = []
    started = time.perf_counter()
    for step in range(round(args.seconds / cfg.dt)):
        p.step()
        if step % max(1, round(.01 / cfg.dt)):
            continue
        p.forward()
        payload = film_payload_geometry(p.m, p.d, p.envelope["faces"])
        support = film_box_geometry(p.m, p.d, p.envelope["faces"], p.m.geom("table").id)
        finite = bool(np.isfinite(p.d.qpos).all() and np.isfinite(p.d.qvel).all())
        warnings = int(p.d.warning.number.sum())
        trace.append(dict(time_s=float(p.d.time), payload_geometry=payload, support_geometry=support,
                          com=p.com().tolist(), finite=finite, warnings=warnings))
        if not finite or warnings or payload["outside_corners"] or max(
                payload["geometric_depth_upper_m"], support["geometric_depth_upper_m"]) > .001:
            (args.output / "trace.json").write_text(json.dumps(trace, indent=2))
            raise RuntimeError(f"parcel preparation failed at {p.d.time}")
    p.forward()
    state = args.output / "initial_state.npz"
    backend.save_initial_state(p, p.d, cfg, state)
    p.save(args.output / "model")
    result = dict(passed=True, machine=platform.node(), seconds=args.seconds,
                  wall_s=time.perf_counter()-started, source_sha256=hashes,
                  state=str(state), final_com=p.com().tolist(),
                  peak_payload_depth_m=max(row["payload_geometry"]["geometric_depth_upper_m"] for row in trace),
                  peak_support_depth_m=max(row["support_geometry"]["geometric_depth_upper_m"] for row in trace))
    (args.output / "trace.json").write_text(json.dumps(trace, indent=2))
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
