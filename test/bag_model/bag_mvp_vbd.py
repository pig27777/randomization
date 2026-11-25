"""One VBD fallback for the same unsupported patch of the rigid payload."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from bag_mvp import PEAK_DEPTH, geometry, make_trial, metrics, save_json
from pe_physics_vbd import VBDBagBackend

DT = .0002


def run(cpu, output):
    output.mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    trial = make_trial("through", DT, "exposed-underface")
    edge = .015
    half_x = (.40 + edge) / 2
    backend = VBDBagBackend(trial, 0, table_box=((edge - half_x, 0., -.009), (half_x, .18, .009)))
    with np.load(cpu / "trace.npz") as trace:
        commands = trace["pad_commands"][::4]
    rows = []

    def sample(t):
        film, velocity = backend.get_bag_state()
        payload, _ = backend.get_payload_state()
        tools = backend.get_fixture_state()
        reaction = backend.get_contact_wrench()
        state = backend.get_physics_diagnostics(film)
        floor, outside, depth, escaped, corners = geometry(backend, film, payload)
        warning = state["contact_overflow"] or not all(np.isfinite(x).all() for x in (film, velocity, payload))
        assert np.isfinite(film).all(), f"Nonfinite VBD film at {t:.6f} s"
        pairs = reaction["pairs"]
        jaw = np.maximum(0., [-reaction["wrench"][2, 2], reaction["wrench"][3, 2]])
        rows.append(dict(time=t, q=film, payload=payload, fixtures=tools, wrench=reaction["wrench"],
                         pairs=pairs, pair_depth=reaction["pair_depth"], floor=floor, outside=outside,
                         core_penetration=depth, escaped_center=escaped, outside_corners=corners,
                         table_penetration=0., shelf_intrusion=0., solved_warning=int(warning), jaw_normal=jaw))

    sample(0.)
    reason = None
    for i, step in enumerate(commands, 1):
        tools = {name: (item[:3], item[3:6], item[6:])
                 for name, item in zip(("left_pad", "right_pad"), step) if np.isfinite(item[0])}
        backend.step(dict(tools=tools, clamps={}, load=np.zeros(3)))
        if i % 25 == 0 or i == len(commands):
            sample(i * DT)
            row = rows[-1]
            if row["escaped_center"] or max(row["outside"], row["core_penetration"]) > PEAK_DEPTH or row["solved_warning"]:
                reason = "containment or solver limit"
                break
    raw = {k: np.array([row[k] for row in rows]) for k in rows[0]}
    if reason:
        result = dict(status="FAIL", reason=reason,
                      max_corner_escape_mm=float(raw["outside"].max() * 1000),
                      max_film_inside_mm=float(raw["core_penetration"].max() * 1000),
                      peak_jaw_N=float(raw["jaw_normal"].max()))
    else:
        result = metrics("through", raw, "exposed-underface", trial.size)
    result.update(device=backend.device.name, dt_s=DT, completed_s=float(raw["time"][-1]),
                  runtime_s=perf_counter() - start)
    raw.update({k: backend.envelope[k] for k in ("reference", "faces", "edges")})
    np.savez_compressed(output / "trace.npz", **raw)
    save_json(output / "config.json", backend.metadata)
    save_json(output / "summary.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("cpu", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.cpu, args.output)
