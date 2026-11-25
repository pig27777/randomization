"""Run the archived PE fixture with an updated acceptance and maximum-side feedback."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
import warp as wp
from bag_sim.wrap import triangle_box_intersections

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "gpu-rl-gates-r1"))
from flex_fixture import Batch, TunedFlex, TUNING, drive, hold_window, longest, make_trial, snapshot

CPU_REFERENCE = Path("test/runs/2026-11-25/bag/pe-mvp-r1")


@wp.kernel
def measure(ncon: wp.array(dtype=int), geom: wp.array(dtype=wp.vec2i), address: wp.array2d(dtype=int),
            efc: wp.array2d(dtype=float), left: int, right: int, core: int, state: wp.array(dtype=float)):
    forces = wp.vec2(0., 0.)
    for i in range(wp.min(ncon[0], geom.shape[0])):
        index = address[i, 0]
        if index >= 0 and index < efc.shape[1]:
            pair = geom[i]
            n = wp.max(efc[0, index], 0.)
            if pair[0] == left or pair[1] == left:
                forces[0] += n
            if pair[0] == right or pair[1] == right:
                forces[1] += n
            fixture = pair[0] == left or pair[1] == left or pair[0] == right or pair[1] == right
            if fixture and n > 1.e-6 and (pair[0] == core or pair[1] == core):
                state[2] += 1.
    state[0] = wp.max(state[0], forces[0])
    state[1] = wp.max(state[1], forces[1])


def run(tuning_name, output, *, young_scale=1., case="through", contact_profile=None):
    output.mkdir()
    tuning = TUNING[tuning_name]
    dt = tuning[0]
    source = CPU_REFERENCE / ("through-r3" if case == "through" else "film-r3")
    config = json.loads((source / "config.json").read_text())
    variant = config["variant"]
    trial = make_trial(case, dt, variant)

    class Fixture(TunedFlex):
        def fixture_xml(self, xml):
            root = ET.fromstring(super().fixture_xml(xml))
            elasticity = root.find(".//flexcomp/elasticity")
            if young_scale != 1.:
                elasticity.set("young", str(float(elasticity.get("young")) * young_scale))
            return ET.tostring(root, encoding="unicode")

    backend = Fixture(trial, config["mesh_pads"], config["support_edge"], variant, tuning)
    stride = round(dt / config["dt_s"])
    with np.load(source / "trace.npz") as trace:
        commands = trace["pad_commands"][::stride].astype(np.float32)
    feedback = case == "through"
    start = perf_counter()
    hold, release = hold_window(variant)
    faces = backend.envelope["faces"]
    edges = backend.envelope["edges"]
    ref = backend.envelope["reference"]
    rest = np.linalg.norm(ref[edges[:, 0]] - ref[edges[:, 1]], axis=1)

    def audit(row, native):
        pose = row["payload"]
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, pose[3:])
        row["inset_triangles"] = int(triangle_box_intersections(
            row["q"], faces, np.asarray(trial.size) / 2 - .002, pose[:3], rotation.reshape(3, 3)).sum())
        row["max_edge_ratio"] = float(np.max(np.linalg.norm(
            row["q"][edges[:, 0]] - row["q"][edges[:, 1]], axis=1) / rest))
        row["peak_substep_N"] = native[:2]
        row["direct_force_contacts"] = int(native[2])
        return row

    with wp.ScopedDevice("cuda:0"):
        batch = Batch(1, parcel=backend, randomize=False, contact_capacity=1024, constraint_capacity=4096)
        d, m = batch.d, backend.m
        q = wp.array(backend.tool_q, dtype=int)
        v = wp.array(backend.tool_v, dtype=int)
        command = wp.array(commands, dtype=float)
        tick = wp.zeros(1, dtype=int)
        jaw = wp.array(np.array([[.046, 0.]], dtype=np.float32), dtype=float)
        state = wp.zeros(3, dtype=float)
        ids = wp.array(np.arange(d.naconmax), dtype=int)
        local = wp.zeros(d.naconmax, dtype=wp.spatial_vector)
        world = wp.zeros_like(local)
        geoms = [m.geom(name).id for name in ("left_pad", "right_pad", "payload_geom_0")]
        if contact_profile is not None:
            from policy_v3.yam_gpu import install_pad_film_contact

            install_pad_film_contact(batch.m, wp.array(geoms[:2], dtype=int), contact_profile)

        def step():
            wp.launch(drive, 1, inputs=[tick, command, q, v, False, 0., 0., feedback, dt,
                      release, trial.size[2] - .00075, jaw, d.qpos, d.qvel, d.qfrc_applied])
            batch.step()
            wp.launch(measure, 1, inputs=[d.nacon, d.contact.geom, d.contact.efc_address,
                      d.efc.force, *geoms, state])

        batch.reset()
        step()
        wp.synchronize_device()
        batch.reset()
        tick.zero_()
        state.zero_()
        block = 20
        with wp.ScopedCapture() as cap:
            for _ in range(block):
                step()
        wp.synchronize_device()
        data = [audit(snapshot(batch, 0., ids, local, world)[0], np.zeros(3))]
        peak = np.zeros(3)
        initial = data[0]
        stop = "initial_film_core_geometry" if (initial["inset_triangles"] or initial["outside_corners"]) else None
        for i in range(0, len(commands), block):
            if stop:
                break
            count = min(block, len(commands) - i)
            state.zero_()
            if count == block:
                wp.capture_launch(cap.graph)
            else:
                for _ in range(count):
                    step()
            t = (i + count) * dt
            native = state.numpy()
            peak[:2] = np.maximum(peak[:2], native[:2])
            peak[2] += native[2]
            row = audit(snapshot(batch, t, ids, local, world)[0], native)
            data.append(row)
            if native[2]:
                stop = "robot_payload_direct_force"
            elif row["solved_warning"]:
                stop = "nonfinite_or_capacity"
            elif row["inset_triangles"] or row["outside_corners"] or row["escaped_center"]:
                stop = "film_core_geometry"
            elif max(native[:2].max(), row["jaw_normal"].max()) > 25.:
                stop = "grasp_peak_force"
            elif row["max_edge_ratio"] > 1.15:
                stop = "film_edge_ratio_1p15"
            if stop:
                break
            if (i + count) % (100 * block) == 0:
                print(tuning_name, case, f"{t:.3f}s simulated, {peak[:2].max():.3f}N native peak", flush=True)
            if feedback and .47 <= t < release:
                measured = row["jaw_normal"].max()
                ctrl = jaw.numpy()
                ctrl[0, 1] = .15 if measured > 18. else np.clip(.006 * (measured - 8.), -.070, .10)
                jaw.assign(ctrl)
        health = batch.diagnostics()
    raw = {key: np.array([row[key] for row in data]) for key in data[0]}
    np.savez_compressed(output / "trace.npz", **raw, reference=ref, faces=faces, edges=edges)
    t = raw["time"]
    floor = raw["floor"] - raw["floor"][0]
    window = (t >= hold - 1e-8) & (t <= release + 1e-8)
    stable = longest(window & (floor >= .01) & np.all(raw["jaw_normal"] > 0., axis=1), t) if len(t) > 1 else 0.
    medians = np.median(raw["jaw_normal"][window], axis=0).tolist() if window.any() else None
    drop = 1000 * float(floor[window][-1] - floor[-1]) if stop is None and window.any() else None
    force_peak = max(float(peak[:2].max()), float(raw["jaw_normal"].max()))
    passed = (stop is None and stable >= 1. - 1e-7 and medians is not None
              and all(3. <= f <= 15. for f in medians) and drop is not None and drop >= 10.)
    result = dict(case=case, config=tuning_name, dt_s=dt, iterations=tuning[1], ls_iterations=tuning[2],
                  impratio=tuning[3], margin_m=tuning[4], young_scale=young_scale, pad_film_contact=contact_profile,
                  feedback="maximum side" if feedback else "archived film motion",
                  pass_hold=bool(passed), stop_reason=stop, completed_s=float(t[-1]), expected_s=len(commands) * dt,
                  hold_window_s=[hold, release], stable_hold_s=stable, hold_pad_median_N=medians,
                  hold_min_clearance_mm=1000 * float(floor[window].min()) if window.any() else None,
                  peak_lift_mm=1000 * float(floor.max()), release_drop_mm=drop, peak_N=force_peak,
                  peak_native_pad_N=peak[:2].tolist(), max_edge_ratio=float(raw["max_edge_ratio"].max()),
                  max_outside_corners=int(raw["outside_corners"].max()), max_inset_triangles=int(raw["inset_triangles"].max()),
                  direct_force_contacts=int(peak[2]), overflow=health["overflow_bits"], nonfinite=health["nonfinite"],
                  wall_s=perf_counter()-start)
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(result, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--film", choices=("a2", "a3"))
    parser.add_argument("--young-scale", type=float, default=1.)
    parser.add_argument("--contact", nargs=4, type=float)
    args = parser.parse_args()
    if args.film:
        run(args.film, ROOT / "film-fixture", case="film", young_scale=args.young_scale,
            contact_profile=args.contact)
    else:
        for name in ("a2", "a3"):
            run(name, ROOT / name)
        candidates = [(name, json.loads((ROOT/name/"summary.json").read_text())) for name in ("a2", "a3")]
        for name, result in candidates:
            if result["pass_hold"] and 1.05 < result["max_edge_ratio"] <= 1.10:
                run(name, ROOT / f"{name}-stiffer-film", young_scale=1.75)
                break
