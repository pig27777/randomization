"""Minimum operational PE-bag trials using the existing Flex parcel and fixture."""
import argparse
from dataclasses import asdict, replace
from pathlib import Path
import subprocess
from time import perf_counter
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from pe_physics import BASE, FIXTURES, PAD, PAIRS, FlexBagBackend, Parcel, save_json, vec
from pe_physics_protocol import CORNERS, IDENTITY, Trial, action, ramp
from bag_sim.wrap import point_triangle_distance, winding_number

CASES = ("contain-200", "contain-350", "try-left", "try-center", "try-right", "film", "through")
PEAK_DEPTH = .002
DT = .00005
PAD_HALF = .003
SEAM_TABLE_EDGE = .095
PROBE_K, PROBE_D, PROBE_MASS = 95., 2.2, .020
SAFE_JAW_FORCE = 16.
VARIANTS = ("baseline", "take-up-slack", "compliant-jaws", "exposed-underface", "near-seam")



def hold_window(variant):
    start = 1.74 if variant in ("take-up-slack", "near-seam") else 1.16
    return start, start + 1.


def make_trial(case, dt, variant="baseline"):
    common = dict(dt=dt, plane_table=True, mass=.20, tool_mu=.80,
                  size=(.088, .068, .024))
    if case.startswith("contain"):
        return Trial("B05-C", .55, **(common | dict(
            table=False, mass=.35 if case.endswith("350") else .20,
            initial_tools={"platen": (0., 0., -.005)})))
    if case.startswith("try"):
        offset = {"try-left": -.055, "try-center": 0., "try-right": .055}[case]
        return Trial(case, 1.86, **common, offset=(offset, 0.),
                     initial_tools={"contact_probe": (-.095, 0., .045)})
    end = hold_window(variant)[1]
    long_lift = variant in ("take-up-slack", "near-seam")
    offset = (.05, 0.) if variant == "near-seam" else (0., 0.)
    return Trial(case, end + (.45 if long_lift else .36), **common, offset=offset)


class SmokeFlex(FlexBagBackend):
    def __init__(self, spec, mesh_pads=True, edge=SEAM_TABLE_EDGE, variant="baseline"):
        self.variant = variant
        self.shelf = spec.case == "film" or variant == "exposed-underface"
        self.mesh_pads = mesh_pads
        self.edge = edge
        super().__init__(spec, 0)
        self.tool_q = np.array([self.m.jnt_qposadr[j] for j in self.fixture_joints.values()])
        self.tool_v = np.array([self.m.jnt_dofadr[j] for j in self.fixture_joints.values()])

    def fixture_xml(self, xml):
        tree = ET.fromstring(super().fixture_xml(xml))
        world = tree.find("worldbody")
        if self.spec.case.startswith("try"):
            node = world.find("body[@name='contact_probe']/inertial")
            node.set("mass", str(PROBE_MASS))
            node.set("diaginertia", "8e-7 8e-7 8e-7")
        if self.shelf:
            table = world.find("geom[@name='table']")
            h = (.40 + self.edge) / 2
            table.attrib.update(type="box", pos=vec([self.edge - h, 0., -.009]),
                                size=vec([h, .18, .009]))
            ET.SubElement(world, "geom", name="floor", type="plane", size=".5 .5 .03",
                          pos="0 0 -.14", contype="1", conaffinity="10")
        if self.mesh_pads:
            asset = tree.find("asset")
            ET.SubElement(asset, "mesh", name="smoke_pad", vertex=vec(CORNERS * [PAD_HALF, *PAD / 2]))
            for name in ("left_pad", "right_pad"):
                geom = world.find(f"body[@name='{name}']/geom")
                geom.set("type", "mesh")
                geom.set("mesh", "smoke_pad")
                geom.attrib.pop("size")
        if self.variant in ("compliant-jaws", "exposed-underface"):
            contact_time = ".005 1" if self.variant == "compliant-jaws" else ".002 1"
            for name in ("left_pad", "right_pad"):
                geom = world.find(f"body[@name='{name}']/geom")
                geom.set("solref", contact_time)
        return ET.tostring(tree, encoding="unicode")

    def drive(self, moves, probe):
        # Prescribe only moving fixtures. The compliant probe's vertical DOF
        # remains dynamic, so its contact impulse can move it out of the film.
        self.apply_action(dict(tools=moves, load=np.zeros(3), clamps={}))
        if probe is not None:
            x, vx, target = probe
            q, v = self.tool_q[0], self.tool_v[0]
            self.d.qpos[q:q + 2] = [x, 0.]
            self.d.qvel[v:v + 2] = [vx, 0.]
            self.d.qfrc_applied[v + 2] = PROBE_K * (target - self.d.qpos[q + 2]) - PROBE_D * self.d.qvel[v + 2]
        Parcel.step(self)


def probe_command(t):
    drop, _ = ramp(t, .18, .58)
    sweep, speed = ramp(t, .62, 1.54)
    lift, _ = ramp(t, 1.62, 1.84)
    return (-.095 + .19 * sweep, .19 * speed, .045 - .018 * (drop - lift))


def quaternion(frame):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, frame.ravel())
    return q


class Grasp:
    def __init__(self, backend, edge):
        self.backend = backend
        self.edge = edge
        self.origin = None
        self.floor = .0
        self.jaw_gap = None
        self.force = np.zeros(2)
        self.reach = None
        self.mode = backend.spec.case
        self.variant = backend.variant
        self.hold_start, self.release = hold_window(self.variant)

    def aim(self):
        q, _ = self.backend.get_bag_state()
        p, _ = self.backend.get_payload_state()
        if self.variant == "exposed-underface":
            self.origin = p[:3] + [.028, 0., 0.]
            self.axis = np.array([0., 0., 1.])
            self.orientation = quaternion(np.column_stack((self.axis, [0., 1., 0.], [-1., 0., 0.])))
            self.open_gap = .046
            self.min_gap = self.backend.spec.size[2] - .00075
            self.reach = self.origin + [0., .074, 0.]
        elif self.mode == "through":
            self.origin = np.array([p[0], p[1], max(.011, p[2])])
            self.axis = np.array([0., 1., 0.])
            self.orientation = quaternion(np.column_stack((self.axis, [-1., 0., 0.], [0., 0., 1.])))
            self.open_gap = self.backend.spec.size[1] + .018
            self.min_gap = self.backend.spec.size[1] - .001
            self.reach = self.origin + [0., 0., .035]
        else:
            r = self.backend.envelope["reference"]
            edge_vertex = int(np.argmin((r[:, 0] - r[:, 0].max())**2 + r[:, 1]**2))
            y = r[edge_vertex, 1]
            inner = np.flatnonzero((abs(r[:, 1] - y) < .001) & (r[:, 0] < r[edge_vertex, 0] - .01))
            adjacent = inner[np.isclose(r[inner, 0], r[inner, 0].max())]
            inside = q[adjacent].mean(axis=0)
            tangent = q[edge_vertex] - inside
            tangent[1] = 0.
            tangent /= np.linalg.norm(tangent)
            self.axis = np.cross(tangent, [0., 1., 0.])
            if self.axis[2] < 0:
                self.axis *= -1
                tangent *= -1
            self.origin = q[edge_vertex] - .004 * tangent
            self.orientation = quaternion(np.column_stack((self.axis, [0., 1., 0.], -tangent)))
            self.open_gap = .015
            self.min_gap = .00025
            self.reach = self.origin + [0., .026, 0.]
        self.jaw_gap = self.open_gap

    def command(self, time, dt):
        if time < .24:
            return {}
        if self.origin is None:
            self.aim()
        approach, speed = ramp(time, .24, .47)
        raise_up, lift_speed = ramp(time, .87, self.hold_start)
        travel = .160 if self.variant in ("take-up-slack", "near-seam") else .045
        center = self.reach + (self.origin - self.reach) * approach + [0., 0., travel * raise_up]
        velocity = (self.origin - self.reach) * speed + [0., 0., travel * lift_speed]
        old_gap = self.jaw_gap
        if .47 <= time < self.release:
            normal = self.force.mean()
            gain = .012 if self.variant in ("compliant-jaws", "exposed-underface") else .0035
            change = np.clip(gain * (normal - 8.), -.055, .16)
            if max(self.force) > SAFE_JAW_FORCE:
                change = .16
            self.jaw_gap = float(np.clip(old_gap + change * dt, self.min_gap, self.open_gap))
        if time >= self.release:
            self.jaw_gap = min(self.open_gap, old_gap + .14 * dt)
            if self.variant in ("exposed-underface", "near-seam"):
                retreat, retreat_speed = ramp(time, self.release + .135, self.release + .315)
                center += [0., .074 * retreat, 0.]
                velocity += [0., .074 * retreat_speed, 0.]
        rate = (self.jaw_gap - old_gap) / dt
        return {name: (center + sign * self.axis * (PAD_HALF + self.jaw_gap / 2),
                       velocity + sign * rate * self.axis / 2, self.orientation)
                for name, sign in (("left_pad", -1.), ("right_pad", 1.))}


def geometry(backend, q, payload):
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, payload[3:])
    rotation = rotation.reshape(3, 3)
    half = np.asarray(backend.spec.size) / 2
    corners = CORNERS * half @ rotation.T + payload[:3]
    faces = backend.envelope["faces"]
    probes = np.vstack((payload[:3], corners))
    outside = np.abs(winding_number(q, faces, probes)) < .5
    distances = point_triangle_distance(corners[outside[1:]], q, faces)
    material = np.abs((q - payload[:3]) @ rotation) - half
    core_penetration = np.maximum(0., -material.max(axis=1)).max()
    return (float(corners[:, 2].min()), float(max(distances, default=0.)),
            float(core_penetration), bool(outside[0]), int(outside[1:].sum()))


def sample(backend, grasp, t):
    q, _ = backend.get_bag_state()
    payload, _ = backend.get_payload_state()
    fixtures = backend.get_fixture_state()
    reactions = backend.get_contact_wrench()
    if not all(np.isfinite(a).all() for a in (q, payload, reactions["wrench"])):
        raise FloatingPointError(f"Nonfinite state at {t:.5f} s")
    floor, outside, penetration, center_free, corners = geometry(backend, q, payload)
    force = reactions["wrench"][[2, 3], :3]
    shelf_intrusion = 0.
    if grasp is not None and grasp.origin is not None:
        grasp.force = np.maximum(0., np.array([-force[0] @ grasp.axis, force[1] @ grasp.axis]))
        if backend.shelf:
            for p in fixtures[2:4]:
                rot = np.empty(9)
                mujoco.mju_quat2Mat(rot, p[3:])
                points = (CORNERS * [PAD_HALF, *PAD / 2]) @ rot.reshape(3, 3).T + p[:3]
                if points[:, 2].min() < 0. and points[:, 2].max() > -.018:
                    shelf_intrusion = max(shelf_intrusion, grasp.edge - float(points[:, 0].min()))
    return dict(time=t, q=q, payload=payload, fixtures=fixtures, **reactions, floor=floor, outside=outside,
                core_penetration=penetration, escaped_center=center_free, outside_corners=corners,
                table_penetration=max(0., -float(q[:, 2].min())) if backend.spec.plane_table and
                backend.spec.table and not backend.shelf else 0.,
                shelf_intrusion=shelf_intrusion, solved_warning=sum(w.number for w in backend.d.warning),
                jaw_normal=grasp.force.copy() if grasp is not None else np.zeros(2),
                jaw_gap=grasp.jaw_gap if grasp is not None and grasp.jaw_gap is not None else np.nan)


def longest(mask, times):
    indices = np.flatnonzero(mask)
    if not len(indices):
        return 0.
    intervals = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
    return max(float(times[i[-1]] - times[i[0]]) for i in intervals)


def rotations(quat):
    r = np.empty((len(quat), 3, 3))
    for q, mat in zip(quat, r):
        mujoco.mju_quat2Mat(mat.reshape(9), q)
    return r


def jaw_core_overlap(payload, fixtures, size):
    """Count sampled jaw/solid overlap deeper than the geometric error budget."""
    core_axes = rotations(payload[:, 3:])
    core_half = np.asarray(size) / 2 - PEAK_DEPTH
    pad_half = np.array([PAD_HALF, *PAD / 2])
    overlaps = np.zeros(len(payload), dtype=bool)
    for pad in fixtures[:, 2:4].transpose(1, 0, 2):
        delta = pad[:, :3] - payload[:, :3]
        near = np.linalg.norm(delta, axis=1) < np.linalg.norm(core_half) + np.linalg.norm(pad_half)
        if not np.any(near):
            continue
        core = core_axes[near].transpose(0, 2, 1)
        finger = rotations(pad[near, 3:]).transpose(0, 2, 1)
        cross = np.cross(core[:, :, None], finger[:, None, :]).reshape(-1, 9, 3)
        axes = np.concatenate((core, finger, cross), axis=1)
        distance = np.abs(np.einsum("ijk,ik->ij", axes, delta[near]))
        reach = np.abs(axes @ core.transpose(0, 2, 1)) @ core_half
        reach += np.abs(axes @ finger.transpose(0, 2, 1)) @ pad_half
        meaningful = np.linalg.norm(axes, axis=2) > 1e-9
        overlaps[near] |= np.all((distance < reach) | ~meaningful, axis=1)
    return int(overlaps.sum())


def metrics(case, d, variant="baseline", size=(.088, .068, .024)):
    t = d["time"]
    active = d["pairs"][:, PAIRS.index("payload_fixture")]
    numerical = bool(d["solved_warning"].max() or np.any(d["escaped_center"])
                     or max(d["outside"].max(), d["core_penetration"].max(), d["table_penetration"].max()) > PEAK_DEPTH)
    bypass = bool(np.any(active) or d["shelf_intrusion"].max() > .001)
    result = dict(case=case, pass_geometry=not numerical, bypass=bypass,
                  max_corner_escape_mm=1000 * float(d["outside"].max()),
                  max_payload_vertex_penetration_mm=1000 * float(d["core_penetration"].max()),
                  max_table_penetration_mm=1000 * float(d["table_penetration"].max()),
                  max_solver_depth_mm=1000 * float(d["pair_depth"].max()),
                  payload_move_mm=1000 * float(np.linalg.norm(d["payload"][-1, :3] - d["payload"][0, :3])),
                  max_outside_corners=int(d["outside_corners"].max()), escaped_center=bool(np.any(d["escaped_center"])),
                  warnings=int(d["solved_warning"].max()), max_shelf_intrusion_mm=1000 * float(d["shelf_intrusion"].max()))
    if case.startswith("contain"):
        passed = not numerical and not bypass and result["payload_move_mm"] > 1.
    elif case.startswith("try"):
        scan = (t >= .62) & (t <= 1.54)
        x = d["fixtures"][:, 0, 0]
        force = np.maximum(0., d["wrench"][:, 0, 2])
        result.update(peak_N=float(force.max()), scan_peak_N=float(force[scan].max()),
                      sustained_s=longest(scan & (force >= .08), t),
                      scan_force_N=[float(force[scan & (x < -.025)].mean()),
                                    float(force[scan & (abs(x) <= .025)].mean()),
                                    float(force[scan & (x > .025)].mean())])
        passed = not numerical and not bypass and result["sustained_s"] >= .020
    else:
        right_after, release = hold_window(variant)
        hold = (t >= right_after - 1e-8) & (t <= release + 1e-8)
        gap = d["floor"] - d["floor"][0]
        raised = gap >= .010
        bilateral = np.all(d["jaw_normal"] > 0., axis=1)
        bilateral &= np.all(d["pairs"][:, [PAIRS.index("film_left_pad"), PAIRS.index("film_right_pad")]] > 0, axis=1)
        overlap = jaw_core_overlap(d["payload"], d["fixtures"], size)
        result.update(peak_lift_mm=1000 * float(gap.max()), hold_min_lift_mm=1000 * float(gap[hold].min()),
                      loaded_hold_s=longest(hold & bilateral & raised, t),
                      bilateral_fraction=float(np.mean(bilateral[hold])),
                      bilateral_above_025N_fraction=float(np.mean(np.all(d["jaw_normal"][hold] > .25, axis=1))),
                      severe_core_jaw_overlap_samples=overlap,
                      release_drop_mm=1000 * float(d["floor"][hold][-1] - d["floor"][-1]),
                      max_jaw_N=float(d["jaw_normal"].max()))
        passed = (not numerical and not bypass and not overlap and bool(np.all(raised[hold])) and
                  result["bilateral_fraction"] >= .95 and result["release_drop_mm"] >= 10.)
    result["status"] = "PASS" if passed else "FAIL"
    return result


def run(case, output, dt, mesh_pads, edge, variant):
    output.mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    spec = make_trial(case, dt, variant)
    backend = SmokeFlex(spec, mesh_pads, edge, variant)
    assert backend.m.neq == 0
    grasp = Grasp(backend, edge) if case in ("film", "through") else None
    rows = [sample(backend, grasp, 0.)]
    commands = []
    interval = max(1, round(.005 / dt))
    for i in range(1, round(spec.duration / dt) + 1):
        t = (i - 1) * dt
        moves = grasp.command(t, dt) if grasp else (
            action(spec, t)["tools"] if case.startswith("contain") else {})
        probe = probe_command(t) if case.startswith("try") else None
        backend.drive(moves, probe)
        if grasp:
            commands.append([np.r_[*moves[name]] for name in ("left_pad", "right_pad")] if moves
                            else np.full((2, 10), np.nan))
        if i % interval == 0 or i == round(spec.duration / dt):
            rows.append(sample(backend, grasp, i * dt))
            if rows[-1]["solved_warning"]:
                break
    d = {key: np.array([row[key] for row in rows]) for key in rows[0]}
    d.update({k: backend.envelope[k] for k in ("reference", "faces", "edges",
                                               "masses", "panels", "current")})
    if commands:
        d["pad_commands"] = np.asarray(commands)
    np.savez_compressed(output / "trace.npz", **d)
    (output / "model.xml").write_text("\n".join(line.rstrip() for line in backend.xml.splitlines()) + "\n")
    data = metrics(case, d, variant, spec.size)
    if d["time"][-1] < spec.duration - dt:
        data["status"] = "FAIL"
    save_json(output / "summary.json", data)
    save_json(output / "config.json", dict(case=case, dt_s=dt, variant=variant, spec=asdict(spec), model=backend.metadata,
                                          engine=mujoco.__version__, mesh_pads=mesh_pads, support_edge=edge,
                                          runtime_s=perf_counter() - start, revision=subprocess.check_output(
                                              ["git", "rev-parse", "HEAD"], text=True).strip()))
    print(data, flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("case", choices=CASES)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--box-pads", action="store_true")
    parser.add_argument("--variant", choices=VARIANTS, default="baseline")
    parser.add_argument("--edge", type=float, default=SEAM_TABLE_EDGE)
    args = parser.parse_args()
    if args.variant != "baseline" and (args.case, args.variant) not in (
            ("film", "take-up-slack"), ("film", "near-seam"),
            ("through", "compliant-jaws"), ("through", "exposed-underface")):
        parser.error("Grasp variant does not match the trial")
    run(args.case, args.output, args.dt, not args.box_pads, args.edge, args.variant)


if __name__ == "__main__":
    main()
