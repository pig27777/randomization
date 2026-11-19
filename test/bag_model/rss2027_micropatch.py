"""Instrument a native thin-shell coupon under two opposing rounded pads."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "experience")]
from yam_platform.geometry import fingertip_xml  # noqa: E402
from bag_sim.physical_parcel import material_parameters, parcel_profile, vec  # noqa: E402

SIDE = 0.060
RADIUS = 0.0001
OPEN_HALF = 0.0115
CLOSED_HALF = 0.0042
PAD_HALF_THICKNESS = .004
PAD_BOTTOM = 0.020
PULL_SPEED = 0.025
PHASES = (("settle", .08), ("press", .45), ("gather", .24))


def refine_axis(axis, half_extent, levels):
    for _ in range(levels):
        mid = (axis[:-1] + axis[1:]) / 2
        axis = np.sort(np.r_[axis, mid[np.abs(mid) <= half_extent]])
    return axis


def coupon_grid(cells, slack, prefold, radius, *, prefold_width=None, flat_waist=False,
                refine_contact=False, refine_levels=1):
    """Keep the initial x-edge lengths exactly equal to the flat reference."""
    axis = np.linspace(-SIDE / 2, SIDE / 2, cells + 1)
    width = (prefold_width or 2 * SIDE / cells) if prefold else SIDE / 2
    if flat_waist:
        ridges = ((-.030, -.025, -.020), (.020, .025, .030))
    else:
        ridges = ((-width / 2, 0., width / 2),)
    x_axis = np.unique(np.round(np.r_[axis, np.ravel(ridges)], 12))
    y_axis = axis
    if refine_contact in (True, "x", "both"):
        x_axis = refine_axis(x_axis, OPEN_HALF + .004, refine_levels)
    if refine_contact in (True, "y", "both"):
        y_axis = refine_axis(y_axis, .010, refine_levels)
    x, y = np.meshgrid(x_axis, y_axis)
    length = np.diff(x_axis)
    mid = (x_axis[:-1] + x_axis[1:]) / 2
    slope = np.zeros(len(length))
    bent_length = sum(end - start for start, _, end in ridges)
    angle = np.arccos(1 - slack / bent_length)
    for start, apex, end in ridges:
        slope[(start < mid) & (mid < apex)] = angle
        slope[(apex < mid) & (mid < end)] = -angle
    current_x = np.r_[0., np.cumsum(length * np.cos(slope))]
    current_x -= current_x[-1] / 2
    current_z = radius + np.r_[0., np.cumsum(length * np.sin(slope))]
    ref = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, radius)))
    current = np.column_stack((np.tile(current_x, len(y_axis)), y.ravel(), np.tile(current_z, len(y_axis))))
    faces = []
    for row in range(len(y_axis) - 1):
        for col in range(len(x_axis) - 1):
            a = row * len(x_axis) + col
            b = a + len(x_axis)
            faces.extend(((a, a + 1, b + 1), (a, b + 1, b)) if (row + col) % 2 == 0 else
                         ((a, a + 1, b), (a + 1, b + 1, b)))
    edges = np.arange(x.size).reshape(x.shape)[:, [0, -1]].ravel()
    return ref, current, np.asarray(faces), edges


def build(args):
    if args.match_whole_bag_pe:
        shell = parcel_profile("pe")["shell"]
        thickness = shell["material_thickness"]
        young, density = shell["membrane"] / thickness, shell["areal_mass"] / thickness
        poisson, radius = shell["poisson"], shell["radius"]
        support_friction, self_friction, support_time = .45, .45, .001
        ccd = 'ccd_tolerance="1e-9"'
    else:
        young, thickness, density, poisson = 255e6, 50.8e-6, 924., .45
        radius = RADIUS
        support_friction, self_friction, support_time = .3, .5, .0005
        ccd = ""
    material = material_parameters(young, thickness, density, poisson)
    ref, current, faces, edges = coupon_grid(
        args.cells, args.slack, args.prefold, radius, prefold_width=args.prefold_width,
        flat_waist=args.match_whole_bag_pe and not args.prefold, refine_contact=args.refine_contact,
        refine_levels=args.refine_levels)
    model = ET.fromstring(f"""
      <mujoco model="rss2027_micropatch">
        <option timestep="{args.dt}" integrator="discrete" solver="Newton"
          iterations="150" cone="elliptic" tolerance="1e-8" {ccd}/>
        <asset/>
        <worldbody>
          <geom name="support" type="plane" size=".08 .08 .01" contype="1"
            conaffinity="2" priority="1" friction="{support_friction} .005 .0001"
            solref="{support_time} 1" solimp=".999 .999 .001"/>
          <flexcomp name="coupon" type="direct" dim="2" mass="{material['areal_mass'] * SIDE**2}"
            radius="{radius}" point="{vec(ref)}" element="{' '.join(map(str, faces.ravel()))}">
            <elasticity young="{young}" poisson="{poisson}" thickness="{thickness}"
              damping=".004" elastic2d="both"/>
            <contact contype="2" conaffinity="7" condim="3" selfcollide="auto"
              friction="{self_friction} .005 .0001" solref=".001 1" solimp=".999 .999 .001"/>
          </flexcomp>
        </worldbody>
        <equality/>
      </mujoco>
    """)
    pad_footprint = np.abs(current[:, 0]) <= OPEN_HALF + .004
    local_top = (current[pad_footprint, 2].max() + radius if args.match_whole_bag_pe
                 else current[:, 2].max())
    start_z = PAD_BOTTOM + (.0003 if args.prefold else local_top + .002)
    open_half = OPEN_HALF
    if args.prefold:
        open_half = .005 if args.grip_target is None else .0065
    for name, sign in (("left", -1), ("right", 1)):
        tip = ET.fromstring(fingertip_xml(name, 0, (sign * open_half, 0, start_z)))
        for part in tip.findall("geom"):
            part.attrib.update(conaffinity="2", condim="3", priority="1", friction=f"{args.friction} .005 .0001",
                               solref=f"{args.pad_contact_time} 1", solimp=".999 .999 .001")
            if part.get("type") == "box":
                half = np.fromstring(part.attrib.pop("size"), sep=" ")
                corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
                ET.SubElement(model.find("asset"), "mesh", name=f"{name}_core_mesh", vertex=vec(corners * half))
                part.attrib.update(type="mesh", mesh=f"{name}_core_mesh")
        model.find("worldbody").append(tip)
    for node in edges:
        ET.SubElement(model.find("equality"), "connect", body1=f"coupon_{node}", anchor="0 0 0",
                      solref=".001 1", solimp=".999 .999 .001")
    xml = ET.tostring(model, encoding="unicode")
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    bodies = m.flex_vertbodyid
    if args.match_whole_bag_pe:
        triangles = ref[faces]
        area = .5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                            triangles[:, 2] - triangles[:, 0]), axis=1)
        masses = np.zeros(len(ref))
        np.add.at(masses, faces.ravel(), np.repeat(area * material["areal_mass"] / 3, 3))
        m.body_mass[bodies] = masses
        mujoco.mj_setConst(m, d)
    m.eq_data[:, 3:6] = current[edges]
    qpos = m.jnt_qposadr[m.body_jntadr[bodies, None] + np.arange(3)]
    d.qpos[qpos] = current - ref
    mujoco.mj_forward(m, d)
    if not np.allclose(d.flexvert_xpos, current) or np.max(np.abs(d.efc_pos[:3 * len(edges)])) > 1e-9:
        raise RuntimeError("Coupon initialization does not match its prescribed anchor positions")
    material.update(young_pa=young, thickness_m=thickness, density_kg_m3=density,
                    pad_friction=args.friction, self_friction=self_friction, support_friction=support_friction,
                    contact_radius_m=radius, support_contact_time_s=support_time,
                    vertex_mass_distribution="triangle_area" if args.match_whole_bag_pe else "equal_per_vertex",
                    total_film_mass_kg=float(m.body_mass[bodies].sum()),
                    constitutive_model="native stretch and bending")
    return xml, m, d, material, ref, current, faces


def pad_reactions(m, d, pad_ids, pull_speed=None):
    force = np.zeros((2, 3))
    inner = np.zeros(2)
    active = np.zeros(2, dtype=int)
    cone_normal = np.zeros(2)
    sliding_normal = np.zeros(2)
    local = np.zeros(6)
    for index, contact in enumerate(d.contact):
        if contact.efc_address < 0 or 0 not in contact.flex:
            continue
        for slot, geom_id in enumerate(contact.geom):
            if geom_id not in pad_ids:
                continue
            pad = pad_ids[geom_id]
            mujoco.mj_contactForce(m, d, index, local)
            active[pad] += local[0] > 0
            frame = contact.frame.reshape(3, 3)
            sign = 2 * slot - 1
            force[pad] += sign * (frame.T @ local[:3])
            # Separate inward-facing pinch contacts from downward press contacts.
            if (2 * pad - 1) * sign * frame[0, 0] > .7:
                normal = local[0] * abs(frame[0, 0])
                inner[pad] += normal
                if pull_speed is not None and contact.dim == 3 and normal > 0:
                    shear = local[1:3] / contact.friction[:2]
                    if np.linalg.norm(shear) >= .95 * local[0]:
                        cone_normal[pad] += normal
                        adr = contact.efc_address
                        pull_slip = sign * float(frame[1:, 2] @ d.efc_vel[adr + 1:adr + 3])
                        if pull_slip > .10 * pull_speed:
                            sliding_normal[pad] += normal
    return force, inner, active, cone_normal, sliding_normal


def aperture_lengths(nodes, reference_lengths, half, z):
    """Clip the central material line to the 3D volume between the pads."""
    half_gap = half - PAD_HALF_THICKNESS
    lower = np.array([-half_gap, -.010, z - PAD_BOTTOM])
    upper = np.array([half_gap, .010, z + PAD_BOTTOM])
    start = nodes[:-1]
    delta = np.diff(nodes, axis=0)
    moving = np.abs(delta) > 1e-14
    safe_delta = np.where(moving, delta, 1.)
    entry = (lower - start) / safe_delta
    exit = (upper - start) / safe_delta
    first = np.where(moving, np.minimum(entry, exit), -np.inf).max(axis=1).clip(0., 1.)
    last = np.where(moving, np.maximum(entry, exit), np.inf).min(axis=1).clip(0., 1.)
    within = (moving | ((start >= lower) & (start <= upper))).all(axis=1)
    fraction = np.maximum(last - first, 0.) * within
    included = fraction > 0.
    if not included.any():
        return 0., 0., 0.
    x = start[included, 0] + delta[included, 0] * np.array([first[included], last[included]])
    arc = float(fraction @ np.linalg.norm(delta, axis=1))
    return float(fraction @ reference_lengths), arc, max(0., arc - float(np.ptp(x)))


def sustained_starts(time_s, mask):
    bounds = np.flatnonzero(np.diff(np.r_[False, mask, False])).reshape(-1, 2)
    return [start for start, end in bounds if time_s[end - 1] - time_s[start] >= .010 - 1e-9]


def run(args):
    started = time.perf_counter()
    xml, m, d, material, ref, initial, faces = build(args)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "model.xml").write_text(xml)
    mujoco.mj_saveModel(m, str(args.output / "model.mjb"), None)
    bodies = [m.body(name).id for name in ("left", "right")]
    pad_joints = m.body_jntadr[bodies, None] + np.arange(3)
    pad_qpos = m.jnt_qposadr[pad_joints]
    pad_dofs = m.jnt_dofadr[pad_joints]
    pad_origins = m.body_pos[bodies].copy()
    pad_ids = {int(geom): pad for pad, body in enumerate(bodies) for geom in np.flatnonzero(m.geom_bodyid == body)}
    gather_seconds = .012 if args.grip_target is None else .30
    phases = (("gather", gather_seconds),) if args.prefold else PHASES
    phases = (*phases, ("hold", args.hold_seconds), ("pull", args.pull_seconds))
    controlled_gather = args.gather_seconds is not None
    controlled_pull = args.pull_force_target is not None
    if controlled_gather:
        phases = tuple((phase, args.gather_seconds if phase == "gather" else duration)
                       for phase, duration in phases)
    if args.gather_only:
        phases = phases[:3]
    row = np.flatnonzero(np.isclose(ref[:, 1], 0., atol=1e-12))
    reference_lengths = np.linalg.norm(np.diff(ref[row], axis=0), axis=1)
    local_pitch = [SIDE / args.cells / (2**args.refine_levels if args.refine_contact in (axis, "both") else 1)
                   for axis in ("x", "y")]
    shape = dict(vertices=len(ref), triangles=len(faces), nominal_local_pitch_m=min(local_pitch),
                 nominal_local_pitch_xy_m=local_pitch,
                 reference_span_m=float(reference_lengths.sum()), projected_span_m=float(np.ptp(initial[row, 0])),
                 central_rise_m=float(initial[row[len(row) // 2], 2] - initial[row[0], 2]),
                 prefold_width_m=(args.prefold_width or 2 * SIDE / args.cells) if args.prefold else None)
    config = vars(args) | {"output": str(args.output), "material": material, "phases_s": phases,
                           "initial_geometry": shape, "ccd_tolerance_m": float(m.opt.ccd_tolerance),
                           "hostname": platform.node(), "mujoco": mujoco.__version__,
                           "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    rows, snapshots = [], []
    z = pad_origins[0, 2]
    open_half = pad_origins[1, 0]
    close_half = CLOSED_HALF if args.closed_gap is None else PAD_HALF_THICKNESS + args.closed_gap / 2
    min_grip_half = .00405 if args.prefold else PAD_HALF_THICKNESS + 1.5 * material["contact_radius_m"]
    half = open_half
    previous = pad_origins
    marker = np.flatnonzero((abs(ref[:, 0]) < 1e-9) & (abs(ref[:, 1]) < .006))
    press = 0.
    peak_penetration, failure = 0., None
    press_band_s = peak_press_band_s = peak_press_N = 0.
    press_filtered_N = gather_band_s = peak_gather_band_s = 0.
    gather_pressure_window = None
    grip_filtered_N = grip_band_s = peak_grip_band_s = 0.
    pull_filtered_N = pull_band_s = peak_pull_band_s = 0.
    minimum_pad_height = z - PAD_BOTTOM
    sample_steps = max(1, round(.001 / args.dt))
    for phase, duration in phases[:1] if args.smoke else phases:
        nsteps = 2 if args.smoke else round(duration / args.dt)
        for step in range(nsteps):
            pull_rate = 0.
            gripping = args.grip_target is not None and (args.prefold or phase in ("hold", "pull"))
            if gripping:
                rate = float(np.clip(.0015 * (grip_filtered_N - args.grip_target), -.012, .012))
                half = float(np.clip(half + args.dt * rate, min_grip_half, open_half))
            elif phase == "gather":
                half = open_half - (open_half - close_half) * (step + 1) / nsteps
            elif phase in ("hold", "pull"):
                half = close_half
            if not args.prefold and phase in ("press", "gather"):
                if controlled_gather and phase == "gather":
                    z_rate = np.clip(.01 * (press_filtered_N - args.press_force), -.008, .008)
                else:
                    z_rate = np.clip(.02 * (press - args.press_force), -.02, .02)
                z = max(PAD_BOTTOM + .00002, z + args.dt * z_rate)
            elif phase == "hold" and args.hold_downforce is not None:
                z += args.dt * float(np.clip(.02 * (press - args.hold_downforce), 0., .006))
            elif phase == "pull":
                pull_rate = args.pull_speed
                if controlled_pull:
                    pull_goal = args.pull_force_target * min(1., (step + 1) * args.dt / .40)
                    if step == 0:
                        pull_filtered_N = -press
                    pull_rate = float(np.clip(.1 * (pull_goal - pull_filtered_N), 0., args.pull_speed))
                z += pull_rate * args.dt
            positions = np.array([[-half, 0, z], [half, 0, z]])
            # Supply the prescribed rigid-body velocity to the contact Jacobian.
            d.qpos[pad_qpos] = positions - pad_origins
            d.qvel[pad_dofs] = (positions - previous) / args.dt
            previous = positions
            minimum_pad_height = min(minimum_pad_height, z - PAD_BOTTOM)
            t_eval = float(d.time)
            mujoco.mj_step(m, d)
            penetration = max(0., -float(d.contact.dist.min())) if d.ncon else 0.
            peak_penetration = max(penetration, peak_penetration)
            sampled = step % sample_steps == 0 or step + 1 == nsteps
            watching_normal = (args.match_whole_bag_pe and phase == "press") or (controlled_gather and phase == "gather")
            if sampled or args.grip_target is not None or watching_normal:
                traction_speed = pull_rate if args.match_whole_bag_pe and sampled and pull_rate > 0 else None
                force, inner, active, cone, sliding = pad_reactions(
                    m, d, pad_ids, pull_speed=traction_speed)
                press = float(force[:, 2].sum())
                if controlled_pull and phase == "pull":
                    pull_filtered_N += args.dt / (.005 + args.dt) * (-press - pull_filtered_N)
                    in_band = .9 * args.pull_force_target <= -press <= 1.1 * args.pull_force_target
                    both_loaded = min(inner) > .001 and (step + 1) * args.dt >= .40
                    pull_band_s = pull_band_s + args.dt if in_band and both_loaded else 0.
                    peak_pull_band_s = max(peak_pull_band_s, pull_band_s)
                if args.grip_target is not None:
                    grip_N = float(inner.sum())
                    grip_filtered_N += args.dt / (.005 + args.dt) * (grip_N - grip_filtered_N)
                    if phase == "hold":
                        in_band = .9 * args.grip_target <= grip_N <= 1.1 * args.grip_target
                        grip_band_s = grip_band_s + args.dt if in_band else 0.
                        peak_grip_band_s = max(peak_grip_band_s, grip_band_s)
                if phase == "press" and args.match_whole_bag_pe:
                    peak_press_N = max(peak_press_N, press)
                    in_band = .8 * args.press_force <= press <= 1.2 * args.press_force
                    press_band_s = press_band_s + args.dt if in_band else 0.
                    peak_press_band_s = max(peak_press_band_s, press_band_s)
                if controlled_gather and watching_normal:
                    press_filtered_N += args.dt / (.005 + args.dt) * (press - press_filtered_N)
                    if phase == "gather":
                        in_band = .8 * args.press_force <= press <= 1.2 * args.press_force
                        gather_band_s = gather_band_s + args.dt if in_band else 0.
                        if gather_band_s > peak_gather_band_s:
                            peak_gather_band_s = gather_band_s
                            gather_pressure_window = [t_eval - gather_band_s + args.dt, t_eval]
            if sampled:
                marker_z = float(d.flexvert_xpos[marker, 2].mean())
                contained = aperture_lengths(d.flexvert_xpos[row], reference_lengths, half, z)
                rows.append((phase, t_eval, open_half - half, z, *force.ravel(), *inner,
                             -press, penetration, marker_z, z - marker_z, *contained, *active, *cone, *sliding,
                             pull_rate))
                snapshots.append(d.flexvert_xpos.copy())
            if d.warning.number.sum() or not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
                failure = f"Unhealthy simulation in {phase} at {t_eval:.6g} s"
                break
        if failure:
            break
    with (args.output / "curve.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        columns = ("t_s", "gather_per_pad_m", "pad_z_m", "left_fx_N", "left_fy_N", "left_fz_N",
                   "right_fx_N", "right_fy_N", "right_fz_N", "left_inner_N", "right_inner_N",
                   "signed_pull_N", "penetration_m", "marker_z_m", "pad_minus_marker_z_m",
                   "aperture_material_length_m", "aperture_current_arc_length_m", "fold_excess_length_m",
                   "left_active", "right_active", "left_cone_normal_N", "right_cone_normal_N",
                   "left_sliding_normal_N", "right_sliding_normal_N", "pad_pull_speed_m_s")
        writer.writerow(("phase", *columns))
        writer.writerows(rows)
    np.savez_compressed(args.output / "coupon_nodes.npz", reference_m=ref, faces=faces, current_m=snapshots)
    pull_rows = np.asarray([row[1:] for row in rows if row[0] == "pull"]).reshape(-1, len(columns))
    pull = dict(zip(columns, pull_rows.T, strict=True))
    time_s = pull["t_s"]
    tension = pull["signed_pull_N"]
    bilateral = (pull["left_inner_N"] > .001) & (pull["right_inner_N"] > .001)
    loaded = bilateral & (tension > .001)
    transitions = np.flatnonzero(np.diff(np.r_[False, loaded, False]))
    longest_load = max((time_s[end - 1] - time_s[begin] for begin, end in transitions.reshape(-1, 2)), default=0.)
    capture = sustained_starts(time_s, loaded)
    release, slip, traction_slip = [], [], []
    if capture:
        after_capture = time_s > time_s[capture[0]]
        detached = (pull["left_active"] == 0) & (pull["right_active"] == 0)
        release = sustained_starts(time_s, after_capture & detached)
        # Require both loss of material from the aperture and relative motion
        # of the tracked material line while the pads still carry tensile load.
        capture_index = capture[0]
        retained = pull["aperture_material_length_m"][capture_index:]
        distance = pull["pad_minus_marker_z_m"][capture_index:]
        radius = material["contact_radius_m"]
        feed_loss = np.maximum.accumulate(retained) - retained
        offset = distance - np.minimum.accumulate(distance)
        slip = [capture_index + i for i in sustained_starts(
            time_s[capture_index:], loaded[capture_index:] & (feed_loss > radius) & (offset > radius))]
        loaded_sliding = loaded & (pull["left_sliding_normal_N"] >= .5 * pull["left_inner_N"])
        loaded_sliding &= pull["right_sliding_normal_N"] >= .5 * pull["right_inner_N"]
        traction_slip = sustained_starts(time_s, after_capture & loaded_sliding)
    valid = failure is None and peak_penetration <= .001
    summary = dict(numerically_valid=valid, failure=failure, peak_penetration_m=peak_penetration,
                   warnings=d.warning.number.tolist(), bilateral_pull_samples=int(bilateral.sum()),
                   bilateral_loaded_pull_samples=int(loaded.sum()), longest_bilateral_load_s=longest_load,
                   peak_pull_N=float(tension.max(initial=0.)),
                   press_peak_N=peak_press_N, longest_press_band_s=peak_press_band_s,
                   longest_gather_band_s=peak_gather_band_s if controlled_gather else None,
                   gather_pressure_window_s=gather_pressure_window,
                   grip_target_N=args.grip_target, longest_grip_hold_band_s=peak_grip_band_s,
                   pull_force_target_N=args.pull_force_target,
                   longest_pull_force_band_s=peak_pull_band_s if controlled_pull else None,
                   minimum_pad_height_m=minimum_pad_height,
                   peak_aperture_material_length_m=max((row[16] for row in rows), default=0.),
                   peak_fold_excess_length_m=max((row[18] for row in rows), default=0.),
                   capture_t_s=float(time_s[capture[0]]) if capture else None,
                   release_t_s=float(time_s[release[0]]) if release else None,
                   slip_candidate_t_s=float(time_s[slip[0]]) if slip else None,
                   slip_candidate_force_N=float(tension[slip[0]]) if slip else None,
                   traction_slip_candidate_t_s=float(time_s[traction_slip[0]]) if traction_slip else None,
                   traction_slip_candidate_force_N=float(tension[traction_slip[0]]) if traction_slip else None,
                   simulated_s=float(d.time), wall_s=time.perf_counter() - started,
                   smoke_only=args.smoke, gather_only=args.gather_only, material_parameters_identified=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return valid


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cells", type=int, default=12)
    parser.add_argument("--slack", type=float, default=.002)
    parser.add_argument("--dt", type=float, default=.0001)
    parser.add_argument("--friction", type=float, help="pad friction (default .5, or .8 for matched PE)")
    parser.add_argument("--press-force", type=float, default=.5)
    parser.add_argument("--pad-contact-time", type=float, default=.001, help="pad contact time constant in seconds")
    parser.add_argument("--prefold", action="store_true", help="test pulling an initially folded coupon")
    parser.add_argument("--prefold-width", type=float, help="unfolded width of the central ridge, in metres")
    parser.add_argument("--refine-contact", nargs="?", const="both", choices=("x", "y", "both"),
                        help="halve mesh spacing locally around the pads on the selected axes")
    parser.add_argument("--refine-levels", type=int, default=1, help="number of local halvings")
    parser.add_argument("--match-whole-bag-pe", action="store_true", help="use the YAM empty-bag PE material and contact skin")
    parser.add_argument("--pull-seconds", type=float, default=.5)
    parser.add_argument("--pull-speed", type=float, default=PULL_SPEED, help="vertical pull speed, in metres per second")
    parser.add_argument("--pull-force-target", type=float, help="force feedback target during pull, in newtons")
    parser.add_argument("--closed-gap", type=float, help="gap between the pads at the end of gathering, in metres")
    parser.add_argument("--hold-seconds", type=float, default=.06)
    parser.add_argument("--hold-downforce", type=float, help="relieve support compression to this level during hold")
    parser.add_argument("--grip-target", type=float, help="prefold comparison using total inner normal force, in newtons")
    parser.add_argument("--gather-seconds", type=float, help="duration of pressure-controlled gathering for flat matched PE")
    parser.add_argument("--gather-only", action="store_true", help="end after pressure-controlled gathering")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.cells < 4 or args.cells % 4:
        parser.error("cells must be a multiple of four")
    if args.refine_levels < 1 or (args.refine_levels > 1 and args.refine_contact is None):
        parser.error("additional refinement levels require selected contact axes")
    if not 0 <= args.slack < SIDE / 2 or not 0 < args.dt <= .001:
        parser.error("slack must be in [0, .03) m and dt in (0, .001] s")
    if args.prefold_width is not None and (not args.prefold or not 0 < args.prefold_width <= SIDE):
        parser.error("prefold width requires --prefold and must be in (0, .06] m")
    if args.prefold and not 0 < args.slack < (args.prefold_width or 2 * SIDE / args.cells):
        parser.error("prefold slack must be positive and below the unfolded ridge width")
    if args.match_whole_bag_pe and not args.prefold and args.slack >= .020:
        parser.error("slack must be below the .020 m of material reserved for the side ridges")
    if not np.isfinite(args.pull_seconds) or args.pull_seconds <= 0:
        parser.error("pull duration must be positive and finite")
    if not np.isfinite(args.hold_seconds) or args.hold_seconds <= 0:
        parser.error("hold duration must be positive and finite")
    if not np.isfinite(args.pull_speed) or args.pull_speed <= 0:
        parser.error("pull speed must be positive and finite")
    if args.pull_force_target is not None and (args.grip_target is None or
                                              not np.isfinite(args.pull_force_target) or args.pull_force_target <= 0):
        parser.error("pull force target must be positive and finite and needs --grip-target")
    initial_half = .005 if args.prefold else OPEN_HALF
    if args.closed_gap is not None and (
            not 0 < args.closed_gap < 2 * (initial_half - PAD_HALF_THICKNESS) or
            (args.prefold and args.grip_target is not None)):
        parser.error("closed gap must be between zero and the initial opening, without prefold grip feedback")
    if args.grip_target is not None and (not args.match_whole_bag_pe or not np.isfinite(args.grip_target) or
                                        args.grip_target <= 0 or (not args.prefold and
                                        (args.gather_seconds is None or args.closed_gap is None))):
        parser.error("grip target needs positive force and matched PE with a prefold or a controlled gather and gap")
    if args.hold_downforce is not None and (args.prefold or args.grip_target is None or
                                            not np.isfinite(args.hold_downforce) or args.hold_downforce < 0):
        parser.error("hold downforce requires flat-specimen grip feedback and a finite nonnegative target")
    if args.gather_seconds is not None and (args.prefold or not args.match_whole_bag_pe or
                                           not np.isfinite(args.gather_seconds) or args.gather_seconds <= 0):
        parser.error("gather duration must be positive, finite, and used with flat matched PE")
    if args.gather_only and args.gather_seconds is None:
        parser.error("gather-only requires pressure-controlled gathering")
    if args.friction is None:
        args.friction = .8 if args.match_whole_bag_pe else .5
    if not 0 <= args.friction <= 1 or not np.isfinite(args.press_force) or args.press_force <= 0:
        parser.error("friction must be in [0, 1] and press force must be positive")
    if not np.isfinite(args.pad_contact_time) or args.pad_contact_time < 2 * args.dt:
        parser.error("pad contact time must be at least twice the timestep")
    if not run(args):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
