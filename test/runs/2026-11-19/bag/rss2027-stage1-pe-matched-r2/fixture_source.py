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
PAD_BOTTOM = 0.020
PHASES = (("settle", .08), ("press", .45), ("gather", .24), ("hold", .06), ("pull", .50))


def coupon_grid(cells, slack, prefold, radius, *, prefold_width=None, flat_waist=False, refine_contact=False):
    """Keep the initial x-edge lengths exactly equal to the flat reference."""
    axis = np.linspace(-SIDE / 2, SIDE / 2, cells + 1)
    width = (prefold_width or 2 * SIDE / cells) if prefold else SIDE / 2
    if flat_waist:
        ridges = ((-.030, -.025, -.020), (.020, .025, .030))
    else:
        ridges = ((-width / 2, 0., width / 2),)
    x_axis = np.unique(np.round(np.r_[axis, np.ravel(ridges)], 12))
    y_axis = axis
    if refine_contact:
        x_mids = (x_axis[:-1] + x_axis[1:]) / 2
        y_mids = (axis[:-1] + axis[1:]) / 2
        x_axis = np.sort(np.r_[x_axis, x_mids[np.abs(x_mids) <= OPEN_HALF + .004]])
        y_axis = np.sort(np.r_[axis, y_mids[np.abs(y_mids) <= .010]])
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
        flat_waist=args.match_whole_bag_pe and not args.prefold, refine_contact=args.refine_contact)
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
    open_half = .005 if args.prefold else OPEN_HALF
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


def pad_reactions(m, d, pad_ids):
    force = np.zeros((2, 3))
    inner = np.zeros(2)
    active = np.zeros(2, dtype=int)
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
                inner[pad] += local[0] * abs(frame[0, 0])
    return force, inner, active


def aperture_lengths(nodes, reference_lengths, half, z):
    """Clip the central material line to the 3D volume between the pads."""
    half_gap = half - .004
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
    phases = (("gather", .012), ("hold", .06)) if args.prefold else PHASES[:-1]
    phases = (*phases, ("pull", args.pull_seconds))
    row = np.flatnonzero(np.isclose(ref[:, 1], 0., atol=1e-12))
    reference_lengths = np.linalg.norm(np.diff(ref[row], axis=0), axis=1)
    shape = dict(vertices=len(ref), triangles=len(faces),
                 nominal_local_pitch_m=SIDE / args.cells / (2 if args.refine_contact else 1),
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
    previous = pad_origins
    marker = np.flatnonzero((abs(ref[:, 0]) < 1e-9) & (abs(ref[:, 1]) < .006))
    press = 0.
    peak_penetration, failure = 0., None
    press_band_s = peak_press_band_s = peak_press_N = 0.
    minimum_pad_height = z - PAD_BOTTOM
    sample_steps = max(1, round(.001 / args.dt))
    for phase, duration in phases[:1] if args.smoke else phases:
        nsteps = 2 if args.smoke else round(duration / args.dt)
        for step in range(nsteps):
            half = open_half
            if phase == "gather":
                half -= (open_half - CLOSED_HALF) * (step + 1) / nsteps
            elif phase in ("hold", "pull"):
                half = CLOSED_HALF
            if not args.prefold and phase in ("press", "gather"):
                z_rate = np.clip(.02 * (press - args.press_force), -.02, .02)
                z = max(PAD_BOTTOM + .00002, z + args.dt * z_rate)
            elif phase == "pull":
                z += .025 * args.dt
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
            if sampled or (args.match_whole_bag_pe and phase == "press"):
                force, inner, active = pad_reactions(m, d, pad_ids)
                press = float(force[:, 2].sum())
                if phase == "press" and args.match_whole_bag_pe:
                    peak_press_N = max(peak_press_N, press)
                    in_band = .8 * args.press_force <= press <= 1.2 * args.press_force
                    press_band_s = press_band_s + args.dt if in_band else 0.
                    peak_press_band_s = max(peak_press_band_s, press_band_s)
            if sampled:
                marker_z = float(d.flexvert_xpos[marker, 2].mean())
                contained = aperture_lengths(d.flexvert_xpos[row], reference_lengths, half, z)
                rows.append((phase, t_eval, open_half - half, z, *force.ravel(), *inner,
                             -press, penetration, marker_z, z - marker_z, *contained, *active))
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
                   "left_active", "right_active")
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
    release, slip = [], []
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
    valid = failure is None and peak_penetration <= .001
    summary = dict(numerically_valid=valid, failure=failure, peak_penetration_m=peak_penetration,
                   warnings=d.warning.number.tolist(), bilateral_pull_samples=int(bilateral.sum()),
                   bilateral_loaded_pull_samples=int(loaded.sum()), longest_bilateral_load_s=longest_load,
                   peak_pull_N=float(tension.max(initial=0.)),
                   press_peak_N=peak_press_N, longest_press_band_s=peak_press_band_s,
                   minimum_pad_height_m=minimum_pad_height,
                   peak_aperture_material_length_m=max((row[16] for row in rows), default=0.),
                   peak_fold_excess_length_m=max((row[18] for row in rows), default=0.),
                   capture_t_s=float(time_s[capture[0]]) if capture else None,
                   release_t_s=float(time_s[release[0]]) if release else None,
                   slip_candidate_t_s=float(time_s[slip[0]]) if slip else None,
                   slip_candidate_force_N=float(tension[slip[0]]) if slip else None,
                   simulated_s=float(d.time), wall_s=time.perf_counter() - started,
                   smoke_only=args.smoke, material_parameters_identified=False)
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
    parser.add_argument("--refine-contact", action="store_true", help="halve mesh spacing locally around the pads")
    parser.add_argument("--match-whole-bag-pe", action="store_true", help="use the YAM empty-bag PE material and contact skin")
    parser.add_argument("--pull-seconds", type=float, default=.5)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.cells < 4 or args.cells % 4:
        parser.error("cells must be a multiple of four")
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
