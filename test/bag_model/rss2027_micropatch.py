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
from bag_sim.physical_parcel import material_parameters, vec  # noqa: E402

SIDE = 0.060
RADIUS = 0.0001
OPEN_HALF = 0.0115
CLOSED_HALF = 0.0042
PAD_BOTTOM = 0.020
PHASES = (("settle", .08), ("press", .45), ("gather", .24), ("hold", .06), ("pull", .50))


def coupon_grid(cells, slack, prefold):
    """Keep the initial x-edge lengths exactly equal to the flat reference."""
    axis = np.linspace(-SIDE / 2, SIDE / 2, cells + 1)
    x, y = np.meshgrid(axis, axis)
    length = SIDE / cells
    # The middle half of the patch starts as a symmetric tent. Solving for
    # the facet angle sets the exact prescribed difference in projected span.
    slope = np.zeros(cells)
    middle = cells // 2
    legs = 1 if prefold else cells // 4
    angle = np.arccos(1 - slack / (2 * legs * length))
    slope[middle - legs:middle] = angle
    slope[middle:middle + legs] = -angle
    current_x = np.r_[0., np.cumsum(length * np.cos(slope))]
    current_x -= current_x[-1] / 2
    current_z = RADIUS + np.r_[0., np.cumsum(length * np.sin(slope))]
    ref = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, RADIUS)))
    current = np.column_stack((np.tile(current_x, cells + 1), y.ravel(), np.tile(current_z, cells + 1)))
    faces = []
    for row in range(cells):
        for col in range(cells):
            a = row * (cells + 1) + col
            b = a + cells + 1
            faces.extend(((a, a + 1, b + 1), (a, b + 1, b)) if (row + col) % 2 == 0 else
                         ((a, a + 1, b), (a + 1, b + 1, b)))
    edges = np.arange(x.size).reshape(x.shape)[:, [0, -1]].ravel()
    return ref, current, np.asarray(faces), edges


def build(cells, slack, dt, friction, prefold):
    young, thickness, density, poisson = 255e6, 50.8e-6, 924., .45
    material = material_parameters(young, thickness, density, poisson)
    ref, current, faces, edges = coupon_grid(cells, slack, prefold)
    model = ET.fromstring(f"""
      <mujoco model="rss2027_micropatch">
        <option timestep="{dt}" integrator="discrete" solver="Newton"
          iterations="150" cone="elliptic" tolerance="1e-8"/>
        <asset/>
        <worldbody>
          <geom name="support" type="plane" size=".08 .08 .01" contype="1"
            conaffinity="2" priority="1" friction=".3 .005 .0001"
            solref=".0005 1" solimp=".999 .999 .001"/>
          <flexcomp name="coupon" type="direct" dim="2" mass="{material['areal_mass'] * SIDE**2}"
            radius="{RADIUS}" point="{vec(ref)}" element="{' '.join(map(str, faces.ravel()))}">
            <elasticity young="{young}" poisson="{poisson}" thickness="{thickness}"
              damping=".004" elastic2d="both"/>
            <contact contype="2" conaffinity="7" condim="3" selfcollide="auto"
              friction=".5 .005 .0001" solref=".001 1" solimp=".999 .999 .001"/>
          </flexcomp>
        </worldbody>
        <equality/>
      </mujoco>
    """)
    start_z = PAD_BOTTOM + (.0003 if prefold else current[:, 2].max() + .002)
    open_half = .005 if prefold else OPEN_HALF
    for name, sign in (("left", -1), ("right", 1)):
        tip = ET.fromstring(fingertip_xml(name, 0, (sign * open_half, 0, start_z)))
        for part in tip.findall("geom"):
            part.attrib.update(conaffinity="2", condim="3", priority="1", friction=f"{friction} .005 .0001",
                               solref=".001 1", solimp=".999 .999 .001")
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
    m.eq_data[:, 3:6] = current[edges]
    bodies = m.flex_vertbodyid
    qpos = m.jnt_qposadr[m.body_jntadr[bodies, None] + np.arange(3)]
    d.qpos[qpos] = current - ref
    mujoco.mj_forward(m, d)
    if not np.allclose(d.flexvert_xpos, current) or np.max(np.abs(d.efc_pos[:3 * len(edges)])) > 1e-9:
        raise RuntimeError("Coupon initialization does not match its prescribed anchor positions")
    material.update(young_pa=young, thickness_m=thickness, density_kg_m3=density,
                    pad_friction=friction, self_friction=.5, support_friction=.3, contact_radius_m=RADIUS)
    return xml, m, d, material, ref, faces


def pad_reactions(m, d, pad_ids):
    force = np.zeros((2, 3))
    inner = np.zeros(2)
    local = np.zeros(6)
    for index, contact in enumerate(d.contact):
        if contact.efc_address < 0 or 0 not in contact.flex:
            continue
        for slot, geom_id in enumerate(contact.geom):
            if geom_id not in pad_ids:
                continue
            pad = pad_ids[geom_id]
            mujoco.mj_contactForce(m, d, index, local)
            frame = contact.frame.reshape(3, 3)
            sign = 2 * slot - 1
            force[pad] += sign * (frame.T @ local[:3])
            # Separate inward-facing pinch contacts from downward press contacts.
            if (2 * pad - 1) * sign * frame[0, 0] > .7:
                inner[pad] += local[0] * abs(frame[0, 0])
    return force, inner


def run(args):
    start = time.perf_counter()
    xml, m, d, material, ref, faces = build(args.cells, args.slack, args.dt, args.friction, args.prefold)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "model.xml").write_text(xml)
    mujoco.mj_saveModel(m, str(args.output / "model.mjb"), None)
    bodies = [m.body(name).id for name in ("left", "right")]
    pad_joints = m.body_jntadr[bodies, None] + np.arange(3)
    pad_qpos = m.jnt_qposadr[pad_joints]
    pad_dofs = m.jnt_dofadr[pad_joints]
    pad_origins = m.body_pos[bodies].copy()
    pad_ids = {int(geom): pad for pad, body in enumerate(bodies) for geom in np.flatnonzero(m.geom_bodyid == body)}
    phases = (("gather", .012), *PHASES[-2:]) if args.prefold else PHASES
    config = vars(args) | {"output": str(args.output), "material": material, "phases_s": phases,
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
            t_eval = float(d.time)
            mujoco.mj_step(m, d)
            penetration = max(0., -float(d.contact.dist.min())) if d.ncon else 0.
            peak_penetration = max(penetration, peak_penetration)
            if step % sample_steps == 0 or step + 1 == nsteps:
                force, inner = pad_reactions(m, d, pad_ids)
                press = float(force[:, 2].sum())
                marker_z = float(d.flexvert_xpos[marker, 2].mean())
                rows.append((phase, t_eval, open_half - half, z, *force.ravel(), *inner,
                             -press, penetration, marker_z, z - marker_z))
                snapshots.append(d.flexvert_xpos.copy())
            if d.warning.number.sum() or not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
                failure = f"Unhealthy simulation in {phase} at {t_eval:.6g} s"
                break
        if failure:
            break
    with (args.output / "curve.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("phase", "t_s", "gather_per_pad_m", "pad_z_m", "left_fx_N", "left_fy_N", "left_fz_N",
                         "right_fx_N", "right_fy_N", "right_fz_N", "left_inner_N", "right_inner_N",
                         "signed_pull_N", "penetration_m", "marker_z_m", "pad_minus_marker_z_m"))
        writer.writerows(rows)
    np.savez_compressed(args.output / "coupon_nodes.npz", reference_m=ref, faces=faces, current_m=snapshots)
    pull = np.asarray([row[1:] for row in rows if row[0] == "pull"]).reshape(-1, 15)
    bilateral = (pull[:, 9:11] > .001).all(axis=1)
    loaded = bilateral & (pull[:, 11] > .001)
    transitions = np.flatnonzero(np.diff(np.r_[False, loaded, False]))
    longest_load = max((pull[end - 1, 0] - pull[begin, 0] for begin, end in transitions.reshape(-1, 2)),
                       default=0.)
    valid = failure is None and peak_penetration <= .001
    summary = dict(numerically_valid=valid, failure=failure, peak_penetration_m=peak_penetration,
                   warnings=d.warning.number.tolist(), bilateral_pull_samples=int(bilateral.sum()),
                   bilateral_loaded_pull_samples=int(loaded.sum()), longest_bilateral_load_s=longest_load,
                   peak_pull_N=float(pull[:, 11].max(initial=0.)),
                   simulated_s=float(d.time), wall_s=time.perf_counter() - start,
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
    parser.add_argument("--friction", type=float, default=.5)
    parser.add_argument("--press-force", type=float, default=.5)
    parser.add_argument("--prefold", action="store_true", help="test pulling an initially folded coupon")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.cells < 4 or args.cells % 4:
        parser.error("cells must be a multiple of four")
    if not 0 <= args.slack < SIDE / 2 or not 0 < args.dt <= .001:
        parser.error("slack must be in [0, .03) m and dt in (0, .001] s")
    if args.prefold and not 0 < args.slack < 2 * SIDE / args.cells:
        parser.error("prefold slack must be positive and below the two central edge lengths")
    if not 0 <= args.friction <= 1 or not np.isfinite(args.press_force) or args.press_force <= 0:
        parser.error("friction must be in [0, 1] and press force must be positive")
    if not run(args):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
