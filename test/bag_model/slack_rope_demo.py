"""Exploratory demo: a bagged package as 3 jointed boxes hung from a grip point by slack ropes.

No bag body, no film, no arm/fingers. The grip is a mocap body moved by script. Each rope is a
spatial tendon with an upper length limit (taut-only). Not a formal model; results are not
acceptance evidence.

Rope rule: each segment has 2 ropes, from the grip point to the midpoints of its two bottom long
edges (y = +-W/2, z = 0). Rope i length = distance from its anchor to the grip's start point raised
by the slack s, so a flat package lifts off with every rope going taut together once the grip has
risen s. s = 0 means each rope is taut at the start.

Usage (repository root):
  MUJOCO_GL=egl python -B test/bag_model/slack_rope_demo.py --output test/runs/<date>/bag/<run-id>
"""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")

L, W, H, MASS = 0.30, 0.20, 0.04, 0.5
NSEG = 3
DT = 0.001
ROPE_SOLREF = "0.005 1"          # rope stretch ~1 mm under the package weight
# hinge stiffness [N m/rad], damping [N m s/rad]; None = locked (one rigid body)
CONTENTS = {"hard": None, "bendable": (0.25, 0.007), "pieces": (0.002, 0.005)}
CASES = [  # id, contents, slack [m], grip offset along x [m]
    (1, "hard", 0.0, 0.0), (2, "hard", 0.06, 0.0), (3, "bendable", 0.0, 0.0), (4, "bendable", 0.06, 0.0),
    (5, "pieces", 0.06, 0.0), (6, "hard", 0.06, 0.06), (7, "bendable", 0.06, 0.06), (8, "bendable", 0.0, 0.06),
]
# schedule [s]: settle, lift 0.3, hold, move 0.5 in x, hold, lower 0.3, release
T_SETTLE, T_LIFT, T_HOLD1, T_MOVE, T_HOLD2, T_LOWER, T_REST = 0.5, 2.0, 2.0, 2.0, 5.0, 2.0, 3.0
LIFT, MOVE = 0.3, 0.5
FPS, VW, VH = 30, 480, 360
RATE = f"1000/{round(1 / (FPS * DT))}" if DT == 0.001 else str(FPS)   # true frame rate: one frame every 33 steps
# VP9 WebM: the app viewer and Chromium cannot decode H.264
VP9 = ["-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p", "-crf", "32", "-b:v", "0", "-row-mt", "1"]


def phases():
    t, out = 0.0, {}
    for name, dur in [("settle", T_SETTLE), ("lift", T_LIFT), ("hold1", T_HOLD1), ("move", T_MOVE),
                      ("hold2", T_HOLD2), ("lower", T_LOWER), ("rest", T_REST)]:
        out[name] = (t, t + dur)
        t += dur
    return out


PH = phases()
T_END = PH["rest"][1]


def smooth(u):
    u = min(max(u, 0.0), 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * u)


def grip_pos(t, p0):
    dz = LIFT * (smooth((t - PH["lift"][0]) / T_LIFT) - smooth((t - PH["lower"][0]) / T_LOWER))
    dx = MOVE * smooth((t - PH["move"][0]) / T_MOVE)
    return p0 + np.array([dx, 0.0, dz])


def seg_x(i):
    return (i - 1) * L / NSEG


def anchors():
    """World anchor points at t = 0 (package bottom at z = 0, centered at the origin)."""
    return [(i, s, np.array([seg_x(i), s * W / 2, 0.0])) for i in range(NSEG) for s in (-1, 1)]


def build_xml(contents, slack, offset):
    seg = L / NSEG
    m_seg = MASS / NSEG
    half = f"{seg / 2} {W / 2} {H / 2}"
    stiff = CONTENTS[contents]
    grip0 = np.array([offset, 0.0, H])
    geom = (f'type="box" size="{half}" mass="{m_seg}" contype="1" conaffinity="0" condim="3" '
            f'friction="0.8 0.005 0.0001"')
    sites = lambda i: "".join(
        f'<site name="a{i}{"np"[s > 0]}" pos="0 {s * W / 2} {-H / 2}" size="0.004"/>' for s in (-1, 1))

    def side(i, sign):
        rgba = ".85 .55 .25 1" if i != 1 else ".25 .55 .85 1"
        if stiff is None:   # locked: extra geoms on the same body
            return f'<body name="s{i}" pos="{seg_x(i)} 0 0"><geom name="g{i}" {geom} rgba="{rgba}"/>{sites(i)}</body>'
        k, c = stiff
        hp = f'pos="{-sign * seg / 2} 0 0"'
        return (f'<body name="s{i}" pos="{seg_x(i)} 0 0">'
                f'<joint name="bend{i}" type="hinge" axis="0 1 0" {hp} stiffness="{k}" damping="{c}"/>'
                f'<joint name="twist{i}" type="hinge" axis="1 0 0" {hp} stiffness="{k}" damping="{c}"/>'
                f'<geom name="g{i}" {geom} rgba="{rgba}"/>{sites(i)}</body>')

    lengths = {}
    tendons = []
    for i, s, a in anchors():
        name = f"r{i}{'np'[s > 0]}"
        lengths[name] = float(np.linalg.norm(grip0 + [0, 0, slack] - a))
        tendons.append(f'<spatial name="{name}" limited="true" range="0 {lengths[name]:.6f}" '
                       f'solreflimit="{ROPE_SOLREF}" width="0.0015" rgba=".1 .1 .1 1">'
                       f'<site site="grip"/><site site="a{i}{"np"[s > 0]}"/></spatial>')
    xml = f"""<mujoco model="slack_rope_demo">
<option timestep="{DT}" integrator="implicitfast" solver="Newton" iterations="50" tolerance="1e-8" cone="elliptic"/>
<visual><global offwidth="{VW}" offheight="{VH}"/><quality shadowsize="0"/><headlight ambient=".45 .45 .45" diffuse=".5 .5 .5"/></visual>
<worldbody>
<light pos="0.2 -1 1.5" dir="0 0.6 -1"/>
<geom name="table" type="plane" size="2 1 0.01" rgba=".8 .8 .75 1" contype="0" conaffinity="1"/>
<camera name="side" pos="0.30 -1.2 0.22" xyaxes="1 0 0 0 0 1" fovy="45"/>
<body name="gripper" mocap="true" pos="{grip0[0]} 0 {grip0[2]}">
  <geom type="sphere" size="0.008" rgba=".8 .1 .1 1" contype="0" conaffinity="0"/><site name="grip" size="0.002"/>
</body>
<body name="s1" pos="0 0 {H / 2}"><freejoint name="root"/>
  <geom name="g1" {geom} rgba=".25 .55 .85 1"/>{sites(1)}
  {side(0, -1)}{side(2, 1)}
</body>
</worldbody>
<tendon>{"".join(tendons)}</tendon>
</mujoco>"""
    return xml, lengths


class Run:
    def __init__(self, case, render=True):
        self.cid, self.contents, self.slack, self.offset = case
        self.xml, self.lengths = build_xml(self.contents, self.slack, self.offset)
        self.m = mujoco.MjModel.from_xml_string(self.xml)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.bodies = [m.body(f"s{i}").id for i in range(NSEG)]
        self.geoms = {m.geom(f"g{i}").id for i in range(NSEG)}
        self.table = m.geom("table").id
        self.p0 = m.body_pos[m.body("gripper").id].copy()
        self.renderer = mujoco.Renderer(m, VH, VW) if render else None
        self.cam = m.camera("side").id

    def contact_count(self):
        d = self.d
        n = 0
        for j in range(d.ncon):
            g1, g2 = d.contact[j].geom1, d.contact[j].geom2
            if (g1 == self.table and g2 in self.geoms) or (g2 == self.table and g1 in self.geoms):
                n += 1
        return n

    def run(self):
        m, d = self.m, self.d
        mujoco.mj_forward(m, d)
        nsteps = int(round(T_END / DT))
        frame_every = int(round(1 / (FPS * DT)))
        rec = {k: [] for k in ("t", "grip", "com", "R", "lin", "ang", "ncon", "taut")}
        frames, step_time, bad, solver_capped = [], 0.0, None, 0
        released = False
        for k in range(nsteps):
            t = k * DT
            if not released and t >= PH["rest"][0]:
                m.tendon_limited[:] = 0          # release: ropes no longer act
                released = True
            d.mocap_pos[0] = grip_pos(t + DT, self.p0)
            t0 = time.perf_counter()
            mujoco.mj_step(m, d)
            step_time += time.perf_counter() - t0
            if d.solver_niter[0] >= m.opt.iterations:
                solver_capped += 1
            if bad is None and not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
                bad = round(t, 4)
                break
            if k % 5 == 0:
                rec["t"].append(t + DT)
                rec["grip"].append(d.mocap_pos[0].copy())
                rec["com"].append(d.subtree_com[self.bodies[1]].copy())
                rec["R"].append(np.array([d.xmat[b].reshape(3, 3) for b in self.bodies]))
                vel = np.zeros(6)
                lin, ang = [], []
                for b in self.bodies:
                    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, vel, 0)
                    ang.append(np.linalg.norm(vel[:3]))
                    lin.append(np.linalg.norm(vel[3:]))
                rec["lin"].append(max(lin))
                rec["ang"].append(max(ang))
                rec["ncon"].append(self.contact_count())
                rec["taut"].append(int(sum(d.ten_length[j] >= m.tendon_range[j, 1] - 1e-4
                                           for j in range(m.ntendon))) if not released else 0)
            if self.renderer is not None and k % frame_every == 0:
                self.renderer.update_scene(d, camera=self.cam)
                frames.append(self.renderer.render().copy())
        if self.renderer is not None:
            self.renderer.close()           # several open EGL renderers corrupt each other's frames
        warnings = {mujoco.mjtWarning(i).name: int(d.warning[i].number) for i in range(len(d.warning))
                    if d.warning[i].number}
        rec = {k: np.array(v) for k, v in rec.items()}
        return rec, frames, dict(steps=k + 1, step_time_s=step_time, nan_at=bad, warnings=warnings,
                                 solver_capped_steps=solver_capped)


def droop_deg(R):
    """Side-segment droop relative to the middle segment, about y; + = outer end lower."""
    out = []
    for i, sign in ((0, -1), (2, 1)):
        x = R[1].T @ R[i][:, 0]          # side x-axis in the middle frame
        out.append(float(np.degrees(-np.arctan2(sign * x[2], x[0]))))
    return out


def pitch_deg(R):
    x = R[:, 0]
    return float(np.degrees(np.arctan2(x[2], np.hypot(x[0], x[1]))))


def metrics(rec, info):
    t = rec["t"]
    win = lambda a, b: (t >= a) & (t < b)
    out = {}
    # lift-off: first time after lift start with no table contact held for 50 ms
    lo = None
    idx = np.where(t >= PH["lift"][0])[0]
    for j in idx:
        k = int(round(0.05 / (5 * DT)))
        if (rec["ncon"][j:j + k] == 0).all() and j + k <= len(t):
            lo = j
            break
    out["liftoff_grip_rise_mm"] = None if lo is None else round(1e3 * (rec["grip"][lo, 2] - rec["grip"][0, 2]), 1)
    out["liftoff_time_s"] = None if lo is None else round(float(t[lo]), 3)
    # steady hang: last 0.5 s of hold1
    h = win(PH["hold1"][1] - 0.5, PH["hold1"][1])
    Rh = rec["R"][h]
    out["hang_pitch_mid_deg"] = round(float(np.mean([pitch_deg(R[1]) for R in Rh])), 1)
    ends = [R for R in Rh]
    out["hang_droop_deg"] = [round(float(v), 1) for v in np.mean([droop_deg(R) for R in ends], axis=0)]
    out["hang_taut_ropes"] = int(np.median(rec["taut"][h]))
    out["hang_still"] = bool(rec["lin"][h].max() < 0.01)
    # swing after the horizontal move stops: x(com) - x(grip)
    s = win(PH["hold2"][0], PH["hold2"][1])
    ts, sx = t[s] - PH["hold2"][0], rec["com"][s, 0] - rec["grip"][s, 0]
    sx = sx - sx[-int(1.0 / (5 * DT)):].mean()
    peaks = [j for j in range(1, len(sx) - 1) if abs(sx[j]) >= abs(sx[j - 1]) and abs(sx[j]) > abs(sx[j + 1])]
    out["swing_amp0_mm"] = round(1e3 * abs(sx[peaks[0]]), 1) if peaks else 0.0
    half = None
    if peaks and abs(sx[peaks[0]]) > 1e-3:
        a0 = abs(sx[peaks[0]])
        later = [j for j in peaks if abs(sx[j]) <= a0 / 2]
        # envelope: first peak after which no peak exceeds half
        for j in peaks:
            if all(abs(sx[p]) <= a0 / 2 for p in peaks if p >= j):
                half = round(float(ts[j]), 2)
                break
    out["swing_half_time_s"] = half
    out["swing_amp_end_mm"] = round(1e3 * float(np.abs(sx[-int(0.5 / (5 * DT)):]).max()), 1)
    # rest after release
    r = win(T_END - 0.5, T_END + 1)
    out["rest_still"] = bool(rec["lin"][r].max() < 0.005 and rec["ang"][r].max() < 0.05 and rec["ncon"][r].min() > 0)
    out["rest_mid_pitch_deg"] = round(float(np.mean([pitch_deg(R[1]) for R in rec["R"][r]])), 1)
    out["rest_droop_deg"] = [round(float(v), 1) for v in np.mean([droop_deg(R) for R in rec["R"][r]], axis=0)]
    out["min_seg_center_z_mm"] = round(1e3 * float(rec["com"][:, 2].min()), 1)
    out["nan_at_s"] = info["nan_at"]
    out["warnings"] = info["warnings"]
    out["solver_capped_steps"] = info["solver_capped_steps"]
    out["mean_step_us"] = round(1e6 * info["step_time_s"] / info["steps"], 2)
    return out


def write_video(frames, path):
    p = subprocess.Popen([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                          "-pixel_format", "rgb24", "-video_size", f"{VW}x{VH}", "-framerate", RATE,
                          "-i", "-", *VP9, str(path)],
                         stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    if p.wait():
        raise RuntimeError(f"ffmpeg failed for {path}")


def overview(paths, labels, out):
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y"]
    for p in paths:
        cmd += ["-i", str(p)]
    f = [f"[{i}:v]drawtext=fontfile={font}:text='{lab}':fontsize=18:fontcolor=white:x=6:y=6[v{i}]"
         for i, lab in enumerate(labels)]
    layout = "|".join(f"{(i % 4) * VW}_{(i // 4) * VH}" for i in range(8))
    f.append("".join(f"[v{i}]" for i in range(8)) + f"xstack=inputs=8:layout={layout}[o]")
    cmd += ["-filter_complex", ";".join(f), "-map", "[o]", *VP9, str(out)]
    subprocess.run(cmd, check=True)


def v13_step_us(nsteps=2000):
    """Mean CPU step time of the frozen v1.3 bendable anchor resting on the table (Sim.step, bushing mode)."""
    sys.path[:0] = [str(ROOT / "contents/segment")]
    import model as SM
    import mj_adapter as A
    desc = SM.build(SM.ANCHORS["bendable"])
    built = A.scene(desc, (0.0, 0.0, desc.kappa.size[2] / 2 + 1e-4))
    sim = A.Sim(built)
    for _ in range(200):
        sim.step()
    t0 = time.perf_counter()
    for _ in range(nsteps):
        sim.step()
    dt = time.perf_counter() - t0
    return dict(model="v1.3 bendable anchor (6x3x2 segments), resting on table, Sim.step",
                timestep_s=float(sim.m.opt.timestep), nbody=int(sim.m.nbody), nv=int(sim.m.nv),
                ntendon=int(sim.m.ntendon), steps=nsteps, mean_step_us=round(1e6 * dt / nsteps, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    results, videos, labels = [], [], []
    for case in CASES:
        run = Run(case, render=not args.no_video)
        rec, frames, info = run.run()
        res = dict(case=case[0], contents=case[1], slack_mm=round(1e3 * case[2]), offset_mm=round(1e3 * case[3]),
                   rope_lengths_m={k: round(v, 4) for k, v in run.lengths.items()}, **metrics(rec, info))
        results.append(res)
        print(json.dumps(res), flush=True)
        np.savez_compressed(out / f"case{case[0]}_traj.npz", **rec)
        if frames:
            path = out / f"case{case[0]}.webm"
            write_video(frames, path)
            videos.append(path)
            labels.append(f"{case[0]} {case[1]} slack {res['slack_mm']} off {res['offset_mm']}")
        if case[0] == 1:
            (out / "case1_model.xml").write_text(run.xml)
    if videos:
        overview(videos, labels, out / "overview_2x4.webm")
    v13 = v13_step_us()
    summary = dict(host=platform.node(), cpu=platform.processor(), mujoco=mujoco.__version__,
                   dt=DT, schedule_s=PH, cases=results, v13=v13,
                   demo_mean_step_us=round(float(np.mean([r["mean_step_us"] for r in results])), 2))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(dict(v13=v13, demo_mean_step_us=summary["demo_mean_step_us"])))


if __name__ == "__main__":
    main()
