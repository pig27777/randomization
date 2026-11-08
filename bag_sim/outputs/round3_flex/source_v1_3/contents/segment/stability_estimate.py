"""Explicit-step limit of the CPU bushing model (first version 2026-11-04, read-only).

Linearized: whole-object highest frequency from the bushing forces (finite differences of the generalized
force) and the mass matrix, at rest and at y / y+z compaction near and beyond the lock; dt <= 2(sqrt(z^2+1) - z)
/ omega for explicit damped springs under semi-implicit Euler, dt <= 2 / omega once the damping is implicit
native tendon damping. No contacts, gravity or native constraints (those are solved implicitly). Compared for
the current model (gains 0, native stops and damping) and the verified_50us model (explicit damping, compaction
gain 99, stop gains 1000).
Dynamic: native stops are measured instead. A single connection of each axis carries a couple about each
transverse axis, ramped over 0.2 s and held 0.2 s, that would bend it to 1.3 rad without stops. Pass: no failure,
settled (< 1 mm/s), and the 0.4 / 0.2 ms end angles agree within 1 deg. (A dead axial squeeze on an isolated
pair buckles elastically before reaching the compression lock, so compression stops are exercised by M2.)
Usage: python stability_estimate.py bendable|compressible [--dt 0.0004 0.0002] [--out DIR]
"""
import argparse, json, sys, math; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np, mujoco
import bushing as B, mj_adapter as A
from model import ANCHORS, build, load_table

def setup(name):
    k = ANCHORS[name]
    desc = build(k, omega_max=math.inf, table=load_table(), geometry=name)
    built = A.scene(desc, (0, 0, 0.5), [], table=False, gravity=False, mode="bushing")
    sim = A.Sim(built, [])
    return k, desc, sim

def free_dofs(sim):
    m = sim.m
    return [m.jnt_dofadr[m.body_jntadr[b]] + i for b in sim.seg_bodies for i in range(6)], \
           [m.jnt_qposadr[m.body_jntadr[b]] for b in sim.seg_bodies]

def compress(sim, desc, fy, fz):
    """Place segments so every y-joint (z-joint) is compressed by fy (fz) x its joint lock."""
    m, d = sim.m, sim.d
    jl = {}
    for j in desc.joints: jl.setdefault(j.axis, j.lock_comp)
    for i, s in enumerate(desc.segments):
        p = s.pos.copy()
        p[1] *= 1 - fy * jl.get(1, 0)
        p[2] *= 1 - fz * jl.get(2, 0)
        qa = m.jnt_qposadr[m.body_jntadr[sim.seg_bodies[i]]]
        d.qpos[qa:qa + 3] = np.array([0, 0, 0.5]) + p
        d.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
    d.qvel[:] = 0
    mujoco.mj_kinematics(m, d)

def gen_force(sim):
    m, d = sim.m, sim.d
    d.xfrc_applied[:] = 0
    sim.bushings.apply(m, d)
    q = np.zeros(m.nv)
    for b in sim.seg_bodies:
        mujoco.mj_applyFT(m, d, d.xfrc_applied[b, :3].copy(), d.xfrc_applied[b, 3:].copy(), d.xipos[b].copy(), b, q)
    d.xfrc_applied[:] = 0
    return q

def hessian(sim, h=1e-6):
    m, d = sim.m, sim.d
    dofs, _ = free_dofs(sim)
    q0 = d.qpos.copy(); n = len(dofs)
    def F(i, sgn):
        dq = np.zeros(m.nv); dq[dofs[i]] = sgn * h
        q = q0.copy(); mujoco.mj_integratePos(m, q, dq, 1.0)
        d.qpos[:] = q; d.qvel[:] = 0; mujoco.mj_forward(m, d)
        return gen_force(sim)[dofs]
    H = np.zeros((n, n))
    for i in range(n):
        H[:, i] = -(F(i, 1) - F(i, -1)) / (2 * h)
    d.qpos[:] = q0; mujoco.mj_forward(m, d)
    return 0.5 * (H + H.T)

def mass(sim):
    m, d = sim.m, sim.d
    mujoco.mj_forward(m, d)
    M = np.zeros((m.nv, m.nv)); mujoco.mj_fullM(m, d, M)
    dofs, _ = free_dofs(sim)
    return M[np.ix_(dofs, dofs)]

def omega_max(H, M):
    Li = np.linalg.inv(np.linalg.cholesky(M))
    w = np.linalg.eigvalsh(Li @ H @ Li.T)
    return math.sqrt(max(w.max(), 0))

S = lambda z: 2 * (math.sqrt(z * z + 1) - z)
STATES = [("rest", 0, 0), ("y at lock (0.98)", 0.98, 0), ("y beyond lock (1.05)", 1.05, 0),
          ("y,z at lock (0.98)", 0.98, 0.98), ("y,z beyond (1.05)", 1.05, 1.05)]


def linearized(name):
    k, desc, sim = setup(name)
    z = k.damping_ratio
    M = mass(sim)
    out = {}
    for label, gain, stop, s_max in (("current", 0.0, 0.0, 2.0), ("verified_50us", 99.0, 1000.0, S(z))):
        B.COMPACTION_BEND_GAIN, B.COMPRESSION_STOP_GAIN, B.ANGLE_STOP_GAIN = gain, stop, stop
        rows = []
        for state, fy, fz in STATES:
            compress(sim, desc, fy, fz)
            w = omega_max(hessian(sim), M)
            rows.append({"state": state, "omega": w, "dt_max_us": s_max / w * 1e6})
        out[label] = rows
        print(f"{name} {label} (gain {gain:.0f}, stop {stop:.0f}): "
              + "; ".join(f"{r['state']}: dt<={r['dt_max_us']:,.0f} us" for r in rows), flush=True)
    B.COMPACTION_BEND_GAIN, B.COMPRESSION_STOP_GAIN, B.ANGLE_STOP_GAIN = 0.0, 0.0, 0.0
    return out


def loaded(name, axis, bend, dt, ramp=0.2, hold=0.2):
    """Single connection under a couple about one transverse axis, ramped over 0.2 s and held 0.2 s, that
    would bend it to 1.3 rad without stops. Returns the extremes, the end state and the end speed."""
    from tests.test_bushing import pair
    A.BUSHING_DT = dt
    sim = pair(name, axis)
    j, d, b = sim.desc.joints[0], sim.d, sim.bushings
    bi = [a for a in range(3) if a != axis].index(bend)
    torque = 1.3 * j.lock_bend * j.k_bend[bi]
    a_body, b_body = sim.seg_bodies
    out = {"anchor": name, "dt": dt, "axis": axis, "bend": bend, "fail": None,
           "bend_ratio_max": 0.0, "compression_ratio_max": 0.0}
    n, n_ramp = round((ramp + hold) / dt), round(ramp / dt)
    tail = []
    for i in range(n):
        s_ = min(1.0, (i + 1) / n_ramp)
        d.xfrc_applied[:] = 0
        d.xfrc_applied[b_body, 3 + bend], d.xfrc_applied[a_body, 3 + bend] = s_ * torque, -s_ * torque
        sim.step()
        *_, u, phi = b.coordinates(d)
        bend_ratio = float(np.abs(phi[0, bend]) / b.limit[0, bend])
        comp_ratio = float(-u[0, axis] / (j.rest * j.lock_comp))
        out["bend_ratio_max"] = max(out["bend_ratio_max"], bend_ratio)
        out["compression_ratio_max"] = max(out["compression_ratio_max"], comp_ratio)
        if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))):
            out["fail"] = "non-finite state"
        else:
            out["fail"] = b.check(d)
        if out["fail"]:
            break
        if i >= n - round(.05 / dt):
            tail.append(float(np.linalg.norm(sim.seg_velocity() - sim.seg_velocity().mean(0), axis=1).max()))
    out.update({"bend_ratio_end": bend_ratio, "compression_ratio_end": comp_ratio,
                "speed_last_50ms": max(tail) if tail else None})
    return out


def dynamic(name, dts):
    rows = [loaded(name, axis, bend, dt) for dt in dts for axis in range(3) for bend in range(3) if bend != axis]
    ref = {(r["axis"], r["bend"]): r for r in rows if r["dt"] == min(dts)}
    for r in rows:
        q = ref[(r["axis"], r["bend"])]
        r["bend_diff_deg"] = abs(r["bend_ratio_end"] - q["bend_ratio_end"]) * math.degrees(1.0)
        r["pass"] = r["fail"] is None and r["speed_last_50ms"] < 1e-3 and r["bend_diff_deg"] < 1
        print(f"dt {r['dt']*1e6:.0f} us {name} axis {r['axis']} bend about {r['bend']}: pass {r['pass']}, fail {r['fail']}, "
              f"bend max/end {r['bend_ratio_max']:.4f}/{r['bend_ratio_end']:.4f}, compression max/end "
              f"{r['compression_ratio_max']:.4f}/{r['compression_ratio_end']:.4f}, end speed {r['speed_last_50ms']} m/s, "
              f"diff {r['bend_diff_deg']:.3f} deg", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("anchor", choices=("bendable", "compressible"))
    ap.add_argument("--dt", type=float, nargs="+", default=[0.0004, 0.0002])
    ap.add_argument("--out", help="directory for stability_<anchor>.json")
    args = ap.parse_args()
    result = {"linearized": linearized(args.anchor), "dynamic": dynamic(args.anchor, args.dt)}
    result["pass"] = (all(r["dt_max_us"] >= 400 for r in result["linearized"]["current"])
                      and all(r["pass"] for r in result["dynamic"]))
    print(f"{args.anchor}: pass={result['pass']}", flush=True)
    if args.out:
        out = __import__("pathlib").Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"stability_{args.anchor}.json").write_text(json.dumps(result, indent=2) + "\n")
