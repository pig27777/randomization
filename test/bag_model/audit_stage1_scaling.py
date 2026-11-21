"""Small Stage 1 film coupon tests using its MuJoCo membrane and hinge law."""
import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bag_sim.physical_parcel import _topology, dihedrals, material_parameters, vec

E, RHO, NU = 4e8, 920., .30
LENGTH, WIDTH = .020, .004
ELASTIC_DT, CONTACT_DT, RADIUS, MU = 4e-6, 25e-6, .00025, .45


def coupon(nx, ny, length, width, z=0.):
    xyz = np.array([[x, y, z] for x in np.linspace(0., length, nx)
                    for y in np.linspace(-width / 2., width / 2., ny)])
    tris = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a = i * ny + j
            tris.extend(((a, a + ny, a + ny + 1), (a, a + ny + 1, a + 1)))
    return xyz, np.array(tris)


def model(points, faces, thickness, fixed=(), gravity=0., contact_xml="", dt=CONTACT_DT):
    material = material_parameters(E, thickness, RHO, NU)
    triangle = points[faces]
    area = .5 * np.linalg.norm(np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]), axis=1)
    masses = np.zeros(len(points))
    np.add.at(masses, faces.ravel(), np.repeat(area * material["areal_mass"] / 3., 3))
    pins = f'<pin id="{vec(fixed)}"/>' if len(fixed) else ""
    xml = f"""<mujoco model="Stage 1 PE film coupon">
      <option timestep="{dt}" integrator="discrete" solver="Newton" iterations="120" tolerance="1e-10"
        cone="elliptic" gravity="0 0 {gravity}"/>
      <worldbody>
        <flexcomp name="film" type="direct" dim="2" point="{vec(points)}" element="{vec(faces)}"
                  mass="{sum(masses)}" radius="{RADIUS}">
          <elasticity young="{E}" poisson="{NU}" thickness="{thickness}" damping=".0001" elastic2d="stretch"/>
          <contact contype="2" conaffinity="1" condim="3" friction="{MU} .005 .0001"
                   solref=".001 1" solimp=".999 .999 .001" selfcollide="none"/>
          {pins}
        </flexcomp>
        {contact_xml}
      </worldbody>
    </mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    movers = np.setdiff1d(np.arange(len(points)), fixed).astype(int)
    bodies = [m.body(f"film_{i}").id for i in movers]
    dofs = m.body_dofadr[bodies, None] + np.arange(3)
    m.body_mass[bodies] = masses[movers]
    mujoco.mj_setConst(m, mujoco.MjData(m))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d, (movers, dofs, masses), material


def cantilever(thickness, load):
    nx, ny = 9, 3
    points, faces = coupon(nx, ny, LENGTH, WIDTH)
    m, d, (movers, dofs, masses), material = model(points, faces, thickness, np.arange(2 * ny),
                                                  gravity=-9.81 if load == "weight" else 0., dt=ELASTIC_DT)
    _, _, hinges, weights, *_ = _topology(points, faces)
    rest, gradient = dihedrals(points, hinges)
    stiffness = material["bending"] * weights
    inverse = np.zeros(len(points))
    inverse[movers] = 1. / masses[movers]
    inverse_inertia = (gradient**2 * inverse[hinges, None]).sum(axis=(1, 2))
    damping = np.zeros_like(stiffness)
    free = inverse_inertia > 0.
    damping[free] = .20 * np.sqrt(stiffness[free] / inverse_inertia[free])
    assert ELASTIC_DT * np.sqrt((stiffness * inverse_inertia).max()) <= .2
    m.dof_damping[dofs] = masses[movers, None] * 45.
    end = dofs[-ny:]
    force = {"weight": 0., "bend": -1e-5, "stretch": .002}[load]
    if load != "weight":
        d.qfrc_applied[end[:, 2 if load == "bend" else 0]] = force / ny
    history = []
    v = np.zeros_like(points)
    forces = np.zeros_like(points)
    for i in range(round(.28 / ELASTIC_DT)):
        mujoco.mj_step1(m, d)
        angles, gradients = dihedrals(d.flexvert_xpos, hinges)
        v[movers] = d.qvel[dofs]
        hinge_rate = (gradients * v[hinges]).sum(axis=(1, 2))
        moment = stiffness * (angles - rest) + damping * hinge_rate
        forces.fill(0.)
        np.add.at(forces, hinges.ravel(), (-moment[:, None, None] * gradients).reshape(-1, 3))
        d.qfrc_passive[dofs] += forces[movers]
        if i % 500 == 0:
            offset = (d.flexvert_xpos[-ny:] - points[-ny:]).mean(axis=0)
            history.append((d.time, *offset, float(np.max(np.abs(d.qvel)))))
        mujoco.mj_step2(m, d)
    rows = np.asarray(history)
    assert np.isfinite(rows).all()
    actual = float(np.mean(rows[-12:, 1 if load == "stretch" else 3]))
    l = LENGTH * (nx - 2) / (nx - 1)
    beam_ei = material["bending"] * (1. - NU**2) * WIDTH
    theory = {
        "weight": -material["areal_mass"] * WIDTH * 9.81 * l**4 / (8. * beam_ei),
        "bend": force * l**3 / (3. * beam_ei),
        "stretch": force * l / (E * thickness * WIDTH),
    }[load]
    return dict(thickness_m=thickness, load=load, applied_N=force, tip_m=actual, theory_m=theory,
                numerical_over_analytical=actual / theory, settled_speed_mps=float(rows[-12:, 4].max()),
                warnings=int(d.warning.number.sum())), rows


def friction(target):
    points, faces = coupon(5, 3, .020, .012, RADIUS)
    xml = f"""
        <geom name="bottom" type="plane" size=".1 .1 .01" contype="1" conaffinity="2"
              friction="{MU} .005 .0001" solref=".001 1" solimp=".999 .999 .001"/>
        <body name="upper" pos=".01 0 {2 * RADIUS + .0021}">
          <joint name="press" type="slide" axis="0 0 1" damping="1"/>
          <geom name="upper_pad" type="box" size=".04 .035 .002" mass=".012"
                contype="1" conaffinity="2" friction="{MU} .005 .0001"
                solref=".001 1" solimp=".999 .999 .001"/>
        </body>"""
    m, d, (_, dofs, masses), _ = model(points, faces, 50e-6, contact_xml=xml)
    press = m.joint("press").dofadr[0]
    pad_ids = [m.geom(name).id for name in ("upper_pad", "bottom")]
    cf = np.empty(6)
    histories = []
    origin = None
    creep_load = None
    for i in range(round((.15 + 1.35 * 2. * MU * target / 2.) / CONTACT_DT)):
        t = i * CONTACT_DT
        pull = 2. * max(0., t - .15)
        d.qfrc_applied[press] = -target * min(1., t / .035)
        d.qfrc_applied[dofs[:, 0]] = pull * masses / masses.sum()
        mujoco.mj_step(m, d)
        if i % 20:
            continue
        normal = [0., 0.]
        shear = [0., 0.]
        for j in range(d.ncon):
            geom = d.contact[j].geom
            for k, pad in enumerate(pad_ids):
                if pad in geom:
                    mujoco.mj_contactForce(m, d, j, cf)
                    normal[k] += max(0., cf[0])
                    shear[k] += float(np.linalg.norm(cf[1:3]))
        position = float(np.average(d.flexvert_xpos[:, 0], weights=masses))
        if t < .15:
            origin = position
        displacement = position - origin
        speed = float(np.average(d.qvel[dofs[:, 0]], weights=masses))
        histories.append((t, pull, *normal, displacement, speed, *shear))
        if t >= .15 and displacement >= .00005 and speed >= .003 and creep_load is None:
            creep_load = pull
        if t >= .15 and speed >= .08:
            break
    trace = np.array(histories)
    t, pull, _, _, displ, speed, _, _ = trace[-1]
    detected = bool(speed >= .08)
    before_slip = trace[(trace[:, 0] > t - .020) & (trace[:, 0] <= t)]
    attained = float(np.median(before_slip[:, 2:4].sum(axis=1)))
    return dict(target_per_pad_N=target, detected=detected, start_slide_N=float(pull) if detected else None,
                creep_threshold_pull_N=creep_load, measured_displacement_m=float(displ),
                measured_total_normal_N=attained, effective_mu=float(pull / attained) if detected and attained else None,
                warnings=int(d.warning.number.sum())), trace


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    elastic, contact = [], []
    for h in (50e-6, 100e-6):
        for mode in ("weight", "bend", "stretch"):
            result, trace = cantilever(h, mode)
            elastic.append(result)
            np.savetxt(output / f"{round(h * 1e6)}um-{mode}.csv", trace, delimiter=",", comments="",
                       header="time_s,tip_dx_m,tip_dy_m,tip_dz_m,max_speed_mps")
            print(result, flush=True)
    for normal in (.5, 1., 2.):
        result, trace = friction(normal)
        contact.append(result)
        np.savetxt(output / f"{normal:g}N-friction.csv", trace, delimiter=",", comments="",
                   header="time_s,pull_N,top_normal_N,bottom_normal_N,displacement_m,speed_mps,top_shear_N,bottom_shear_N")
        print(result, flush=True)
    ratio = {k: elastic[i]["tip_m"] / elastic[i + 3]["tip_m"] for k, i in (("sag_h_over_2h", 0), ("bending_2h_over_h", 1),
                                                                         ("stretch_2h_over_h", 2))}
    fit = {}
    if all(r["detected"] for r in contact):
        x = np.array([r["measured_total_normal_N"] for r in contact])
        y = np.array([r["start_slide_N"] for r in contact])
        slope, intercept = np.polyfit(x, y, 1)
        fit = dict(slope=float(slope), intercept_N=float(intercept), r2=float(1 - np.sum((y - slope*x-intercept)**2) /
                                                                            np.sum((y - y.mean())**2)))
    summary = dict(elastic=elastic, friction=contact, ratios=ratio, friction_fit=fit)
    (output / "scaling-summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
