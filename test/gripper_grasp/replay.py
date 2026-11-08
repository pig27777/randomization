"""Replay a recorded grasp (grasp.py --record) in the MuJoCo viewer. Needs a display.

The sponge surface is drawn as a wireframe; pads are blue boxes; nodes of inverted tetrahedra are red.
The replay loops. Space pauses the viewer's clock; close the window to quit.

  DISPLAY=:1 /home/ubuntu/Dev/Envs/mjwarp/bin/python -B replay.py results/rec_soft_off0.npz
"""
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np


def boundary_faces(tets):
    faces = {}
    for t in tets:
        for f in ((t[0], t[2], t[1]), (t[0], t[1], t[3]), (t[0], t[3], t[2]), (t[1], t[2], t[3])):
            key = tuple(sorted(f))
            faces[key] = None if key in faces else f
    return np.array([f for f in faces.values() if f is not None])


def surface_edges(faces):
    return np.array(sorted({tuple(sorted((f[i], f[(i + 1) % 3]))) for f in faces for i in range(3)}))


def jacobians(q, tets, rest):
    def volumes(p):
        a, b, c, d = (p[tets[:, k]] for k in range(4))
        return np.einsum('ij,ij->i', np.cross(b - a, c - a), d - a)
    return volumes(q) / volumes(rest)


def main(path):
    data = np.load(path)
    frames, pads, tets = data['frames'], data['pads'], data['tets']
    half = data['pad_half']
    edges = surface_edges(boundary_faces(tets))
    jac = np.array([jacobians(q, tets, frames[0]) for q in frames])
    bad_frames = np.flatnonzero((jac <= 0).any(axis=1))
    first = f'first inversion at {bad_frames[0] * float(data["dt_frame"]):.2f} s' if len(bad_frames) else 'no inversion'
    print(f'{data["label"]}: {len(frames)} frames ({len(frames) * float(data["dt_frame"]):.2f} s); '
          f'events: {list(data["events"]) or "none"}; {first}', flush=True)
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><visual><global offwidth="1280"/></visual><worldbody><light pos="0 -.3 .6" dir="0 .3 -.6"/>'
        '<geom type="plane" size=".4 .4 .01" rgba=".85 .87 .9 1"/></worldbody></mujoco>')
    state = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, state) as viewer:
        viewer.cam.lookat[:] = (0, 0, .05)
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = .45, 125, -20
        k = 0
        while viewer.is_running():
            start = time.perf_counter()
            q, pose = frames[k], pads[k]
            bad = np.unique(tets[jac[k] <= 0])
            with viewer.lock():
                scene = viewer.user_scn
                scene.ngeom = 0
                for i, j in edges:
                    g = scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                                        np.array([.9, .55, .25, 1], np.float32))
                    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, 2., q[i].astype(float), q[j].astype(float))
                    scene.ngeom += 1
                for node in bad:
                    g = scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, .004), q[node], np.eye(3).ravel(),
                                        np.array([1, 0, 0, 1], np.float32))
                    scene.ngeom += 1
                for p in pose:
                    g = scene.geoms[scene.ngeom]
                    rot = np.zeros(9)
                    mujoco.mju_quat2Mat(rot, np.array([p[6], p[3], p[4], p[5]], float))
                    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX, half.astype(float), p[:3].astype(float), rot,
                                        np.array([.3, .5, .85, .9], np.float32))
                    scene.ngeom += 1
            viewer.sync()
            k = (k + 1) % len(frames)
            if k == 0:
                time.sleep(1.)
            time.sleep(max(0., float(data['dt_frame']) - (time.perf_counter() - start)))


if __name__ == '__main__':
    main(sys.argv[1])
