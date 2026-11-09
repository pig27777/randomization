"""Interactive MuJoCo viewer for the particle soft model. Usage: view_soft.py flat|blob|pillow

Double-click a body to select it, then Ctrl + right-drag to pull it (Ctrl + left-drag rotates the box).
Space pauses. The free box can be dropped on or pushed into the item.
"""
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

import particle_soft as ps
import particle_tests as pt

shape = sys.argv[1] if len(sys.argv) > 1 else 'pillow'
x = ps.SoftX(**pt.SHAPE_DEFAULTS.get(shape, {}))  # authored defaults; the saved calibration gains diverged
box = ('<body name="box" pos="0 -0.32 0.05"><freejoint/>'
       '<geom name="box" type="box" size=".06 .06 .04" mass="1.5" rgba=".3 .5 .85 1" contype="3" conaffinity="3" condim="3" '
       'friction=".8 .005 .0001" solref="-2e4 -60" solimp="0.95 0.99 0.0005"/></body>')
soft = ps.Soft(pt.BUILDERS[shape](x), x, box, pt.FLOOR)
model, data = soft.model, soft.data
model.vis.global_.offwidth = 1280

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat[:] = soft.com()
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.0, 135, -25
    frame = 0.01
    while viewer.is_running():
        start = time.perf_counter()
        with viewer.lock():
            for _ in range(int(round(frame / soft.dt))):
                soft.apply()
                mujoco.mj_step(model, data)
                soft.advance_history()
            if not np.isfinite(data.qpos).all():
                mujoco.mj_resetData(model, data)
        viewer.sync()
        time.sleep(max(0.0, frame - (time.perf_counter() - start)))
