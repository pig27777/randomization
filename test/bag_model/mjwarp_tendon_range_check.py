"""Minimal check: can MJWarp use a different tendon length limit (rope length) in each world?

One ball hangs from a fixed point by a taut-only spatial tendon. World 0 gets range upper 0.10 m,
world 1 gets 0.20 m, world 2 gets 10 m (rope effectively released). After 1 s each ball should hang
at its own rope length below the anchor (or fall freely for world 2).
"""
import json
import sys

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

XML = """<mujoco><option timestep="0.001" integrator="implicitfast"/>
<worldbody><site name="top" pos="0 0 1"/>
<body name="ball" pos="0 0 0.95"><freejoint/><geom type="sphere" size="0.01" mass="0.1"/><site name="b"/></body>
</worldbody>
<tendon><spatial limited="true" range="0 0.05" solreflimit="0.005 1"><site site="top"/><site site="b"/></spatial></tendon>
</mujoco>"""

UPPER = [0.10, 0.20, 10.0]


def main():
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    mjw_model = mjw.put_model(m)
    mjw_data = mjw.put_data(m, d, nworld=len(UPPER))
    rng = np.tile(m.tendon_range[None], (len(UPPER), 1, 1)).astype(np.float32)
    rng[:, 0, 1] = UPPER
    mjw_model.tendon_range = wp.array(rng, dtype=wp.vec2)   # leading dim = nworld
    for _ in range(1000):
        mjw.step(mjw_model, mjw_data)
    wp.synchronize()
    z = mjw_data.xpos.numpy()[:, m.body("ball").id, 2]
    out = dict(mujoco=mujoco.__version__, warp=wp.__version__, upper=UPPER,
               ball_z_after_1s=[round(float(v), 4) for v in z],
               expected_z=[1 - UPPER[0], 1 - UPPER[1], "free fall (< 0)"])
    out["per_world_range_works"] = bool(abs(z[0] - (1 - UPPER[0])) < 0.005 and abs(z[1] - (1 - UPPER[1])) < 0.005
                                        and z[2] < 0)
    print(json.dumps(out))
    return 0 if out["per_world_range_works"] else 1


if __name__ == "__main__":
    sys.exit(main())
