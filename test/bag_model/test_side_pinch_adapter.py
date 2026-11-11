"""Static checks of side_pinch_adapter.py that need no YAM assets (rope geometry, rules, XML parts)."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import side_pinch_adapter as S  # noqa: E402

SIZE = np.array([0.30, 0.20, 0.04])


def sites_for(size, depth=0.0):
    """Rope sites +-10 mm about the long-face centre (box frame), pressed `depth` into the face."""
    g = np.array([size[0] / 2, depth, size[2] / 2])
    return {"rope_ga": g - [0, 0, .01], "rope_gb": g + [0, 0, .01]}


def site_of(anchors, size):
    return ["rope_ga" if (p[2] < size[2] / 2 - 1e-9 or (abs(p[2] - size[2] / 2) < 1e-9 and p[0] < size[0] / 2))
            else "rope_gb" for _, p in anchors]


class Geometry(unittest.TestCase):
    def test_anchors(self):
        a = S.box_anchors(SIZE)
        self.assertEqual(len(a), 20)
        self.assertEqual(len({n for n, _ in a}), 20)

    def test_rope_limits_match_preview(self):
        a = S.box_anchors(SIZE)
        lim = dict(zip([n for n, _ in a], 1e3 * S.rope_limits(sites_for(SIZE), site_of(a, SIZE), a, SIZE, (.04, .04))))
        # long-side centre -> that face's top/bottom edge midpoints: 10 mm surface path + the width-loop slack
        self.assertAlmostEqual(lim["amnn"], 50.0, delta=.2)
        self.assertAlmostEqual(lim["amnp"], 50.0, delta=.2)
        self.assertAlmostEqual(lim["ampp"], 250.0, delta=.5)     # 10 up + 200 across the top, all width loop
        self.assertAlmostEqual(lim["apnn"], 1e3 * np.hypot(.15, .01) + 40.0, delta=.5)   # along the long side: length loop

    def test_loop_split(self):
        surf = S.BoxSurface(.04, length=.3, width=.2)
        q = np.array([.15, 0., .02])
        (l1, pl1, pw1), (l2, pl2, pw2), (l3, pl3, pw3) = surf.geodesic_loops(
            q, [np.array([.15, .2, .04]), np.array([.3, 0., .02]), np.array([0., .1, .02])])
        self.assertAlmostEqual(pl1, 0.0, places=9)                       # up the long side, across the top in y
        self.assertAlmostEqual(pw1, l1, places=6)
        self.assertAlmostEqual(pl2, l2, places=6)                        # horizontal on the long side
        self.assertAlmostEqual(pw2, 0.0, places=9)
        self.assertGreater(pl3, 0.0)                                     # mixed path to the short side
        self.assertGreater(pw3, 0.0)
        self.assertAlmostEqual(S.loop_slack(0.0, 1.0, (.04, .03)), .03)
        self.assertAlmostEqual(S.loop_slack(1.0, 0.0, (.04, .03)), .04)
        self.assertEqual(S.loop_slack(0.0, 0.0, (.04, .03)), 0.0)

    def test_close_consumes_slack(self):
        a = S.box_anchors(SIZE)
        names = [n for n, _ in a]
        args = (sites_for(SIZE), site_of(a, SIZE), a, SIZE)
        off = dict(zip(names, S.rope_limits(*args, S.effective_slack((.04, .04), [1, 0, 0], .095, 0., False))))
        on = dict(zip(names, S.rope_limits(*args, S.effective_slack((.04, .04), [1, 0, 0], .095, 0., True))))
        self.assertEqual(list(S.effective_slack((.04, .04), [1, 0, 0], .095, 0., True)), [0.0, .04])
        self.assertAlmostEqual(on["ampp"], off["ampp"], places=6)   # width loop: unchanged
        self.assertLess(on["apnn"], off["apnn"] - .03)              # length loop: slack taken by the fingers
        self.assertAlmostEqual(on["amnn"], off["amnn"], places=6)
        partial = S.effective_slack((.04, .04), [1, 0, 0], .05, .01, True)
        self.assertAlmostEqual(partial[0], .02)

    def test_fit_frame(self):
        rng = np.random.default_rng(0)
        rest = rng.normal(size=(8, 3))
        q = np.zeros(4)
        mujoco.mju_axisAngle2Quat(q, np.array([.3, -.5, .8]) / np.linalg.norm([.3, -.5, .8]), .7)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        R = R.reshape(3, 3)
        Rf, t = S.fit_frame(rest, rest @ R.T + [1, 2, 3])
        np.testing.assert_allclose(Rf, R, atol=1e-9)
        np.testing.assert_allclose(t, [1, 2, 3], atol=1e-9)


class Rules(unittest.TestCase):
    def test_modes_for_preview_conditions(self):
        c = [1, 0, 0]
        # (slack, depth) of the armless pre-check cases 1-5 (long side): expected r2 modes
        for slack, depth, mode in [((.04, .04), 0., "rope"), ((.04, .04), .0042, "rope"), ((0, 0), .0126, "rope+connect"),
                                   ((0, 0), 0., "miss"), ((0, 0), .0182, "rope+connect"), ((0, 0), .007, "miss")]:
            self.assertEqual(S.decide(True, S.gather_amount(slack, c, depth), depth), mode)
        self.assertEqual(S.decide(False, 1.0, .02), "miss")
        self.assertTrue(S.film_at([.15, -.03, .02], SIZE, (.04, .04)))
        self.assertFalse(S.film_at([.15, -.03, .02], SIZE, (0, 0)))

    def test_hold_damping_gate(self):
        v = np.array([[.1, 0, 0], [0, .2, 0]])
        np.testing.assert_array_equal(S.hold_damping_forces(v, [1, 1], [0, 0, 0], 10., False), 0)
        f = S.hold_damping_forces(v, [1, 3], [.1, 0, 0], 10., True)
        np.testing.assert_allclose(f, [[0, 0, 0], [.75, -1.5, 0]], atol=1e-12)
        self.assertAlmostEqual(S.hold_damping_coef(.5, SIZE, "y"), 2 * .5 * np.sqrt(9.81 / .1))
        cfg = S.Config(category="hard", size=(.3, .2, .04), mass=.5)
        self.assertTrue(cfg.hold_damping and cfg.close_consumes_slack)     # default on


class Xml(unittest.TestCase):
    def test_sites_on_v13_contents(self):
        for cat, size, mass, face in [("hard", (.3, .2, .04), .5, "y"), ("bendable", (.3, .2, .04), .5, "y"),
                                      ("compressible", (.3, .2, .12), .29, "y"), ("hard", (.3, .2, .04), .5, "x")]:
            cfg = S.Config(category=cat, size=size, mass=mass, slack=(.04, .04), face=face, fixed_block=True)
            k = replace(S.ANCHORS[cat], size=size, mass=mass)
            desc = S.build(k, omega_max=.8 / cfg.dt, geometry=cat)
            anchors, sites, tendons, connects, segs, block = S.content_xml_parts(desc, cfg, np.array(size))
            built = S.A.scene(desc, [*S.CENTER, .002 + size[2] / 2], dt=cfg.dt, worldbody=block)
            from xml.etree import ElementTree as ET
            root = ET.fromstring(built.xml)
            for body, items in sites.items():
                node = root.find(f".//body[@name='{body}']")
                for s in items:
                    node.append(ET.fromstring(s))
            m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
            d = mujoco.MjData(m)
            mujoco.mj_forward(m, d)
            origin = np.r_[S.CENTER, .002] - np.r_[np.array(size[:2]) / 2, 0]
            for nm, p in anchors:
                np.testing.assert_allclose(d.site_xpos[m.site(nm).id], origin + p, atol=1e-6, err_msg=f"{cat} {nm}")
            self.assertEqual(tendons.count("<spatial"), 20)
            self.assertEqual(len(segs) == 0, desc.rigid)
            if not desc.rigid:
                self.assertEqual(connects.count("<connect"), len(segs))

    def test_scene_refuses_without_platform_approach(self):
        with self.assertRaisesRegex(RuntimeError, "approach direction"):
            S.scene(S.Config(category="hard", size=(.3, .2, .04), mass=.5, slack=(.04, .04)))


if __name__ == "__main__":
    unittest.main()
