"""Backport the 2D curved-reference fix without changing 3.17 material constants.

Upstream: https://github.com/google-deepmind/mujoco/commit/
afa16414ca62075676e24a3003a8825cc4da0d26
Only coefficient 16 of each 17-value hinge record changes. The correction
cancels rest-shape bending forces normal to each flap; in-plane forces remain.
"""
import hashlib
from xml.etree import ElementTree as ET

import mujoco
import numpy as np


UPSTREAM_COMMIT = "afa16414ca62075676e24a3003a8825cc4da0d26"


def _hinge_coefficients(points, young, poisson, thickness):
    def cot(origin, one, two):
        a, b = points[:, one] - points[:, origin], points[:, two] - points[:, origin]
        area2 = np.linalg.norm(np.cross(a, b), axis=1)
        if np.any(area2 <= 0):
            raise ValueError("curved-reference correction requires nondegenerate triangles")
        return np.einsum("ij,ij->i", a, b) / area2

    a01, a02 = cot(0, 1, 2), cot(0, 3, 1)
    a03, a04 = cot(1, 2, 0), cot(1, 0, 3)
    c = np.column_stack((a03 + a04, a01 + a02, -a01 - a03, -a02 - a04))
    e0, e1, e2 = (points[:, i] - points[:, 0] for i in (1, 2, 3))
    e3, e4 = points[:, 2] - points[:, 1], points[:, 3] - points[:, 1]
    normal = np.cross(e0, e1)
    area = (np.linalg.norm(normal, axis=1) + np.linalg.norm(np.cross(e2, e0), axis=1)) / 2
    length2 = np.einsum("ij,ij->i", e0, e0)
    # Preserve ComputeBending's 3.17 prefactor; do not backport 7ba52b533 here.
    shear = young / (2 * (1 + poisson))
    stiffness = 3 * shear * thickness**3 / (24 * area)
    t0 = -(a03[:, None] * e1 + a01[:, None] * e3)
    t1 = -(a04[:, None] * e2 + a02[:, None] * e4)
    cosine = -np.einsum("ij,ij->i", t0, t1) / length2
    matrix = c[:, :, None] * c[:, None, :] * (cosine * stiffness)[:, None, None]
    determinant = np.einsum("ij,ij->i", normal, e2)
    old = determinant * (a01 - a03) * (a04 - a02) * stiffness / length2**1.5
    corrected = -determinant * (c[:, 2] * c[:, 3])**2 * stiffness / length2**2
    return matrix, old, corrected


def apply_curved_reference_fix(model, xml, flex_name="bag"):
    """Correct one compiled standard 2D flex in place, before stepping/transfer.

    ``xml`` must be the same MJCF used to compile ``model`` and must contain
    explicit elasticity parameters. A separate qpos0 data object supplies the
    rest coordinates; existing simulation state is neither read nor changed.
    The old matrix is checked before any write to reject mismatched inputs or
    a compiler with different material constants. Reapplying is idempotent.
    """
    root = ET.fromstring(xml)
    flexcomp = next((node for node in root.iter("flexcomp") if node.get("name") == flex_name), None)
    elasticity = None if flexcomp is None else flexcomp.find("elasticity")
    if elasticity is None:
        raise ValueError(f"missing explicit elasticity for flexcomp {flex_name!r}")
    try:
        young, poisson, thickness = (float(elasticity.attrib[key]) for key in ("young", "poisson", "thickness"))
    except (KeyError, ValueError) as error:
        raise ValueError("curved-reference correction needs explicit young/poisson/thickness") from error
    if not np.all(np.isfinite([young, poisson, thickness])) or young <= 0 or thickness <= 0 or not -1 < poisson < .5:
        raise ValueError("invalid elasticity parameters for curved-reference correction")
    flex = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, flex_name)
    if flex < 0:
        raise ValueError(f"compiled model has no flex {flex_name!r}")
    if model.flex_dim[flex] != 2 or model.flex_interp[flex] != 0 or model.flex_bendingadr[flex] < 0:
        raise ValueError("curved-reference correction requires standard 2D bending")

    rest = mujoco.MjData(model)
    mujoco.mj_kinematics(model, rest)
    mujoco.mj_comPos(model, rest)
    mujoco.mj_flex(model, rest)
    first, count = int(model.flex_edgeadr[flex]), int(model.flex_edgenum[flex])
    vertices = np.column_stack((model.flex_edge[first:first + count], model.flex_edgeflap[first:first + count]))
    interior = np.flatnonzero(vertices[:, 3] >= 0)
    offsets = int(model.flex_bendingadr[flex]) + 17 * interior
    points = rest.flexvert_xpos[int(model.flex_vertadr[flex]) + vertices[interior]]
    matrix, old_formula, corrected = _hinge_coefficients(points, young, poisson, thickness)
    records = model.flex_bending[offsets[:, None] + np.arange(17)]
    if not np.all(np.isfinite(corrected)):
        raise ValueError("nonfinite corrected curvature coefficients")
    if not np.allclose(records[:, :16].reshape(-1, 4, 4), matrix, rtol=1e-9, atol=1e-9):
        raise ValueError("compiled bending matrix differs from the 3.17 geometry/material formula")
    old = records[:, 16].copy()
    known = np.isclose(old, old_formula, rtol=1e-8, atol=1e-10) | np.isclose(old, corrected, rtol=1e-8, atol=1e-10)
    if not np.all(known):
        raise ValueError("compiled curved-reference coefficients match neither old nor corrected formula")
    before_sha = hashlib.sha256(model.flex_bending.tobytes()).hexdigest()
    model.flex_bending[offsets + 16] = corrected
    return dict(method="curved_reference_3_17_backport", upstream_commit=UPSTREAM_COMMIT,
                flex_name=flex_name, interior_edges=len(interior), changed_edges=int(np.count_nonzero(old != corrected)),
                old_coefficient_l2=float(np.linalg.norm(old)), corrected_coefficient_l2=float(np.linalg.norm(corrected)),
                maximum_coefficient_change=float(np.max(np.abs(corrected - old), initial=0)),
                bending_sha256_before=before_sha, bending_sha256_after=hashlib.sha256(model.flex_bending.tobytes()).hexdigest(),
                material_prefactor="MuJoCo 3.17 unchanged", young=young, poisson=poisson, thickness=thickness)
