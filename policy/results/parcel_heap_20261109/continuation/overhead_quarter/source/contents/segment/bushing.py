"""Conservative, objective six-DOF face connections for the CPU reference.

u is the separation of the two initially coincident face centres, in body A's
frame. phi = log(R_A.T R_B). The force includes the moving-frame torque;
omitting it would create angular momentum for anisotropic shear springs.
Only the elastic part is applied here. Damping is native tendon damping (mj_adapter dampers),
integrated implicitly by implicitfast: explicit pair damping limited the step to about 0.3 ms.
"""
import numpy as np

# Compression, bending and compaction support are native zero-stiffness tendon limits (mj_adapter
# edge stops), so these penalty/gain terms are off. verified_50us used 1000 / 1000 / 99 at 50 us;
# stability_estimate.py sets them to compare step limits.
ANGLE_STOP_GAIN = 0.0
COMPRESSION_STOP_GAIN = 0.0
COMPACTION_BEND_GAIN = 0.0


def rotate(r, v):
    return np.einsum("nij,nj->ni", r, v)


def cross(a, b):
    """Row-wise 3D cross product, avoiding NumPy's general axis dispatch per step."""
    out = np.empty_like(a)
    out[:, 0] = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
    out[:, 1] = a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]
    out[:, 2] = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return out


def log_rotation(qa, qb):
    w = np.sum(qa * qb, axis=1)
    v = qa[:, :1] * qb[:, 1:] - qb[:, :1] * qa[:, 1:] - cross(qa[:, 1:], qb[:, 1:])
    sign = np.where(w < 0, -1.0, 1.0)
    v, w = v * sign[:, None], w * sign
    s = np.linalg.norm(v, axis=1)
    scale = np.divide(2 * np.arctan2(s, w), s, out=np.full_like(s, 2.0), where=s > 1e-12)
    return v * scale[:, None]


def joint_damping(j):
    """Damping of one connection in its frame, indexed by global axis: translational (axial, shear) and
    rotational (twist, bending), each 2 zeta sqrt(k x effective mass or inertia)."""
    other = [a for a in range(3) if a != j.axis]
    kt, kr, inertia = np.empty(3), np.empty(3), np.empty(3)
    kt[j.axis], kt[other] = j.k_comp, j.k_shear
    kr[j.axis], kr[other] = j.k_twist, j.k_bend
    inertia[j.axis], inertia[other] = j.i_eff[2], j.i_eff[:2]
    return 2 * j.damping_ratio * np.sqrt(kt * j.m_eff), 2 * j.damping_ratio * np.sqrt(kr * inertia)


def log_jacobian(phi, v, transpose=False):
    """Left log Jacobian inverse, or its transpose, acting on v."""
    theta = np.linalg.norm(phi, axis=1)
    a = np.full_like(theta, 1 / 12)
    sel = theta > 1e-4
    t = theta[sel]
    a[sel] = (1 - 0.5 * t / np.tan(0.5 * t)) / t**2
    first = cross(phi, v)
    return v + (0.5 if transpose else -0.5) * first + a[:, None] * cross(phi, first)


class Bushings:
    def __init__(self, desc, bodies, joint_indices):
        js = [desc.joints[i] for i in joint_indices]
        self.a = np.array([bodies[j.a] for j in js], int)
        self.b = np.array([bodies[j.b] for j in js], int)
        self.n = len(js)
        self.node_a = np.array([j.a for j in js], int)
        self.node_b = np.array([j.b for j in js], int)
        self.node_count = len(desc.segments)
        self.centered = all(np.all(s.com == 0) for s in desc.segments)
        self.axis = np.array([j.axis for j in js], int)
        self.row = np.arange(self.n)
        counts = np.zeros((self.node_count, 3))
        np.add.at(counts, (self.node_a, self.axis), 1)
        np.add.at(counts, (self.node_b, self.axis), 1)
        self.weight_a = 1 / counts[self.node_a, self.axis]
        self.weight_b = 1 / counts[self.node_b, self.axis]
        self.ra = np.zeros((self.n, 3))
        self.ra[self.row, self.axis] = [j.rest / 2 for j in js]
        self.kt, self.kr = np.zeros_like(self.ra), np.zeros_like(self.ra)
        self.tension = np.array([j.k_tens for j in js])
        self.limit = np.full_like(self.ra, np.inf)
        self.compression_limit = np.array([j.lock_comp for j in js])
        self.compression_travel = np.array([j.rest * j.lock_comp for j in js])
        for i, j in enumerate(js):
            other = [a for a in range(3) if a != j.axis]
            self.kt[i, j.axis], self.kt[i, other] = j.k_comp, j.k_shear
            self.kr[i, j.axis], self.kr[i, other] = j.k_twist, j.k_bend
            self.limit[i, other] = j.lock_bend
        self.last_wrenches = None
        self.orthogonal_compaction = np.zeros(self.n)

    def check(self, d):
        _, _, _, _, u, phi = self.coordinates(d)
        strain = -u[self.row, self.axis] / (2 * self.ra[self.row, self.axis])
        if strain.max() > 0.5:
            return f"adjacent overlap {strain.max():.2f} of segment size"
        ratio = strain / self.compression_limit
        if ratio.max() > 1.2:
            return f"compression {ratio.max():.2f} x limit"
        ratio = np.abs(phi) / self.limit
        if ratio.max() > 1.2:
            return f"bending {ratio.max():.2f} x limit"
        return None

    def coordinates(self, d):
        ra = d.xmat[self.a].reshape(-1, 3, 3)
        rb = d.xmat[self.b].reshape(-1, 3, 3)
        arm_a, arm_b = rotate(ra, self.ra), rotate(rb, -self.ra)
        delta = d.xpos[self.b] + arm_b - d.xpos[self.a] - arm_a
        u = rotate(ra.transpose(0, 2, 1), delta)
        phi = log_rotation(d.xquat[self.a], d.xquat[self.b])
        return ra, arm_a, arm_b, delta, u, phi

    def elastic(self, u, phi, compute_energy=True):
        k = self.kt.copy()
        ax = u[self.row, self.axis]
        k[self.row, self.axis] = np.where(ax > 0, self.tension, k[self.row, self.axis])
        f = k * u
        if COMPACTION_BEND_GAIN == ANGLE_STOP_GAIN == COMPRESSION_STOP_GAIN == 0:
            # Current native-stop model: all three historical penalty terms are
            # disabled. Skip their incidence maps and gradients, exactly zero.
            moment = self.kr * phi
            energy = .5 * np.sum(k * u**2 + self.kr * phi**2) if compute_energy else 0.0
            return f, moment, float(energy), np.ones(self.n)
        # Orthogonal compression also supports bending. Sum each segment's
        # directional compaction, averaging the two neighbours along each axis.
        compact = np.maximum(-ax / self.compression_travel, 0)
        node_compact = np.zeros(self.node_count)
        np.add.at(node_compact, self.node_a, compact * self.weight_a)
        np.add.at(node_compact, self.node_b, compact * self.weight_b)
        density = .5 * (node_compact[self.node_a] + node_compact[self.node_b])
        gain = 1 + COMPACTION_BEND_GAIN * density**2
        angular_energy2 = np.sum(self.kr * phi**2, axis=1)
        moment = gain[:, None] * self.kr * phi
        # Differentiate the same energy through the compaction incidence map.
        # Leaving this axial term out would inject energy when pressure changes.
        node_gradient = np.zeros(self.node_count)
        pressure = COMPACTION_BEND_GAIN * density * angular_energy2
        np.add.at(node_gradient, self.node_a, .5 * pressure)
        np.add.at(node_gradient, self.node_b, .5 * pressure)
        gradient = node_gradient[self.node_a] * self.weight_a + node_gradient[self.node_b] * self.weight_b
        f[self.row, self.axis] -= np.where(compact > 0, gradient / self.compression_travel, 0)
        # Bending stops act on angles, so compression cannot falsely activate them.
        beyond = np.maximum(np.abs(phi) - self.limit, 0.0)
        moment += ANGLE_STOP_GAIN * self.kr * beyond * np.sign(phi)
        energy = 0.5 * np.sum(k * u**2 + gain[:, None] * self.kr * phi**2
                              + ANGLE_STOP_GAIN * self.kr * beyond**2)
        compression_stop = np.minimum(ax + self.compression_travel, 0.0)
        stop_k = COMPRESSION_STOP_GAIN * self.kt[self.row, self.axis]
        f[self.row, self.axis] += stop_k * compression_stop
        energy += 0.5 * np.sum(stop_k * compression_stop**2)
        return f, moment, float(energy), gain

    def energy(self, d):
        return self.elastic(*self.coordinates(d)[-2:])[2] if self.n else 0.0

    def apply(self, m, d):
        if not self.n:
            return
        ra, arm_a, arm_b, delta, u, phi = self.coordinates(d)
        f, moment, _, _ = self.elastic(u, phi, compute_energy=False)
        # Per connection, the larger of its two segments' compaction along the other two axes (each axis:
        # mean normalized compression of that segment's connections along it): a compacted segment stiffens
        # all its connections. It switches the native compaction welds; the connection's own axis is
        # excluded so a squeezed column cannot lock itself.
        compact = np.maximum(-u[self.row, self.axis] / self.compression_travel, 0)
        node = np.zeros((self.node_count, 3))
        np.add.at(node, (self.node_a, self.axis), compact * self.weight_a)
        np.add.at(node, (self.node_b, self.axis), compact * self.weight_b)
        across = node.sum(1, keepdims=True) - node
        self.orthogonal_compaction = np.maximum(across[self.node_a, self.axis], across[self.node_b, self.axis])
        f = rotate(ra, f)
        torque = rotate(ra, log_jacobian(phi, moment, transpose=True))
        ta = torque + cross(arm_a + delta, f)
        tb = -torque - cross(arm_b, f)
        # xfrc_applied torques are about the body COM, not the body origin.
        if not self.centered:
            ta += cross(d.xpos[self.a] - d.xipos[self.a], f)
            tb += cross(d.xpos[self.b] - d.xipos[self.b], -f)
        np.add.at(d.xfrc_applied[:, :3], self.a, f)
        np.add.at(d.xfrc_applied[:, :3], self.b, -f)
        np.add.at(d.xfrc_applied[:, 3:], self.a, ta)
        np.add.at(d.xfrc_applied[:, 3:], self.b, tb)
        self.last_wrenches = (f, ta, tb)
