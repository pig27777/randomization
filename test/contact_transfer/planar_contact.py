"""Area-weighted contact quadrature for the benchmark's vertically guided flat box.

This covers the actual projected triangle/footprint intersection, including tools
between vertices. It is deliberately limited to the benchmark's flat top surface
and box bottom; it is not a replacement for arbitrary mesh or gripper collision.
"""
import warp as wp
import newton

polygon8 = wp.types.matrix(shape=(8, 3), dtype=wp.float32)


@wp.func
def _clip(poly: polygon8, count: int, axis: int, bound: float, sign: float,
          p0: wp.vec3, p1: wp.vec3, p2: wp.vec3):
    out = polygon8()
    size = int(0)
    for i in range(count):
        a = poly[i]
        b = poly[(i + 1) % count]
        xa = p0 * a[0] + p1 * a[1] + p2 * a[2]
        xb = p0 * b[0] + p1 * b[1] + p2 * b[2]
        da = sign * (xa[axis] - bound)
        db = sign * (xb[axis] - bound)
        if da >= 0.0:
            out[size] = a
            size += 1
        if (da >= 0.0) != (db >= 0.0):
            out[size] = a + (da / (da - db)) * (b - a)
            size += 1
    return out, size


@wp.kernel
def build_planar_contacts(faces: wp.array[wp.vec3i], q: wp.array[wp.vec3], body_q: wp.array[wp.transform],
        body: int, shape: int, half: wp.vec3, reference_area: float, capacity: int,
        count: wp.array[int], indices: wp.array[wp.vec3i], barycentric: wp.array[wp.vec3],
        contact_shape: wp.array[int], body_pos: wp.array[wp.vec3], body_vel: wp.array[wp.vec3],
        normal: wp.array[wp.vec3], area_weight: wp.array[float]):
    tri = faces[wp.tid()]
    inv = wp.transform_inverse(body_q[body])
    p0 = wp.transform_point(inv, q[tri[0]])
    p1 = wp.transform_point(inv, q[tri[1]])
    p2 = wp.transform_point(inv, q[tri[2]])
    poly = polygon8()
    poly[0] = wp.vec3(1., 0., 0.)
    poly[1] = wp.vec3(0., 1., 0.)
    poly[2] = wp.vec3(0., 0., 1.)
    size = int(3)
    for axis in range(2):
        poly, size = _clip(poly, size, axis, -half[axis], 1., p0, p1, p2)
        poly, size = _clip(poly, size, axis, half[axis], -1., p0, p1, p2)
    for i in range(1, size - 1):
        b0 = poly[0]
        b1 = poly[i]
        b2 = poly[i + 1]
        a = p0 * b0[0] + p1 * b0[1] + p2 * b0[2]
        b = p0 * b1[0] + p1 * b1[1] + p2 * b1[2]
        c = p0 * b2[0] + p1 * b2[1] + p2 * b2[2]
        area = 0.5 * wp.abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        if area > 1.e-12:
            for point in range(3):
                weights = (b0 + b1 + b2) / 6.0
                if point == 0:
                    weights += 0.5 * b0
                elif point == 1:
                    weights += 0.5 * b1
                else:
                    weights += 0.5 * b2
                local = p0 * weights[0] + p1 * weights[1] + p2 * weights[2]
                row = wp.atomic_add(count, 0, 1)
                if row < capacity:
                    indices[row] = tri
                    barycentric[row] = weights
                    contact_shape[row] = shape
                    body_pos[row] = wp.vec3(local[0], local[1], -half[2])
                    body_vel[row] = wp.vec3(0.)
                    normal[row] = wp.transform_vector(body_q[body], wp.vec3(0., 0., -1.))
                    area_weight[row] = area / (3.0 * reference_area)


@wp.kernel
def _weight_materials(count: wp.array[int], weight: wp.array[float], ke: wp.array[float],
                       kd: wp.array[float], material_ke: wp.array[float]):
    i = wp.tid()
    if i < count[0]:
        ke[i] *= weight[i]
        kd[i] *= weight[i]
        material_ke[i] *= weight[i]


class PlanarVBD(newton.solvers.SolverVBD):
    def __init__(self, model, *, quadrature_weights, **kwargs):
        self.quadrature_weights = quadrature_weights
        super().__init__(model, **kwargs)

    def _refresh_body_particle_contact_state(self, contacts, refresh):
        super()._refresh_body_particle_contact_state(contacts, refresh)
        if refresh and contacts is not None:
            wp.launch(_weight_materials, dim=contacts.soft_contact_max,
                      inputs=[contacts.soft_contact_count, self.quadrature_weights,
                              self.body_particle_contact_penalty_k, self.body_particle_contact_material_kd,
                              self.body_particle_contact_material_ke])


def install_planar_contacts(exp, weights):
    """Retain native rigid collision, replace soft records with unique surface-area samples."""
    original = exp.pipeline.collide
    faces = wp.array(exp.top_faces, dtype=wp.vec3i, device=exp.model.device)
    # ke retains the benchmark's N/m scale at a nominal 25 x 25 mm surface patch.
    reference_area = .025 ** 2

    def collide(state, contacts):
        original(state, contacts)
        contacts.soft_contact_count.zero_()
        wp.launch(build_planar_contacts, dim=len(exp.top_faces), inputs=[faces, state.particle_q,
            state.body_q, exp.body, exp.tool_shape, wp.vec3(*exp.half), reference_area, contacts.soft_contact_max,
            contacts.soft_contact_count, contacts.soft_contact_indices, contacts.soft_contact_barycentric,
            contacts.soft_contact_shape, contacts.soft_contact_body_pos, contacts.soft_contact_body_vel,
            contacts.soft_contact_normal, weights])

    exp.pipeline.collide = collide
