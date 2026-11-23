"""Presentation mesh for the four-block parcel, in table coordinates (metres)."""
from itertools import product

import numpy as np


def pad_corners(scene, positions, rotations):
    signs = np.array(list(product((-1., 1.), repeat=3)))
    # The right box has a 180-degree yaw, which preserves its corner set.
    centers = np.array([pad["box_center_m"] for pad in scene["pads"]])
    offsets = centers[:, None] + signs * np.array(scene["pad_box_size_m"]) / 2
    return positions[:, None] + offsets @ rotations.transpose(0, 2, 1)


def finger_meshes(asset):
    """Read the visible finger triangles in their own rigid-link frames."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(asset))
    cache = UsdGeom.XformCache()
    meshes = []
    for name in ("tip_left", "tip_right"):
        prim = next(p for p in stage.Traverse() if p.GetName() == name and p.IsA(UsdGeom.Mesh))
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get())
        relative, _ = cache.ComputeRelativeTransform(prim, prim.GetParent())
        matrix = np.asarray(relative)
        points = points @ matrix[:3, :3] + matrix[3, :3]
        if not np.all(np.asarray(mesh.GetFaceVertexCountsAttr().Get()) == 3):
            raise ValueError(f"Expected triangulated finger mesh in {asset}")
        meshes.append((points, np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)))
    return meshes


class ParcelSkin:
    def __init__(self, asset, resolution=101):
        self.fingers = finger_meshes(asset)
        grid = np.linspace(-1., 1., resolution)
        u, v = np.meshgrid(grid, grid)
        self.uv = np.stack((u.ravel(), v.ravel()), axis=1)
        ids = np.arange(u.size).reshape(u.shape)
        a, b, c, d = (x.ravel() for x in (ids[:-1, :-1], ids[:-1, 1:], ids[1:, :-1], ids[1:, 1:]))
        self.top_faces = np.stack((a, b, d, a, d, c), axis=1).reshape(-1, 3)
        self.rim = np.r_[ids[0], ids[1:, -1], ids[-1, -2::-1], ids[-2:0:-1, 0]]
        r = np.arange(len(self.rim)) + u.size
        s = np.roll(r, -1)
        nxt = np.roll(self.rim, -1)
        wall = np.stack((self.rim, r, s, self.rim, s, nxt), axis=1).reshape(-1, 3)
        base = np.stack((s, r, np.full_like(r, s.max() + 1)), axis=1)
        self.indices = np.concatenate((self.top_faces, wall, base)).astype(np.int32)
        self.spacing = grid[1] - grid[0]

    def surface(self, vertices, block_z, corners, pads, positions, rotations, captured):
        blocks = vertices.reshape(4, 8, 3)
        centers = blocks.mean(axis=1)
        x = (centers[1] - centers[0])[:2]
        x = x / np.linalg.norm(x)
        axes = np.array([x, [-x[1], x[0]]])
        local = vertices[:, :2] @ axes.T
        lo, hi = local.min(0), local.max(0)
        half = .5 * (hi - lo)
        # Round the sealed corners and taper the rolled edge of the PE pouch.
        u, v = self.uv.T
        xy = self.uv * (1. - .09 * self.uv[:, ::-1] ** 8) * half + .5 * (hi + lo)
        xy = xy @ axes
        bottom = max(0., block_z[:, 0].min())
        top = block_z[:, 1].max()
        # Match the kernel's world-vertical support columns even when the
        # four-point material frame tilts. Orienting a whole box by that
        # frame makes its far corner rise by half a block width times tilt.
        a, b = np.clip(self.uv + .5, 0., 1.).T
        weights = np.column_stack(((1-a)*(1-b), a*(1-b), (1-a)*b, a*b))
        roof = weights @ block_z[:, 1]
        edge = np.maximum(abs(u), abs(v))
        lip = (.30 * (top - bottom)) * edge ** 10
        folds = .0008 * np.sin(16 * u + 4 * v) ** 2 * np.sin(7 * v) ** 2 * edge ** 3
        z = roof - lip - folds
        z += top - z.max()
        # A whole mesh cell can cross the finger projection with all its
        # vertices outside. Pad the affected footprint by its maximum diagonal.
        cell = self.spacing * np.linalg.norm(half) * 1.3
        bounds = []
        ceilings = []
        for i, (points, faces) in enumerate(self.fingers):
            world = points @ rotations[i].T + positions[i]
            touched = faces[world[faces, 2].min(axis=1) <= top + .002]
            if not len(touched) and corners[i, :, 2].min() > top + .002:
                continue
            footprint = np.concatenate((world[touched.ravel()], corners[i]))
            low, high = footprint[:, :2].min(0), footprint[:, :2].max(0)
            target = min(world[:, 2].min(), corners[i, :, 2].min()) - .00155
            outside = np.maximum(np.maximum(low - xy, xy - high), 0.)
            distance = np.linalg.norm(outside, axis=1)
            t = np.clip((distance - cell) / .016, 0., 1.)
            smooth = t * t * (3. - 2. * t)
            dent = np.minimum(z, target) + (z - np.minimum(z, target)) * smooth
            rim_rise = .0008 * np.sin(np.pi * t) ** 2 * np.clip((top - target) / .012, 0., 1.)
            ceilings.append(np.minimum(top, dent + rim_rise))
            bounds.append((low, high, target))
        for ceiling in ceilings:
            z = np.minimum(z, ceiling)
        points = np.vstack((np.column_stack((xy, z)), np.column_stack((xy[self.rim],
                            np.full(len(self.rim), bottom))), [*xy.mean(axis=0), bottom]))
        faces = self.indices
        if np.all(captured):
            # A closed rail gap can be much narrower than the grid pitch.
            # Build the two sides of the gathered fold inside the measured
            # inner-face planes. Its bottom edge enters the surrounding roof.
            middle = pads.mean(axis=0)
            root_z = z[np.argmin(np.linalg.norm(xy - middle[:2], axis=1))] - .0001
            neck = []
            for side in range(2):
                tangent = rotations[side, :, 0]
                if tangent @ rotations[0, :, 0] < 0:
                    tangent = -tangent
                tip = .9 * pads[side] + .1 * middle
                lip = tip + np.array([-.005, .005])[:, None] * tangent
                base = lip.copy()
                up = rotations[side, :, 2]
                if abs(up[2]) > .1:
                    base += ((root_z - lip[:, 2]) / up[2])[:, None] * up
                else:
                    base[:, 2] = root_z
                neck.extend([*base, *lip])
            neck_faces = np.array([[0,1,3],[0,3,2],[4,7,5],[4,6,7],[2,3,7],[2,7,6],
                                   [0,2,6],[0,6,4],[1,5,7],[1,7,3]], dtype=np.int32)
            faces = np.vstack((faces, len(points) + neck_faces))
            points = np.vstack((points, neck))
        return points.astype(np.float32), faces, bounds


class NewtonParcelSkin(ParcelSkin):
    def __init__(self, asset, viewer, device):
        import warp as wp

        super().__init__(asset)
        self.viewer = viewer
        self.device = device
        self.triangles = wp.array(self.indices.ravel(), dtype=wp.int32, device=device)

    def draw(self, env):
        import warp as wp
        from isaaclab.utils.math import matrix_from_quat
        from src.low_order_parcel.lumped_contact_proxy import ContactState

        port = env.ports[0]
        pose = env.robots[0].data.body_link_pose_w.torch[0, port.pad_ids]
        pos = (pose[:, :3] - env.scene.env_origins[0]).cpu().numpy()
        rot = matrix_from_quat(pose[:, 3:]).cpu().numpy()
        scene = env.scene_definition
        corners = pad_corners(scene, pos, rot)
        # Always use the CAD inner-face centers, independent of the work port.
        pads = corners.mean(axis=1) + rot[:, :, 1] * np.array([[1.], [-1.]]) * scene["pad_box_size_m"][1] / 2
        vertices = env.projectors[0].vertices(env.parcel_state.payload)[0].cpu().numpy()
        captured = env.parcel_state.contacts_A[0, :, ContactState.CAPTURE].cpu().numpy() > .5
        state = env.parcel_state.payload
        half = env.parcel_sim.payload.half_thickness(state.d_surf)[0].cpu().numpy()
        center_z = state.pos[0, :, 2].cpu().numpy()
        block_z = np.column_stack((center_z - half, center_z + half))
        mesh, faces, bounds = self.surface(vertices, block_z, corners, pads, pos, rot, captured)
        self.last = (vertices, corners, pads, mesh, bounds)
        world = wp.array(mesh + env.scene.env_origins[0].cpu().numpy(), dtype=wp.vec3, device=self.device)
        indices = self.triangles if faces is self.indices else wp.array(faces.ravel(), dtype=wp.int32, device=self.device)
        self.viewer.log_mesh("/Parcel/film", world, indices, color=(.25, .49, .46),
                             roughness=.52, backface_culling=False, dynamic=True)
