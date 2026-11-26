"""Geometry from a segmented metric depth image, shared with a real camera adapter."""

import numpy as np

from .geometry import VisualObservation, VisualUnavailable


def outer_geometry(mask, depth, intrinsics, camera_to_world, camera_position, *, table_z=0.):
    mask = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth)
    if depth.shape != mask.shape or depth.ndim != 2:
        raise ValueError("Aligned two-dimensional mask and depth are required")
    rows, cols = np.where(mask & np.isfinite(depth) & (depth > 0))
    if len(rows) < 8:
        raise VisualUnavailable("Outer bag mask has too few valid depth pixels")
    fx, fy, cx, cy = intrinsics
    z = depth[rows, cols]
    optical = np.column_stack(((cols - cx) * z / fx, (rows - cy) * z / fy, z))
    points = optical @ camera_to_world.T + camera_position
    xy = points[:, :2] - points[:, :2].mean(axis=0)
    _, axes = np.linalg.eigh(xy.T @ xy)
    x = axes[:, -1]
    if x[np.argmax(np.abs(x))] < 0:
        x = -x
    rotation = np.array([[x[0], -x[1], 0], [x[1], x[0], 0], [0, 0, 1.]])
    local = points @ rotation
    low, high = np.quantile(local, [.01, .99], axis=0)
    low[2] = min(table_z, low[2])
    return VisualObservation(rotation @ ((low + high) / 2), rotation,
                             np.maximum((high - low) / 2, .002), "camera_depth_mask")


class SimDepthCamera:
    def __init__(self, model, obj_type, obj_id):
        import mujoco

        self.renderer = mujoco.Renderer(model, height=360, width=480)
        self.camera = model.camera("abc_top").id
        self.focal = 180 / np.tan(np.deg2rad(model.cam_fovy[self.camera]) / 2)
        self.obj_type = int(obj_type)
        self.obj_id = obj_id

    def observe(self, data):
        r = self.renderer
        try:
            r.enable_segmentation_rendering()
            r.update_scene(data, camera=self.camera)
            segmented = r.render()
            mask = (segmented[..., 1] == self.obj_type) & (segmented[..., 0] == self.obj_id)
            r.disable_segmentation_rendering()
            r.enable_depth_rendering()
            r.update_scene(data, camera=self.camera)
            depth = r.render()
        finally:
            r.disable_segmentation_rendering()
            r.disable_depth_rendering()
        rotation = data.cam_xmat[self.camera].reshape(3, 3) @ np.diag([1., -1., -1.])
        return outer_geometry(mask, depth, (self.focal, self.focal, 239.5, 179.5),
                              rotation, data.cam_xpos[self.camera])

    def close(self):
        self.renderer.close()
