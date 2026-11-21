"""Rasterize a calibrated color silhouette for the YAM probe diagnostics."""
import numpy as np
from PIL import Image, ImageDraw
from src.low_order_parcel.lift_probe import CAMERA_OFFSET, CAMERA_ROTATION

SIGNS = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
FACES = ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))


def exposure(robot):
    sim = robot.sim
    payload, state = sim.parcel.payload, sim.state.payload
    xy = .5 * payload.params.bag_half_extents[0, :2].numpy()
    half = np.column_stack((np.tile(xy, (4, 1)), payload.half_thickness(state.d_surf)[0].numpy()))
    vertices = state.pos[0].numpy()[:, None] + (half[:, None] * SIGNS) @ payload.frame(state)[0].numpy().T
    camera = np.r_[sim.built.center, 0.] + CAMERA_OFFSET
    def pixels(points):
        local = (points - camera) @ CAMERA_ROTATION
        return local[..., :2] / local[..., 2:] * 440. + [320., 240.]
    image = Image.new("RGB", (640, 480), (40, 40, 40))
    draw = ImageDraw.Draw(image)
    for block in pixels(vertices).round().astype(int):
        for face in FACES:
            draw.polygon([tuple(block[i]) for i in face], fill=(230, 130, 30))
    rgb = np.asarray(image)
    ys = np.nonzero((rgb[..., 0] > 180) & (rgb[..., 2] < 70))[0]
    center_y = float((ys.max() + ys.min()) / 2) if ys.size else None
    return image, center_y, float(pixels(sim.d.site_xpos[robot.site])[1])
