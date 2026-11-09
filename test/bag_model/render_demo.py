"""Render recorded Newton drops with MuJoCo's offscreen renderer, then tile them into a GIF.

Step 1, MuJoCo python:  test/bag_model/render_demo.py frames outputs/pe_drop.npz [...]   -> outputs/<name>_frames.npy
Step 2, python with PIL: test/bag_model/render_demo.py gif outputs/newton_drop.gif [--seconds 2] outputs/pe_frames.npy [...]
Rendering only; the simulation lives in demo_newton.py and demo_contents.py. Content recordings
draw the bag translucent so the rigid content stays visible.
"""
import json
import os
from pathlib import Path
import sys

import numpy as np

BAG = Path(__file__).resolve().parents[2] / "bag_sim"
sys.path.insert(0, str(BAG))


COLORS = {"pe": ".80 .82 .84 1", "kraft": ".70 .50 .30 1", "paperboard": ".82 .77 .66 1", "bubble": ".93 .94 .96 1"}
TITLES = {"pe": "PE film mailer", "kraft": "Kraft paper mailer", "paperboard": "Paperboard envelope", "bubble": "Bubble mailer (12 g)",
          "bubble_book": "Bubble mailer + book 250 g", "bubble_bottle": "Bubble mailer + bottle 300 g",
          "bubble_ball": "Bubble mailer + ball 150 g", "bubble_torus": "Bubble mailer + torus 120 g"}


def _quat_wxyz(pose):
    """Newton body_q stores position then quaternion (x, y, z, w)."""
    return pose[6], pose[3], pose[4], pose[5]


def render_frames(path, size=(360, 480)):
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    d = np.load(path)
    name = Path(path).stem.replace("_drop", "")
    frames, faces = d["frames"], d["faces"]
    double = np.concatenate([faces, faces[:, ::-1]])
    has_step, has_content = "step_center" in d, "content_vertices" in d
    extent = float(np.ptp(frames[0][:, :2], axis=0).max())
    images = []
    for k, q in enumerate(frames):
        extra_asset, extra_geom = "", ""
        if has_step:
            center, half = d["step_center"], d["step_half"]
            extra_geom += f'<geom type="box" pos="{center[0]} {center[1]} {center[2]}" size="{half[0]} {half[1]} {half[2]}" rgba=".33 .45 .62 1"/>'
        if has_content:
            cv, cf, pose = d["content_vertices"], d["content_faces"], d["content_poses"][k]
            extra_asset = (f'<mesh name="content" vertex="{" ".join(f"{v:.6f}" for v in cv.ravel())}" '
                           f'face="{" ".join(map(str, cf.ravel()))}"/>')
            extra_geom += (f'<geom type="mesh" mesh="content" pos="{pose[0]} {pose[1]} {pose[2]}" '
                           f'quat="{" ".join(map(str, _quat_wxyz(pose)))}" rgba=".80 .25 .15 1" contype="0" conaffinity="0"/>')
        bag_rgba = ".93 .94 .96 .55" if has_content else COLORS.get(name, ".85 .55 .35 1")
        xml = ('<mujoco><visual><global offwidth="960" offheight="720"/><quality shadowsize="4096"/></visual><asset>'
               f'<mesh name="bag" vertex="{" ".join(f"{v:.6f}" for v in q.ravel())}" face="{" ".join(map(str, double.ravel()))}" inertia="shell"/>'
               + extra_asset + '</asset><worldbody><light pos=".3 -.5 1.4" dir="-.2 .35 -1" castshadow="true"/><light pos="-.6 .4 1" dir=".4 -.3 -1" diffuse=".35 .35 .35"/>'
               '<geom type="plane" size="2 2 .01" rgba=".78 .80 .83 1"/>' + extra_geom
               + f'<geom type="mesh" mesh="bag" rgba="{bag_rgba}" contype="0" conaffinity="0"/>'
               '</worldbody></mujoco>')
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        renderer = mujoco.Renderer(model, *size)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = (0.0, 0.0, 0.02 if has_content else 0.05)
        camera.distance = (1.2 * extent + 0.05) if has_content else (1.55 * extent + 0.15)
        camera.azimuth, camera.elevation = 145, -32 if has_content else -24
        renderer.update_scene(data, camera)
        images.append(renderer.render().copy())
        renderer.close()
    out = Path(path).with_name(f"{name}_frames.npy")
    np.save(out, np.array(images))
    print(json.dumps(dict(rendered=str(out), frames=len(images))))


def tile_gif(out, paths, seconds=2.0, fps=30):
    from PIL import Image, ImageDraw

    clips = {Path(p).stem.replace("_frames", ""): np.load(p) for p in paths}
    names = list(clips)
    count = min(len(clips[n]) for n in names)
    h, w = clips[names[0]].shape[1:3]
    pages = []
    for i in range(count):
        canvas = Image.new("RGB", (2 * w, ((len(names) + 1) // 2) * h), (255, 255, 255))
        for k, name in enumerate(names):
            tile = Image.fromarray(clips[name][i])
            draw = ImageDraw.Draw(tile)
            draw.rectangle((0, 0, w, 22), fill=(255, 255, 255))
            draw.text((8, 5), f"{TITLES.get(name, name.replace('_', ' + '))}   t = {i * seconds / (count - 1):.2f} s", fill=(20, 20, 20))
            canvas.paste(tile, ((k % 2) * w, (k // 2) * h))
        pages.append(canvas)
    pages[0].save(out, save_all=True, append_images=pages[1:] + [pages[-1]] * 20, duration=int(1000 / fps), loop=0, optimize=True)
    pages[-1].save(Path(out).with_suffix(".png"))
    print(json.dumps(dict(gif=str(out), last_frame=str(Path(out).with_suffix(".png")), frames=count)))


if __name__ == "__main__":
    if sys.argv[1] == "frames":
        for p in sys.argv[2:]:
            render_frames(p)
    else:
        args = sys.argv[3:]
        seconds = 2.0
        if args[0] == "--seconds":
            seconds, args = float(args[1]), args[2:]
        tile_gif(sys.argv[2], args, seconds=seconds)
