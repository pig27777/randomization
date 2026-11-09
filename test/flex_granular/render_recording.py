"""Replay saved GPU grain/membrane states using the shared Mesa renderer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

os.environ.setdefault("LP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test/flex3d"))
from render_video import SoftwareContext
import mujoco
import numpy as np


def render(result_path, output):
    if output.exists():
        raise FileExistsError(output)
    result = json.loads(result_path.read_text())
    source = result_path.with_suffix(".trajectory.npz")
    with np.load(source, allow_pickle=False) as recorded:
        times, states = recorded["time"].copy(), recorded["qpos"].copy()
    if len(times) != len(states) or not np.isfinite(states).all() or np.any(np.diff(times) <= 0):
        raise ValueError("invalid saved trajectory")
    if len(times) < 2 or not np.allclose(np.diff(times), .01, atol=1e-9):
        raise ValueError("expected 100 Hz saved GPU states")
    from mujoco.rendering.classic import gl_context
    gl_context.GLContext = SoftwareContext
    tree = ET.fromstring(result_path.with_suffix(".xml").read_text())
    asset = ET.SubElement(tree, "asset")
    ET.SubElement(asset, "texture", type="skybox", builtin="flat", width="128", height="128",
                  rgb1=".95 .97 .98", rgb2=".95 .97 .98")
    model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
    model.vis.global_.offwidth = 640
    model.vis.global_.offheight = 480
    model.flex_rgba[0] = [.12, .68, .74, .22]
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    renderer_name = renderer._gl_context.renderer
    options = [mujoco.MjvOption(), mujoco.MjvOption()]
    for opt in options:
        opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXEDGE] = True
    options[1].flags[mujoco.mjtVisFlag.mjVIS_FLEXFACE] = False
    options[1].flags[mujoco.mjtVisFlag.mjVIS_FLEXSKIN] = False
    cameras = [mujoco.MjvCamera(), mujoco.MjvCamera()]
    cameras[0].lookat[:] = [0, 0, .075]
    cameras[0].distance = .48
    cameras[0].azimuth, cameras[0].elevation = 125, -25
    cameras[1].lookat[:] = [0, 0, .045]
    cameras[1].distance = .40
    cameras[1].azimuth, cameras[1].elevation = 100, -65
    trial = result["trials"][0]
    warning = f'Solver flags {trial["overflow_or"]} - diagnostic recording' if trial["overflow_or"] else "No solver flags recorded"
    filters = [
        "drawtext=text='GPU simulation - 64 free grains and a 2D flex membrane':x=22:y=12:fontsize=27:fontcolor=0x18313c",
        "drawtext=text='Solid view - translucent film':x=22:y=52:fontsize=20:fontcolor=0x334155",
        "drawtext=text='Wireframe view - internal grain motion':x=662:y=52:fontsize=20:fontcolor=0x334155",
        "drawtext=text='1x playback - recorded states - simulation t %{pts}':x=22:y=565:fontsize=19:fontcolor=0x334155",
        f"drawtext=text='{warning}':x=720:y=565:fontsize=17:fontcolor=0x334155",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
               "-video_size", "1280x600", "-framerate", "25", "-i", "pipe:0", "-vf", ",".join(filters),
               "-c:v", "libx264", "-threads", "2", "-preset", "fast", "-crf", "19", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    indices = list(range(0, len(times), 4))
    if indices[-1] != len(times)-1:
        indices.append(len(times)-1)
    try:
        for frame_number, index in enumerate(indices):
            data.qpos[:] = states[index]
            mujoco.mj_forward(model, data)
            frame = np.full((600, 1280, 3), 248, dtype=np.uint8)
            for panel, (camera, opt) in enumerate(zip(cameras, options)):
                renderer.update_scene(data, camera=camera, scene_option=opt)
                frame[80:560, panel*640:(panel+1)*640] = renderer.render()
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("ffmpeg encoding failed")
    finally:
        renderer.close()
        if process.poll() is None:
            process.kill()
            process.wait()
    manifest = dict(renderer=renderer_name, physics_backend="MJWarp CUDA", physics_recomputed=False,
                    interpolated=False, fps=25, frames=len(indices), recorded_samples=len(times),
                    simulated_time_s=[float(times[0]), float(times[-1])], valid=result["valid"],
                    solver_flags=trial["overflow_or"], source=str(source),
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    shared_context_sha256=hashlib.sha256((ROOT/"test/flex3d/render_video.py").read_bytes()).hexdigest(),
                    renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), command=command)
    output.with_suffix(".render.json").write_text(json.dumps(manifest, indent=2))
    print(output, renderer_name, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.result, args.output)
