import argparse
import json
from pathlib import Path
import runpy
import subprocess

import numpy as np
import torch
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim.utils.stage import use_stage

from src.isaac_yam_parcel.yam_parcel_base_env import YAMPickEnv, yam_parcel_cfg
from src.low_order_parcel.domain_randomizer import CornerSuite, DomainRandomizer


acquisition = runpy.run_path("test/isaaclab/check_route_b_closed_loop.py")["acquisition"]


class RecordingEnv(YAMPickEnv):
    encoder = None
    captures = 0

    def step(self, actions):
        result = super().step(actions)
        self._draw_parcel()
        self.sim.render()
        frame = self.sim._visualizers[0].render_rgb_array()
        pixels = np.ascontiguousarray(frame[..., :3])
        if self.encoder is None:
            h, w = pixels.shape[:2]
            self.encoder = subprocess.Popen([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{w}x{h}", "-r", "25", "-i", "pipe:0", "-an", "-vf", "scale=1280:-2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-threads", "2",
                "-movflags", "+faststart", str(self.video),
            ], stdin=subprocess.PIPE)
        self.encoder.stdin.write(pixels.tobytes())
        self.captures += 1
        return result

    def finish(self):
        if self.encoder is not None:
            self.encoder.stdin.close()
            if self.encoder.wait() != 0:
                raise RuntimeError("FFmpeg failed to build the grasp clip")


parser = argparse.ArgumentParser()
parser.add_argument("--output-clip", type=Path, required=True)
parser.add_argument("--seat-fold", action="store_true")
add_launcher_args(parser)
args = parser.parse_args()
cfg = yam_parcel_cfg(1)
# This conversion already has a fixed root joint. Use a preview-only USD
# with that authored joint removed, so the spawner authors one fixed joint.
cfg.scene.arm_a.spawn.usd_path = "/tmp/route-b-abc-ui.2aaqMr/preview-yam.usdc"
cfg.scene.arm_b.spawn.usd_path = cfg.scene.arm_a.spawn.usd_path
with launch_simulation(cfg.sim, args):
    import pyglet
    pyglet.options.shadow_window = False
    import pyglet.window.xlib as pyglet_x11

    pyglet_x11._have_utf8 = False
    randomizer = DomainRandomizer("cuda:0", 17)
    episode = CornerSuite(randomizer).select(torch.tensor([4], device="cuda", dtype=torch.long))
    env = RecordingEnv(cfg, episode=episode)
    env.video = args.output_clip
    try:
        with use_stage(env.sim.stage):
            if env.markers is None:
                env.markers = env._make_markers()
        env.sim.set_camera_view((.47, -.67, .38), (-.09, -.23, .07))
        row = ("C5", "top", -.036, 0., 4.)
        scores, trace = acquisition(env, [row], 7.2, rapid_close=True)
        env.finish()
        print("ABC_GRASP_VIDEO " + json.dumps(dict(frames=env.captures, result=scores[0])), flush=True)
        args.output_clip.with_suffix(".json").write_text(json.dumps(trace))
    finally:
        env.close()
