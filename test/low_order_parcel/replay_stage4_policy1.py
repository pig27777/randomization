"""Replay the fixed Policy 1 corner holdout and render a C3/C6 comparison."""

import argparse
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from src.low_order_parcel.domain_randomizer import CornerSuite, DomainRandomizer
from src.low_order_parcel.envs.pick_to_air_env import PickToAirEnv
from src.low_order_parcel.lumped_contact_proxy import ContactState
from src.low_order_parcel.pad_torsion import _orientation, rotation_quaternion
from src.low_order_parcel.rl.asymmetric_ppo import AsymmetricPPO


FPS = 25
PANEL = (640, 492)
STAGES = ("触探测高", "进给与闭合", "25 mm 微提拉", "正式提拉验证")
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
MASS = ("C3  ·  1.5 kg 偏心硬书", "C6  ·  65 g 低摩擦轻包")


class Recorder(PickToAirEnv):
    def __init__(self, episode, rows):
        super().__init__(episode, seed=88004, cuda_graph=True)
        self.rows = rows
        self.recorded = []
        self.stages = []
        self.ticks = 0
        self.stage = 0

    def _tick(self, command, active, *, lift=False):
        super()._tick(command, active, lift=lift)
        if self.ticks % 20 == 0:
            i = self.rows
            body = self.state.payload
            template = self.sim.payload
            sampled_body = type(body)(*(v[i] for v in body))
            self.recorded.append(torch.cat((
                sampled_body.pos.flatten(1), _orientation(template.frame(sampled_body)),
                template.half_thickness(body.d_surf)[i], self.pads[i, 0].flatten(1),
                rotation_quaternion(self.command.angle[i, 0]),
                self.result.table_clearance[i], self.state.contacts_A[i, :, ContactState.CAPTURE],
            ), dim=1))
            self.stages.append(2 if lift else self.stage)
        self.ticks += 1


@torch.no_grad()
def record(run, output):
    results = json.loads((run / "result.json").read_text())
    corner = torch.arange(4096, device="cuda") % 8
    episode = CornerSuite(DomainRandomizer("cuda", 88003)).select(corner)
    candidates = torch.tensor([2, *range(5, 101, 8)], device="cuda")
    env = Recorder(episode, candidates)
    p = AsymmetricPPO(env.static_size, env.seq_size, env.priv_size, env.decisions).cuda().eval()
    p.load_state_dict(torch.load(run / "best.pt", map_location="cuda", weights_only=True)["model"])
    obs = env.reset()
    choices = []
    for _ in range(6):
        if env.done.all():
            break
        env.stage = 1 if bool((obs.phase == 0).any()) else 3
        action = p.act(obs, deterministic=True).action
        choices.append(action.choice[candidates])
        obs = env.step(action).obs
    assert env.done.all()
    actual = {f"C{i+1}": int(env.success[corner == i].sum()) for i in range(8)}
    expected = {k: v["success"]["successes"] for k, v in results["corners"].items()}
    if actual != expected:
        raise RuntimeError(f"Checkpoint replay differs from the corner holdout: {actual} vs {expected}")
    selection = [0, 1 + int(torch.where(~env.success[candidates[1:]])[0][0])]
    rows = candidates[selection]
    assert env.success[rows].tolist() == [True, False]
    trace = torch.stack(env.recorded).cpu().numpy()[:, selection]
    stage = np.asarray(env.stages)
    half = episode.payload.bag_half_extents[rows].cpu().numpy()
    if not np.isfinite(trace).all() or (stage == 2).sum() not in (13, 14):
        raise RuntimeError("Expected finite replay poses and a 0.55 s probe window")
    assert np.max(trace[stage == 2, :, 30]) <= 1e-6
    summary = dict(rows=rows.tolist(), successes=actual, selected_success=env.success[rows].tolist(),
                   selected_dropped=env.dropped[rows].tolist(),
                   choices=torch.stack(choices)[:, selection].tolist(), frames=len(trace), fps=FPS,
                   checkpoint_sha256=results["checkpoint_sha256"], seed=88004)
    np.savez_compressed(output / "trajectory.npz", trace=trace, stage=stage, half_extents=half)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return trace, stage, half


def scene():
    blocks = "".join(f'<body name="block{i}" mocap="true" pos="{i} 0 0">'
                     f'<geom name="block{i}" type="box" size=".04 .04 .016" rgba="{color} 1"/>'
                     '</body>' for i, color in enumerate((".73 .49 .26", ".79 .55 .31", ".77 .52 .28", ".70 .46 .23")))
    pads = "".join(f'<body name="pad{i}" mocap="true"><geom type="box" size=".008 .018 .022" '
                   f'pos="0 0 .022" rgba="{color} 1"/></body>'
                   for i, color in enumerate((".27 .72 .98", ".98 .44 .26")))
    xml = f"""<mujoco model="Policy 1 corner replay">
    <visual><global offwidth="640" offheight="492"/>
    <headlight ambient=".53 .53 .53" diffuse=".68 .68 .68" specular=".14 .14 .14"/></visual>
    <asset><texture name="grid" type="2d" builtin="checker" width="256" height="256"
    rgb1=".20 .24 .26" rgb2=".23 .28 .31"/>
    <material name="floor" texture="grid" texrepeat="9 9" texuniform="true" reflectance=".06"/></asset>
    <worldbody>
    <light directional="true" pos="-.2 -.35 .65" dir=".15 .2 -1"/>
    <geom type="plane" size=".6 .6 .02" material="floor"/>
    {blocks}{pads}</worldbody></mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def render(trace, stages, half, output):
    width, height = PANEL
    canvas_size = (2 * width, 584)
    fonts = [ImageFont.truetype(FONT, n) for n in (26, 20, 16)]
    model = scene()
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 113., -21., .92
    camera.lookat[:] = [0., 0., .144]
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-video_size", f"{canvas_size[0]}x{canvas_size[1]}", "-framerate", str(FPS), "-i", "-",
               "-c:v", "libx264", "-crf", "19", "-threads", "2", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(output / "policy1-c3-c6.mp4")]
    with mujoco.Renderer(model, height=height, width=width) as renderer, subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        for t in range(len(trace) + FPS):
            n = min(t, len(trace) - 1)
            canvas = Image.new("RGB", canvas_size, "#151b23")
            draw = ImageDraw.Draw(canvas)
            for i, row in enumerate(trace[n]):
                offset = width * i
                model.geom_size[1:5, :2] = .5 * half[i, :2]
                model.geom_size[1:5, 2] = row[16:20]
                data.mocap_pos[:4] = row[:12].reshape(4, 3)
                data.mocap_quat[:4] = row[12:16]
                data.mocap_pos[4:] = row[20:26].reshape(2, 3)
                data.mocap_quat[4:] = row[26:30]
                mujoco.mj_fwdPosition(model, data)
                renderer.update_scene(data, camera=camera)
                canvas.paste(Image.fromarray(renderer.render()), (offset, 56))
                draw.text((offset+18, 11), MASS[i], font=fonts[0], fill="#f0f5f9")
                status = ("稳悬成功", "提拉失败")[i] if t >= len(trace) else STAGES[stages[n]]
                color = "#99edb8" if i == 0 else "#ffd1a6"
                draw.text((offset+18, 551), f"{status}  ·  净空 {row[30]*1000:5.1f} mm  ·  双指捕获 {int(row[31:33].sum())}/2",
                          font=fonts[1], fill=color)
            draw.line((width, 56, width, 548), fill="#151b23", width=4)
            draw.text((width-126, 64), f"{n/FPS:05.2f} s", font=fonts[2], fill="#e8ecf1")
            encoder.stdin.write(canvas.tobytes())
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError("ffmpeg failed to encode the replay")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("checkpoint_directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    trace, stages, half = record(args.checkpoint_directory, args.output)
    render(trace, stages, half, args.output)


if __name__ == "__main__":
    main()
