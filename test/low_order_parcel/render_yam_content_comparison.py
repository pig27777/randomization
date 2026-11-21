"""Render the content layout and lift phase of the sampled MuJoCo trials."""

import argparse
import gzip
import json
from pathlib import Path
import subprocess
import sys
from contextlib import ExitStack

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experience.yam_platform.render_video import SoftwareContext  # noqa: E402
import mujoco  # noqa: E402
from mujoco.rendering.classic import gl_context  # noqa: E402

WIDTH, HEIGHT = 520, 335
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
PHASE_LABELS = {
    "settle": "准备", "approach": "接近与进给", "close": "闭合聚拢",
    "lift": "抬升", "lift_hold": "抬升后停留", "open": "松夹",
    "post_release": "释放后落桌",
}


def read_trial(run):
    trace_file = run / "trace.json"
    trace = json.loads(trace_file.read_bytes() if trace_file.is_file()
                       else gzip.decompress((run / "trace.json.gz").read_bytes()))
    result = json.loads((run / "result.json").read_bytes())
    with np.load(run / "trajectory.npz") as saved:
        poses, times = saved["qpos"], saved["time"]
    model = mujoco.MjModel.from_xml_path(str(run / "model.xml"))
    model.vis.global_.offwidth = WIDTH
    model.vis.global_.offheight = HEIGHT
    return trace, result, poses, times, model


def draw_trial(run):
    trace, result, poses, _, model = read_trial(run)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    option = mujoco.MjvOption()
    option.geomgroup[3] = False
    option.sitegroup[4] = False
    camera.lookat[:] = [-.25, .04, .02]
    camera.distance = .52
    camera.azimuth = 90.
    camera.elevation = -89.
    option.geomgroup[2] = False
    hold = [i for i, sample in enumerate(trace) if sample["phase"] == "lift_hold"]
    frame = hold[len(hold) // 2] if hold else len(trace) - 1

    with mujoco.Renderer(model, HEIGHT, WIDTH) as renderer:
        data.qpos[:] = poses[0]
        mujoco.mj_fwdPosition(model, data)
        renderer.update_scene(data, camera, option)
        initial = Image.fromarray(renderer.render().copy())
        option.geomgroup[2] = True
        camera.lookat[:] = [-.13, -.07, .24]
        camera.distance = 1.12
        camera.azimuth = 43.
        camera.elevation = -24.
        data.qpos[:] = poses[frame]
        mujoco.mj_fwdPosition(model, data)
        renderer.update_scene(data, camera, option)
        lift = Image.fromarray(renderer.render().copy())
    return initial, lift, trace[frame], result


def video_trial(run, stack):
    trace, result, poses, times, model = read_trial(run)
    tail = run / "post-release.npz"
    if tail.is_file():
        with np.load(tail) as saved:
            poses = np.concatenate((poses, saved["qpos"]))
            times = np.concatenate((times, saved["time"]))
            trace += [dict(phase="post_release", proxy_capture=[0., 0.], proxy_clearance_m=float(h))
                      for h in saved["clearance_m"]]
    data = mujoco.MjData(model)
    renderer = stack.enter_context(mujoco.Renderer(model, HEIGHT, WIDTH))
    option = mujoco.MjvOption()
    option.geomgroup[3] = False
    option.sitegroup[4] = False
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [-.14, -.08, .245]
    camera.distance = 1.24
    camera.azimuth = 43.
    camera.elevation = -24.
    colors = np.array([[.13, .78, 1., 1.], [1., .49, .17, 1.]], dtype=np.float32)
    origin, rotation = np.zeros(3), np.eye(3).ravel()

    def frame(t):
        i = int(np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 1))
        k = min(i + 1, len(times) - 1)
        blend = float(np.clip((t - times[i]) / (times[k] - times[i]), 0., 1.)) if k > i else 0.
        data.qpos[:] = (1. - blend) * poses[i] + blend * poses[k]
        mujoco.mj_normalizeQuat(model, data.qpos)
        mujoco.mj_fwdPosition(model, data)
        renderer.update_scene(data, camera, option)
        row = trace[i]
        for side, name in enumerate(("left", "right")):
            if row["proxy_capture"][side]:
                scene = renderer.scene
                line = scene.geoms[scene.ngeom]
                scene.ngeom += 1
                mujoco.mjv_initGeom(line, mujoco.mjtGeom.mjGEOM_CAPSULE, origin, origin, rotation, colors[side])
                mujoco.mjv_connector(line, mujoco.mjtGeom.mjGEOM_CAPSULE, .0016,
                                     data.site(f"proxy_{name}").xpos, np.asarray(row["proxy_pad_m"][side]))
        return Image.fromarray(renderer.render().copy()), row

    if tail.is_file():
        end_caption = "已释放并落桌"
    elif not any(result["model_metrics"]["proxy_capture_s"]):
        end_caption = "空夹结束"
    else:
        end_caption = "回合结束"
    return frame, result["dimensions"], float(times[-1]), end_caption


def render_video(run, samples):
    output = run / "yam-content-randomization.mp4"
    if output.exists():
        raise FileExistsError(output)
    margin = 22
    width, height = len(samples) * (WIDTH + margin) + margin, 552
    fps = 20
    title_font = ImageFont.truetype(FONT, 32, index=2)
    label_font = ImageFont.truetype(FONT, 22, index=2)
    text_font = ImageFont.truetype(FONT, 17, index=2)
    with ExitStack() as stack:
        trials = [video_trial(run / sample["name"], stack) for sample in samples]
        frames = round((max(trial[2] for trial in trials) + 1.) * fps) + 1
        template = Image.new("RGB", (width, height), "#17212b")
        draw = ImageDraw.Draw(template)
        draw.text((margin, 10), "YAM 侧抓  ·  内容物随机化", font=title_font, fill="#f1f5fa")
        draw.text((margin, 54), "同一抓取协议，青线标记袋口封边内的初始轮廓。",
                  font=text_font, fill="#b9c9d6")
        for j, (sample, (_, dims, _, _)) in enumerate(zip(samples, trials, strict=True)):
            x = margin + j * (WIDTH + margin)
            size = " × ".join(f"{value:.0f}" for value in dims["size_mm"])
            draw.text((x, 92), f"{sample['fill']:.0%} 填充率", font=label_font, fill="#f1f5fa")
            draw.text((x, 124), f"{size} mm  ·  {1000 * dims['mass_kg']:.0f} g  ·  余料上限 {dims['slack_budget_mm']:.1f} mm",
                      font=text_font, fill="#b9c9d6")
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                   "-video_size", f"{width}x{height}", "-framerate", str(fps), "-i", "pipe:0",
                   "-c:v", "libx264", "-threads", "4", "-preset", "fast", "-crf", "19",
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-n", str(output)]
        with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
            for number in range(frames):
                t = number / fps
                canvas = template.copy()
                label = ImageDraw.Draw(canvas)
                label.text((width - 174, 28), f"{t:05.2f} s", font=label_font, fill="#f1f5fa")
                for j, (frame, _, final_time, end_caption) in enumerate(trials):
                    x = margin + j * (WIDTH + margin)
                    picture, row = frame(t)
                    canvas.paste(picture, (x, 155))
                    if t > final_time:
                        state = end_caption
                    else:
                        captured = int(sum(row["proxy_capture"]))
                        state = f"{PHASE_LABELS[row['phase']]}  ·  承载指 {captured}/2  ·  离台 {1000 * row['proxy_clearance_m']:.0f} mm"
                    label.text((x, 506), state, font=text_font, fill="#f1f5fa")
                process.stdin.write(canvas.tobytes())
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("ffmpeg could not encode the MuJoCo comparison")
    print(output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    samples = json.loads((args.run / "samples.json").read_bytes())["samples"]
    gl_context.GLContext = SoftwareContext
    if args.video:
        render_video(args.run, samples)
        return
    gap = 22
    canvas = Image.new("RGB", (len(samples) * (WIDTH + gap) + gap, 928), "#17212b")
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.truetype(FONT, 33, index=2)
    label_font = ImageFont.truetype(FONT, 22, index=2)
    text_font = ImageFont.truetype(FONT, 18, index=2)
    draw.text((gap, 12), "YAM 侧抓  ·  内容物随机化", font=title_font, fill="#f1f5fa")
    draw.text((gap, 58), "上排为桌面初态，青线是封边内轮廓。下排为抬升后的同阶段画面。",
              font=text_font, fill="#b9c9d6")
    for j, sample in enumerate(samples):
        x = gap + j * (WIDTH + gap)
        initial, lift, frame, result = draw_trial(args.run / sample["name"])
        dims = result["dimensions"]
        size = " × ".join(f"{value:.0f}" for value in dims["size_mm"])
        dx, dy = np.asarray(sample["options"]["payload_offset_xy_m"]) * 1000.
        draw.text((x, 95), f"填充率 {sample['fill']:.0%}", font=label_font, fill="#f1f5fa")
        draw.text((x, 129), f"内容物 {size} mm  ·  {1000 * dims['mass_kg']:.0f} g",
                  font=text_font, fill="#b9c9d6")
        draw.text((x, 155), f"初始偏置 ({dx:+.0f}, {dy:+.0f}) mm  ·  余料上限 {dims['slack_budget_mm']:.1f} mm",
                  font=text_font, fill="#b9c9d6")
        canvas.paste(initial, (x, 190))
        canvas.paste(lift, (x, 540))
        if all(frame["proxy_capture"]):
            summary = f"双指承载  ·  离台 {1000 * frame['proxy_clearance_m']:.0f} mm"
        else:
            summary = "未夹住  ·  包裹留在桌面"
        draw.text((x, 889), summary, font=label_font, fill="#f1f5fa")
    target = args.run / "mujoco-comparison.png"
    if target.exists():
        raise FileExistsError(target)
    canvas.save(target)


if __name__ == "__main__":
    main()
