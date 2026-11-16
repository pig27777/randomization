"""Record two seeded parcel variants with the shared side and top YAM grasps.

Run this driver in an existing Pillow environment and select the MuJoCo Python
with --python. Dynamics, preparation and rendering use the existing entry points.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]


def command(arguments, log):
    with log.open("w") as stream:
        subprocess.run([str(value) for value in arguments], cwd=ROOT,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=ROOT / ".venv-physical-parcel/bin/python")
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--font", type=Path, default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "driver_source.py").write_bytes(Path(__file__).read_bytes())
    rng = random.Random(args.seed)
    cases = []
    for number, (grasp, mode) in enumerate((("side", "rigid"), ("top", "bendable")), 1):
        folder = output / f"case-{number}-{grasp}"
        folder.mkdir()
        cfg = dict(geometry="geometry/kraft_edge_sealed_35.npz", mode=mode,
            bag_size=[rng.uniform(.32, .38), rng.uniform(.23, .27)],
            material=dict(young_pa=rng.uniform(4e6, 10e6), thickness_m=rng.uniform(.0001, .0002),
                          density_kg_m3=rng.uniform(700, 1000)),
            friction=rng.uniform(.35, .65), bend_damping_ratio=rng.uniform(.4, .8),
            rayleigh=rng.uniform(.0003, .0008),
            payload_size=[rng.uniform(.10, .145), rng.uniform(.065, .095), rng.uniform(.02, .04)],
            payload_mass=rng.uniform(.12, .35), joint_stiffness=rng.uniform(.03, .15),
            dt=.0001, contact_time=.001, pinch_coeff=0., support_height=.15,
            lift_height=.35, gripper_overtravel=.005, simulation_seconds=10., wall_limit=3600.,
            grasp_face=grasp, yaw=90. if grasp == "side" else 0., tilt=15. if grasp == "side" else 0.,
            side_start_distance=.08, side_height_offset=0.,
            initial_state=str(folder / "preparation/initial_state.npz"))
        (folder / "config.json").write_text(json.dumps(cfg, indent=2))
        cases.append(dict(number=number, grasp=grasp, mode=mode, config=cfg, folder=str(folder)))
    (output / "cases.json").write_text(json.dumps(dict(seed=args.seed, cases=cases,
        sampling="independent uniform bag/content parameters; one rigid side grasp and one bendable top grasp",
        material="Et, D and areal mass derived together from E, thickness, density; engineering presets"), indent=2))

    clips = []
    summaries = []
    for case in cases:
        folder, cfg = Path(case["folder"]), case["config"]
        preparation = dict(cfg)
        preparation.pop("initial_state")
        commands = [
            [args.python, "-u", "-B", ROOT / "test/bag_model/prepare_physical_parcel.py",
             "--output", folder / "preparation", "--seconds", ".5", "--model-options", json.dumps(preparation)],
            [args.python, "-u", "-B", ROOT / "test/bag_model/record_physical_parcel.py",
             "--config", folder / "config.json", "--output", folder / "recording"],
            [args.python, "-B", ROOT / "test/bag_model/render_physical_parcel_video.py",
             folder / "recording/trial", folder / "raw.mp4", "--ffmpeg", args.ffmpeg],
        ]
        (folder / "commands.json").write_text(json.dumps([[str(v) for v in row] for row in commands], indent=2))
        for label, arguments in zip(("prepare", "record", "render"), commands):
            print(f"case {case['number']} {case['grasp']}: {label}", flush=True)
            command(arguments, folder / f"{label}.log")
        result = json.loads((folder / "recording/trial/result.json").read_text())
        trace = json.loads((folder / "recording/trial/trace.json").read_text())
        lifted = max(row["com"][2] - result["initial_com"][2] for row in trace)
        hanging = [row for row in trace if row["support_N"] < .05 and row["com"][2]-result["initial_com"][2] >= .05]
        summary = dict(number=case["number"], grasp=case["grasp"], mode=case["mode"],
                       outcome=result["outcome"], reason=result["reason"],
                       peak_lift_m=lifted, free_hang_samples=len(hanging),
                       peak_penetration_mm=result["peaks"]["penetration_mm"],
                       simulation_s=result["sim_time_s"], wall_s=result["wall_s"])
        summaries.append(summary)
        material = cfg["material"]
        bag = " × ".join(f"{v*100:.1f}" for v in cfg["bag_size"])
        payload = " × ".join(f"{v*100:.1f}" for v in cfg["payload_size"])
        mode_label = "刚体" if case["mode"] == "rigid" else f"可弯曲，铰链 {cfg['joint_stiffness']:.3f} N·m/rad"
        phase_label = "侧抓" if case["grasp"] == "side" else "顶面搓起"
        outcome_label = {"grasp_fail": "抓取失败", "drop": "滑落", "incomplete": "录制窗口结束", "success": "完成搬运"}.get(result["outcome"], result["outcome"])
        lines = [f"随机组 {case['number']} · {phase_label} · CPU 1倍速 · 种子 {args.seed}",
            f"袋子 {bag} cm  |  E {material['young_pa']/1e6:.2f} MPa  |  厚度 {material['thickness_m']*1e6:.0f} μm  |  摩擦 {cfg['friction']:.2f}  |  弯曲阻尼 {cfg['bend_damping_ratio']:.2f}",
            f"内部物体 {payload} cm  |  {cfg['payload_mass']*1000:.0f} g  |  {mode_label}",
            f"实际最高抬升 {max(0.,lifted)*100:.1f} cm  |  {outcome_label}"]
        header = Image.new("RGB", (1280, 120), "#141920")
        draw = ImageDraw.Draw(header)
        font = ImageFont.truetype(str(args.font), 22)
        for index, line in enumerate(lines):
            draw.text((16, 3+29*index), line, font=font, fill="#84d3e6" if index == 0 else "#eef1f4")
        header.save(folder / "parameters.png")
        render = json.loads((folder / "raw_render.json").read_text())
        duration = render["frames"] / render["fps"]
        clip = folder / f"{case['grasp']}-randomized.mp4"
        command([args.ffmpeg, "-hide_banner", "-loglevel", "error", "-i", folder / "raw.mp4",
            "-loop", "1", "-i", folder / "parameters.png", "-filter_complex",
            "[0:v]pad=iw:ih+120:0:120:black[v];[v][1:v]overlay=0:0", "-t", duration,
            "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", clip], folder / "caption.log")
        command([args.ffmpeg, "-hide_banner", "-loglevel", "error", "-i", clip, "-f", "null", "-"], folder / "decode.log")
        clips.append(clip)
        (output / "summary.json").write_text(json.dumps(dict(seed=args.seed, cases=summaries), indent=2))

    video = output / "randomized-bag-and-contents.mp4"
    command([args.ffmpeg, "-hide_banner", "-loglevel", "error", "-i", clips[0], "-i", clips[1],
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", video], output / "combine.log")
    command([args.ffmpeg, "-hide_banner", "-loglevel", "error", "-i", video, "-f", "null", "-"], output / "decode.log")
    (output / "summary.json").write_text(json.dumps(dict(seed=args.seed, cases=summaries, video=str(video),
        video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(), full_decode_passed=True), indent=2))
    print(video, flush=True)


if __name__ == "__main__":
    main()
