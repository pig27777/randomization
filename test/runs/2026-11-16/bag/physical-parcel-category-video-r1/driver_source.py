"""Record paper, film and bubble parcel categories with shared YAM grasps.

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
    parser.add_argument("--reuse-film-run", type=Path)
    parser.add_argument("--font", type=Path, default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "driver_source.py").write_bytes(Path(__file__).read_bytes())
    rng = random.Random(args.seed)
    cases = []
    variants = (("kraft", "纸袋", "side", "rigid"),
                ("pe", "塑料薄膜袋", "side", "rigid"),
                ("pe", "塑料薄膜袋", "top", "bendable"),
                ("bubble", "气泡信封袋", "side", "compressible"))
    for number, (profile, title, grasp, mode) in enumerate(variants, 1):
        folder = output / f"case-{number}-{profile}-{grasp}"
        folder.mkdir()
        if profile == "pe" and args.reuse_film_run is not None:
            reused = args.reuse_film_run.resolve() / ("case-1-side" if grasp == "side" else "case-2-top")
            cfg = json.loads((reused / "config.json").read_text())
            (folder / "config.json").write_text(json.dumps(cfg, indent=2))
            cases.append(dict(number=number, grasp=grasp, mode=mode, config=cfg, folder=str(folder),
                              title=title, reused_from=str(reused)))
            continue
        cfg = dict(geometry="geometry/kraft_edge_sealed_35.npz", mode=mode,
            bag_size=[rng.uniform(.32, .38), rng.uniform(.23, .27)],
            bag_profile=profile,
            friction=rng.uniform(.35, .65), bend_damping_ratio=rng.uniform(.4, .8),
            rayleigh=rng.uniform(.0003, .0008),
            payload_size=[rng.uniform(.10, .145), rng.uniform(.065, .095), rng.uniform(.02, .04)],
            payload_mass=rng.uniform(.12, .35), joint_stiffness=rng.uniform(.03, .15),
            dt=.0002, contact_time=.001, pinch_coeff=0., support_height=.15,
            lift_height=.35, gripper_overtravel=.005, simulation_seconds=8., wall_limit=3600.,
            grasp_face=grasp, yaw=90. if grasp == "side" else 0., tilt=15. if grasp == "side" else 0.,
            side_start_distance=.08, side_height_offset=0.,
            initial_state=str(folder / "preparation/initial_state.npz"))
        (folder / "config.json").write_text(json.dumps(cfg, indent=2))
        cases.append(dict(number=number, grasp=grasp, mode=mode, config=cfg, folder=str(folder), title=title))
    (output / "cases.json").write_text(json.dumps(dict(seed=args.seed, cases=cases,
        sampling="random bag size/contact/damping and content dimensions/mass; selected category and grasp coverage",
        material="existing kraft/pe/bubble engineering profiles; film recordings may be reused"), indent=2))

    clips = []
    summaries = []
    for case in cases:
        folder, cfg = Path(case["folder"]), case["config"]
        source = Path(case.get("reused_from", folder))
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
        if "reused_from" not in case:
            for label, arguments in zip(("prepare", "record", "render"), commands):
                print(f"case {case['number']} {case['title']}: {label}", flush=True)
                command(arguments, folder / f"{label}.log")
        result = json.loads((source / "recording/trial/result.json").read_text())
        trace = json.loads((source / "recording/trial/trace.json").read_text())
        lifted = max(row["com"][2] - result["initial_com"][2] for row in trace)
        hanging = [row for row in trace if row["support_N"] < .05 and row["com"][2]-result["initial_com"][2] >= .05]
        summary = dict(number=case["number"], grasp=case["grasp"], mode=case["mode"],
                       outcome=result["outcome"], reason=result["reason"],
                       peak_lift_m=lifted, free_hang_samples=len(hanging),
                       peak_penetration_mm=result["peaks"]["penetration_mm"],
                       simulation_s=result["sim_time_s"], wall_s=result["wall_s"])
        summaries.append(summary)
        phase_label = "侧抓" if case["grasp"] == "side" else "顶面搓起"
        outcome_label = {"grasp_fail": "抓取失败", "drop": "滑落", "incomplete": "录制窗口结束", "success": "完成搬运"}.get(result["outcome"], result["outcome"])
        lines = [f"{case['title']} · {phase_label}尝试 · {outcome_label}"]
        header = Image.new("RGB", (1280, 54), "#141920")
        draw = ImageDraw.Draw(header)
        font = ImageFont.truetype(str(args.font), 28)
        for index, line in enumerate(lines):
            draw.text((16, 7), line, font=font, fill="#eef1f4")
        header.save(folder / "category.png")
        render = json.loads((source / "raw_render.json").read_text())
        duration = render["frames"] / render["fps"]
        clip = folder / f"{case['grasp']}-randomized.mp4"
        command([args.ffmpeg, "-hide_banner", "-loglevel", "error", "-i", source / "raw.mp4",
            "-loop", "1", "-i", folder / "category.png", "-filter_complex",
            "[0:v]pad=iw:ih+54:0:54:black[v];[v][1:v]overlay=0:0", "-t", duration,
            "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", clip], folder / "caption.log")
        command([args.ffmpeg, "-hide_banner", "-loglevel", "error", "-i", clip, "-f", "null", "-"], folder / "decode.log")
        clips.append(clip)
        (output / "summary.json").write_text(json.dumps(dict(seed=args.seed, cases=summaries), indent=2))

    video = output / "randomized-bag-and-contents.mp4"
    inputs = [argument for clip in clips for argument in ("-i", clip)]
    streams = "".join(f"[{index}:v]" for index in range(len(clips)))
    command([args.ffmpeg, "-hide_banner", "-loglevel", "error", *inputs,
        "-filter_complex", f"{streams}concat=n={len(clips)}:v=1:a=0[v]", "-map", "[v]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", video], output / "combine.log")
    command([args.ffmpeg, "-hide_banner", "-loglevel", "error", "-i", video, "-f", "null", "-"], output / "decode.log")
    (output / "summary.json").write_text(json.dumps(dict(seed=args.seed, cases=summaries, video=str(video),
        video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(), full_decode_passed=True), indent=2))
    print(video, flush=True)


if __name__ == "__main__":
    main()
