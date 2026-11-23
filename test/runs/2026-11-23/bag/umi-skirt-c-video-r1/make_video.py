"""Replay the calibrated skirt C package in the shared YAM studio."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "umi-skirt-c-r1"
sys.path.insert(0, str(SOURCE))
from model_c import BAG, CENTER, CLOTH, lengths, make_scene, mj, pad_faces, template
from project import audit, project
from render import render
from compose import stamp

FPS = 25
FRAME_COUNT = 240


def prepared_frames(frame_ids=range(FRAME_COUNT + 1)):
    with np.load(SOURCE / "right.npz") as data:
        proxy = dict(data)
    with np.load(SOURCE / "skirt_hem_replay.npz") as data:
        native = dict(data)
    ref, faces, edges, sides, rings, seam, initial = template()
    np.testing.assert_array_equal(native["F"], faces)
    np.testing.assert_allclose(native["V_frames"][0], initial, atol=5e-7)
    rest = lengths(initial, edges)
    flat = lengths(ref, edges)
    top = sides > 0
    patch = top & (rings <= .066)
    radial = np.abs((ref[:, :2] - CENTER) / (BAG / 2))
    radial_distance = radial.max(axis=1)
    short_axis = radial[:, 0] / np.maximum(radial_distance, 1e-10)
    edge_height = -.0055 + .0076 * short_axis**4
    theta = np.arctan2(ref[:, 1] - CENTER[1], ref[:, 0] - CENTER[0])
    folds = .0023 * (1 - radial_distance) * radial_distance * np.cos(8 * theta)
    blocks_rest = proxy["centers"][0]
    offsets = blocks_rest - blocks_rest.mean(axis=0)
    distances = (ref[:, None, :2] - CENTER) - offsets[None, :, :2]
    sigma = .45 * np.linalg.norm(np.ptp(blocks_rest, axis=0))
    weights = np.exp(-.5 * np.sum(distances**2, axis=2) / sigma**2)
    weights /= weights.sum(axis=1, keepdims=True)

    sequence, diagnostics = [], []
    for i in frame_ids:
        centers, rotation, thickness, pads, captured = (
            proxy[key][i] for key in ("centers", "frame", "half_height", "pads", "capture")
        )
        half = np.tile(CLOTH / [4, 4, 2], (4, 1))
        half[:, 2] = thickness
        plane = (centers + half[:, 2, None] * rotation[:, 2]).mean(axis=0)
        grip = (pads.mean(axis=0) - plane) @ rotation
        # Support the resting skin on the physical skirt. Once the captured
        # patch rises above the textile, transition to the calibrated folds.
        blend = np.clip((grip[2] - .001) / .034, 0, 1) if captured.all() else 0.
        blend = blend**2 * (3 - 2 * blend)
        resting = initial - [*CENTER, .0207]
        hanging = resting.copy()
        table_blend = np.clip(1 - proxy["clearance"][i] / .025, 0, 1)
        # Let the sides fold inward during the lift from support. Keep the
        # bottom centred on the skirt until there is room for it to swing.
        draw_in = .025 * np.clip((grip[2] - .015) / .014, 0, 1) * table_blend
        hanging[:, :2] *= [.983 - draw_in, .990]
        swing = np.where(top, 1., 1. - table_blend)
        hanging[:, :2] += grip[:2] * ((1 - radial_distance) * swing)[:, None]
        hanging[top, 2] = grip[2] * (1 - radial_distance[top]) + edge_height[top] * radial_distance[top] + folds[top]
        hanging[~top, 2] = initial[~top, 2] - .0211
        hanging[seam, 2] = edge_height[seam]
        hanging[patch] = grip + np.column_stack((ref[patch, :2] - CENTER, np.zeros(patch.sum())))
        local_vertices = (1 - blend) * resting + blend * hanging
        underside = (centers - plane) @ rotation
        underside_z = (underside[:, 2] - half[:, 2]).min() - .0008
        # Use a common underside plane through lift-off. A vertex-wise world
        # table clamp followed by separate block projections stretches short
        # adjacent edges as the four moving obstacles begin to unload.
        lower = (~top) & (rings < 1)
        local_vertices[lower, 2] -= table_blend * np.maximum(local_vertices[lower, 2] - underside_z, 0)
        vertices = local_vertices @ rotation.T + plane
        vertices[:, 2] = np.maximum(vertices[:, 2], .00008)
        displacement = centers - centers.mean(axis=0) - offsets @ rotation.T
        pinned = patch & (blend == 1)
        vertices[~pinned] += (weights @ displacement)[~pinned]
        vertices = project(vertices, ref, faces, edges, pinned, centers, rotation, half,
                           finite_contact=True, initial=initial)
        crossing, outside = audit(vertices, faces, centers, rotation, half)
        current = lengths(vertices, edges)
        diagnostics.append([np.max(current / flat), np.max(current / rest), crossing, outside, vertices[:, 2].min()])
        sequence.append(vertices)
    rendered = np.asarray(sequence)
    if 200 in frame_ids:
        with np.load(SOURCE / "box-fit.npz") as accepted:
            index = list(frame_ids).index(200)
            np.testing.assert_allclose(rendered[index], accepted["V"], atol=4e-7, rtol=0)
    return native, proxy, rendered, np.asarray(diagnostics), ref, faces, edges


def robot_frames(time, names):
    with np.load(SOURCE / "q_ref.npz") as path:
        np.testing.assert_allclose(time, path["time"][::20], atol=1e-10, rtol=0)
        angles = path["q_path"][::20]
    robot = make_scene()
    model, data = robot.sim.m, robot.kin
    body_ids = [model.body(str(name)).id for name in names]
    joints = np.r_[robot.qadr, robot.fingers]
    links, quaternions, pads = [], [], []
    for q in angles:
        data.qpos[joints] = q
        mj.mj_kinematics(model, data)
        links.append(data.xpos[body_ids].copy())
        quaternions.append(data.xquat[body_ids].copy())
        pads.append(pad_faces(robot, data))
    return tuple(map(np.asarray, (links, quaternions, pads)))


def prepare(out):
    native, proxy, right, diagnostics, ref, faces, edges = prepared_frames()
    left = native["V_frames"]
    time = proxy["time"]
    np.testing.assert_allclose(time, native["trace"][:, 0], atol=1e-10, rtol=0)
    step_ratio = float(native["peak"])
    failed = np.flatnonzero((diagnostics[:, :2].max(axis=1) > 1.05) | (diagnostics[:, 2:4].max(axis=1) != 0))
    if len(failed):
        raise ValueError(f"Proxy geometry at failed frames {failed.tolist()}: {diagnostics[failed].round(5)}")
    if step_ratio > 1.05:
        raise ValueError(f"Native film reached {step_ratio:.6f}")
    names = native["names"]
    links, quaternions, pads = robot_frames(time, names)
    np.testing.assert_allclose(pads, proxy["pads"], rtol=0, atol=1e-8)
    native_velocity = np.gradient(native["COM_frames"], 1 / FPS, axis=0)
    com_speed = np.stack((np.linalg.norm(native_velocity, axis=1), np.linalg.norm(proxy["velocity"], axis=1)))
    clearance = np.stack((left[:, :, 2].min(axis=1), right[:, :, 2].min(axis=1)))
    both_capture = native["tip_and_capture"][:, 3].astype(bool) & proxy["capture"].all(axis=1)
    closed = int(np.flatnonzero(both_capture)[0])
    lifted = int(np.flatnonzero(both_capture & (clearance.min(axis=0) > .005))[0])
    hold = (time >= 6.56 - 1e-8) & (time <= 9.56 + 1e-8)
    if not (both_capture[hold].all() and (clearance[:, hold] > .15).all()
            and (com_speed[:, hold] < .020).all() and (proxy["support"][hold] == 0).all()):
        raise ValueError("Incomplete carried hold")
    report = {
        "native_physics_max_edge_ratio": step_ratio,
        "proxy_video_max_ratio_flat": float(diagnostics[:, 0].max()),
        "proxy_video_max_ratio_packed": float(diagnostics[:, 1].max()),
        "proxy_max_intersections": int(diagnostics[:, 2].max()),
        "proxy_max_outside_corners": int(diagnostics[:, 3].max()),
        "proxy_min_table_clearance_mm": float(diagnostics[:, 4].min() * 1000),
        "hold_min_clearance_mm": (clearance[:, hold].min(axis=1) * 1000).tolist(),
        "hold_max_com_speed_mm_s": (com_speed[:, hold].max(axis=1) * 1000).tolist(),
        "keyframes": {"grasp": closed, "liftoff": lifted, "hold": 200},
        "fps": FPS,
        "frames": FRAME_COUNT,
    }
    print(json.dumps(report, indent=2))
    (out / "video_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(out / "frames.npz", left=left, right=right, faces=faces, ref=ref, names=names,
        links=links, quaternions=quaternions, pads=pads, time=time, com_speed=com_speed,
        clearance=clearance, proxy_geometry_checks=diagnostics, native_steps_s=native["steps_s"])


def render_side(out, side):
    if side == "left":
        with np.load(SOURCE / "skirt_hem_replay.npz") as a:
            names = a["names"]
            links, quaternions, pads = robot_frames(a["trace"][:, 0], names)
            render(out / "flex_raw.mp4", a["V_frames"], a["F"], links, quaternions, names, pads,
                   False, material_xy=a["ref"][:, :2])
        return
    with np.load(out / "frames.npz") as a:
        render(out / "proxy_skinned_raw.mp4", a["right"], a["faces"], a["links"],
               a["quaternions"], a["names"], a["pads"], False, material_xy=a["ref"][:, :2])


def compose(out):
    audit_data = json.loads((out / "video_audit.json").read_text())
    left_title = "High-Fidelity MuJoCo flex / Thin-Shell"
    right_title = "Ours: Low-Order Passive Proxy (Visual Skin)"
    rinfo = json.loads((SOURCE / "right.json").read_text())
    events = []
    def label(start, end, x, y, text, style):
        events.append(f"Dialogue: 0,{stamp(start)},{stamp(end)},{style},,0,0,0,,{{\\pos({x},{y})}}{text}")

    fps = FPS
    duration = FRAME_COUNT / fps
    with np.load(out / "frames.npz") as data:
        for x, title in [(24, left_title), (984, right_title)]:
            label(0, duration, x, 8, title, "Title")
        label(0, duration, 24, 43, f"UMI C  |  PE 39 × 29 × 2 cm  |  500 g  |  CPU, 1 env: {data['native_steps_s']:.0f} steps/s", "Note")
        label(0, duration, 984, 39, f"UMI C  |  CPU, 1 env: {rinfo['steps_s']:.0f} steps/s", "Note")
        label(0, duration, 984, 59, "Prior unloaded GPU benchmark   4,096 envs: 157k steps/s   |   30,000 envs: 636k steps/s", "Note")
        clearance = data["clearance"]
        speed = data["com_speed"]
        for i, t in enumerate(data["time"][:FRAME_COUNT]):
            phase = "Approach" if t < 1.56 else "Grasp" if t < 3.36 else "Lift" if t < 5.15 else "Hold"
            for side, x in enumerate((24, 984)):
                label(t, t + 1 / fps, x, 661, f"{t:04.2f} s   {phase.upper()}", "HUD")
                label(t, t + 1 / fps, x + 289, 661, f"Clearance  {max(0, 1000 * clearance[side, i]):.0f} mm", "HUD")
                label(t, t + 1 / fps, x + 598, 661, f"COM speed  {1000 * speed[side, i]:.1f} mm/s", "HUD")
    ass = out / "metrics.ass"
    styles = [
        f"Style: {name},Liberation Sans,{size},&H00F4F7F7,&H00FFFFFF,&H00000000,&H00000000,{bold},0,0,0,100,100,0,0,1,0,0,7,0,0,0,1"
        for name, size, bold in (("Title", 29, -1), ("Note", 18, 0), ("HUD", 24, -1))
    ]
    ass.write_text(
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 720\nScaledBorderAndShadow: yes\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        + "\n".join(styles)
        + "\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        + "\n".join(events) + "\n", encoding="utf-8")
    film = "compare_flex_vs_proxy_side_by_side.mp4"
    filters = ("[0:v][1:v]hstack=inputs=2,drawbox=x=0:y=0:w=iw:h=80:c=0x17242B:t=fill,"
               "drawbox=x=0:y=646:w=iw:h=74:c=0x17242B:t=fill,"
               "drawbox=x=959:y=0:w=2:h=ih:c=0x42535B:t=fill,subtitles=metrics.ass")
    subprocess.run(["ffmpeg", "-v", "error", "-n", "-i", "flex_raw.mp4", "-i", "proxy_skinned_raw.mp4",
                   "-filter_complex", filters, "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "16",
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", film], cwd=out, check=True)
    sheet = Image.new("RGB", (1280, 1440))
    for row, (name, i) in enumerate(audit_data["keyframes"].items()):
        file = out / f"{name}.png"
        subprocess.run(["ffmpeg", "-v", "error", "-n", "-i", film,
                        "-vf", f"select=eq(n\\,{i})", "-frames:v", "1", file.name], cwd=out, check=True)
        with Image.open(file) as image:
            sheet.paste(image.resize((1280, 480), Image.Resampling.LANCZOS), (0, 480 * row))
    sheet.save(out / "keyframes.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "left", "right", "compose"])
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.output)
    elif args.stage == "compose":
        compose(args.output)
    else:
        render_side(args.output, args.stage)
