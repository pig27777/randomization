"""Render a recorded YAM grasp beside a close-up of one material edge."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experience"))
from yam_platform.render_video import SoftwareContext  # noqa: E402
import mujoco  # noqa: E402
from mujoco.rendering.classic import gl_context  # noqa: E402
import numpy as np  # noqa: E402
from view_rss2027_fold import load_yam_recording, show_pose  # noqa: E402


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--edge", type=int, nargs=2, required=True)
    parser.add_argument("--start", type=float, default=.30)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--highlight-time", type=float, required=True)
    parser.add_argument("--detail-azimuth", type=float, default=-55., help="close-up camera azimuth in degrees")
    parser.add_argument("--detail-elevation", type=float, default=-24., help="close-up camera elevation in degrees")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    if not np.isfinite([args.detail_azimuth, args.detail_elevation]).all() or not -90 <= args.detail_elevation <= 90:
        parser.error("close-up camera angles must be finite and elevation in [-90, 90] degrees")
    if args.output.exists():
        raise FileExistsError(args.output)
    model, data, poses, times, _, rows = load_yam_recording(args.run)
    if not times[0] <= args.start < args.highlight_time < args.end <= times[-1]:
        parser.error("start, highlight and end must be ordered times within the recording")
    edge = np.asarray(args.edge)
    faces = model.flex_elem.reshape(-1, 3)
    adjacent = faces[np.isin(faces, edge).sum(axis=1) == 2]
    if len(adjacent) != 2:
        parser.error("edge must belong to two recorded membrane triangles")

    os.environ.setdefault("LP_NUM_THREADS", "4")
    gl_context.GLContext = SoftwareContext
    width, height, fps = 640, 512, 25
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    option = mujoco.MjvOption()
    option.geomgroup[3] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_FLEXFACE] = True
    detail_option = mujoco.MjvOption()
    detail_option.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = False
    detail_option.flags[mujoco.mjtVisFlag.mjVIS_FLEXFACE] = True
    # Keep the real pads visible in the close-up while removing the palm and
    # other rigid bodies that otherwise occlude the membrane edge.
    for index in range(model.ngeom):
        geom = model.geom(index)
        if geom.group[0] < 3 and not geom.name.startswith(("left_pad", "right_pad")):
            geom.group[0] = 5
    option.geomgroup[5] = True
    full = mujoco.MjvCamera()
    full.azimuth, full.elevation, full.distance = -61., -26., .59
    detail = mujoco.MjvCamera()
    detail.azimuth, detail.elevation, detail.distance = args.detail_azimuth, args.detail_elevation, .20
    args.output.parent.mkdir(parents=True, exist_ok=True)
    font = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    filters = ",".join(f"drawtext=fontfile={font}:text={title}:x={x}:y=16:fontsize=25:"
                       "fontcolor=white:box=1:boxcolor=black@0.62:boxborderw=8"
                       for title, x in (("YAM 侧夹与提拉", 18), ("褶皱近景", width + 18)))
    command = [args.ffmpeg, "-n", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
               "-pixel_format", "rgb24", "-video_size", f"{2 * width}x{height}",
               "-framerate", str(fps), "-i", "pipe:0", "-vf", filters, "-c:v", "libx264",
               "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "19", "-movflags", "+faststart", str(args.output)]
    rendered_times = args.start + np.arange(round((args.end - args.start) * fps) + 1) / fps
    rendered_times = np.minimum(rendered_times, args.end)
    tips = np.asarray([row["tip_mid"] for row in rows])
    renderer = mujoco.Renderer(model, height=height, width=width)
    software_renderer = renderer._gl_context.renderer
    try:
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            with encoder.stdin:
                for t in rendered_times:
                    index = int(np.clip(np.searchsorted(times, t) - 1, 0, len(times) - 2))
                    fraction = np.clip((t - times[index]) / (times[index + 1] - times[index]), 0, 1)
                    show_pose(model, data, poses[index] * (1 - fraction) + poses[index + 1] * fraction)
                    xyz = data.flexvert_xpos[edge]
                    midpoint = xyz.mean(axis=0)
                    tip = tips[index] * (1 - fraction) + tips[index + 1] * fraction
                    full.lookat[:] = .55 * tip + .45 * data.flexvert_xpos.mean(axis=0)
                    detail.lookat[:] = midpoint
                    renderer.update_scene(data, camera=full, scene_option=option)
                    left = renderer.render().copy()
                    renderer.update_scene(data, camera=detail, scene_option=detail_option)
                    if t >= args.highlight_time - .35:
                        faces_on_edge = data.flexvert_xpos[adjacent]
                        normals = np.cross(faces_on_edge[:, 1] - faces_on_edge[:, 0],
                                           faces_on_edge[:, 2] - faces_on_edge[:, 0])
                        outward = normals.sum(axis=0)
                        # Keep the visual marker just outside the opaque membrane.
                        marker_points = xyz + .001 * outward / max(np.linalg.norm(outward), 1e-12)
                        marker = renderer.scene.geoms[renderer.scene.ngeom]
                        mujoco.mjv_initGeom(marker, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                                            np.eye(3).ravel(), np.array([1., .72, .18, 1.]))
                        marker.emission = .7
                        mujoco.mjv_connector(marker, mujoco.mjtGeom.mjGEOM_CAPSULE, .00075,
                                             marker_points[0], marker_points[1])
                        renderer.scene.ngeom += 1
                    right = renderer.render()
                    encoder.stdin.write(np.concatenate((left, right), axis=1).tobytes())
            if encoder.wait() != 0:
                raise RuntimeError("Video encoding failed")
    finally:
        renderer.close()

    evidence = []
    for t in (args.start, args.highlight_time):
        index = int(np.argmin(np.abs(times - t)))
        show_pose(model, data, poses[index])
        triangles = data.flexvert_xpos[adjacent]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        row = rows[index]
        evidence.append(dict(t_s=float(times[index]), phase=row["phase"],
                             edge_angle_deg=float(np.rad2deg(np.arccos(np.clip(normals[0] @ normals[1], -1., 1.)))),
                             support_N=row["support_N"], gripper_gap_m=row["gap"],
                             payload_intersections=row["payload_geometry"]["intersecting_triangles"]))
    metadata = dict(source=str(args.run), edge=args.edge, start_s=args.start, end_s=args.end, frames=len(rendered_times),
                    fps=fps, resolution=[2 * width, height], renderer=software_renderer, speed=1.,
                    physics_recomputed=False, interpolated_between_saved_poses=True,
                    highlight_is_render_only=True, detail_rigid_geometry="pads only",
                    detail_camera_deg=[args.detail_azimuth, args.detail_elevation], evidence=evidence,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(args.output, software_renderer, flush=True)


if __name__ == "__main__":
    main()
