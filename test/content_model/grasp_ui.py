"""Local parameter editing and live frames of the shared YAM GPU grasp trial.

Start with .venv-continuum/bin/python -B test/content_model/grasp_ui.py.
The web process never advances dynamics. Each submitted run is a separate,
GPU-locked invocation of grasp_platform.run with a read-only image observer.
"""
import argparse
from dataclasses import asdict, replace
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import threading
import time
import traceback
from uuid import uuid4
import zlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "contents"))
sys.path.insert(0, str(ROOT / "test/flex3d"))
sys.path.insert(0, str(ROOT))
from continuum.model import Parameters, Template, geometry, preset, require_mujoco4, shape_parameters

RANGES = dict(young=[1000, 300000], poisson=[0, .45], damping=[0, .1],
              yield_strain=[.001, .2], density=[30, 1000], friction=[0, 2],
              roundness=[0, 1], bulge=[0, .8], taper=[-.5, .5], yaw=[0, 180], tilt=[-30, 30])
SIZE_RANGES = [[.08, .5], [.06, .35], [.015, .15]]


def clean(value):
    if isinstance(value, dict): return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, allow_nan=False, indent=2))
    temporary.replace(path)


def parse_settings(payload):
    if not isinstance(payload, dict): raise ValueError("参数必须是对象")
    unknown = set(payload) - (set(Parameters.__dataclass_fields__) | {"material", "shape", "yaw", "tilt"})
    if unknown: raise ValueError(f"未知参数: {sorted(unknown)}")
    material, shape = payload.get("material", "hard"), payload.get("shape", "box")
    p = replace(preset(material), **shape_parameters(shape))
    values = asdict(p) | {k: v for k, v in payload.items() if k in Parameters.__dataclass_fields__}
    for name, limits in RANGES.items():
        value = payload.get(name, 0.) if name in ("yaw", "tilt") else values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} 必须为有限数值")
        if not limits[0] <= value <= limits[1]: raise ValueError(f"{name} 超出界面范围 {limits}")
    size = values["size"]
    if not isinstance(size, (list, tuple)) or len(size) != 3: raise ValueError("尺寸需要长、宽、高")
    for v, limits in zip(size, SIZE_RANGES):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not limits[0] <= v <= limits[1]:
            raise ValueError(f"尺寸超出界面范围 {SIZE_RANGES} m")
    p = Parameters(**values)
    geometry(p, Template())
    return dict(material=material, shape=shape, yaw=payload.get("yaw", 0.), tilt=payload.get("tilt", 0.), **asdict(p))


def meta():
    return dict(presets={name: asdict(preset(name)) for name in ("hard", "bendable", "compressible")},
                shapes={name: shape_parameters(name) for name in ("box", "rounded", "pillow", "tapered")},
                ranges=RANGES, size_ranges=SIZE_RANGES, template=asdict(Template()),
                default=parse_settings({}), model="MuJoCo 4 · full 3D StVK / J₂ · GPU Newton")


def png_bytes(rgb):
    height, width, _ = rgb.shape
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    scanlines = b"".join(b"\0" + row.tobytes() for row in rgb)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(scanlines, 2)) + chunk(b"IEND", b""))


class Frames:
    def __init__(self, directory):
        self.directory, self.last, self.serial, self.renderer = directory, 0., 0, None

    def __call__(self, robot, sample):
        if time.monotonic() - self.last < .6: return
        self.draw(robot, sample)

    def draw(self, robot, sample):
        import mujoco
        import numpy as np
        from mujoco.rendering.classic import gl_context
        from render_video import SoftwareContext
        gl_context.GLContext = SoftwareContext
        sim = robot.sim
        if self.renderer is None:
            sim.m.vis.global_.offwidth, sim.m.vis.global_.offheight = 640, 480
            sim.m.vis.headlight.ambient[:] = .55
            sim.m.vis.headlight.diffuse[:] = .55
            self.renderer = mujoco.Renderer(sim.m, 480, 640)
            self.option = mujoco.MjvOption()
            self.option.geomgroup[3] = False
            self.option.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = False
            self.full, self.detail = mujoco.MjvCamera(), mujoco.MjvCamera()
            self.full.lookat[:] = [0, -.02, .20]
            self.full.distance, self.full.azimuth, self.full.elevation = 1.35, -45, -25
            self.detail.distance, self.detail.azimuth, self.detail.elevation = .67, -45, -30
        self.detail.lookat[:] = sim.d.flexvert_xpos.mean(0)
        images = []
        for camera in (self.full, self.detail):
            self.renderer.update_scene(sim.d, camera=camera, scene_option=self.option)
            self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            images.append(self.renderer.render().copy())
        target = self.directory / "live.png"
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(png_bytes(np.concatenate(images, axis=1)))
        temporary.replace(target)
        self.serial += 1
        sample = {"mass_kg": float(sim.m.body_mass[sim.m.flex_vertbodyid].sum()), **sample}
        write_json(self.directory / "live.json", dict(serial=self.serial, sample=sample,
                   renderer=self.renderer._gl_context.renderer))
        self.last = time.monotonic()

    def close(self):
        if self.renderer is not None: self.renderer.close()


def worker(mode, directory):
    # Set the software-rendering environment before importing MuJoCo GL modules.
    import render_video
    import continuum.grasp_adapter as backend
    from grasp_platform import run
    settings = parse_settings(json.loads((directory / "settings.json").read_text()))
    category = "compressible" if settings["material"] == "compressible" else "bendable"
    cfg = backend.Config(**settings, category=category, dt=.0005, pinch_coeff=0., wall_limit=600., simulation_seconds=10.)
    frames = Frames(directory)
    try:
        if mode == "preview":
            robot = backend.scene(cfg)
            frames.draw(robot, dict(t=0., phase="preview", **backend.dimensions(robot.sim)))
            result = dict(outcome="geometry_preview", reason="初始几何与机械臂姿态；未推进动力学", config=asdict(cfg),
                          dimensions=backend.dimensions(robot.sim))
        else:
            result = run(backend, cfg, directory / "trial", on_sample=frames)
    except Exception as error:
        result = dict(outcome="execution_error", reason=str(error), traceback=traceback.format_exc(), config=asdict(cfg))
    finally:
        frames.close()
    write_json(directory / "result.json", result)
    if mode == "run" and (directory / "trial/trajectory.npz").exists():
        completed = subprocess.run([sys.executable, "-B", str(ROOT / "test/flex3d/render_video.py"),
            str(directory / "trial"), str(directory / "grasp.mp4"), "--fps", "15"], cwd=ROOT)
        if completed.returncode: result["render_error"] = "录像导出失败；实时采样图与原始轨迹已保留"
    write_json(directory / "done.json", result)


class Jobs:
    def __init__(self, directory):
        self.directory, self.lock, self.active = directory, threading.Lock(), None
        self.process = None

    def start(self, mode, settings):
        if mode not in ("preview", "run"): raise ValueError("未知操作")
        settings = parse_settings(settings)
        with self.lock:
            if self.process is not None and self.process.poll() is None: raise RuntimeError("当前任务仍在运行")
            job_id = datetime.now().strftime("%H%M%S") + "-" + uuid4().hex[:8]
            folder = self.directory / job_id
            folder.mkdir(parents=True)
            write_json(folder / "settings.json", settings)
            write_json(folder / "job.json", dict(id=job_id, mode=mode, created=datetime.now().isoformat()))
            command = [sys.executable, "-u", "-B", str(Path(__file__).resolve()), "--worker", mode, "--output", str(folder)]
            if mode == "run": command = ["flock", "/tmp/claude-gpu.lock", *command]
            with (folder / "worker.log").open("w") as log:
                self.process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            self.active = job_id
        return self.snapshot(job_id)

    def snapshot(self, job_id=None):
        job_id = job_id or self.active
        if job_id is None: return dict(status="idle")
        if not re.fullmatch(r"\d{6}-[0-9a-f]{8}", job_id): raise ValueError("无效任务编号")
        folder = self.directory / job_id
        if not (folder / "job.json").exists(): raise FileNotFoundError("任务不存在")
        info = json.loads((folder / "job.json").read_text())
        info["settings"] = json.loads((folder / "settings.json").read_text())
        info["status"] = "running" if info["mode"] == "preview" or (folder / "trial").exists() else "waiting_gpu"
        for name in ("live", "result", "done"):
            if (folder / f"{name}.json").exists(): info[name] = json.loads((folder / f"{name}.json").read_text())
        if "result" in info: info["status"] = "rendering" if info["mode"] == "run" else "done"
        if "done" in info: info["status"] = "done"
        if job_id == self.active and self.process is not None and self.process.poll() is not None and "done" not in info:
            info["status"], info["error"] = "error", "工作进程异常退出，详见日志"
        info["image"] = f"/files/{job_id}/live.png" if (folder / "live.png").exists() else None
        info["video"] = f"/files/{job_id}/grasp.mp4" if "done" in info and (folder / "grasp.mp4").exists() else None
        info["log"] = f"/files/{job_id}/worker.log"
        info["result_url"] = f"/files/{job_id}/result.json" if "result" in info else None
        return clean(info)


def handler_for(jobs):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_): pass

        def respond(self, value, status=200, mime="application/json; charset=utf-8"):
            body = value if isinstance(value, bytes) else json.dumps(clean(value), ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try: self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError): pass

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            try:
                if path == "/": self.respond(Path(__file__).with_suffix(".html").read_bytes(), mime="text/html; charset=utf-8")
                elif path == "/api/meta": self.respond(meta())
                elif path == "/api/state": self.respond(jobs.snapshot())
                elif path == "/api/history":
                    self.respond([jobs.snapshot(p.parent.name) for p in sorted(jobs.directory.glob("*/job.json"), reverse=True)][:30])
                elif path.startswith("/api/jobs/"): self.respond(jobs.snapshot(path.rsplit("/", 1)[-1]))
                elif match := re.fullmatch(r"/files/(\d{6}-[0-9a-f]{8})/(live.png|grasp.mp4|result.json|worker.log)", path):
                    file = jobs.directory / match[1] / match[2]
                    mime = {".png": "image/png", ".mp4": "video/mp4", ".json": "application/json", ".log": "text/plain; charset=utf-8"}[file.suffix]
                    self.respond(file.read_bytes(), mime=mime)
                else: self.respond(dict(error="未找到"), 404)
            except (FileNotFoundError, ValueError) as error: self.respond(dict(error=str(error)), 404)

        def do_POST(self):
            try:
                origin = self.headers.get("Origin")
                if origin and origin != "http://" + self.headers.get("Host", ""): raise ValueError("需要同源请求")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8192: raise ValueError("请求大小无效")
                payload = json.loads(self.rfile.read(length))
                if self.path != "/api/jobs" or not isinstance(payload, dict): raise ValueError("无效操作")
                self.respond(jobs.start(payload.get("mode"), payload.get("settings")), 202)
            except RuntimeError as error: self.respond(dict(error=str(error)), 409)
            except (ValueError, TypeError) as error: self.respond(dict(error=str(error)), 400)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--output", type=Path, default=ROOT / f"test/runs/{datetime.now():%Y-%m-%d}/contents/continuum-ui")
    parser.add_argument("--worker", choices=("preview", "run"))
    args = parser.parse_args()
    require_mujoco4()
    if args.worker: return worker(args.worker, args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(Jobs(args.output.resolve())))
    print(f"连续体调参与机械臂抓取: http://127.0.0.1:{server.server_port}", flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__ == "__main__": main()
