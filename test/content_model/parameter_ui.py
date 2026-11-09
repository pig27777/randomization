"""Local, CPU-only response-parameter preview of frozen v1.3 contents.

Run from the repository root: .venv/bin/python -B test/content_model/parameter_ui.py
This is a material preview with prescribed loads/boundaries, not a grasp evaluator.
"""
import argparse
from dataclasses import asdict, dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import sys
import threading
import time
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bag_sim.frozen_content import activate

RELEASE_SHA = activate()
import mj_adapter as A
from model import ANCHORS, build, load_table, table_entry

DT = .0004
FIELDS = {'ky': 'squeeze_stiffness_y', 'kz': 'squeeze_stiffness_z',
          'droop': 'droop_ratio', 'damping': 'damping_ratio'}


def presets():
    result = {}
    for category in ('bendable', 'compressible'):
        k = ANCHORS[category]
        entry = table_entry(k, load_table())
        ranges = {}
        for field, column in (('ky', 'squeeze_y'), ('kz', 'squeeze_z'), ('droop', 'droop')):
            values = [row[0] for row in entry[column]]
            ranges[field] = [min(values), max(values)]
        ranges.update(damping=[.05, 1.5], force=[0., 15.])
        result[category] = dict(defaults={key: getattr(k, name) for key, name in FIELDS.items()},
                                ranges=ranges, size=list(k.size), mass=k.mass,
                                segments=list(k.segments))
    return result


PRESETS = presets()


@dataclass
class Settings:
    category: str = 'bendable'
    mode: str = 'squeeze_y'
    ky: float = .5
    kz: float = .3
    droop: float = .32
    damping: float = .5
    force: float = 5.

    @classmethod
    def parse(cls, payload):
        if not isinstance(payload, dict):
            raise ValueError('参数必须为对象')
        unknown = set(payload) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f'未知参数: {sorted(unknown)}')
        cfg = cls(**payload)
        if cfg.category not in PRESETS or cfg.mode not in ('squeeze_y', 'squeeze_z', 'sag'):
            raise ValueError('未知的内容物或预览动作')
        for field, bounds in PRESETS[cfg.category]['ranges'].items():
            value = getattr(cfg, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{field} 必须为有限数值')
            if not bounds[0] <= value <= bounds[1]:
                raise ValueError(f'{field} 超出映射范围 {bounds}')
        return cfg


class Preview:
    def __init__(self, cfg):
        self.cfg = cfg
        k = replace(ANCHORS[cfg.category], **{field: getattr(cfg, key) for key, field in FIELDS.items()})
        desc = build(k, omega_max=.8 / DT)
        self.origin = np.array([0., 0., .30 if cfg.mode == 'sag' else .002 + k.size[2] / 2])
        supports = ''
        if cfg.mode == 'sag':
            supports = ''.join(f'<weld name="preview_support_{i}" body1="s{i}" solref=".002 2"/>'
                               for i, s in enumerate(desc.segments) if s.idx[0] == 0)
        built = A.scene(desc, self.origin, table=cfg.mode != 'sag', dt=DT, equality=supports)
        self.sim = A.Sim(built)
        self.reference = self.sim.seg_pos()
        self.half = np.array([s.half for s in desc.segments])
        self.layers = np.array([s.idx for s in desc.segments])
        self.failure = None
        self.load = 0.
        self.history = []

    def advance(self, steps=25):
        sim, cfg = self.sim, self.cfg
        for _ in range(steps):
            if self.failure or sim.d.time >= 3.:
                break
            self.load = cfg.force * min(1., sim.d.time / .3)
            sim.d.xfrc_applied[:] = 0
            if cfg.mode == 'squeeze_y':
                for layer, sign in ((0, 1), (sim.desc.kappa.segments[1] - 1, -1)):
                    selected = np.flatnonzero(self.layers[:, 1] == layer)
                    sim.d.xfrc_applied[np.array(sim.seg_bodies)[selected], 1] = sign * self.load / len(selected)
            elif cfg.mode == 'squeeze_z':
                selected = np.flatnonzero(self.layers[:, 2] == sim.desc.kappa.segments[2] - 1)
                sim.d.xfrc_applied[np.array(sim.seg_bodies)[selected], 2] = -self.load / len(selected)
            previous = float(sim.d.time)
            sim.step()
            self.failure = sim.failure()
            if sim.d.time < previous + DT / 2:
                self.failure = 'MuJoCo 自动重置'
            if self.failure:
                break

    def snapshot(self):
        sim = self.sim
        # Kinematics only: do not overwrite the step's frozen-model force data.
        mujoco.mj_kinematics(sim.m, sim.d)
        centers = sim.seg_pos()
        rotations = sim.d.xmat[sim.seg_bodies].reshape(-1, 3, 3)
        if not np.isfinite(centers).all() or not np.isfinite(rotations).all():
            return dict(settings=asdict(self.cfg), failure=self.failure or '非有限模型状态',
                        finished=True, status='error', source_sha256=RELEASE_SHA, preview_only=True)
        extent = np.einsum('nij,nj->ni', np.abs(rotations), self.half)
        width = float(np.max(centers[:, 1] + extent[:, 1]) - np.min(centers[:, 1] - extent[:, 1]))
        height = float(np.max(centers[:, 2] + extent[:, 2]) - np.min(centers[:, 2] - extent[:, 2]))
        end = self.layers[:, 0] == sim.desc.kappa.segments[0] - 1
        droop_mm = float((self.reference[end, 2].mean() - centers[end, 2].mean()) * 1000)
        compression = (1 - width / sim.desc.kappa.size[1]) * 100
        metrics = dict(compression_pct=compression, height_mm=height * 1000, droop_mm=droop_mm,
                       load_N=self.load if self.cfg.mode != 'sag' else 0.,
                       angle_deg=sim.max_relative_angle_deg())
        self.history.append([float(sim.d.time), compression, height * 1000, droop_mm])
        self.history = self.history[-350:]
        return dict(settings=asdict(self.cfg), t=float(sim.d.time), failure=self.failure,
                    finished=bool(sim.d.time >= 3. or self.failure), centers=centers.tolist(),
                    rotations=rotations.tolist(), half=self.half.tolist(), reference=self.reference.tolist(),
                    joints=[[j.a, j.b] for j in sim.desc.joints], metrics=metrics, history=self.history,
                    internal={key: float(sim.desc.internal[key]) for key in ('E_y', 'E_z', 'D')},
                    warnings={mujoco.mjtWarning(i).name: int(w.number)
                              for i, w in enumerate(sim.d.warning) if w.number},
                    source_sha256=RELEASE_SHA, preview_only=True)


class LivePreview:
    def __init__(self):
        self.lock = threading.Lock()
        self.pending = Settings()
        self.state = dict(status='building')
        self.generation = 0
        self.paused = False
        self.closed = threading.Event()
        self.last_client = 0.
        self.worker = threading.Thread(target=self.loop, daemon=True)
        self.worker.start()

    def configure(self, payload):
        cfg = Settings.parse(payload)
        with self.lock:
            self.generation += 1
            self.pending = cfg
            self.paused = False
            self.last_client = time.monotonic()
            return self.generation

    def snapshot(self):
        with self.lock:
            self.last_client = time.monotonic()
            return dict(self.state, paused=self.paused, requested_generation=self.generation)

    def loop(self):
        preview, generation = None, -1
        while not self.closed.is_set():
            with self.lock:
                pending = self.pending
                self.pending = None
                requested = self.generation
                paused = self.paused or time.monotonic() - self.last_client > 2.
            if pending is not None:
                generation = requested
                try:
                    preview = Preview(pending)
                    data = preview.snapshot()
                except (ValueError, RuntimeError) as error:
                    preview = None
                    data = dict(failure=str(error), status='error')
                with self.lock:
                    self.state = dict(data, generation=generation)
            elif preview is not None and not paused and not self.state.get('finished'):
                preview.advance()
                data = preview.snapshot()
                with self.lock:
                    self.state = dict(data, generation=generation)
            else:
                self.closed.wait(.03)


def handler_for(live):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, payload, status=200, content_type='application/json; charset=utf-8'):
            body = payload if isinstance(payload, bytes) else json.dumps(payload, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/':
                self.respond(Path(__file__).with_suffix('.html').read_bytes(), content_type='text/html; charset=utf-8')
            elif self.path == '/api/meta':
                self.respond(dict(presets=PRESETS, timestep=DT, source_sha256=RELEASE_SHA))
            elif self.path == '/api/state':
                self.respond(live.snapshot())
            else:
                self.respond(dict(error='未找到'), 404)

        def do_POST(self):
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 4096:
                    raise ValueError('请求大小无效')
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError('请求必须为对象')
                if self.path == '/api/config':
                    self.respond(dict(generation=live.configure(payload)))
                elif self.path == '/api/pause' and isinstance(payload.get('paused'), bool):
                    with live.lock:
                        live.paused = payload['paused']
                    self.respond(dict(paused=payload['paused']))
                else:
                    self.respond(dict(error='无效操作'), 400)
            except (ValueError, TypeError) as error:
                self.respond(dict(error=str(error)), 400)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    live = LivePreview()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(live))
    print(f'内容物参数预览: http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        live.closed.set()
        server.server_close()


if __name__ == '__main__':
    main()
