"""Render saved Newton states with MuJoCo's renderer. Usage: render_newton.py STATE.npz [...]"""
import os
import struct
import sys
import zlib
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
import numpy as np

NEWTON = Path(__file__).resolve().parents[2] / "contents/newton"
sys.path.insert(0, str(NEWTON))



def png(path, img):
    h, w, _ = img.shape
    raw = b''.join(b'\0' + img[i].tobytes() for i in range(h))

    def chunk(tag, body):
        return struct.pack('>I', len(body)) + tag + body + struct.pack('>I', zlib.crc32(tag + body) & 0xffffffff)
    Path(path).write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
                          + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))


def render(path):
    d = np.load(path)
    q, faces, body_q, boxes = d['q'], d['faces'], d['body_q'], d['boxes']
    geoms = []
    if len(faces):
        tris = faces.reshape(-1, 3)
        double = np.concatenate([tris, tris[:, ::-1]])
        asset = (f'<mesh name="m" vertex="{" ".join(map(str, q.ravel()))}" face="{" ".join(map(str, double.ravel()))}" inertia="shell"/>')
        geoms.append('<geom type="mesh" mesh="m" rgba=".85 .55 .35 1" contype="0" conaffinity="0"/>')
    else:
        asset = ''
        geoms += [f'<geom type="sphere" size=".006" pos="{p[0]} {p[1]} {p[2]}" rgba=".85 .55 .35 1"/>' for p in q]
    for box in boxes:
        link, hx, hy, hz = int(box[0]), box[1], box[2], box[3]
        p, r = body_q[link][:3], body_q[link][3:]
        geoms.append(f'<geom type="box" size="{hx} {hy} {hz}" pos="{p[0]} {p[1]} {p[2]}" quat="{r[3]} {r[0]} {r[1]} {r[2]}" rgba=".3 .5 .85 .9"/>')
    xml = ('<mujoco><visual><global offwidth="1280" offheight="800"/></visual><asset>' + asset + '</asset><worldbody>'
           '<light pos=".4 -.4 1.5" dir="-.3 .3 -1"/><light pos="-.5 .5 1.2" dir=".3 -.3 -1"/>'
           '<geom type="plane" size="2 2 .01" rgba=".82 .84 .87 1"/>' + ''.join(geoms) + '</worldbody></mujoco>')
    m = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(m)
    mujoco.mj_forward(m, data)
    r = mujoco.Renderer(m, 480, 640)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (q[:, 0].mean(), q[:, 1].mean(), q[:, 2].mean())
    cam.distance, cam.azimuth, cam.elevation = max(0.6, 2.0 * float(np.ptp(q, axis=0).max())), 130, -20
    r.update_scene(data, cam)
    img = r.render().copy()
    r.close()
    png(str(path).replace('.npz', '.png'), img)


for p in sys.argv[1:]:
    render(p)
    print('rendered', p)
