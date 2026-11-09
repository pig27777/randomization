"""Export actual continuum reference tetrahedra and a fixed-camera SVG preview."""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "contents"))
from continuum.model import Template, geometry, preset, shape_parameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".npz").exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="840" height="600" viewBox="0 0 840 600">',
           '<rect width="840" height="600" fill="#f4f7f8"/>',
           '<text x="30" y="34" font-family="sans-serif" font-size="19">Same B: 60 nodes / 144 tetrahedra. Change geometric X.</text>']
    b, meshes, records = Template(), {}, []
    camera = np.array([[.866,-.5,0],[-.25,-.433,.866],[.433,.75,.5]])
    for panel,name in enumerate(("box","rounded","pillow","tapered")):
        p = replace(preset("bendable"), **shape_parameters(name))
        g = geometry(p,b)
        faces = Counter(tuple(sorted(face)) for t in g.elements for face in
                        ((t[0],t[1],t[2]),(t[0],t[1],t[3]),(t[0],t[2],t[3]),(t[1],t[2],t[3])))
        boundary = np.array([face for face,count in faces.items() if count==1])
        projected = g.points @ camera.T
        offset = np.array([210+420*(panel%2),175+265*(panel//2)])
        points = projected[:,:2]*[1050,-1050]+offset
        for face in sorted(boundary,key=lambda f:projected[f,2].mean()):
            normal = np.cross(g.points[face[1]]-g.points[face[0]],g.points[face[2]]-g.points[face[0]])
            normal /= np.linalg.norm(normal)
            light = 40 + 25*abs(float(normal @ np.array([.2,-.3,.9327])))
            coords = " ".join(f"{x:.2f},{y:.2f}" for x,y in points[face])
            svg.append(f'<polygon points="{coords}" fill="hsl(176,45%,{light:.2f}%)" stroke="#276368" stroke-width=".65"/>')
        x,y = offset+[0,102]
        svg.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-family="sans-serif" font-size="17">{name} | {g.mass:.3f} kg</text>')
        svg.append(f'<text x="{x}" y="{y+21}" text-anchor="middle" font-family="sans-serif" font-size="12">roundness={p.roundness:g}, bulge={p.bulge:g}, taper={p.taper:g}</text>')
        meshes[name+"_vertices"],meshes[name+"_tetrahedra"] = g.points,g.elements
        records.append(dict(shape=name,parameters_X=asdict(p),mass_kg=g.mass,volume_m3=g.volume))
    svg.append('</svg>')
    args.output.write_text("\n".join(svg))
    np.savez_compressed(args.output.with_suffix(".npz"),**meshes)
    manifest = dict(template_B=asdict(b),variants=records,
                    model_sha256=hashlib.sha256((ROOT/"contents/continuum/model.py").read_bytes()).hexdigest(),
                    exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    note="Reference geometry only; no dynamics or physical validation implied")
    args.output.with_suffix(".json").write_text(json.dumps(manifest,indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
