"""Read archived Newton snapshots without starting GPU or modifying the model."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from bag_sim.wrap import triangle_box_intersections


ROOT = Path(__file__).resolve().parent
R1 = ROOT.parent / "rss-premises-verify-r1"
CONTACT_MARGIN = .00025
THICKNESS = .000193


def diagnose(path):
    trial = json.loads(path.with_suffix(".json").read_text())
    dt = trial["dt_s"]
    rows = {round(r["time_s"] / dt): r for r in trial["rows"]}
    out = []
    with np.load(path.with_suffix(".npz")) as data:
        pad_ids, pad_local = data["pad_body_ids"], data["pad_local_centers"]
        pad_q, half = data["pad_local_rotation"], data["pad_half_sizes"]
        faces = data["faces"]
        ref = data["film"][0]
        for step, bodies, film in zip(data["steps"], data["bodies"], data["film"], strict=True):
            if step * dt < .08:
                continue
            parents = Rotation.from_quat(bodies[pad_ids, 3:])
            centers = parents.apply(pad_local) + bodies[pad_ids, :3]
            rot = (parents * Rotation.from_quat(pad_q)).as_matrix()
            close = rot[0, :, 1].copy()
            if close @ (centers[1] - centers[0]) < 0:
                close = -close
            parallel_error = np.linalg.norm(np.cross(close, rot[1, :, 1]))
            if parallel_error > 1.e-4:
                raise ValueError("Expected the two actual ABC inner faces to be parallel")
            gap = float((centers[1] - centers[0]) @ close - (half[:, 1].sum()))
            mid = centers.mean(0)
            # The grasp prism covers the common face footprint between the opposing faces.
            # A radius on both sides includes nodes close enough to either pad to be candidates.
            grip_box = half.min(0) + CONTACT_MARGIN
            grip_box[1] = max(0., .5 * gap) + CONTACT_MARGIN
            local = (film - mid) @ rot[0]
            vertices_in_prism = np.all(np.abs(local) <= grip_box, axis=1)
            intersect = triangle_box_intersections(film, faces, grip_box, mid, rot[0])
            selected = faces[intersect]
            edge_now, edge_ref = [], []
            if len(selected):
                edge_now = np.linalg.norm(film[selected] - film[np.roll(selected, 1, axis=1)], axis=-1).ravel()
                edge_ref = np.linalg.norm(ref[selected] - ref[np.roll(selected, 1, axis=1)], axis=-1).ravel()
            proximity = []
            for center, frame, pad_half in zip(centers, rot, half, strict=True):
                residual = np.abs((film - center) @ frame) - pad_half
                proximity.append(int((np.linalg.norm(np.maximum(residual, 0.), axis=1)
                                      <= 2. * CONTACT_MARGIN).sum()))
            lowest = int(film[:, 2].argmin())
            source = rows[int(step)]
            wrench = np.array(source["pad_proxy_wrench"])[:, :3]
            axial = wrench @ close
            tangent = np.linalg.norm(wrench - axial[:, None] * close, axis=1)
            out.append(dict(t_s=float(step * dt), jaw_gap_mm=gap * 1000.,
                            mid_z_mm=float(mid[2] * 1000.),
                            inward_force_N=(-axial * [1., -1.]).tolist(),
                            tangent_force_N=tangent.tolist(),
                            candidate_contacts=source["pad_soft_candidates"],
                            pad_nearby_vertex_counts=proximity,
                            vertices_in_prism=int(vertices_in_prism.sum()),
                            film_min_z_mm=float(film[lowest, 2]*1000.),
                            film_max_z_mm=float(film[:, 2].max()*1000.),
                            lowest_film_xy_mm=(film[lowest, :2]*1000.).tolist(),
                            floor_near_vertex_count=int((film[:, 2] <= 2.*CONTACT_MARGIN).sum()),
                            triangles_in_prism=int(intersect.sum()),
                            local_edge_max_mm=float(np.max(edge_now, initial=0.) * 1000.),
                            local_reference_edge_max_mm=float(np.max(edge_ref, initial=0.) * 1000.)))
    return dict(case=trial["case"], sampled_hz=25, double_layer_criterion_mm=2*(THICKNESS+CONTACT_MARGIN)*1000.,
                pad_face_mm=(half[0, [0, 2]] * 2000.).tolist(),
                original_reference_longest_edge_mm=trial["max_reference_edge_m"]*1000., frames=out,
                source="original pad-body wrench and candidate counts; vertex prism computed from archived meshes")


if __name__ == "__main__":
    findings = [diagnose(R1 / "film-10g" / "1A"),
                diagnose(R1 / "joint-10g" / "1C-empty")]
    (ROOT / "jaw_diagnosis.json").write_text(json.dumps(findings, indent=2, allow_nan=False) + "\n")
    for finding in findings:
        rows = finding["frames"]
        print(finding["case"], "double layer", finding["double_layer_criterion_mm"])
        for t in (.08, .24, .32, .4, .48, .6, .8, 1.04, 2.4, 3.2, 4.2):
            r = min(rows, key=lambda r: abs(r["t_s"] - t))
            print(r)
