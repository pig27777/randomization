"""Reduce the native membrane energy in two already enclosed preparation states."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from diagnose_initial_load import RUN, backend, physical_parcel, prepared_wrap
from continue_wrap import FACES, MIN_GAP, PANELS, edge_targets, incident_faces, measurements, panel_pairs
from relax_tangent import open_pairs, project_step, separating_planes

ROOT = RUN.parents[4]


class FilmEnergy:
    def __init__(self, parcel):
        self.parcel = parcel
        model = parcel.m
        self.a, self.b = FACES[:, [1, 2, 0]], FACES[:, [2, 0, 1]]
        self.rest_sq = model.flexedge_length0[model.flex_elemedge.reshape(-1, 3)] ** 2
        start = model.flex_stiffnessadr[0]
        packed = model.flex_stiffness[start:start + 21 * len(FACES)].reshape(-1, 21)[:, :6]
        i, j = np.triu_indices(3)
        self.stiffness = np.zeros((len(FACES), 3, 3))
        self.stiffness[:, i, j] = packed
        self.stiffness[:, j, i] = packed
        self.weight = parcel.envelope["masses"][:, None] * np.asarray(parcel.cfg.gravity)
        self.edges = np.unique(np.sort(np.stack((self.a, self.b), axis=-1).reshape(-1, 2), axis=1), axis=0)
        self.edge_a, self.edge_b = self.edges.T
        reference = parcel.envelope["reference"]
        self.edge_lengths = np.linalg.norm(reference[self.edge_a] - reference[self.edge_b], axis=1)

    def evaluate(self, points):
        parcel = self.parcel
        delta = points[self.a] - points[self.b]
        strain = np.sum(delta ** 2, axis=2) - self.rest_sq
        tension = np.einsum("nij,nj->ni", self.stiffness, strain)
        membrane = .25 * float(np.sum(strain * tension))
        force = -tension[:, :, None] * delta
        net = self.weight.copy()
        np.add.at(net, self.a, force)
        np.add.at(net, self.b, -force)
        angle, gradient = physical_parcel.dihedrals(points, parcel.hinges)
        rotation = np.arctan2(np.sin(angle - parcel.rest_angles), np.cos(angle - parcel.rest_angles))
        moment = parcel.hinge_stiffness * rotation
        bend = .5 * float(moment @ rotation)
        np.add.at(net, parcel.hinges, -moment[:, None, None] * gradient)
        return membrane + bend - float(np.sum(points * self.weight)), net, membrane + bend


def assess(reference, local, item, pairs, energy, round_id):
    measured = measurements(reference, local, energy.edges, item)
    total, force, elastic = energy.evaluate(local)
    wall = float(np.min(local[pairs[:, 0], 2] - local[pairs[:, 1], 2]))
    return dict(iteration=round_id, elastic_J=elastic, potential_J=total,
                peak_unconstrained_node_force_N=float(np.linalg.norm(force, axis=1).max()), wall_gap_m=wall,
                **measured)


def project_edge_band(target, groups):
    for a, b, length in groups:
        direction = target[b] - target[a]
        current = np.linalg.norm(direction, axis=1)
        desired = np.clip(current, .9525 * length, 1.0475 * length)
        correction = .45 * ((current - desired) / np.maximum(current, 1e-12))[:, None] * direction
        target[a] += correction
        target[b] -= correction


def prepare(case, output, iterations, keep_strain_bound):
    raw_cfg = json.loads((case / "smoke.json").read_bytes())
    cfg = backend.Config(**raw_cfg)
    with patch.object(physical_parcel, "wrap_generic", prepared_wrap(cfg.geometry)):
        parcel, _ = backend.parcel_world(cfg)
    parcel.forward()
    reference = parcel.envelope["reference"]
    origin = parcel.envelope["payload_origin"]
    start = parcel.envelope["current"] - origin
    energy = FilmEnergy(parcel)
    _, nodal_force, _ = energy.evaluate(start)
    native = parcel.d.qfrc_passive[parcel.film_dofs] + energy.weight
    np.testing.assert_allclose(nodal_force, native, atol=1e-10, rtol=0.)
    error = float(np.max(abs(nodal_force - native)))
    half = np.asarray(cfg.payload_size) / 2. + parcel.cfg.radius
    item = dict(position=np.zeros(3), half_size=half, rotation=np.eye(3))
    pairs = panel_pairs(reference, FACES, PANELS, [0], [3])
    edge_groups = edge_targets(reference, energy.edges) if keep_strain_bound else []
    adjacent = np.column_stack((incident_faces(), np.full(len(start), len(FACES))))
    table_z = parcel.cfg.radius - origin[2]
    rows = [assess(reference, start, item, pairs, energy, 0)]
    best, final = start, rows[0]
    local = start
    step_size = .0001
    for round_id in range(1, iterations + 1):
        potential, force, _ = energy.evaluate(local)
        normals, bounds = separating_planes(local, half)
        normals = np.vstack((normals, [0., 0., 1.]))
        bounds = np.r_[bounds, table_z]
        for _ in range(16):
            target = local + step_size * force
            if keep_strain_bound:
                project_edge_band(target, edge_groups)
            open_pairs(target, pairs, MIN_GAP)
            candidate = project_step(local, target, normals, bounds, adjacent)
            candidate_potential, _, _ = energy.evaluate(candidate)
            gap = float(np.min(candidate[pairs[:, 0], 2] - candidate[pairs[:, 1], 2]))
            within_band = True
            if keep_strain_bound:
                lengths = np.linalg.norm(candidate[energy.edge_a] - candidate[energy.edge_b], axis=1)
                within_band = bool(np.max(abs(lengths / energy.edge_lengths - 1.)) <= .05)
            if candidate_potential < potential and gap >= MIN_GAP - 1e-9 and within_band:
                local = candidate
                step_size = min(step_size * 1.25, .002)
                break
            step_size *= .5
        else:
            break
        if round_id % 50 == 0 or round_id == iterations:
            rows.append(assess(reference, local, item, pairs, energy, round_id))
            if rows[-1]["enclosed"] and rows[-1]["wall_gap_m"] >= MIN_GAP - 1e-9:
                best, final = local, rows[-1]
    if rows[-1]["iteration"] != round_id:
        latest = assess(reference, local, item, pairs, energy, round_id)
        rows.append(latest)
        if latest["enclosed"] and latest["wall_gap_m"] >= MIN_GAP - 1e-9:
            best, final = local, latest
    improved = final["potential_J"] < rows[0]["potential_J"]
    if improved:
        destination = output / case.name
        destination.mkdir(exist_ok=False)
        source = physical_parcel.HERE / cfg.geometry
        with np.load(source) as archive:
            raw = dict(archive.items())
        center = np.r_[parcel.cfg.center, 0.]
        raw["current_m"] = best + origin - center
        np.savez_compressed(destination / "prepared_mesh.npz", **raw)
        new_cfg = raw_cfg | {"geometry": f"../{(destination / 'prepared_mesh.npz').resolve().relative_to(ROOT)}"}
        for name, seconds in (("smoke", .02), ("settle", .30)):
            (destination / f"{name}.json").write_text(json.dumps(new_cfg | {"smoke_seconds": seconds}, indent=2) + "\n")
    return dict(case=case.name, exported=improved, projected_strain_band=keep_strain_bound,
                max_native_node_force_error_N=error,
                start=rows[0], selected=final, checkpoints=rows)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-directory", type=Path, default=RUN / "randomized-entry")
    parser.add_argument("--sample", type=int)
    parser.add_argument("--keep-strain-bound", action="store_true")
    args = parser.parse_args()
    pattern = f"sample-{args.sample:03d}" if args.sample is not None else "sample-???"
    cases = sorted(args.source_directory.glob(pattern))
    if not cases:
        parser.error(f"no matching preparation cases in {args.source_directory}")
    args.output.mkdir(exist_ok=False)
    summaries = []
    for case in cases:
        summary = prepare(case, args.output, 1500, args.keep_strain_bound)
        summaries.append(summary)
        print(summary["case"], summary["start"]["elastic_J"], summary["selected"], flush=True)
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
