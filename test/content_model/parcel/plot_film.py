"""Plot measured membrane rebound and the saved physical surface topology."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


FACES = ("TOP", "BOTTOM", "LEFT", "RIGHT", "FRONT", "BACK")
COLORS = ("#e4a44f", "#9467bd", "#4c98cc", "#d56d6d", "#65a67c", "#b8aa60")


def rebound(run):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for variant, label in (("previous", "Previous parameters"), ("damped", "Revised parameters")):
        rows = json.loads((run / f"{variant}.json").read_text())
        times = [r["t"] for r in rows]
        energy = np.array([r["total_energy"] for r in rows])
        axes[0].plot(times, np.array([r["displacement"] for r in rows]) * 1000, label=label)
        axes[1].semilogy(times, energy / energy[0], label=label)
    axes[0].set(xlabel="Time (s)", ylabel="Top patch displacement relative to COM (mm)")
    axes[1].set(xlabel="Time (s)", ylabel="Mechanical energy / initial energy")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(run / "rebound.png", dpi=170)
    fig.savefig(run / "rebound.svg")
    plt.close(fig)


def surface(ax, points, triangles, faces):
    mesh = Poly3DCollection(points[triangles] * 1000, alpha=.45,
                            facecolors=[COLORS[i] for i in faces],
                            edgecolors="#33434b", linewidths=.55)
    ax.add_collection3d(mesh)
    ax.auto_scale_xyz(points[:, 0] * 1000, points[:, 1] * 1000, points[:, 2] * 1000)
    ax.set_box_aspect(np.maximum(np.ptp(points, axis=0), .01))
    ax.set(xlabel="x (mm)", ylabel="y (mm)", zlabel="z (mm)")
    ax.view_init(elev=25, azim=-55)


def envelope(run, topology_path):
    topo = json.loads(topology_path.read_text())
    triangles = np.array(topo["triangles"])
    faces = np.array(topo["face_id"])
    points = np.load(run / "p3.npz")["rest"][0]
    fig = plt.figure(figsize=(8, 5.5), layout="constrained")
    ax = fig.add_subplot(projection="3d")
    surface(ax, points, triangles, faces)
    edges = np.array(topo["edges"])
    adjacent = np.array(topo["adjacent"])
    for a, b in edges[faces[adjacent[:, 0]] != faces[adjacent[:, 1]]]:
        ax.plot(*((points[[a, b]] * 1000).T), color="#c72e42", linewidth=2)
    ax.legend(handles=[Patch(color=c, label=f) for c, f in zip(COLORS, FACES)], loc="upper left")
    ax.set_title("56 shared physical vertices, 108 triangles\nRed edges connect adjacent faces")
    fig.savefig(run / "topology.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for face, ax in enumerate(axes.flat):
        tri = triangles[faces == face]
        vertices = np.unique(tri)
        normal_axis = (2, 2, 0, 0, 1, 1)[face]
        u, v = [a for a in range(3) if a != normal_axis]
        ax.triplot(points[:, u] * 1000, points[:, v] * 1000, tri, color=COLORS[face])
        ax.scatter(points[vertices, u] * 1000, points[vertices, v] * 1000, s=12, color="#33434b")
        for vertex in vertices:
            ax.annotate(str(vertex), points[vertex, [u, v]] * 1000, xytext=(3, 3),
                        textcoords="offset points", fontsize=8)
        ax.set(title=FACES[face], xlabel=f"{'xyz'[u]} (mm)", ylabel=f"{'xyz'[v]} (mm)", aspect="equal")
    fig.suptitle("Identical vertex IDs denote the same physical node across faces")
    fig.savefig(run / "face_nodes.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(11, 8), layout="constrained")
    for column, proto in enumerate((2, 3)):
        data = np.load(run / f"p{proto}.npz")
        for row, key in enumerate(("rest", "film")):
            ax = fig.add_subplot(2, 2, row * 2 + column + 1, projection="3d")
            surface(ax, data[key][0], triangles, faces)
            state = "Reference" if row == 0 else "Settled at 2 s"
            ax.set_title(f"{'Tight' if proto == 2 else 'Loose'} bag / {state}")
    fig.savefig(run / "tight_loose.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relaxation", type=Path)
    parser.add_argument("--support", type=Path)
    parser.add_argument("--topology", type=Path)
    args = parser.parse_args()
    if args.relaxation:
        rebound(args.relaxation)
    if args.support:
        if args.topology is None:
            parser.error("--support requires --topology")
        envelope(args.support, args.topology)
