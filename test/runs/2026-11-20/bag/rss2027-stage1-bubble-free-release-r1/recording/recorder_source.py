"""Record whole-bag release using the shared YAM protocol and an unloaded movable support."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from unittest.mock import patch
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.parents[3] / "bag_model"))
import physical_parcel_yam as backend  # noqa: E402
from record_physical_parcel import penetration_peak, release_contacts  # noqa: E402
from yam_platform.protocol import run  # noqa: E402

SUPPORT = "withdrawable_support"
RETRACTION_M = .120
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
parcel_world = backend.parcel_world


def world_with_movable_support(cfg):
    parcel, xml = parcel_world(cfg)
    root = ET.fromstring(xml)
    world = root.find("worldbody")
    table = world.find("geom[@name='table']")
    support = ET.SubElement(world, "body", name=SUPPORT, mocap="true", pos=table.attrib.pop("pos"))
    world.remove(table)
    support.append(table)
    return parcel, ET.tostring(root, encoding="unicode")


def integration_state(sim):
    state = np.empty(mujoco.mj_stateSize(sim.m, STATE_SPEC))
    mujoco.mj_getState(sim.m, sim.d, state, STATE_SPEC)
    return state


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, default=RUN / "config.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = backend.Config(**json.loads(args.config.read_text()))
    if not cfg.support_height > RETRACTION_M or not cfg.pickup_release_seconds:
        parser.error("the recorded protocol requires a raised support and pickup release")
    if args.smoke:
        cfg = replace(cfg, smoke_seconds=.02)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "recorder_source.py").write_bytes(Path(__file__).read_bytes())
    event = None
    peaks, contacts = [], []
    last_phase = None

    def on_sample(robot, row):
        nonlocal event, last_phase
        phase = row["phase"]
        if phase != last_phase:
            print(json.dumps(dict(phase=phase, t_s=row["t"], support_N=row["support_N"])), flush=True)
            last_phase = phase
        if phase in ("lift_hold", "open"):
            contacts.append(dict(t_s=row["t"], phase=phase, contacts=release_contacts(robot)))
        if phase != "open" or event is not None:
            return
        sim = robot.sim
        mocap = sim.m.body_mocapid[sim.m.body(SUPPORT).id]
        if mocap < 0:
            raise RuntimeError("The support has no MuJoCo kinematic pose")
        position = sim.d.mocap_pos[mocap]
        event = dict(first_open_t_s=row["t"], before_m=position.tolist(), after_m=None,
                     support_N=row["support_N"], support_geometry=row["support_geometry"])
        if row["support_N"] == 0. and row["support_geometry"]["intersecting_triangles"] == 0:
            position[2] -= RETRACTION_M
            event["after_m"] = position.tolist()
        (args.output / "withdrawal.json").write_text(json.dumps(event, indent=2) + "\n")

    def on_peak(robot, phase):
        peaks.append(penetration_peak(robot, phase))

    with patch.multiple(backend, parcel_world=world_with_movable_support, record_state=integration_state,
                        STATE_SPEC=STATE_SPEC, create=True):
        result = run(backend, cfg, args.output / "trial", on_sample=on_sample, on_penetration_peak=on_peak)
    for name, payload in (("summary.json", result), ("penetration-peaks.json", peaks)):
        (args.output / name).write_text(json.dumps(payload, indent=2) + "\n")
    if contacts:
        (args.output / "release-contacts.json").write_text(json.dumps(contacts, indent=2) + "\n")


if __name__ == "__main__":
    main()
