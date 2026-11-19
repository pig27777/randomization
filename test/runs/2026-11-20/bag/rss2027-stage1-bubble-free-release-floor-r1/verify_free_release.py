"""Verify the recorded unsupported interval and replay the exact final geometry."""

import json
from pathlib import Path
import sys

import mujoco
import numpy as np

RUN = Path(__file__).resolve().parent
PRIOR = RUN.with_name("rss2027-stage1-bubble-free-release-r1")
sys.path.insert(0, str(RUN.parents[3] / "bag_model"))
from physical_parcel_checks import film_payload_geometry  # noqa: E402


def load(path):
    return json.loads(path.read_bytes())


def main():
    trial = RUN / "recording/trial"
    result = load(trial / "result.json")
    trace = load(trial / "trace.json")
    prior = load(PRIOR / "recording/trial/trace.json")
    release = load(RUN / "release-check.json")
    event = load(RUN / "recording/withdrawal.json")
    slip = load(RUN / "release-traction.json")
    loss = float(release["first_final_contact_loss_sample"]["t_s"])
    open_frames = [row for row in trace if row["phase"] == "open" and row["t"] <= loss]
    hold_frames = [row for row in trace if row["phase"] == "lift_hold"]
    hold = next(phase for phase in result["phases"] if phase["phase"] == "lift_hold")
    model = mujoco.MjModel.from_binary_path(str(trial / "model.mjb"))
    data = mujoco.MjData(model)
    # The frozen recorder stores mjSTATE_INTEGRATION. Read the numerical state
    # without deserializing the old enum-valued metadata entry.
    state_spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    with np.load(trial / "trajectory.npz") as states:
        mujoco.mj_setState(model, data, states["final_integration_state"], state_spec)
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    mujoco.mj_fwdPosition(model, data)
    geometry = film_payload_geometry(model, data, model.flex_elem.reshape(-1, 3))
    table_z = float(data.geom_xpos[model.geom("table").id, 2])
    floor_z = float(data.geom_xpos[model.geom("floor").id, 2])
    np.testing.assert_allclose([table_z, floor_z], [event["after_m"][2], event["floor_after_m"][2]])
    before = {row["t"]: row for row in prior if row["t"] <= event["first_open_t_s"]}
    matched = [row for row in trace if row["t"] <= event["first_open_t_s"]]
    assert set(row["t"] for row in matched) == set(before)
    com_difference = max(float(np.linalg.norm(np.asarray(row["com"]) - before[row["t"]]["com"])) for row in matched)
    support_difference = max(abs(row["support_N"] - before[row["t"]]["support_N"]) for row in matched)
    hold_support = max(row["support_N"] for row in hold_frames)
    pre_loss_support = max(row["support_N"] for row in open_frames)
    checks = dict(outcome=result["outcome"] == "pickup_release_ok",
                  numerical=not result["warnings"] and result["peaks"]["penetration_mm"] <= 1.,
                  suspended_hold=hold["end_s"] - hold["start_s"] >= .2 - result["config"]["dt"] / 2
                  and hold_support <= .05,
                  unsupported_through_loss=pre_loss_support <= .05,
                  enclosure=geometry["intersecting_triangles"] == geometry["outside_corners"] == 0)
    summary = dict(verified_free_release=all(checks.values()), checks=checks, opening_s=event["first_open_t_s"],
                   loss_s=loss, release_time_to_loss_s=loss - event["first_open_t_s"],
                   simulated_s=result["sim_time_s"], wall_s=result["wall_s"],
                   peak_penetration_mm=result["peaks"]["penetration_mm"],
                   pre_loss_support_max_N=pre_loss_support, hold_max_support_N=hold_support,
                   hold_min_grip_per_finger_N=min(min(row["grip_each_N"]) for row in hold_frames),
                   samples_before_withdrawal=len(matched), max_prefix_com_difference_m=com_difference,
                   max_prefix_support_difference_N=support_difference,
                   native_bilateral_release_samples=slip["release_bilateral"],
                   native_sustained_sliding=slip["first_sustained_bilateral_sliding"],
                   final_table_z_m=table_z, final_floor_z_m=floor_z, state_recording_spec=state_spec,
                   final_geometry=geometry)
    with (RUN / "free-release-verification.json").open("x") as stream:
        stream.write(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
