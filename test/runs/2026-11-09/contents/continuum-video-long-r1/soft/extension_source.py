"""Continue a saved, failed continuum task with unchanged held control on GPU.

This is post-task observation, not another grasp controller. Run under flock.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'contents'))
from continuum.gpu import Simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--until', type=float, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    original = json.loads((source / 'result.json').read_text())
    old = np.load(source / 'trajectory.npz')
    spec = int(old['integration_state_spec'])
    if spec != int(mujoco.mjtState.mjSTATE_INTEGRATION) or not spec & int(mujoco.mjtState.mjSTATE_PLASTIC):
        raise ValueError('full integration state including PLASTIC is required')
    state = old['final_integration_state'].astype(np.float32)
    if not np.isfinite(state).all() or original['outcome'] != 'grasp_fail':
        raise ValueError('expected a finite, failed grasp task terminal state')
    output.mkdir(parents=True, exist_ok=False)
    for name in ('model.xml', 'config.json'):
        shutil.copyfile(source / name, output / name)
    shutil.copyfile(__file__, output / 'extension_source.py')
    model = mujoco.MjModel.from_xml_path(str(output / 'model.xml'))
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, state.astype(float), spec)
    start_time = float(data.time)
    if not start_time < args.until <= 12:
        raise ValueError('observation end must be after source end and no later than 12 s')
    # CPU only restores and computes geometry; every dynamics step stays on GPU.
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_flex(model, data)
    started = time.perf_counter()
    sim = Simulation(model, data, nconmax=1024, njmax=4096)
    sim.restore(state[None, :])
    restored = sim.state()[0]
    error = float(np.max(np.abs(restored - state)))
    if error != 0:
        raise RuntimeError(f'integration restore changed saved state: {error}')
    held_ctrl = data.ctrl.copy()
    initial_plastic = float(np.linalg.norm(data.plastic))
    times = list(old['time'])
    positions = list(old['qpos'])
    integration = list(old['integration_state'])
    trace = []
    last_safe = state.copy()
    failure = None
    every = max(1, round(.02 / model.opt.timestep))

    def append_safe(value, health):
        # Set-state plus geometry is sufficient to export the last known safe qpos.
        mujoco.mj_setState(model, data, value.astype(float), spec)
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_flex(model, data)
        if times and float(data.time) <= float(times[-1]):
            return
        times.append(float(data.time))
        positions.append(data.qpos.copy())
        integration.append(value.copy())
        trace.append(dict(time_s=float(data.time), **health))

    append_safe(last_safe, sim.health()[0])
    step = 0
    while float(data.time) < args.until:
        sim.step()  # ctrl and all external inputs remain exactly as restored.
        step += 1
        health = sim.health()[0]
        value = sim.state()[0]
        if int(health['failure_bits']):
            failure = dict(time_s=float(sim.data.time.numpy()[0]), step=step, **health)
            np.savez_compressed(output / 'first_failed_state.npz', integration_state=value,
                                integration_state_spec=spec, qpos=sim.data.qpos.numpy(),
                                qvel=sim.data.qvel.numpy(), plastic=sim.data.plastic.numpy())
            append_safe(last_safe, last_health)
            break
        last_safe, last_health = value, health
        current_time = float(sim.data.time.numpy()[0])
        if step % every == 0 or current_time >= args.until:
            sim.sync(data)
            append_safe(last_safe, health)
            if step % 1000 == 0:
                print(f'valid post-failure observation at {current_time:.3f} s', flush=True)
        # Keep loop time current without a CPU dynamics call.
        data.time = current_time
    append_safe(last_safe, sim.health()[0] if failure is None else last_health)
    mujoco.mj_setState(model, data, last_safe.astype(float), spec)
    np.savez_compressed(output / 'trajectory.npz', qpos=np.asarray(positions), time=np.asarray(times),
                        final_qpos=data.qpos, final_qvel=data.qvel,
                        integration_state=np.asarray(integration), integration_state_spec=spec,
                        final_integration_state=last_safe)
    result = dict(original)
    observation = dict(source=str(source), start_s=start_time, requested_end_s=args.until,
                       last_safe_time_s=float(data.time), steps=step, failure=failure,
                       status='sim_fail' if failure else 'time_budget_reached',
                       held_ctrl=held_ctrl.tolist(), initial_plastic_norm=initial_plastic,
                       final_plastic_norm=float(np.linalg.norm(data.plastic)),
                       restore_max_abs_difference=error, state_includes_plastic=True,
                       dynamics='GPU Newton', software=sim.software(), gpu_health=sim.health(),
                       wall_s=time.perf_counter()-started, hostname=platform.node(),
                       label='失夹后保持姿态观察', original_task_outcome=original['outcome'],
                       original_task_reason=original['reason'], original_task_end_s=original['sim_time_s'],
                       source_sha256={name: hashlib.sha256((source/name).read_bytes()).hexdigest()
                                      for name in ('model.xml','trajectory.npz','result.json')})
    result.update(sim_time_s=float(data.time), phase='post_grasp_fail_hold', observation=observation)
    (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    (output / 'observation.json').write_text(json.dumps(observation, ensure_ascii=False, indent=2))
    (output / 'observation_trace.json').write_text(json.dumps(trace, indent=2))
    print(json.dumps(observation, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
