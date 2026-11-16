"""Shared YAM controller and recorder with GPU physical parcel stepping.

The controller and IK remain on CPU. State/contact transfers are included in
end-to-end trial timings and reported separately from resident GPU throughput.
"""
import time
from pathlib import Path
import numpy as np
import mujoco_warp as mjw
import warp as wp
import physical_parcel_yam as cpu
from physical_parcel_yam import (Config, measure, attachment, gathering_spring, check_state,
                                  dimensions, refresh, summarize, save_model)
from bag_sim.physical_parcel_gpu import Batch


def scene(cfg):
    robot=cpu.scene(cfg)
    sim=robot.sim
    begin=time.perf_counter()
    sim.gpu=Batch(1,sim.parcel.cfg,parcel=sim.parcel,model=sim.m,data=sim.d,
                  randomize=False,contact_capacity=2048,constraint_capacity=8192)
    sim.gpu_graph=sim.gpu.graph()
    sim.gpu.reset(initial_qpos=sim.d.qpos,initial_qvel=sim.d.qvel,initial_ctrl=sim.d.ctrl)
    wp.synchronize()
    sim.gpu_timing=dict(setup_s=time.perf_counter()-begin,upload_control_s=0.,physics_s=0.,download_s=0.,steps=0)
    sim.original_step=sim.step
    def step():
        begin=time.perf_counter()
        sim.gpu.d.ctrl.assign(np.asarray(sim.d.ctrl)[None,:])
        wp.synchronize()
        uploaded=time.perf_counter()
        wp.capture_launch(sim.gpu_graph)
        wp.synchronize()
        stepped=time.perf_counter()
        mjw.get_data_into(sim.d,sim.m,sim.gpu.d)
        finished=time.perf_counter()
        sim.gpu_timing['upload_control_s']+=uploaded-begin
        sim.gpu_timing['physics_s']+=stepped-uploaded
        sim.gpu_timing['download_s']+=finished-stepped
        sim.gpu_timing['steps']+=1
        if np.any(sim.gpu.bits.numpy()):
            raise RuntimeError('GPU contact/constraint buffer overflow')
    sim.step=step
    return robot


def source_files():
    return dict(**cpu.source_files(),cpu_yam_adapter=Path(cpu.__file__),
                gpu_model=cpu.ROOT/'bag_sim/physical_parcel_gpu.py')
