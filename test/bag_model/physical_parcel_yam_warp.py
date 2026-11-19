"""GPU dynamics for the shared YAM protocol with live host-side feedback."""

from importlib.metadata import version
import os
from pathlib import Path
import subprocess

import mujoco_warp as mjw
import numpy as np
import warp as wp

import physical_parcel_yam as cpu
from physical_parcel_yam import (Config, attachment, dimensions, gathering_spring,
                                measure, save_model, summarize)
from bag_sim.physical_parcel_gpu import bending


class WarpParcelSim(cpu.ParcelSim):
    def upload(self):
        if len(self.parcel.compression_joints):
            raise ValueError("Warp YAM trials require a parcel without compression joints")
        self.device = wp.get_device("cuda:0")
        with wp.ScopedDevice(self.device):
            self.gpu_model = mjw.put_model(self.m)
            # Capacities from the archived 200-vertex YAM GPU probe.
            self.gpu = mjw.put_data(self.m, self.d, nworld=1, nconmax=2048,
                                    naconmax=2048, njmax=8192)
            parcel = self.parcel
            self.bending_inputs = [
                self.gpu.flexvert_xpos, self.gpu.qvel,
                wp.array(self.film_dofs[:, 0], dtype=int),
                wp.array(parcel.hinges, dtype=wp.vec4i),
                wp.array(parcel.rest_angles, dtype=float),
                wp.array(parcel.hinge_stiffness, dtype=float),
                wp.array(parcel.hinge_damping, dtype=float),
                self.gpu.qfrc_passive,
            ]
        self.graph = None
        self.overflow = 0

    def prepare_step(self):
        mjw.step1(self.gpu_model, self.gpu)
        wp.launch(bending, (1, len(self.parcel.hinges)), self.bending_inputs)

    def advance(self):
        self.prepare_step()
        mjw.step2(self.gpu_model, self.gpu)

    def read_feedback(self):
        self.overflow |= int(self.gpu.overflow.numpy()[0])
        if not self.overflow:
            # This also restores the ordered constraint rows used by measure().
            mjw.get_data_into(self.d, self.m, self.gpu)

    def step(self):
        with wp.ScopedDevice(self.device):
            self.gpu.ctrl.assign(self.d.ctrl[None, :].astype(np.float32))
            if self.graph is None:
                self.advance()
                with wp.ScopedCapture(device=self.device) as capture:
                    self.advance()
                self.graph = capture.graph
            else:
                wp.capture_launch(self.graph)
            self.read_feedback()
        if self.overflow:
            raise RuntimeError(f"Warp capacity overflow bits {self.overflow}")

    def refresh(self):
        with wp.ScopedDevice(self.device):
            self.prepare_step()
            mjw.fwd_actuation(self.gpu_model, self.gpu)
            mjw.fwd_acceleration(self.gpu_model, self.gpu)
            mjw.solve(self.gpu_model, self.gpu)
            self.read_feedback()

    def telemetry(self):
        output = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ], text=True)
        samples = [int(value) for pid, value in (line.split(", ") for line in output.splitlines())
                   if pid == str(os.getpid()) and value.isdecimal()]
        return dict(device=self.device.name, warp=version("warp-lang"),
                    mujoco_warp=version("mujoco-warp"),
                    process_memory_mib=sum(samples) if samples else None,
                    overflow_bits=self.overflow)


def scene(cfg):
    if cfg.pinch_coeff:
        raise ValueError("Warp YAM trials require pinch_coeff=0 for native-contact feedback")
    robot = cpu.scene(cfg, sim_factory=WarpParcelSim)
    robot.sim.upload()
    return robot


def refresh(sim):
    sim.refresh()


def check_state(sim):
    if sim.overflow:
        return f"Warp capacity overflow bits {sim.overflow}", dict(overflow_bits=sim.overflow)
    return cpu.check_state(sim)


def source_files():
    return {**cpu.source_files(), "cpu_adapter": Path(cpu.__file__),
            "warp_bending": cpu.ROOT / "bag_sim/physical_parcel_gpu.py"}
