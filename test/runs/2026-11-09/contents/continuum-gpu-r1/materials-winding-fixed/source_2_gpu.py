"""CUDA-only native Newton dynamics, plastic state and independent diagnostics.

CPU compilation/kinematics are allowed; this module never calls mj_step. The
existing effective-metric diagnostic is reused without the 3.17 film patches.
"""
import importlib.metadata

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from bag_sim.flex_residual import ResidualState, step2_with_residual
from .model import require_mujoco4


@wp.kernel
def _health(qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float),
            acceleration: wp.array2d(dtype=float), plastic: wp.array2d(dtype=float),
            addresses: wp.array(dtype=int), rest: wp.array(dtype=wp.vec3d),
            tets: wp.array(dtype=wp.vec4i), det0: wp.array(dtype=wp.float64),
            overflow: wp.array(dtype=int), iterations: wp.array(dtype=int), cap: int,
            residual: wp.array2d(dtype=wp.float64), residual_bad: wp.array(dtype=int),
            output: wp.array2d(dtype=wp.float64)):
    w = wp.tid()
    flags = int(0)
    for i in range(qpos.shape[1]):
        if not wp.isfinite(qpos[w, i]): flags = flags | 1
    for i in range(qvel.shape[1]):
        if not wp.isfinite(qvel[w, i]) or not wp.isfinite(acceleration[w, i]): flags = flags | 1
    norm = wp.float64(0.)
    for i in range(plastic.shape[1]):
        v = wp.float64(plastic[w, i])
        if not wp.isfinite(v): flags = flags | 1
        norm += v*v
    minimum = wp.float64(wp.inf)
    for i in range(tets.shape[0]):
        tet = tets[i]
        positions = wp.mat44d()
        for j in range(4):
            v = tet[j]
            a = addresses[v]
            position = rest[v] + wp.vec3d(wp.float64(qpos[w,a]), wp.float64(qpos[w,a+1]), wp.float64(qpos[w,a+2]))
            for c in range(3): positions[j,c] = position[c]
        a = wp.vec3d(positions[0,0], positions[0,1], positions[0,2])
        b = wp.vec3d(positions[1,0], positions[1,1], positions[1,2])
        c = wp.vec3d(positions[2,0], positions[2,1], positions[2,2])
        d = wp.vec3d(positions[3,0], positions[3,1], positions[3,2])
        value = wp.dot(b-a, wp.cross(c-a, d-a)) / det0[i]
        if not wp.isfinite(value): flags = flags | 1
        if value <= wp.float64(0.): flags = flags | 4
        minimum = wp.min(minimum, value)
    if overflow[w] != 0: flags = flags | 2
    if iterations[w] >= cap: flags = flags | 8
    if residual_bad[w] >= 0 or not wp.isfinite(residual[w,0]): flags = flags | 16
    output[w,0] = wp.float64(int(output[w,0]) | flags)
    output[w,1] = wp.min(output[w,1], minimum)
    output[w,2] = wp.max(output[w,2], residual[w,0])
    output[w,3] = wp.float64(int(output[w,3]) | overflow[w])
    output[w,4] = wp.max(output[w,4], wp.float64(iterations[w]))
    output[w,5] = wp.sqrt(norm)
    output[w,6] += wp.float64(1.)


class Simulation:
    """Homogeneous GPU worlds of one X; different X use separate model groups."""

    def __init__(self, model, data, *, nworld=1, nconmax=512, njmax=2048):
        require_mujoco4()
        if (model.opt.solver != mujoco.mjtSolver.mjSOL_NEWTON
                or model.opt.integrator != mujoco.mjtIntegrator.mjINT_DISCRETE):
            raise ValueError("continuum GPU requires native Newton and discrete integration")
        if model.nflex != 1 or model.flex_dim[0] != 3 or model.flex_interp[0] != 0 or model.nplastic <= 0:
            raise ValueError("expected one full-3D elasto-plastic flex")
        if model.opt.enableflags & mujoco.mjtEnableBit.mjENBL_SLEEP:
            raise ValueError("sleep is not enabled for this residual-checked candidate")
        wp.init()
        devices = wp.get_cuda_devices()
        if not devices:
            raise RuntimeError("CUDA is required; no CPU or CG fallback is implemented")
        self.device = devices[0]
        self.cpu_model = model
        bodies = model.flex_vertbodyid
        self.addresses = np.array([model.jnt_qposadr[model.body_jntadr[b]] for b in bodies], dtype=np.int32)
        for body in bodies:
            first = model.body_jntadr[body]
            if (model.body_parentid[body] != 0 or model.body_jntnum[body] != 3
                    or not np.all(model.jnt_type[first:first+3] == mujoco.mjtJoint.mjJNT_SLIDE)
                    or not np.allclose(model.jnt_axis[first:first+3], np.eye(3))):
                raise ValueError("continuum vertices must use world-parent XYZ slides")
        # Use a separate qpos0 geometry, independent of deformed initial data.
        reference = mujoco.MjData(model)
        mujoco.mj_kinematics(model, reference)
        mujoco.mj_comPos(model, reference)
        mujoco.mj_flex(model, reference)
        self.rest = reference.flexvert_xpos.copy()
        tets = model.flex_elem.reshape(-1,4)
        p = self.rest[tets]
        determinants = np.einsum("ij,ij->i", p[:,1]-p[:,0], np.cross(p[:,2]-p[:,0], p[:,3]-p[:,0]))
        # MuJoCo canonicalizes tetrahedron winding (opposite to the input mesh
        # convention). Inversion is the signed current/reference ratio, not the
        # sign of the compiler's reference determinant itself.
        if not np.isfinite(determinants).all() or np.any(np.abs(determinants) <= 1e-15):
            raise ValueError("reference tetrahedra must be nondegenerate")
        with wp.ScopedDevice(self.device):
            self.model = mjw.put_model(model)
            self.data = mjw.put_data(model, data, nworld=nworld, nconmax=nconmax, njmax=njmax)
            self.residual = ResidualState(self.model, self.data, relative_tolerance=1e-5, refinement_steps=0)
            self._addresses = wp.array(self.addresses, dtype=int)
            self._rest = wp.array(self.rest, dtype=wp.vec3d)
            self._tets = wp.array(tets, dtype=wp.vec4i)
            self._det0 = wp.array(determinants, dtype=wp.float64)
            self.metrics = wp.array(np.tile([0.,1.,0.,0.,0.,0.,0.], (nworld,1)), dtype=wp.float64)
            self.state_spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
            self._state = wp.empty((nworld, mujoco.mj_stateSize(model, self.state_spec)), dtype=float)
            # GPU-only preparation; no physics step or reset is hidden here.
            mjw.forward(self.model, self.data)
            initial_time = self.data.time.numpy().copy()
            with wp.ScopedCapture() as capture:
                self._advance()
            self.graph = capture.graph
            if not np.array_equal(initial_time, self.data.time.numpy()):
                raise RuntimeError("graph capture unexpectedly advanced the initial state")

    def _advance(self):
        mjw.step1(self.model, self.data)
        step2_with_residual(self.model, self.data, state=self.residual)
        wp.launch(_health, self.data.nworld, inputs=[self.data.qpos, self.data.qvel,
            self.data.qacc_smooth, self.data.plastic, self._addresses, self._rest,
            self._tets, self._det0, self.data.overflow, self.data.solver_niter,
            self.cpu_model.opt.iterations, self.residual.last, self.residual.first_bad_step, self.metrics])

    def step(self, *, ctrl=None, external=None):
        with wp.ScopedDevice(self.device):
            if ctrl is not None:
                self.data.ctrl.assign(np.broadcast_to(ctrl, self.data.ctrl.shape).astype(np.float32))
            if external is not None:
                self.data.xfrc_applied.assign(np.broadcast_to(external, self.data.xfrc_applied.shape).astype(np.float32))
            wp.capture_launch(self.graph)

    def health(self):
        rows = self.metrics.numpy()
        names = ("failure_bits", "minimum_J", "peak_relative_residual", "overflow_or",
                 "peak_iterations", "plastic_norm", "steps")
        return [dict(zip(names, map(float, row))) for row in rows]

    def assert_valid(self):
        bad = [(i,row) for i,row in enumerate(self.health()) if row["failure_bits"]]
        if bad:
            raise RuntimeError(f"continuum GPU validity failure: {bad[:3]}")

    def sync(self, destination, world=0):
        # The exporter reallocates contact buffers. Reject invalid capacities
        # before invoking it; never hide an overflow by exporting truncated data.
        self.assert_valid()
        mjw.get_data_into(destination, self.cpu_model, self.data, world)

    def state(self):
        mjw.get_state(self.model, self.data, self._state, self.state_spec)
        return self._state.numpy().copy()

    def restore(self, state):
        value = np.asarray(state, dtype=np.float32)
        if value.shape != self._state.shape or not np.isfinite(value).all():
            raise ValueError("integration state has wrong shape or nonfinite values")
        self._state.assign(value)
        mjw.set_state(self.model, self.data, self._state, self.state_spec)
        self.data.overflow.zero_()
        self.metrics.assign(np.tile([0.,1.,0.,0.,0.,0.,0.], (self.data.nworld,1)))
        self.residual.first_bad_step.fill_(-1)
        self.residual.peak_relative.zero_()
        mjw.forward(self.model, self.data)

    def software(self):
        return dict(mujoco=mujoco.__version__, mujoco_warp=importlib.metadata.version("mujoco-warp"),
                    warp=wp.__version__, device=str(self.device), solver="Newton", dynamics="GPU")
