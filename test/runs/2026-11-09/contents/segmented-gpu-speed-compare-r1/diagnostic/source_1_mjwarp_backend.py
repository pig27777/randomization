"""MJWarp batched stepping with the CPU-validated v1.2 Warp force operators.

The first gate is unforced resting stability. Full grasp controllers are gated
by that check. Native MJWarp state is float32; material kernels remain float64.
"""
import mujoco_warp as mjw
from mujoco_warp._src.types import vec12
import numpy as np
import warp as wp

import mj_adapter as A
import warp_effects as E
from warp_mechanics import face_forces

wp.set_module_options({"enable_backward": False, "enable_fp_fma": False})


@wp.kernel
def promote_state(
    xpos: wp.array2d(dtype=wp.vec3), xmat: wp.array2d(dtype=wp.mat33),
    xquat: wp.array2d(dtype=wp.quat), xipos: wp.array2d(dtype=wp.vec3),
    qpos: wp.array2d(dtype=float),
    position: wp.array2d(dtype=wp.vec3d), orientation: wp.array2d(dtype=wp.mat33d),
    quaternion: wp.array2d(dtype=wp.vec4d), center: wp.array2d(dtype=wp.vec3d),
    q: wp.array2d(dtype=wp.float64),
):
    env, i = wp.tid()
    if i < xpos.shape[1]:
        p = xpos[env, i]
        r = xmat[env, i]
        a = xquat[env, i]
        c = xipos[env, i]
        position[env, i] = wp.vec3d(wp.float64(p[0]), wp.float64(p[1]), wp.float64(p[2]))
        center[env, i] = wp.vec3d(wp.float64(c[0]), wp.float64(c[1]), wp.float64(c[2]))
        quaternion[env, i] = wp.vec4d(wp.float64(a[0]), wp.float64(a[1]), wp.float64(a[2]), wp.float64(a[3]))
        orientation[env, i] = wp.mat33d(wp.float64(r[0, 0]), wp.float64(r[0, 1]), wp.float64(r[0, 2]),
            wp.float64(r[1, 0]), wp.float64(r[1, 1]), wp.float64(r[1, 2]),
            wp.float64(r[2, 0]), wp.float64(r[2, 1]), wp.float64(r[2, 2]))
    if i < qpos.shape[1]:
        q[env, i] = wp.float64(qpos[env, i])


@wp.kernel
def write_forces(values: wp.array3d(dtype=wp.float64), target: wp.array2d(dtype=wp.spatial_vector)):
    env, body = wp.tid()
    target[env, body] = wp.spatial_vector(float(values[env, body, 0]), float(values[env, body, 1]),
        float(values[env, body, 2]), float(values[env, body, 3]), float(values[env, body, 4]), float(values[env, body, 5]))


@wp.kernel
def write_equalities(values: wp.array3d(dtype=wp.float64), target: wp.array2d(dtype=vec12)):
    env, eq = wp.tid()
    value = target[env, eq]
    for i in range(12):
        value[i] = float(values[env, eq, i])
    target[env, eq] = value


@wp.kernel
def pack_contacts(
    nacon: wp.array(dtype=int), world: wp.array(dtype=int), geom: wp.array(dtype=wp.vec2i),
    frame: wp.array(dtype=wp.mat33), address: wp.array2d(dtype=int),
    counters: wp.array(dtype=int), overflow: wp.array(dtype=int),
    out_geom: wp.array3d(dtype=int), out_frame: wp.array3d(dtype=wp.float64), out_address: wp.array2d(dtype=int),
):
    i = wp.tid()
    if i >= nacon[0]:
        return
    env = world[i]
    if env < 0 or env >= counters.shape[0]:
        return
    slot = wp.atomic_add(counters, env, 1)
    if slot >= out_address.shape[1]:
        wp.atomic_add(overflow, 0, 1)
        return
    g = geom[i]
    out_geom[env, slot, 0] = g[0]
    out_geom[env, slot, 1] = g[1]
    r = frame[i]
    for row in range(3):
        for col in range(3):
            out_frame[env, slot, 3*row+col] = wp.float64(r[row, col])
    out_address[env, slot] = address[i, 0]


@wp.kernel
def promote_constraints(source: wp.array2d(dtype=float), target: wp.array2d(dtype=wp.float64)):
    env, row = wp.tid()
    target[env, row] = wp.float64(source[env, row])


class BatchStepper:
    """Homogeneous-model batch with independent native equality state per world."""
    def __init__(self, sim, nworld, nconmax=1024, njmax=8192):
        self.sim, self.nworld = sim, nworld
        m = sim.m
        self.model = mjw.put_model(m, batch_sizes={"eq_data": nworld})
        self.data = mjw.put_data(m, sim.d, nworld=nworld, nconmax=nconmax, njmax=njmax)
        self.external = wp.zeros((nworld,m.nbody,6),dtype=wp.float64)
        self.forces = wp.zeros_like(self.external)
        self.xpos = wp.zeros((nworld,m.nbody),dtype=wp.vec3d)
        self.xmat = wp.zeros((nworld,m.nbody),dtype=wp.mat33d)
        self.xquat = wp.zeros((nworld,m.nbody),dtype=wp.vec4d)
        self.xipos = wp.zeros((nworld,m.nbody),dtype=wp.vec3d)
        self.qpos = wp.zeros((nworld,m.nq),dtype=wp.float64)
        self.eq_data = wp.array(np.broadcast_to(m.eq_data,(nworld,*m.eq_data.shape)).copy(),dtype=wp.float64)
        self.overflow = wp.zeros(1,dtype=int)
        self._material = sim.bushings is not None
        if not self._material:
            return
        b = sim.bushings
        arrays = ((b.a,int),(b.b,int),(b.node_a,int),(b.node_b,int),(b.axis,int),
            (b.ra,wp.vec3d),(b.kt,wp.vec3d),(b.kr,wp.vec3d),(b.tension,wp.float64),
            (b.compression_travel,wp.float64),(b.weight_a,wp.float64),(b.weight_b,wp.float64))
        self.face_inputs = [wp.array(value,dtype=dtype) for value,dtype in arrays]
        self.wrenches = [wp.zeros((nworld,b.n),dtype=wp.vec3d) for _ in range(3)]
        self.compact = wp.zeros((nworld,b.n),dtype=wp.float64)
        self.node = wp.zeros((nworld,b.node_count),dtype=wp.vec3d)
        self.rho = wp.zeros((nworld,b.n),dtype=wp.float64)
        self.contact_geom = wp.zeros((nworld,nconmax,2),dtype=int)
        self.contact_frame = wp.zeros((nworld,nconmax,9),dtype=wp.float64)
        self.contact_address = wp.full((nworld,nconmax),-1,dtype=int)
        self.contact_count = wp.zeros(nworld,dtype=int)
        self.efc_force = wp.zeros((nworld,njmax),dtype=wp.float64)
        self.filtered = wp.zeros(nworld,dtype=wp.float64)
        self.raw = wp.zeros(nworld,dtype=wp.float64)
        arrays = ((sim.seg_bodies,int),(sim._obj_geom_mask,bool),(sim.welds,int),
                  (b.node_a,int),(b.node_b,int),(sim._seg_qadr,int))
        self.compaction_inputs = [wp.array(value,dtype=dtype) for value,dtype in arrays]

    def promote(self):
        d, m = self.data, self.sim.m
        wp.launch(promote_state,dim=(self.nworld,max(m.nbody,m.nq)),inputs=[d.xpos,d.xmat,d.xquat,d.xipos,d.qpos,
            self.xpos,self.xmat,self.xquat,self.xipos,self.qpos])

    def step(self):
        if not self._material:
            mjw.step(self.model,self.data)
            return
        d, m = self.data, self.sim.m
        wp.copy(self.forces,self.external)
        mjw.step1(self.model,d)
        self.promote()
        wp.launch(face_forces,dim=self.nworld,inputs=[*self.face_inputs,self.sim.bushings.centered,
            self.xpos,self.xmat,self.xquat,self.xipos,self.forces,*self.wrenches,self.compact,self.node,self.rho])
        wp.launch(write_forces,dim=(self.nworld,m.nbody),inputs=[self.forces,d.xfrc_applied])
        mjw.step2(self.model,d)
        self.promote()
        self.contact_count.zero_()
        self.contact_address.fill_(-1)
        wp.launch(pack_contacts,dim=d.naconmax,inputs=[d.nacon,d.contact.worldid,d.contact.geom,d.contact.frame,
            d.contact.efc_address,self.contact_count,self.overflow,self.contact_geom,self.contact_frame,self.contact_address])
        wp.launch(promote_constraints,dim=self.efc_force.shape,inputs=[d.efc.force,self.efc_force])
        wp.launch(E.compaction,dim=self.nworld,inputs=[*self.compaction_inputs,self.external,
            self.contact_geom,self.contact_frame,self.contact_address,self.efc_force,self.rho,self.qpos,
            m.opt.timestep,A.COMPACTION_LOAD_TAU,A.WELD_ON,A.WELD_RELEASE,A.COMPACTION_LOAD_MIN,
            self.filtered,self.raw,d.eq_active,self.eq_data])
        wp.launch(write_equalities,dim=(self.nworld,m.neq),inputs=[self.eq_data,self.model.eq_data])
        wp.launch(write_forces,dim=(self.nworld,m.nbody),inputs=[self.external,d.xfrc_applied])


@wp.kernel
def prescribe_pads(qadr: wp.array(dtype=int), vadr: wp.array(dtype=int),
                   position: wp.array2d(dtype=float), velocity: wp.array2d(dtype=float),
                   qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float)):
    env, i = wp.tid()
    qpos[env,qadr[i]] = position[env,i]
    qvel[env,vadr[i]] = velocity[env,i]


class DiagnosticSim(A.Sim):
    """Closed-loop R1--R3 diagnostic, driven by the frozen host controller.

    Only prescribed pad DOFs and external loads are uploaded. Material state is
    integrated on GPU. Per-step readback is intentional here and is not a batch
    grasp executor or a throughput benchmark.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._qadr = np.concatenate([self.pad_q[p] for p in self.pad_names]).astype(np.int32)
        self._vadr = np.concatenate([self.pad_v[p] for p in self.pad_names]).astype(np.int32)
        with wp.ScopedDevice("cuda:0"):
            self.batch = BatchStepper(self,1)
            self._qindices = wp.array(self._qadr)
            self._vindices = wp.array(self._vadr)
            self._pad_position = wp.zeros((1,len(self._qadr)),dtype=float)
            self._pad_velocity = wp.zeros_like(self._pad_position)

    def step(self):
        with wp.ScopedDevice("cuda:0"):
            self._pad_position.assign(self.d.qpos[self._qadr].astype(np.float32)[None])
            self._pad_velocity.assign(self.d.qvel[self._vadr].astype(np.float32)[None])
            self.batch.external.assign(self.d.xfrc_applied[None])
            wp.launch(prescribe_pads,dim=self._pad_position.shape,inputs=[self._qindices,self._vindices,
                self._pad_position,self._pad_velocity,self.batch.data.qpos,self.batch.data.qvel])
            self.batch.step()
            if int(self.batch.overflow.numpy()[0]):
                raise RuntimeError("GPU contact packing overflow in response diagnostic")
            if int(self.batch.data.nacon.numpy()[0])>self.batch.data.naconmax or np.any(self.batch.data.nefc.numpy()>self.batch.data.njmax):
                raise RuntimeError("GPU contact/constraint buffer overflow in response diagnostic")
            mjw.get_data_into(self.d,self.m,self.batch.data)
            synchronize_contact_ids(self.d)


def synchronize_contact_ids(data):
    # MJWarp 3.17 fills geom only; MuJoCo's legacy fields are separate storage.
    # The frozen v1.2 force/anchor measurements still consume those fields.
    data.contact.geom1[:] = data.contact.geom[:, 0]
    data.contact.geom2[:] = data.contact.geom[:, 1]
