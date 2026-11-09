"""Diagnostics and optional refinement of MJWarp's effective-metric solve.

Measure immediately after fwd_acceleration, before constraint solving/integration:
    state = ResidualState(model, data, relative_tolerance=1e-5)
    step2_callback = functools.partial(step2_with_residual, state=state)
The native operator remains float32; only diagnostic reductions use float64.
The recomputed residual can differ from PCG's recursively updated residual.
``refinement_steps=1`` or ``2`` explicitly enables correction of qacc_smooth in
the step2 hook. The default is zero; measure() is always read-only. Refinement
solves the same equation with its explicit defect as RHS, then accepts only a
finite candidate with a strictly smaller explicit residual. It cannot guarantee
convergence beyond float32 operator/solution accuracy; the original threshold
still decides acceptance after all configured corrections.

GPU PCG uses max(opt.tolerance, 1e-5); double-precision CPU MuJoCo uses
opt.tolerance (currently 1e-8). Both share opt.iterations with the constraint
solver. MJWarp does not expose PCG cap/breakdown status, so finite output or
solver_niter alone cannot establish this equation's convergence.

With sleep disabled, discrete step2 leaves the previous equation caches in place,
but its qpos/qvel/time have advanced. Sleep can rebuild efm_c during integration.
The explicit pre-constraint hook below avoids relying on stale cache timing.
"""
import math

import warp as wp
from mujoco_warp._src import derivative, forward, sensor, smooth, solver, types


@wp.func
def _residual_summary(
    lhs: wp.array2d(dtype=float), rhs: wp.array2d(dtype=float), acceleration: wp.array2d(dtype=float),
    w: int, tolerance: wp.float64, minval: wp.float64,
):
    rr = wp.float64(0.)
    bb = wp.float64(0.)
    largest = wp.float64(0.)
    worst = int(-1)
    first_nonfinite = int(-1)
    flags = int(0)
    for j in range(lhs.shape[1]):
        a = wp.float64(lhs[w, j])
        b = wp.float64(rhs[w, j])
        good_a = wp.isfinite(a)
        good_b = wp.isfinite(b)
        good_q = wp.isfinite(acceleration[w, j])
        if not good_a:
            flags = flags | 1
        if not good_b:
            flags = flags | 2
        if not good_q:
            flags = flags | 4
        if not good_a or not good_b or not good_q:
            if first_nonfinite < 0:
                first_nonfinite = j
        if good_b:
            bb += b*b
        if good_a and good_b:
            r = a-b
            rr += r*r
            if wp.abs(r) > largest:
                largest = wp.abs(r)
                worst = j
    absolute = wp.sqrt(rr)
    rhs_norm = wp.sqrt(bb)
    relative = wp.sqrt(rr / wp.max(bb, minval))
    tiny = bb <= minval and (flags & 2) == 0
    # Native PCG treats a RHS squared norm <= MJ_MINVAL as zero before iterating.
    # Use that same absolute scale only on this branch; normal RHS uses relative L2.
    bad = relative >= tolerance
    if tiny:
        bad = rr > minval
    if flags != 0:
        relative = wp.float64(wp.inf)
        absolute = wp.float64(wp.inf)
        largest = wp.float64(wp.inf)
        worst = first_nonfinite
        bad = True
        if (flags & 2) != 0:
            rhs_norm = wp.float64(wp.inf)
    return wp.vec4d(relative, absolute, rhs_norm, largest), worst, flags, tiny, bad


@wp.kernel
def residual_metrics(
    lhs: wp.array2d(dtype=float), rhs: wp.array2d(dtype=float), acceleration: wp.array2d(dtype=float),
    tolerance: wp.float64, minval: wp.float64,
    step: wp.array(dtype=int), last: wp.array2d(dtype=wp.float64),
    worst_dof: wp.array(dtype=int), nonfinite: wp.array(dtype=int), small_rhs: wp.array(dtype=int),
    peak_relative: wp.array(dtype=wp.float64), first_bad_step: wp.array(dtype=int),
    first_bad_dof: wp.array(dtype=int), first_bad_metrics: wp.array2d(dtype=wp.float64),
    first_bad_nonfinite: wp.array(dtype=int),
):
    w = wp.tid()
    metrics, worst, flags, tiny, bad = _residual_summary(lhs, rhs, acceleration, w, tolerance, minval)
    for j in range(4):
        last[w, j] = metrics[j]
    worst_dof[w] = worst
    nonfinite[w] = flags
    small_rhs[w] = int(tiny)
    peak_relative[w] = wp.max(peak_relative[w], metrics[0])
    if bad and first_bad_step[w] < 0:
        first_bad_step[w] = step[w]
        first_bad_dof[w] = worst
        first_bad_nonfinite[w] = flags
        for j in range(4):
            first_bad_metrics[w, j] = last[w, j]
    step[w] += 1


@wp.kernel
def _refinement_rhs(
    lhs: wp.array2d(dtype=float), rhs: wp.array2d(dtype=float), acceleration: wp.array2d(dtype=float),
    tolerance: wp.float64, minval: wp.float64, first_pass: bool,
    before: wp.array2d(dtype=wp.float64), active: wp.array(dtype=bool),
    attempted: wp.array(dtype=int), defect: wp.array2d(dtype=float),
):
    w = wp.tid()
    metrics, worst, flags, tiny, bad = _residual_summary(lhs, rhs, acceleration, w, tolerance, minval)
    if first_pass:
        for j in range(4):
            before[w, j] = metrics[j]
    active[w] = bad and flags == 0
    if active[w]:
        attempted[w] += 1
    for j in range(lhs.shape[1]):
        defect[w, j] = 0.
        if active[w]:
            defect[w, j] = float(wp.float64(rhs[w, j]) - wp.float64(lhs[w, j]))


@wp.kernel
def _refinement_candidate(
    acceleration: wp.array2d(dtype=float), correction: wp.array2d(dtype=float),
    active: wp.array(dtype=bool), candidate: wp.array2d(dtype=float),
):
    w, j = wp.tid()
    candidate[w, j] = acceleration[w, j]
    if active[w]:
        candidate[w, j] = float(wp.float64(acceleration[w, j]) + wp.float64(correction[w, j]))


@wp.kernel
def _accept_refinement(
    lhs: wp.array2d(dtype=float), rhs: wp.array2d(dtype=float), acceleration: wp.array2d(dtype=float),
    candidate_lhs: wp.array2d(dtype=float), candidate: wp.array2d(dtype=float),
    tolerance: wp.float64, minval: wp.float64, active: wp.array(dtype=bool),
    applied: wp.array(dtype=int), rejected: wp.array(dtype=int), nonfinite: wp.array(dtype=int),
):
    w = wp.tid()
    if active[w]:
        old, old_worst, old_flags, old_tiny, old_bad = _residual_summary(lhs, rhs, acceleration, w, tolerance, minval)
        new, worst, flags, tiny, bad = _residual_summary(candidate_lhs, rhs, candidate, w, tolerance, minval)
        nonfinite[w] = nonfinite[w] | flags
        if flags == 0 and new[1] < old[1]:
            applied[w] += 1
            for j in range(lhs.shape[1]):
                acceleration[w, j] = candidate[w, j]
                lhs[w, j] = candidate_lhs[w, j]
        else:
            rejected[w] += 1


class ResidualState:
    """Per-world residuals; measure() never changes solver state.

``last``/``first_bad_metrics`` columns are normalized L2, absolute L2, RHS L2,
and maximum absolute residual component. ``nonfinite`` bits are lhs=1, rhs=2,
qacc_smooth=4. ``first_bad_step`` is a zero-based measurement index, -1 if none.
``small_rhs`` marks the native near-zero RHS branch described above.
The caller must explicitly choose the relative threshold; no acceptance policy
is inferred from finite values or from the iteration count.
When refinement_steps is nonzero, refine() changes only qacc_smooth and its own
scratch. refinement_before preserves pre-correction metrics; attempted/applied/
rejected/nonfinite arrays describe the latest refine() call, not cumulative
counts. The main first_bad latch records only calls to measure().
    """

    def __init__(self, model, data, *, relative_tolerance, refinement_steps=0):
        if model.opt.integrator != types.IntegratorType.DISCRETE:
            raise ValueError("effective-metric residual requires the discrete integrator")
        if not math.isfinite(relative_tolerance) or relative_tolerance <= 0:
            raise ValueError("relative_tolerance must be finite and positive")
        if refinement_steps not in (0, 1, 2):
            raise ValueError("refinement_steps must be 0, 1, or 2")
        self.model, self.data = model, data
        self.relative_tolerance = float(relative_tolerance)
        self.refinement_steps = int(refinement_steps)
        with wp.ScopedDevice(data.qpos.device):
            self.lhs = wp.zeros_like(data.qacc_smooth)
            self.rhs = wp.zeros_like(data.qacc_smooth)
            self.step = wp.zeros(data.nworld, dtype=int)
            self.last = wp.zeros((data.nworld, 4), dtype=wp.float64)
            self.worst_dof = wp.full(data.nworld, -1, dtype=int)
            self.nonfinite = wp.zeros(data.nworld, dtype=int)
            self.small_rhs = wp.zeros(data.nworld, dtype=int)
            self.peak_relative = wp.zeros(data.nworld, dtype=wp.float64)
            self.first_bad_step = wp.full(data.nworld, -1, dtype=int)
            self.first_bad_dof = wp.full(data.nworld, -1, dtype=int)
            self.first_bad_metrics = wp.zeros((data.nworld, 4), dtype=wp.float64)
            self.first_bad_nonfinite = wp.zeros(data.nworld, dtype=int)
            if self.refinement_steps:
                self.refinement_before = wp.zeros((data.nworld, 4), dtype=wp.float64)
                self.refinement_attempted = wp.zeros(data.nworld, dtype=int)
                self.refinement_applied = wp.zeros(data.nworld, dtype=int)
                self.refinement_rejected = wp.zeros(data.nworld, dtype=int)
                self.refinement_nonfinite = wp.zeros(data.nworld, dtype=int)
                self._refinement_active = wp.zeros(data.nworld, dtype=bool)
                self._defect = wp.zeros_like(data.qacc_smooth)
                self._correction = wp.zeros_like(data.qacc_smooth)
                self._candidate = wp.zeros_like(data.qacc_smooth)
                self._candidate_lhs = wp.zeros_like(data.qacc_smooth)

    def reset(self):
        """Clear diagnostics after warmup/capture; never reset physics arrays."""
        for value in (self.step, self.last, self.nonfinite, self.small_rhs, self.peak_relative,
                      self.first_bad_metrics, self.first_bad_nonfinite):
            value.zero_()
        for value in (self.worst_dof, self.first_bad_step, self.first_bad_dof):
            value.fill_(-1)
        if self.refinement_steps:
            for value in (self.refinement_before, self.refinement_attempted, self.refinement_applied,
                          self.refinement_rejected, self.refinement_nonfinite):
                value.zero_()

    def _equation(self):
        m, d = self.model, self.data
        derivative.eff_mul_m(m, d, self.lhs, d.qacc_smooth)
        wp.launch(derivative._eff_rhs, dim=(d.nworld, m.nv), inputs=[
            m.opt.enableflags, m.dof_treeid, d.tree_awake,
            d.qfrc_smooth, d.efm_c, d.efm_ca, self.rhs])

    def refine(self):
        """Optionally correct the current equation; never latch a measurement.

eff_solve's explicit qfrc argument is the complete correction RHS, so efm_c and
efm_ca must NOT be added again. The same operator, preconditioner and iteration
budget are reused. Inactive worlds get zero correction RHS and are not changed.
Native PCG cap/breakdown flags are still unavailable: an explicit residual is
required after refinement, and failed/nonfinite candidates remain visible.
        """
        if not self.refinement_steps:
            return
        m, d = self.model, self.data
        with wp.ScopedDevice(d.qpos.device):
            for value in (self.refinement_attempted, self.refinement_applied,
                          self.refinement_rejected, self.refinement_nonfinite):
                value.zero_()
            self._equation()
            for iteration in range(self.refinement_steps):
                wp.launch(_refinement_rhs, d.nworld, inputs=[
                    self.lhs, self.rhs, d.qacc_smooth, self.relative_tolerance, types.MJ_MINVAL,
                    iteration == 0, self.refinement_before, self._refinement_active,
                    self.refinement_attempted, self._defect])
                derivative.eff_solve(m, d, self._correction, qfrc=self._defect)
                wp.launch(_refinement_candidate, (d.nworld, m.nv), inputs=[
                    d.qacc_smooth, self._correction, self._refinement_active, self._candidate])
                derivative.eff_mul_m(m, d, self._candidate_lhs, self._candidate)
                wp.launch(_accept_refinement, d.nworld, inputs=[
                    self.lhs, self.rhs, d.qacc_smooth, self._candidate_lhs, self._candidate,
                    self.relative_tolerance, types.MJ_MINVAL, self._refinement_active,
                    self.refinement_applied, self.refinement_rejected, self.refinement_nonfinite])

    def measure(self):
        """Re-evaluate the current smooth equation before solver/integration.

The native metric operator also accounts for flex contact, tendon, actuator,
fluid, bending and stretch contributions. Its temporary buffers are separate
from the physical state, and its contact metric is rebuilt for this sample.
        """
        m, d = self.model, self.data
        with wp.ScopedDevice(d.qpos.device):
            self._equation()
            wp.launch(residual_metrics, dim=d.nworld, inputs=[
                self.lhs, self.rhs, d.qacc_smooth, self.relative_tolerance, types.MJ_MINVAL,
                self.step, self.last, self.worst_dof, self.nonfinite, self.small_rhs,
                self.peak_relative, self.first_bad_step, self.first_bad_dof,
                self.first_bad_metrics, self.first_bad_nonfinite])


def step2_with_residual(m, d, *, state):
    """Native discrete step2 with opt-in refinement, then a residual sample."""
    if m is not state.model or d is not state.data:
        raise ValueError("ResidualState must belong to this model and data")
    if m.opt.integrator != types.IntegratorType.DISCRETE:
        raise ValueError("step2_with_residual supports only discrete integration")
    forward.fwd_actuation(m, d)
    forward.fwd_acceleration(m, d)
    state.refine()
    state.measure()
    solver.solve(m, d)
    forward._launch_discrete_free_gyro(m, d, d.qacc, d.qacc, False)
    if m.opt.run_rne_postconstraint or (not (m.opt.disableflags & types.DisableBit.SENSOR) and m.sensor_rne_postconstraint):
        smooth.rne_postconstraint(m, d)
    sensor.sensor_acc(m, d, skip_rne_postconstraint=True)
    forward.discrete(m, d)
