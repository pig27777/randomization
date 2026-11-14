"""Reviewed v1.3 gathering spring; model attachments remain in adapters."""
import math

GRIP_STIFFNESS = 40000.  # N/m, fixed by the reviewed effect rule
GATHER_TRAVEL = .010    # total reduction in the fingertip gap before force begins


class PinchSpring:
    def __init__(self):
        self.reference_gap = None

    def arm(self, gap, eligible):
        if not math.isfinite(gap) or gap < 0:
            raise ValueError('finite nonnegative fingertip gap required')
        self.reference_gap = float(gap) if eligible else None

    def force(self, gap):
        if not math.isfinite(gap):
            raise ValueError('finite fingertip gap required')
        if self.reference_gap is None:
            return 0.
        return GRIP_STIFFNESS * max(self.reference_gap - GATHER_TRAVEL - gap, 0.)

    def release(self):
        self.reference_gap = None
