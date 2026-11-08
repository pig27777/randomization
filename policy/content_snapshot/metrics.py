"""Engine-independent response metrics and failure geometry (test/content_model/README.md, 2026-11-04)."""
import numpy as np

E0 = 0.02            # secant start
E_MAX = 0.15         # secant end cap
LOCK_FRAC = 0.6      # secant end <= 0.6 x lock strain


def first_peak(e, F, drop=0.1, f_min=1.0):
    """Index of the first force peak followed by a drop of more than `drop` (buckling), else None."""
    run_max, i_max = -np.inf, None
    for i, f in enumerate(F):
        if f > run_max:
            run_max, i_max = f, i
        elif run_max > f_min and f < (1 - drop) * run_max:
            return i_max
    return None


def r1_metrics(arr, W, f_cap=50.0):
    """arr rows: (t, strain from both-pad contact, mean pad force).

    Secant from E0 to min(E_MAX, LOCK_FRAC * lock), only before the first force peak (buckling); when the
    force limit comes first the value is a lower bound. Lock: first strain >= 0.05 where the tangent exceeds
    5 x the initial (E0 to 0.05) stiffness."""
    e, F = arr[:, 1], arr[:, 2]
    keep = np.concatenate([[True], np.diff(e) > 1e-7])         # loading branch only
    e, F = e[keep], F[keep]
    out = {"buckling_force": None, "lower_bound": False}
    ip = first_peak(e, F)
    if ip is not None and e[ip] > E0:
        out["buckling_force"] = float(F[ip])
        e, F = e[: ip + 1], F[: ip + 1]
    emax = float(e.max())
    mm = W * 1e3
    lock = None
    k0 = None
    if emax > 0.05:
        k0 = (np.interp(0.05, e, F) - np.interp(E0, e, F)) / ((0.05 - E0) * mm)
        h = 0.005
        grid = np.arange(0.05, emax - h, 0.0025)
        if len(grid) and k0 > 0:
            kt = (np.interp(grid + h, e, F) - np.interp(grid - h, e, F)) / (2 * h * mm)
            hit = np.where(kt > 5 * k0)[0]
            lock = float(grid[hit[0]]) if len(hit) else None
    end = min(E_MAX, LOCK_FRAC * lock) if lock else E_MAX
    if emax >= end and end > E0:
        k = (np.interp(end, e, F) - np.interp(E0, e, F)) / ((end - E0) * mm)
        rng = [E0, end]
    else:
        # force limit (or buckling) before the interval end: lower bound over what was measured
        out["lower_bound"] = True
        i1 = len(F) - 1
        below = np.where(F < 0.1 * F[i1])[0]               # skip contact-onset spikes
        i0 = int(below[-1]) if len(below) else 0
        e_lo = max(E0, float(e[i0])) if emax > E0 else float(e[i0])
        de = (emax - e_lo) * mm
        k = (F[i1] - np.interp(e_lo, e, F)) / de if de > 0 else float("nan")
        rng = [e_lo, emax]
    out.update({"stiffness": float(k), "secant_range": rng, "lock_strain": lock,
                "initial_stiffness": None if k0 is None else float(k0),
                "f30": float(np.interp(0.30, e, F)) if emax >= 0.30 else None})
    return out


def period_mean(t, x, t0, t1, fallback=0.5):
    """Mean of x over an integer number of oscillation periods inside [t0, t1] (zero crossings of x - mean);
    falls back to the last `fallback` seconds when no oscillation is found."""
    t, x = np.asarray(t), np.asarray(x)
    sel = (t >= t0) & (t <= t1)
    ts, xs = t[sel], x[sel]
    if len(xs) < 4:
        return float(np.mean(xs)) if len(xs) else float("nan")
    s = np.sign(xs - xs.mean())
    cross = np.where(np.diff(s) != 0)[0]
    if len(cross) >= 3:
        n = len(cross) - 1 if (len(cross) - 1) % 2 == 0 else len(cross) - 2
        a, b = cross[-1 - n], cross[-1]
        if b > a:
            return float(np.mean(xs[a:b]))
    return float(np.mean(xs[ts >= t1 - fallback]))


class JointGeometry:
    """Per-joint arrays to check README failure modes from segment poses (any joint implementation)."""

    def __init__(self, desc, edge_arm=0.9):
        from model import _bend_lower_length
        js = [j for j in desc.joints]
        self.n = len(js)
        if not self.n:
            return
        self.a = np.array([j.a for j in js])
        self.b = np.array([j.b for j in js])
        self.rest = np.array([j.rest for j in js])
        self.lock = np.array([j.lock_comp for j in js])
        offs, lows = [], []
        for j in js:
            half = desc.segments[j.a].half
            o = [x for x in range(3) if x != j.axis]
            jo, jl = [], []
            for along in (o[1], o[0]):
                arm = edge_arm * half[along]
                lo = min(_bend_lower_length(j.rest, arm, j.lock_bend), j.rest * (1 - j.lock_comp) - 1e-5)
                for sgn in (1, -1):
                    v = np.zeros(3)
                    v[along] = sgn * arm
                    jo.append(v)
                    jl.append(lo)
            offs.append(jo)
            lows.append(jl)
        self.off = np.array(offs)            # (n, 4, 3) in the segment frame (= object frame at rest)
        self.low = np.array(lows)            # (n, 4)

    def check(self, pos, rot):
        """pos (nseg,3), rot (nseg,3,3) world. Returns failure string or None."""
        if not self.n:
            return None
        pa, pb = pos[self.a], pos[self.b]
        L = np.linalg.norm(pb - pa, axis=1)
        strain = (self.rest - L) / self.rest
        if strain.max() > 0.5:
            return f"adjacent overlap {strain.max():.2f} of segment size"
        r = strain / self.lock
        if r.max() > 1.2:
            return f"compression {r.max():.2f} x limit"
        ea = pa[:, None, :] + np.einsum("nij,nkj->nki", rot[self.a], self.off)
        eb = pb[:, None, :] + np.einsum("nij,nkj->nki", rot[self.b], self.off)
        Le = np.linalg.norm(eb - ea, axis=2)
        trav = (self.rest[:, None] - Le) / (self.rest[:, None] - self.low)
        if trav.max() > 1.2:
            return f"bending {trav.max():.2f} x limit"
        return None

    def max_strain(self, pos):
        if not self.n:
            return 0.0
        return float(np.max((self.rest - np.linalg.norm(pos[self.b] - pos[self.a], axis=1)) / self.rest))
