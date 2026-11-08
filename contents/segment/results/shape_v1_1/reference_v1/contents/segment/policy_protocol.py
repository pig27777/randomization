"""Read-only adapter of policy/env.py to the current content model.

Keep the frozen task implementation in one place. Only imports, the stepping
entry point required by bushing, and an optional observer are adapted. The policy
snapshot and its source files are never changed by this module.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import mj_adapter  # ensure the live content implementation is already imported
import model
import sampling

SOURCE = Path(__file__).resolve().parents[2] / "policy" / "env.py"


def load_protocol(table=None):
    source = SOURCE.read_text()
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    changes = {
        'sys.path.insert(0, str(Path(__file__).resolve().parent / "content_snapshot"))': '',
        'from model import G, PAD, Kappa, build, load_table, sample_kappa':
            'from model import G, PAD, Kappa, build, load_table\nfrom sampling import sample_kappa',
        'def evaluate(object_spec: dict, grasp_params, seed: int) -> dict:':
            'def evaluate(object_spec: dict, grasp_params, seed: int, observer=None) -> dict:',
        'mujoco.mj_step(m, dd)': 'sim.step()',
        # README's already-approved I-28 limit; policy/env.py still carries the old 4 s.
        'or tau > 4.0:': 'or tau > 8.0:',
        '        if n % check_every == 0:\n            gripped =':
            '        if n % check_every == 0:\n            if observer is not None:\n                observer(sim, phase, t, f_ctrl)\n            gripped =',
    }
    for before, after in changes.items():
        if source.count(before) != 1:
            raise RuntimeError(f"policy protocol changed; review adapter for {before!r}")
        source = source.replace(before, after)
    namespace = {"__name__": "content_policy_protocol", "__file__": str(SOURCE)}
    exec(compile(source, str(SOURCE), "exec"), namespace)
    if table is not None:
        namespace["_TABLE"] = table
    return namespace["evaluate"], source_hash
