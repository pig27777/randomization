"""Matched replays for the causal chain, modality interaction and controls."""


import numpy as np
import torch

from .geometry import InfoMode, vector
from .history import HISTORY_DIM
from .training import InformationProfile, score_candidates


MODALITY_PROFILES = {
    "J00": InformationProfile(InfoMode.V0_ONLY, "none"),
    "J10": InformationProfile(InfoMode.V1_ALLOWED, "none"),
    "J01": InformationProfile(InfoMode.V0_ONLY, "all"),
    "J11": InformationProfile(InfoMode.V1_ALLOWED, "all"),
    "without_effort": InformationProfile(InfoMode.V1_ALLOWED, "all", False),
    "without_temporal_history": InformationProfile(InfoMode.V1_ALLOWED, "last"),
}


def interaction(scores):
    return scores["J11"] - scores["J10"] - scores["J01"] + scores["J00"]


def fixed_try(action):
    action = vector(action, 13)
    if np.max(np.abs(action)) > 1.:
        raise ValueError("Fixed Try action must lie in [-1, 1]")
    return lambda _v0: action.copy()


def learned_try(actor):
    @torch.no_grad()
    def choose(v0):
        device = next(actor.parameters()).device
        _, a, _ = actor.sample(torch.as_tensor(v0.features(), device=device), deterministic=True)
        return a.cpu().numpy()
    return choose


def scored_grasp(encoder, scorer, profile):
    def choose(context, history, grasps):
        if not len(grasps):
            return None
        score = score_candidates(encoder, scorer, context, history, grasps, profile)
        return grasps[score.argmax()]
    return choose


def _run(make_env, seed, choose, profile, try_action=None, frozen_grasp=None, privileged=None,
         candidate_mode=None):
    env = make_env(seed, profile.mode)
    try:
        v0, _, info = env.reset(seed=seed)
        history = np.empty((0, HISTORY_DIM), dtype=np.float32)
        initial_id = np.asarray(info["initial_state"]).copy()
        if try_action is not None:
            history, _, _, _ = env.execute_probe(try_action)
        post_state = np.asarray(env.backend.state_identity()).copy()
        g = None
        if env.phase in ("initial", "ready"):
            if frozen_grasp is not None:
                g = frozen_grasp.copy()
            else:
                candidates = env.generate_grasp_candidates(candidate_mode)
                if privileged is not None:
                    # Explicit isolated evaluator. Ordinary selectors receive no simulator state.
                    g = privileged(env.backend.private_state(), candidates)
                else:
                    g = choose(env.public_context(), history, candidates)
            env.execute_grasp(g)
        summary = env.get_task_result()
        return dict(v0=v0, initial_state=initial_id, probe_state=post_state, history=history.copy(),
                    grasp_world=None if g is None else g.copy(), summary=summary)
    finally:
        env.backend.close()


def causal_chain(make_env, seed, no_try_selector, no_history_selector, full_selector, probe_policy, *,
                 public_mode=InfoMode.V1_ALLOWED):
    """Supply independently fitted selectors for A, C and D."""
    public_mode = InfoMode(public_mode)
    a_mode = InfoMode.WORLD_COMMON if public_mode is InfoMode.WORLD_COMMON else InfoMode.V0_ONLY
    a = _run(make_env, seed, no_try_selector, InformationProfile(a_mode, "none"))
    if a["grasp_world"] is None:
        raise ValueError("A needs a feasible g_A to evaluate B")
    action = vector(probe_policy(a["v0"]), 13)
    b = _run(make_env, seed, None, InformationProfile(public_mode, "none"),
             try_action=action, frozen_grasp=a["grasp_world"])
    c = _run(make_env, seed, no_history_selector, InformationProfile(public_mode, "none"), try_action=action)
    d = _run(make_env, seed, full_selector, InformationProfile(public_mode), try_action=action)
    for result in (b, c, d):
        if not np.array_equal(a["initial_state"], result["initial_state"]):
            raise RuntimeError("Initial states differed across the causal chain")
        if any(not np.array_equal(b[key], result[key]) for key in ("probe_state", "history")):
            raise RuntimeError("Post-Try state or history differed across the causal chain")
    return dict(A=a, B=b, C=c, D=d)


def matched_modalities(make_env, seed, action, selectors):
    """The selectors map contains independently trained heads for all four cells."""
    if set(selectors) != {"J00", "J10", "J01", "J11"}:
        raise ValueError("Four independently trained modality selectors are required")
    results = {name: _run(make_env, seed, select, MODALITY_PROFILES[name], try_action=action,
                          candidate_mode=InfoMode.WORLD_COMMON)
               for name, select in selectors.items()}
    ref = results["J00"]
    if any(not np.array_equal(ref[key], result[key]) for result in results.values()
           for key in ("initial_state", "probe_state", "history")):
        raise RuntimeError("The modality cells must replay the same initial state and Try")
    return results


def supplemental_control(make_env, seed, action, *, profile, selector=None, privileged=None):
    if (selector is None) == (privileged is None):
        raise ValueError("Provide either a public selector or an explicitly privileged evaluator")
    return _run(make_env, seed, selector, profile, try_action=action, privileged=privileged)
