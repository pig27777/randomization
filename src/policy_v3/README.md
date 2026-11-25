# Two-Stage Active Proprioceptive Grasping V3-MVP

This package wraps Yam_B's 7D control contract. Run it from the repository root with the existing physical parcel Python environment. The learner uses CPU PyTorch and the rigid smoke uses native MuJoCo. The GPU adapter uses the shared MJWarp interface and the PE Flex scene uses the existing `Parcel` model and GPU bending kernel.

```sh
export PYTHONPATH=src:experience:.
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
P=.venv-physical-parcel/bin/python
R=test/runs/YYYY-MM-DD/policy/policy-v3-local

"$P" -B -m unittest discover -s tests/policy_v3 -v
"$P" -B -m policy_v3.curriculum smoke "$R/smoke" --devices cpu,cpu \
  --episodes 100 --matched-repeats 2 --candidates 8 --ppo-episodes 16
"$P" -B -m policy_v3.curriculum c1 "$R/collect" --devices cpu,cpu,cpu,cpu \
  --episodes 1000 --matched-repeats 2 --candidates 8
"$P" -B -m policy_v3.curriculum c2 "$R/c2" --records "$R/collect/c1" --bce-epochs 20
"$P" -B -m policy_v3.curriculum c3 "$R/c3" --devices cpu,cpu --episodes 16 \
  --candidates 8 --checkpoint "$R/c2/checkpoint.pt"
"$P" -B -m policy_v3.curriculum c4 "$R/c4" --devices cpu,cpu --episodes 16 \
  --candidates 8 --records "$R/collect/c1" --checkpoint "$R/c2/checkpoint.pt" --rounds 2
```

Keep each output directory unique. The default candidate budget is 256. The sampler returns the feasible set up to that budget and records its size. The smoke uses eight to limit runtime. Each Try executes one grasp. Matched repeats independently reset and replay the initial state and action, then draw a different feasible grasp. The learner groups by both initial and probe state IDs, and pins the holdouts through C4 updates.

## GPU backends

```sh
flock /tmp/claude-gpu.lock env MUJOCO_GL=egl "$P" -B -m policy_v3.curriculum \
  c0 "$R/flex-probe" --devices 0 --scene flex --probe-only

MUJOCO_GL=egl "$P" -B -m policy_v3.curriculum c1 "$R/flex-multi" \
  --scene flex --devices 0,1 --episodes 100
```

Use the lock around single-GPU repository runs. Multi-GPU workers start as independent subprocesses with one physical GPU ID each. Each process addresses its local `cuda:0` and uses `/tmp/warp_cache_gpu_<id>` for Warp compilation. The launcher has no dependency on the shared lock. It releases the central update only after every worker returns the requested batch with the same version and checksum. Learner checkpoints include the critic, optimizer state and split assignments. The worker snapshot contains the actor, encoder, scorer and collection configuration.

The built-in scenes are `rigid` and `flex`. The optional `--factory module:callable` accepts another backend implementing the contract used by `MacroBagGraspEnv`, including `reset(seed=...)`, `observe()`, `state_identity()`, `private_state()`, and the motion and planning methods used by `ProbeActionExecutor` and `SE3GraspCandidateSampler`. The existing Newton/VBD fixture experiments require an articulated backend adapter to use this entry. The pre-existing 7D PPO environments remain available through their original entry points.

## Observations, actions and records

- Flex visual features come from a simulated exterior segmentation mask and metric depth. A real camera provider can use `outer_geometry()` with a calibrated camera and runtime exterior mask. The rigid default uses tagged `fixture_geometry` for software verification. Pass `--camera` for rigid image-based geometry.
- Learner public inputs consist of permitted exterior features, the 13D action, permitted history and the 11D candidate. `world_common` and `v0_only` skip the post-Try capture and accessors. The private critic uses only the initial privileged state. The collision planner uses a separate empty ABC workcell.
- The episode archive stores `H_try` as 48D encoder frames and `H_try_raw` as 54D full feedback. `joint_tracking_error` is the six-channel difference between joint position command and joint position. It remains in the raw layout. The archive also stores original timestamps, measured world TCP positions, joint feedback, the actual 7D control command and per-frame phase. Absolute TCP orientation is available through FK from the raw joint positions.
- Pose parameters are in the world frame at execution and expressed in the mode-specific frame for the scorer. Rotation 6D encodes the first two matrix columns. Jaw width is in meters. Effort is the DM4310 output-shaft torque cap in N·m. The native slider force limit uses `force = torque / (0.096 / 6.57)`, consistent with the existing Yam_B transmission model. Physical deployment requires calibrated camera, TCP, effort and safety parameters.
- GPU safety checks latch penetration, collision, excess normal contact load, solver overflow, nonfinite state or excessive film elongation. A latched violation zeros actuation before the next solve and ends the macro motion. These simulator diagnostics are used for safety and labeling. Deployable proprioception uses motor feedback and FK.
- Replay retains physics-invalid episodes for accounting. BCE reads only valid episodes with an executed grasp. PPO charges the unsafe cost for preflight rejection or unsafe action-induced probe invalidity. Other physics-invalid episodes are excluded. Worker `on_policy.npz` preserves `u`, `a_try` and the corrected old log probability for audited updates.

## Evaluation

`evaluation.py` exposes A/B/C/D with a frozen world-frame `g_A`, matched post-Try state checks, and the four independent modality profiles with the interaction contrast. Fixed Try, learned Try, an explicitly privileged evaluator, and effort and temporal ablations are selectable.

For a shared 2 × 2 training pool, collect with `--info-mode v1_allowed --shared-world-candidates`. Train four independent C2 checkpoints by combining `--info-mode v0_only|v1_allowed` with `--history none|all`. The evaluator also draws a common world candidate set for the four cells. C2 reports both held-out score AUC and same-post-Try paired concordance when paired labels are available.
