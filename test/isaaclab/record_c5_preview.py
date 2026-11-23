"""Record the C5 top-grasp preview and audit its measured geometry and loads."""
import argparse
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
import torch
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim.utils.stage import use_stage

from check_route_b_closed_loop import acquisition, transition
from src.isaac_yam_parcel.yam_parcel_base_env import YAMPickEnv, yam_parcel_cfg
from src.isaac_yam_parcel.yam_scene import YAM_USD
from src.low_order_parcel.domain_randomizer import CornerSuite, DomainRandomizer


def preview_target(env, time, start, offer, start_rail, closed):
    approach = transition(time, .35, 1.85)
    top = 2 * env.episode.payload.bag_half_extents[:, 2]
    command = start.clone()
    command[:, 0] += .055 * approach + .040 * transition(time, 3.45, 4.1)
    # Once clear, retract towards the base so the two hardware variants
    # can realize the same wrist pose through the end of the replay.
    command[:, 0] -= .120 * transition(time, 4.12, 6.4)
    command[:, 1] += .050 * approach - .045 * transition(time, 4.12, 6.4)
    height = torch.lerp(start[:, 2], top + .025, approach)
    height = torch.lerp(height, top - .020, transition(time, 1.85, 3.26))
    command[:, 2] = height + .065 * transition(time, 3.52, 4.30) + .13 * transition(time, 4.8, 7.2)
    rail = torch.lerp(start_rail, closed, transition(time, 1.50, 2.48))
    rail += env.step_limit[-1] * 4 * transition(time, 2.40, 2.55) / (1000 * env.rail_overtravel)
    if env.result is not None:
        retreat = env.ports[0].read().pads[:, :, 2].mean(dim=1) + .001
        command[:, 2] = torch.where(env.result.press_N[:, 0] > 3.4,
                                    torch.maximum(command[:, 2], retreat), command[:, 2])
    return command, rail


class RecordingEnv(YAMPickEnv):
    def _apply_action(self):
        super()._apply_action()
        robot = self.robots[0]
        state = self.parcel_state.payload
        half = self.parcel_sim.payload.half_thickness(state.d_surf)[0]
        center = state.pos[0, :, 2]
        self.substep_bounds.append(torch.stack(((center - half).min(), (center + half).max())))
        self.substeps.append(torch.cat((robot.data.joint_pos.torch[0], self.result.press_N[0, :1],
            robot.rail_load[0], robot.actuators.applied_effort.torch[0, robot.rail_ids],
            robot.data.body_link_pose_w.torch[0, self.ports[0].pad_ids].flatten())).clone())

    def step(self, actions):
        result = super().step(actions)
        self._draw_parcel()
        if self.flex is not None:
            import warp as wp
            index = len(self.frames)
            measured = self.robots[0].data.body_link_pose_w.torch[0, self.ports[0].pad_ids].cpu().numpy()
            if not np.allclose(measured, self.reference[index], atol=1e-6, rtol=0.):
                raise ValueError(f"Arm pose differs from the reference at frame {index}")
            viewer = self.sim._visualizers[0]
            points = wp.array(self.flex["vertices"][index], dtype=wp.vec3, device=str(self.device))
            viewer.log_mesh("/Parcel/film", points, self.flex_faces, color=(.25,.49,.46),
                            roughness=.52, backface_culling=False, dynamic=True)
        self.sim.render()
        frame = self.sim._visualizers[0].render_rgb_array()[..., :3]
        if self.encoder is None:
            h, w = frame.shape[:2]
            self.encoder = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-n", "-f", "rawvideo",
                "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "25", "-i", "-", "-an",
                "-vf", "scale=1280:-2", "-c:v", "libx264", "-crf", "17", "-threads", "2",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.out / ("flex.mp4" if self.flex is not None else "simplified.mp4"))],
                stdin=subprocess.PIPE)
        self.encoder.stdin.write(frame.tobytes())
        verts, corners, pads, mesh, bounds = self.markers.last
        # Conservative test on triangles intersecting each projected finger box.
        triangles = mesh[self.markers.top_faces]
        clearances = []
        for low, high, target in bounds:
            hit = ((triangles[:, :, :2].max(1) >= low) & (triangles[:, :, :2].min(1) <= high)).all(1)
            if hit.any():
                clearances.append(target + .00155 - triangles[hit, :, 2].max())
        robot = self.robots[0]
        state = self.parcel_state.payload
        half = self.parcel_sim.payload.half_thickness(state.d_surf)[0].cpu().numpy()
        center_z = state.pos[0, :, 2].cpu().numpy()
        self.frames.append(dict(core_z=[min(center_z - half), max(center_z + half)], vertices=verts, corners=corners, pads=pads, skin_z=mesh[:, 2][[mesh[:,2].argmin(),mesh[:,2].argmax()]],
            clearance=min(clearances, default=np.inf), press=self.result.press_N[0, 0].item(),
            rail=robot.rail_load[0].cpu().numpy().copy(), motor=robot.actuators.applied_effort.torch[0,robot.rail_ids].cpu().numpy(),
            capture=self.parcel_state.contacts_A[0,:,1].cpu().numpy().copy(),
            finger_pose=robot.data.body_link_pose_w.torch[0,self.ports[0].pad_ids].cpu().numpy().copy()))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--flex-surface", type=Path, help="Render native flex vertices on their registered finger trajectory.")
    parser.add_argument("--reference", type=Path, help="Original 25 Hz proxy geometry archive for pose verification.")
    add_launcher_args(parser)
    args = parser.parse_args()
    if bool(args.flex_surface) != bool(args.reference):
        parser.error("--flex-surface and --reference must be supplied together")
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = yam_parcel_cfg(1)
    with TemporaryDirectory() as tmp:
        # Keep one authored fixed root after spawning the converted USD.
        with launch_simulation(cfg.sim, args):
            from pxr import Usd
            stage = Usd.Stage.Open(str(YAM_USD))
            stage.RemovePrim("/yam/Physics/root_joint")
            preview = str(Path(tmp) / "yam.usdc")
            stage.GetRootLayer().Export(preview)
            cfg.scene.arm_a.spawn.usd_path = cfg.scene.arm_b.spawn.usd_path = preview
            import pyglet
            pyglet.options.shadow_window = False
            import pyglet.window.xlib as xlib
            xlib._have_utf8 = False
            randomizer = DomainRandomizer("cuda:0", 17)
            episode = CornerSuite(randomizer).select(torch.tensor([4], device="cuda"))
            env = RecordingEnv(cfg, episode=episode)
            env.out, env.encoder, env.frames, env.substeps = args.output, None, [], []
            env.substep_bounds = []
            env.flex = None
            if args.flex_surface:
                import warp as wp
                with np.load(args.flex_surface) as archive:
                    env.flex = {"vertices": archive["vertices"]}
                    env.flex_faces = wp.array(archive["triangles"].ravel(), dtype=wp.int32, device=str(env.device))
                env.reference = np.load(args.reference)["finger_pose"]
            try:
                # Use the lower inner edge for vertical pressing. The port uses
                # this same lever for velocity, Jacobian, reaction and IK.
                env.ports[0].frames.offsets[1:, 2] -= env.scene_definition["pad_box_size_m"][2] / 2
                with use_stage(env.sim.stage):
                    if env.markers is None:
                        env.markers = env._make_markers()
                env.sim.set_camera_view((.47, -.67, .38), (-.09, -.23, .07))
                result, trace = acquisition(env, [("C5", "top", -.036, 0., 4.)], 7.2,
                                            rapid_close=True, preview_target=preview_target)
                np.savez_compressed(env.out / "geometry.npz", **{k: np.stack([f[k] for f in env.frames])
                    for k in env.frames[0]}, substeps=torch.stack(env.substeps).cpu().numpy(),
                    core_bounds_2ms=torch.stack(env.substep_bounds).cpu().numpy(),
                    final_q=env.robots[0].data.joint_pos.torch[0].cpu().numpy())
                (env.out / "trace.json").write_text(json.dumps(trace))
                (env.out / "result.json").write_text(json.dumps(result[0], indent=2))
            finally:
                if env.encoder is not None:
                    env.encoder.stdin.close()
                    code = env.encoder.wait()
                    if code:
                        raise RuntimeError(f"ffmpeg exited with {code}")
                env.close()


if __name__ == "__main__":
    main()
