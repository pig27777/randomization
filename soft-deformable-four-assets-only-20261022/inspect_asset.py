"""Compile a classified deformable asset; optionally step or open a paused viewer."""
import argparse
import json
import time
from load_assets import load_asset

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset',choices=['pe','kraft','paperboard','legacy'],default='pe')
    parser.add_argument('--steps',type=int,default=0,help='Owner steps, each 0.01 s at the supplied baselines')
    parser.add_argument('--viewer',action='store_true')
    args=parser.parse_args()
    if args.steps<0: parser.error('steps must be nonnegative')
    scene=load_asset(args.asset)
    try:
        import mujoco
        import numpy as np
        for _ in range(args.steps): scene.step()
        if not (np.isfinite(scene.data.qpos).all() and np.isfinite(scene.data.qvel).all()):
            raise RuntimeError('Nonfinite state')
        report=dict(asset=args.asset,compiled=True,mujoco=mujoco.__version__,owner_steps=args.steps,
                    time_s=float(scene.data.time),dofs=int(scene.model.nv),
                    warnings=scene.data.warning.number.tolist(),physical_calibration=False)
        print(json.dumps(report),flush=True)
        if not args.viewer: return
        import mujoco.viewer
        paused=[True]
        def key_callback(key):
            if key==32: paused[0]=not paused[0]
        with mujoco.viewer.launch_passive(scene.model,scene.data,key_callback=key_callback) as viewer:
            viewer.opt.geomgroup[5]=0
            viewer.cam.lookat[:]=scene.model.stat.center
            viewer.cam.distance=.65
            viewer.cam.azimuth=125
            viewer.cam.elevation=-30
            print('Starts paused. SPACE toggles physics. Close the window to exit.',flush=True)
            while viewer.is_running():
                if not paused[0]: scene.step()
                viewer.sync()
                time.sleep(.016)
    finally:
        scene.close()

if __name__=='__main__': main()
