"""Open the prepared physical parcel in the local MuJoCo Studio web UI."""
import argparse
import json
from pathlib import Path
import threading
from unittest.mock import patch

import mujoco
import numpy as np
from mujoco.studio import app, launch_web, messages, plugin, sim, viewer_handle, viewer_protocol
from mujoco.studio.web import web_server

import physical_parcel_yam as backend


class RestoreInitialState(plugin.SimPlugin):
    def __init__(self, model, data):
        self.model = model
        self.signature = mujoco.mjtState.mjSTATE_INTEGRATION
        self.state = np.empty(mujoco.mj_stateSize(model, self.signature))
        mujoco.mj_getState(model, data, self.state, self.signature)

    @messages.handler(priority=messages.Priority.INTERNAL - 1)
    def restore(self, event: messages.ResetEvent):
        if self.handle.model is self.model:
            mujoco.mj_setState(self.model, self.handle.data, self.state, self.signature)
            mujoco.mj_forward(self.model, self.handle.data)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8088)
    args = parser.parse_args()
    robot = backend.scene(backend.Config(**json.loads(args.config.read_text())))
    model, data = robot.sim.m, robot.sim.d
    model.stat.center[:] = [-.20, .02, .22]
    model.stat.extent = .8
    callback_count = 0

    def passive(m, d):
        nonlocal callback_count
        if m is model:
            robot.sim.parcel.apply_internal(m, d, robot.sim.film_dofs)
            callback_count += 1

    config = viewer_protocol.ViewerConfig(gfx="webgl", http_port=args.port,
        title="双膜片袋 · 同形凸网格接触", open_browser=False)
    plugins = [app.App(), *app.SIM_PLUGINS, RestoreInitialState(model, data)]
    bind = web_server.bind_public_socket
    ready = threading.Event()

    def bind_local(host, port):
        sock = bind("127.0.0.1", port)
        ready.set()
        return sock

    try:
        with patch.object(web_server, "bind_public_socket", bind_local):
            with launch_web.launch(config, plugins=plugins) as handle:
                # Studio first compiles an empty viewer model. Register the
                # callback after that compilation has completed.
                if not ready.wait(10):
                    raise RuntimeError("Studio did not start its local server")
                mujoco.set_mjcb_passive(passive)
                mujoco.mj_forward(model, data)
                if callback_count == 0:
                    raise RuntimeError("Studio did not invoke the parcel's custom passive forces")
                handle.set_model(model, data=data)
                handle.dispatch(messages.RequestPauseEvent(sim.PauseState.NORMAL_PAUSED))
                print(json.dumps(dict(url=f"http://localhost:{args.port}", paused=True,
                    custom_passive_forces=True, config=str(args.config))), flush=True)
                viewer_handle.run_sim_loop(handle)
    finally:
        mujoco.set_mjcb_passive(None)


if __name__ == "__main__":
    main()
