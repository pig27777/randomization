"""Replay the existing 2 ms environment tick with live sensor queue cursors."""

import torch
from torch.utils._pytree import tree_map


class TickGraph:
    def __init__(self, env, command, active, lift):
        self.command = tree_map(torch.Tensor.clone, command)
        self.active = active.clone()
        self.lift = lift.clone()
        self.graph = torch.cuda.CUDAGraph()
        yam, rgb = env.yam, env.rgb_sensor
        # The frozen filters use modulo indexing. A one-element device tensor
        # preserves that indexing while avoiding a host scalar in the graph.
        for owner, name in ((yam, "_obs_index"), (rgb, "_index"), (rgb, "_frame_clock")):
            setattr(owner, name, torch.tensor([getattr(owner, name)], dtype=torch.long, device=env.device))
        for generator in (yam.generator, rgb.generator):
            self.graph.register_generator_state(generator)
        carries = (
            (env, ("state", "command", "offset", "offset_velocity", "load", "pads")),
            (yam, ("_obs_index", "_obs_age", "_filtered", "_opening")),
            (rgb, ("_index", "_frame_clock", "_age", "_last", "_lift_origin", "_lifting",
                   "_gamma", "_previous_wrist", "_wrist_seen")),
        )
        stable = [(obj, key, getattr(obj, key)) for obj, keys in carries for key in keys]
        current = torch.cuda.current_stream(env.device)
        stream = torch.cuda.Stream(device=env.device)
        stream.wait_stream(current)
        with torch.cuda.graph(self.graph, stream=stream):
            env._tick_eager(self.command, self.active, self.lift)
            for obj, key, storage in stable:
                tree_map(torch.Tensor.copy_, storage, getattr(obj, key))
        current.wait_stream(stream)
        for obj, key, storage in stable:
            setattr(obj, key, storage)

    def step(self, command, active, lift):
        tree_map(torch.Tensor.copy_, self.command, command)
        self.active.copy_(active)
        self.lift.copy_(lift)
        self.graph.replay()
