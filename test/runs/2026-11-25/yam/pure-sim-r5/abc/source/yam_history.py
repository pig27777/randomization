"""Device-side rolling and optional full Try history."""

import warp as wp

wp.set_module_options({"enable_backward": False})


@wp.kernel
def append(obs: wp.array2d[float], action: wp.array2d[float], timestamp: wp.array[float],
           total: wp.array[int], active: wp.array[bool], try_total: wp.array[int],
           frames: wp.array3d[float], actions: wp.array3d[float], times: wp.array2d[float],
           try_frames: wp.array3d[float], try_actions: wp.array3d[float], try_times: wp.array2d[float]):
    env = wp.tid()
    i = total[env] % frames.shape[1]
    j = try_total[env]
    keep = active[env] and j < try_frames.shape[1]
    for k in range(28):
        frames[env, i, k] = obs[env, k]
        if keep:
            try_frames[env, j, k] = obs[env, k]
    for k in range(7):
        actions[env, i, k] = action[env, k]
        if keep:
            try_actions[env, j, k] = action[env, k]
    times[env, i] = timestamp[env]
    if keep:
        try_times[env, j] = timestamp[env]
    if active[env]:
        try_total[env] = j + 1
    total[env] += 1


@wp.kernel
def ordered(total: wp.array[int], rolling: bool, frames: wp.array3d[float], actions: wp.array3d[float],
            times: wp.array2d[float], out_obs: wp.array3d[float], out_action: wp.array3d[float],
            out_time: wp.array2d[float], mask: wp.array2d[bool]):
    env, i = wp.tid()
    length = frames.shape[1]
    count = total[env]
    j = i
    valid = i < count
    if rolling:
        valid = i >= length - wp.min(length, count)
        j = (i + count) % length
    mask[env, i] = valid
    for k in range(28):
        out_obs[env, i, k] = frames[env, j, k] if valid else 0.0
    for k in range(7):
        out_action[env, i, k] = actions[env, j, k] if valid else 0.0
    out_time[env, i] = times[env, j] if valid else 0.0


@wp.kernel
def set_recording(mask: wp.array[bool], start: bool, active: wp.array[bool], count: wp.array[int]):
    env = wp.tid()
    if mask[env]:
        active[env] = start
        if start:
            count[env] = 0


class History:
    def __init__(self, worlds, length, try_capacity, device):
        self.worlds = worlds
        self.device = device
        self.count = wp.zeros(worlds, dtype=int, device=device)
        self.try_count = wp.zeros_like(self.count)
        self.try_active = wp.zeros(worlds, dtype=bool, device=device)
        self.rolling = self._arrays(length)
        # Size one for the default disabled archive, including graph arguments.
        self.archive = self._arrays(max(1, try_capacity))
        self.try_capacity = try_capacity
        self.rolling_output = self._arrays(length, with_mask=True)
        self.archive_output = self._arrays(max(1, try_capacity), with_mask=True)

    def _arrays(self, length, with_mask=False):
        obs = wp.zeros((self.worlds, length, 28), dtype=float, device=self.device)
        action = wp.zeros((self.worlds, length, 7), dtype=float, device=self.device)
        times = wp.zeros((self.worlds, length), dtype=float, device=self.device)
        if with_mask:
            return obs, action, times, wp.zeros(times.shape, dtype=bool, device=self.device)
        return obs, action, times

    def record(self, obs, action, timestamp):
        wp.launch(append, self.worlds, [obs, action, timestamp, self.count, self.try_active,
                                       self.try_count, *self.rolling, *self.archive], device=self.device)

    def start_try(self, mask):
        if not self.try_capacity:
            raise ValueError("Allocate try_capacity policy frames when constructing the interface")
        wp.launch(set_recording, self.worlds, [mask, True, self.try_active, self.try_count], device=self.device)

    def stop_try(self, mask):
        wp.launch(set_recording, self.worlds, [mask, False, self.try_active, self.try_count], device=self.device)

    def read(self, *, full_try=False):
        count = self.try_count if full_try else self.count
        source = self.archive if full_try else self.rolling
        output = self.archive_output if full_try else self.rolling_output
        wp.launch(ordered, source[2].shape, [count, not full_try, *source, *output], device=self.device)
        result = dict(zip(("proprio_history", "action_history", "measurement_timestamp", "valid_mask"), output))
        if full_try:
            result["total_frames"] = self.try_count
        return result
