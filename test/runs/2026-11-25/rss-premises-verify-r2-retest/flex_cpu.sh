#!/usr/bin/env bash
set -euo pipefail
run_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$run_dir/../../../.."
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out="${1:-$run_dir/flex-cpu}"
mkdir -- "$out"
exec flock /tmp/claude-gpu.lock bash -s -- "$out" <<'RUN'
set -u
out="$1"
python=.venv-physical-parcel/bin/python
runner=test/bag_model/bag_mvp.py
"$python" -B -c 'import mujoco; from importlib.metadata import version; print("MuJoCo", mujoco.__version__, "MJWarp", version("mujoco-warp"))'
failed=0
for side in left center right; do
    name="try-$side"
    date -Iseconds
    printf '%s\n' "$name"
    "$python" -B "$runner" "$name" "$out/$name" --dt .00005 > "$out/$name.log" 2>&1 || failed=1
done
date -Iseconds
printf '%s\n' film
"$python" -B "$runner" film "$out/film" --variant near-seam --dt .00005 > "$out/film.log" 2>&1 || failed=1
date -Iseconds
printf '%s\n' through
"$python" -B "$runner" through "$out/through" --variant exposed-underface --edge .015 --dt .00005 > "$out/through.log" 2>&1 || failed=1
exit "$failed"
RUN
