#!/usr/bin/env bash
set -euo pipefail
run_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$run_dir/../../../.."
parcel=(src/low_order_parcel/lumped_contact_proxy.py src/low_order_parcel/unified_payload.py)
port=src/isaac_yam_parcel/yam_load_port.py
current=src/isaac_yam_parcel/joint_current_filter.py
fix41=(configs/stage3_dr_and_corners.yaml src/low_order_parcel/domain_randomizer.py
       src/low_order_parcel/macro_parcel_sim.py)
workcell=(configs experience/yam_platform src/isaac_yam_parcel/yam_parcel_base_env.py
          src/isaac_yam_parcel/yam_bin_env.py src/isaac_yam_parcel/joint_bc_ppo.py)
git diff --exit-code fix4.1-route-b-base -- "${parcel[@]}" "${fix41[@]}"
printf '%s\n' 'fix4.1-route-b-base: exit 0'
git diff --exit-code 2ef4a394 -- "${parcel[@]}" "${fix41[@]}"
printf '%s\n' '2ef4a394: exit 0'
git diff --exit-code 736db9b2 -- "$port" "$current"
printf '%s\n' '736db9b2: exit 0'
git diff --exit-code 2229e650 -- "${workcell[@]}" "${parcel[@]}" "$port"
printf '%s\n' '2229e650: exit 0'
git diff --exit-code b5d89681 -- configs experience/yam_platform src/low_order_parcel src/isaac_yam_parcel
printf '%s\n' 'b5d89681 (本轮起点，冻结底座和既有历史差异): exit 0'
git diff --exit-code 736db9b2 -- src/low_order_parcel/neck_damping.py
printf '%s\n' '736db9b2 neck damping (supersedes Fix 4.1): exit 0'
git diff --exit-code 98692aef -- src/low_order_parcel/dual_arm_parcel_sim.py
printf '%s\n' '98692aef dual arm (supersedes 736db9b2): exit 0'
git diff --exit-code 2229e650 -- bag_sim/physical_parcel.py bag_sim/physical_parcel_gpu.py
printf '%s\n' '2229e650 PE Flex and bending: exit 0'
git diff --exit-code caa5228b -- configs experience/yam_platform bag_sim/physical_parcel.py \
  bag_sim/physical_parcel_gpu.py src/low_order_parcel src/isaac_yam_parcel
printf '%s\n' 'caa5228b mission start, all frozen source and configurations: exit 0'
