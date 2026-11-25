#!/usr/bin/env bash
set -euo pipefail
run_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$run_dir/../../../.."
contact=src/low_order_parcel/lumped_contact_proxy.py
payload=src/low_order_parcel/unified_payload.py
port=src/isaac_yam_parcel/yam_load_port.py
git diff --exit-code 2ef4a3943183c527946c3ebc0ba171ea6c163094 -- "$contact" "$payload"
printf '%s\n' "2ef4a394 contact and payload: exit 0"
git diff --exit-code 736db9b2 -- "$port"
printf '%s\n' "736db9b2 actuator/load port: exit 0"
git diff --exit-code 917a7871 -- configs "$contact" "$payload" "$port"
printf '%s\n' "Starting revision 917a7871, configs and frozen source: exit 0"
git diff --exit-code 72b558fe -- configs "$contact" "$payload" "$port"
printf '%s\n' "Fetched revision 72b558fe, configs and frozen source: exit 0"
sha256sum "$contact" "$payload" "$port" configs/route_b_*
printf '\n%s\n' "Installed IsaacLab release commit:"
git -C /home/ubuntu/workspace/czx/system0/sim/IsaacLab rev-parse HEAD
