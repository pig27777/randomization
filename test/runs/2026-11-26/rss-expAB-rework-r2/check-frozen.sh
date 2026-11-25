#!/usr/bin/env bash
set -euo pipefail
run="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$run/../../../.."
check() {
    printf '\n$ git diff --exit-code'
    printf ' %q' "$@"
    printf '\n'
    git diff --exit-code "$@"
    printf 'exit 0\n'
}
date -Iseconds
git rev-parse HEAD
check "$(cat "$run/frozen-head.txt")" -- configs isaaclab \
    src/low_order_parcel/lumped_contact_proxy.py src/low_order_parcel/unified_payload.py \
    src/isaac_yam_parcel/yam_load_port.py tests/policy_v3 test/policy_v3 src/policy_v3 \
    test/runs/2026-11-23/isaaclab/route-b-final-gates-r1/plan-b-step1-summary.json
check fix4.1-route-b-base -- src/low_order_parcel/lumped_contact_proxy.py src/low_order_parcel/unified_payload.py
check 736db9b2 -- src/isaac_yam_parcel/yam_load_port.py
check 2229e650 -- configs experience bag_sim src/low_order_parcel src/isaac_yam_parcel \
    test/bag_model/bag_mvp.py test/bag_model/pe_physics.py test/bag_model/pe_physics_protocol.py
printf '\n$ sha256sum --check --quiet frozen-before.sha256\n'
sha256sum --check --quiet "$run/frozen-before.sha256"
printf 'Tracked and untracked baseline files verified: %s\n' "$(wc -l < "$run/frozen-before.sha256")"
rg ' (tests/policy_v3/|configs/route_b_)' "$run/frozen-before.sha256"
