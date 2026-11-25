#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.."
core=(src/low_order_parcel/lumped_contact_proxy.py src/low_order_parcel/unified_payload.py)
port=(src/isaac_yam_parcel/yam_load_port.py src/isaac_yam_parcel/joint_current_filter.py)
check() {
    local revision=$1
    shift
    git diff --exit-code "$revision" -- "$@"
    printf '%s path-level git diff: exit 0\n' "$revision"
}
date -Iseconds
git rev-parse HEAD
check fix4.1-route-b-base "${core[@]}"
check 2ef4a394 "${core[@]}" configs/stage3_dr_and_corners.yaml
check 736db9b2 "${port[@]}"
check 2229e650 configs experience bag_sim src/low_order_parcel src/isaac_yam_parcel \
    test/bag_model/bag_mvp.py test/bag_model/pe_physics.py test/bag_model/pe_physics_protocol.py
sha256sum "${core[@]}" "${port[@]}" configs/route_b_* test/bag_model/{bag_mvp,pe_physics,pe_physics_protocol,rss_exp_ab,rss_exp_ab_analysis}.py
