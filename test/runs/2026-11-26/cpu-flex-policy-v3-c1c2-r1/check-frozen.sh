#!/usr/bin/env bash
set -euo pipefail
run="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$run/../../../.."
bash test/runs/2026-11-26/rss-exp-ab-r1/check-frozen.sh
git show 94d44457:test/bag_model/rss_exp_ab.py | cmp - "$run/source/rss_exp_ab_94d44457.py"
printf '%s\n' 'Archived Exp A/B adapter matches 94d44457 byte-for-byte'
sha256sum "$run"/source/rss_exp_ab_94d44457.py test/bag_model/rss_cpu_flex_c1c2*.py
