#!/bin/bash
# C2 (REPORT round 1): MuJoCo loose-bag representations, CPU. R2, R3 and four task-protocol grasps per content.
# Usage: run_c2.sh plates|flex NAME SET_JSON [content ...]   (default contents: box_loose bendable compressible)
# NAME labels the output (c2_NAME.jsonl); SET_JSON overrides LooseBag fields.
cd "$(dirname "$0")/../.."
P=/home/ubuntu/Dev/Envs/mjwarp/bin/python
bag=$1; name=$2; set=$3; shift 3
contents=${@:-box_loose bendable compressible}
for c in $contents; do
  case $c in box*) F=20;; bendable) F=15;; compressible) F=10;; esac
  jobs=("r2" "r3"
        "eval --params [0,0,1.5708,$F,0.1,2] --seed 0 --tag centre"
        "eval --params [0,0,1.5708,$F,0.1,2] --seed 1 --tag centre"
        "eval --params [0.42,0,1.5708,$F,0.1,2] --seed 0 --tag empty_end"
        "eval --params [0.42,0.42,0.7854,$F,0.1,2] --seed 0 --tag corner")
  for j in "${jobs[@]}"; do
    $P -B loose_protocols.py $j --bag $bag --content $c --set "$set" --tag "$name ${j##*--tag }" --out c2_$name.jsonl >> outputs/round1/logs/c2_${name}_$c.log 2>&1 &
  done
done
wait
