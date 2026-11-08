"""Write the round-2 behaviour job list (F3, F4, F5) for xargs; one loose_protocols.py call per line."""
import json, math
SET = '{"edge": 0.035, "radius": 0.003}'
F_CLS = {"box_loose": 20, "bendable": 15, "compressible": 10}
jobs = []
for c, F in F_CLS.items():                                    # F3: v1.1 side grasp, 8 s squeeze
    for tag, (u, v, psi) in {"centre": (0, 0, math.pi / 2), "empty_end": (0.42, 0, math.pi / 2),
                             "corner": (0.42, 0.42, math.pi / 4)}.items():
        for seed in range(3):
            jobs.append(f"eval --bag flex --content {c} --set '{SET}' --params '[{u},{v},{psi:.6f},{F},0.1,2]' "
                        f"--seed {seed} --tag 'F3 {tag}' --out f3.jsonl")
for t in ("r2", "r3"):                                        # F4: bagged tight-fit box
    jobs.append(f"{t} --bag flex --content box_fit --set '{SET}' --tag 'F4 bag' --out f4.jsonl")
for pos, (u, v) in {"middle": (0, 0), "empty": (0.4, 0)}.items():   # F5: v2 top grasp, hard box
    for w in (0.03, 0.06, 0.09):
        for d in (0.002, 0.006, 0.012):
            for F in (10, 30):
                for seed in range(3):
                    jobs.append(f"eval --bag flex --content box_loose --set '{SET}' "
                                f"--params '[{u},{v},{math.pi / 2:.6f},{w},{d},{F},0.1,2]' --seed {seed} "
                                f"--tag 'F5 {pos}' --out f5.jsonl")
SC = '{"edge": 0.035, "radius": 0.003, "selfcollide": "auto"}'
for w in (0.03, 0.06, 0.09):                                  # F5: flex self-collision on, empty area
    for d in (0.002, 0.006, 0.012):
        jobs.append(f"eval --bag flex --content box_loose --set '{SC}' --params '[0.4,0,{math.pi / 2:.6f},{w},{d},10,0.1,2]' "
                    f"--seed 0 --tag 'F5 empty selfcollide' --out f5.jsonl")
# slow ones first (large bags of the soft anchors)
jobs.sort(key=lambda j: 0 if "compressible" in j else 1 if "bendable" in j else 2)
open("jobs.txt", "w").write("\n".join(jobs) + "\n")
print(len(jobs), "jobs")
