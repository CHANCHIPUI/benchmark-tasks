"""dev/results.py RUN: a run's score, events, end state and per-arm peaks (Markdown), from dev/RUN.log."""
import json
import os
import re
import sys

D = os.path.dirname(os.path.abspath(__file__))
run = sys.argv[1]
text = open(f"{D}/{run}.log", errors="replace").read()
score = re.findall(r"^\s+pair_task_score: ([0-9.]+)", text, re.M)
reps = re.findall(r"^\[rep\] step (\d+) (\{.*\})$", text, re.M)
rep = json.loads(reps[-1][1]) if reps else {}
print(f"**{run}**: pair_task_score {score[-1] if score else 'none'} (last report at step {reps[-1][0] if reps else '-'})\n")
for step, ev in re.findall(r"^\[clp\] event step (\d+): (.*)$", text, re.M):
    print(f"- step {step} ({int(step) / 20:.1f} s): {ev}")
print()
for s in "ab":
    print(f"- {s.upper()}: fastened {rep.get(f'{s}_fastened')}, seated {rep.get(f'{s}_seated')}, flush {rep.get(f'{s}_flush')}, "
          f"clamp {rep.get(f'{s}_clamp', 0):.0f} N, head gap {rep.get(f'{s}_gap', 0) * 1000:.2f} mm, edge gap "
          f"{rep.get(f'{s}_edge_gap', 0) * 1000:.2f} mm, driver turned {rep.get(f'{s}_turn', 0):.0f} deg in the hand while "
          f"driving, peak push {rep.get(f'peak_push_{s}', 0):.1f} N")
print(f"- dropped {rep.get('dropped')}, hangs {rep.get('hangs')}, cross-station contact peak "
      f"{rep.get('cross_contact_peak_n', 0):.2f} N\n")
arms = (("a_left", "A outer (left)"), ("a_right", "A inner (right)"), ("b_left", "B inner (left)"), ("b_right", "B outer (right)"))
print("| Arm | " + " | ".join(f"J{i} N m" for i in range(1, 7)) + " | Housing contact, N |")
print("|---|" + "---|" * 7)
print("| limit | 27 | 27 | 27 | 7 | 7 | 7 | |")
for key, name in arms:
    js = [rep.get(f"peak_torque_{key}_j{i}", 0.0) for i in range(1, 7)]
    print(f"| {name} | " + " | ".join(f"{v:.1f}" for v in js) + f" | {rep.get(f'peak_housing_{key}', 0.0):.1f} |")
