"""dev/summary.py NAME[:VARIANT:STEPS] ...: each run's final score and its last report line from dev/NAME.log."""
import json
import os
import re
import sys

D = os.path.dirname(os.path.abspath(__file__))
KEYS = ("dropped", "hangs", "a_fastened", "a_tight", "a_gap", "a_edge_gap", "a_clamp", "a_turn",
        "b_fastened", "b_tight", "b_gap", "b_edge_gap", "b_clamp", "b_turn", "cross_contact_peak_n")
for arg in sys.argv[1:]:
    name, *rest = arg.split(":")
    text = open(f"{D}/{name}.log", errors="replace").read()
    score = re.findall(r"^\s+pair_task_score: ([0-9.]+)", text, re.M)
    reps = re.findall(r"^\[rep\] step (\d+) (\{.*\})$", text, re.M)
    rep = json.loads(reps[-1][1]) if reps else {}
    events = re.findall(r"^\[clp\] event step (\d+): (.*)$", text, re.M)
    print(f"{name:14s} variant={rest[0] if rest else '':11s} score={score[-1] if score else 'NONE':5s} "
          f"last_rep_step={reps[-1][0] if reps else '-'} "
          + " ".join(f"{k}={rep.get(k)}" for k in KEYS))
    for step, text_ in events:
        print(f"    step {step}: {text_}")
