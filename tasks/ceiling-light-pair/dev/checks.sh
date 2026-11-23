#!/bin/bash
# dev/checks.sh [NAME:VARIANT:STEPS ...]: the scorer checks, one yam_pair run each, in turn (dev/run.sh), then a
# table of the end scores in dev/checks.txt. With no arguments, all of them.
D=$(cd "$(dirname "$0")/.." && pwd)
cd "$D" || exit 1
RUNS=("$@")
[ ${#RUNS[@]} -eq 0 ] && RUNS=(c_nothing:nothing:300 c_alone:alone:1100 c_drop:drop:800 c_unseated:unseated:3200
                               c_one_tight:one_tight:3200 c_one_end:one_end:3200 c_still_held:still_held:3200
                               c_slip:slip:3200)
for r in "${RUNS[@]}"; do
  IFS=: read -r name variant steps <<<"$r"
  rm -rf dev/logs/$name
  TAILN=3 dev/run.sh "$name" "$steps" "$variant" > /dev/null
done
python3 dev/summary.py "${RUNS[@]}" | tee -a dev/checks.txt
