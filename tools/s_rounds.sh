#!/bin/bash
# Final runs of the plain submission file (python lab06_final.py, no runner or import), two at a time, each in its own
# directory. Usage: nohup bash tools/s_rounds.sh > final/cand/s_rounds.log 2>&1 &
source ~/deep_learning/bin/activate
cd ~/p6/final/cand
run() { (cd "$1" && python -u lab06_final.py > run.log 2>&1; echo "$(date +%T) $1 exit $?"); }
for pair in "S_1 S_2" "S_3 S_4"; do
  echo "$(date +%T) pair start: $pair"
  for d in $pair; do run "$d" & done
  wait
  for d in $pair; do echo "$(date +%T) $d: $(grep -h 'KID x 1000' $d/run.log | tail -1)"; done
done
echo "$(date +%T) all S runs done"
