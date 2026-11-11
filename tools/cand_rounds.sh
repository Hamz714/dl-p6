#!/bin/bash
# Final-selection seed runs: rounds of at most two concurrent candidates; each runs once in its own directory.
# Usage: nohup bash tools/cand_rounds.sh > final/cand/rounds.log 2>&1 &
source ~/deep_learning/bin/activate
cd ~/p6/final/cand
run() { (cd "$1" && python -u ~/p6/tools/run_seed.py lab06_final.py "$2" > run.log 2>&1; echo "$(date +%T) $1 exit $?"); }
for round in "B_3801:3801 A_3802:3802" "B_3802:3802 A_3803:3803" "B_3803:3803"; do
  echo "$(date +%T) round start: $round"
  for job in $round; do run "${job%%:*}" "${job##*:}" & done
  wait
  for job in $round; do d=${job%%:*}; echo "$(date +%T) $d: $(grep -h 'KID x 1000' $d/run.log | tail -1)"; done
done
echo "$(date +%T) all rounds done"
