#!/bin/bash
# Usage: bash logrun.sh exps/exp_XXX.py [runner args...]  -> logs/<exp>_s<seed>.log with timestamps
source ~/deep_learning/bin/activate
cd ~/p6
exp=$1; shift
name=$(basename $exp .py)
seed=3801; prev=""
for a in "$@"; do [ "$prev" = "--seed" ] && seed=$a; prev=$a; done
log=logs/${name}_s${seed}.log
nohup bash -c "python -u runner.py $exp $* 2>&1 | while IFS= read -r l; do echo \"\$(date +%T) \$l\"; done" > $log 2>&1 &
echo "started $name -> $log"
