#!/usr/bin/env bash
# One-time start (and restart after the NCC server ends): bash ~/p6/bridge/start.sh
cd ~/p6 || exit 1
git config user.name "p6-ncc-driver"
git config user.email "p6-driver@ncc.invalid"
git config pull.rebase true
[ -d ~/data ] && [ ! -e data ] && ln -s ~/data data
mkdir -p logs runs exps final
chmod +x bridge/*.sh

AGENT=$(head -n1 ~/.p6_agent 2>/dev/null)
[ -x "$AGENT" ] || { echo "~/.p6_agent must hold the absolute path of the agent CLI (ask the reviewer)"; exit 1; }
nvidia-smi -L || echo "WARNING: no GPU visible on $(hostname) - start the JupyterHub server with a GPU"

if command -v tmux >/dev/null; then
  tmux kill-session -t p6 2>/dev/null
  tmux new-session -d -s p6 "bash bridge/driver.sh 2>&1 | tee -a logs/driver.log"
  echo "Driver running in tmux session 'p6' on $(hostname). Watch it with:  tmux attach -t p6   (detach: Ctrl-b then d)"
else
  nohup setsid bash bridge/driver.sh >> logs/driver.log 2>&1 < /dev/null &
  echo "Driver running in the background on $(hostname). Watch it with:  tail -f ~/p6/logs/driver.log"
fi
