#!/usr/bin/env bash
# Practical 6 bridge driver (runs on the NCC GPU node).
# Waits for the reviewer's next instruction in bridge/inbox/NNN.md (arrives via git pull),
# runs one headless Claude Code turn with it, saves the reply as bridge/outbox/NNN.md,
# and commits + pushes everything so the reviewer can read it. Repeats forever.
cd "$(dirname "$0")/.." || exit 1
B=bridge
mkdir -p "$B/inbox" "$B/outbox" logs
source ~/deep_learning/bin/activate 2>/dev/null
export PATH="$HOME/.local/bin:$HOME/.claude/local:$PATH"
TOPIC=$(cat "$B/ntfy_topic" 2>/dev/null)

log()    { echo "$(date '+%F %T') $*"; }
notify() { [ -n "$TOPIC" ] && curl -s -m 10 -H "Title: P6" -d "$1" "https://ntfy.sh/$TOPIC" >/dev/null 2>&1; true; }
field()  { python3 -c "import json;print(json.load(open('$B/status.json')).get('$1',''))" 2>/dev/null; }
status() { printf '{"turn": %d, "state": "%s", "updated": "%s", "host": "%s"}\n' \
             "$1" "$2" "$(date -u +%FT%TZ)" "$(hostname)" > "$B/status.json"; }

sync() {  # commit local work, fetch the reviewer's files, push. Serialised with a lock.
  (
    flock -w 300 9 || exit 0
    git add -A >/dev/null 2>&1
    git commit -qm "${1:-sync}" >/dev/null 2>&1
    if ! git pull -q --rebase >/dev/null 2>&1; then git rebase --abort >/dev/null 2>&1; fi
    git push -q >/dev/null 2>&1 || log "push failed"
  ) 9>"$B/.lock"
}

N=$(field turn); N=${N:-0}
if [ "$(field state)" = "working" ]; then   # the server died in the middle of a turn
  printf 'Turn %d was interrupted at %s (driver restarted on %s). Runs started in that turn may be incomplete; check logs/ and runs/index.jsonl.\n' \
    "$N" "$(date -u +%FT%TZ)" "$(hostname)" >> "$B/outbox/$(printf %03d "$N").md"
  status "$N" waiting; sync "turn $N interrupted"; notify "P6: turn $N was interrupted, waiting for reviewer"
fi
status "$N" waiting; sync "driver started on $(hostname)"
log "driver started, last finished turn $N"; notify "P6 driver running on $(hostname) (last turn $N)"

while true; do
  sync "poll"
  NEXT=$((N + 1)); ID=$(printf %03d "$NEXT"); PROMPT="$B/inbox/$ID.md"
  if [ ! -f "$PROMPT" ]; then sleep 60; continue; fi

  log "turn $NEXT starting"
  status "$NEXT" working; sync "turn $NEXT start"
  SID=$(cat "$B/session_id" 2>/dev/null)
  ARGS=(-p "$(cat "$PROMPT")" --output-format json --dangerously-skip-permissions)
  [ -n "$SID" ] && ARGS+=(--resume "$SID")
  timeout 8h claude "${ARGS[@]}" > "$B/outbox/$ID.raw.json" 2> "$B/outbox/$ID.stderr.log" &
  CP=$!

  # Heartbeat: push progress every ~5 min; an abort_NNN file from the reviewer stops the turn.
  i=0
  while kill -0 "$CP" 2>/dev/null; do
    sleep 30; i=$((i + 1))
    if [ $((i % 10)) -eq 0 ]; then
      status "$NEXT" working; sync "heartbeat turn $NEXT"
      if [ -f "$B/abort_$ID" ]; then log "abort requested"; kill "$CP"; pkill -P "$CP" 2>/dev/null; fi
    fi
  done
  wait "$CP"; RC=$?

  python3 - "$B" "$ID" "$RC" <<'EOF'
import json, sys
b, i, rc = sys.argv[1], sys.argv[2], sys.argv[3]
raw = open(f"{b}/outbox/{i}.raw.json").read().strip()
err = open(f"{b}/outbox/{i}.stderr.log").read().strip()
try:
    d = json.loads(raw.splitlines()[-1]) if raw else {}
except Exception:
    d = {}
text = d.get("result") or ""
meta = (f"\n\n---\n[driver] exit={rc} is_error={d.get('is_error')} turns={d.get('num_turns')} "
        f"minutes={round((d.get('duration_ms') or 0)/60000, 1)}")
if not text:
    text = "(no result text)\nRAW TAIL:\n" + raw[-2000:] + "\nSTDERR TAIL:\n" + err[-2000:]
    open(f"{b}/outbox/{i}.failed", "w").write("1")
with open(f"{b}/outbox/{i}.md", "a") as f:
    f.write(text + meta + "\n")
if d.get("session_id"):
    open(f"{b}/session_id", "w").write(d["session_id"])
EOF
  if [ -f "$B/outbox/$ID.failed" ] && [ -n "$SID" ]; then
    log "turn failed while resuming; the next turn will start a fresh session"; rm -f "$B/session_id"
  fi
  N=$NEXT
  status "$N" waiting; sync "turn $N done"
  notify "P6 turn $N done: $(head -c 150 "$B/outbox/$ID.md" | tr '\n' ' ')"
  log "turn $N done (exit $RC)"
done
