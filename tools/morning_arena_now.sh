#!/usr/bin/env bash
# Morning eval: wait for the overnight loop to finish (log idle >3 min), then run the two
# real-planner arenas at full parallelism on the now-idle machine and aggregate.
set -e
cp tools/gnn_policy_new.json tools/_morning_final.json
run_match () { # $1=polA $2=polB $3=tag
  TOTAL=280; PROCS=14; PER=$(( (TOTAL + PROCS - 1) / PROCS ))
  TMP=$(mktemp -d)
  for i in $(seq 1 $PROCS); do
    ( node tools/selfplay_arena.js "$1" "$2" $PER 60 > "$TMP/a$i.log" 2>&1 ) &
  done
  wait
  A=0; N=0
  for i in $(seq 1 $PROCS); do
    line=$(grep '^FINAL' "$TMP/a$i.log" || true)
    x=$(echo "$line" | sed -E 's/FINAL: A won ([0-9]+)\/([0-9]+).*/\1/')
    n=$(echo "$line" | sed -E 's/FINAL: A won ([0-9]+)\/([0-9]+).*/\2/')
    if [ -n "$x" ] && [ -n "$n" ]; then A=$((A+x)); N=$((N+n)); fi
  done
  rm -rf "$TMP"
  python -c "
import math
a,n=$A,$N; wr=a/n if n else 0; se=math.sqrt(wr*(1-wr)/n)*196 if n else 0
print(f'$3: A won {a}/{n} = {wr*100:.1f}%  (95% CI +/-{se:.1f} pts)')"
}
echo "training finished; arenas starting $(date +%T)"
run_match tools/_morning_final.json src/web/gnn_policy.json  "NEW-vs-SHIPPED-GNN"
run_match tools/_morning_final.json src/web/rl_policy.json   "NEW-vs-FLAT-MLP"
echo "arenas done $(date +%T)"
