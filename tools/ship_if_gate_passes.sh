#!/usr/bin/env bash
# Watch a running gate_fader.py log; the instant it prints a PROMOTABLE verdict, ship the counter
# net OTA (copy -> bump -> push). On NOT promotable, do nothing and report. Idempotent: exits after
# one decision. Usage: bash tools/ship_if_gate_passes.sh <gate_output_log>
set -e
LOG="$1"
cd "$(dirname "$0")/.."

echo "watching $LOG for a verdict..."
until grep -qE "^VERDICT:" "$LOG" 2>/dev/null; do sleep 20; done

echo "==== gate result ===="; cat "$LOG"
if ! grep -q "PROMOTABLE" "$LOG"; then
  echo ">> NOT promotable — nothing shipped. v2.2 stays champion."
  exit 0
fi

echo ">> PROMOTABLE — shipping the counter net OTA."
cp tools/counter_gnn.json src/web/gnn_policy.json
py build/bump_web.py "Smarter Nightmare: the AI was trained specifically to beat the farm-and-snowball strategy under the new depth rules — it out-territories a hoarder instead of trading stacks"
git add src/web/gnn_policy.json web-version.json
git commit -m "Ship the trained counter-net — first net to beat v2.2 (retire the champion)

67-round depth-rules best-response run against the fitted Fader-bot. Passes the full
triple gate at high sample: beats shipped v2.2 head-to-head, beats the Fader-bot more
than v2.2 does, no league regression. Same gnn-v1 arch (10-feature, embed 24), so index.html
dispatches on node_features unchanged and it drops straight in.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push origin main
git branch -f release main
git push origin release
echo ">> shipped. live web: $(git show origin/release:src/web/index.html | grep -o 'const WEB_VERSION = [0-9]*')"
