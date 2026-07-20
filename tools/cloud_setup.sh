#!/usr/bin/env bash
# One-shot setup for running the beat-Fader pipeline on a Linux cloud box (e.g. the 24-vCPU
# EPYC host of a TPU v5e VM — the TPU itself is unused; the CPU runs the self-play arena).
# Usage, from a fresh VM:
#   git clone https://github.com/kltef/Anthony.io.git && cd Anthony.io
#   bash tools/cloud_setup.sh
#   python3 tools/beat_fader.py phase3 --hours 24 --device cpu     # detached; survives SSH drop
#   python3 tools/beat_fader.py status
#   python3 tools/beat_fader.py gate tools/counter_gnn.json        # when done
# Pull the result back (from your PC):  scp VM:.../tools/counter_gnn.json tools/
set -e

echo "== node (>=16 needed: the arena uses modern JS) =="
if ! command -v node >/dev/null || [ "$(node -e 'console.log(+process.versions.node.split(".")[0])')" -lt 16 ]; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
  sudo apt-get install -y nodejs
fi
node --version

echo "== python deps (CPU torch is all we need — the net is 8.8k params) =="
sudo apt-get install -y python3-pip >/dev/null 2>&1 || true
pip3 install --quiet numpy torch --index-url https://download.pytorch.org/whl/cpu

echo "== sanity: arena extracts the shipped planner and plays 2 depth-rules games =="
DEPTH_RULES=1 GAMECAP=60 node tools/selfplay_arena.js src/web/gnn_policy.json script:rush 2 40 | tail -2

echo
echo "ready. suggested run:  python3 tools/beat_fader.py phase3 --hours 24 --device cpu"
