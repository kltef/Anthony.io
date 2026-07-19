#!/usr/bin/env python3
"""The triple ship-gate for the beat-Fader program (BEAT_FADER.md).

A candidate net is promotable ONLY if it passes all three legs, measured at the deploy
budget with the actual shipped planner:

  1. FADER  — beats the fitted Fader-bot MORE than the incumbent does (the goal metric),
              played at a long game cap so the snowball has time to matter.
  2. ANCHOR — no regression head-to-head vs the incumbent (src/web/gnn_policy.json):
              win-rate >= --anchor-floor (default 0.45, the trainer's own floor).
  3. LEAGUE — no forgetting: vs each scripted opponent (turtle/rush/econ/farmer/human),
              win-rate >= --league-floor (default 0.80).

Prints a PASS/FAIL verdict per leg and exits 0 only on full pass, so it can gate a
promotion step in a shell chain. Run times scale with --games; defaults are a real
measurement (~30-60 min), use --quick for a smoke read.

Usage:
  python tools/gate_fader.py candidate.json                    # full gate vs the shipped net
  python tools/gate_fader.py candidate.json --quick            # fast smoke read
  python tools/gate_fader.py --baseline                        # measure the INCUMBENT's fader
                                                               # number (phase-2 baseline)
"""
import argparse
import json
import os
import re
import subprocess
import sys

SCR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCR)
ANCHOR = os.path.join(ROOT, 'src', 'web', 'gnn_policy.json')
BASELINE_PATH = os.path.join(SCR, 'fader_baseline.json')
LEAGUE = ['script:turtle', 'script:rush', 'script:econ', 'script:farmer', 'script:human']


def arena(a, b, games, budget, gamecap=None, extra_env=None):
    """Run one arena match, return side A's win-rate (parses 'FINAL: A won X/N')."""
    env = os.environ.copy()
    env['DEPTH_RULES'] = '1'   # gate under the SHIPPED game's rules (depth v2, build 13+)
    if gamecap:
        env['GAMECAP'] = str(gamecap)
    if extra_env:
        env.update(extra_env)
    r = subprocess.run(['node', 'tools/selfplay_arena.js', a, b, str(games), str(budget)],
                       capture_output=True, text=True, cwd=ROOT, env=env, timeout=7200)
    m = re.search(r'FINAL: A won (\d+)/(\d+)', r.stdout)
    if not m:
        print(r.stdout[-2000:], file=sys.stderr)
        raise RuntimeError('arena produced no FINAL line (%s vs %s)' % (a, b))
    return int(m.group(1)) / int(m.group(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('candidate', nargs='?', help='candidate net JSON (omit with --baseline)')
    ap.add_argument('--baseline', action='store_true',
                    help='measure the incumbent vs the fader bot and store it as the bar to beat')
    ap.add_argument('--anchor', default=ANCHOR)
    ap.add_argument('--games', type=int, default=40)
    ap.add_argument('--budget', type=int, default=60)
    ap.add_argument('--fader-cap', type=int, default=240,
                    help='GAMECAP for the fader leg — long, so the snowball has time to matter')
    ap.add_argument('--anchor-floor', type=float, default=0.45)
    ap.add_argument('--league-floor', type=float, default=0.80)
    ap.add_argument('--league-games', type=int, default=20)
    ap.add_argument('--quick', action='store_true', help='smoke read: fewer games everywhere')
    args = ap.parse_args()
    if args.quick:
        args.games, args.league_games = 16, 10

    if args.baseline:
        wr = arena(args.anchor, 'script:fader', args.games, args.budget, args.fader_cap)
        json.dump({'anchor_vs_fader': wr, 'games': args.games, 'budget': args.budget,
                   'gamecap': args.fader_cap}, open(BASELINE_PATH, 'w'), indent=2)
        print('BASELINE: incumbent beats fader-bot %.1f%%  -> %s' % (wr * 100, BASELINE_PATH))
        return

    if not args.candidate:
        ap.error('candidate required (or --baseline)')
    cand = args.candidate
    base = json.load(open(BASELINE_PATH))['anchor_vs_fader'] if os.path.exists(BASELINE_PATH) else 0.5
    ok = True

    # Leg 1 — FADER (the goal): beat the bot more than the incumbent does
    wr_f = arena(cand, 'script:fader', args.games, args.budget, args.fader_cap)
    leg = wr_f > base
    ok &= leg
    print('LEG 1 FADER : cand %.1f%% vs bot   (incumbent %.1f%%)   %s'
          % (wr_f * 100, base * 100, 'PASS' if leg else 'FAIL'))

    # Leg 2 — ANCHOR: no regression vs the shipped champion
    wr_a = arena(cand, args.anchor, args.games, args.budget)
    leg = wr_a >= args.anchor_floor
    ok &= leg
    print('LEG 2 ANCHOR: cand %.1f%% vs shipped net   (floor %.0f%%)   %s'
          % (wr_a * 100, args.anchor_floor * 100, 'PASS' if leg else 'FAIL'))

    # Leg 3 — LEAGUE: no forgetting vs the scripted exploiters
    for opp in LEAGUE:
        wr = arena(cand, opp, args.league_games, args.budget)
        leg = wr >= args.league_floor
        ok &= leg
        print('LEG 3 LEAGUE: cand %.1f%% vs %-14s (floor %.0f%%)   %s'
              % (wr * 100, opp, args.league_floor * 100, 'PASS' if leg else 'FAIL'))

    print('\nVERDICT: %s' % ('PROMOTABLE — all legs passed' if ok else 'NOT promotable'))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
