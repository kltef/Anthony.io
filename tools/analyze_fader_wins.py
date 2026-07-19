#!/usr/bin/env python3
"""Isolate HOW the play-tester beats the AI: contrast his WINS vs his LOSSES.

The winning behavior is whatever differs between the two groups. Also tracks the
economy trajectory (his troops vs the AI's over time) to test the 'farms troops by
transferring in/out' exploit hypothesis.

Usage: python tools/analyze_fader_wins.py [playdata.json]   (default: phone_export/playdata_fader.json)
"""
import json
import statistics
import sys
from collections import Counter

PLAYER = 1  # owner id of the human in the logs (0 = neutral, >=2 = AI)


def board_totals(own, tr):
    myT = sum(tr[i] for i in range(len(own)) if own[i] == PLAYER)
    aiT = sum(tr[i] for i in range(len(own)) if own[i] >= 2)
    myS = sum(1 for o in own if o == PLAYER)
    aiS = sum(1 for o in own if o >= 2)
    return myT, aiT, myS, aiS


def econ_at(game, frac):
    """His share of total troops at a fraction (0..1) through the game's snapshots."""
    snaps = game.get("snaps") or []
    if not snaps:
        return None
    s = snaps[min(len(snaps) - 1, int(frac * (len(snaps) - 1)))]
    if "own" not in s or "tr" not in s:
        return None
    myT, aiT, _, _ = board_totals(s["own"], s["tr"])
    tot = myT + aiT
    return myT / tot if tot else None


def group_stats(games, label):
    print("=" * 66)
    print("  %s  (%d games)" % (label, len(games)))
    print("=" * 66)
    if not games:
        return

    # move-level aggregates
    fr = Counter()
    neutral = own = enemy = nmoves = 0
    srcTs, tgtTs = [], []
    reinforce_share = []
    for g in games:
        gr = gn = go = ge = 0
        st, tt = [], []
        for m in g.get("moves", []):
            a = m.get("act")
            if not a:
                continue
            nmoves += 1; gr += 1
            fr[a.get("frac")] += 1
            o = a.get("tgtOwner")
            if o == 0: neutral += 1; gn += 1
            elif o == PLAYER: own += 1; go += 1
            else: enemy += 1; ge += 1
            if isinstance(a.get("srcT"), (int, float)): st.append(a["srcT"]); srcTs.append(a["srcT"])
            if isinstance(a.get("tgtT"), (int, float)): tt.append(a["tgtT"]); tgtTs.append(a["tgtT"])
        if gr: reinforce_share.append(go / gr)

    def pct(n): return 100.0 * n / nmoves if nmoves else 0
    print("  moves/game: %.1f    avg duration: %.0fs" % (
        statistics.mean(len(g.get("moves", [])) for g in games),
        statistics.mean(g.get("duration", 0) for g in games)))
    print("  split:  all %.0f%%   half %.0f%%   quarter %.0f%%" % (
        pct(fr.get(1, 0)), pct(fr.get(2, 0)), pct(fr.get(4, 0))))
    print("  target: neutral %.0f%%   attack-enemy %.0f%%   REINFORCE-own %.0f%%" % (
        pct(neutral), pct(enemy), pct(own)))
    print("  avg troops sent-from: %.1f   avg target size: %.1f" % (
        statistics.mean(srcTs) if srcTs else 0, statistics.mean(tgtTs) if tgtTs else 0))

    # economy trajectory: his troop share at 25/50/75/end
    def share(frac):
        vals = [econ_at(g, frac) for g in games]
        vals = [v for v in vals if v is not None]
        return statistics.mean(vals) if vals else 0
    print("  his troop-share over the game:  25%%=%.0f%%  50%%=%.0f%%  75%%=%.0f%%  end=%.0f%%" % (
        100 * share(0.25), 100 * share(0.5), 100 * share(0.75), 100 * share(1.0)))

    # economy efficiency at the end: troops per owned state (his vs AI) — high = 'farming'
    eff_mine, eff_ai = [], []
    for g in games:
        fb = g.get("finalBoard")
        if not fb or "own" not in fb:
            continue
        myT, aiT, myS, aiS = board_totals(fb["own"], fb["tr"])
        if myS: eff_mine.append(myT / myS)
        if aiS: eff_ai.append(aiT / aiS)
    if eff_mine and eff_ai:
        print("  end troops-per-state:  HIM %.1f   vs   AI %.1f" % (
            statistics.mean(eff_mine), statistics.mean(eff_ai)))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "phone_export/playdata_fader.json"
    games = json.load(open(path))
    wins = [g for g in games if g.get("won")]
    losses = [g for g in games if not g.get("won")]
    print("\nTotal %d games — he WON %d, LOST %d\n" % (len(games), len(wins), len(losses)))
    group_stats(wins, "HIS WINS (the AI's losses — how he beats it)")
    print()
    group_stats(losses, "HIS LOSSES (for contrast)")


if __name__ == "__main__":
    main()
