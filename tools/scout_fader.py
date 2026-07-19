#!/usr/bin/env python3
"""Scouting report from a State.io play-data export (playdata.json).

Aggregates a player's exported games into a readable profile: win/loss record, how he
splits troops (all / half / quarter), what he targets (neutral vs attack vs reinforce),
his pace and aggression, and a per-game deep-dive on the games he LOST. Built to run the
moment a fresh export lands in phone_export/.

Usage:
  python tools/scout_fader.py [path/to/playdata.json]     # default: phone_export/playdata.json
  python tools/scout_fader.py new.json --vs old.json      # compare two exports (e.g. 158 vs 83)

Move record fields used (per moves[].act): frac (1=all,2=half,4=quarter), src/tgt,
srcT/tgtT (troop counts), tgtOwner (0=neutral, player's own=reinforce, else enemy),
dist, joint. Per-game `profile` metrics are averaged when present.
"""
import argparse
import json
import statistics
from collections import Counter

FRAC_LABEL = {1: "all (full stack)", 2: "half", 4: "quarter", 3: "third"}


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("games", [])


def pct(n, d):
    return (100.0 * n / d) if d else 0.0


def summarize(games, label):
    print("=" * 70)
    print("  SCOUTING REPORT — %s   (%d games)" % (label, len(games)))
    print("=" * 70)
    if not games:
        print("  (no games)")
        return {}

    # --- names / dates -------------------------------------------------------
    names = Counter(g.get("name", "?") for g in games)
    ts = [g.get("ts") for g in games if g.get("ts")]
    print("  Player(s): " + ", ".join("%s x%d" % (n, c) for n, c in names.most_common()))
    if ts:
        import datetime
        lo = datetime.datetime.utcfromtimestamp(min(ts) / 1000).date()
        hi = datetime.datetime.utcfromtimestamp(max(ts) / 1000).date()
        print("  Date range: %s -> %s" % (lo, hi))

    # --- record --------------------------------------------------------------
    wins = sum(1 for g in games if g.get("won"))
    losses = len(games) - wins
    print("\n  RECORD: %d W - %d L   (%.1f%% win rate)" % (wins, losses, pct(wins, len(games))))
    by_diff = {}
    for g in games:
        d = g.get("difficulty", "?")
        w, l = by_diff.get(d, (0, 0))
        by_diff[d] = (w + (1 if g.get("won") else 0), l + (0 if g.get("won") else 1))
    for d, (w, l) in sorted(by_diff.items(), key=lambda kv: -(kv[1][0] + kv[1][1])):
        print("    %-12s %d W - %d L  (%.0f%%)" % (d, w, l, pct(w, w + l)))

    # --- gather all moves ----------------------------------------------------
    moves = []
    for g in games:
        for m in g.get("moves", []):
            a = m.get("act")
            if a:
                moves.append(a)
    nmoves = len(moves)

    # --- fraction usage ------------------------------------------------------
    if nmoves:
        fr = Counter(a.get("frac") for a in moves)
        print("\n  TROOP SPLIT (how he sends):")
        for k, c in sorted(fr.items(), key=lambda kv: -kv[1]):
            print("    %-18s %5d  (%.0f%%)" % (FRAC_LABEL.get(k, "frac=%s" % k), c, pct(c, nmoves)))

        # --- targeting -------------------------------------------------------
        neutral = sum(1 for a in moves if a.get("tgtOwner") == 0)
        own = sum(1 for a in moves if a.get("tgtOwner") == 1)
        enemy = nmoves - neutral - own
        print("\n  TARGETING:")
        print("    Grab neutral   %5d  (%.0f%%)" % (neutral, pct(neutral, nmoves)))
        print("    Attack enemy   %5d  (%.0f%%)" % (enemy, pct(enemy, nmoves)))
        print("    Reinforce own  %5d  (%.0f%%)" % (own, pct(own, nmoves)))
        joint = sum(1 for a in moves if a.get("joint"))
        print("    Joint (multi-source) attacks: %.0f%%" % pct(joint, nmoves))

        srcT = [a["srcT"] for a in moves if isinstance(a.get("srcT"), (int, float))]
        tgtT = [a["tgtT"] for a in moves if isinstance(a.get("tgtT"), (int, float))]
        if srcT and tgtT:
            print("    Avg troops sent-from: %.1f   Avg target size: %.1f" % (
                statistics.mean(srcT), statistics.mean(tgtT)))
            overkill = [a["srcT"] / max(1, a["tgtT"]) for a in moves
                        if isinstance(a.get("srcT"), (int, float)) and isinstance(a.get("tgtT"), (int, float))]
            safe = sum(1 for a in moves if isinstance(a.get("srcT"), (int, float))
                       and isinstance(a.get("tgtT"), (int, float)) and a["srcT"] > a["tgtT"])
            print("    Attacks with more troops than target (safe): %.0f%%   median force ratio %.2fx" % (
                pct(safe, len(overkill)), statistics.median(overkill) if overkill else 0))
        dist = [a["dist"] for a in moves if isinstance(a.get("dist"), (int, float))]
        if dist:
            print("    Avg move distance (reach): %.2f" % statistics.mean(dist))

    # --- pace ----------------------------------------------------------------
    durs = [g.get("duration") for g in games if isinstance(g.get("duration"), (int, float))]
    mpg = [len(g.get("moves", [])) for g in games]
    if durs:
        print("\n  PACE:")
        print("    Avg game length: %.0fs   (median %.0fs, range %.0f-%.0f)" % (
            statistics.mean(durs), statistics.median(durs), min(durs), max(durs)))
    if mpg:
        print("    Avg moves/game: %.1f   (median %d, max %d)" % (
            statistics.mean(mpg), int(statistics.median(mpg)), max(mpg)))
    if durs and moves:
        total_min = sum(durs) / 60.0
        print("    Move rate: %.1f moves/min" % (nmoves / total_min if total_min else 0))

    # --- averaged style profile ---------------------------------------------
    prof_keys = ["aggression", "earlyAggro", "targetWeakness", "armySize", "reach",
                 "avgTargetT", "reinforceRate", "jointRate", "neutralBias",
                 "fracAll", "fracHalf", "fracQuarter"]
    prof = {}
    for k in prof_keys:
        vals = [g["profile"][k] for g in games
                if isinstance(g.get("profile"), dict) and isinstance(g["profile"].get(k), (int, float))]
        if vals:
            prof[k] = statistics.mean(vals)
    if prof:
        print("\n  STYLE PROFILE (avg of per-game metrics, 0-1 unless noted):")
        for k in prof_keys:
            if k in prof:
                print("    %-15s %.3f" % (k, prof[k]))

    # --- loss deep-dive ------------------------------------------------------
    lost = [g for g in games if not g.get("won")]
    if lost:
        print("\n  LOSSES (%d) — the games to learn from:" % len(lost))
        for g in lost:
            fb = g.get("finalBoard") or {}
            own = fb.get("own", [])
            share = pct(sum(1 for o in own if o == 1), len(own)) if own else 0
            print("    %-11s %3.0fs  %2d moves  final territory %2.0f%%" % (
                g.get("difficulty", "?"), g.get("duration", 0), len(g.get("moves", [])), share))

    return {"wins": wins, "losses": losses, "n": len(games),
            "prof": prof, "nmoves": nmoves,
            "frac": Counter(a.get("frac") for a in moves) if moves else Counter()}


def main():
    ap = argparse.ArgumentParser(description="Scouting report from a State.io play-data export.")
    ap.add_argument("path", nargs="?", default="phone_export/playdata.json")
    ap.add_argument("--vs", default=None, help="second export to compare against (e.g. the old one)")
    ap.add_argument("--label", default=None, help="label for the main export")
    args = ap.parse_args()

    games = load(args.path)
    summarize(games, args.label or args.path)

    if args.vs:
        print("\n")
        summarize(load(args.vs), "COMPARISON: " + args.vs)


if __name__ == "__main__":
    main()
