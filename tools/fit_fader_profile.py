#!/usr/bin/env python3
"""Phase 2a: fit the Fader-bot's parameters from his real games.

Reads a play-data export and distills the tester's measured behaviour into the small
parameter set the arena's `script:fader` opponent consumes (selfplay_arena.js SCRIPTS.fader).
Writes tools/fader_profile.json; the arena reads it via the FADER_PROFILE env var or the
orchestrator bakes it in.

Every parameter is measured, not guessed — rerun this whenever a fresh export lands and the
bot tracks his current style automatically.

Usage: python tools/fit_fader_profile.py [playdata.json] [-o tools/fader_profile.json]
"""
import argparse
import json
import statistics

PLAYER = 1


def board_totals(own, tr):
    myT = sum(tr[i] for i in range(len(own)) if own[i] == PLAYER)
    aiT = sum(tr[i] for i in range(len(own)) if own[i] >= 2)
    return myT, aiT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="phone_export/playdata_fader.json")
    ap.add_argument("-o", "--out", default="tools/fader_profile.json")
    args = ap.parse_args()
    games = json.load(open(args.path))

    frac_all = frac_half = 0
    n = neutral = own = enemy = joint = 0
    attack_ratios = []     # myT/enemyT at the moment he chooses to ATTACK (war-phase trigger)
    reinforce_ratios = []  # same at the moment he REINFORCES (econ-phase evidence)
    force_ratios = []      # srcT/tgtT on his attacks (how big a fight he picks)
    max_stacks = []
    for g in games:
        peak = 0
        for m in g.get("moves", []):
            a = m.get("act")
            if not a:
                continue
            n += 1
            if a.get("frac") == 1: frac_all += 1
            elif a.get("frac") == 2: frac_half += 1
            if (a.get("joint") or 1) > 1: joint += 1
            o = a.get("tgtOwner")
            myT, aiT = board_totals(m.get("own", []), m.get("tr", []))
            ratio = myT / max(1, aiT)
            if o == 0:
                neutral += 1
            elif o == PLAYER:
                own += 1; reinforce_ratios.append(ratio)
            else:
                enemy += 1; attack_ratios.append(ratio)
                if isinstance(a.get("srcT"), (int, float)) and isinstance(a.get("tgtT"), (int, float)):
                    force_ratios.append(a["srcT"] / max(1, a["tgtT"]))
        for s in g.get("snaps", []):
            ownv, trv = s.get("own"), s.get("tr")
            if ownv and trv:
                for i in range(len(ownv)):
                    if ownv[i] == PLAYER:
                        peak = max(peak, trv[i])
        if peak:
            max_stacks.append(peak)

    prof = {
        "fitted_from": args.path,
        "games": len(games),
        "moves": n,
        # behaviour mix (his measured move distribution)
        "reinforceRate": round(own / n, 3),
        "neutralRate": round(neutral / n, 3),
        "attackRate": round(enemy / n, 3),
        "fracAll": round(frac_all / n, 3),
        "jointRate": round(joint / n, 3),
        # war-phase trigger: the troop-advantage at which his ATTACK moves typically happen.
        # median of myT/enemyT across all his attack decisions — below this he farms, above he strikes.
        "warRatio": round(statistics.median(attack_ratios), 2) if attack_ratios else 1.8,
        # how big a fight he picks per-send (joint attacks combine several of these)
        "forceRatio": round(statistics.median(force_ratios), 2) if force_ratios else 0.6,
        # snowball bank: the stack size he consolidates toward before converting to territory
        "stackTarget": round(statistics.mean(max_stacks)) if max_stacks else 100,
        "jointMax": 4,
    }
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(prof, f, indent=2)
    print("fitted profile from %d games / %d moves -> %s" % (len(games), n, args.out))
    for k, v in prof.items():
        if k not in ("fitted_from",):
            print("  %-14s %s" % (k, v))


if __name__ == "__main__":
    main()
