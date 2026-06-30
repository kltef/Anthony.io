#!/usr/bin/env python3
# Ingest exported State.io play-data logs (the JSON from the in-game "Data" button) and fit a
# baseline opponent model: P(human's move | board). Each human move is a discrete choice among the
# legal (owned-source -> any-other-target) candidates, so we fit a conditional-logit (softmax over
# candidates) scorer on board+action features, and compare it to simple heuristics. Also aggregates
# the behavioural profile into a plain-English scouting report.
#
#   usage: python3 tools/parse_play_data.py stateio_playdata.json
#
# The loader is tolerant of a truncated trailing record (large exports can get cut when copied):
# it auto-closes any unterminated brackets at EOF so a partial last game is salvaged, not lost.
import json, sys, math
import numpy as np

PLAYER = 1   # the human's owner id in the logs

def tolerant_load(raw):
    # close any brackets/braces left open at EOF (ignoring those inside strings), then parse
    depth = []; in_str = False; esc = False
    for ch in raw:
        if in_str:
            if esc: esc = False
            elif ch == '\\': esc = True
            elif ch == '"': in_str = False
            continue
        if ch == '"': in_str = True
        elif ch in '[{': depth.append(']' if ch == '[' else '}')
        elif ch in ']}':
            if depth: depth.pop()
    fixed = raw
    if in_str: fixed += '"'
    fixed += ''.join(reversed(depth))
    try:
        return json.loads(fixed)
    except Exception:
        # last resort: trim to the last top-level '}' that yields valid JSON array
        cut = fixed.rfind('}')
        while cut > 0:
            try: return json.loads(fixed[:cut+1] + ']')
            except Exception: cut = fixed.rfind('}', 0, cut)
        return []

def candidates(move, header):
    """Return list of (src,tgt,feat[]) for every legal move available at this board, plus the
    index of the one the human actually chose (or -1 if it wasn't reconstructable)."""
    own, tr = move['own'], move['tr']
    cx, cy = header['cx'], header['cy']; diag = header.get('rlDiag', 1) or 1
    n = len(own)
    srcs = [i for i in range(n) if own[i] == PLAYER and tr[i] > 1]
    tgts = [i for i in range(n) if own[i] != PLAYER]
    cands, chosen = [], -1
    a = move['act']
    for s in srcs:
        for t in tgts:
            d = math.hypot(cx[s]-cx[t], cy[s]-cy[t])
            feat = [tr[s]/50.0, tr[t]/50.0, (tr[s]-tr[t])/50.0, d/diag,
                    1.0 if own[t] == 0 else 0.0,           # target is neutral
                    1.0 if tr[s] > tr[t] else 0.0,         # can take it outright
                    1.0]                                   # bias
            if s == a['src'] and t == a['tgt']: chosen = len(cands)
            cands.append((s, t, feat))
    return cands, chosen

def build_dataset(games):
    samples = []   # (X [k,d], chosen_idx)
    for g in games:
        h = g.get('header');
        if not h: continue
        for mv in g.get('moves', []):
            cands, ch = candidates(mv, h)
            if ch < 0 or len(cands) < 2: continue
            X = np.array([c[2] for c in cands], dtype=np.float64)
            samples.append((X, ch, cands))
    return samples

def softmax(z): z = z - z.max(); e = np.exp(z); return e/e.sum()

def train_condlogit(samples, epochs=400, lr=0.3):
    d = samples[0][0].shape[1]; w = np.zeros(d)
    for _ in range(epochs):
        grad = np.zeros(d)
        for X, ch, _ in samples:
            p = softmax(X @ w)
            grad += X[ch] - p @ X
        w += lr * grad / len(samples)
    return w

def top1(score_fn, samples):
    hit = 0
    for X, ch, cands in samples:
        if int(np.argmax(score_fn(X, cands))) == ch: hit += 1
    return hit/len(samples) if samples else 0.0

def scouting_report(prof):
    L = []
    def has(k): return k in prof and prof[k] is not None
    if has('aggression'): L.append(("very aggressive (high attack rate)" if prof['aggression']>0.7 else
                                     "measured pace" if prof['aggression']<0.35 else "moderately aggressive"))
    if has('neutralBias'): L.append(("fights enemies, doesn't farm neutral land" if prof['neutralBias']<0.35 else
                                      "expands into neutral land first" if prof['neutralBias']>0.65 else "mixes expansion and fighting"))
    if has('targetWeakness') and prof['targetWeakness']>0.65: L.append("pounces on the weakest available target")
    if has('armySize'): L.append("strikes with small stacks" if prof['armySize']<0.25 else
                                  "builds big stacks before committing" if prof['armySize']>0.6 else "uses medium stacks")
    if has('jointRate') and prof['jointRate']>0.5: L.append("favours multi-source coordinated attacks")
    if has('fracAll') and prof['fracAll']>0.8: L.append("almost always sends ALL troops")
    if has('reinforceRate') and prof['reinforceRate']<0.25: L.append("rarely reinforces — almost purely offensive")
    if has('earlyAggro') and prof['earlyAggro']>0.6: L.append("attacks enemies early")
    return L

def exploit_tips(prof):
    tips = []
    if prof.get('targetWeakness',0)>0.65: tips.append("leave a weak-LOOKING bait state that's reinforced-on-arrival — they'll pounce")
    if prof.get('fracAll',0)>0.8 and prof.get('reinforceRate',1)<0.3: tips.append("they send-all and don't reinforce — counter-attack the square they just captured (it's left thin)")
    if prof.get('jointRate',0)>0.5: tips.append("read their multi-source buildup and defend the convergence point a beat early")
    if prof.get('neutralBias',1)<0.35: tips.append("they ignore neutral land — out-economy them by grabbing it safely while they fight")
    return tips

def main():
    if len(sys.argv) < 2:
        print("usage: python3 tools/parse_play_data.py <playdata.json>"); return
    games = tolerant_load(open(sys.argv[1]).read())
    if isinstance(games, dict): games = [games]
    print(f"loaded {len(games)} game(s)")
    by_diff = {}
    profs = []
    total_moves = 0
    for g in games:
        by_diff[g.get('difficulty','?')] = by_diff.get(g.get('difficulty','?'),0)+1
        total_moves += len(g.get('moves',[]))
        if g.get('profile'): profs.append(g['profile'])
    print(f"  by difficulty: {by_diff}")
    print(f"  total human moves logged: {total_moves}   games with profile: {len(profs)}")

    # aggregate profile -> scouting report
    if profs:
        keys = sorted({k for p in profs for k in p})
        agg = {k: float(np.mean([p[k] for p in profs if k in p and p[k] is not None])) for k in keys}
        print("\n== averaged behavioural profile ==")
        for k in keys: print(f"  {k:15s} {agg[k]:.3f}")
        print("\n== scouting report ==")
        for line in scouting_report(agg): print("  •", line)
        print("\n== how an exploiter would punish this ==")
        for line in exploit_tips(agg): print("  →", line)

    # fit the move predictor
    samples = build_dataset(games)
    print(f"\n== P(move|board) predictor ==  trainable decisions: {len(samples)}")
    if len(samples) >= 4:
        w = train_condlogit(samples)
        learned = top1(lambda X,c: X @ w, samples)
        weakest = top1(lambda X,c: -X[:,1], samples)                 # pick lowest target troops
        nearest = top1(lambda X,c: -X[:,3], samples)                 # pick nearest
        overkill= top1(lambda X,c:  X[:,2], samples)                 # biggest troop advantage
        rand    = float(np.mean([1.0/len(c) for _,_,c in samples]))  # expected random top-1
        print(f"  baseline  random        : {rand:.2f}")
        print(f"  heuristic weakest-target: {weakest:.2f}")
        print(f"  heuristic nearest       : {nearest:.2f}")
        print(f"  heuristic max-overkill  : {overkill:.2f}")
        print(f"  LEARNED  cond-logit     : {learned:.2f}  (top-1 move-match)")
        print("  learned weights:", {n_:round(v,2) for n_,v in zip(
            ['srcT','tgtT','delta','dist','neutral','can-take','bias'], w)})
        print("\n  NOTE: with few games this is in-sample/overfit — it's the pipeline, not a verdict.")
        print("  Accuracy and the model sharpen as more exported games accumulate.")
        # learning curve: train on the first K games, test on the rest — shows when more data stops helping
        if len(games) >= 6:
            print("\n== learning curve (train on first K games, top-1 on held-out rest) ==")
            for frac in (0.25, 0.5, 0.75):
                k = max(1, int(len(games)*frac))
                tr = build_dataset(games[:k]); te = build_dataset(games[k:])
                if len(tr) >= 4 and te:
                    wk = train_condlogit(tr); acc = top1(lambda X,c: X @ wk, te)
                    print(f"  first {k:3d} games ({len(tr):4d} decisions) -> held-out top-1 {acc:.2f}")
            print("  Rising = keep playing. Flat = you have enough data for a stable model (step 3).")
        else:
            print("  (need >=6 games for a held-out learning curve — keep playing + exporting.)")
    else:
        print("  need more logged moves to fit a stable model (keep playing + export).")

if __name__ == '__main__':
    main()
