#!/usr/bin/env python3
# Compatibility screen: run ANY gnn-v1 policy JSON — including the pre-split 3-move-feature
# generation (the shipped src/web/gnn_policy.json) — through the same scripted-opponent screen
# as train_gnn_torch.vs_script, so old and new nets can be compared apples-to-apples.
#
# The only architecture-dependent piece is the move chooser: a move_feats=3 net predates unit
# splitting, so it is scored exactly on its own contract — full sends only (frac fixed at 1.0)
# and 3 move features [(sent-tt)/FN, dist, sent>tt] — mirroring how it was trained and how the
# in-game scorer effectively treats it (the 4th input falls off the end of its weight rows).
#
#   usage: python3 tools/compat_screen.py <netA.json> [netB.json ...] [--games N] [--opps a,b,c]
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gnn_net_torch as G
import train_gnn_torch as TG
import train_rl as T


def choose_compat(net, g, owner, device='cpu'):
    if net.move_feats >= 4:
        return TG.choose_gnn(net, g, owner, device)   # current split-capable contract
    # ---- pre-split (move_feats=3) contract: full sends only, no frac feature ----
    o = g['owner']; t = g['troops']
    srcs = np.where((o == owner) & (t >= 5))[0]
    if len(srcs) == 0:
        return None
    with torch.no_grad():
        h = TG.board_embeddings(net, o.tolist(), t.tolist(), g['adj'], g['armies'], owner, device)
        FN = G.FN
        best = float(net.noop_bias.item()); mv = None
        for si in srcs:
            tgts = T.legal_targets(g, int(si), owner)
            if len(tgts) == 0:
                continue
            st = float(t[si])
            pairs = torch.tensor([[si, tj] for tj in tgts], dtype=torch.long, device=device)
            move_x = torch.tensor(
                [[(st - float(t[tj])) / FN,
                  float(np.hypot(g['pos'][tj, 0] - g['pos'][si, 0], g['pos'][tj, 1] - g['pos'][si, 1])) / g['rlDiag'],
                  1.0 if st > t[tj] else 0.0]
                 for tj in tgts], dtype=torch.float32, device=device)
            scores = net.move_scores(h, pairs, move_x)
            j = int(torch.argmax(scores).item())
            if scores[j].item() > best:
                best = scores[j].item(); mv = (int(si), int(tgts[j]))
        return mv


def screen(net, opp, rng, n, device='cpu', N=18, max_t=80.0):
    # identical to train_gnn_torch.vs_script except the chooser is contract-aware
    w = 0
    for _ in range(n):
        Pn = int(rng.choice([2, 3])); ci = int(rng.integers(Pn))
        g = T.new_game(N, Pn, rng)
        timers = {p: rng.uniform(0.2, 1.0) for p in range(1, Pn + 1)}
        t = 0.0
        while t < max_t:
            T.step(g, 0.25); t += 0.25
            for p in range(1, Pn + 1):
                timers[p] -= 0.25
                if timers[p] <= 0:
                    timers[p] = rng.uniform(0.6, 1.0)
                    if p - 1 == ci:            mv = choose_compat(net, g, p, device)
                    elif opp == 'turtle':      mv = T.turtle(g, p)
                    elif opp == 'rush':        mv = T.rush(g, p)
                    elif opp == 'farmer':      mv = T.farmer(g, p)
                    elif opp == 'economist':   mv = T.economist(g, p)
                    else:                      mv = T.heuristic(g, p)
                    if mv:
                        T.send(g, mv[0], mv[1], mv[2] if len(mv) > 2 else 1.0)
            if len(T.alive_owners(g)) <= 1:
                break
        o = g['owner']; al = T.alive_owners(g); res = {}
        for p in range(1, Pn + 1):
            s = int((o == p).sum()) / N
            if al == {p}: s += 1.0
            res[p] = s
        if res[ci + 1] >= max(res.values()) - 1e-9:
            w += 1
    return w / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('nets', nargs='+')
    ap.add_argument('--games', type=int, default=60)
    ap.add_argument('--opps', default='farmer,turtle,rush')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--seed', type=int, default=11)
    a = ap.parse_args()
    opps = a.opps.split(',')
    print(f"{'net':38s} feats  " + '  '.join(f'{o:>8s}' for o in opps))
    for path in a.nets:
        net = G.from_json(path, device=a.device)
        rng = np.random.default_rng(a.seed)   # same seed per net -> same board sequence
        row = []
        for opp in opps:
            row.append(screen(net, opp, rng, a.games, device=a.device))
        print(f"{os.path.basename(path):38s} mf={net.move_feats}   " + '  '.join(f'{v:8.3f}' for v in row), flush=True)


if __name__ == '__main__':
    main()
