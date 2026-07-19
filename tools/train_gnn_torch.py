#!/usr/bin/env python3
# GPU training loop for the board-aware (GNN) policy net — FINDINGS.md future-path #1, trained with
# the same real-planner visit-count distillation as tools/train_rl_torch.py (future-path #3), and
# meant to be run for as long as you can afford (future-path #2, "deep self-play at scale") since a
# structured net is exactly the thing more self-play data has somewhere to go, per FINDINGS.md.
#
# Same round structure and same no-regression posture as train_rl_torch.py — see that file's header
# for the full rationale; this one only differs in HOW a candidate is trained: node embeddings must
# be recomputed from the raw board (owner/troops/adjacency/armies) per decision, not read off a flat
# per-move feature vector, so the training loop here operates one decision (one board) at a time
# rather than a single flattened batch. Self-play generation still dominates wall-clock (see
# FINDINGS.md 5.8), so this is an acceptable trade for correctness — a batched/vectorized version
# across boards of equal size is a reasonable future optimization, not attempted here.
#
# Data collection (2026-07-11): the arena now (a) subsamples noop-dominated decisions (93% of raw
# decisions were "sit still"), (b) prunes boards to include real-map-style corner seats, (c) stamps
# every decision with the game's final graded outcome (=> value co-training below), and this loop
# (d) generates data at a much deeper search budget than the deploy/gate budget (--selfplay-budget),
# (e) mixes scripted exploiter + human-profile opponents into data generation (--script-mix), and
# (f) adds AlphaZero-style Dirichlet root noise + opening move-sampling temperature (--explore-*),
# breaking the "splits never explored -> never learned" feedback loop. The GATE is untouched by all
# of this: promotion is still measured noise-free at the deploy budget.
#
#   usage: python3 tools/train_gnn_torch.py --hours 5 --device cuda
import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_rl as T
import train_value as V
import gnn_net_torch as G

SCR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCR)


# ---------------- board -> GNN scoring (Python-side, mirrors index.html's gnn* functions) ----------------
def incoming_from_armies(armies, N, mover):
    inc_mine = [0.0] * N
    inc_enemy = [0.0] * N
    for a in armies:
        if a['owner'] == mover:
            inc_mine[a['ti']] += a['count']
        else:
            inc_enemy[a['ti']] += a['count']
    return inc_mine, inc_enemy


def board_embeddings(net, owner, troops, adj, armies, mover, device):
    inc_mine, inc_enemy = incoming_from_armies(armies, len(owner), mover)
    nf = G.node_features(owner, troops, adj, inc_mine, inc_enemy, mover)
    node_x = torch.tensor(nf, dtype=torch.float32, device=device)
    return net.node_embeddings(node_x, adj)


def choose_gnn(net, g, owner, device='cpu'):
    # GNN-net move chooser over train_rl.py's lightweight environment `g` — used for the no-forget
    # screen (vs_script) and self-play (value co-training data generation), mirroring train_rl.py's
    # choose()/choose16() but scoring via node embeddings instead of the flat 12-feature MLP.
    o = g['owner']
    t = g['troops']
    srcs = np.where((o == owner) & (t >= 5))[0]
    if len(srcs) == 0:
        return None
    with torch.no_grad():
        h = board_embeddings(net, o.tolist(), t.tolist(), g['adj'], g['armies'], owner, device)
        FN = G.FN
        best = float(net.noop_bias.item())
        mv = None
        for si in srcs:
            tgts = T.legal_targets(g, int(si), owner)
            if len(tgts) == 0:
                continue
            st = float(t[si])
            dists = [float(np.hypot(g['pos'][tj, 0] - g['pos'][si, 0],
                                    g['pos'][tj, 1] - g['pos'][si, 1])) / g['rlDiag'] for tj in tgts]
            pairs = torch.tensor([[si, tj] for tj in tgts], dtype=torch.long, device=device)
            # each split fraction is a separate candidate: sent = st*frac (see MOVE_FEATS contract)
            for frac in G.FRACTIONS:
                sent = st * frac
                move_x = torch.tensor(
                    [[(sent - float(t[tj])) / FN, dists[k], 1.0 if sent > t[tj] else 0.0, frac]
                     for k, tj in enumerate(tgts)], dtype=torch.float32, device=device)
                scores = net.move_scores(h, pairs, move_x)
                j = int(torch.argmax(scores).item())
                if scores[j].item() > best:
                    best = scores[j].item()
                    mv = (int(si), int(tgts[j]), frac)
    return mv


def vs_script(net, opp, rng, n=60, N=18, device='cpu', max_t=80.0):
    # self-contained no-forget screen, same shape as train_rl_torch.py's vs_script — see that file's
    # comment for why train_selfplay_loop.py is deliberately not imported/reused here.
    # max_t is intentionally shorter than train_rl.py's play()/train_rl_torch.py's default: every
    # decision here runs a real PyTorch graph-attention forward pass (~3ms on CPU, per the profiling
    # that motivated this), not a cheap numpy matmul, so a full-length game-count screen would be
    # slow enough to dominate a round's wall-clock — this is a screen, not a final evaluation.
    w = 0
    for _ in range(n):
        Pn = int(rng.choice([2, 3]))
        ci = int(rng.integers(Pn))
        g = T.new_game(N, Pn, rng)
        timers = {p: rng.uniform(0.2, 1.0) for p in range(1, Pn + 1)}
        air = {}
        t = 0.0
        while t < max_t:
            T.step(g, 0.25)
            t += 0.25
            for p in range(1, Pn + 1):
                timers[p] -= 0.25
                if timers[p] <= 0:
                    timers[p] = rng.uniform(0.6, 1.0)
                    if p - 1 == ci:
                        mv = choose_gnn(net, g, p, device)
                    elif opp == 'turtle':
                        mv = T.turtle(g, p)
                    elif opp == 'rush':
                        mv = T.rush(g, p)
                    elif opp == 'farmer':
                        mv = T.farmer(g, p)
                    else:
                        mv = T.heuristic(g, p)
                    if mv:
                        T.send(g, mv[0], mv[1], mv[2] if len(mv) > 2 else 1.0)
            if len(T.alive_owners(g)) <= 1:
                break
        o = g['owner']
        al = T.alive_owners(g)
        res = {}
        for p in range(1, Pn + 1):
            s = int((o == p).sum()) / N
            if al == {p}:
                s += 1.0
            res[p] = s
        if res[ci + 1] >= max(res.values()) - 1e-9:
            w += 1
    return w / n


# ---------------- arena plumbing (identical shape to train_rl_torch.py, format-specific write_net) ----------------
def write_net(net, value_json, path):
    j = G.to_json(net)
    if value_json:
        j['value'] = value_json
    json.dump(j, open(path, 'w'))


def _split_games(games, workers):
    # spread `games` as evenly as possible across `workers` parallel arena processes
    base = games // workers
    rem = games % workers
    return [base + (1 if i < rem else 0) for i in range(workers)]


def _run_arena(argv, env=None, timeout=3600):
    try:
        return subprocess.run(argv, capture_output=True, text=True, cwd=REPO_ROOT, timeout=timeout, env=env).stdout
    except Exception as e:
        print("  arena worker error:", e, flush=True)
        return ""


def _gate_envs(n_jobs, long_cap, long_mix):
    # Long-game mix (2026-07-12): the arena's default 38s cap means gates and training data never
    # contain late-game boards (cap economics, 300+ mega-stacks). A 52k candidate that held ~50%
    # in capped games lost 35% untimed — hoard-locking on 3 states for minutes (out-of-distribution
    # paralysis). So a slice of gate/data-gen jobs now runs with GAMECAP=long_cap.
    envs = []
    n_long = int(round(n_jobs * long_mix)) if long_cap else 0
    for i in range(n_jobs):
        env = os.environ.copy()
        if i < n_long:
            env['GAMECAP'] = str(long_cap)
        envs.append(env)
    return envs


def anchor_gate(cand_net, value_json, anchor_path, games, budget, cand_path, workers=1, long_cap=None, long_mix=0.0):
    # Anti-drift check (added 2026-07-12): a chain of small promotion gates is NOT transitive — the
    # 2026-07-11 overnight run promoted 14 times, each beating its predecessor at n=40, yet the final
    # checkpoint lost 35% vs the SHIPPED net on game-like openings (47.5% pooled over 200 games).
    # So on top of beating the previous checkpoint, a candidate must also not clearly LOSE to the
    # fixed shipped anchor. The anchor file is used verbatim (it carries its own value net).
    write_net(cand_net, value_json, cand_path)
    per_worker = _split_games(games, max(1, workers))
    argvs = [['node', 'tools/selfplay_arena.js', cand_path, anchor_path, str(g), str(budget)]
             for g in per_worker if g > 0]
    if not argvs:
        return None
    envs = _gate_envs(len(argvs), long_cap, long_mix)
    with ThreadPoolExecutor(max_workers=len(argvs)) as ex:
        outs = list(ex.map(lambda t: _run_arena(t[0], env=t[1], timeout=1800), zip(argvs, envs)))
    total_a, total_n = 0, 0
    for out in outs:
        m = re.search(r'A won (\d+)/(\d+)', out)
        if m:
            total_a += int(m.group(1))
            total_n += int(m.group(2))
    return (total_a / total_n) if total_n else None


def planner_gate(cand_net, best_net, value_json, games, budget, cand_path, best_path, workers=1,
                 long_cap=None, long_mix=0.0):
    # Both sides are FIXED files read by every worker (written once, before any process starts), so
    # running N independent arena processes concurrently and summing their win counts is equivalent
    # to one big sequential run — Node's Math.random() isn't seeded identically across processes, so
    # each worker naturally plays different boards, not N copies of the same games.
    write_net(cand_net, value_json, cand_path)
    write_net(best_net, value_json, best_path)
    per_worker = _split_games(games, max(1, workers))
    argvs = [['node', 'tools/selfplay_arena.js', cand_path, best_path, str(g), str(budget)]
             for g in per_worker if g > 0]
    if not argvs:
        return None
    envs = _gate_envs(len(argvs), long_cap, long_mix)
    with ThreadPoolExecutor(max_workers=len(argvs)) as ex:
        outs = list(ex.map(lambda t: _run_arena(t[0], env=t[1], timeout=1800), zip(argvs, envs)))
    total_a, total_n = 0, 0
    for out in outs:
        m = re.search(r'A won (\d+)/(\d+)', out)
        if m:
            total_a += int(m.group(1))
            total_n += int(m.group(2))
    return (total_a / total_n) if total_n else None


# Scripted opponents mixed into data generation (see selfplay_arena.js SCRIPTS): pure net-vs-net
# self-play never produces the board states these styles force (hoards, distant farming, joint
# human-style salvos), so the no-forget screen kept catching regressions the training data had no
# way to prevent. Only the planner seat's decisions are dumped; the scripted seat never plans.
# 'script:fader' is the play-tester's fitted farm-and-snowball bot (fit_fader_profile.py).
# ARENA_SCRIPTS env overrides the league (comma-separated, repeats allowed to oversample one
# opponent) — beat_fader.py phase 3 uses this for best-response training against the fader bot.
SCRIPT_OPPONENTS = (os.environ['ARENA_SCRIPTS'].split(',') if os.environ.get('ARENA_SCRIPTS') else
                    ['script:turtle', 'script:rush', 'script:farmer', 'script:econ', 'script:human', 'script:fader'])


def self_play_dump(net, value_json, games, budget, dump_path, net_path, workers=1,
                   script_mix=0.0, explore=None, keep_noop=None, long_cap=None, long_mix=0.0,
                   anchor_path=None, anchor_mix=0.0, midgame_pool=None, midgame_p=0.0):
    write_net(net, value_json, net_path)
    # jobs: (opponent, games, is_long) — net-vs-net split across workers, plus one job per scripted
    # opponent. A --long-mix slice of the NET-VS-NET jobs runs untimed-ish (GAMECAP=--long-cap) so
    # late-game boards (cap economics, mega-stacks) are finally in-distribution — see _gate_envs.
    # An --anchor-mix slice plays the SHIPPED anchor instead of self: the challenger trains on the
    # exact opponent it must beat, and (since both planner seats dump) it distills the anchor's own
    # search verdicts — inheriting the champion's lineage as training signal instead of racing it.
    n_script = int(round(games * script_mix))
    jobs = []
    if n_script > 0:
        per_opp = _split_games(n_script, len(SCRIPT_OPPONENTS))
        jobs += [(opp, g, False) for opp, g in zip(SCRIPT_OPPONENTS, per_opp) if g > 0]
    n_net = games - n_script
    n_anchor = int(round(n_net * anchor_mix)) if anchor_path and os.path.exists(anchor_path) else 0
    if n_anchor > 0:
        jobs += [(anchor_path, g, i % 3 == 2) for i, g in enumerate(_split_games(n_anchor, max(1, workers // 3))) if g > 0]
    net_jobs = [g for g in _split_games(n_net - n_anchor, max(1, workers)) if g > 0]
    n_long = int(round(len(net_jobs) * long_mix)) if long_cap else 0
    jobs += [(net_path, g, i < n_long) for i, g in enumerate(net_jobs)]
    if not jobs:
        return
    base_env = os.environ.copy()
    if explore:   # Dirichlet root noise + opening temperature — data generation only, never the gate
        base_env['EXPLORE_EPS'] = str(explore['eps'])
        base_env['EXPLORE_ALPHA'] = str(explore['alpha'])
        base_env['EXPLORE_TEMP'] = str(explore['temp'])
        base_env['EXPLORE_TEMP_MOVES'] = str(explore['temp_moves'])
    if keep_noop is not None:   # noop-dominated decision subsampling (see selfplay_arena.js flushDump)
        base_env['DUMP_KEEP_NOOP'] = str(keep_noop)
    if midgame_pool and midgame_p > 0 and os.path.exists(midgame_pool):
        base_env['MIDGAME_POOL'] = midgame_pool   # start a slice of games from sampled mid-game boards
        base_env['MIDGAME_P'] = str(midgame_p)
    worker_dumps = [f"{dump_path}.w{i}" for i in range(len(jobs))]

    def run_one(i):
        opp, g, is_long = jobs[i]
        env = base_env.copy()
        env['DUMP_VISITS'] = worker_dumps[i]
        if is_long:
            env['GAMECAP'] = str(long_cap)
        _run_arena(['node', 'tools/selfplay_arena.js', net_path, opp, str(g), str(budget)], env)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        list(ex.map(run_one, range(len(jobs))))
    with open(dump_path, 'w') as out:
        for wp in worker_dumps:
            if os.path.exists(wp):
                out.write(open(wp).read())
                try:
                    os.remove(wp)
                except OSError:
                    pass


def load_dump(dump_path):
    records = []
    if not os.path.exists(dump_path):
        return records
    with open(dump_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if 'adj' in r:   # only records with the raw-board fields are usable for GNN training
                    records.append(r)
            except Exception:
                continue
    return records


# ---------------- value-net co-training from dumped outcomes ----------------
# The arena now stamps every dumped decision with the game's final graded result (`result`, mover
# perspective — selfplay_arena.js gradeResult, same formula as train_value.py result()). So unlike
# train_rl_torch.py, which generates SEPARATE cheap-physics self-play for value data, the value net
# here trains directly on the REAL planner's own self-play boards — the exact distribution its leaf
# evaluations are queried on. Recipe otherwise mirrors train_rl_torch.train_value_round.
def value_theta_from_json(path):
    p = json.load(open(path))
    parts = []
    for l in p['layers']:
        parts.append(np.array(l['W'], dtype=np.float64).ravel())
        parts.append(np.array(l['b'], dtype=np.float64).ravel())
    return np.concatenate(parts)


def vfeats_of_record(rec):
    armies = [{'owner': a['owner'], 'count': a['count']} for a in rec['armies']]
    return V.vfeats_from(rec['mover'], rec['owner'], rec['troops'], armies, len(rec['owner']))


def train_value_round(value_theta, vbuf, rng, epochs=3, bs=256, lr=1e-3):
    X = np.array([f for f, _ in vbuf], dtype=np.float64)
    Y = np.array([y for _, y in vbuf], dtype=np.float64)
    n = len(value_theta)
    m = np.zeros(n)
    v = np.zeros(n)
    step = 0
    th = value_theta.copy()
    for _ in range(epochs):
        perm = rng.permutation(len(X))
        for k in range(0, len(perm), bs):
            bi = perm[k:k + bs]
            xb, yb = X[bi], Y[bi]
            pred, cache = V.vfwd(th, xb)
            dout = (2.0 / len(bi)) * (pred - yb)
            grad = V.vbwd(cache, dout)
            step += 1
            b1, b2, eps = 0.9, 0.999, 1e-8
            m = b1 * m + (1 - b1) * grad
            v = b2 * v + (1 - b2) * grad * grad
            th = th - lr * (m / (1 - b1 ** step)) / (np.sqrt(v / (1 - b2 ** step)) + eps)
    pred, _ = V.vfwd(th, X)
    mask = np.abs(Y) > 0.1
    sign_acc = float(np.mean(np.sign(pred[mask]) == np.sign(Y[mask]))) if mask.any() else 0.0
    return th, sign_acc


# ---------------- distillation loss ----------------
# decision_loss() scores one board at a time — kept as the reference implementation the batched path
# is verified numerically identical to (see tools/test_gnn_batch_equiv.py). The training loop uses the
# batched path below; this stays for that equivalence test and as readable documentation of the math.
def decision_loss(net, rec, device):
    FN = G.FN
    h = board_embeddings(net, rec['owner'], rec['troops'], rec['adj'], rec['armies'], rec['mover'], device)
    scores = []
    for mv in rec['moves']:
        if mv is None:
            scores.append(net.noop_bias.squeeze())
            continue
        si, ti = mv[0], mv[1]
        frac = mv[2] if len(mv) > 2 else 1.0      # 2-element legacy move => all-in (frac=1.0)
        st, tt = rec['troops'][si], rec['troops'][ti]
        sent = st * frac
        dist = float(np.hypot(rec['cx'][ti] - rec['cx'][si], rec['cy'][ti] - rec['cy'][si])) / rec['rlDiag']
        move_x = torch.tensor([(sent - tt) / FN, dist, 1.0 if sent > tt else 0.0, frac], dtype=torch.float32, device=device)
        pair = torch.tensor([[si, ti]], dtype=torch.long, device=device)
        scores.append(net.move_scores(h, pair, move_x.unsqueeze(0)).squeeze())
    scores = torch.stack(scores)
    visits = torch.tensor(rec['visits'], dtype=torch.float32, device=device)
    target = visits / visits.sum().clamp_min(1e-9)
    logp = F.log_softmax(scores, dim=0)
    loss = -(target * logp).sum()
    # board-aware value head (vhead): learn the game's graded outcome from the same embeddings.
    # Records without `result` (pre-outcome-stamping dumps) contribute exactly zero value loss.
    if getattr(net, 'value_head', False) and ('result' in rec):
        loss = loss + 0.5 * (net.board_value(h) - float(rec['result'])) ** 2
    return loss


def build_batch(records, device):
    # Pack a list of records that ALL share the same board node-count N into dense tensors. Candidates
    # are padded to the batch's max count and masked out in the loss, so per-decision softmax is
    # reproduced exactly. Grouping by N (see train_round) guarantees the uniform-N precondition.
    B = len(records)
    N = len(records[0]['owner'])
    FN = G.FN
    node_x = torch.zeros(B, N, G.NODE_FEATS, dtype=torch.float32)
    adj_mask = torch.zeros(B, N, N, dtype=torch.float32)
    for b, r in enumerate(records):
        inc_mine, inc_enemy = incoming_from_armies(r['armies'], N, r['mover'])
        nf = G.node_features(r['owner'], r['troops'], r['adj'], inc_mine, inc_enemy, r['mover'])
        node_x[b] = torch.tensor(nf, dtype=torch.float32)
        for i, nbrs in enumerate(r['adj']):
            for j in nbrs:
                adj_mask[b, i, j] = 1.0
    Mmax = max(len(r['moves']) for r in records)
    src_idx = torch.zeros(B, Mmax, dtype=torch.long)
    tgt_idx = torch.zeros(B, Mmax, dtype=torch.long)
    move_x = torch.zeros(B, Mmax, G.MOVE_FEATS, dtype=torch.float32)
    noop_mask = torch.zeros(B, Mmax, dtype=torch.float32)
    valid_mask = torch.zeros(B, Mmax, dtype=torch.float32)
    visits = torch.zeros(B, Mmax, dtype=torch.float32)
    for b, r in enumerate(records):
        for k, mv in enumerate(r['moves']):
            valid_mask[b, k] = 1.0
            visits[b, k] = r['visits'][k]
            if mv is None:
                noop_mask[b, k] = 1.0        # scored by noop_bias, not the move head (src/tgt stay 0 = dummy)
                continue
            si, ti = mv[0], mv[1]
            frac = mv[2] if len(mv) > 2 else 1.0    # 2-element legacy move => all-in (frac=1.0)
            src_idx[b, k] = si
            tgt_idx[b, k] = ti
            st, tt = r['troops'][si], r['troops'][ti]
            sent = st * frac
            dist = float(np.hypot(r['cx'][ti] - r['cx'][si], r['cy'][ti] - r['cy'][si])) / r['rlDiag']
            move_x[b, k, 0] = (sent - tt) / FN
            move_x[b, k, 1] = dist
            move_x[b, k, 2] = 1.0 if sent > tt else 0.0
            move_x[b, k, 3] = frac
    # value-head targets: the game's graded outcome per record (mask=0 rows contribute exactly zero)
    res_val = torch.zeros(B, dtype=torch.float32)
    res_mask = torch.zeros(B, dtype=torch.float32)
    for b, r in enumerate(records):
        if 'result' in r:
            res_val[b] = float(r['result'])
            res_mask[b] = 1.0
    return tuple(t.to(device) for t in
                 (node_x, adj_mask, src_idx, tgt_idx, move_x, noop_mask, valid_mask, visits, res_val, res_mask))


def batched_loss(net, batch):
    # Mean cross-entropy over a batch of decisions — numerically identical to averaging decision_loss()
    # over the same records (verified in tools/test_gnn_batch_equiv.py, forward AND gradients).
    node_x, adj_mask, src_idx, tgt_idx, move_x, noop_mask, valid_mask, visits, res_val, res_mask = batch
    h = net.node_embeddings_batched(node_x, adj_mask)                       # (B, N, dim)
    scores = net.move_scores_batched(h, src_idx, tgt_idx, move_x)           # (B, M)
    scores = torch.where(noop_mask > 0, net.noop_bias.to(scores.dtype), scores)
    scores = scores.masked_fill(valid_mask == 0, -1e9)                      # padded candidates -> ~0 weight
    logp = F.log_softmax(scores, dim=1)
    target = visits / visits.sum(dim=1, keepdim=True).clamp_min(1e-9)
    loss = -(target * logp * valid_mask).sum(dim=1).mean()
    # value head: 0.5*MSE vs the graded outcome, masked rows contribute zero — same per-record
    # denominator as averaging decision_loss(), so the equivalence test stays exact.
    if getattr(net, 'value_head', False):
        v = net.board_value_batched(h)
        loss = loss + 0.5 * ((v - res_val) ** 2 * res_mask).mean()
    return loss


def train_round(best, buf, device, lr, batch_size, epochs, rng):
    cand = G.GNNPolicyNet(best.node_feats, best.embed_dim, best.hops, best.move_feats, best.head_dim,
                          value_head=getattr(best, 'value_head', False)).to(device)
    cand.load_state_dict(best.state_dict())
    opt = torch.optim.Adam(cand.parameters(), lr=lr)
    # bucket by board node-count so every batch is uniform-N (arena is fixed N=16, so in practice one
    # bucket — but this stays correct and loud, not silently wrong, if boards of mixed size ever appear)
    by_n = {}
    for j, r in enumerate(buf):
        by_n.setdefault(len(r['owner']), []).append(j)
    total_loss, nsteps = 0.0, 0
    for _ in range(epochs):
        for n_nodes, idxs in by_n.items():
            order = rng.permutation(len(idxs))
            for i in range(0, len(order), batch_size):
                sel = [idxs[k] for k in order[i:i + batch_size]]
                batch = build_batch([buf[j] for j in sel], device)
                opt.zero_grad()
                loss = batched_loss(cand, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(cand.parameters(), 5.0)
                opt.step()
                total_loss += loss.item()
                nsteps += 1
    return cand, total_loss / max(1, nsteps)


# ---------------- live training visualizer (http://localhost:<port>) ----------------
# A dependency-free threaded HTTP server started when training begins. It serves one self-refreshing
# page that shows live training stats plus a grid of the most recent self-play boards (territories at
# their real positions, coloured by owner, border edges, and the planner's chosen move drawn as an
# arrow labelled with its split fraction). VIZ_STATE is a plain dict the training loop writes into.
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VIZ_STATE = {'stats': {}, 'boards': [], 'log': []}

def viz_publish_stats(**kw):
    VIZ_STATE['stats'].update(kw)

def viz_log(line):
    VIZ_STATE['log'].append(line)
    del VIZ_STATE['log'][:-40]          # keep the last 40 lines

def viz_publish_boards(dump_path, k=9):
    # sample up to k evenly-spaced decisions from the latest dump for display
    try:
        with open(dump_path) as f:
            lines = f.readlines()
    except OSError:
        return
    if not lines:
        return
    step = max(1, len(lines) // k)
    boards = []
    for ln in lines[::step][:k]:
        try:
            r = json.loads(ln)
        except Exception:
            continue
        vis = r.get('visits') or []
        ch = r.get('chosen', 0)
        mv = r['moves'][ch] if r.get('moves') and ch < len(r['moves']) and vis else None
        boards.append({'owner': r['owner'], 'troops': [round(float(x), 1) for x in r['troops']],
                       'cx': r['cx'], 'cy': r['cy'], 'adj': r['adj'], 'mover': r['mover'], 'move': mv})
    if boards:
        VIZ_STATE['boards'] = boards

VIZ_HTML = r'''<!doctype html><html><head><meta charset="utf-8"><title>GNN training</title>
<style>
 body{margin:0;background:#0e1420;color:#dfe6ee;font:13px/1.5 Segoe UI,system-ui,sans-serif}
 h1{font-size:15px;margin:14px 16px 2px}
 #stats{margin:0 16px 10px;color:#9fb0c2}#stats b{color:#e7edf3}
 #grid{display:flex;flex-wrap:wrap;gap:10px;padding:8px 16px}
 .card{background:#151d2b;border:1px solid #223047;border-radius:8px;padding:6px}
 .card div{font-size:10px;color:#8a97a6;text-align:center;margin-top:2px}
 #log{white-space:pre-wrap;font:11px/1.45 Consolas,monospace;color:#8fa;background:#0b1018;
      margin:8px 16px;padding:8px;border-radius:6px;max-height:160px;overflow:auto}
</style></head><body>
<h1>GNN training — live self-play</h1><div id="stats">connecting…</div>
<div id="grid"></div><div id="log"></div>
<script>
const COL=['#c9d2dc','#7f8a99','#46a6ef','#ef5b52','#f0a63a','#8a5cf0','#3ac47d','#e05299'];
const SZ=190,PAD=16;
function board(b){
 const c=document.createElement('canvas');c.width=SZ;c.height=SZ;const x=c.getContext('2d');
 const P=i=>[PAD+b.cx[i]*(SZ-2*PAD),PAD+b.cy[i]*(SZ-2*PAD)];
 x.strokeStyle='rgba(150,170,190,.18)';x.lineWidth=1;
 for(let i=0;i<b.adj.length;i++)for(const j of b.adj[i])if(j>i){const a=P(i),d=P(j);x.beginPath();x.moveTo(a[0],a[1]);x.lineTo(d[0],d[1]);x.stroke();}
 if(b.move){const a=P(b.move[0]),d=P(b.move[1]);const fr=b.move.length>2?b.move[2]:1;
  x.strokeStyle='#5fd07a';x.fillStyle='#5fd07a';x.lineWidth=2;x.beginPath();x.moveTo(a[0],a[1]);x.lineTo(d[0],d[1]);x.stroke();
  const an=Math.atan2(d[1]-a[1],d[0]-a[0]);x.beginPath();x.moveTo(d[0],d[1]);
  x.lineTo(d[0]-8*Math.cos(an-.4),d[1]-8*Math.sin(an-.4));x.lineTo(d[0]-8*Math.cos(an+.4),d[1]-8*Math.sin(an+.4));x.closePath();x.fill();
  x.fillStyle='#bdf0cd';x.font='9px monospace';x.fillText('x'+fr,(a[0]+d[0])/2,(a[1]+d[1])/2-3);}
 for(let i=0;i<b.owner.length;i++){const p=P(i),o=b.owner[i];const r=3+Math.min(7,b.troops[i]/12);
  x.fillStyle=COL[o]||'#888';x.beginPath();x.arc(p[0],p[1],r,0,7);x.fill();
  x.strokeStyle='rgba(0,0,0,.45)';x.lineWidth=1;x.stroke();
  x.fillStyle='#0b1018';x.font='8px monospace';x.textAlign='center';x.fillText(Math.round(b.troops[i]),p[0],p[1]+3);}
 const w=document.createElement('div');w.className='card';w.appendChild(c);
 const cap=document.createElement('div');cap.textContent='mover '+b.mover+(b.move?'':' · waiting');w.appendChild(cap);return w;
}
async function tick(){try{
 const s=await fetch('/state').then(r=>r.json());const st=s.stats||{};
 document.getElementById('stats').innerHTML=
  'round <b>'+(st.round??'–')+'</b> · '+(st.elapsed??'–')+' · device <b>'+(st.device??'–')+'</b> · '
  +'buffer <b>'+(st.buffer??'–')+'</b> · loss <b>'+(st.loss??'–')+'</b> · '
  +'turtle '+(st.turtle??'–')+' rush '+(st.rush??'–')+' · gate <b>'+(st.gate??'–')+'</b> · '
  +'promotions <b>'+(st.promotions??0)+'</b> · fractions '+(st.fractions??'');
 const g=document.getElementById('grid');g.innerHTML='';for(const b of (s.boards||[]))g.appendChild(board(b));
 document.getElementById('log').textContent=(s.log||[]).join('\n');
}catch(e){document.getElementById('stats').textContent='waiting for training… ('+e+')';}}
setInterval(tick,1500);tick();
</script></body></html>'''

def start_viz_server(port):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass   # silence per-request logging
        def _send(self, body, ctype):
            self.send_response(200)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def do_GET(self):
            if self.path.startswith('/state'):
                self._send(json.dumps(VIZ_STATE).encode(), 'application/json')
            else:
                self._send(VIZ_HTML.encode(), 'text/html; charset=utf-8')
    srv = ThreadingHTTPServer(('127.0.0.1', port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--hours', type=float, default=5.0)
    ap.add_argument('--device', default=None)
    ap.add_argument('--games-per-round', type=int, default=150)
    ap.add_argument('--selfplay-budget', type=int, default=500, help='ms/move think budget during DATA '
                     'generation. Deliberately much larger than the gate/deploy budget (50ms): the '
                     'training target is "what did deep search conclude", and deeper search per dumped '
                     'decision is worth more than more shallow decisions — train on a stronger expert '
                     'than you deploy.')
    ap.add_argument('--gate-games', type=int, default=40)
    ap.add_argument('--gate-budget', type=int, default=50)
    ap.add_argument('--gate-threshold', type=float, default=0.56)
    ap.add_argument('--script-mix', type=float, default=0.3, help='fraction of data-generation games '
                     'played vs the scripted exploiter league (turtle/rush/farmer/econ/human-profile) '
                     'instead of net-vs-net — fills the blind spots pure self-play never visits')
    ap.add_argument('--explore-eps', type=float, default=0.25, help='Dirichlet root-noise weight during '
                     'data generation (0 disables). Never applied to the gate.')
    ap.add_argument('--explore-alpha', type=float, default=0.8)
    ap.add_argument('--explore-temp', type=float, default=1.0, help='visit-proportional move sampling '
                     'temperature for the first --explore-temp-moves decisions of each data-gen game')
    ap.add_argument('--explore-temp-moves', type=int, default=12)
    ap.add_argument('--keep-noop', type=float, default=0.1, help='keep-probability for noop-dominated '
                     'dumped decisions (~93%% of raw decisions are "sit still"; see arena flushDump)')
    ap.add_argument('--real-maps', type=float, default=0.5, help='fraction of ALL arena games (data-gen '
                     'and gates) played on the real game maps (tools/real_maps.json, 11-49 states) '
                     'instead of synthetic 16-state Delaunay boards')
    ap.add_argument('--long-cap', type=int, default=180, help='GAMECAP (sim seconds) for the long-game '
                     'slice of data generation and gates (0 disables long games entirely)')
    ap.add_argument('--long-mix', type=float, default=0.33, help='fraction of net-vs-net data-gen games '
                     'played long (late-game boards in-distribution)')
    ap.add_argument('--gate-long-mix', type=float, default=0.5, help='fraction of gate/anchor games played '
                     'long — no candidate ships on sprint performance alone')
    ap.add_argument('--anchor-mix', type=float, default=0.25, help='fraction of net-vs-net data-gen games '
                     'played against the ANCHOR net instead of self — best-response training plus '
                     'distillation of the champion lineage (both seats dump)')
    ap.add_argument('--midgame-mix', type=float, default=0.2, help='fraction of data-gen games started from '
                     'MID-GAME positions sampled from earlier rounds (dense late-game coverage)')
    ap.add_argument('--anchor', default=os.path.join(REPO_ROOT, 'src', 'web', 'gnn_policy.json'),
                     help='fixed shipped net a passing candidate must ALSO not clearly lose to '
                     '(anti-drift; empty string disables). Default: the shipped gnn_policy.json.')
    ap.add_argument('--anchor-floor', type=float, default=0.45, help='min win-rate vs the anchor to promote')
    ap.add_argument('--anchor-games', type=int, default=40)
    ap.add_argument('--no-value-cotrain', action='store_true',
                     help='keep the value net fixed instead of retraining it on dumped outcomes')
    ap.add_argument('--value-retrain-every', type=int, default=3, help='retrain the value net every N rounds')
    ap.add_argument('--value-buffer-cap', type=int, default=40000)
    ap.add_argument('--buffer-cap', type=int, default=8000, help='decisions kept — smaller than train_rl_torch.py '
                     'since each decision here carries a full board+adjacency, not just a 12-float vector')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--batch-size', type=int, default=64, help='much smaller than the flat-MLP trainer: each '
                     'example here runs its own graph-attention forward pass, not a shared matmul batch')
    ap.add_argument('--epochs-per-round', type=int, default=2)
    ap.add_argument('--embed-dim', type=int, default=24)
    ap.add_argument('--hops', type=int, default=2)
    ap.add_argument('--value-head', action='store_true', help='give FRESHLY-initialized nets a board-aware '
                     'GNN value head (vhead) trained on dumped outcomes — the planner leaf eval upgrade. '
                     'Warm-started checkpoints infer it from their JSON, so this only affects new nets.')
    ap.add_argument('--workers', type=int, default=max(1, os.cpu_count() or 1),
                     help='parallel Node arena processes for self-play/gate — this is the actual '
                     'bottleneck (single-threaded Node), not the GPU, so this is the main lever for '
                     'using more than 1 of your CPU cores. Defaults to all cores.')
    ap.add_argument('--viz-port', type=int, default=7500, help='port for the live training visualizer web page')
    ap.add_argument('--no-viz', action='store_true', help='disable the live visualizer web server')
    ap.add_argument('--out', default=None, help='path to write/resume the trained net '
                    '(default: tools/gnn_policy_new.json). Use a separate file to avoid touching the checkpoint.')
    args = ap.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device}  cuda_available={torch.cuda.is_available()}  cores={os.cpu_count()}  "
          f"workers={args.workers}", flush=True)
    # every arena subprocess (data-gen AND gates) inherits this via os.environ.copy()
    if args.real_maps > 0:
        os.environ['REAL_MAPS'] = str(args.real_maps)
        print(f"real-map mix: {args.real_maps:.0%} of arena games on tools/real_maps.json boards", flush=True)

    if not args.no_viz:
        try:
            start_viz_server(args.viz_port)
            print(f"live training visualizer: http://localhost:{args.viz_port}  (open it to watch self-play)", flush=True)
        except Exception as e:
            print(f"(viz server failed to start on :{args.viz_port}: {e} — training continues without it)", flush=True)
    viz_publish_stats(device=device, fractions=str(G.FRACTIONS), round=0, elapsed='0s',
                      buffer=0, loss='–', turtle='–', rush='–', gate='–', promotions=0)

    resume_path = os.path.abspath(args.out) if args.out else os.path.join(SCR, 'gnn_policy_new.json')
    if os.path.exists(resume_path):
        best = G.from_json(resume_path, device=device)
        if best.node_feats != G.NODE_FEATS:
            print(f"WARNING: {resume_path} has node_features={best.node_feats} but this build expects "
                  f"{G.NODE_FEATS} (v2 features). Ignoring the old checkpoint and starting a fresh net. "
                  f"Move/rename that file to keep it.", flush=True)
            best = G.GNNPolicyNet(embed_dim=args.embed_dim, hops=args.hops, value_head=args.value_head).to(device)
        elif best.move_feats != G.MOVE_FEATS:
            # the checkpoint predates unit-splitting (move_feats=3); its move head is a different shape,
            # so it can't be warm-started into the split-capable (move_feats=4) architecture. Start fresh.
            print(f"WARNING: {resume_path} has move_feats={best.move_feats} but this build expects "
                  f"{G.MOVE_FEATS} (unit-splitting). Ignoring the old checkpoint and starting a fresh "
                  f"split-capable net. Move/rename that file to keep it.", flush=True)
            best = G.GNNPolicyNet(embed_dim=args.embed_dim, hops=args.hops, value_head=args.value_head).to(device)
        else:
            print(f"warm-started from {resume_path}", flush=True)
    else:
        best = G.GNNPolicyNet(embed_dim=args.embed_dim, hops=args.hops, value_head=args.value_head).to(device)
        print(f"no {resume_path} found — starting from a freshly-initialized (untrained) GNN net. "
              f"Expect it to lose badly for a while; this is the honest starting point for a genuinely "
              f"new architecture, not a warm-start from the shipped flat-MLP net (the two aren't "
              f"weight-compatible).", flush=True)

    # value net: warm-start from the co-trained checkpoint when present, else the shipped net.
    # Co-training feeds on the `result` field the arena now stamps into every dumped decision.
    value_net_path = os.path.join(SCR, 'value_net.json')
    value_resume_path = os.path.join(SCR, 'value_gnn_new.json')
    value_warm_path = value_resume_path if os.path.exists(value_resume_path) else value_net_path
    value_theta = value_theta_from_json(value_warm_path) if os.path.exists(value_warm_path) else None
    value_cotrain = not args.no_value_cotrain and value_theta is not None
    value_json = V.vto_json(value_theta) if value_theta is not None else None
    if value_json is None:
        print("WARNING: no value_net.json found — arena leaf eval falls back to the hand heuristic alone "
              "(value co-training disabled)", flush=True)
    else:
        print(f"value net: warm-started from {value_warm_path}  co-train={value_cotrain}", flush=True)

    rng = np.random.default_rng(7)
    base_turtle = vs_script(best, 'turtle', rng, 15, device=device)
    base_rush = vs_script(best, 'rush', rng, 15, device=device)
    base_farmer = vs_script(best, 'farmer', rng, 15, device=device)
    print(f"baseline vs scripts: turtle {base_turtle:.2f} rush {base_rush:.2f} farmer {base_farmer:.2f}", flush=True)
    # The no-forget floors are for catching style-COLLAPSE, not for demanding perfection: once a
    # warm-started checkpoint saturates a screen (~1.0), an uncapped floor plus the 12-game screen's
    # variance skips nearly every gate (observed 2026-07-11: 10/10 rounds no-forget-FAIL after a
    # resume recomputed baselines from a strong checkpoint — one dropped screen game = "forgot").
    # Cap the floors so a saturated screen still tolerates normal sampling noise.
    base_turtle = min(base_turtle, 0.85)
    base_rush = min(base_rush, 0.85)
    base_farmer = min(base_farmer, 0.85)

    midgame_pool_path = os.path.join(SCR, '_midgame_pool.jsonl')
    dump_path = os.path.join(SCR, '_gnn_visits_dump.jsonl')
    cand_path = os.path.join(SCR, '_gnn_cand_net.json')
    best_path = os.path.join(SCR, '_gnn_best_net.json')
    selfplay_net_path = os.path.join(SCR, '_gnn_selfplay_net.json')

    explore = {'eps': args.explore_eps, 'alpha': args.explore_alpha,
               'temp': args.explore_temp, 'temp_moves': args.explore_temp_moves} if args.explore_eps > 0 else None
    buf = []
    vbuf = []
    t0 = time.time()
    rnd = 0
    promotions = 0
    gates = 0
    try:
        while time.time() - t0 < args.hours * 3600:
            rnd += 1
            if os.path.exists(dump_path):
                os.remove(dump_path)
            self_play_dump(best, value_json, args.games_per_round, args.selfplay_budget, dump_path,
                           selfplay_net_path, args.workers,
                           script_mix=args.script_mix, explore=explore, keep_noop=args.keep_noop,
                           long_cap=args.long_cap or None, long_mix=args.long_mix,
                           anchor_path=args.anchor if args.anchor_mix > 0 else None, anchor_mix=args.anchor_mix,
                           midgame_pool=midgame_pool_path, midgame_p=args.midgame_mix)
            new_records = load_dump(dump_path)
            # roll the mid-game position pool forward: sample this round's boards in, keep it bounded
            if args.midgame_mix > 0 and new_records:
                sample = new_records[:: max(1, len(new_records) // 300)][:300]
                with open(midgame_pool_path, 'a') as mp:
                    for r in sample:
                        mp.write(json.dumps(r) + '\n')
                lines = open(midgame_pool_path).readlines()
                if len(lines) > 4000:
                    open(midgame_pool_path, 'w').writelines(lines[-4000:])
            viz_publish_boards(dump_path)   # show the latest self-play positions on the live page
            buf.extend(new_records)
            if len(buf) > args.buffer_cap:
                buf = buf[-args.buffer_cap:]
            if value_cotrain:   # every dumped decision doubles as a Monte-Carlo value sample
                vbuf.extend((vfeats_of_record(r), r['result']) for r in new_records if 'result' in r)
                if len(vbuf) > args.value_buffer_cap:
                    vbuf = vbuf[-args.value_buffer_cap:]
            el = f"{time.time()-t0:.0f}s"
            viz_publish_stats(round=rnd, elapsed=el, buffer=len(buf), promotions=promotions)
            if len(buf) < args.batch_size:
                msg = f"r{rnd:3d} t={time.time()-t0:6.0f}s warmup buf={len(buf)} (+{len(new_records)})"
                print(msg, flush=True); viz_log(msg)
                continue

            cand, avg_loss = train_round(best, buf, device, args.lr, args.batch_size, args.epochs_per_round, rng)
            viz_publish_stats(loss=f"{avg_loss:.4f}")

            tt = vs_script(cand, 'turtle', rng, 12, device=device)
            rs = vs_script(cand, 'rush', rng, 12, device=device)
            fm = vs_script(cand, 'farmer', rng, 12, device=device)
            viz_publish_stats(turtle=f"{tt:.2f}", rush=f"{rs:.2f}", farmer=f"{fm:.2f}")
            if tt < base_turtle - 0.06 or rs < base_rush - 0.08 or fm < base_farmer - 0.08:
                msg = (f"r{rnd:3d} t={time.time()-t0:6.0f}s buf={len(buf)} loss={avg_loss:.4f} "
                       f"tt={tt:.2f} rs={rs:.2f} fm={fm:.2f} no-forget-FAIL (skip gate)")
                print(msg, flush=True); viz_log(msg); viz_publish_stats(gate='skip(no-forget)')
                continue

            wr = planner_gate(cand, best, value_json, args.gate_games, args.gate_budget, cand_path, best_path,
                              args.workers, long_cap=args.long_cap or None, long_mix=args.gate_long_mix)
            gates += 1
            promoted = False
            anchor_str = ""
            if wr is not None and wr >= args.gate_threshold:
                promoted = True
                if args.anchor and os.path.exists(args.anchor):   # anti-drift: also hold ground vs the SHIPPED net
                    awr = anchor_gate(cand, value_json, args.anchor, args.anchor_games, args.gate_budget, cand_path,
                                      args.workers, long_cap=args.long_cap or None, long_mix=args.gate_long_mix)
                    anchor_str = f" anchor={awr:.2f}" if awr is not None else " anchor=ERR"
                    if awr is not None and awr < args.anchor_floor:
                        promoted = False
                        anchor_str += "(VETO)"
                if promoted:
                    best = cand
                    promotions += 1
                    write_net(best, value_json, resume_path)

            # value co-training on the accumulated dumped outcomes — as the policy improves, the
            # value net keeps training on increasingly realistic planner self-play (the same
            # compounding-teacher rationale as train_rl_torch.py, minus its separate cheap-physics
            # data generation: the dump already carries outcomes).
            vtag = ""
            if value_cotrain and rnd % args.value_retrain_every == 0:
                if len(vbuf) >= 2000:
                    value_theta, sign_acc = train_value_round(value_theta, vbuf, rng)
                    value_json = V.vto_json(value_theta)
                    json.dump(value_json, open(value_resume_path, 'w'))
                    vtag = f" value(vbuf={len(vbuf)} sign-acc={sign_acc:.3f})"
                else:
                    vtag = f" value(warmup vbuf={len(vbuf)})"

            gate_str = f"{wr:.2f}" if wr is not None else "ERR"
            msg = (f"r{rnd:3d} t={time.time()-t0:6.0f}s buf={len(buf)} loss={avg_loss:.4f} "
                   f"tt={tt:.2f} rs={rs:.2f} fm={fm:.2f} gate={gate_str}{anchor_str} {'PROMOTED #%d' % promotions if promoted else 'kept'}{vtag}")
            print(msg, flush=True); viz_log(msg)
            viz_publish_stats(gate=gate_str, promotions=promotions)
    except KeyboardInterrupt:
        print("\ninterrupted — writing final state", flush=True)
    finally:
        cleanup_paths = [dump_path, cand_path, best_path, selfplay_net_path]
        # leftover per-job shards if interrupted mid-round (net-vs-net workers + one job per scripted opponent)
        cleanup_paths += [f"{dump_path}.w{i}" for i in range(args.workers + len(SCRIPT_OPPONENTS))]
        for p in cleanup_paths:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

    print(f"DONE rounds={rnd} gates={gates} promotions={promotions}", flush=True)
    write_net(best, value_json, resume_path)
    print(f"wrote {resume_path} — this is a NEW architecture (gnn-v1), not a drop-in replacement for "
          f"src/web/rl_policy.json's format. Only copy it over rl_policy.json if the gate log above "
          f"actually shows it beating the flat-MLP net; the live game reads POL.format and dispatches "
          f"correctly either way (see index.html's gnnEmbedBoard/gnnMoveScore).", flush=True)


if __name__ == '__main__':
    main()
