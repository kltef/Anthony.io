#!/usr/bin/env python3
# Benchmark for the three fixes applied to the training pipeline.
# Run from the repo root:
#   python3 tools/bench_training.py
#   python3 tools/bench_training.py --device cuda
#
# Prints wall-clock times for each component so you can see where your machine
# actually spends time and project the end-to-end round improvement.
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_rl as T
import gnn_net_torch as G
import train_gnn_torch as GNN

REPS = 200   # decisions per choose_gnn bench
N_NODES = 18


# ── helpers ──────────────────────────────────────────────────────────────────

def make_net(device):
    net = G.GNNPolicyNet().to(device)
    net.eval()
    return net


def make_fake_game(rng):
    g = T.new_game(N_NODES, 2, rng)
    return g


def make_fake_records(n, rng):
    """Synthetic records that look like selfplay_arena JSONL output."""
    records = []
    for _ in range(n):
        owner = rng.integers(0, 3, size=N_NODES).tolist()
        troops = rng.integers(1, 30, size=N_NODES).tolist()
        adj = [[j for j in range(N_NODES) if j != i and abs(j - i) <= 3]
               for i in range(N_NODES)]
        cx = rng.uniform(0, 1, N_NODES).tolist()
        cy = rng.uniform(0, 1, N_NODES).tolist()
        mover = 1
        n_moves = rng.integers(2, 6)
        moves = [None] + [[int(rng.integers(N_NODES)), int(rng.integers(N_NODES))]
                           for _ in range(n_moves - 1)]
        visits = rng.dirichlet(np.ones(n_moves)).tolist()
        records.append({
            'owner': owner, 'troops': troops, 'adj': adj,
            'armies': [], 'mover': mover,
            'cx': cx, 'cy': cy,
            'rlDiag': 1.414,
            'moves': moves, 'visits': visits,
        })
    return records


# ── benchmark 1: choose_gnn (old loop vs new vectorized) ─────────────────────

def _choose_gnn_old(net, g, owner, device):
    """Original per-source-node version (before vectorization)."""
    o = g['owner']
    t = g['troops']
    srcs = np.where((o == owner) & (t >= 5))[0]
    if len(srcs) == 0:
        return None
    with torch.no_grad():
        h = GNN.board_embeddings(net, o.tolist(), t.tolist(), g['adj'], g['armies'], owner, device)
        FN = G.FN
        best = float(net.noop_bias.item())
        mv = None
        for si in srcs:
            tgts = T.legal_targets(g, int(si), owner)
            if len(tgts) == 0:
                continue
            st = float(t[si])
            pairs = torch.tensor([[si, tj] for tj in tgts], dtype=torch.long, device=device)
            move_x = torch.tensor(
                [[(st - float(t[tj])) / FN,
                  float(np.hypot(g['pos'][tj, 0] - g['pos'][si, 0],
                                 g['pos'][tj, 1] - g['pos'][si, 1])) / g['rlDiag'],
                  1.0 if st > t[tj] else 0.0] for tj in tgts],
                dtype=torch.float32, device=device)
            scores = net.move_scores(h, pairs, move_x)
            j = int(torch.argmax(scores).item())
            if scores[j].item() > best:
                best = scores[j].item()
                mv = (int(si), int(tgts[j]))
    return mv


def bench_choose_gnn(net, device, rng):
    game = make_fake_game(rng)

    # warm-up
    for _ in range(10):
        _choose_gnn_old(net, game, 1, device)
        GNN.choose_gnn(net, game, 1, device)
    if device == 'cuda':
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(REPS):
        _choose_gnn_old(net, game, 1, device)
    if device == 'cuda':
        torch.cuda.synchronize()
    old_ms = (time.perf_counter() - t0) * 1000 / REPS

    t0 = time.perf_counter()
    for _ in range(REPS):
        GNN.choose_gnn(net, game, 1, device)
    if device == 'cuda':
        torch.cuda.synchronize()
    new_ms = (time.perf_counter() - t0) * 1000 / REPS

    print(f"\n── choose_gnn ({REPS} reps, device={device}) ──────────────────")
    print(f"  old (per-source loop):  {old_ms:.3f} ms/decision")
    print(f"  new (vectorized):       {new_ms:.3f} ms/decision")
    print(f"  speedup:                {old_ms/new_ms:.2f}×")
    return old_ms, new_ms


# ── benchmark 2: build_batch (old recompute vs new pre-computed stacks) ───────

def _build_batch_old(records, device):
    """build_batch without pre-computed arrays — recomputes every call."""
    B = len(records)
    N = len(records[0]['owner'])
    FN = G.FN
    node_x = torch.zeros(B, N, G.NODE_FEATS, dtype=torch.float32)
    adj_mask = torch.zeros(B, N, N, dtype=torch.float32)
    for b, r in enumerate(records):
        inc_mine, inc_enemy = GNN.incoming_from_armies(r['armies'], N, r['mover'])
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
                noop_mask[b, k] = 1.0
                continue
            si, ti = mv
            src_idx[b, k] = si
            tgt_idx[b, k] = ti
            st, tt = r['troops'][si], r['troops'][ti]
            dist = float(np.hypot(r['cx'][ti] - r['cx'][si], r['cy'][ti] - r['cy'][si])) / r['rlDiag']
            move_x[b, k, 0] = (st - tt) / G.FN
            move_x[b, k, 1] = dist
            move_x[b, k, 2] = 1.0 if st > tt else 0.0
    return tuple(t.to(device) for t in
                 (node_x, adj_mask, src_idx, tgt_idx, move_x, noop_mask, valid_mask, visits))


BATCH_REPS = 500
BATCH_SIZE = 64


def bench_build_batch(device, rng):
    raw = make_fake_records(BATCH_SIZE, rng)
    precomputed = make_fake_records(BATCH_SIZE, rng)
    for r in precomputed:
        GNN._precompute_record(r)

    # warm-up
    for _ in range(5):
        _build_batch_old(raw, device)
        GNN.build_batch(precomputed, device)

    t0 = time.perf_counter()
    for _ in range(BATCH_REPS):
        _build_batch_old(raw, device)
    old_ms = (time.perf_counter() - t0) * 1000 / BATCH_REPS

    t0 = time.perf_counter()
    for _ in range(BATCH_REPS):
        GNN.build_batch(precomputed, device)
    new_ms = (time.perf_counter() - t0) * 1000 / BATCH_REPS

    print(f"\n── build_batch (batch={BATCH_SIZE}, {BATCH_REPS} reps) ────────")
    print(f"  old (recompute each call): {old_ms:.3f} ms/batch")
    print(f"  new (pre-computed stacks): {new_ms:.3f} ms/batch")
    print(f"  speedup:                   {old_ms/new_ms:.2f}×")

    # project savings over a full training round
    steps_per_round = 8000 // BATCH_SIZE * 2   # buf_cap=8000, epochs=2
    saved_s = (old_ms - new_ms) * steps_per_round / 1000
    print(f"  projected saving/round ({steps_per_round} steps): {saved_s:.1f}s")
    return old_ms, new_ms


# ── benchmark 3: self-play parallelism (dry-run, no Node required) ───────────

def bench_selfplay_parallelism(cores):
    print(f"\n── self-play parallelism projection (cores={cores}) ─────────")
    # Typical single-worker time from FINDINGS.md §5.8: ~90s for 200 games at 50ms/move
    single_worker_s = 90.0
    for w in [1, 2, 4, cores]:
        if w > cores:
            continue
        projected = single_worker_s / w
        print(f"  {w:2d} worker(s): ~{projected:.0f}s  ({single_worker_s/projected:.1f}× vs baseline)")
    print(f"  (actual time will differ; run one round and read the t= column to calibrate)")


# ── round-time projection ─────────────────────────────────────────────────────

def project_round_time(choose_old_ms, choose_new_ms, batch_old_ms, batch_new_ms, cores):
    print(f"\n── projected round time (cores={cores}) ─────────────────────")

    # Self-play: dominates; assume 90s baseline (single worker, 200 games × 50ms/move)
    sp_old = 90.0
    sp_new = sp_old / max(1, cores)

    # vs_script no-forget screen: 12 games × ~320 decisions/game × ms/decision / 1000
    decisions_per_game = 320
    screen_games = 12
    screen_old = screen_games * decisions_per_game * choose_old_ms / 1000
    screen_new = screen_games * decisions_per_game * choose_new_ms / 1000

    # build_batch during train_round: 8000/64 × 2 epochs batches
    steps = 8000 // 64 * 2
    bb_old = steps * batch_old_ms / 1000
    bb_new = steps * batch_new_ms / 1000

    # gate: ~40 games, same parallelism as self-play, ~18s baseline
    gate_old = 18.0
    gate_new = gate_old / max(1, cores)

    total_old = sp_old + screen_old + bb_old + gate_old
    total_new = sp_new + screen_new + bb_new + gate_new

    rows = [
        ("self-play (200 games)", sp_old, sp_new),
        ("vs_script screen",      screen_old, screen_new),
        ("build_batch",           bb_old, bb_new),
        ("gate (40 games)",       gate_old, gate_new),
    ]
    print(f"  {'component':<26} {'before':>8}  {'after':>8}")
    print(f"  {'-'*26} {'-'*8}  {'-'*8}")
    for label, old, new in rows:
        print(f"  {label:<26} {old:>7.1f}s  {new:>7.1f}s")
    print(f"  {'-'*26} {'-'*8}  {'-'*8}")
    print(f"  {'TOTAL / round':<26} {total_old:>7.1f}s  {total_new:>7.1f}s")
    print(f"  {'speedup':<26} {total_old/total_new:>7.2f}×")

    hours_budget = 10.0
    rounds_old = int(hours_budget * 3600 / total_old)
    rounds_new = int(hours_budget * 3600 / total_new)
    print(f"\n  rounds in {hours_budget:.0f}h: {rounds_old} → {rounds_new}  "
          f"({rounds_new - rounds_old:+d} extra promotion opportunities)")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default=None, help='cuda | cpu (default: auto)')
    args = ap.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    cores = os.cpu_count() or 1
    rng = np.random.default_rng(42)

    print(f"device={device}  cores={cores}  torch={torch.__version__}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    net = make_net(device)

    choose_old, choose_new = bench_choose_gnn(net, device, rng)
    batch_old, batch_new   = bench_build_batch(device, rng)
    bench_selfplay_parallelism(cores)
    project_round_time(choose_old, choose_new, batch_old, batch_new, cores)


if __name__ == '__main__':
    main()
