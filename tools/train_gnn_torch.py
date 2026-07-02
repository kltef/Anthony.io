#!/usr/bin/env python3
# GPU training loop for the board-aware (GNN) policy net — FINDINGS.md future-path #1, trained with
# the same real-planner visit-count distillation as tools/train_rl_torch.py (future-path #3), and
# meant to be run for as long as you can afford (future-path #2, "deep self-play at scale") since a
# structured net is exactly the thing more self-play data has somewhere to go, per FINDINGS.md.
#
# Same round structure and same no-regression posture as train_rl_torch.py — see that file's header
# for the full rationale; this one only differs in HOW a candidate is trained: node embeddings are
# computed from the raw board (owner/troops/adjacency/armies) via graph-attention, not read off a
# flat per-move feature vector. The training loop batches decisions that share the same node-count N
# (see train_round's by_n bucketing) and runs a single vectorized forward pass per batch via
# node_embeddings_batched() + move_scores_batched() — numerically verified equivalent to the
# per-decision reference path (see tools/test_gnn_batch_equiv.py).
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
    #
    # All (src, tgt) pairs are collected in a single Python pass, then scored with ONE call to
    # move_scores() and ONE .item() sync — previously there was one forward pass + two .item() syncs
    # per source node, which serialized the GPU and added ~2ms of kernel-launch overhead per decision.
    o = g['owner']
    t = g['troops']
    srcs = np.where((o == owner) & (t >= 5))[0]
    if len(srcs) == 0:
        return None
    FN = G.FN
    all_pairs = []    # (si, tj) int tuples — flat list across all sources
    all_move_x = []   # parallel move-feature rows
    for si in srcs:
        tgts = T.legal_targets(g, int(si), owner)
        st = float(t[si])
        for tj in tgts:
            tt = float(t[tj])
            dist = float(np.hypot(g['pos'][tj, 0] - g['pos'][si, 0],
                                  g['pos'][tj, 1] - g['pos'][si, 1])) / g['rlDiag']
            all_pairs.append((int(si), int(tj)))
            all_move_x.append([(st - tt) / FN, dist, 1.0 if st > tt else 0.0])
    if not all_pairs:
        return None
    with torch.no_grad():
        h = board_embeddings(net, o.tolist(), t.tolist(), g['adj'], g['armies'], owner, device)
        pairs_t = torch.tensor(all_pairs, dtype=torch.long, device=device)
        move_x_t = torch.tensor(all_move_x, dtype=torch.float32, device=device)
        scores = net.move_scores(h, pairs_t, move_x_t)
        best_idx = int(torch.argmax(scores).item())   # single CPU-GPU sync point
        if scores[best_idx].item() > float(net.noop_bias.item()):
            return all_pairs[best_idx]
    return None


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
                    else:
                        mv = T.heuristic(g, p)
                    if mv:
                        T.send(g, mv[0], mv[1])
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


def planner_gate(cand_net, best_net, value_json, games, budget, cand_path, best_path, workers=1):
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
    with ThreadPoolExecutor(max_workers=len(argvs)) as ex:
        outs = list(ex.map(lambda a: _run_arena(a, timeout=1800), argvs))
    total_a, total_n = 0, 0
    for out in outs:
        m = re.search(r'A won (\d+)/(\d+)', out)
        if m:
            total_a += int(m.group(1))
            total_n += int(m.group(2))
    return (total_a / total_n) if total_n else None


def self_play_dump(net, value_json, games, budget, dump_path, net_path, workers=1):
    write_net(net, value_json, net_path)
    per_worker = _split_games(games, max(1, workers))
    if len(per_worker) == 1 or workers <= 1:
        env = os.environ.copy()
        env['DUMP_VISITS'] = dump_path
        _run_arena(['node', 'tools/selfplay_arena.js', net_path, net_path, str(games), str(budget)], env)
        return
    worker_dumps = [f"{dump_path}.w{i}" for i in range(len(per_worker))]

    def run_one(i):
        if per_worker[i] <= 0:
            return
        env = os.environ.copy()
        env['DUMP_VISITS'] = worker_dumps[i]
        _run_arena(['node', 'tools/selfplay_arena.js', net_path, net_path, str(per_worker[i]), str(budget)], env)

    with ThreadPoolExecutor(max_workers=len(per_worker)) as ex:
        list(ex.map(run_one, range(len(per_worker))))
    with open(dump_path, 'w') as out:
        for wp in worker_dumps:
            if os.path.exists(wp):
                out.write(open(wp).read())
                try:
                    os.remove(wp)
                except OSError:
                    pass


def _precompute_record(r):
    # Called once per record when it enters the buffer. Stores node features and the dense adjacency
    # mask as numpy arrays so build_batch() can just stack them — avoids re-running the Python
    # node_features() loop and the O(N*deg) adj inner loop on every batch that touches this record.
    N = len(r['owner'])
    inc_mine, inc_enemy = incoming_from_armies(r['armies'], N, r['mover'])
    r['_nf'] = np.array(G.node_features(r['owner'], r['troops'], r['adj'],
                                         inc_mine, inc_enemy, r['mover']), dtype=np.float32)
    adj_mask = np.zeros((N, N), dtype=np.float32)
    for i, nbrs in enumerate(r['adj']):
        for j in nbrs:
            adj_mask[i, j] = 1.0
    r['_adj_mask'] = adj_mask


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
                    _precompute_record(r)
                    records.append(r)
            except Exception:
                continue
    return records


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
        si, ti = mv
        st, tt = rec['troops'][si], rec['troops'][ti]
        dist = float(np.hypot(rec['cx'][ti] - rec['cx'][si], rec['cy'][ti] - rec['cy'][si])) / rec['rlDiag']
        move_x = torch.tensor([(st - tt) / FN, dist, 1.0 if st > tt else 0.0], dtype=torch.float32, device=device)
        pair = torch.tensor([[si, ti]], dtype=torch.long, device=device)
        scores.append(net.move_scores(h, pair, move_x.unsqueeze(0)).squeeze())
    scores = torch.stack(scores)
    visits = torch.tensor(rec['visits'], dtype=torch.float32, device=device)
    target = visits / visits.sum().clamp_min(1e-9)
    logp = F.log_softmax(scores, dim=0)
    return -(target * logp).sum()


def build_batch(records, device):
    # Pack a list of records that ALL share the same board node-count N into dense tensors. Candidates
    # are padded to the batch's max count and masked out in the loss, so per-decision softmax is
    # reproduced exactly. Grouping by N (see train_round) guarantees the uniform-N precondition.
    #
    # Node features and adj masks are read from r['_nf'] / r['_adj_mask'] pre-computed at load time
    # by _precompute_record() — no Python loops here, just numpy stacks → single torch conversion.
    B = len(records)
    N = len(records[0]['owner'])
    FN = G.FN
    node_x = torch.as_tensor(np.stack([r['_nf'] for r in records]), dtype=torch.float32)
    adj_mask = torch.as_tensor(np.stack([r['_adj_mask'] for r in records]), dtype=torch.float32)
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
            si, ti = mv
            src_idx[b, k] = si
            tgt_idx[b, k] = ti
            st, tt = r['troops'][si], r['troops'][ti]
            dist = float(np.hypot(r['cx'][ti] - r['cx'][si], r['cy'][ti] - r['cy'][si])) / r['rlDiag']
            move_x[b, k, 0] = (st - tt) / FN
            move_x[b, k, 1] = dist
            move_x[b, k, 2] = 1.0 if st > tt else 0.0
    return tuple(t.to(device) for t in
                 (node_x, adj_mask, src_idx, tgt_idx, move_x, noop_mask, valid_mask, visits))


def batched_loss(net, batch):
    # Mean cross-entropy over a batch of decisions — numerically identical to averaging decision_loss()
    # over the same records (verified in tools/test_gnn_batch_equiv.py, forward AND gradients).
    node_x, adj_mask, src_idx, tgt_idx, move_x, noop_mask, valid_mask, visits = batch
    h = net.node_embeddings_batched(node_x, adj_mask)                       # (B, N, dim)
    scores = net.move_scores_batched(h, src_idx, tgt_idx, move_x)           # (B, M)
    scores = torch.where(noop_mask > 0, net.noop_bias.to(scores.dtype), scores)
    scores = scores.masked_fill(valid_mask == 0, -1e9)                      # padded candidates -> ~0 weight
    logp = F.log_softmax(scores, dim=1)
    target = visits / visits.sum(dim=1, keepdim=True).clamp_min(1e-9)
    return -(target * logp * valid_mask).sum(dim=1).mean()


def train_round(best, buf, device, lr, batch_size, epochs, rng):
    cand = G.GNNPolicyNet(best.node_feats, best.embed_dim, best.hops, best.move_feats, best.head_dim).to(device)
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--hours', type=float, default=5.0)
    ap.add_argument('--device', default=None)
    ap.add_argument('--games-per-round', type=int, default=150)
    ap.add_argument('--selfplay-budget', type=int, default=50)
    ap.add_argument('--gate-games', type=int, default=40)
    ap.add_argument('--gate-budget', type=int, default=50)
    ap.add_argument('--gate-threshold', type=float, default=0.56)
    ap.add_argument('--buffer-cap', type=int, default=8000, help='decisions kept — smaller than train_rl_torch.py '
                     'since each decision here carries a full board+adjacency, not just a 12-float vector')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--batch-size', type=int, default=64, help='much smaller than the flat-MLP trainer: each '
                     'example here runs its own graph-attention forward pass, not a shared matmul batch')
    ap.add_argument('--epochs-per-round', type=int, default=2)
    ap.add_argument('--embed-dim', type=int, default=24)
    ap.add_argument('--hops', type=int, default=2)
    ap.add_argument('--workers', type=int, default=max(1, os.cpu_count() or 1),
                     help='parallel Node arena processes for self-play/gate — this is the actual '
                     'bottleneck (single-threaded Node), not the GPU, so this is the main lever for '
                     'using more than 1 of your CPU cores. Defaults to all cores.')
    args = ap.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device}  cuda_available={torch.cuda.is_available()}  cores={os.cpu_count()}  "
          f"workers={args.workers}", flush=True)

    resume_path = os.path.join(SCR, 'gnn_policy_new.json')
    if os.path.exists(resume_path):
        best = G.from_json(resume_path, device=device)
        print(f"warm-started from {resume_path}", flush=True)
    else:
        best = G.GNNPolicyNet(embed_dim=args.embed_dim, hops=args.hops).to(device)
        print(f"no {resume_path} found — starting from a freshly-initialized (untrained) GNN net. "
              f"Expect it to lose badly for a while; this is the honest starting point for a genuinely "
              f"new architecture, not a warm-start from the shipped flat-MLP net (the two aren't "
              f"weight-compatible).", flush=True)

    value_net_path = os.path.join(SCR, 'value_net.json')
    value_json = json.load(open(value_net_path)) if os.path.exists(value_net_path) else None
    if value_json is None:
        print("WARNING: no value_net.json found — arena leaf eval falls back to the hand heuristic alone", flush=True)

    rng = np.random.default_rng(7)
    base_turtle = vs_script(best, 'turtle', rng, 15, device=device)
    base_rush = vs_script(best, 'rush', rng, 15, device=device)
    print(f"baseline vs scripts: turtle {base_turtle:.2f} rush {base_rush:.2f}", flush=True)

    dump_path = os.path.join(SCR, '_gnn_visits_dump.jsonl')
    cand_path = os.path.join(SCR, '_gnn_cand_net.json')
    best_path = os.path.join(SCR, '_gnn_best_net.json')
    selfplay_net_path = os.path.join(SCR, '_gnn_selfplay_net.json')

    buf = []
    t0 = time.time()
    rnd = 0
    promotions = 0
    gates = 0
    try:
        while time.time() - t0 < args.hours * 3600:
            rnd += 1
            if os.path.exists(dump_path):
                os.remove(dump_path)
            self_play_dump(best, value_json, args.games_per_round, args.selfplay_budget, dump_path, selfplay_net_path, args.workers)
            new_records = load_dump(dump_path)
            buf.extend(new_records)
            if len(buf) > args.buffer_cap:
                buf = buf[-args.buffer_cap:]
            if len(buf) < args.batch_size:
                print(f"r{rnd:3d} t={time.time()-t0:6.0f}s warmup buf={len(buf)} (+{len(new_records)})", flush=True)
                continue

            cand, avg_loss = train_round(best, buf, device, args.lr, args.batch_size, args.epochs_per_round, rng)

            tt = vs_script(cand, 'turtle', rng, 12, device=device)
            rs = vs_script(cand, 'rush', rng, 12, device=device)
            if tt < base_turtle - 0.06 or rs < base_rush - 0.08:
                print(f"r{rnd:3d} t={time.time()-t0:6.0f}s buf={len(buf)} loss={avg_loss:.4f} "
                      f"tt={tt:.2f} rs={rs:.2f} no-forget-FAIL (skip gate)", flush=True)
                continue

            wr = planner_gate(cand, best, value_json, args.gate_games, args.gate_budget, cand_path, best_path, args.workers)
            gates += 1
            promoted = False
            if wr is not None and wr >= args.gate_threshold:
                best = cand
                promotions += 1
                promoted = True
                write_net(best, value_json, resume_path)

            gate_str = f"{wr:.2f}" if wr is not None else "ERR"
            print(f"r{rnd:3d} t={time.time()-t0:6.0f}s buf={len(buf)} loss={avg_loss:.4f} "
                  f"tt={tt:.2f} rs={rs:.2f} gate={gate_str} {'PROMOTED #%d' % promotions if promoted else 'kept'}",
                  flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted — writing final state", flush=True)
    finally:
        cleanup_paths = [dump_path, cand_path, best_path, selfplay_net_path]
        cleanup_paths += [f"{dump_path}.w{i}" for i in range(args.workers)]   # leftover per-worker shards if interrupted mid-round
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
