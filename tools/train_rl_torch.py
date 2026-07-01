#!/usr/bin/env python3
# GPU-trainable retraining loop for the State.io policy net, built for the Risk-style adjacency +
# connected-path movement rules (see FINDINGS.md "Next: retraining for Risk-style adjacency
# movement" for the design rationale). Run this on your own machine with a GPU — it's built to run
# unattended for hours and checkpoint safely; it never ships a worse net than it started with.
#
# What it does, each round:
#   1. Self-play `--games-per-round` games (current best net vs itself) through the REAL MCTS
#      planner (tools/selfplay_arena.js), capturing every decision's root visit-count distribution
#      (--dump-visits) — this is the signal the candidate net is trained to imitate: not "what won
#      the game" but "what did deep search conclude was best, and how confident was it."
#   2. Train a candidate (warm-started from best) via Adam, minimizing cross-entropy against that
#      visit distribution over a rolling buffer of captured decisions.
#   3. Cheap no-forget screen vs scripted turtle/rush opponents — skip the expensive real-planner
#      gate on an obvious regression.
#   4. REAL-PLANNER gate: candidate vs best, played out through the actual arena. Promote ONLY if
#      the candidate wins clearly (>= --gate-threshold). This is the same no-regression discipline
#      every other trainer in this directory uses — a long unattended run can't make things worse.
#   5. On promotion, export tools/policy_torch_new.json (loadable back into src/web/rl_policy.json's
#      schema — see policy_net_torch.to_json/from_json for the exact contract).
#
# Loss weighting, buffer size, learning rate, and the gate threshold below are reasonable STARTING
# POINTS, not tuned values — watch the logged loss/gate numbers over the first 10-20 rounds and
# adjust via the CLI flags rather than trusting these blindly, the same way every other tool in this
# directory was tuned empirically against the arena rather than derived analytically.
#
#   usage: python3 tools/train_rl_torch.py --hours 5 --device cuda
import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_rl as T
import policy_net_torch as P

SCR = os.path.dirname(os.path.abspath(__file__))     # tools/ — where nets and JSONL dumps live
REPO_ROOT = os.path.dirname(SCR)                       # repo root — `node tools/selfplay_arena.js` is relative to this


# ---------------- bridging PolicyNet <-> train_rl.py's (layers, noop) seat format ----------------
def net_to_layers(net):
    return [(lin.weight.detach().cpu().numpy().astype(np.float64),
              lin.bias.detach().cpu().numpy().astype(np.float64)) for lin in net.linears]


def vs_script(net, opp, rng, n=60, N=18):
    # Self-contained no-forget screen (deliberately NOT importing train_selfplay_loop.py — that
    # module mutates train_rl.ARCH to [12,64,64,1] as an import side-effect and has its own stale,
    # non-adjacency-aware move generator; reusing it here would silently break both this script's
    # architecture assumption and its legality guarantees).
    seat_net = ('net', net_to_layers(net), float(net.noop_bias.item()))
    seat_opp = (opp, None, None)
    w = 0
    for _ in range(n):
        Pn = int(rng.choice([2, 3]))
        ci = int(rng.integers(Pn))
        seats = [seat_net if p == ci else seat_opp for p in range(Pn)]
        res = T.play(seats, rng, N=N)
        if res[ci + 1] >= max(res.values()) - 1e-9:
            w += 1
    return w / n


# ---------------- arena plumbing (real MCTS planner, off-process via Node) ----------------
def write_net(net, value_json_path, path):
    j = P.to_json(net)
    if value_json_path:
        j['value'] = json.load(open(value_json_path))
    json.dump(j, open(path, 'w'))


def planner_gate(cand_net, best_net, value_json_path, games, budget, cand_path, best_path):
    write_net(cand_net, value_json_path, cand_path)
    write_net(best_net, value_json_path, best_path)
    try:
        out = subprocess.run(['node', 'tools/selfplay_arena.js', cand_path, best_path, str(games), str(budget)],
                              capture_output=True, text=True, cwd=REPO_ROOT, timeout=1800).stdout
    except Exception as e:
        print("  gate error:", e, flush=True)
        return None
    m = re.search(r'A won (\d+)/(\d+)', out)
    return (int(m.group(1)) / int(m.group(2))) if m else None


def self_play_dump(net, value_json_path, games, budget, dump_path, net_path):
    write_net(net, value_json_path, net_path)
    env = os.environ.copy()
    env['DUMP_VISITS'] = dump_path
    try:
        subprocess.run(['node', 'tools/selfplay_arena.js', net_path, net_path, str(games), str(budget)],
                        capture_output=True, text=True, cwd=REPO_ROOT, timeout=3600, env=env)
    except Exception as e:
        print("  self-play error:", e, flush=True)


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
                records.append(json.loads(line))
            except Exception:
                continue
    return records


# ---------------- distillation batch/loss ----------------
def make_batch(records, device):
    maxK = max(len(r['feats']) for r in records)
    B = len(records)
    X = np.zeros((B, maxK, P.ARCH[0]), dtype=np.float32)
    noop_mask = np.zeros((B, maxK), dtype=np.float32)
    valid_mask = np.zeros((B, maxK), dtype=np.float32)
    visits = np.zeros((B, maxK), dtype=np.float32)
    for bi, r in enumerate(records):
        for ki, f in enumerate(r['feats']):
            valid_mask[bi, ki] = 1.0
            if f is None:
                noop_mask[bi, ki] = 1.0   # no-op candidate: scored by noop_bias alone, never through the MLP
            else:
                X[bi, ki, :] = f
            visits[bi, ki] = r['visits'][ki]
    return (torch.tensor(X, device=device), torch.tensor(noop_mask, device=device),
            torch.tensor(valid_mask, device=device), torch.tensor(visits, device=device))


def distill_loss(net, X, noop_mask, valid_mask, visits):
    # cross-entropy of the net's own softmax-over-candidates against the real planner's empirical
    # visit distribution pi(a) ~ N(s,a) — the standard AlphaZero policy-target recipe, applied here
    # for the first time in this codebase (every prior phase optimized win-rate, never distilled the
    # search's own move distribution — see FINDINGS.md).
    B, K, Fdim = X.shape
    mlp = net(X.reshape(B * K, Fdim)).reshape(B, K)
    scores = torch.where(noop_mask > 0, net.noop_bias.expand(B, K), mlp)
    scores = scores.masked_fill(valid_mask == 0, -1e9)
    logp = F.log_softmax(scores, dim=1)
    target = visits / visits.sum(dim=1, keepdim=True).clamp_min(1e-9)
    return -(target * logp * valid_mask).sum(dim=1).mean()


def train_round(best, buf, device, lr, batch_size, epochs, rng):
    cand = P.PolicyNet(best.arch).to(device)   # match best's actual architecture, NOT policy_net_torch's module default
    cand.load_state_dict(best.state_dict())
    opt = torch.optim.Adam(cand.parameters(), lr=lr)
    total_loss, nsteps = 0.0, 0
    for _ in range(epochs):
        perm = rng.permutation(len(buf))
        for i in range(0, len(buf), batch_size):
            idx = perm[i:i + batch_size]
            batch = [buf[j] for j in idx]
            X, noop_mask, valid_mask, visits = make_batch(batch, device)
            loss = distill_loss(cand, X, noop_mask, valid_mask, visits)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cand.parameters(), 5.0)
            opt.step()
            total_loss += loss.item()
            nsteps += 1
    return cand, total_loss / max(1, nsteps)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--hours', type=float, default=5.0, help='wall-clock budget for the whole run')
    ap.add_argument('--device', default=None, help='cuda | cpu (default: auto-detect)')
    ap.add_argument('--games-per-round', type=int, default=200, help='self-play games generated per round')
    ap.add_argument('--selfplay-budget', type=int, default=50, help='ms/move think budget during self-play data generation')
    ap.add_argument('--gate-games', type=int, default=40, help='games played in the real-planner promotion gate')
    ap.add_argument('--gate-budget', type=int, default=50, help='ms/move think budget during the gate')
    ap.add_argument('--gate-threshold', type=float, default=0.56, help='candidate win-rate required to promote')
    ap.add_argument('--buffer-cap', type=int, default=20000, help='max decisions kept in the rolling training buffer')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--epochs-per-round', type=int, default=4)
    args = ap.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device}  cuda_available={torch.cuda.is_available()}  cores={os.cpu_count()}", flush=True)

    resume_path = os.path.join(SCR, 'policy_torch_new.json')
    warm_path = resume_path if os.path.exists(resume_path) else T.POLICY
    best = P.from_json(warm_path, device=device)
    print(f"warm-started from {warm_path}", flush=True)

    value_json_path = os.path.join(SCR, 'value_net.json')
    if not os.path.exists(value_json_path):
        value_json_path = None
        print("WARNING: no value_net.json found — arena leaf eval falls back to the hand heuristic alone", flush=True)

    rng = np.random.default_rng(7)
    base_turtle = vs_script(best, 'turtle', rng, 60)
    base_rush = vs_script(best, 'rush', rng, 60)
    print(f"baseline vs scripts: turtle {base_turtle:.2f} rush {base_rush:.2f}", flush=True)

    dump_path = os.path.join(SCR, '_visits_dump.jsonl')
    cand_path = os.path.join(SCR, '_torch_cand_net.json')
    best_path = os.path.join(SCR, '_torch_best_net.json')
    selfplay_net_path = os.path.join(SCR, '_torch_selfplay_net.json')

    buf = []
    t0 = time.time()
    rnd = 0
    promotions = 0
    gates = 0
    try:
        while time.time() - t0 < args.hours * 3600:
            rnd += 1
            # 1) self-play, capture the real planner's own visit-count distributions
            if os.path.exists(dump_path):
                os.remove(dump_path)
            self_play_dump(best, value_json_path, args.games_per_round, args.selfplay_budget, dump_path, selfplay_net_path)
            new_records = load_dump(dump_path)
            buf.extend(new_records)
            if len(buf) > args.buffer_cap:
                buf = buf[-args.buffer_cap:]
            if len(buf) < args.batch_size:
                print(f"r{rnd:3d} t={time.time()-t0:6.0f}s warmup buf={len(buf)} (+{len(new_records)})", flush=True)
                continue

            # 2) train a candidate on the buffer
            cand, avg_loss = train_round(best, buf, device, args.lr, args.batch_size, args.epochs_per_round, rng)

            # 3) cheap no-forget screen (skip the expensive real-planner gate on an obvious regression)
            tt = vs_script(cand, 'turtle', rng, 40)
            rs = vs_script(cand, 'rush', rng, 40)
            if tt < base_turtle - 0.06 or rs < base_rush - 0.08:
                print(f"r{rnd:3d} t={time.time()-t0:6.0f}s buf={len(buf)} loss={avg_loss:.4f} "
                      f"tt={tt:.2f} rs={rs:.2f} no-forget-FAIL (skip gate)", flush=True)
                continue

            # 4) REAL-PLANNER gate
            wr = planner_gate(cand, best, value_json_path, args.gate_games, args.gate_budget, cand_path, best_path)
            gates += 1
            promoted = False
            if wr is not None and wr >= args.gate_threshold:
                best = cand
                promotions += 1
                promoted = True
                write_net(best, value_json_path, resume_path)

            gate_str = f"{wr:.2f}" if wr is not None else "ERR"
            print(f"r{rnd:3d} t={time.time()-t0:6.0f}s buf={len(buf)} loss={avg_loss:.4f} "
                  f"tt={tt:.2f} rs={rs:.2f} gate={gate_str} {'PROMOTED #%d' % promotions if promoted else 'kept'}",
                  flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted — writing final state", flush=True)
    finally:
        for p in (dump_path, cand_path, best_path, selfplay_net_path):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

    print(f"DONE rounds={rnd} gates={gates} promotions={promotions}", flush=True)
    ft = vs_script(best, 'turtle', rng, 80)
    fr = vs_script(best, 'rush', rng, 80)
    print(f"final best vs scripts: turtle {ft:.2f}(base {base_turtle:.2f}) rush {fr:.2f}(base {base_rush:.2f})", flush=True)
    write_net(best, value_json_path, resume_path)
    print(f"wrote {resume_path} — copy to src/web/rl_policy.json to ship (after your own review; "
          f"this script never overwrites the shipped net automatically)", flush=True)


if __name__ == '__main__':
    main()
