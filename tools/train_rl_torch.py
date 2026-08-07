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
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_rl as T
import train_value as V   # pure functions only (vfwd/vbwd/vfeats/result/vto_json) — does NOT mutate T.ARCH on import
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
def write_net(net, value_json, path):
    # value_json: an already-loaded dict (kept in-memory in main() so co-training updates propagate
    # to every subsequent gate/self-play call without a redundant re-read from disk), or None.
    j = P.to_json(net)
    if value_json:
        j['value'] = value_json
    json.dump(j, open(path, 'w'))


def planner_gate(cand_net, best_net, value_json, games, budget, cand_path, best_path):
    write_net(cand_net, value_json, cand_path)
    write_net(best_net, value_json, best_path)
    try:
        out = subprocess.run(['node', 'tools/selfplay_arena.js', cand_path, best_path, str(games), str(budget)],
                              capture_output=True, text=True, cwd=REPO_ROOT, timeout=1800).stdout
    except Exception as e:
        print("  gate error:", e, flush=True)
        return None
    m = re.search(r'A won (\d+)/(\d+)', out)
    return (int(m.group(1)) / int(m.group(2))) if m else None


def self_play_dump(net, value_json, games, budget, dump_path, net_path):
    write_net(net, value_json, net_path)
    env = os.environ.copy()
    env['DUMP_VISITS'] = dump_path
    try:
        subprocess.run(['node', 'tools/selfplay_arena.js', net_path, net_path, str(games), str(budget)],
                        capture_output=True, text=True, cwd=REPO_ROOT, timeout=3600, env=env)
    except Exception as e:
        print("  self-play error:", e, flush=True)


# ---------------- value-net co-training (fixed-VARCH Monte-Carlo regression, same recipe as
# train_value.py, just fed self-play from the CURRENT best policy instead of a static one) ----------------
def value_theta_from_json(path):
    p = json.load(open(path))
    parts = []
    for l in p['layers']:
        parts.append(np.array(l['W'], dtype=np.float64).ravel())
        parts.append(np.array(l['b'], dtype=np.float64).ravel())
    return np.concatenate(parts)


def gen_value_data(args):
    # standalone copy of train_value.py's gen_value_data() that takes (layers, noop) directly instead
    # of a flat SNES theta — avoids depending on train_rl.unpack()/train_rl.ARCH, which other trainers
    # in this directory monkeypatch as an import side-effect (see the vs_script() comment above for
    # why that's a real, previously-hit bug, not just caution).
    layers, noop, seed = args
    rng = np.random.default_rng(seed)
    N = 18
    Pn = int(rng.choice([2, 3, 4]))
    g = T.new_game(N, Pn, rng)
    timers = {p: rng.uniform(0.2, 1.0) for p in range(1, Pn + 1)}
    caps = []
    t = 0.0
    nextcap = rng.uniform(2.0, 5.0)
    while t < 75.0:
        T.step(g, 0.5)
        t += 0.5
        for p in range(1, Pn + 1):
            timers[p] -= 0.5
            if timers[p] <= 0:
                timers[p] = rng.uniform(0.6, 1.0)
                mv = T.choose(layers, noop, g, p)
                if mv:
                    T.send(g, mv[0], mv[1])
        if t >= nextcap:
            nextcap = t + rng.uniform(3.0, 7.0)
            for p in range(1, Pn + 1):
                if (g['owner'] == p).any():
                    caps.append((p, V.vfeats(g, p)))
        if len(T.alive_owners(g)) <= 1:
            break
    return [(f, V.result(g, p)) for (p, f) in caps]


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
    ap.add_argument('--gate-budget', type=int, default=240, help='ms/move during the gate. 240 == the '
                     'shipped Nightmare tier (LEVELS workerMs); 50 measured in a regime the real game '
                     'never runs in.')
    ap.add_argument('--real-maps', type=float, default=0.5, help='fraction of arena games played on the '
                     'real game maps (tools/real_maps.json, 11-49 states) instead of synthetic boards')
    ap.add_argument('--classic-rules', action='store_true', help='use the OLD classic physics instead of '
                     'the shipped depth rules (Lanchester/terrain/supply/attrition). Old measurements only.')
    ap.add_argument('--features', type=int, default=12, choices=(12, 24), help='policy input width. 24 '
                     'selects the EXTENDED vector (board-derived features + split-send support, see '
                     'train_rl.feats24 and rlMoveFeats/polMoveFeats in src/web/index.html). The two are '
                     'not weight-compatible, so --features 24 starts from a FRESH net unless a '
                     '24-feature checkpoint exists; run tools/test_feats24_parity.py first.')
    ap.add_argument('--hidden', type=int, default=64, help='hidden width for both trunk layers')
    ap.add_argument('--gate-threshold', type=float, default=0.56, help='candidate win-rate required to promote')
    ap.add_argument('--buffer-cap', type=int, default=20000, help='max decisions kept in the rolling training buffer')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--epochs-per-round', type=int, default=4)
    ap.add_argument('--no-value-cotrain', action='store_true',
                     help='keep the value net fixed (old behavior) instead of retraining it alongside the policy')
    ap.add_argument('--value-retrain-every', type=int, default=3, help='retrain the value net every N rounds')
    ap.add_argument('--value-games-per-retrain', type=int, default=120,
                     help='self-play games (cheap physics, not the real planner) generated per value retrain')
    ap.add_argument('--value-buffer-cap', type=int, default=40000)
    args = ap.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device}  cuda_available={torch.cuda.is_available()}  cores={os.cpu_count()}", flush=True)

    # Match the shipped game in every arena subprocess (data-gen AND gates) — inherited via
    # os.environ.copy(). Before 2026-08-06 this loop set neither, so candidates were trained and
    # promoted on 16-state synthetic boards under classic physics, then deployed onto the 50-state
    # US map under depth rules. See FINDINGS.md; index.html defaults both adjacencyMode and
    # depthMode ON for solo.
    if args.real_maps > 0:
        os.environ['REAL_MAPS'] = str(args.real_maps)
        print(f"real-map mix: {args.real_maps:.0%} of arena games on tools/real_maps.json boards", flush=True)
    if not args.classic_rules:
        os.environ['DEPTH_RULES'] = '1'
        print("depth rules: ON for all arena subprocesses (matches the shipped game)", flush=True)

    # Input width drives BOTH the net and make_batch's unpacking of the dumped feature vectors,
    # so it must be set before anything reads P.ARCH. selfplay_arena.js sizes its --dump-visits
    # vectors off the captured net's own num_features, so exporting a 24-feature net is all it
    # takes to get 24-feature training data (see candFeats24 there).
    P.ARCH = (args.features, args.hidden, args.hidden, 1)
    print(f"policy arch: {P.ARCH}  (input width {args.features})", flush=True)
    if args.features == 24:
        # Same monkeypatch convention train_rl.choose16 documents: the no-forget screen and the
        # value co-training data generator both go through T.choose(), which is hardcoded to
        # feats12. Point them at the extended chooser so they evaluate the net they are training.
        T.choose = T.choose24
        print("train_rl.choose -> choose24 (extended features + split candidates)", flush=True)

    resume_path = os.path.join(SCR, f'policy_torch_new{"" if args.features == 12 else "_f24"}.json')
    warm_path = resume_path if os.path.exists(resume_path) else T.POLICY
    best = None
    try:
        cand = P.from_json(warm_path, device=device)
        if cand.arch[0] == args.features:
            best = cand
            print(f"warm-started from {warm_path}", flush=True)
        elif cand.arch[0] < args.features and tuple(cand.arch[1:]) == tuple(P.ARCH[1:]):
            # Function-preserving widen instead of a cold start. MEASURED: a cold-started net's
            # own search sits still on 39% of decisions, so distilling its visits teaches "do
            # nothing" (move logits fell 0.6 below noop_bias; greedy move rate 0.017 -> 0.000) and
            # the next round's data is more no-op heavy still. Visit-count distillation needs a
            # competent teacher; it cannot bootstrap from noise. Widening starts the extended net
            # at EXACTLY the shipped net's strength (verified 0 difference at frac=1) so round 1
            # already has a real teacher.
            best = P.widen_inputs(cand, P.ARCH, device=device)
            print(f"widened {warm_path} ({cand.arch[0]} -> {args.features} features, "
                  f"function-preserving: identical scores on all-in candidates)", flush=True)
        else:
            print(f"{warm_path} is {cand.arch}; cannot reach {P.ARCH} by input-widening "
                  f"— starting FRESH (expect it to lose until trained)", flush=True)
    except Exception as e:
        print(f"could not warm-start from {warm_path} ({e}) — starting FRESH", flush=True)
    if best is None:
        best = P.PolicyNet(P.ARCH).to(device)
        print(f"fresh net {P.ARCH}: expect it to LOSE its first gates until it has trained",
              flush=True)

    # The value net is co-trained (Monte-Carlo regression, same recipe as train_value.py) on self-play
    # generated with the CURRENT best policy — so as the policy improves, the value net keeps training
    # on increasingly realistic games instead of staying frozen at whatever value_net.json shipped
    # with. This is the "teacher evolves more fully" extension: the MCTS search that generates the
    # policy's own training signal blends policy prior + value leaf eval, so both halves improving
    # together compounds, rather than only the policy half moving while leaf evaluation stays capped.
    value_net_path = os.path.join(SCR, 'value_net.json')
    value_resume_path = os.path.join(SCR, 'value_torch_new.json')
    value_warm_path = value_resume_path if os.path.exists(value_resume_path) else value_net_path
    value_cotrain = not args.no_value_cotrain and os.path.exists(value_warm_path)
    value_theta = value_theta_from_json(value_warm_path) if os.path.exists(value_warm_path) else None
    if value_theta is None:
        print("WARNING: no value_net.json found — arena leaf eval falls back to the hand heuristic alone "
              "(value co-training disabled)", flush=True)
        value_cotrain = False
    value_json = V.vto_json(value_theta) if value_theta is not None else None
    print(f"value net: warm-started from {value_warm_path}  co-train={value_cotrain}", flush=True)

    rng = np.random.default_rng(7)
    base_turtle = vs_script(best, 'turtle', rng, 60)
    base_rush = vs_script(best, 'rush', rng, 60)
    print(f"baseline vs scripts: turtle {base_turtle:.2f} rush {base_rush:.2f}", flush=True)

    dump_path = os.path.join(SCR, '_visits_dump.jsonl')
    cand_path = os.path.join(SCR, '_torch_cand_net.json')
    best_path = os.path.join(SCR, '_torch_best_net.json')
    selfplay_net_path = os.path.join(SCR, '_torch_selfplay_net.json')

    buf = []
    vbuf = []
    vpool = Pool(processes=min(4, os.cpu_count() or 4)) if value_cotrain else None
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
            self_play_dump(best, value_json, args.games_per_round, args.selfplay_budget, dump_path, selfplay_net_path)
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
            wr = planner_gate(cand, best, value_json, args.gate_games, args.gate_budget, cand_path, best_path)
            gates += 1
            promoted = False
            if wr is not None and wr >= args.gate_threshold:
                best = cand
                promotions += 1
                promoted = True
                write_net(best, value_json, resume_path)

            gate_str = f"{wr:.2f}" if wr is not None else "ERR"
            vtag = ""

            # 5) value co-training, every --value-retrain-every rounds — generates its own self-play
            # (cheap physics engine, not the real planner) using the CURRENT best policy, so the
            # value net's training data keeps pace with policy improvement instead of staying static.
            if value_cotrain and rnd % args.value_retrain_every == 0:
                layers = net_to_layers(best)
                noop = float(best.noop_bias.item())
                seeds = [int(rng.integers(1 << 30)) for _ in range(args.value_games_per_retrain)]
                for chunk in vpool.map(gen_value_data, [(layers, noop, s) for s in seeds]):
                    vbuf.extend(chunk)
                if len(vbuf) > args.value_buffer_cap:
                    vbuf = vbuf[-args.value_buffer_cap:]
                if len(vbuf) >= 2000:
                    value_theta, sign_acc = train_value_round(value_theta, vbuf, rng)
                    value_json = V.vto_json(value_theta)
                    json.dump(value_json, open(value_resume_path, 'w'))
                    vtag = f" value(vbuf={len(vbuf)} sign-acc={sign_acc:.3f})"
                else:
                    vtag = f" value(warmup vbuf={len(vbuf)})"

            print(f"r{rnd:3d} t={time.time()-t0:6.0f}s buf={len(buf)} loss={avg_loss:.4f} "
                  f"tt={tt:.2f} rs={rs:.2f} gate={gate_str} {'PROMOTED #%d' % promotions if promoted else 'kept'}{vtag}",
                  flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted — writing final state", flush=True)
    finally:
        if vpool is not None:
            vpool.close()
            vpool.join()
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
    write_net(best, value_json, resume_path)
    print(f"wrote {resume_path} — copy to src/web/rl_policy.json to ship (after your own review; "
          f"this script never overwrites the shipped net automatically)", flush=True)
    if value_cotrain:
        json.dump(value_json, open(value_resume_path, 'w'))
        print(f"wrote {value_resume_path} — copy to tools/value_net.json if you want to keep the "
              f"co-trained value net independent of this policy run", flush=True)


if __name__ == '__main__':
    main()
