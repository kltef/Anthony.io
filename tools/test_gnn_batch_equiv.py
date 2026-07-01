#!/usr/bin/env python3
# Correctness gate for the batched GNN training path: the batched loss + its gradients MUST equal the
# per-decision reference loop (decision_loss) to floating-point tolerance. A forward-only check isn't
# enough — a bug that gets the loss value right but the gradient wrong would silently mistrain the net
# over an entire overnight run — so this compares BOTH the loss value and the .grad on every parameter.
#
#   usage: python3 tools/test_gnn_batch_equiv.py
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_rl as T
import train_gnn_torch as R
import gnn_net_torch as G


def make_record(rng, with_armies=False, isolate_node=None):
    # Build a realistic training record from train_rl's environment (real Delaunay adjacency), with a
    # fabricated candidate list + visit counts — same schema the arena dumps and the trainer consumes.
    P = int(rng.integers(2, 5))
    g = T.new_game(16, P, rng)
    mover = 1
    adj = [list(a) for a in g['adj']]
    if isolate_node is not None:
        k = isolate_node
        for i in range(len(adj)):
            adj[i] = [x for x in adj[i] if x != k]
        adj[k] = []
    o = g['owner']
    srcs = [int(i) for i in range(16) if o[i] == mover and g['troops'][i] >= 5]
    if not srcs:
        srcs = [int(np.argmax(o == mover))] if (o == mover).any() else [0]
    moves = []
    gg = dict(g); gg['adj'] = adj
    for si in srcs:
        tgts = T.legal_targets(gg, si, mover)
        for ti in tgts[:4]:
            moves.append([int(si), int(ti)])
    moves = moves[:9]
    moves.append(None)                       # noop candidate
    if len(moves) == 1:                       # ensure >=2 candidates so softmax is non-trivial
        moves = [[srcs[0], (srcs[0] + 1) % 16], None]
    visits = [float(int(rng.integers(1, 500))) for _ in moves]
    armies = []
    if with_armies:
        for _ in range(int(rng.integers(1, 4))):
            armies.append(dict(owner=int(rng.integers(1, P + 1)), count=float(rng.integers(5, 40)),
                                ti=int(rng.integers(0, 16))))
    return dict(owner=o.tolist(), troops=g['troops'].tolist(), adj=adj, armies=armies, mover=mover,
                cx=g['pos'][:, 0].tolist(), cy=g['pos'][:, 1].tolist(), rlDiag=float(g['rlDiag']),
                moves=moves, visits=visits)


def clone_net(net):
    n2 = G.GNNPolicyNet(net.node_feats, net.embed_dim, net.hops, net.move_feats, net.head_dim)
    n2.load_state_dict(net.state_dict())
    return n2


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    device = 'cpu'

    records = [make_record(rng) for _ in range(20)]
    records += [make_record(rng, with_armies=True) for _ in range(10)]   # exercise in-flight node feats
    records.append(make_record(rng, isolate_node=5))                      # exercise the no-neighbor path
    # a single-candidate-plus-noop decision (M=2, minimal softmax)
    r_small = make_record(rng)
    r_small['moves'] = [r_small['moves'][0], None]
    r_small['visits'] = [7.0, 3.0]
    records.append(r_small)

    net = G.GNNPolicyNet(embed_dim=24, hops=2)

    # ---- forward value: batched vs mean of per-decision reference ----
    net_ref = clone_net(net)
    ref = torch.stack([R.decision_loss(net_ref, r, device) for r in records]).mean()

    net_bat = clone_net(net)
    batch = R.build_batch(records, device)
    bat = R.batched_loss(net_bat, batch)

    fwd_err = abs(ref.item() - bat.item())
    print(f"forward loss:  ref={ref.item():.8f}  batched={bat.item():.8f}  |diff|={fwd_err:.2e}")
    assert fwd_err < 1e-5, f"FORWARD MISMATCH: {fwd_err}"

    # ---- gradients: backward both, compare .grad on every parameter ----
    ref.backward()
    bat.backward()
    max_grad_err = 0.0
    for (n1, p1), (n2, p2) in zip(net_ref.named_parameters(), net_bat.named_parameters()):
        assert n1 == n2
        g1 = p1.grad if p1.grad is not None else torch.zeros_like(p1)
        g2 = p2.grad if p2.grad is not None else torch.zeros_like(p2)
        e = (g1 - g2).abs().max().item()
        max_grad_err = max(max_grad_err, e)
        print(f"  grad {n1:20s} |diff|={e:.2e}")
    print(f"max gradient |diff| = {max_grad_err:.2e}")
    assert max_grad_err < 1e-5, f"GRADIENT MISMATCH: {max_grad_err}"

    print("EQUIVALENCE OK (forward + gradients match to <1e-5)")


if __name__ == '__main__':
    main()
