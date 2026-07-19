#!/usr/bin/env python3
# FUNCTION-PRESERVING net growth (net2net) — grow the shipped champion instead of racing it.
#
# Post-mortem of the 2026-07 strength program: three challenger generations died to the same
# structural handicap — fresh random weights racing v2.2's ~40h of compounded distillation lineage
# on a plateau where the last points are the hardest (43.8-47.5% at their final evals). The repo
# already solved this once at small scale: train_rl.init_from_baseline warm-started the 8-feature
# flat net into 12 features with zero-initialized new inputs, "so training begins EXACTLY like
# today's net and only improves from there." This script is that trick at architecture scale:
#
#   v2.2 (7 feats, embed 24, 2 hops, ~7.9k params)
#     -> grown (10 feats, embed 64, 2 hops, + value head, ~49k params)
#
# such that the grown net computes the SAME policy function as v2.2 (to f32) on every board:
#   * embed: old block top-left; new feature columns ZERO; new units get eps-random incoming
#     weights (their outputs exist but nothing consumes them yet — zero downstream avoids the
#     dead-unit deadlock of all-zero init: downstream weights wake first, then incoming).
#   * attention: old rows' new input-columns ZERO (old outputs see only old dims). K's new rows
#     EXACTLY zero so new dims contribute nothing to any attention score; Q's old rows scaled by
#     sqrt(new_dim/old_dim) to cancel the 1/sqrt(dim) softmax rescale — scores are bit-preserved.
#     V/update new rows eps-random (harmless: consumed by nothing old).
#   * head1: column remap for the [h_src | h_tgt | move_feats] concat layout shift; head2,
#     noop_bias verbatim.
#   * value head: fresh eps-random (v2.2 has none). Policy preservation is untouched by it, but
#     arena leaf eval will be noisy until the value loss trains it (same brief dip the 10f graft
#     showed, recovered within ~6 rounds; the anchor gate guards promotions meanwhile).
#
# Verification (run on import as __main__): grown vs source move scores on randomized boards must
# match to <=1e-4 (f32 pack/unpack), and the printed instructions include the arena check
# (grown-vs-source must open at ~50% by construction).
#   usage: python3 tools/grow_net.py [src.json] [out.json] [--dim 64]
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gnn_net_torch as G

EPS = 1e-3   # scale of the dormant new-unit weights


def grow(src_path, out_path, new_dim=64):
    old = G.from_json(src_path)
    assert old.hops == 2, f"growth is written for 2-hop sources (got {old.hops})"
    assert old.node_feats <= G.NODE_FEATS, "source has MORE features than this build"
    d0, d1 = old.embed_dim, new_dim
    nf0, nf1 = old.node_feats, G.NODE_FEATS
    assert d1 > d0 or nf1 > nf0, "nothing to grow (same feats and same embed)"
    # d1 == d0 is allowed: FEATURE-ONLY growth (add the 3 v2 node-features, keep embed size) so the
    # net stays v2.2's speed in the equal-time arena — the exact thing that beat the bigger nets. The
    # preservation check below is the real guardrail (refuses to write if not function-preserving).
    assert d1 >= d0, "target embed_dim must be >= the source's"
    new = G.GNNPolicyNet(node_feats=nf1, embed_dim=d1, hops=old.hops,
                         move_feats=old.move_feats, head_dim=old.head_dim, value_head=True)
    s = math.sqrt(d1 / d0)   # Q rescale: cancels the sqrt(dim) attention normalizer change

    def eps_(t):
        t.copy_(torch.randn_like(t) * EPS)

    with torch.no_grad():
        # ---- embed: (d1 x nf1) ----
        eps_(new.embed.weight); eps_(new.embed.bias)
        new.embed.weight[:d0, :] = 0.0
        new.embed.weight[:d0, :nf0] = old.embed.weight
        new.embed.bias[:d0] = old.embed.bias
        # ---- attention hops ----
        for a_new, a_old in zip(new.attn, old.attn):
            for lin_new, lin_old, mode in ((a_new.q, a_old.q, 'q'), (a_new.k, a_old.k, 'k'),
                                           (a_new.v, a_old.v, 'v')):
                eps_(lin_new.weight); eps_(lin_new.bias)
                lin_new.weight[:d0, :] = 0.0
                if mode == 'q':
                    lin_new.weight[:d0, :d0] = lin_old.weight * s
                    lin_new.bias[:d0] = lin_old.bias * s
                else:
                    lin_new.weight[:d0, :d0] = lin_old.weight
                    lin_new.bias[:d0] = lin_old.bias
                if mode == 'k':          # K's NEW rows must be exactly zero: kills new-dim score terms
                    lin_new.weight[d0:, :] = 0.0
                    lin_new.bias[d0:] = 0.0
            # update: (d1 x 2*d1), concat layout [h(d1) | agg(d1)] vs old [h(d0) | agg(d0)]
            eps_(a_new.update.weight); eps_(a_new.update.bias)
            a_new.update.weight[:d0, :] = 0.0
            a_new.update.weight[:d0, :d0] = a_old.update.weight[:, :d0]          # h part
            a_new.update.weight[:d0, d1:d1 + d0] = a_old.update.weight[:, d0:]   # agg part
            a_new.update.bias[:d0] = a_old.update.bias
        # ---- move head: head1 (head_dim x 2*d1+mf), layout [src(d1)|tgt(d1)|mf] ----
        mf = old.move_feats
        new.head1.weight.zero_()
        new.head1.weight[:, :d0] = old.head1.weight[:, :d0]
        new.head1.weight[:, d1:d1 + d0] = old.head1.weight[:, d0:2 * d0]
        new.head1.weight[:, 2 * d1:2 * d1 + mf] = old.head1.weight[:, 2 * d0:2 * d0 + mf]
        new.head1.bias.copy_(old.head1.bias)
        new.head2.weight.copy_(old.head2.weight)
        new.head2.bias.copy_(old.head2.bias)
        new.noop_bias.copy_(old.noop_bias)
        # ---- value head: fresh (dormant scale) ----
        eps_(new.v1.weight); eps_(new.v1.bias); eps_(new.v2.weight); eps_(new.v2.bias)

    # ---- preservation check: identical move scores on randomized boards ----
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(40):
        N = int(rng.integers(8, 24))
        adj = [[j for j in range(N) if j != i and abs(j - i) <= 2] for i in range(N)]
        owner = rng.integers(0, 3, N).tolist()
        owner[0] = 2
        troops = (rng.random(N) * 60 + 3).tolist()
        armies = [{'owner': int(rng.integers(1, 3)), 'count': float(rng.random() * 20), 'ti': int(rng.integers(N))}
                  for _ in range(int(rng.integers(0, 4)))]
        inc_m = [0.0] * N; inc_e = [0.0] * N
        for a in armies:
            (inc_m if a['owner'] == 2 else inc_e)[a['ti']] += a['count']
        nf_old = G.node_features(owner, troops, adj, inc_m, inc_e, 2)
        with torch.no_grad():
            # old net consumes only its original nf0 features
            h_old = old.node_embeddings(torch.tensor([r[:nf0] for r in nf_old], dtype=torch.float32), adj)
            h_new = new.node_embeddings(torch.tensor(nf_old, dtype=torch.float32), adj)
            pairs = torch.tensor([[0, j] for j in adj[0][:3]], dtype=torch.long)
            if len(pairs) == 0:
                continue
            mx = torch.rand(len(pairs), mf)
            so = old.move_scores(h_old, pairs, mx)
            sn = new.move_scores(h_new, pairs, mx)
            worst = max(worst, float((so - sn).abs().max()))
    print(f"policy preservation: max |score diff| over randomized boards = {worst:.3e}")
    assert worst <= 1e-4, "growth is NOT function-preserving — refusing to write"

    json.dump(G.to_json(new), open(out_path, 'w'))
    n_params = sum(p.numel() for p in new.parameters())
    print(f"wrote {out_path}: node_feats {nf0}->{nf1}, embed {d0}->{d1}, +vhead, {n_params} params")
    print("next: arena-verify ~50% open vs the source net (strip vhead for a pure-policy check), "
          "then train with the anchor gate — every point gained is on TOP of the champion.")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('src', nargs='?', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'web', 'gnn_policy.json'))
    ap.add_argument('out', nargs='?', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gnn_grown_policy.json'))
    ap.add_argument('--dim', type=int, default=64)
    a = ap.parse_args()
    grow(a.src, a.out, a.dim)
