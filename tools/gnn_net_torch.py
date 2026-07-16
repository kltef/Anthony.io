#!/usr/bin/env python3
# Board-aware policy net — FINDINGS.md future-path #1 ("Board-aware architecture").
#
# The 12-feature flat MLP scores each candidate move in isolation: it never sees the shape of the
# map, only 12 numbers about one (source, target) pair. Phase B (see FINDINGS.md) proved this is a
# real ceiling: adding more hand-picked per-move features didn't help, because the net structurally
# had no way to reason about board topology.
#
# This net instead computes ONE embedding per territory from the actual adjacency graph — a small
# graph-attention stack restricted to real border edges (NOT a full O(N^2) transformer: we already
# know the graph, so there's no reason to make the net rediscover it, and full attention would cost
# more of the per-move time budget for no benefit — see the "smarter methods of planning" / "would a
# transformer help" discussion this design comes out of). Move candidates are then scored from a pair
# of node embeddings + the existing cheap move-specific features, so the net can finally represent
# things like "this attack looks fine locally but strands my other border" — exactly what Phase B's
# bolt-on features couldn't teach it.
#
# Export format ("gnn-v1") is a NEW schema, additive to the existing "mlp-v1" flat net — the live game
# (src/web/index.html) format-dispatches on load, so this never breaks or replaces the shipped net
# until a trained gnn-v1 net actually beats it in the real-planner arena (same no-regression posture
# as every other change in this project).
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

# v2 node features (2026-07-12): the original 7 plus 3 the net could not derive on its own —
#   capHeadroom/FN  : how much passive generation this state has left before the 150 cap gates it
#                     (the cap-dodge economy — see the play-test mega-stack findings — was invisible)
#   borderPressure/FN: total hostile troops directly adjacent (threat concentration at this node)
#   isFrontier      : any neighbor owned differently (border vs interior state)
# CONTRACT: mirrored exactly in index.html gnnNodeFeats (worker AND main-thread copies). A 10-feature
# net exports node_features:10; the JS side dispatches on that field, so old 7-feature nets keep
# working unchanged.
NODE_FEATS = 10  # [is_mine, is_neutral, is_enemy, troops/FN, degree/maxDeg, incMine/FN, incEnemy/FN, headroom/FN, borderPressure/FN, isFrontier]
# MOVE_FEATS is 4 with unit-splitting enabled: the AI can send a FRACTION of a source's troops, not
# just all of them, so the move head sees how much is being committed. `sent = st*frac` is the troop
# count actually leaving the source; the feature vector is:
#   [(sent-tt)/FN, dist/rlDiag, sent>tt, frac]
# (with frac=1.0 this reduces to the old all-in [(st-tt)/FN, dist, st>tt, 1.0], so 2-element legacy
# moves [si,ti] map cleanly to frac=1.0.) CONTRACT: this exact formula is mirrored in index.html
# (worker + main-thread computeUntried/snapGreedy, and gnnMoveScore callers), selfplay_arena.js's
# dump, and train_gnn_torch.py's build_batch/decision_loss — keep all of them in sync.
MOVE_FEATS = 4
# The split fractions the planner enumerates as separate candidates per (source,target). GNN/Impossible
# only — the flat MLP (Grandmaster) stays all-in. Mirrored as FRACTIONS in index.html and the arena.
FRACTIONS = [1.0, 0.75, 0.5, 0.25]
FN = 50.0


class AttnHop(nn.Module):
    # single-head scaled dot-product attention, restricted to actual graph edges (adjacency), plus a
    # residual self-update — deliberately NOT full pairwise attention over all N nodes: the graph
    # structure is already known, so only real neighbors are attended over (cheaper, and doesn't
    # force the net to rediscover adjacency it could just be told).
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.update = nn.Linear(dim * 2, dim)
        self.scale = dim ** 0.5

    def forward(self, h, adj):
        # h: (N, dim) node embeddings. adj: list of neighbor-index lists (python ints), length N.
        N = h.shape[0]
        Q, K, V = self.q(h), self.k(h), self.v(h)
        agg = torch.zeros_like(h)
        for i in range(N):
            nbrs = adj[i]
            if not nbrs:
                continue
            qi = Q[i:i + 1]                       # (1, dim)
            ki = K[nbrs]                           # (deg, dim)
            vi = V[nbrs]                           # (deg, dim)
            scores = (qi @ ki.T).squeeze(0) / self.scale   # (deg,)
            w = F.softmax(scores, dim=0)
            agg[i] = w @ vi
        return torch.tanh(self.update(torch.cat([h, agg], dim=1)))

    def forward_batched(self, h, adj_mask):
        # Vectorized equivalent of forward() over a whole batch of boards at once — the training-time
        # hot path (single-board forward() stays for inference and the no-forget screen).
        #   h: (B, N, dim). adj_mask: (B, N, N) with adj_mask[b,i,j]=1 iff j is a neighbor of i (no
        #   self-loop, matching adj which never contains i). Row i entirely 0 => node i has no
        #   neighbors => its aggregate must be 0, exactly like forward()'s `if not nbrs: continue`.
        Q, K, V = self.q(h), self.k(h), self.v(h)                     # (B, N, dim)
        scores = torch.matmul(Q, K.transpose(1, 2)) / self.scale      # (B, N, N): scores[b,i,j]=qi·kj
        neg = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(adj_mask == 0, neg)               # attend real neighbors only
        w = torch.softmax(scores, dim=2)                              # (B, N, N) over j
        # a fully-masked row (no neighbors) softmaxes to uniform, not 0 — zero those rows so their
        # aggregate is 0, exactly matching the per-node loop's skip. (Delaunay boards never actually
        # produce isolated nodes, but this keeps the two paths bit-for-bit equivalent regardless.)
        has_nbr = (adj_mask.sum(dim=2, keepdim=True) > 0).to(w.dtype)  # (B, N, 1)
        agg = torch.matmul(w, V) * has_nbr                            # (B, N, dim)
        return torch.tanh(self.update(torch.cat([h, agg], dim=2)))


class GNNPolicyNet(nn.Module):
    def __init__(self, node_feats=NODE_FEATS, embed_dim=24, hops=2, move_feats=MOVE_FEATS, head_dim=32,
                 value_head=False):
        super().__init__()
        self.node_feats, self.embed_dim, self.hops, self.move_feats, self.head_dim = \
            node_feats, embed_dim, hops, move_feats, head_dim
        self.embed = nn.Linear(node_feats, embed_dim)
        self.attn = nn.ModuleList(AttnHop(embed_dim) for _ in range(hops))
        self.head1 = nn.Linear(embed_dim * 2 + move_feats, head_dim)
        self.head2 = nn.Linear(head_dim, 1)
        self.noop_bias = nn.Parameter(torch.zeros(1))
        # optional board-aware VALUE head (planner leaf evaluation): mean-pool the node embeddings,
        # small tanh MLP -> expected result in [-1,1] for the mover. Additive — old nets (and old
        # exported JSON) simply don't have it; format stays "gnn-v1" (see to_json's "vhead" key).
        self.value_head = value_head
        if value_head:
            self.v1 = nn.Linear(embed_dim, 32)
            self.v2 = nn.Linear(32, 1)

    def board_value(self, h):
        # h: (N, embed_dim) node embeddings (from node_embeddings()). Returns a scalar in [-1,1] —
        # the mover's expected graded outcome. Mirrored in index.html gnnValueScore() (both copies).
        pooled = h.mean(dim=0)
        z = torch.tanh(self.v1(pooled))
        return torch.tanh(self.v2(z)).squeeze(-1)

    def board_value_batched(self, h):
        # h: (B, N, embed_dim) -> (B,). Mean-pool per board — trivial here since batches are
        # uniform-N (train_gnn_torch.build_batch groups records by node count).
        pooled = h.mean(dim=1)
        z = torch.tanh(self.v1(pooled))
        return torch.tanh(self.v2(z)).squeeze(-1)

    def node_embeddings(self, node_x, adj):
        # node_x: (N, node_feats) tensor. adj: list of N neighbor-index lists. Returns (N, embed_dim).
        h = torch.tanh(self.embed(node_x))
        for layer in self.attn:
            h = layer(h, adj)
        return h

    def move_scores(self, h, pairs, move_x):
        # h: (N, embed_dim) node embeddings (already computed once per board via node_embeddings()).
        # pairs: (M, 2) long tensor of (source_idx, target_idx). move_x: (M, move_feats).
        hs, ht = h[pairs[:, 0]], h[pairs[:, 1]]
        x = torch.cat([hs, ht, move_x], dim=1)
        z = torch.tanh(self.head1(x))
        return self.head2(z).squeeze(-1)

    def node_embeddings_batched(self, node_x, adj_mask):
        # node_x: (B, N, node_feats). adj_mask: (B, N, N). Returns (B, N, embed_dim).
        h = torch.tanh(self.embed(node_x))
        for layer in self.attn:
            h = layer.forward_batched(h, adj_mask)
        return h

    def move_scores_batched(self, h, src_idx, tgt_idx, move_x):
        # h: (B, N, embed_dim). src_idx/tgt_idx: (B, M) long. move_x: (B, M, move_feats). -> (B, M).
        B = h.shape[0]
        bidx = torch.arange(B, device=h.device).unsqueeze(1)         # (B, 1)
        hs = h[bidx, src_idx]                                         # (B, M, embed_dim)
        ht = h[bidx, tgt_idx]
        x = torch.cat([hs, ht, move_x], dim=2)
        z = torch.tanh(self.head1(x))
        return self.head2(z).squeeze(-1)                             # (B, M)


def to_json(net):
    def lin(l):
        return {"W": l.weight.detach().cpu().tolist(), "b": l.bias.detach().cpu().tolist()}
    j = {
        "format": "gnn-v1",
        "feat_norm": FN,
        "node_features": net.node_feats,
        "move_features": net.move_feats,
        "embed_dim": net.embed_dim,
        "hops": net.hops,
        "noop_bias": float(net.noop_bias.item()),
        "embed": lin(net.embed),
        "attn": [
            {"q": lin(a.q), "k": lin(a.k), "v": lin(a.v), "update": lin(a.update)}
            for a in net.attn
        ],
        "head1": lin(net.head1),
        "head2": lin(net.head2),
    }
    # additive: exported ONLY when the net actually has a value head — old nets' JSON is unchanged,
    # and the JS side (index.html gnnValueScore) dispatches on the presence of "vhead".
    if getattr(net, "value_head", False):
        j["vhead"] = {"v1": lin(net.v1), "v2": lin(net.v2)}
    return j


def from_json(path, device="cpu"):
    j = json.load(open(path))
    net = GNNPolicyNet(j["node_features"], j["embed_dim"], j["hops"], j["move_features"],
                        len(j["head1"]["b"]), value_head=("vhead" in j)).to(device)

    def load_lin(l, d):
        with torch.no_grad():
            l.weight.copy_(torch.tensor(d["W"], dtype=torch.float32))
            l.bias.copy_(torch.tensor(d["b"], dtype=torch.float32))
    load_lin(net.embed, j["embed"])
    for a, ad in zip(net.attn, j["attn"]):
        load_lin(a.q, ad["q"]); load_lin(a.k, ad["k"]); load_lin(a.v, ad["v"]); load_lin(a.update, ad["update"])
    load_lin(net.head1, j["head1"])
    load_lin(net.head2, j["head2"])
    if "vhead" in j:
        load_lin(net.v1, j["vhead"]["v1"])
        load_lin(net.v2, j["vhead"]["v2"])
    with torch.no_grad():
        net.noop_bias.copy_(torch.tensor([j["noop_bias"]], dtype=torch.float32))
    return net


# ---------------- board -> node features (CONTRACT: replicate exactly in index.html gnnNodeFeats) ----------------
def node_features(owner, troops, adj, incoming_mine, incoming_enemy, mover):
    # owner/troops: arrays over states. adj: list of neighbor-index lists. incoming_*: arrays over
    # states (enemy/mine in-flight troop totals, mirrors incoming_for()/incomingTo()). mover: owner id
    # whose perspective this is scored from. Returns an (N, NODE_FEATS) list of lists.
    N = len(owner)
    max_deg = max((len(a) for a in adj), default=1) or 1
    CAP = 150.0
    out = []
    for i in range(N):
        o = owner[i]
        border_pressure = 0.0
        frontier = 0.0
        for j in adj[i]:
            if owner[j] != o:
                frontier = 1.0
            if owner[j] != 0 and owner[j] != mover:
                border_pressure += troops[j]
        out.append([
            1.0 if o == mover else 0.0,
            1.0 if o == 0 else 0.0,
            1.0 if (o != 0 and o != mover) else 0.0,
            troops[i] / FN,
            len(adj[i]) / max_deg,
            incoming_mine[i] / FN,
            incoming_enemy[i] / FN,
            max(0.0, CAP - troops[i]) / FN,
            border_pressure / FN,
            frontier,
        ])
    return out


if __name__ == "__main__":
    # Sanity check: a small random board, one forward pass, verify shapes and that the net actually
    # responds to graph structure (isolating one node from the rest should change its own embedding
    # relative to a fully-connected version, since it can no longer aggregate neighbor info).
    import numpy as np
    torch.manual_seed(0)
    N = 10
    net = GNNPolicyNet()
    adj = [[j for j in range(N) if j != i and abs(j - i) <= 2] for i in range(N)]   # small path-ish graph
    node_x = torch.rand(N, NODE_FEATS)
    h = net.node_embeddings(node_x, adj)
    print("node embeddings shape:", h.shape)
    assert h.shape == (N, net.embed_dim)
    pairs = torch.tensor([[0, 1], [2, 5], [3, 4]], dtype=torch.long)
    move_x = torch.rand(3, MOVE_FEATS)   # 4 move features now: [(sent-tt)/FN, dist, sent>tt, frac]
    scores = net.move_scores(h, pairs, move_x)
    print("move scores:", scores.tolist())
    assert scores.shape == (3,)

    # isolate node 0 (no neighbors) and confirm its embedding actually differs from the connected case
    adj_isolated = [[] if i == 0 else adj[i] for i in range(N)]
    h_iso = net.node_embeddings(node_x, adj_isolated)
    diff = (h[0] - h_iso[0]).abs().max().item()
    print(f"node 0 embedding changes when isolated: max diff = {diff:.4f}")
    assert diff > 1e-6, "graph structure isn't actually affecting node embeddings — bug"

    j = to_json(net)
    net2 = GNNPolicyNet(j["node_features"], j["embed_dim"], j["hops"], j["move_features"], len(j["head1"]["b"]))
    net2.load_state_dict(net.state_dict())
    h2 = net2.node_embeddings(node_x, adj)
    assert torch.allclose(h, h2, atol=1e-6), "state_dict round-trip mismatch"
    print("SELF-TEST OK")
