#!/usr/bin/env python3
# PyTorch reimplementation of the State.io move-scoring policy net, for GPU-trainable distillation
# of the real MCTS planner's own search (see train_rl_torch.py). Mirrors train_rl.py's numpy
# ARCH=[12,32,32,1] / forward() exactly (same architecture, same tanh/linear activations) so the
# two implementations are numerically interchangeable — the export format is the actual contract
# (the JS client in src/web/index.html only ever reads rl_policy.json, never the training code).
import json
import torch
import torch.nn as nn

ARCH = (12, 32, 32, 1)


class PolicyNet(nn.Module):
    def __init__(self, arch=ARCH):
        super().__init__()
        self.arch = tuple(arch)
        self.linears = nn.ModuleList(nn.Linear(arch[i], arch[i + 1]) for i in range(len(arch) - 1))
        self.noop_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (batch, arch[0]) -> (batch,), matching the numpy forward()'s a[:,0] output shape
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if i < len(self.linears) - 1:
                x = torch.tanh(x)
        return x.squeeze(-1)


def widen_inputs(src, arch, device="cpu"):
    """Function-preserving widen of a narrower net onto a wider input (net2net style, same idea as
    tools/grow_net.py but along the INPUT axis).

    This exists because visit-count distillation cannot cold-start: a randomly-initialised net's
    search concludes most moves are bad and sits still (measured: the no-op wins 39% of decisions),
    so distilling its own visits teaches "do nothing", which makes the next round's data more no-op
    heavy still. Warm-starting removes the bootstrap problem entirely.

    It is exact ONLY because the extended feature layout was designed for it: features 0..11 of
    feats24 are IDENTICAL to feats12 whenever frac==1 (see train_rl.feats24). Copying the first
    src.arch[0] input columns and ZEROING the new ones therefore reproduces the source net's output
    bit-for-bit on every all-in candidate; the new features start with no influence and the trainer
    grows it from there. Hidden widths must match — a hidden-size change is grow_net.py's job.
    """
    if tuple(arch[1:]) != tuple(src.arch[1:]):
        raise ValueError(f"widen_inputs only widens the INPUT axis: {src.arch} -> {tuple(arch)} "
                         f"differs past layer 0; use grow_net.py for hidden-width changes")
    if arch[0] < src.arch[0]:
        raise ValueError(f"target input width {arch[0]} is narrower than the source's {src.arch[0]}")
    net = PolicyNet(arch).to(device)
    with torch.no_grad():
        net.linears[0].weight.zero_()
        net.linears[0].weight[:, :src.arch[0]] = src.linears[0].weight
        net.linears[0].bias.copy_(src.linears[0].bias)
        for i in range(1, len(net.linears)):
            net.linears[i].weight.copy_(src.linears[i].weight)
            net.linears[i].bias.copy_(src.linears[i].bias)
        net.noop_bias.copy_(src.noop_bias)
    return net


def to_json(net):
    layers = []
    for i, lin in enumerate(net.linears):
        layers.append({
            "W": lin.weight.detach().cpu().tolist(),
            "b": lin.bias.detach().cpu().tolist(),
            "act": "tanh" if i < len(net.linears) - 1 else "linear",
        })
    return {
        "format": "mlp-v1",
        # splits: whether the live planner may enumerate partial-stack candidates for this net.
        # OFF unless the net was actually trained on them — a widened net has zeroed weights on the
        # split features and cannot score a fraction, and the candidates still consume root-cut slots.
        "splits": bool(getattr(net, "splits", False)),
        "feat_norm": 50.0,
        "num_features": net.arch[0],
        "noop_bias": float(net.noop_bias.item()),
        "layers": layers,
    }


def from_json(path, device="cpu"):
    j = json.load(open(path))
    layers = j["layers"]
    arch = [len(layers[0]["W"][0])] + [len(l["b"]) for l in layers]
    net = PolicyNet(arch).to(device)
    with torch.no_grad():
        for lin, l in zip(net.linears, layers):
            lin.weight.copy_(torch.tensor(l["W"], dtype=torch.float32))
            lin.bias.copy_(torch.tensor(l["b"], dtype=torch.float32))
        net.noop_bias.copy_(torch.tensor([j["noop_bias"]], dtype=torch.float32))
    return net


def from_theta(theta, arch=ARCH, device="cpu"):
    # Warm-start from an existing numpy SNES theta vector (train_rl.py's flat parameter layout:
    # concatenated [W0.ravel(), b0.ravel(), W1.ravel(), b1.ravel(), ..., noop_bias]).
    net = PolicyNet(arch).to(device)
    idx = 0
    with torch.no_grad():
        for lin in net.linears:
            ni, no = lin.in_features, lin.out_features
            W = theta[idx:idx + ni * no].reshape(no, ni); idx += ni * no
            b = theta[idx:idx + no]; idx += no
            lin.weight.copy_(torch.tensor(W, dtype=torch.float32))
            lin.bias.copy_(torch.tensor(b, dtype=torch.float32))
        net.noop_bias.copy_(torch.tensor([theta[idx]], dtype=torch.float32))
    return net


def _numpy_forward_from_json(j, X):
    # Independent, ARCH-agnostic numpy forward pass built directly from the JSON's own layer
    # shapes — deliberately does NOT go through train_rl.py's unpack()/ARCH global, since that
    # global gets silently monkeypatched by other trainers (e.g. train_selfplay_planner.py sets
    # T.ARCH=[12,64,64,1]) and a stale default would truncate/misread a differently-shaped net
    # without erroring — exactly the bug this self-test exists to catch.
    a = X
    for i, l in enumerate(j["layers"]):
        W = np.array(l["W"], dtype=np.float64)
        b = np.array(l["b"], dtype=np.float64)
        z = a @ W.T + b
        a = np.tanh(z) if l["act"] == "tanh" else z
    return a[:, 0]


if __name__ == "__main__":
    # Correctness gate: load the current shipped net's OWN JSON-declared shape through an
    # independent numpy forward pass and through the PyTorch implementation, and confirm identical
    # outputs on random inputs, before anything else in the training pipeline is built on top of
    # this module.
    import sys
    import numpy as np
    sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
    import train_rl as T

    j = json.load(open(T.POLICY))
    print(f"shipped net architecture: {[len(j['layers'][0]['W'][0])] + [len(l['b']) for l in j['layers']]}")
    net = from_json(T.POLICY)

    rng = np.random.default_rng(0)
    X = rng.standard_normal((1000, net.arch[0])).astype(np.float32)
    numpy_out = _numpy_forward_from_json(j, X.astype(np.float64))
    torch_out = net(torch.tensor(X)).detach().numpy()

    max_err = float(np.abs(numpy_out - torch_out).max())
    print(f"max |numpy - torch| over 1000 random inputs: {max_err:.3e}")
    assert max_err < 1e-4, "PyTorch/numpy parity check FAILED"
    print("PARITY OK")

    # round-trip through to_json/from_json against the actual shipped rl_policy.json schema
    j1 = to_json(net)
    print("to_json/from_json round-trip OK, num_features =", j1["num_features"], "arch =", net.arch)
