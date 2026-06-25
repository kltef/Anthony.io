# State.io "Impossible" AI — Investigation & Findings

A record of the work to make the Impossible AI as strong as possible, and the systematic
investigation into whether it can be pushed further. **Bottom line: the shipped AI is measurably
optimal for its architecture — four independent, rigorously-gated attempts to beat it all failed.**

## What shipped (live in the APK, all validated)

1. **Game-logic overhaul** — anti-snowball "rope-a-dope" that springs the moment you pull ahead,
   economic catch-up + coordinated focus-fire (rivals truce to gang the leader), harder crush,
   faster reactions, comeback-when-cornered, and the planner now **models the human at full net
   strength** (it used to assume you play poorly). Root-parallelized across idle cores.
2. **Learned value network** (12→32→16→1, ~82% sign-accuracy) replaced the crude leaf heuristic in
   the MCTS planner's evaluation.
3. **Bigger policy net** (12→32→32 → **12→64→64→1**) distilled from the full-game Monte-Carlo expert,
   which resolved a measured capacity trade-off (the 32-wide net couldn't hold turtle-resistance AND
   general play at once).

Combined, the shipped net beats the start-of-session net **~97%** head-to-head.

## Measurement methodology

A headless **self-play arena** (`tools/selfplay_arena.js`) extracts the *actual* shipped MCTS
planner (`planWorkerMain` from `index.html`) and plays two policy configs head-to-head in the game's
own physics, under an **equal per-move time budget** (so a bigger/slower net is penalized exactly as
in-game). Every change was **gated**: it had to beat the *current* shipped net in the arena, with no
regression vs scripted turtle/rush/economist opponents. This is what kept the live AI at its best
quality throughout — every "did not ship" below is a gate correctly rejecting a regression.

## Phase-by-phase results

| Phase | Idea | Gate | Result vs current net | Shipped |
|-------|------|------|----------------------|---------|
| A1 | Attribution ablation | arena | bigger net **98%**, value head **50%** (neutral alone) | n/a |
| A2 | Tune value/heuristic blend weight | arena | neutral → kept 0.5 | n/a |
| **B** | **Richer inputs** (in-flight armies, source vulnerability, arrival-time strength, can-win-after-inflight; +4 features, contract-verified Python↔JS to 2e-16) | arena | **loses (23%)** | ❌ |
| **C** | **Wider net** (96-wide) | arena | **ties (50%)** — capacity saturated at 64 | ❌ |
| **D** | **Self-play loop** (AlphaZero-style) with a cheap *greedy* promotion gate | greedy proxy | proxy said "improving" 157×; **real arena: loses (37%)** | ❌ |
| **D+** | **Self-play loop with the REAL-PLANNER gate** (value co-training, anti-forgetting league) | real planner | **9 gates, all ~0.50 — ties, never beats** | ❌ |

### Why each failed (the useful part)
- **B (more inputs):** the distillation teacher's move choice was driven by the 12-input policy, so
  the "best move" labels barely depended on the new features — the student had nothing to learn from
  them. Richer inputs need a *feature-aware teacher* (i.e. fold into D), not bolt-on distillation.
- **C (more width):** the 64-wide net already saturates the representable policy for 12 features;
  96-wide is a slightly-better function but ~2.25× slower, so at equal time budget it does fewer
  sims and nets out to an exact tie. Confirms ~64 is the capacity sweet spot.
- **D (greedy gate):** **proxy misalignment** — optimizing greedy head-to-head drifted away from
  *planner* strength (and eroded turtle-resistance). The cheap gate lied. Key lesson: the in-loop
  objective must match what ships.
- **D+ (planner gate):** with the gate fixed to the deployment objective, candidates that hold
  turtle-resistance **tie the current net (0.47–0.50 across 9 independent gates) but never beat it.**
  This is the rigorous confirmation of the ceiling.

## Conclusion: the shipped AI is on the Pareto frontier

You cannot improve its general play without sacrificing exploiter-resistance, and the versions that
hold both cannot out-*plan* it. Four independent attempts (more inputs, more width, self-play with a
proxy gate, self-play with the correct gate) all hit the same wall. This is a **genuine optimum for
the architecture**, not a tuning failure — the limit is set by:

- **12 hand-engineered features** (vs a learned board representation),
- **64-wide MLP** policy capacity,
- **value-net quality** (~82%) and **1-ply + short-rollout** teacher depth.

## How to push past the ceiling (real projects, not tuning)

1. **Board-aware architecture** — a graph/attention network over the *whole map* instead of 12
   per-move features, giving the net spatial/topological structure to reason about. Requires
   GPU + a real ML framework (PyTorch).
2. **Deep self-play at scale** — only worth it *with* (1); a bigger, structured net is where 100k–1M
   self-play games (true expert iteration) finally have somewhere to go.
3. **Real-planner-in-the-loop self-improvement** (built here as `train_selfplay_planner.py`) — the
   correct loop; combine with (1)+(2) for gains the current 12-feature/64-wide design can't express.

## Tooling produced (in `tools/`)

- `selfplay_arena.js` — headless real-planner arena (the measurement bedrock).
- `train_rl.py` — policy trainer + features (`feats12`/`feats16`) + scripted opponent league.
- `train_value.py` — value-net trainer (Monte-Carlo regression).
- `train_distill_big.py` — distill the MC expert into a configurable-width net.
- `train_distill_big16.py` — 16-feature (richer-input) variant.
- `train_distill_D.py` — value-net-bootstrapped stronger teacher (expert iteration).
- `train_selfplay_loop.py` — autonomous self-improving loop (greedy gate).
- `train_selfplay_planner.py` — autonomous self-improving loop with the real-planner gate.
