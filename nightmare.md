# Nightmare — the actually-impossible tier

> **Status (2026-07-11):** Phase 1 shipped earlier (tier entry, multiMove salvo commits, workerMs/
> cadence — see the LEVELS entry and onPlanWorkerMsg in index.html; play-tested in v2.1). Now also
> implemented: **Lever 1c** (multiMove tiers get the FULL worker pool — workersFor), **Lever 4
> root bias** (OppModel prediction threaded into the planner via req.oppBias; worker raises root
> priors on pre-empting moves), and **Lever 4 offline** (script:human profile bot in
> selfplay_arena.js joins the training league via train_gnn_torch.py --script-mix). Phase 2 (GPU
> nights on the upgraded data pipeline: outcome-stamped dumps + value co-training, noop
> subsampling, corner-pruned boards, deep-expert budget, Dirichlet+temperature exploration) is
> ready to run — see tools/train_gnn_torch.py's header.

Design doc for a top difficulty above Impossible. Goal: an AI no human realistically beats,
achieved **without touching the stat sheet** — same growth, same caps, same start, same rules.
The tier's tagline extends the Impossible doctrine one step:

> Impossible out-thinks you. Nightmare out-thinks you *and* out-clicks you.
> Neither ever out-cheats you.

## Why "unbeatable" is reachable here

Every cap that kept Impossible human-beatable is a deliberate self-restriction, not a limit of
the machine. The evidence, all measured in this repo (2026-07-02 → 07-04 sessions):

- **Impossible decides once per ~0.85 s** (`aiSpeed 0.85` — "human-fair reaction cadence, same
  as Hard"), and the planner **commits exactly one move per decision** (`onPlanWorkerMsg` takes
  the single most-visited root move). Meanwhile winning *human* play is coordinated multi-state
  attacks: ANTHONY's play-data profiles show **jointRate 0.70–0.82** — the human routinely does
  the thing the AI cannot express.
- The in-game **worker search budget is 350 ms** (`workerMs`), while the same tier's server
  budget is 4000 ms. Search strength compounds: the planner beats its own raw net 55.7% at just
  60 ms/move (280-game arena, 2026-07-04); the raw net alone already ties the flat-MLP-with-
  search (47.1%).
- The overnight training run (11.6 h, 339 rounds, **167 promotions**) was *still promoting when
  stopped* — the strength ladder had not flattened. More GPU-hours are unspent strength.
- The game already ships an **opponent model** (`OppModel`, the Adaptive tier): it learns the
  human's habits across sessions and best-responds — and it is currently paired with the
  *weakest* brain in the lineup.

In every RTS-like ever, bots pass humans on mechanics (speed + parallel attention) multiplied
by a competent policy. Nightmare is exactly that recipe, with the policy we already trained.

---

## The four levers

### Lever 1 — Mechanical superiority (biggest win, ~free)

**1a. Fast cadence.** Ace already runs `aiSpeed 0.45` as its whole identity. Nightmare pairs a
fast cadence (0.30–0.45) with the *full planner*. Guard rail: plan latency must fit inside the
cadence, so this lever interacts with 1c below (parallel workers hide latency; a plan computed
from a ≤0.45 s-old board is still fresher than a human's reaction loop).

**1b. Multi-move commits (the important one).** Today `onPlanWorkerMsg` merges the workers'
root tallies and executes ONE move. Change: execute the top-K root moves that are mutually
compatible, in one tick — a true salvo.

- Source of truth already exists: `roots` is the full per-move visit distribution.
- Compatibility filter: no two commits share a source state; each commit must individually
  out-visit the noop tally; each passes the existing `s.owner===owner && !freshThreat(s)`
  guards; respect split fractions per commit (`bestMove[2]`).
- K: start at 3. The human's own joint attacks run 2–8 states (play-data `joint:` field), so
  K=3 is not even superhuman yet — it is parity. Raise after testing.
- Risk: the search evaluated these moves as *alternatives*, not as a package — the second-best
  root move may be redundant with the best (both hitting the same target). Mitigations: skip a
  commit whose target already has an inbound friendly army this tick; cap total committed
  troops at a fraction of frontline strength. If that proves too crude, the principled version
  is sequential re-planning: commit move 1, advance the root snapshot, re-pick — the worker
  already has `snapSend` to do this cheaply inside one plan reply.

**1c. Attention everywhere.** Root parallelization already merges independent worker searches
(`workersFor`, tally merge). Give Nightmare the full worker pool per decision instead of 1–2,
so every front is searched every tick. A human allocates attention; Nightmare doesn't have to.

### Lever 2 — More search

`workerMs 350 → 800–1000` for the tier (the `LEVELS` entry carries per-tier `workerMs`; no
engine change). With 1c, effective simulations per decision go up ~5–10×. Measured trend says
each config step of extra search is worth points, and it stacks with cadence because faster
decisions on fresher boards waste fewer simulations on stale states.

### Lever 3 — A stronger net (the compute lever)

Resume `tools/train_gnn_torch.py` (warm-starts from the promoted checkpoint, arena-gated so it
can only improve) with the two 2026-07-04 fixes now in the pipeline:

- **Corner-aware boards**: `train_rl.new_game` prunes ~15% of nodes to degree-1 pendants, so
  the Maine-style hoard-lock is trained away instead of hand-shaped away (see the `freeLunch`
  clamp in `computeUntried` — the goal is to retire it once a corner-trained net stops needing
  it).
- **Farmer gate**: the distant-expander exploit is a hard no-forget veto (`fm=` in the round
  log) — candidates that regress against it never promote.

Budget guidance: rounds cost ~2.2 min; the 11.6 h run bought 167 promotions and was still
climbing. Accumulate 50–100 h across idle nights (the loop is resumable across invocations)
before considering architecture scale-ups (`--embed-dim`, `--hops`) — data first, size second.

### Lever 4 — Anti-human specialization

`OppModel` (Adaptive tier) predicts P(human's next move | board) online, with confidence
gating, persisted across sessions. Today its predictions only bias the *greedy* chooser
(`rlEvaluate`'s `adaptPred`). Nightmare folds it into the planner:

- Root bias: raise priors on moves that pre-empt the predicted human action (racing the state
  the human is about to grab; pouncing on the state they are about to empty — the model's two
  strongest reads).
- Offline: `phone_export/` + `tools/parse_play_data.py` already capture real human profiles
  (aggression, joint rate, split habits — the exact fields quoted above). A scripted
  "human-profile bot" driven by those distributions joins the training league next to
  turtle/rush/farmer, so the net trains against human-shaped play, not just self-play.

---

## The tier entry (draft)

```js
// Nightmare: everything Impossible has, with the human-fairness caps removed. Full planner +
// GNN, Ace-class reflexes (aiSpeed 0.35), salvo commits (multiMove 3), deep worker search
// (workerMs 900), and the opponent model biasing the search. Stats stay 100% symmetric.
{ name:'Nightmare', enemies:1, aiSpeed:0.35, aiAggro:1.0, neutralMin:18, neutralMax:55,
  growRate:1.0, playerStart:22, enemyStart:22, enemyGrow:1.0, rl:true, plan:true, adapt:true,
  planDt:0.3, mctsK:9, tbudget:4000, workerMs:900, hunt:0.7, multiMove:3 },
```

Coin multiplier: `Nightmare: 4.0` in `awardCoins`. `selectPolicyForLevel`: Nightmare loads the
GNN (same as Impossible).

## Build order & gates

Phase 1 (afternoon): tier entry + `multiMove` commit logic in `onPlanWorkerMsg` + workerMs/
cadence. **Gate:** beats current Impossible head-to-head in the arena — but note the arena is
synchronous (equal turns) and cannot see the cadence advantage, so the arena number
*understates* Nightmare; it only needs to confirm no regression from multi-move commits.
The real gate is play-testing: it should beat its author.

Phase 2 (GPU-nights): accumulate training per Lever 3; swap `gnn_policy.json` on each gated
promotion.

Phase 3 (a day): OppModel → planner bias + human-profile league bot. **Gate:** measured lift
against replayed human profiles, and the author still can't beat it.

## Honest limits

- "Literally unbeatable" does not exist in a fair symmetric game; the target is "no human
  realistically beats it." Mechanics (Lever 1) is what makes that stick — policy alone
  plateaus near parity (every net+search config we measured sits within a few points of 50%
  against every other at equal time).
- The synchronous arena cannot measure Levers 1a/1c/4. Evaluation must include live play and
  profile-bot matches, or the numbers will lie reassuringly.
- Multi-move commits change the AI's *feel* sharply. If Nightmare is meant to be fun-hard
  rather than demoralizing-hard, K and cadence are the difficulty dials — expose them per
  scenario rather than maxing both by default.
