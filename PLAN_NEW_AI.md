# Plan: rebuild the State.io AI from scratch

Status: **DRAFT, awaiting approval. Nothing in this plan has been started.**

## 1. Goal and honest limits

Replace the entire existing AI (scripted layer, planner, nets, opponent model) with a new one
built on a single shared rules engine. Keep everything else: UI, maps, netplay, shop, scenarios.

**Target:** win at least 99% of games against your play-tester and other strong humans, measured,
with no repeatable exploit.

**What I cannot promise:** a mathematically unbeatable AI. Players move simultaneously in real
time, so no proof exists, and one lucky human game on a volatile map can still happen. I will
report measured numbers, not claims. "Impossible" in the game means "measured win rate above the
target", and the plan has a way to measure it (section 8).

## 2. What is wrong with the current AI (why a rebuild)

Evidence from the repo (`FINDINGS.md`, `BEAT_FADER.md`, commits `bd5fd35 af8adf2 ec9c93c 9924661`,
and `RULES.md`):

1. **No memory or intent.** The search re-decides every move, so it ping-pongs troops between two
   states, makes pointless chip attacks, and sometimes waits forever. Each was patched with a veto.
2. **It plans against the wrong game.** The forward model ignores supply cutoffs, tiles, invest
   tiers, neutral restock and its own investing. The real game uses all of them.
3. **Rules duplicated by hand.** The main thread and `planWorkerMain` each keep a copy of the
   scoring and legality code. Combat sizing already disagreed with real combat once.
4. **Layers pulling against each other.** Scripted aggression (`aiHunt`, rope-a-dope, opening
   drive) fights the planner.
5. **Weak on farm-and-snowball.** Your tester banks troops, then strikes at about 7x local force.
6. **Cannot be debugged.** No record of why a move was chosen.

## 3. Principles

1. **One rules engine.** The real game, the search and the training all run the same code.
2. **One decision system.** No scripted behaviours. Hunting, waiting and investing come out of
   search, not switches.
3. **Commitments.** The AI holds an intent until it succeeds or the board changes enough to cancel it.
4. **Everything measured.** A change ships only if it beats the previous best in the arena.
5. **Every decision is explainable.** Each move is logged with its intent and reason.
6. **Old AI is kept as the baseline** (`Legacy`) so progress is a number, not a feeling.

## 4. Architecture

```
 Game (UI, netplay, shop, scenarios)
        |
        v
 Rules engine  <------------------------------+---------------+
 (single source of truth,                     |               |
  deterministic, headless)                    |               |
        ^                                     |               |
        |                                     |               |
 Agent: observe -> intent layer -> search -> executor      Arena / training
        |             (commitments)  (uses engine +        (same engine,
        |                             learned priors)       thousands of games)
        v
 Decision log (move, intent, reason, alternatives)
```

### 4.1 Rules engine
- Implements everything in `RULES.md`: orbs in flight, fractional troops, Lanchester combat with
  sequential resolution, terrain, supply components, tiles, invest tiers, cap upgrades,
  over-cap attrition, neutral restock, win and loss.
- Deterministic given a seed, with a fast state copy for search.
- Written in **Rust from the start** (decided). One crate, three targets: WASM for the game and
  the on-phone search, a native binary for the arena, and a Python extension (PyO3) for training.
- Game, search and training all import it. The two hand-synced copies disappear.
- A small build script compiles the WASM and inlines it into `index.html` (decided: build step OK),
  so the shipped app is still one self-contained file.

### 4.2 Action space
- `send(source, target, fraction)` for fractions like 1, 1/2, 1/4. Legality via the engine.
- `invest(state)` and `upgrade(state)` as first-class actions, chosen by the same search.
- `wait`.
- A "plan" is a set of sends that land together (a salvo is one decision, sized by the engine so
  the sequential Lanchester resolution actually flips the target).

### 4.3 Intent layer (fixes the ping-pong and aimless moves)
- An intent is a goal such as "take state X with a salvo", "mass at Y for 8 s", "invest in interior
  Z", or "defend W".
- The agent keeps its intents and re-evaluates them each tick: continue, cancel (target changed,
  force too small) or complete. A moved troop group cannot be undone without cancelling the intent.
- Intents come from the search and the learned prior. They are not hand-written rules.

### 4.4 Search
- Rolling-horizon search over the real engine: sample plans for the next 10 to 20 seconds,
  simulate, score, and execute the first steps.
- Opponent moves are sampled from a learned opponent policy, including your tester's style.
  Simultaneous clashes use regret matching so the AI is not predictable.
- Time budget per decision is the difficulty knob (plus cadence and salvo width). One brain at
  every tier.
- Fallbacks: if a Web Worker cannot start, a cheaper search on the main thread.

### 4.5 Learned parts
- **Graph network** over the territory graph: move priors and a position value. Trained by
  self-play against a league (past versions, scripted exploiters, a bot fitted to your tester).
- Starts with a hand-built value function (troops, territory, supply, growth, tempo) so the AI is
  strong before any training.
- Training is offline in PyTorch on your RTX 3080 (10 GB). That is enough for a graph network of
  this size; self-play generation on the CPU will be the bottleneck, which the native Rust engine
  is there to solve.
- The result ships as a weights file with a fixed schema. Inference runs inside the WASM engine.

### 4.6 Fairness stance (decided)
- Same economy, same information, same caps as the human. The AI wins by planning and precision,
  not by cheating.
- The AI can buy cap upgrades with troops, at the same costs and tiers as the human. Upgrading
  becomes one more action the search chooses.
- **Open:** whether the top cap stays 250 for both sides or is raised above 250 (section 11).

## 5. Phases

### Phase 0: baseline and measurement (about 1 to 2 days)
- Freeze the current AI as `Legacy` in the arena.
- Add a deterministic game-trace recorder to the live game (seeded RNG, every send, state hash).
- **Fix the in-game Data export** (currently not working for the tester). Diagnose it first: I
  need the symptom (nothing downloads, empty file, crash, too large). Until it works there are no
  real tester games, so the opponent model and the tester win-rate metric have no data.
- Fallback if the export cannot be fixed quickly: pull the logs off the phone over USB with the
  existing `tools/extract_idb_playdata.py`.
- Define the metrics (section 8) and measure Legacy on them.

**Exit:** a reproducible Legacy benchmark with numbers.

### Phase 1: rules engine (about 1.5 to 2 weeks, Rust)
- Write the engine from `RULES.md` as a Rust crate, with WASM, native and Python builds.
- Add the build script that inlines the WASM into `index.html`.
- **Conformance test:** replay recorded real-game traces through the engine and compare the board
  at each step. Zero drift on a large trace set, including supply, tiles and attrition.
- Unit tests for each rule (combat, supply, restock, caps).
- Add multiplayer rules (now identical to solo).

**Exit:** engine matches the live game on 100% of test traces.

### Phase 2: harness and baselines (about 3 days)
- Headless arena running the engine, any two agents, equal time budget, seeded maps.
- Simple baseline agents (greedy, random, scripted turtle/rush) to sanity check.
- Decision-log format and viewer.

**Exit:** arena runs 1,000 games in under an hour.

### Phase 3: agent v1, search only (about 1 to 2 weeks)
- Intent layer, rolling-horizon search, hand-built value function, salvo sizing from the engine.
- No neural net yet.
- Property tests: no ping-pong, no attack that cannot flip, no idle stall longer than N seconds.

**Exit:** v1 beats Legacy more than 60% in the arena, and passes the property tests.

### Phase 4: learned priors and value (about 2 to 4 weeks, GPU-dependent)
- Self-play data from the engine, graph network policy and value.
- Opponent model fitted from tester games.
- League training with exploiters; each candidate gated against the best so far.

**Exit:** v2 beats v1 more than 60%, with no regression against the exploiter set.

### Phase 5: integration into the game (about 1 week)
- Replace the game's sim code with the engine, and the old AI with the new agent.
- Keep netplay host-authoritative. AI runs on the host only.
- Remove old code: `planWorkerMain`, scripted layer, old nets, old opponent model.
- Tiers: Easy to Nightmare differ only in search time, cadence and salvo width.
- Rebuild the APK with the existing signing process (`BUILD.md`, same key).

**Exit:** the game plays end to end on a phone with no frame hitches during search.

### Phase 6: tester loop (ongoing)
- Tester plays, exports games, we refit the opponent model, retrain, regate, ship.
- Track tester win rate per build.

## 6. Evaluation and promotion gates

A new agent ships only if all pass:
1. **Strength:** beats the current best more than 55% (statistically significant) at equal time.
2. **No regression:** at least 95% against scripted turtle / rush / economist / farmer bots.
3. **Exploiter test:** a dedicated exploiter trained against the candidate wins less than 5%.
4. **Behaviour tests:** property tests pass (no oscillation, no suicidal sends, no stalls).
5. **Latency:** decision time fits the tier budget on the target phones (Pixel 10 Pro and
   Galaxy S24 Ultra), measured on a real device, with no dropped frames during search.

## 7. What gets deleted vs kept

| Deleted | Kept |
|---|---|
| `planWorkerMain` and its duplicated scoring code | Map, rendering, UI |
| Scripted layer (`aiHunt`, `aiInfight`, `coordinatedAssault`, rope-a-dope, opening drive) | Netplay, scenarios, shop |
| Old nets (`rl_policy.json`, `gnn_policy*.json`), `value_net.json` | Build and signing process |
| Old opponent model | Play-data recorder |
| Old training scripts that depend on the above | Arena ideas (rewritten on the new engine) |

## 8. Metrics

- Win rate vs Legacy, vs each bot, vs the fitted tester bot.
- Tester win rate per build (the number you actually care about).
- Exploitability: best exploiter's win rate against the agent.
- Behaviour: oscillations per game, wasted sends, longest idle stretch.
- Latency: p50 and p95 decision time on a phone.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Engine drifts from the real game | Conformance traces, run on every change |
| Search too slow on phone | Time-budget tiers; Rust/WASM port; precomputed priors |
| Training stalls or overfits to the tester bot | League with several exploiters; fresh tester games each round |
| Rebuild regresses the game feel | Legacy stays in the arena until v2 clearly beats it |
| Game still beatable at the top | Report measured numbers; keep the tester loop running |
| Single-file `index.html` makes the engine awkward | Add a small build step that inlines the engine (see section 11) |

## 10. Timeline (rough, one person working with me)

Phase 0-2: about 3 to 4 weeks (Rust from the start adds about a week over JavaScript-first).
Phase 3: 1 to 2 weeks. Phase 4: 2 to 4 weeks on the 3080. Phase 5-6: about 1 week, then ongoing.
Total to a first playable replacement: roughly 7 to 10 weeks.
Phase 3 gives a usable new AI before any training.

## 11. Decisions

Decided (2026-10-13):
1. **Fairness:** same economy and caps as the human. No bonuses.
2. **AI cap upgrades:** yes, the AI can upgrade caps like the human.
3. **Engine language:** Rust compiled to WASM from the start.
4. **Build step:** approved.
5. **GPU:** RTX 3080.
6. **Phone target:** Pixel 10 Pro and Galaxy S24 Ultra.
7. **Tester export:** not working today. Fixing it is now a Phase 0 task.

Still open:
- **Top cap value.** Today the human's top tier is 250. Either the AI shares that same 250 ceiling,
  or the ceiling is raised above 250 for both sides (new tiers, with costs to choose).
- **Data export symptom.** What goes wrong when the tester tries to export.

## 12. What happens next

Nothing starts until you approve. When you do, I begin Phase 0 and Phase 1 and report back at each
exit criterion.
