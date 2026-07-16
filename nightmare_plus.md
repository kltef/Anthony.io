# Nightmare Plus — the asymptote tier

Design doc for a tier past Nightmare. Nightmare's contract was "no human realistically beats it."
Nightmare Plus drops the word *realistically* as far as a fair game mathematically allows: drive the
human win rate toward zero **asymptotically** — measured, not asserted — while keeping every stat
100% symmetric. The doctrine extends one more step:

> Impossible out-thinks you. Nightmare out-thinks you *and* out-clicks you.
> Nightmare Plus out-thinks you, out-clicks you, and has read your file.
> None of them ever out-cheat you.

**Fun disclaimer, up front (the play-tester's ceiling rule, 2026-07-12):** past a point, a stronger
AI stops being fun — "it's just not going to get fun if it can beat you all the time." Nightmare
Plus is deliberately past that point. It is the opt-in mountain peak for players who want to be
crushed, a lab instrument for measuring our distance from perfect play, and a marketing bullet. It
must never leak into the tiers below; the ladder from Easy to Grandmaster keeps its current feel
forever, and Nightmare stays the hardest tier a determined human should still occasionally beat.

## Why the asymptote is reachable (all measured in this repo, 2026-07-11 → 07-13)

- **Mathematical ceiling, for honesty:** a fair symmetric real-time game cannot be made *provably*
  unbeatable (symmetry ⇒ game value is a draw; simultaneous moves ⇒ even optimal play only
  guarantees expectation, not individual games — the matching-pennies caveat). The reachable target
  is "no measurable human strategy wins above noise." Everything below is engineering toward that.
- **Humans are the bottleneck, not the policy.** Winning human play runs ~2–5 meaningful actions/s
  in bursts, one attention spotlight, ~250ms reaction loop, degrading with fatigue. Nightmare
  already decides every 0.30s with 4-move salvos over the full worker pool; the human out-tempo
  exploit (442 moves in 243s, play-data 2026-07-04) was closed by fitting plan latency inside the
  cadence.
- **Policy strength has plateaued near parity — mechanics haven't.** Three independent
  measurements: every net+search config sits within a few points of 50% against every other at
  equal think time; the teacher–student gap (search over raw net) is down to ~7–8 pts
  (57.5%/40g, 2026-07-11); two 200-game ship gates in a row landed at 47.5%. The "three ceilings"
  analysis (2026-07-12): equal search equalizes, distillation fuel is nearly spent, small boards
  compress skill. Conclusion: past Nightmare, strength comes from mechanics, anti-human
  specialization, and grind — not from a bigger net.
- **The AI already discovers superhuman grind on its own.** The cap-dodge economy (growth stops at
  the 150 cap; transfers stack past it) is worth +60 troops/min for a two-state pair vs sitting
  capped — the planner rediscovers it live via search, no training needed, and runs it relentlessly
  (787 cap-shuffles and a 394-troop mega-stack across 120 recorded untimed games). A human *can*
  match this micro; no human *wants to* for a whole game. Stamina is a weapon the machine gets free.
- **Anti-human specialization works and has no parity ceiling.** The human-profile bot built from
  ANTHONY's play data (jointRate 0.90, half-stack sends 0.97, aggression 0.61): the retrained net
  beats it 97.5% vs the old net's 75%. Generic strength converges to ~50/50 vs strong opponents;
  anti-*you* strength doesn't, because a person is a fixed style with habits, not a moving
  equilibrium.

---

## The six levers

### Lever 1 — Mechanics past parity (cadence 0.20, salvo 6+)

Nightmare runs aiSpeed 0.30 / multiMove 4 / workerMs 240. Nightmare Plus targets **aiSpeed 0.20,
multiMove 6, workerMs ≤ 160** — sustained ~30 committed actions/second across every front at once.
Guard rail (learned the hard way): **plan latency must fit inside the cadence** or the tier is
latency-bound and a fast human out-tempos stale plans. That makes this lever gated on Lever 6
(WASM): the JS typed-array pass only bought 1.1–1.3× (V8 already optimizes packed arrays), so the
remaining 5–20× lives in a WASM/SIMD forward pass. Ship Lever 6 first; then raise these dials until
latency, not fairness, says stop.

### Lever 2 — Weaponized attention (the important one)

The human defends one crisis well. The machine's true edge is not acting faster — it is
**manufacturing more simultaneous crises than one attention spotlight can service**. Today the
planner treats multi-front pressure as incidental; make it an objective:

- Add a search bias term for *the number of distinct fronts the opponent must answer within one
  human reaction window* (~300ms). The MCTS already simulates futures; count threatened
  human-adjacent states in the root child's snapshot and bias priors toward high-fan-out lines.
- The salvo commit (onPlanWorkerMsg) already executes compatible multi-front moves; this lever
  makes the search *prefer* lines that create them, instead of merely permitting them.
- Evidence this is the right axis: the human's own winning style is exactly this (jointRate 0.90 —
  he does to the AI what the AI couldn't do back). Turn the weapon around.
- Arena-gateable like any planner change (HTML_A/HTML_B A/B), though the synchronous arena
  understates it — the profile-bot and live play are the real gauges.

### Lever 3 — Commitment-lag exploitation

Every human action is irreversible for the orb's flight time, and the OppModel already predicts
both "about to act from" and "about to hit" (soft/threat, conf-gated). Today that only biases the
ROOT priors (req.oppBias, 2026-07-11). Extend it into the tree:

- Model the human inside rollouts with the *fitted personal policy* (see Lever 4) instead of the
  generic snapGreedyNet, so lookahead anticipates *his* replies, not a textbook player's.
- Time strikes to commitments: raise priors on attacks that land during the window where the
  human's forces are provably in flight elsewhere (incoming-army features already exist per node —
  the v2 feature set even carries border pressure explicitly).
- Effect as felt by the player: it always arrives exactly where you just left. Nothing is peeked;
  everything used is on-screen information both sides can see.

### Lever 4 — The personalization stack ("it has read your file")

From the 300-games analysis (2026-07-12). Data path exists end-to-end: in-game logging → Data
export → phone_export/ → tools/parse_play_data.py.

- **4a. Warm-start the OppModel** from the fitted profile per player name, so confidence is high
  from move one instead of after ~5 observed moves. The opening is where games are decided; today
  the model is blind exactly there. (~30 games of data suffice for the 7-param model.)
- **4b. Behavioral clone in the training league** — per-phase conditional-logit imitation of the
  target human (needs ~300 games), trained against as an explicit best-response opponent with the
  no-forget screens guarding generality. `script:human` (aggregate stats) is the v0; the clone is
  contextual.
- **4c. Personal rollout model** — Lever 3's in-search human model, fed by 4b's fit.
- Ceiling note: a counter-style teaches the human a new style; the clone is always one meta behind.
  4a (live-updating) compounds; 4b/4c are point-in-time exploits. Refresh from fresh exports.

### Lever 5 — Stamina and the long game

Untimed games median 23s but the p90 runs 120s+ (measured 2026-07-12), and long games are where
human error accumulates — fatigue, tilt, attention lapses — while the machine plays move 400
exactly like move 4. Two applications:

- Style dial: a slightly defensive, economy-first posture (the cap-dodge grind the AI already
  loves) extends games at no cost to win probability and converts human degradation into wins.
- Training support: the long-game mix (GAMECAP env, --long-mix / --gate-long-mix, 2026-07-12)
  keeps late-game boards in-distribution so the hoard-lock failure (3 states, 384-troop stack,
  never attacks — the 267s autopsy) stays trained away. No candidate ships on sprint performance
  alone.

### Lever 6 — The WASM unlock (enabling tech, not a feature)

Everything above is throttled by inference latency in scalar JS. A WASM/SIMD forward pass
(5–20× on the embedding math) converts directly into: deeper search per decision at the same
budget, lower workerMs (enabling Lever 1's cadence), and headroom for bigger nets if the
real-map arena (11–49 state boards, tools/real_maps.json) ever gives capacity a reason to matter.
Needs a toolchain (emscripten or AssemblyScript) — the one big item deliberately deferred
2026-07-12. Ship as: wasm bytes passed through the worker init message, JS fallback kept for the
extraction/arena contract, bit-compatibility verified like the typed-array pass was.

---

## Measurement — the only honest definition of "crushes a human"

- **Primary metric: human win rate over real sessions**, from the play-data pipeline. Target:
  statistically zero (no wins above what seat-luck noise predicts) across 50+ session games.
- **Staged proxy: the profile bot.** Can anything shaped like the target human take 1 game in 50?
  (Current: retrained net concedes 2.5% to the ANTHONY profile at Nightmare settings.)
- **Lab instrument: exploiter ε.** Freeze the tier's full config (net + dials); train a dedicated
  best-response exploiter against it with everything the pipeline has. Its win rate above 50% is a
  certificate: "no strategy beats this by more than ε." This is the approximate-Nash program from
  the theory discussion — ε is the tier's true distance from unbeatable, and the number that goes
  in the lab notebook, not the changelog.
- The synchronous arena **cannot see Levers 1, 2, 3, or 5** (equal turns, no attention model, no
  fatigue). Arena gates remain the no-regression floor; they are not the success metric here.

## The tier entry (draft)

```js
// Nightmare Plus: the asymptote tier. Everything Nightmare has, with the remaining human-fairness
// margins spent: Ace-class-plus reflexes (aiSpeed 0.20), 6-move salvos, WASM-deep search inside
// the cadence, attention-pressure bias, and the personalized opponent model driving the tree.
// Stats stay 100% symmetric. Opt-in; never part of the auto-difficulty ladder.
{ name:'Nightmare+', enemies:1, aiSpeed:0.20, aiAggro:1.0, neutralMin:18, neutralMax:55,
  growRate:1.0, playerStart:22, enemyStart:22, enemyGrow:1.0, rl:true, plan:true, adapt:true,
  planDt:0.25, mctsK:9, tbudget:4000, workerMs:160, hunt:0.7, multiMove:6,
  attnPressure:1.0, personal:true },   // Lever 2 bias weight; Lever 3/4 personal models on
```

Coin multiplier: `Nightmare+: 5.0` in awardCoins. selectPolicyForLevel: loads the GNN (whatever
champion holds src/web/gnn_policy.json — currently the v2.2 net after repelling two challengers at
200-game gates). Excluded from auto-difficulty (autoAdjust caps at Nightmare). The difficulty
picker should label it honestly: "You will lose. Bring proof otherwise."

## Build order & gates

Phase 0 (prerequisite): **Lever 6** — WASM inference. Gate: bit-compatibility vs the JS scorer
(the ~1e-7 discipline), then latency benchmark proving workerMs 160 fits under aiSpeed 0.20.

Phase 1 (a day): tier entry + dials + **Lever 2** (attention-pressure bias in computeUntried /
salvo selection). Gate: no regression in the mixed-regime arena (real maps + long games,
OPENING_MIX/GAMECAP/REAL_MAPS envs), then profile-bot win rate vs Nightmare's.

Phase 2 (a day): **Levers 3+4a** — oppBias into the tree, personal rollout model, OppModel
warm-start from parsed exports. Gate: measured lift vs replayed human profiles; live play-test.

Phase 3 (needs ~300 exported games from the target human): **4b/4c** — the clone in the league,
best-response training with no-forget + anchor gates guarding generality.

Phase 4 (ongoing): **exploiter-ε runs** after each promotion — the lab metric that tells us when
to stop, which per the fun ceiling is: as soon as the humans stop winning, not later.

## Honest limits

- "Mathematically impossible" does not exist in a fair symmetric game (see the theory discussion:
  symmetry ⇒ draw value; simultaneity ⇒ expectation-only guarantees; the map generator still deals
  theoretically-lost seats — the cursed degree-2 corner openings are dealt losses no policy saves).
  Nightmare Plus claims an asymptote, and backs it with ε, not proofs.
- The personalization levers make losses feel *studied* rather than outplayed. That is the point of
  this tier and poison for every other. Keep the file-reading strictly behind the Nightmare+ label.
- Each anti-human exploit teaches the human; profiles go stale. The asymptote is approached by the
  live-updating pieces (OppModel, adaptive search), not the frozen ones.
- If a human still wins 1 in 50 after all six levers: that residue is seat luck plus
  matching-pennies variance, and it is mathematically irreducible. Print the loss screen with
  pride — it proves the game stayed fair.
