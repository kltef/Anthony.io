# Nightmare Ω — closing the last doors

Third doc in the line (nightmare.md → nightmare_plus.md → this). The first made the AI out-click
humans; the second made it read their file. This one is the terminal program: enumerate every way
a human can still win, close each one that a fair game permits closing, and measure the residue
honestly. Target: a tier where a human victory is not "hard" but *statistically indistinguishable
from zero* — while every stat stays symmetric.

> Impossible out-thinks you. Nightmare out-clicks you. Nightmare Plus has read your file.
> Nightmare Ω closes the doors. What remains open is mathematics, and it is counted.

## The five doors (from the 2026-07-14 analysis) and their status

A good human beats the current system through exactly five doors:

1. **Invent an unseen strategy** (the turtle/economist/farmer/out-tempo lineage) — CLOSABLE
2. **Poison the opponent model** (feed it fake habits, then pivot) — CLOSABLE
3. **Out-update it between retrains** (human learns nightly; policy is frozen) — CLOSABLE
4. **Win the map lottery** (cursed/blessed seats are dealt losses; measured repeatedly) — CLOSABLE,
   and this is the big one nobody noticed: seat luck is a property of the MAP GENERATOR, not the game
5. **Guess right on simultaneous commits** (matching pennies) — IRREDUCIBLE per decision, but see
   the aggregation argument below: it nearly vanishes per GAME

The mathematical ceiling (2026-07-14 discussion) stands: a fair symmetric real-time game admits no
per-game no-loss proof. But the practical floor is much lower than it first appears:

**The aggregation argument.** Door 5 is a coin flip per simultaneous commitment — but a game
contains dozens to hundreds of them, and the outcomes aggregate. One lucky read wins a skirmish;
winning the game requires the human's *sum* of reads to beat the AI's sum, and the law of large
numbers crushes that as game length grows. Combined with door 4 closed (symmetric seats), the
theoretical human win floor is not "a few percent" — it decays toward zero with decision count.
The AI's stamina lever (longer games, more aggregated decisions) actively drives it down.

---

## The program

### Ω1 — Kill the map lottery (door 4): mirror-symmetric duel maps

The single largest remaining human win source is dealt, not earned: degree-2 corner seats behind
50-troop walls are theoretically lost positions (loss autopsies, 2026-07-12). For the Ω tier only,
generate PERFECTLY MIRROR-SYMMETRIC maps: reflect the board, swap the seats, mirror every neutral
garrison. Both players face the identical strategic problem; the game value is exactly a draw by
construction; every loss is now earned. Implementation: a symmetrized genBoard variant (synthetic
Delaunay reflected about an axis; real maps get mirrored-pair spawns with mirrored garrisons —
approximate for map shape, exact for the resource layout, which is what the autopsies say matters).
This is fully fair — fairer than the current game — and it removes the human's best door.

### Ω2 — Dial to the latency floor (door: none — pure pressure)

WASM (5.5x measured) moved the latency floor. Current Nightmare+ dials (0.20s / 160ms / salvo 6)
were set pre-WASM. The floor is now: plan latency ≈ budget, so cadence 0.12–0.15s with 80–100ms
budgets searches DEEPER than pre-WASM Nightmare did at 240ms while acting nearly twice as fast.
Tree reuse (83% of decisions promote a subtree) + pondering between commits mean the effective
tree is continuous — the AI never actually stops thinking. Calibrate against the profile bot
(2026-07-15 tournament) and cap salvo width only where over-extension shows.

### Ω3 — Armor the opponent model (door 2)

The personalization stack reads the human; meta-deception is its attack surface. Armor:
- **Confidence decay on miss**: every prediction the human falsifies decays conf multiplicatively
  (fast down, slow up). A player feeding fake habits pays with an AI that simply stops trusting
  the model and reverts to the (already superhuman) generic game — the poison costs the poisoner.
- **Style-shift detection**: track a short-window profile (jointRate/frac/aggression over the last
  N moves) vs the long-window one; on divergence, reset conf and re-learn. The pivot IS the tell.
- The live logit (7 params) relearns in ~5 moves — faster than any human can maintain a fake
  style while also playing well enough to survive Ω2's tempo.

### Ω4 — Never be readable (door 5 hygiene)

The human's only stable strategy against a stronger opponent is prediction. Deny it:
- The search's PUCT visit distribution already randomizes near-tie decisions; keep committing the
  argmax for strength but RANDOMIZE TIMING within the cadence window (a fixed decision rhythm is
  the most human-readable pattern the AI has).
- Vary opening lines deliberately: seed tiny root noise in the first two decisions only —
  strength-neutral (openings are near-symmetric under Ω1) but kills opening prep, the one form of
  between-session human learning (door 3) that survives Ω3.

### Ω5 — Endgame exactness (door 1, partial)

Novel human strategies live in the value function's blind spots. Two hardenings:
- The board-aware value head (shipped 2026-07-14, training since) — its whole purpose is pricing
  slow strategies the flat net missed. Keep co-training it on every dump.
- **Tablebase-lite**: 2-state-vs-2-state positions (and simpler) are exactly solvable with the
  real physics (growth, travel, cap) by forward dynamic programming over quantized troop counts.
  At most a few million states; solve offline, ship as a small lookup, consult at search leaves.
  The endgame becomes literally perfect — the phase where human grinders historically survive.

### Ω6 — The certification loop (measurement, not a lever)

Per game theory (2026-07-14): approximate-equilibrium play is certified by EXPLOITABILITY, not
win rates. Freeze the full Ω config; train a dedicated best-response exploiter against it with the
whole pipeline (it gets the Ω config as its anchor-opponent and unlimited nights); its win rate
above 50% is ε — the tier's true distance from unbeatable. Publish ε in the repo per release.
The human-facing metric stays: tester session win rate, target statistical zero. When Fader's
best remaining weapon produces fewer wins than Ω1 removed, the program is complete.

## What this does NOT do

- No stat asymmetries. Ω is *more* symmetric than the base game (Ω1). Doctrine holds: it
  out-thinks, out-clicks, out-reads, and out-lasts — it never out-cheats.
- No per-game guarantee. A human can still win one: stack every coin flip in one short game on a
  volatile map and the aggregation argument hasn't had time to bite. The design response is
  Ω2+stamina (longer effective games) and honesty about the residue.
- No fun. This is the observatory-grade instrument at the top of the mountain, explicitly past
  the play-tester's fun ceiling (2026-07-12). It exists to answer the research question and to be
  pointed at, not to be the game. The ladder below it never changes.

## Status (2026-07-15)

Ω2, Ω3, Ω4, and Ω5-playout SHIPPED in State.io_v2.4.1-omega-testkey.apk (mirror-gated 51.7%/60g,
no regression; the Ω2 dial set pre-validated 60/60 vs the tester-profile bot). **Ω1 declined by
design**: the owner keeps the map lottery — dealt losses and dealt wins stay part of the game's
character; the tier does as well as possible on whatever board it's dealt. Ω5 full tablebase and
Ω6 exploiter-ε remain open (Phase 2/3).

## Build order

Phase 0 (pre-deadline, if the tester session demands it): Ω2 dial calibration from the
2026-07-15 tournament + Ω3 confidence decay (a dozen lines in OppModel.update). Ship in v2.4.

Phase 1 (days): Ω1 symmetric maps (arena + game genBoard variants, gate on symmetric boards),
Ω4 timing jitter + opening noise.

Phase 2 (a week, background): Ω5 tablebase solver + leaf lookup; value head keeps training.

Phase 3 (ongoing): Ω6 exploiter-ε per release. The number goes in this file:

    ε history:  (unmeasured — first certification pending)

## Honest limits, final form

Assume every phase lands. What beats Nightmare Ω?
- A short game on a volatile map where three big simultaneous guesses all break human — call it
  low single digits percent, shrinking with game length.
- A strategy outside every model AND fast enough to win before the within-game learners adapt —
  historically the human specialty, now racing Ω3's 5-move relearn window and Ω5's endgame wall.
- Nothing else. And when one of those wins lands once in fifty sessions, print the loss screen
  with pride: it is the signature of a fair game, measured to the bottom.
