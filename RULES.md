# State.io rules (as implemented in `src/web/index.html`)

Spec for the rebuilt engine. Extracted by reading the code, not by running it. Line numbers are
for `src/web/index.html` at commit 74e63cf. Anything marked **[not in AI model]** is simulated by
the real game but absent from the planner's forward model (`snapStep` inside `planWorkerMain`).

## Board
- Map: 50 continental US states (default). Each state: `owner`, `troops` (fractional allowed under
  Depth), `growAcc`, `level` (cap tier), `type` (tile), `inv` (invest tier), `offSupply`.
- Adjacency: border graph from TopoJSON (`topojson.neighbors`). Distance = Euclidean between state
  centroids, in screen pixels.
- Owners: 0 = neutral, 1 = human, 2+ = AI teams.
- Setup (`newGame`, ~L1470): every state gets neutral troops `floor(rand(neutralMin, neutralMax))`.
  Human starts on a random state with `playerStart` troops. Each AI starts on the state farthest
  from all existing starters, with `enemyStart` troops. Then ~12% of neutral states (min 2) get a
  special tile: 40% factory, 35% fortress, 25% capital.

## Time
- Continuous. `update(dt)` each frame: growth, invest, supply, orb movement and arrival, AI.
- Orbs (troop groups in flight) move in a straight line at `ORB_SPEED` px/s. Native app:
  `rlDiag / 8` (about 8 s to cross the map diagonal). Web: 70.
- Orbs cannot be intercepted, redirected or recalled. They arrive when within `step + 2` px.

## Moves
- One action: send troops from state A to state B. Amount = all, or `floor(troops / denom)` (human
  split buttons) or `floor(troops * frac)` (AI). Must be at least 1.
- Legality (`isLegalMove`, adjacency mode, the default):
  - Target not owned by mover: must share a border with the source.
  - Target owned by mover: must be connected to the source through an unbroken chain of
    mover-owned states (BFS over adjacency).
  - Classic mode (solo toggle only): send anywhere from an owned state.
- Neutral states cannot send.

## Growth (per state, per second)
- `rate = growRateFor(owner) * tileGrowMult(state)`, then under Depth `* (1 + 0.4*inv)` and `* 0.4`
  if off-supply. Fractions accumulate in `growAcc`; +1 troop each time it reaches 1.
- Human rate = `cfg.growRate`. AI rate = `growRate * enemyGrow * (1 + rubber * aiSnow)`.
  Neutrals do not grow (except Depth restock below).
- Growth only applies while `troops < cap`.
- Tiles: capital x1.35 growth +40 cap; factory x1.6; fortress x1.0 +40 cap and absorbs 25% of
  incoming attacks. Tiles persist through capture. **[not in AI model]** (documented as such)

## Caps
- Human cap by upgrade level: 150 / 175 / 200 / 225 / 250. AI cap = `cfg.aiCap` or 150. Plus tile
  bonus. Human upgrade costs 40 / 70 / 100 / 130 troops (solo only).
- Incoming troops and sends can push a state above cap. Only passive growth is capped.

## Combat (`arrive`)
- Same owner: `troops += count` (reinforce).
- Different owner, classic: `troops -= dmg`; if below 0, ownership flips and troops = -troops.
  `dmg` = `count`, or `round(count * 0.75)` if the target is a fortress with troops.
- Different owner, Depth (default in solo): Lanchester exponent L = 1.5.
  `defEff = troops * terrMult * (offSupply ? 0.85 : 1)`.
  If `dmg > defEff`: flip, survivors = `round((dmg^L - defEff^L)^(1/L))`.
  Else: defender keeps `round((defEff^L - dmg^L)^(1/L) / (terrMult * supplyMult))`.
- Each orb resolves on its own when it lands. A salvo spread over arrival times is strictly weaker
  than one concentrated strike (this is why `coordinatedAssault` kept under-sizing attacks).
- Terrain `terrMult`: states with at most 2 neighbours 1.4, exactly 3 neighbours 1.2, else 1.

## Depth mechanics (solo only, on by default; off in netplay)
1. **Invest**: pay `25 + 15*inv` troops (25/40/55) for +40% growth per tier, max 3 tiers.
   Human taps manually. AIs invest by script (below).
2. **Terrain**: see above.
3. **Supply**: per owner, the largest connected component is the core. All other components are
   off-supply: grow x0.4, defend x0.85. Recomputed every 0.5 s. **[not in AI model]**
4. **Lanchester combat**: see above.
5. **Over-cap attrition**: above cap, loss/s = `0.06 * excess^2 / cap`. At cap 150: 200 troops
   loses 1/s, 300 loses 9/s, 400 loses 25/s.
6. **Neutral restock**: neutral states with under 30 troops gain 0.2/s up to 30. **[not in AI model]**

## AI invest (script, outside the planner)
Every 2 s each AI team invests in up to 2 states, only interior states (no enemy neighbours),
keeping a 15 troop reserve, preferring capital/factory then highest troops.
**[not in AI model]**: the planner does not choose investing at all.

## Win and loss
- Human loses when they own no states and have no orbs in flight.
- Human wins when all states are human-owned, or no AI owns a state or has an orb in flight.

## Per-level knobs (`LEVELS`)
`enemies`, `aiSpeed` (decision cadence), `aiAggro`, `neutralMin/Max`, `growRate`, `playerStart`,
`enemyStart`, `enemyGrow`, `workerMs` (search time per decision), `multiMove` (moves per decision),
`hunt` (how soon the AI attacks the human), `rubber` (catch-up growth).
Nightmare: 1 enemy, 22 vs 22 start troops, symmetric growth.

## What this means for the rebuild
- The engine needs: fractional troops, per-owner supply components, tiles, invest tiers, attrition,
  neutral restock, orb flight with straight-line travel, sequential Lanchester resolution.
- The human also gets cap upgrades (extra value spending troops); the AI never does. Not symmetric.
- The current planner ignores supply, tiles, invest, restock, and AI-side investing. The real game
  runs all of them. The search plans against a game that is not the one being played.

## Resolved details
- `aiCap` and `rubber` are never set in any `LEVELS` entry: AI cap is 150, catch-up growth is 0.
- Scenario mode changes setup only (teams, territory, troops; upgrades/tiles are toggles).
- Multiplayer (changed from earlier versions): host-authoritative. Depth mechanics, cap upgrades,
  tiles and adjacency movement are all on. Human guests get the upgrade-tier caps; AI slots keep 150.
  Shop perks and airdrops stay solo-only.
