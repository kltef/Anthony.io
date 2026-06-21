# State.io — build & the Scenario feature

`State.io` is a WebView-wrapped HTML5 strategy game. A thin native Android shell
(`classes*.dex`) loads the real game from `assets/web/index.html` via a virtual
host (`https://stateio.local/web/index.html`). All game logic — the map, the
troop simulation, and the trained reinforcement-learning AI — lives in that one
self-contained HTML file. The editable source is mirrored at
[`src/web/index.html`](src/web/index.html).

## Per-team top bar

The top strength bar used to lump every enemy into one red segment (blue = you, red = all AIs,
grey = neutral). It now renders **one segment per team in that team's own colour** (you, then
each AI colour, then neutral). `updateBar()` tallies troops per owner and rebuilds the segment
DOM only when the set of live owners changes (e.g. a team is eliminated); otherwise it just
nudges widths.

## Mobile tap fix & Airdrop arm-then-tap

- **Mobile tap selection bug (root cause).** `gameMove()` flagged the gesture as a *drag* on
  **any** pointer movement, so a finger tap's few px of jitter made `gameUp()` clear the
  selection. On a phone, tapping your own state therefore never selected it — leaving the
  **Airdrop** and **Upgrade** buttons greyed out and tap-then-tap sending unreliable. Fixed with
  a **12px tap dead-zone** (`TAP_SLOP`): the gesture only becomes a drag once the pointer leaves
  that radius (`downCss` recorded in `gameDown`). Synthetic test clicks have zero jitter, which is
  why this slipped past earlier automated runs.
- **Airdrop is now arm-then-tap.** Instead of "select a state, then press a greyed-out button,"
  tap **🪂 Airdrop** to *arm* it (it highlights and reads "🪂 Tap a state…"), then tap one of your
  states to drop +25 troops there and spend a charge. No hidden select-first step. (`airdropArmed`,
  handled in `gameUp`/`refreshHud`.)

## Scenario back button & orb-shape perk

- **Fixed "← Back" on menu screens.** The in-game `❮` arrow only helps once a match is
  running; on the Scenario setup screen in landscape on mobile the page scrolls and the mode
  tabs scroll out of view, leaving no way back. A `position:fixed` **← Back** button
  (`#menuBack`, z-index above the overlay) now shows on any non-default menu screen and returns
  to the vs-Computer menu — it stays put no matter how far the panel scrolls. Visibility is
  driven by `updateMenuBack()` on every overlay show/hide and `setMode()` change.
- **Orb-shape perk (replaces the old colour perk).** The cosmetic team-colour option is gone;
  the shop now sells an **orb shape**: Circle (default), Triangle, Jet (F-22), Bomber, Diamond.
  Your own orbs (`t.owner===myOwner`) render in the chosen silhouette, rotated to point at their
  target; everyone else's stay circles. It's a purely local cosmetic, so it works in every mode
  with no networking changes. See `ORB_SHAPES`, `drawOrbShape()` (also used for the shop's canvas
  previews), and `PERK.orbShape`/`orbUnlocked`.

## Cap upgrades, special tiles & a coin shop

Three linked single-player systems (all gated to `netMode === null`, so multiplayer and
its snapshot sync are untouched):

- **Per-state cap upgrades.** Tap one of your states to select it, then the on-screen
  **⬆ Upgrade (cost)** button spends troops to raise that state's troop cap *above* the
  classic 150 (tiers 150 → 175 → 200 → 225 → 250). AIs never upgrade — it's a player-only
  power lever. See `PLAYER_CAP`, `capOf()`, `tryUpgrade()`.
- **Special tiles.** Every solo game sprinkles ~12% of neutral states with **Capital ★**
  (more growth + cap), **Factory ⚙** (most growth), or **Fortress 🛡** (absorbs 25% of
  attacks + cap). Tiles persist through capture, and the heuristic + RL AI both get a
  scoring nudge so they actively fight for them. See `TILE`, `assignTiles()`, `tileGrowMult()`,
  and the bias terms in `updateAI()`/`rlEvaluate()`.
- **Coins & shop.** Finish a solo game to earn ~50 coins (scaled by difficulty, +40 on a win,
  +25% with Investor). The **🪙 Shop** (modelled on the leaderboard modal) sells: extra
  starting state (×3, escalating), +10 start troops, pre-upgraded capital, starting Capital
  tile, Boom Start, Investor, Reinforcement Airdrop charges, and a cosmetic colour. Persisted
  in `localStorage`; perks apply in **vs Computer only** (never Scenario or online). The
  Scenario builder gets **Cap upgrades** / **Special tiles** on-off toggles.

## New feature: Scenario Builder

A new **Scenario** tab on the main menu lets you author a starting board and drop
the AI into it to see how it performs.

- **Choose the map** (East Coast / Midwest / West Coast / Continental U.S.).
- **Choose the AI brain** that drives every AI team (Easy → Grandmaster, including
  the trained RL "Master" policy and the rollout-planner "Grandmaster").
- **Define 2–6 teams**, and for each one set:
  - its **starting share of the map** (e.g. one team 50%, another 25%, the AI 25%), and
  - whether it is controlled by **You** or by the **AI**.
- Any share of the map you don't assign starts **Neutral**.
- Set **every** team to AI to sit back and **watch the AIs fight** (observer mode).
  A standing-army readout and a per-team proportion bar preview the split before
  you start, and the end screen announces which colour won.

One-tap **presets** are included, e.g. `50 / 25 / 25` (the canonical example),
`You vs AI`, `Surrounded`, `Underdog`, `Watch: AI duel`, and `Watch: 4-way`.

### How territory is laid out

Teams are seeded far apart (reusing the game's `farthestPick`) and then grown as
contiguous blobs: on each step the neediest team (lowest owned/target ratio) is
handed the nearest still-unclaimed state. This makes the requested shares hold up
both numerically and visually. Each team's seed state is its "capital" and starts
at full strength; its other states hold a modest garrison; unclaimed land keeps
the usual neutral spread. Scenario games are intentionally **not** ranked on the
leaderboard.

The implementation is entirely within `assets/web/index.html` (search for
`SCENARIO BUILDER`, `startScenario`, `allocateTerritory`).

## Other additions

- **Landscape support.** The native shell was locked to portrait
  (`android:screenOrientation="userPortrait"`, value `12`). It's now
  `fullUser` (value `13`), so the app rotates to any orientation the device
  allows. The activity already declares `configChanges="…|orientation|screenSize|…"`,
  so rotating doesn't recreate the activity — the WebView just resizes and the
  game's existing `resize()` handler reflows the map. The menu overlay is now
  scrollable and tightens its layout on short (landscape) viewports
  (`@media (max-height: 560px)`). This is a one-byte patch of the binary
  `AndroidManifest.xml` (the `screenOrientation` int at file offset `2380`,
  `0x0C` → `0x0D`).
- **0.5× game speed.** The speed button now cycles
  `0.5× · 1× · 2× · 3× · 5× · 10×`. The loop runs whole simulation sub-steps and
  then a fractional step for the remainder (`update(dt * frac)`), which slows the
  whole sim uniformly — handy for watching the AI think in slow motion.

## Rebuilding the APK

The repo ships the rebuilt, signed artifact as
`State.io_v1.3-scenarios.apk` (the original `State.io_v1.2.apk` is kept for
reference). To rebuild after editing the web source:

```sh
# 1. Unpack the base APK
unzip State.io_v1.2.apk -d apk_extract

# 2. Apply your changes to apk_extract/assets/web/index.html
#    (or copy in src/web/index.html)

# 3. Repackage — resources.arsc MUST stay STORED (uncompressed)
#    (see the python snippet in this repo's history) then:
zipalign -f -p 4 unsigned.apk aligned.apk

# 4. Sign with APK Signature Scheme v2+v3 (required for targetSdk 34)
apksigner sign --ks release.keystore --ks-key-alias <alias> \
  --v2-signing-enabled true --v3-signing-enabled true \
  --out State.io_v1.3-scenarios.apk aligned.apk

# 5. Verify
apksigner verify --verbose State.io_v1.3-scenarios.apk
```

`minSdkVersion` is 24 and `targetSdkVersion` is 34, so a v2/v3 signature is
required — a v1 (JAR) signature alone would be rejected on install by modern
Android. The committed APK is signed with a throwaway debug key; re-sign with
your own release key before distributing.
