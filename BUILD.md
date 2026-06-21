# State.io — build & the Scenario feature

`State.io` is a WebView-wrapped HTML5 strategy game. A thin native Android shell
(`classes*.dex`) loads the real game from `assets/web/index.html` via a virtual
host (`https://stateio.local/web/index.html`). All game logic — the map, the
troop simulation, and the trained reinforcement-learning AI — lives in that one
self-contained HTML file. The editable source is mirrored at
[`src/web/index.html`](src/web/index.html).

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
