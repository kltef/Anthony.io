# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**State.io** is a WebView-wrapped HTML5 strategy game shipped as an Android APK. A thin native
Android shell loads the entire game from `assets/web/index.html` over a virtual host
(`https://stateio.local/web/index.html`). **All game logic — map, troop simulation, and the trained
reinforcement-learning AI — lives in that one self-contained HTML file.** The editable source of
truth is `src/web/index.html` (~4600 lines); the APK's `assets/web/index.html` is a copy swapped in
at build time.

There is no bundler, package manager, or server dependency for the game itself — `index.html` is
plain hand-written HTML/CSS/JS that runs standalone in any browser.

## Repository layout

- `src/web/` — editable game source: `index.html` plus the trained nets `rl_policy.json` (flat MLP)
  and `gnn_policy.json` (board-aware graph-attention net). These are the files that get built into
  the APK.
- `build/` — APK repackaging: `repack.py`, the patched binary `AndroidManifest.xml`, `zipalign.py`,
  build intermediates (gitignored).
- `tools/` — the Python/Node ML training pipeline (see below). Not shipped in the app.
- `State.io_v*.apk` — the signed release artifacts, newest is the shipping one; older ones kept for
  reference.
- `phone_export/` — exported device data (play-data logs, localStorage) used by `parse_play_data.py`.
- `BUILD.md` — the authoritative build/signing/feature reference. **Read it before touching the
  build or signing.**
- `FINDINGS.md` — the AI investigation log: what was tried, what shipped, why the flat net is at its
  architectural ceiling, and the plan for the board-aware net + adjacency-movement retrain.

## Building the APK

The APK is rebuilt **on top of the previous signed APK** (which already carries all resources and the
patched manifest) by swapping in the current web assets and re-signing — no Android SDK build-tools
required, just a JDK plus the dependency-free `apksig` library.

```sh
python3 build/repack.py            # -> build/unsigned.apk (swaps in src/web/{index.html,rl_policy.json,gnn_policy.json} + patched manifest)
# then sign v2+v3 with apksig (a ~30-line Java driver around ApkSigner) and verify.
```

`repack.py` copies every entry from the previous APK, replacing the web assets and dropping the old
`META-INF/*.SF/.RSA/.MF`. **Each entry's original `compress_type` must be preserved — `resources.arsc`
MUST stay STORED (uncompressed)** or the app fails to install
(`INSTALL_PARSE_FAILED_RESOURCES_ARSC_COMPRESSED`). `minSdk` 24 / `targetSdk` 34 require a v2/v3
signature; a v1-only (JAR) signature is rejected on install.

### Signing key — do not regenerate (see BUILD.md)

App updates only install over an existing install when signed with the **same** key. The `keytool
-genkeypair` command in BUILD.md makes a **new random key each run** — running it breaks the update
path for existing installs. The shipping line uses one specific keystore (alias `stateio`,
store/key password `android`, signer cert SHA-256 `81:2E:C5:...:15:FA`). That `release.keystore` is
deliberately kept **out of the repo**; keep a private backup and sign every build with it.

## The AI: the Rust engine (`engine/`)

Since web build 43 every AI decision comes from `engine/`, a dependency-free Rust crate compiled to
WebAssembly and inlined into `index.html` (the `ENGINE_WASM_B64` line). `updateAI` sends the board
to a Web Worker running the engine and applies the orders it returns through the normal
`sendTroops` / `doInvest` / `doUpgrade` gates. The old MCTS planner, neural nets, opponent model and
AI-brain panel were removed. See `PLAN_NEW_AI.md` for the design and `RULES.md` for the rules.

- `engine/src/rules.rs` — the rules (single source of truth). `engine/src/policy.rs` — the fast
  scripted policy. `engine/src/search.rs` — the search agent (simulates each candidate order in the
  real rules). `engine/src/ffi.rs` — the WASM interface. `engine/data/policies.txt` — tuned policies.
- Each `LEVELS` entry names a `brain` (`base`, `t2`, `t4`, `search`) and a `think` cadence.

```sh
# from the repo root
python engine/build.py                                   # build the WASM and inline it into index.html
node engine/tools/conformance_gen.js 60                  # record traces from the original JS simulation
node engine/tools/vs_legacy.js --boards 50               # new AI vs the original AI
# from engine/
cargo build --release                                    # native tools
./target/release/conformance target/conformance          # replay the traces through the engine
./target/release/arena search rush --games 120           # head-to-head, both seats per board
./target/release/tune --rounds 5                         # evolve scripted policies by league play
```

**Rule changes must go in `engine/src/rules.rs` AND the game's JS simulation** (the game still
simulates in JS; the engine is checked against it by the conformance traces). Run the conformance
check after any rule change.

The `tools/` Python/Node training pipeline belongs to the removed AI and is kept only for reference.

## Critical constraints when editing

- **LF line endings are mandatory** on `*.js *.py *.json *.html` (enforced via `.gitattributes`).
  The conformance and comparison tools extract functions from `index.html` by text matching. On
  Windows, do not let an editor rewrite these to CRLF.
- **Multiplayer now runs the full rule set** (Depth mechanics, cap upgrades, special tiles, adjacency
  movement). `netMode` is `null` (solo vs AI), `'host'` (runs the sim and broadcasts snapshots), or
  `'guest'` (renders snapshots, relays intents). Only the host simulates; the snapshot carries each
  state's level/type/invest/off-supply (`l`,`y`,`v`,`x`), and guest upgrade/invest presses go to the
  host as action intents (`k:'up'|'inv'`). Still solo-only: coin-shop perks, airdrops, Scenario.
- **The AI falls back** to a simple legal-move heuristic in `updateAI` when WebAssembly or Web
  Workers are unavailable (`engFailed`).

## Landmarks in `src/web/index.html`

Search for these markers rather than scrolling:
- `SCENARIO BUILDER`, `startScenario`, `allocateTerritory` — the scenario authoring feature.
- `NEW AI: the Rust engine`, `engineSyncWorld`, `engineRequest`, `engineApply`, `updateAI` — the AI driver.
- `PLAYER_CAP`, `capOf`, `tryUpgrade` — per-state cap upgrades. `TILE`, `assignTiles`, `tileGrowMult`
  — special tiles.
- `IS_NATIVE_APP` / `stateio.local` — native-app detection.
