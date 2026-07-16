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

## The ML training pipeline (`tools/`)

The game's AI is a policy net trained offline and exported as JSON that `index.html` reads at
runtime. Training is a separate, GPU-oriented track — none of it runs in the app.

- **`selfplay_arena.js`** — the measurement bedrock. A headless Node arena that extracts the *actual*
  shipped MCTS planner (`planWorkerMain`) straight out of `src/web/index.html` via regex and plays
  two net configs head-to-head under an **equal per-move time budget**. Every training change is
  **gated** here: it must beat the currently-shipped net with no regression vs scripted
  turtle/rush/economist opponents. A candidate net is only promoted to `src/web/*.json` (and rebuilt
  into the APK) if it wins this gate.
- `train_rl.py` — SNES (gradient-free) policy trainer, 12-feature flat MLP, scripted opponent league.
- `train_value.py` — value-net trainer (Monte-Carlo regression).
- `train_distill*.py` — distill the Monte-Carlo expert into a net (various widths / 16-feature /
  value-bootstrapped variants).
- `train_selfplay_loop.py` / `train_selfplay_planner.py` — autonomous self-improving loops (greedy
  gate vs. real-planner gate respectively).
- `policy_net_torch.py` + `train_rl_torch.py` — PyTorch flat-MLP net + GPU loop that distills the
  real planner's MCTS **visit-count distribution** (AlphaZero-style policy target), with value-net
  co-training.
- `gnn_net_torch.py` + `train_gnn_torch.py` — the board-aware graph-attention net (`format:"gnn-v1"`)
  and its GPU training loop.
- `parse_play_data.py` — fit an opponent model from exported in-game play-data logs.

Typical invocations (see each file's header comment for full args):
```sh
python3 tools/train_rl.py [seconds]
python3 tools/train_gnn_torch.py --hours 5 --device cuda
node tools/selfplay_arena.js            # the gate / measurement harness
python3 tools/test_gnn_batch_equiv.py   # correctness gate: batched GNN loss+grads == per-decision reference
```

`test_gnn_batch_equiv.py` is a self-contained correctness check (no test framework) — run it after
changing the batched GNN training path; it verifies both loss values and `.grad` on every parameter
against the per-decision reference loop.

## Critical constraints when editing

- **LF line endings are mandatory** on `*.js *.py *.json *.html` (enforced via `.gitattributes`).
  `selfplay_arena.js` extracts `planWorkerMain` from `index.html` with an **LF-anchored regex**; a
  CRLF checkout makes extraction silently fail. On Windows, do not let an editor rewrite these to
  CRLF.
- **`planWorkerMain` runs in a Web Worker built via `.toString()`** and therefore *cannot reference
  any outside-scope function*. The worker holds its own copies of `computeUntried`,
  `snapGreedy`/`snapHeuristic`, `snapLegal`, etc. These are **manually kept in sync** with the
  main-thread versions — if you change the scoring/legal-move logic, change **both** copies (search
  for the duplicated function names; the worker copies are inside `planWorkerMain`).
- **Two net formats coexist.** Scoring functions dispatch on `rlPolicy.format`: `'gnn-v1'` computes
  per-territory node embeddings once per decision, anything else falls through to the flat-MLP path.
  Keep both branches working; the flat net is what currently ships.
- **Solo-only features are gated on `netMode === null`.** `netMode` is `null` (solo vs AI), `'host'`
  (runs the sim and broadcasts snapshots), or `'guest'` (renders snapshots, sends intents). Cap
  upgrades, special tiles, coin-shop perks, and Scenario apply to **solo only** so multiplayer
  snapshot sync stays untouched. Preserve these guards when adding features.
- **The planner has a fallback chain**: inline Web-Worker MCTS → `/api/plan` server (if a worker
  can't be built) → tiny in-page search. New planner work should keep all three paths coherent.
- **JS↔Python inference must stay contract-identical.** Nets are trained in Python and run in JS; the
  JSON export schema is the contract. Changes have historically been verified to ~1e-7 between the
  PyTorch and JS implementations before shipping (and Python↔JS features to ~2e-16). Preserve
  `to_json`/`from_json` schema compatibility.

## Landmarks in `src/web/index.html`

Search for these markers rather than scrolling:
- `SCENARIO BUILDER`, `startScenario`, `allocateTerritory` — the scenario authoring feature.
- `planWorkerMain` (~line 2179) — the off-thread MCTS worker (the extracted, self-contained copy).
- `rlEvaluate`, `computeUntried`, `snapGreedy`, `snapHeuristic` — move scoring / candidate generation
  (each exists in both the main thread and inside `planWorkerMain`).
- `PLAYER_CAP`, `capOf`, `tryUpgrade` — per-state cap upgrades. `TILE`, `assignTiles`, `tileGrowMult`
  — special tiles.
- `IS_NATIVE_APP` / `stateio.local` — native-app detection.
