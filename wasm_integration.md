# WASM+SIMD GNN inference — proof-of-concept results & integration proposal

**Status: PoC only. Nothing in `src/web/index.html` was touched.**

## What was built

- `tools/wasm_gnn/assembly/gnn.ts` — the gnn-v1 forward pass in AssemblyScript:
  embed+tanh → H hops of edge-restricted single-head attention (CSR adjacency:
  `offsets[N+1]` + neighbour indices, softmax over real border edges only) →
  update+tanh → per-candidate `head1(tanh)/head2` scoring of `(si,ti,moveFeats)`
  triples. Generic over `node_features` (7 or 10), `move_features` (3 or 4),
  `embed_dim` and `hops` — all driven by the JSON export, exactly like
  `packPolicy()` in `planWorkerMain`. Weight blob mirrors `packPolicy`'s
  row-major flattening: `embed.W/b`, then per hop `q/k/v/update` `W/b`, plus
  `head1/head2` passed separately to the scoring entry point.
- `tools/wasm_gnn/build.js` — builds both flavours with `asc -O3 --runtime stub`
  (`npm install --no-save assemblyscript`, v0.28.19 used):
  - `gnn_simd.wasm` (7,326 B; base64 ≈ 9.8 KB) — `f32x4` dot products +
    vectorized fast tanh.
  - `gnn_scalar.wasm` (5,100 B; base64 ≈ 6.8 KB) — MVP-only fallback, same
    source (`ASC_FEATURE_SIMD` folds the vector code out at compile time).
- `tools/bench_wasm_gnn.js` — the benchmark/verification harness (below).

Key implementation notes:

- **tanh was the actual hotspot, not the mat-muls.** A first WASM build calling
  AssemblyScript's f64 `Math.tanh` (musl port, ~60 ns/call × 72 calls/node) was
  only ~3x faster overall. Replacing it with a cephes-style f32 `expf`-based
  tanh (~2 ulp, vectorized 4 lanes at a time) roughly doubled the win.
- Dot products accumulate in f32 (SIMD lanes) with an f64 bias-add and tail;
  the JS path accumulates in f64 throughout. This is where the (tiny, measured)
  divergence comes from.
- No imports at all — `Math.*` compiles to AS-internal code, memory is exported.
  The module is fully self-contained, which is exactly what `planWorkerMain`
  needs.

## Benchmark results

Node v22.11.0, Windows 11, shipping net `src/web/gnn_policy.json`
(`nf=7, mf=4, d=24, hops=2, head1=32x52`). 200 random boards per size with
planar-ish (3–6 nearest-neighbour, symmetrized) adjacency; min-of-3 timed runs
after warmup. "JS-flat" is a faithful replica of the *shipping* planner path
(`packPolicy`/`linInto`/`gnnEmbedBoard`/`gnnMoveScore` inside `planWorkerMain`);
"JS-nested" is the pre-typed-array nested-array path. WASM embed numbers
*include* the per-board input copies into WASM memory (`setBoard`).

### ns per board embedding

| board | JS-nested | JS-flat | WASM scalar | WASM SIMD | SIMD vs JS-flat |
|------:|----------:|--------:|------------:|----------:|----------------:|
| N=16  |   334,252 | 343,738 |     193,192 |    56,331 |       **6.10x** |
| N=49  | 1,015,932 | 1,234,779 |   538,984 |   171,433 |       **7.20x** |

### ns per candidate-move score (embedding excluded)

| board | JS-nested | JS-flat | WASM scalar | WASM SIMD | SIMD vs JS-flat |
|------:|----------:|--------:|------------:|----------:|----------------:|
| N=16  |     5,175 |   5,720 |       2,815 |       767 |       **7.46x** |
| N=49  |     5,193 |   6,306 |       2,651 |       782 |       **8.06x** |

Scalar-WASM fallback: ~1.8–2.4x vs JS-flat on both metrics.

### Numeric divergence (every candidate score, 400 boards, 19,635 candidates)

| pair                    | max abs   | max rel   |
|-------------------------|-----------|-----------|
| WASM-SIMD vs JS-flat    | 4.56e-6   | 5.04e-4   |
| WASM-scalar vs JS-flat  | 2.89e-6   | 5.54e-4   |
| JS-nested(f64) vs JS-flat | —       | 4.14e-4   |

The large *relative* numbers are scores crossing zero — note the shipping f32
path itself shows rel 4.1e-4 against its own f64 sibling. Gate used (and
passed on all 19,635 candidates): `abs <= 1e-4 || rel <= 1e-4`. Absolute
divergence never exceeds **4.6e-6** — three orders of magnitude inside the
gate, and far below anything that could flip a move ranking in practice
(adjacent-candidate score gaps are ~1e-2..1).

Reproduce: `npm install --no-save assemblyscript && node tools/wasm_gnn/build.js && node tools/bench_wasm_gnn.js`

## Integration proposal

### 1. Shipping the bytes — base64 in the init message

`planWorkerMain` is built via `.toString()` and cannot reference outer scope,
but its **init message** can carry data. Proposal:

- Embed both wasm binaries as base64 string constants in `index.html`
  (main-thread scope, ~17 KB of text total — negligible next to the ~1 MB
  `gnn_policy.json`), and pass them in the existing init post:
  `worker.postMessage({ type:'init', policy, wasmSimdB64, wasmScalarB64 })`.
- Inside the worker's `init` handler, after `packPolicy()`:
  1. decode with `atob` (available in workers and in Node ≥16, so the arena
     works unchanged),
  2. `new WebAssembly.Instance(new WebAssembly.Module(bytes), {})` — try the
     SIMD module first; on **any** exception (`CompileError` on a non-SIMD
     engine) retry with the scalar module; on any exception again set
     `WASM = null`.
  3. Allocate the weight blob + scratch once (`alloc()`), copy the packed
     `FLAT` weights in. Per decision: copy node feats + CSR adjacency, call
     `embedBoard`, then batch candidate scoring through `scoreMoves`.
- Chrome's 4 KB synchronous-compile limit applies to the **main thread only**;
  workers may sync-compile any size, and Node has no limit — so the two hot
  paths (real Web Worker in the app, fake-self in the arena) both instantiate
  synchronously inside `init`. The rare `/api/plan` and tiny in-page fallback
  planners simply keep the existing JS path (they are already the "worker
  couldn't be built" degraded modes).

### 2. Graceful JS fallback

`gnnEmbedBoard`/`gnnMoveScore` keep their current JS bodies; the WASM path is a
pure fast-path override at the top:

```js
function gnnEmbedBoard(s, mover){
  if (WASM) return wasmEmbedBoard(s, mover);   // returns the same flat N*d view
  ... existing linInto code ...
}
```

Same public signatures, same flat `Float32Array` result contract (the WASM
version returns a subarray view over wasm memory — `gnnMoveScore` is its only
consumer, unchanged pattern from the 2026-07-11 typed-array pass). If
instantiation failed at init, `WASM` is null and nothing changes. The flat-MLP
(`rlScore`) and value-net paths stay in JS in the first cut (they are much
smaller nets; can be moved later with the same `linInto`-shaped kernel).

One structural choice to make at integration time: since `gnnMoveScore` is
called once per candidate from inside the search loops, either (a) keep
per-call `scoreMoves(M=1)` — still ~7x, wasm call overhead is small, or
(b) restructure the two candidate sweeps (`computeUntried`, `snapGreedyNet`)
to collect candidates and batch-score. (a) is the low-risk first step; the
bench's per-candidate numbers were measured batched, so quote (a)
conservatively as ~4–7x until measured.

### 3. Arena shim (`tools/selfplay_arena.js`)

**Verified in this PoC**: the exact fake-self pattern the arena uses
(`new Function('self', '(' + workerSrc + ')();')` + `self.onmessage({data:{type:'init',...}})`)
instantiates the module synchronously in Node with zero changes to the
pattern — `WebAssembly` and `atob` resolve from Node's globals, and the
init-decoded instance exports `alloc/embedBoard/scoreMoves/memory` correctly.

The only arena change needed: `makePlanner` must include the base64 strings in
its init message. Two options — read them from the extracted HTML with a
second regex (fragile), or have the arena read `tools/wasm_gnn/*.wasm` from
disk and pass them (same bytes; keep a build-stamp comparison to detect drift
between the checked-in wasm and what index.html embeds). Recommend the latter,
plus an `ARENA_NO_WASM=1` env toggle so any candidate-vs-champion match can be
re-run on the pure-JS path to bisect wasm-related regressions.

Note the asymmetry hazard: under the arena's **equal per-move time budget**, a
wasm-accelerated config gets more MCTS iterations per move. That is the point
(strength via speed), but when gating a *net* change, both A and B configs must
run with the same engine (both-wasm or both-JS) so the net comparison stays
clean.

### 4. Verification & gating (repo discipline)

1. **Parity test** (new, mirrors `test_gnn_batch_equiv.py`'s role): a Node
   script asserting WASM vs JS-flat candidate scores at `abs<=1e-4 || rel<=1e-4`
   over a few hundred randomized boards, both wasm flavours, plus a fixed-board
   golden check against `tools/gnn_net_torch.py` output (the existing JS↔Python
   contract point). `tools/bench_wasm_gnn.js` already implements the former —
   split the verification part into a fast CI-style check.
2. **Arena gate**: wasm-planner vs JS-planner with the *same* net under the
   standard equal-time budget — expect a strength gain from ~4–7x more
   simulations; require no regression vs scripted turtle/rush/economist. This
   is the promotion gate per CLAUDE.md.
3. **Device check**: the WebView on Android (Chrome ≥91 ships WASM SIMD;
   `minSdk 24` devices with older WebViews fall back — scalar wasm, then JS,
   automatically via the try-chain). Verify on the oldest available device that
   the fallback chain engages rather than crashes, and rebuild/sign the APK per
   BUILD.md.
4. **Line-ending discipline**: `.wasm` marked `-text` in `.gitattributes`
   (done in this PoC); the base64 constants in `index.html` are plain ASCII and
   inherit the existing LF rule.

## Risk assessment

- **Numerics: low.** Max abs divergence 4.6e-6 per candidate score; the
  existing pipeline already tolerates f64→f32 packing (same order of effect).
  The custom f32 tanh is the main deliberate deviation — it is ~2 ulp accurate
  and its effect is inside the measured envelope above.
- **Platform: low-medium.** SIMD needs Chrome/WebView 91+ (2021); the scalar
  module covers older MVP-only engines and still wins ~2x; JS remains the
  final fallback. The failure mode of a bad wasm blob is a caught exception at
  init → JS path, i.e. current behaviour.
- **Integration blast radius: medium.** The change lives inside
  `planWorkerMain` (worker copy) *and* the main-thread `gnnEvaluate`/brain-viz
  path uses `gnnEmbedBoard` too — per repo rules both copies must be updated
  in sync, or the main-thread copy can simply stay JS (viz is not hot).
  Memory ownership needs care: `embedBoard`'s returned H view aliases wasm
  memory and is invalidated by the *next* embed — same lifetime rule as the
  current shared `SCR` scratch, but worth an assertion during bring-up.
  `memory.grow` invalidates JS views — the wrapper must take views once after
  the last `alloc` (the PoC wrapper does this; keep the pattern).
- **Toolchain: low.** AssemblyScript is a dev-only dependency; the ~7 KB
  artifacts are committed, so the training/arena pipeline and the APK build
  need no new tools at all. Rebuilding wasm is only needed when the kernel
  changes, and `build.js` is deterministic.
- **Payoff:** the planner's per-iteration cost is dominated by board
  embeddings inside the MCTS rollouts; a 6–7x cheaper embedding directly
  multiplies simulations per thinking window on phones — the exact axis the
  equal-time arena rewards.
