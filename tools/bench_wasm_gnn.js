#!/usr/bin/env node
// Benchmark: GNN policy inference — current JS planner path vs WASM (SIMD + scalar).
//
//   node tools/bench_wasm_gnn.js
//
// Loads the shipping net (src/web/gnn_policy.json, falling back to
// tools/gnn_policy_new.json), generates 200 random 16-state and 200 random
// 49-state boards with plausible planar-ish adjacency, and times (after warmup):
//   (a) JS-flat   — faithful replica of the planner's flat-Float32Array path
//                   (packPolicy/linInto/gnnEmbedBoard/gnnMoveScore inside
//                   planWorkerMain in src/web/index.html)
//   (b) JS-nested — the original nested-array path (pre-typed-array baseline)
//   (c) WASM SIMD / WASM scalar (tools/wasm_gnn/gnn_{simd,scalar}.wasm)
// and verifies every candidate score matches JS-flat to <= 1e-4 relative.
//
// Reported: ns/board-embedding, ns/candidate-score, speedup ratios, max divergence.

'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const NEUTRAL = 0, PLAYER = 1;   // src/web/index.html:537

// ---------- policy ----------
let POL, polPath;
for (const p of ['src/web/gnn_policy.json', 'tools/gnn_policy_new.json']) {
  const f = path.join(ROOT, p);
  if (fs.existsSync(f)) { POL = JSON.parse(fs.readFileSync(f, 'utf8')); polPath = p; break; }
}
if (!POL || POL.format !== 'gnn-v1') { console.error('no gnn-v1 policy found'); process.exit(1); }
const NFEAT = POL.node_features || 7;
const MFEAT = POL.move_features || 3;
const D = POL.embed.b.length;
const HOPS = POL.attn.length;
const HO = POL.head1.b.length;
const FN = POL.feat_norm || 50;
console.log(`policy: ${polPath}  nf=${NFEAT} mf=${MFEAT} d=${D} hops=${HOPS} head1=${HO}x${2 * D + MFEAT}`);

// ---------- (b) JS-nested: original array-of-arrays path (src/web/index.html gnnLin et al.) ----------
function gnnLin(W, b, x) {
  const out = new Array(b.length);
  for (let o = 0; o < b.length; o++) { let acc = b[o]; const row = W[o]; for (let i = 0; i < row.length; i++) acc += row[i] * x[i]; out[o] = acc; }
  return out;
}
function gnnTanh(v) { return v.map(Math.tanh); }
function gnnAttnHop(layer, h, adj) {
  const N = h.length, dim = h[0].length, scale = Math.sqrt(dim);
  const Q = h.map(x => gnnLin(layer.q.W, layer.q.b, x)), K = h.map(x => gnnLin(layer.k.W, layer.k.b, x)), V = h.map(x => gnnLin(layer.v.W, layer.v.b, x));
  const out = new Array(N);
  for (let i = 0; i < N; i++) {
    const nbrs = adj[i]; let agg = new Array(dim).fill(0);
    if (nbrs.length) {
      const scores = nbrs.map(j => { let sc = 0; for (let d = 0; d < dim; d++) sc += Q[i][d] * K[j][d]; return sc / scale; });
      const mx = Math.max(...scores), exps = scores.map(sc => Math.exp(sc - mx)), sum = exps.reduce((a, b) => a + b, 0);
      for (let k = 0; k < nbrs.length; k++) { const w = exps[k] / sum, vj = V[nbrs[k]]; for (let d = 0; d < dim; d++) agg[d] += w * vj[d]; }
    }
    const cat = h[i].concat(agg); out[i] = gnnTanh(gnnLin(layer.update.W, layer.update.b, cat));
  }
  return out;
}
function nestedEmbed(nfRows, adj) {
  let h = nfRows.map(x => gnnTanh(gnnLin(POL.embed.W, POL.embed.b, x)));
  for (const layer of POL.attn) h = gnnAttnHop(layer, h, adj);
  return h;
}
function nestedMoveScore(h, si, ti, moveFeats) {
  const x = h[si].concat(h[ti]).concat(moveFeats);
  const z = gnnTanh(gnnLin(POL.head1.W, POL.head1.b, x));
  return gnnLin(POL.head2.W, POL.head2.b, z)[0];
}

// ---------- (a) JS-flat: faithful replica of the planner-speed pass (packPolicy/linInto) ----------
var FLAT = null, SCR = null;
function packLin(l) {
  var no = l.b.length, ni = l.W[0].length, W = new Float32Array(no * ni);
  for (var o = 0; o < no; o++) { var row = l.W[o]; for (var i = 0; i < ni; i++) W[o * ni + i] = row[i]; }
  return { W: W, b: Float32Array.from(l.b), no: no, ni: ni };
}
function packPolicy() {
  FLAT = {}; SCR = null;
  FLAT.embed = packLin(POL.embed);
  FLAT.attn = POL.attn.map(function (a) { return { q: packLin(a.q), k: packLin(a.k), v: packLin(a.v), update: packLin(a.update) }; });
  FLAT.head1 = packLin(POL.head1); FLAT.head2 = packLin(POL.head2);
}
function linInto(fl, x, xo, out, oo, tanh) {
  var ni = fl.ni, no = fl.no, W = fl.W, b = fl.b;
  for (var o = 0; o < no; o++) {
    var acc = b[o], base = o * ni;
    for (var i = 0; i < ni; i++) acc += W[base + i] * x[xo + i];
    out[oo + o] = tanh ? Math.tanh(acc) : acc;
  }
}
function gnnScratch(N, d, nf) {
  if (SCR && SCR.N >= N && SCR.d === d && SCR.nf === nf) return SCR;
  var hn = FLAT.head1.ni, ho = FLAT.head1.no;
  SCR = { N: N, d: d, nf: nf, NF: new Float32Array(N * nf), H: new Float32Array(N * d), H2: new Float32Array(N * d),
          Q: new Float32Array(N * d), K: new Float32Array(N * d), V: new Float32Array(N * d),
          CAT: new Float32Array(2 * d), SCW: new Float32Array(N), HEAD: new Float32Array(hn), Z: new Float32Array(ho) };
  return SCR;
}
// NF rows are precomputed for the bench (feature extraction is identical for every impl
// and is NOT what we're measuring); flatEmbed = the linInto/attention core of gnnEmbedBoard.
function flatEmbed(NFbuf, N, adj) {
  var d = FLAT.embed.no, nf = SCR.nf, i, di;
  var H = SCR.H;
  for (i = 0; i < N; i++) linInto(FLAT.embed, NFbuf, i * nf, H, i * d, true);
  var scale = Math.sqrt(d);
  for (var l = 0; l < FLAT.attn.length; l++) {
    var L2 = FLAT.attn[l], Q = SCR.Q, K = SCR.K, V = SCR.V, H2 = SCR.H2, CAT = SCR.CAT, SCW = SCR.SCW;
    for (i = 0; i < N; i++) { linInto(L2.q, H, i * d, Q, i * d, false); linInto(L2.k, H, i * d, K, i * d, false); linInto(L2.v, H, i * d, V, i * d, false); }
    for (i = 0; i < N; i++) {
      var nbrs = adj[i], nn = nbrs.length, base = i * d, k2;
      for (di = 0; di < d; di++) { CAT[di] = H[base + di]; CAT[d + di] = 0; }
      if (nn) {
        var mx = -1e30;
        for (k2 = 0; k2 < nn; k2++) {
          var jb = nbrs[k2] * d, acc = 0;
          for (di = 0; di < d; di++) acc += Q[base + di] * K[jb + di];
          acc /= scale; SCW[k2] = acc; if (SCW[k2] > mx) mx = SCW[k2];
        }
        var sum = 0; for (k2 = 0; k2 < nn; k2++) { SCW[k2] = Math.exp(SCW[k2] - mx); sum += SCW[k2]; }
        for (k2 = 0; k2 < nn; k2++) { var w2 = SCW[k2] / sum, vb = nbrs[k2] * d; for (di = 0; di < d; di++) CAT[d + di] += w2 * V[vb + di]; }
      }
      linInto(L2.update, CAT, 0, H2, i * d, true);
    }
    SCR.H = H2; SCR.H2 = H; H = SCR.H;
  }
  return H;
}
function flatMoveScore(h, si, ti, moveFeats) {
  var d = FLAT.embed.no, HEAD = SCR.HEAD, Z = SCR.Z, di;
  for (di = 0; di < d; di++) { HEAD[di] = h[si * d + di]; HEAD[d + di] = h[ti * d + di]; }
  for (di = 0; di < moveFeats.length; di++) HEAD[2 * d + di] = moveFeats[di];
  linInto(FLAT.head1, HEAD, 0, Z, 0, true);
  var fl = FLAT.head2, acc = fl.b[0];
  for (di = 0; di < fl.ni; di++) acc += fl.W[di] * Z[di];
  return acc;
}

// ---------- (c) WASM wrapper ----------
function makeWasm(file, maxN, maxM) {
  const bytes = fs.readFileSync(path.join(__dirname, 'wasm_gnn', file));
  // Synchronous instantiation — same call shape planWorkerMain / the arena's fake-self would use.
  const inst = new WebAssembly.Instance(new WebAssembly.Module(bytes), {});
  const ex = inst.exports;
  const mem = ex.memory;
  const alloc = n => ex.alloc(n);

  // weight blob: embed W/b, per hop q/k/v/update W/b (packPolicy order)
  const flat = [];
  const push = l => { for (const row of l.W) flat.push(...row); flat.push(...l.b); };
  push(POL.embed);
  for (const a of POL.attn) { push(a.q); push(a.k); push(a.v); push(a.update); }
  const pW = alloc(flat.length * 4);
  const pH1W = alloc(HO * (2 * D + MFEAT) * 4), pH1B = alloc(HO * 4);
  const pH2W = alloc(HO * 4), pH2B = alloc(4);
  // scratch + I/O
  const pNF = alloc(maxN * NFEAT * 4);
  const pOff = alloc((maxN + 1) * 4), pNbr = alloc(maxN * maxN * 4);
  const pH = alloc(maxN * D * 4), pH2 = alloc(maxN * D * 4);
  const pQ = alloc(maxN * D * 4), pK = alloc(maxN * D * 4), pV = alloc(maxN * D * 4);
  const pCAT = alloc(2 * D * 4), pSCW = alloc(maxN * 4);
  const pCand = alloc(maxM * 2 * 4), pMF = alloc(maxM * MFEAT * 4);
  const pHEAD = alloc((2 * D + MFEAT) * 4), pZ = alloc(HO * 4), pOut = alloc(maxM * 4);

  // memory can only grow during alloc() above — safe to take views once, afterwards.
  const F32 = new Float32Array(mem.buffer), I32 = new Int32Array(mem.buffer);
  F32.set(flat, pW >> 2);
  {
    const h1 = []; for (const row of POL.head1.W) h1.push(...row);
    F32.set(h1, pH1W >> 2); F32.set(POL.head1.b, pH1B >> 2);
    F32.set(POL.head2.W[0], pH2W >> 2); F32.set(POL.head2.b, pH2B >> 2);
  }
  return {
    name: file,
    setBoard(NFbuf, N, csrOff, csrNbr) {
      F32.set(NFbuf.subarray(0, N * NFEAT), pNF >> 2);
      I32.set(csrOff, pOff >> 2); I32.set(csrNbr, pNbr >> 2);
    },
    embed(N) {
      return ex.embedBoard(N, NFEAT, D, HOPS, pNF, pOff, pNbr, pW, pH, pH2, pQ, pK, pV, pCAT, pSCW);
    },
    setCands(cands, mfs) { I32.set(cands, pCand >> 2); F32.set(mfs, pMF >> 2); },
    score(M, pHres) {
      ex.scoreMoves(M, D, MFEAT, HO, pHres, pCand, pMF, pH1W, pH1B, pH2W, pH2B, pHEAD, pZ, pOut);
      return F32.subarray(pOut >> 2, (pOut >> 2) + M);
    },
  };
}

// ---------- board generation ----------
function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; var t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

function makeBoard(N, rng) {
  // planar-ish adjacency: random points, connect each to its 3-6 nearest neighbours, symmetrize
  const cx = [], cy = [];
  for (let i = 0; i < N; i++) { cx.push(rng() * 1000); cy.push(rng() * 700); }
  const adjSet = Array.from({ length: N }, () => new Set());
  for (let i = 0; i < N; i++) {
    const order = [...Array(N).keys()].filter(j => j !== i)
      .sort((a, b) => Math.hypot(cx[a] - cx[i], cy[a] - cy[i]) - Math.hypot(cx[b] - cx[i], cy[b] - cy[i]));
    const k = 3 + Math.floor(rng() * 4);   // 3..6
    for (let q = 0; q < Math.min(k, order.length); q++) { adjSet[i].add(order[q]); adjSet[order[q]].add(i); }
  }
  const adj = adjSet.map(s => [...s].sort((a, b) => a - b));
  // owners/troops: mover + 1-3 rivals + neutrals
  const owner = [], troops = [];
  const nRivals = 1 + Math.floor(rng() * 3);
  for (let i = 0; i < N; i++) {
    const r = rng();
    owner.push(r < 0.35 ? NEUTRAL : r < 0.6 ? PLAYER : 2 + Math.floor(rng() * nRivals));
    troops.push(Math.floor(rng() * 120));
  }
  owner[0] = PLAYER; troops[0] = 30 + Math.floor(rng() * 60);   // mover always has a source
  // in-flight armies
  const armies = [];
  const nA = Math.floor(rng() * 6);
  for (let a = 0; a < nA; a++) armies.push({ owner: rng() < 0.5 ? PLAYER : 2, count: 1 + Math.floor(rng() * 40), ti: Math.floor(rng() * N) });
  const s = { n: N, cx, cy, owner, troops, adj, armies };

  // node features — same as gnnNodeFeats(s, mover) in src/web/index.html (7-feature net;
  // the >=10-feature branch mirrors the shipping code and is exercised if the net asks for it)
  const NF10 = NFEAT >= 10;
  let maxDeg = 1; for (let i = 0; i < N; i++) if (adj[i].length > maxDeg) maxDeg = adj[i].length;
  const incM = new Array(N).fill(0), incE = new Array(N).fill(0);
  for (const a of armies) { if (a.owner === PLAYER) incM[a.ti] += a.count; else incE[a.ti] += a.count; }
  const nfRows = [], NFbuf = new Float32Array(N * NFEAT);
  for (let i = 0; i < N; i++) {
    const o = owner[i];
    const f = [o === PLAYER ? 1 : 0, o === NEUTRAL ? 1 : 0, (o !== NEUTRAL && o !== PLAYER) ? 1 : 0, troops[i] / FN, adj[i].length / maxDeg, incM[i] / FN, incE[i] / FN];
    if (NF10) {
      let bp = 0, fr = 0;
      for (const j of adj[i]) { if (owner[j] !== o) fr = 1; if (owner[j] !== NEUTRAL && owner[j] !== PLAYER) bp += troops[j]; }
      f.push(Math.max(0, 150 - troops[i]) / FN, bp / FN, fr);
    }
    nfRows.push(f);
    for (let q = 0; q < NFEAT; q++) NFbuf[i * NFEAT + q] = f[q];
  }
  // CSR adjacency
  const csrOff = new Int32Array(N + 1); let tot = 0;
  for (let i = 0; i < N; i++) { csrOff[i] = tot; tot += adj[i].length; } csrOff[N] = tot;
  const csrNbr = new Int32Array(tot); let w = 0;
  for (let i = 0; i < N; i++) for (const j of adj[i]) csrNbr[w++] = j;

  // candidate moves — every (mover source with >=5 troops, adjacent target), like the planner's
  // candidate sweep; moveFeats mirror the call sites: [ (st-tt)/FN, dist, st>tt?1:0 (, frac) ]
  const diag = Math.hypot(1000, 700);
  const candList = [], mfList = [];
  for (let si = 0; si < N; si++) {
    if (owner[si] !== PLAYER || troops[si] < 5) continue;
    for (const ti of adj[si]) {
      const st = troops[si], tt = troops[ti];
      const dist = Math.hypot(cx[ti] - cx[si], cy[ti] - cy[si]) / diag;
      const mf = [(st - tt) / FN, dist, st > tt ? 1 : 0];
      if (MFEAT >= 4) mf.push(rng() < 0.5 ? 0.5 : 1.0);   // frac (split-move nets)
      candList.push(si, ti); mfList.push(...mf);
    }
  }
  const M = candList.length / 2;
  return { s, nfRows, NFbuf, csrOff, csrNbr, cands: new Int32Array(candList), mfs: new Float32Array(mfList), mfRows: Array.from({ length: M }, (_, m) => Array.from(mfList.slice(m * MFEAT, (m + 1) * MFEAT))), M };
}

// ---------- bench ----------
function hrns() { return Number(process.hrtime.bigint()); }
// min-of-3 timed runs after a dedicated warmup run — resistant to GC/JIT interference
function bench(fn, reps) {
  fn(0);                                   // final warmup
  let best = Infinity;
  for (let run = 0; run < 3; run++) {
    const t0 = hrns();
    for (let r = 0; r < reps; r++) fn(r);
    best = Math.min(best, (hrns() - t0) / reps);
  }
  return best;
}

packPolicy();
const sizes = [16, 49];
const NBOARDS = 200;
const rng = mulberry32(0xC0FFEE);
const results = [];
let globalMaxRel = 0, globalMaxAbs = 0;

for (const N of sizes) {
  const boards = Array.from({ length: NBOARDS }, () => makeBoard(N, rng));
  const totalCands = boards.reduce((a, b) => a + b.M, 0);
  const wasmS = makeWasm('gnn_simd.wasm', N, Math.max(...boards.map(b => b.M)));
  const wasmC = makeWasm('gnn_scalar.wasm', N, Math.max(...boards.map(b => b.M)));
  gnnScratch(N, D, NFEAT); SCR.N = N;   // fresh flat scratch per size (exact-size buffers, like the worker)

  // ---- correctness first: JS-flat is the reference (it IS the shipping planner path).
  // Mixed tolerance (standard for f32 pipelines): a candidate passes if absDiff <= 1e-4
  // OR relDiff <= 1e-4 — pure relative error is meaningless for scores that cross zero
  // (even the nested f64 JS path "fails" pure-relative against its own f32 sibling).
  let maxRelS = 0, maxAbsS = 0, maxRelC = 0, maxAbsC = 0, maxRelNested = 0, failures = 0;
  for (const b of boards) {
    const hFlat = flatEmbed(b.NFbuf, N, b.s.adj).slice();
    const refScores = [];
    for (let m = 0; m < b.M; m++) refScores.push(flatMoveScore(hFlat, b.cands[m * 2], b.cands[m * 2 + 1], b.mfRows[m]));
    const hNested = nestedEmbed(b.nfRows, b.s.adj);

    for (const [w, tag] of [[wasmS, 'simd'], [wasmC, 'scalar']]) {
      w.setBoard(b.NFbuf, N, b.csrOff, b.csrNbr);
      const ph = w.embed(N);
      w.setCands(b.cands, b.mfs);
      const out = w.score(b.M, ph);
      for (let m = 0; m < b.M; m++) {
        const abs = Math.abs(out[m] - refScores[m]);
        const rel = abs / Math.max(1e-12, Math.abs(refScores[m]));
        if (abs > 1e-4 && rel > 1e-4) failures++;
        if (tag === 'simd') { if (rel > maxRelS) maxRelS = rel; if (abs > maxAbsS) maxAbsS = abs; }
        else { if (rel > maxRelC) maxRelC = rel; if (abs > maxAbsC) maxAbsC = abs; }
      }
    }
    for (let m = 0; m < b.M; m++) {
      const ns = nestedMoveScore(hNested, b.cands[m * 2], b.cands[m * 2 + 1], b.mfRows[m]);
      const rel = Math.abs(ns - refScores[m]) / Math.max(1e-12, Math.abs(refScores[m]));
      if (rel > maxRelNested) maxRelNested = rel;
    }
  }
  globalMaxRel = Math.max(globalMaxRel, maxRelS);
  globalMaxAbs = Math.max(globalMaxAbs, maxAbsS);
  console.log(`\nN=${N}: ${NBOARDS} boards, ${totalCands} candidate moves total (${(totalCands / NBOARDS).toFixed(1)}/board)`);
  console.log(`  divergence vs JS-flat:  wasm-simd abs ${maxAbsS.toExponential(2)} rel ${maxRelS.toExponential(2)}  |  wasm-scalar abs ${maxAbsC.toExponential(2)} rel ${maxRelC.toExponential(2)}  |  js-nested(f64) rel ${maxRelNested.toExponential(2)}`);
  if (failures) { console.error(`  FAIL: ${failures} candidates outside abs<=1e-4 || rel<=1e-4`); process.exitCode = 1; }

  // ---- timing (each impl warmed up + timed in isolation; min of 3 runs) ----
  const EMB_REPS = N === 16 ? 20 : 8, SCORE_REPS = 20;
  const t = {};
  // embedding: ns per board embedding. WASM includes the setBoard() input copies — that's the real per-decision cost.
  t.embJS = bench(r => { const b = boards[r % NBOARDS]; flatEmbed(b.NFbuf, N, b.s.adj); }, NBOARDS * EMB_REPS);
  t.embNested = bench(r => { const b = boards[r % NBOARDS]; nestedEmbed(b.nfRows, b.s.adj); }, NBOARDS * Math.max(2, EMB_REPS >> 2));
  t.embWS = bench(r => { const b = boards[r % NBOARDS]; wasmS.setBoard(b.NFbuf, N, b.csrOff, b.csrNbr); wasmS.embed(N); }, NBOARDS * EMB_REPS);
  t.embWC = bench(r => { const b = boards[r % NBOARDS]; wasmC.setBoard(b.NFbuf, N, b.csrOff, b.csrNbr); wasmC.embed(N); }, NBOARDS * EMB_REPS);

  // scoring: ns per candidate, isolated from the embedding — per board the embedding is done
  // untimed, then only the scoring is timed. JS scores one candidate per call (the planner does
  // the same); WASM scores the whole batch per call (how it would integrate), incl. setCands copy.
  const avgM = totalCands / NBOARDS;
  const embFlat = boards.map(b => flatEmbed(b.NFbuf, N, b.s.adj).slice());
  t.scJS = bench(r => {
    const b = boards[r % NBOARDS], h = embFlat[r % NBOARDS];
    for (let m = 0; m < b.M; m++) flatMoveScore(h, b.cands[m * 2], b.cands[m * 2 + 1], b.mfRows[m]);
  }, NBOARDS * SCORE_REPS) / avgM;
  const embNested2 = boards.map(b => nestedEmbed(b.nfRows, b.s.adj));
  t.scNested = bench(r => {
    const b = boards[r % NBOARDS], h = embNested2[r % NBOARDS];
    for (let m = 0; m < b.M; m++) nestedMoveScore(h, b.cands[m * 2], b.cands[m * 2 + 1], b.mfRows[m]);
  }, NBOARDS * Math.max(2, SCORE_REPS >> 2)) / avgM;
  const scoreWasm = (w) => {
    // embed all boards once (untimed) is impossible with shared scratch — instead, per rep:
    // embed board (untimed via pre-loop), then time a burst of score-only calls on that board.
    let total = 0, calls = 0;
    for (let run = 0; run < 3; run++) {
      let tot = 0, c = 0;
      for (const b of boards) {
        w.setBoard(b.NFbuf, N, b.csrOff, b.csrNbr);
        const ph = w.embed(N);
        w.setCands(b.cands, b.mfs); w.score(b.M, ph);   // warm
        const t0 = hrns();
        for (let r = 0; r < SCORE_REPS; r++) { w.setCands(b.cands, b.mfs); w.score(b.M, ph); }
        tot += hrns() - t0; c += SCORE_REPS * b.M;
      }
      if (run === 0 || tot / c < total / calls) { total = tot; calls = c; }
    }
    return total / calls;
  };
  t.scWS = scoreWasm(wasmS);
  t.scWC = scoreWasm(wasmC);

  results.push({ N, avgM, t });
  const f = x => x.toFixed(0).padStart(8);
  console.log(`  ns/board-embedding:   js-nested ${f(t.embNested)}   js-flat ${f(t.embJS)}   wasm-scalar ${f(t.embWC)}   wasm-simd ${f(t.embWS)}`);
  console.log(`  ns/candidate-score:   js-nested ${f(t.scNested)}   js-flat ${f(t.scJS)}   wasm-scalar ${f(t.scWC)}   wasm-simd ${f(t.scWS)}`);
  console.log(`  speedup vs js-flat:   embed  scalar ${(t.embJS / t.embWC).toFixed(2)}x  simd ${(t.embJS / t.embWS).toFixed(2)}x   |  score  scalar ${(t.scJS / t.scWC).toFixed(2)}x  simd ${(t.scJS / t.scWS).toFixed(2)}x`);
  console.log(`  speedup vs js-nested: embed  simd ${(t.embNested / t.embWS).toFixed(2)}x   |  score  simd ${(t.scNested / t.scWS).toFixed(2)}x`);
}

const pass = process.exitCode !== 1;
console.log(`\nglobal max divergence (wasm-simd vs shipping js-flat path): abs ${globalMaxAbs.toExponential(2)}, rel ${globalMaxRel.toExponential(2)}`);
console.log(pass ? 'PASS: every candidate score within abs<=1e-4 || rel<=1e-4 of the JS path'
                 : 'FAIL: candidates diverged beyond abs 1e-4 && rel 1e-4');
