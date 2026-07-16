// GNN forward pass (gnn-v1) — WASM proof-of-concept.
//
// Mirrors the flat-Float32Array planner path in src/web/index.html (packPolicy /
// linInto / gnnEmbedBoard / gnnMoveScore inside planWorkerMain):
//   embed(+tanh) -> H hops of edge-restricted single-head attention (softmax over
//   real border neighbours only) with update(+tanh) -> per-candidate head1(tanh)/head2.
//
// Everything is generic over N (nodes), nf (node features, 7 or 10), d (embed dim),
// hops, mf (move features, 3 or 4) and ho (head1 width) — all driven by the JSON
// export, exactly like packPolicy().
//
// Precision contract (kept deliberately close to the JS path):
//   - weights/activations are f32 (same as the packed Float32Arrays in JS)
//   - dot products accumulate in f32 (SIMD) or f64 (scalar) — JS accumulates in f64;
//     divergence is bounded by f32 rounding, measured in the bench (<= 1e-4 rel).
//   - tanh/exp/sqrt run in f64 (Math.*), matching JS Math.tanh/Math.exp semantics,
//     then round to f32 on store, same as writing into a Float32Array.
//
// Build (see tools/wasm_gnn/build.js):
//   SIMD build:   asc gnn.ts -O3 --enable simd   (dot products use f32x4)
//   scalar build: asc gnn.ts -O3                  (ASC_FEATURE_SIMD folds the v128 code out)
//
// Memory protocol (no runtime, no GC — plain linear memory + bump allocator):
//   JS calls alloc(bytes) for every buffer it needs (weights once per policy,
//   scratch once per max board size), writes Float32Array/Int32Array views into the
//   exported memory, then calls embedBoard()/scoreMoves() with the pointers.

// ---- bump allocator ------------------------------------------------------

let top: usize = __heap_base;

export function alloc(nBytes: i32): i32 {
  let p = (top + 15) & ~(<usize>15);          // 16-byte align (v128 loads)
  top = p + <usize>nBytes;
  let neededPages = <i32>((top + 0xffff) >> 16);
  if (memory.size() < neededPages) memory.grow(neededPages - memory.size());
  return <i32>p;
}

export function resetAlloc(): void { top = __heap_base; }

// ---- fast tanh -----------------------------------------------------------
//
// tanh(x) = sign(x) * (1 - 2/(e^{2|x|}+1)) with a cephes-style f32 expf
// (~2 ulp). JS uses f64 Math.tanh then rounds to f32; the difference is a few
// f32 ulps per activation — measured end-to-end in tools/bench_wasm_gnn.js and
// far inside the 1e-4 gate. Math.tanh (AS's f64 musl port) was the profile's
// hotspot: ~60ns/call x 72 calls/node dominated the embedding.

// e^r minimax coefficients on [-ln2/2, ln2/2] (cephes expf)
const EXP_P0: f32 = 1.9875691500e-4;
const EXP_P1: f32 = 1.3981999507e-3;
const EXP_P2: f32 = 8.3334519073e-3;
const EXP_P3: f32 = 4.1665795894e-2;
const EXP_P4: f32 = 1.6666665459e-1;
const EXP_P5: f32 = 5.0000001201e-1;
const LOG2EF: f32 = 1.44269504088896341;
const LN2_HI: f32 = 0.693359375;
const LN2_LO: f32 = -2.12194440e-4;

// @ts-ignore: decorator
@inline
function expf1(x: f32): f32 {   // e^x for x in [-19, 19] (post-clamp range)
  let n = Mathf.round(x * LOG2EF);
  let r = x - n * LN2_HI - n * LN2_LO;
  let p = EXP_P0;
  p = p * r + EXP_P1; p = p * r + EXP_P2; p = p * r + EXP_P3;
  p = p * r + EXP_P4; p = p * r + EXP_P5;
  let er = r * r * p + r + 1.0;
  return er * reinterpret<f32>((<i32>n + 127) << 23);   // * 2^n
}

// @ts-ignore: decorator
@inline
function tanhf1(x: f32): f32 {
  let ax = Mathf.min(Mathf.abs(x), 9.01);   // tanh(9)=1 to f32; keeps expf in range
  let e2 = expf1(2.0 * ax);
  let t: f32 = 1.0 - 2.0 / (e2 + 1.0);
  return x < 0 ? -t : t;
}

// in-place tanh over an f32 buffer; f32x4 in the SIMD build (4 lanes at once).
// @ts-ignore: decorator
@inline
function tanhInto(p: usize, n: i32): void {
  let i = 0;
  if (ASC_FEATURE_SIMD) {
    const vLOG2E = f32x4.splat(LOG2EF), vHI = f32x4.splat(LN2_HI), vLO = f32x4.splat(LN2_LO);
    const vP0 = f32x4.splat(EXP_P0), vP1 = f32x4.splat(EXP_P1), vP2 = f32x4.splat(EXP_P2);
    const vP3 = f32x4.splat(EXP_P3), vP4 = f32x4.splat(EXP_P4), vP5 = f32x4.splat(EXP_P5);
    const vONE = f32x4.splat(1.0), vTWO = f32x4.splat(2.0), vCLAMP = f32x4.splat(9.01);
    const vBIAS = i32x4.splat(127);
    for (; i + 4 <= n; i += 4) {
      let x = v128.load(p + (<usize>i << 2));
      let ax = f32x4.min(f32x4.abs(x), vCLAMP);
      let t2 = f32x4.mul(vTWO, ax);
      let nv = f32x4.nearest(f32x4.mul(t2, vLOG2E));
      let r = f32x4.sub(f32x4.sub(t2, f32x4.mul(nv, vHI)), f32x4.mul(nv, vLO));
      let q = vP0;
      q = f32x4.add(f32x4.mul(q, r), vP1); q = f32x4.add(f32x4.mul(q, r), vP2);
      q = f32x4.add(f32x4.mul(q, r), vP3); q = f32x4.add(f32x4.mul(q, r), vP4);
      q = f32x4.add(f32x4.mul(q, r), vP5);
      let er = f32x4.add(f32x4.add(f32x4.mul(f32x4.mul(r, r), q), r), vONE);
      let scale = i32x4.shl(i32x4.add(i32x4.trunc_sat_f32x4_s(nv), vBIAS), 23);
      let e2 = f32x4.mul(er, scale);   // e^{2|x|}
      let t = f32x4.sub(vONE, f32x4.div(vTWO, f32x4.add(e2, vONE)));
      // restore sign of x
      let signBit = v128.and(x, i32x4.splat(<i32>0x80000000));
      v128.store(p + (<usize>i << 2), v128.or(t, signBit));
    }
  }
  for (; i < n; i++) store<f32>(p + (<usize>i << 2), tanhf1(load<f32>(p + (<usize>i << 2))));
}

// ---- core kernels --------------------------------------------------------

// dot(w[0..n), x[0..n)) over f32 buffers. SIMD build: f32x4 lanes + scalar tail.
// @ts-ignore: decorator
@inline
function dot(w: usize, x: usize, n: i32): f64 {
  let i = 0;
  if (ASC_FEATURE_SIMD) {
    let vacc = f32x4.splat(0);
    for (; i + 4 <= n; i += 4) {
      vacc = f32x4.add(vacc, f32x4.mul(v128.load(w + (<usize>i << 2)), v128.load(x + (<usize>i << 2))));
    }
    let acc: f64 = <f64>(f32x4.extract_lane(vacc, 0) + f32x4.extract_lane(vacc, 1)
                       + f32x4.extract_lane(vacc, 2) + f32x4.extract_lane(vacc, 3));
    for (; i < n; i++) acc += <f64>load<f32>(w + (<usize>i << 2)) * <f64>load<f32>(x + (<usize>i << 2));
    return acc;
  } else {
    let acc: f64 = 0;
    for (; i < n; i++) acc += <f64>load<f32>(w + (<usize>i << 2)) * <f64>load<f32>(x + (<usize>i << 2));
    return acc;
  }
}

// out[0..no) = W(no x ni) * x[0..ni) + b, optional tanh — mirror of linInto() in index.html.
// @ts-ignore: decorator
@inline
function linInto(pW: usize, pB: usize, ni: i32, no: i32, pX: usize, pOut: usize, tanh: bool): void {
  for (let o = 0; o < no; o++) {
    let acc: f64 = <f64>load<f32>(pB + (<usize>o << 2)) + dot(pW + (<usize>(o * ni) << 2), pX, ni);
    store<f32>(pOut + (<usize>o << 2), <f32>acc);
  }
  if (tanh) tanhInto(pOut, no);
}

// ---- board embedding -----------------------------------------------------
//
// Weight blob layout at pWeights (all f32, matching packPolicy() row-major flattening):
//   embed.W (d*nf), embed.b (d),
//   then per hop: q.W (d*d), q.b (d), k.W (d*d), k.b (d), v.W (d*d), v.b (d),
//                 update.W (d*2d), update.b (d)
//
// Adjacency is CSR: pOff = i32[N+1] offsets, pNbr = i32[totalEdges] neighbour ids.
// Scratch: pH, pH2 = N*d f32; pQ, pK, pV = N*d f32; pCAT = 2d f32; pSCW = maxDeg f32.
// Node features at pNF: N*nf f32, computed JS-side by gnnNodeFeats (contract: caller's job).
//
// Returns the pointer holding the final embeddings (pH or pH2 — hop ping-pong).
export function embedBoard(
  N: i32, nf: i32, d: i32, hops: i32,
  pNF: i32, pOff: i32, pNbr: i32, pWeights: i32,
  pH: i32, pH2: i32, pQ: i32, pK: i32, pV: i32, pCAT: i32, pSCW: i32
): i32 {
  let w: usize = <usize>pWeights;
  let H: usize = <usize>pH, H2: usize = <usize>pH2;
  let Q: usize = <usize>pQ, K: usize = <usize>pK, V: usize = <usize>pV;
  let CAT: usize = <usize>pCAT, SCW: usize = <usize>pSCW;
  let OFF: usize = <usize>pOff, NBR: usize = <usize>pNbr, NF: usize = <usize>pNF;
  let dBytes: usize = <usize>d << 2;

  // embed + tanh
  let embW = w; w += <usize>(d * nf) << 2;
  let embB = w; w += <usize>d << 2;
  for (let i = 0; i < N; i++) {
    linInto(embW, embB, nf, d, NF + <usize>(i * nf) * 4, H + <usize>i * dBytes, true);
  }

  let scale: f64 = Math.sqrt(<f64>d);

  for (let hop = 0; hop < hops; hop++) {
    let qW = w; w += <usize>(d * d) << 2; let qB = w; w += <usize>d << 2;
    let kW = w; w += <usize>(d * d) << 2; let kB = w; w += <usize>d << 2;
    let vW = w; w += <usize>(d * d) << 2; let vB = w; w += <usize>d << 2;
    let uW = w; w += <usize>(d * 2 * d) << 2; let uB = w; w += <usize>d << 2;

    for (let i = 0; i < N; i++) {
      let hOff = H + <usize>i * dBytes;
      let o = <usize>i * dBytes;
      linInto(qW, qB, d, d, hOff, Q + o, false);
      linInto(kW, kB, d, d, hOff, K + o, false);
      linInto(vW, vB, d, d, hOff, V + o, false);
    }

    for (let i = 0; i < N; i++) {
      let base = <usize>i * dBytes;
      // CAT = [ h_i | softmax-weighted neighbour V ]
      memory.copy(CAT, H + base, dBytes);
      memory.fill(CAT + dBytes, 0, dBytes);

      let s0 = load<i32>(OFF + (<usize>i << 2));
      let s1 = load<i32>(OFF + (<usize>(i + 1) << 2));
      let nn = s1 - s0;
      if (nn > 0) {
        // scores (f32-stored like the JS SCW Float32Array), track max
        let mx: f64 = -1e30;
        for (let k = 0; k < nn; k++) {
          let j = load<i32>(NBR + (<usize>(s0 + k) << 2));
          let sc: f64 = dot(Q + base, K + <usize>j * dBytes, d) / scale;
          let sf: f32 = <f32>sc;
          store<f32>(SCW + (<usize>k << 2), sf);
          if (<f64>sf > mx) mx = <f64>sf;
        }
        // softmax weights
        let sum: f64 = 0;
        for (let k = 0; k < nn; k++) {
          let e: f32 = <f32>Math.exp(<f64>load<f32>(SCW + (<usize>k << 2)) - mx);
          store<f32>(SCW + (<usize>k << 2), e);
          sum += <f64>e;
        }
        // agg += w_k * V_j  (accumulated into the f32 CAT tail, same as JS)
        for (let k = 0; k < nn; k++) {
          let j = load<i32>(NBR + (<usize>(s0 + k) << 2));
          let wk: f32 = <f32>(<f64>load<f32>(SCW + (<usize>k << 2)) / sum);
          let vOff = V + <usize>j * dBytes;
          let aggOff = CAT + dBytes;
          if (ASC_FEATURE_SIMD) {
            let vw = f32x4.splat(wk);
            let di = 0;
            for (; di + 4 <= d; di += 4) {
              let a = v128.load(aggOff + (<usize>di << 2));
              let vv = v128.load(vOff + (<usize>di << 2));
              v128.store(aggOff + (<usize>di << 2), f32x4.add(a, f32x4.mul(vw, vv)));
            }
            for (; di < d; di++) {
              let a = load<f32>(aggOff + (<usize>di << 2));
              store<f32>(aggOff + (<usize>di << 2), a + wk * load<f32>(vOff + (<usize>di << 2)));
            }
          } else {
            for (let di = 0; di < d; di++) {
              let a = load<f32>(aggOff + (<usize>di << 2));
              store<f32>(aggOff + (<usize>di << 2), a + wk * load<f32>(vOff + (<usize>di << 2)));
            }
          }
        }
      }
      linInto(uW, uB, 2 * d, d, CAT, H2 + base, true);
    }
    // ping-pong hop buffers
    let t = H; H = H2; H2 = t;
  }
  return <i32>H;
}

// ---- candidate scoring ---------------------------------------------------
//
// Candidates: pCand = i32[M*2] (si,ti) pairs; pMF = f32[M*mf] move features.
// head1: ho x (2d+mf) + tanh; head2: 1 x ho. Scratch pHEAD = (2d+mf) f32, pZ = ho f32.
// Scores written to pOut = f32[M].
export function scoreMoves(
  M: i32, d: i32, mf: i32, ho: i32,
  pH: i32, pCand: i32, pMF: i32,
  pH1W: i32, pH1B: i32, pH2W: i32, pH2B: i32,
  pHEAD: i32, pZ: i32, pOut: i32
): void {
  let H: usize = <usize>pH, HEAD: usize = <usize>pHEAD, Z: usize = <usize>pZ;
  let dBytes: usize = <usize>d << 2;
  let ni = 2 * d + mf;
  for (let m = 0; m < M; m++) {
    let si = load<i32>(<usize>pCand + (<usize>(m * 2) << 2));
    let ti = load<i32>(<usize>pCand + (<usize>(m * 2 + 1) << 2));
    memory.copy(HEAD, H + <usize>si * dBytes, dBytes);
    memory.copy(HEAD + dBytes, H + <usize>ti * dBytes, dBytes);
    memory.copy(HEAD + 2 * dBytes, <usize>pMF + (<usize>(m * mf) << 2), <usize>mf << 2);
    linInto(<usize>pH1W, <usize>pH1B, ni, ho, HEAD, Z, true);
    let acc: f64 = <f64>load<f32>(<usize>pH2B) + dot(<usize>pH2W, Z, ho);
    store<f32>(<usize>pOut + (<usize>m << 2), <f32>acc);
  }
}
