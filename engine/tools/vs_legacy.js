#!/usr/bin/env node
// New engine AI vs the ORIGINAL AI, head to head.
//
// Referee  : the game's own simulation functions, extracted from the last commit that shipped the
//            old AI (the Rust engine is conformance-checked against exactly this code).
// Old AI   : that commit's planWorkerMain (the MCTS planner) with the shipped gnn_policy.json, at
//            the Nightmare tier's settings (240 ms think per decision, 0.30 s cadence, 4-move
//            salvos, hunt 0.7 with the 5 s opening ramp), plus its main-thread helpers: the
//            coordinatedAssault script, the reverse-reinforcement veto, and the AI invest script.
//            Not reproduced: the cross-session opponent model (it needs a human's history),
//            pondering between decisions, multi-core root parallelism, and the 0.5 s flee delay.
// New AI   : the shipped WebAssembly engine, `search` brain, one decision every 0.30 s.
//
// Every board is played twice with the start positions swapped. The old AI is always owner 2 and
// the new AI owner 1, as in the live game (the old planner treats owner 1 as the human).
//
//   usage: node engine/tools/vs_legacy.js [--boards N] [--offset K] [--stride S] [--terrain 0|1]
//                                         [--budget ms] [--max-time s] [--brain spec]
// Prints one JSON line per game. Run several with different --offset and the same --stride to
// use more cores.
const fs = require('fs'), path = require('path'), vm = require('vm'), cp = require('child_process');
const { performance } = require('perf_hooks');
const root = path.join(__dirname, '..', '..');
const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i < 0 ? d : process.argv[i + 1]; };
const BOARDS = +arg('boards', 4), OFFSET = +arg('offset', 0), STRIDE = +arg('stride', 1);
const TERRAIN = arg('terrain', '1') === '1', BUDGET = +arg('budget', 240), MAX_TIME = +arg('max-time', 480);
const BRAIN = arg('brain', 'search'), CADENCE = 0.30, TICK = 1 / 30, REF = '74e63cf';

let html = cp.execSync('git show ' + REF + ':src/web/index.html', { cwd: root, maxBuffer: 1 << 28 }).toString('utf8').replace(/\r\n/g, '\n');
const policy = JSON.parse(cp.execSync('git show ' + REF + ':src/web/gnn_policy.json', { cwd: root, maxBuffer: 1 << 28 }).toString('utf8'));
const workerSrc = html.match(/function planWorkerMain\(\)\s*\{[\s\S]*?\n  \}\n/)[0];
// referee rules = the current game: one cap ladder for every owner, and (TERRAIN) the terrain
// bonus that the shipped code computed from `.length` of a Set and therefore never applied.
if (TERRAIN) html = html.replace(/adjacency\[si\]\.length/g, 'adjacency[si].size');
{ const old = '(s.owner===PLAYER) ? PLAYER_CAP[s.level||0]'; if (!html.includes(old)) throw new Error('capOf patch target missing');
  html = html.replace(old, '(s.owner!==NEUTRAL) ? PLAYER_CAP[s.level||0]'); }

function grabBalanced(start){ let i = html.indexOf('{', start), depth = 0;
  for (; i < html.length; i++){ const c = html[i]; if (c === '{') depth++; else if (c === '}'){ depth--; if (depth === 0) return i + 1; } }
  throw new Error('unbalanced'); }
function grabFn(name){ const at = html.indexOf('\n  function ' + name + '('); if (at < 0) throw new Error('function not found: ' + name);
  return html.slice(at + 1, grabBalanced(html.indexOf(')', at))); }
function grabConst(name){ const at = html.indexOf('\n  const ' + name + ' '); if (at < 0) throw new Error('const not found: ' + name);
  let i = at + 1, depth = 0;
  for (; i < html.length; i++){ const c = html[i]; if ('([{'.includes(c)) depth++; else if (')]}'.includes(c)) depth--; else if (c === ';' && depth === 0) break; }
  return html.slice(at + 1, i + 1); }
const src = [
  'const NEUTRAL=0, PLAYER=1, MAX_TROOPS=150;',
  'let states=[], troops=[], adjacency=[], cfg={}, gameElapsed=0, gameTick=0, netMode=null, myOwner=1;',
  'let matchTrack=null, aiSnow=0, supplyClock=0, aiInvClock=0, slotControl={}, running=false, boomUntil=0;',
  'let ORB_SPEED=70, orbSeq=0, markR=17, lastReinforce={}, adjacencyMode=true, depthMode=true, upgradesOn=true, lastSalvo=null;',
  'const humanSnowball=()=>0, updateThreats=()=>{}, tutHintsTick=()=>{}, updateAI=()=>{};',
  ...['PLAYER_CAP', 'upgradeCost', 'TILE', 'depthOn', 'LANCH', 'ATTRITION_Q', 'OFFSUP_DEF', 'terrMult', 'investCost', 'invGrowMult'].map(grabConst),
  ...['capOf', 'tileGrowMult', 'growRateFor', 'recomputeSupply', 'isLegalMove', 'sendTroops', 'arrive', 'update', 'salvoFlips', 'coordinatedAssault'].map(grabFn),
  'this.api = { update, sendTroops, recomputeSupply, coordinatedAssault, terrMult, OFFSUP_DEF, investCost, upgradeCost, PLAYER_CAP,',
  '  set(o){ if(o.states) states=o.states; if(o.adjacency) adjacency=o.adjacency; if(o.cfg) cfg=o.cfg; if(o.speed) ORB_SPEED=o.speed; if(o.slotControl) slotControl=o.slotControl;',
  '    if(o.reset){ troops=[]; gameElapsed=0; supplyClock=0; aiInvClock=0; orbSeq=0; lastReinforce={}; } },',
  '  get troops(){ return troops; }, get states(){ return states; }, get t(){ return gameElapsed; }, get supplyClock(){ return supplyClock; }, get lastReinforce(){ return lastReinforce; } };',
].join('\n');
const sandbox = { Math, console }; vm.createContext(sandbox);
vm.runInContext('(function(){' + src + '\n}).call(this)', sandbox);
const G = sandbox.api;

// ---- old planner, instantiated the way tools/selfplay_arena.js does ----
const wasmDir = path.join(root, 'tools', 'wasm_gnn');
const b64 = f => { try { return fs.readFileSync(path.join(wasmDir, f)).toString('base64'); } catch (e) { return ''; } };
const self = { postMessage: m => { self._last = m; }, performance };
new Function('self', 'performance', '(' + workerSrc + ')();')(self, performance);
self.onmessage({ data: { type: 'init', policy, wasmSimdB64: b64('gnn_simd.wasm'), wasmScalarB64: b64('gnn_scalar.wasm') } });
const legacyPlan = req => { self.onmessage({ data: { type: 'plan', id: 1, req } }); return self._last || {}; };

// ---- new engine (the same .wasm the game inlines) ----
const ex = new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(path.join(root, 'engine', 'target', 'wasm32-unknown-unknown', 'release', 'stateio.wasm'))), {}).exports;
const view = () => { const ptr = ex.sio_buf(), len = ex.sio_buf_len(); return new Float64Array(ex.memory.buffer, ptr, len); };
const TILE_ID = { capital: 1, factory: 2, fortress: 3 };

// ---- maps ----
const maps = []; { const L = fs.readFileSync(path.join(root, 'engine', 'data', 'maps.txt'), 'utf8').split('\n').filter(Boolean);
  for (let i = 0; i < L.length;){ const h = L[i++].split('|'); const n = +h[2]; const m = { name: h[1], n, cx: [], cy: [], adj: [] };
    for (let k = 0; k < n; k++){ const q = L[i++].split('|'); m.cx.push(+q[1]); m.cy.push(+q[2]); m.adj.push(q[3] ? q[3].split(',').map(Number) : []); }
    maps.push(m); } }
const map = maps.find(m => m.name === 'Continental U.S.'), N = map.n;
const diag = Math.hypot(Math.max(...map.cx) - Math.min(...map.cx), Math.max(...map.cy) - Math.min(...map.cy));
const ORB_SPEED = diag / 8;

function makeBoard(seed){
  let s = (seed * 2654435761) >>> 0; const rnd = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; };
  for (let i = 0; i < 8; i++) rnd();
  const troops = map.cx.map(() => Math.floor(18 + 37 * rnd()));
  const a = Math.floor(rnd() * N); let b = 0, bd = -1;
  for (let i = 0; i < N; i++){ const d = (map.cx[i] - map.cx[a]) ** 2 + (map.cy[i] - map.cy[a]) ** 2; if (i !== a && d > bd){ bd = d; b = i; } }
  const pool = [...Array(N).keys()].filter(i => i !== a && i !== b);
  for (let i = pool.length - 1; i > 0; i--){ const j = Math.floor(rnd() * (i + 1)); [pool[i], pool[j]] = [pool[j], pool[i]]; }
  const tiles = {}; for (let i = 0; i < Math.max(2, Math.round(N * 0.12)); i++){ const r = rnd(); tiles[pool[i]] = r < 0.40 ? 'factory' : (r < 0.75 ? 'fortress' : 'capital'); }
  return { troops, a, b, tiles };
}

function play(board, swap){
  const NEW = 1, OLD = 2;
  const states = map.cx.map((x, i) => ({ id: i, name: '', cx: x, cy: map.cy[i], owner: 0, troops: board.troops[i], growAcc: 0, pulse: 0, level: 0, type: board.tiles[i] || null, inv: 0, offSupply: false }));
  const sNew = swap ? board.b : board.a, sOld = swap ? board.a : board.b;
  states[sNew].owner = NEW; states[sNew].troops = 22; states[sOld].owner = OLD; states[sOld].troops = 22;
  G.set({ states, adjacency: map.adj.map(x => new Set(x)), cfg: { growRate: 1.0, enemyGrow: 1.0 }, speed: ORB_SPEED, slotControl: { 2: 'ai' }, reset: true });
  G.recomputeSupply();
  // describe the world to the engine
  const w = [N, 1, 1, TERRAIN ? 1 : 0, 1, ORB_SPEED, 3, 0, 1.0, 1.0];
  for (let i = 0; i < N; i++) w.push(map.cx[i], map.cy[i], map.adj[i].length, ...map.adj[i]);
  view().set(w); ex.sio_world();
  { const v = view(); for (let i = 0; i < BRAIN.length; i++) v[i] = BRAIN.charCodeAt(i); ex.sio_agent(NEW, BRAIN.length); }
  let oldPeak = 1, newPeak = 1, oldSends = 0;
  let tNew = 0.1, tOld = 0.15, oldMs = 0, oldDecisions = 0, newMs = 0, newDecisions = 0, oldIters = 0;
  const count = o => { let st = 0, tr = 0; for (const s of G.states) if (s.owner === o){ st++; tr += s.troops; } for (const t of G.troops) if (t.owner === o) tr += t.count; return [st, Math.round(tr)]; };
  const alive = o => G.states.some(s => s.owner === o) || G.troops.some(t => t.owner === o);
  while (G.t < MAX_TIME){
    tNew -= TICK; tOld -= TICK;
    if (tNew <= 0 && alive(NEW)){
      tNew = CADENCE;
      const st = [G.t, G.supplyClock];
      for (const s of G.states) st.push(s.owner, s.troops, s.growAcc || 0, s.level || 0, s.type ? TILE_ID[s.type] : 0, s.inv || 0, s.offSupply ? 1 : 0);
      st.push(G.troops.length);
      for (const t of G.troops) st.push(t.owner, t.fromId, t.targetId, t.count, Math.hypot(t.tx - t.x, t.ty - t.y), Math.hypot(t.tx - t.ox, t.ty - t.oy) || 1);
      view().set(st);
      const t0 = performance.now(), n = ex.sio_decide(NEW), acts = Array.from(view().subarray(0, n * 4));
      newMs += performance.now() - t0; newDecisions++;
      for (let k = 0; k + 3 < acts.length; k += 4){
        const from = G.states[acts[k + 1]]; if (!from || from.owner !== NEW) continue;
        if (acts[k] === 0){ const to = G.states[acts[k + 2]], amt = acts[k + 3];
          G.sendTroops(from, to, undefined, amt >= Math.floor(from.troops) ? 1 : (amt + 0.5) / from.troops); }
        else if (acts[k] === 1){ const c = G.investCost(from.inv || 0); if ((from.inv || 0) < 3 && from.troops > c){ from.troops -= c; from.inv = (from.inv || 0) + 1; } }
        else if (acts[k] === 2){ const lv = from.level || 0, c = G.upgradeCost(lv); if (lv < G.PLAYER_CAP.length - 1 && from.troops >= c){ from.troops -= c; from.level = lv + 1; } }
      }
    }
    if (tOld <= 0 && alive(OLD)){
      tOld = CADENCE;
      if (!G.coordinatedAssault(OLD)){
        const S = G.states, lr = G.lastReinforce[OLD];
        const req = { n: N, cx: map.cx, cy: map.cy, owner: S.map(s => s.owner), troops: S.map(s => s.troops), names: S.map(s => s.name),
          armies: G.troops.map(t => ({ owner: t.owner, count: t.count, ti: t.targetId, ttotal: Math.max(0.02, Math.hypot(t.tx - t.x, t.ty - t.y) / ORB_SPEED) })),
          adj: map.adj, oppBias: null, oppW: null, explore: undefined, ponder: false, freeMove: false, depth: true,
          terr: S.map(s => G.terrMult(s.id) * (s.offSupply ? G.OFFSUP_DEF : 1)), reinf: (lr && (G.t - lr.t) < 6) ? [lr.a, lr.b] : null,
          aiSpeed: 0.30, rlDiag: diag, orbSpeed: ORB_SPEED, growRate: 1.0, enemyGrow: 1.0,
          hunt: 0.7 * Math.min(1, G.t / 5), desp: 0, infight: 1, valueBlend: 0.5, attn: 0,
          horizon: 1e9, dt: 0.3, ownerId: OLD, wantViz: false, humans: {},
          mcts: { tbudget: BUDGET, maxBudget: BUDGET, maxIter: 1e9, cPuct: 1.3, K: 9 } };
        const t0 = performance.now(), out = legacyPlan(req);
        oldMs += performance.now() - t0; oldDecisions++; oldIters += out.iters || 0;
        let noopV = -1; const cands = [];
        for (const r of (out.roots || [])){ if (!r.move) noopV = r.visits; else cands.push(r); }
        cands.sort((x, y) => y.visits - x.visits);
        const usedSrc = new Set(), usedTgt = new Set(); let done = 0;
        for (const c of cands){
          if (done >= 4 || c.visits <= noopV) break;
          const m = c.move, s = S[m[0]], t = S[m[1]];
          if (!s || !t || s === t || s.owner !== OLD || usedSrc.has(m[0]) || usedTgt.has(m[1])) continue;
          G.sendTroops(s, t, null, m[2]); usedSrc.add(m[0]); usedTgt.add(m[1]); done++; oldSends++;
        }
      }
    }
    G.update(TICK);
    if (process.env.TRACE && G.t % 10 < TICK){ const a = count(NEW), b = count(OLD); console.error('  t=' + Math.round(G.t) + '  new ' + a[0] + ' states ' + a[1] + ' troops | old ' + b[0] + ' states ' + b[1] + ' troops | orbs new ' + G.troops.filter(t=>t.owner===NEW).length + ' old ' + G.troops.filter(t=>t.owner===OLD).length); }
    if (process.env.TRACE && G.t % 10 < TICK){ const a = count(NEW), b = count(OLD); console.error('  t=' + Math.round(G.t) + '  new ' + a[0] + ' states ' + a[1] + ' troops | old ' + b[0] + ' states ' + b[1] + ' troops | orbs new ' + G.troops.filter(t=>t.owner===NEW).length + ' old ' + G.troops.filter(t=>t.owner===OLD).length); }
    if (G.t % 5 < TICK){ oldPeak = Math.max(oldPeak, count(OLD)[0]); newPeak = Math.max(newPeak, count(NEW)[0]); }
    const an = alive(NEW), ao = alive(OLD);
    if (!an || !ao){ const [sn] = count(NEW), [so] = count(OLD);
      return { winner: an ? 'new' : (ao ? 'old' : 'draw'), t: Math.round(G.t), newStates: sn, oldStates: so, oldPeak, newPeak, oldSends, oldMs: Math.round(oldMs / Math.max(1, oldDecisions)), oldIters: Math.round(oldIters / Math.max(1, oldDecisions)), newMs: +(newMs / Math.max(1, newDecisions)).toFixed(2) }; }
  }
  const [sn, tn] = count(NEW), [so, to] = count(OLD);
  return { winner: 'draw', t: Math.round(G.t), newStates: sn, oldStates: so, newTroops: tn, oldTroops: to, oldPeak, newPeak, oldSends, oldMs: Math.round(oldMs / Math.max(1, oldDecisions)), oldIters: Math.round(oldIters / Math.max(1, oldDecisions)), newMs: +(newMs / Math.max(1, newDecisions)).toFixed(2) };
}

for (let b = OFFSET; b < BOARDS; b += STRIDE){
  const board = makeBoard(1000 + b);
  for (const swap of [false, true]){
    const r = play(board, swap);
    console.log(JSON.stringify(Object.assign({ board: b, swap, terrain: TERRAIN, budget: BUDGET }, r)));
  }
}
