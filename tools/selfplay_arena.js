#!/usr/bin/env node
// Headless self-play arena. Runs the ACTUAL shipped planner (planWorkerMain, extracted verbatim
// from src/web/index.html) with two policy configs playing head-to-head in the game's own physics,
// so we can measure strength changes WITHOUT a human play-tester.
//
// Both sides get the SAME wall-clock think budget per move, so a bigger/slower net that does fewer
// simulations is penalized exactly as it would be in the real game — the win-rate captures the real
// "smarter but slower" trade-off, not just "is it a better function".
//
//   usage: node tools/selfplay_arena.js <policyA.json> <policyB.json> [games] [tbudgetMs]
const fs = require('fs');
const { performance } = require('perf_hooks');

const htmlPath0 = process.env.HTML_DEFAULT || 'src/web/index.html';
const [,, pathA, pathB, gamesArg, budgetArg, blendAArg, blendBArg] = process.argv;
const GAMES = +(gamesArg || 40);
const TBUDGET = +(budgetArg || 60);
const BLEND_A = blendAArg!=null ? +blendAArg : null;   // optional per-config value/heuristic blend
const BLEND_B = blendBArg!=null ? +blendBArg : null;
const NSTATES = 16, HORIZON = 8, DT = 0.4, K = 9, CPUCT = 1.3;
// GAMECAP env override: default 38s keeps gating runs fast, but that means arena games NEVER reach
// the 150 troop cap — long-game dynamics (cap-dodge shuffling, mega-stacks, split habits at scale)
// are invisible at the default. Set GAMECAP=300+ for behavioral analysis runs.
const ORB = 70, AISPD = 0.6, ENVDT = 0.2, GAMECAP = +(process.env.GAMECAP || 38), CADENCE = 0.7;

// ---- extract planWorkerMain() source verbatim (per html file) and instantiate as a callable planner ----
// HTML_A / HTML_B env vars override the worker source per config, so a 16-input net (new index.html)
// can play a 12-input net (snapshotted old index.html) across a feature-set change.
const _srcCache = {};
function loadWorkerSrc(path){
  if (_srcCache[path]) return _srcCache[path];
  // normalize CRLF -> LF: on a Windows checkout git may convert line endings, and the extraction
  // regex below anchors on `\n  }\n`, which would silently fail to match against `\r\n` and leave
  // the whole trainer producing zero self-play data (buf stays 0). Strip \r so extraction is
  // line-ending-agnostic regardless of how the repo was cloned.
  const html = fs.readFileSync(path, 'utf8').replace(/\r\n/g, '\n');
  const m = html.match(/function planWorkerMain\(\)\s*\{[\s\S]*?\n  \}\n/);
  if (!m) { console.error('could not extract planWorkerMain from '+path); process.exit(1); }
  return (_srcCache[path] = m[0]);
}
// WASM fast-path bytes for the worker (wasm_integration.md): read from disk here (same bytes
// index.html embeds as base64 constants). ARENA_NO_WASM=1 forces the pure-JS path — use it to
// bisect wasm-related regressions, and note the equal-time-budget asymmetry: when gating a NET
// change, both sides must run the same engine (both-wasm or both-JS) to keep the comparison clean.
const WASM_B64 = (() => {
  if (process.env.ARENA_NO_WASM) return {};
  try {
    const p = require('path');
    return { simd:   fs.readFileSync(p.join(__dirname, 'wasm_gnn', 'gnn_simd.wasm')).toString('base64'),
             scalar: fs.readFileSync(p.join(__dirname, 'wasm_gnn', 'gnn_scalar.wasm')).toString('base64') };
  } catch (e) { return {}; }
})();
function makePlanner(policy, valueBlend, htmlPath){
  const workerSrc = loadWorkerSrc(htmlPath || htmlPath0);
  const self = { postMessage:(msg)=>{ self._last = msg; }, performance };
  // free variable `self` inside planWorkerMain resolves to this param lexically
  const install = new Function('self', '(' + workerSrc + ')();');
  install(self);
  self.onmessage({ data:{ type:'init', policy, wasmSimdB64: WASM_B64.simd, wasmScalarB64: WASM_B64.scalar } });
  const fn = (req) => { if (valueBlend!=null) req.valueBlend = valueBlend;   // per-planner leaf-eval blend
    self.onmessage({ data:{ type:'plan', id:1, req } });
    fn.lastRoots = self._last && self._last.roots;   // root visit-count distribution, for --dump-visits capture
    return self._last && self._last.move; };
  return fn;
}

// ---- faithful environment (mirrors the worker's snapStep/snapSend: growth + orb travel + combat) ----
function rnd(a,b){ return a + Math.random()*(b-a); }
// Brute-force Delaunay triangulation (N is tiny here — 16 points, ~560 candidate triangles — so an
// O(N^4) circumcircle sweep is trivial). This is the synthetic analogue of "shares a border" for a
// random point set, mirroring the real game's TopoJSON-derived adjacency (see index.html Part 1).
function delaunayAdjacency(cx, cy, N){
  const adj = Array.from({length:N}, () => new Set());
  const orient = (i,j,k) => (cx[j]-cx[i])*(cy[k]-cy[i]) - (cy[j]-cy[i])*(cx[k]-cx[i]);
  const inCircum = (i,j,k,d) => {   // assumes i,j,k CCW; true iff d lies inside their circumcircle
    const ax=cx[i]-cx[d], ay=cy[i]-cy[d], bx=cx[j]-cx[d], by=cy[j]-cy[d], gx=cx[k]-cx[d], gy=cy[k]-cy[d];
    const a2=ax*ax+ay*ay, b2=bx*bx+by*by, g2=gx*gx+gy*gy;
    return ax*(by*g2-b2*gy) - ay*(bx*g2-b2*gx) + a2*(bx*gy-by*gx) > 1e-12;
  };
  for (let i=0;i<N;i++) for (let j=i+1;j<N;j++) for (let k=j+1;k<N;k++){
    let a=i,b=j,c=k;
    if (orient(a,b,c) < 0) { const t=b; b=c; c=t; }
    if (Math.abs(orient(a,b,c)) < 1e-12) continue;   // collinear triple, skip
    let ok = true;
    for (let d=0; d<N; d++){ if (d===a||d===b||d===c) continue; if (inCircum(a,b,c,d)) { ok=false; break; } }
    if (ok) { adj[a].add(b); adj[b].add(a); adj[b].add(c); adj[c].add(b); adj[a].add(c); adj[c].add(a); }
  }
  // defensive fallback for degenerate (near-collinear) point sets: guarantee no isolated node by
  // connecting any still-empty node to its nearest neighbor.
  for (let i=0;i<N;i++) if (adj[i].size===0){
    let bj=-1, bd=Infinity;
    for (let j=0;j<N;j++) if (j!==i){ const d=Math.hypot(cx[i]-cx[j], cy[i]-cy[j]); if (d<bd){ bd=d; bj=j; } }
    if (bj>=0){ adj[i].add(bj); adj[bj].add(i); }
  }
  return adj.map(s => Array.from(s));
}
// REAL_MAPS env (0..1): probability a game is played on one of the REAL game maps (adjacency +
// centroids extracted from src/web/us-states.json by tools/extract_maps.js) instead of a synthetic
// Delaunay board. Real regions run 11-49 states with genuine corner topology (Maine is degree-1),
// so the nets train and gate on the boards the shipped game actually plays.
const REAL_MAPS_P = +(process.env.REAL_MAPS || 0);
const REAL_MAPS = REAL_MAPS_P > 0 ? JSON.parse(fs.readFileSync(require('path').join(__dirname, 'real_maps.json'), 'utf8')) : null;
// MIDGAME_POOL/MIDGAME_P: start a slice of games from MID-GAME positions sampled from earlier
// dump records instead of fresh openings — dense late-game coverage at a fraction of the cost of
// playing whole games there (the AlphaZero position-sampling trick). Records must have both
// owners 2 and 3 alive; in-flight armies are dropped (they carry no travel geometry we can trust).
const MIDGAME_P = +(process.env.MIDGAME_P || 0);
const MIDGAME_POOL = (() => {
  if (!(MIDGAME_P > 0) || !process.env.MIDGAME_POOL) return null;
  try {
    const lines = fs.readFileSync(process.env.MIDGAME_POOL, 'utf8').trim().split('\n');
    const pool = [];
    for (const l of lines){
      try {
        const r = JSON.parse(l);
        if (!r.adj || !r.owner) continue;
        let a = false, b = false;
        for (const o of r.owner){ if (o === 2) a = true; else if (o === 3) b = true; }
        if (a && b) pool.push(r);
      } catch (e) {}
    }
    return pool.length ? pool : null;
  } catch (e) { return null; }
})();
function genBoard(N, owners){
  if (MIDGAME_POOL && Math.random() < MIDGAME_P){
    const r = MIDGAME_POOL[Math.floor(Math.random() * MIDGAME_POOL.length)];
    return { n:r.owner.length, cx:r.cx, cy:r.cy, owner:r.owner.slice(), troops:r.troops.slice(),
             armies:[], rlDiag:r.rlDiag || 1, adj:r.adj.map(a => a.slice()) };
  }
  if (REAL_MAPS && Math.random() < REAL_MAPS_P){
    const m = REAL_MAPS[Math.floor(Math.random() * REAL_MAPS.length)];
    const owner = new Array(m.n).fill(0), troops = new Array(m.n);
    const gameLike = Math.random() < (process.env.OPENING_MIX != null ? +process.env.OPENING_MIX : 0.5);
    for (let i = 0; i < m.n; i++) troops[i] = gameLike ? rnd(18, 55) : rnd(5, 45);
    const cx = m.cx, cy = m.cy;
    let mnx=Math.min(...cx), mxx=Math.max(...cx), mny=Math.min(...cy), mxy=Math.max(...cy);
    const rlDiag = Math.hypot(mxx-mnx, mxy-mny) || 1;
    const seeds = [Math.floor(Math.random() * m.n)];
    while (seeds.length < owners.length){
      let bi = -1, bd = -1;
      for (let i = 0; i < m.n; i++){ let d = 1e9; for (const s of seeds) d = Math.min(d, Math.hypot(cx[i]-cx[s], cy[i]-cy[s])); if (d > bd){ bd = d; bi = i; } }
      seeds.push(bi);
    }
    owners.forEach((o, k) => { owner[seeds[k]] = o; troops[seeds[k]] = gameLike ? 22 : 40; });
    return { n:m.n, cx, cy, owner, troops, armies:[], rlDiag, adj: m.adj.map(a => a.slice()) };
  }
  const cx=[], cy=[], owner=new Array(N).fill(0), troops=new Array(N);
  // Opening-distribution mix (2026-07-11): half the boards use the SHIPPING hard tiers' opening
  // (Impossible/Nightmare: 22-troop starts, 18-55 neutrals). The legacy training regime (40-troop
  // starts, 5-45 neutrals) essentially always offered a beatable neighbor on move one, so the net
  // never learned multi-wave chip attacks — and in the real tiers' openings (frequently NO beatable
  // neighbor) it noop-locked at 99-100% of visits while a human chipped and snowballed (play-test
  // 2026-07-11). Half the boards keep the legacy regime so easy openings stay in-distribution.
  // OPENING_MIX env (0..1) overrides the game-like share — for diagnosing per-regime strength
  // (e.g. OPENING_MIX=1 -> all hard game-like openings, 0 -> all legacy). Default stays 0.5.
  const gameLike = Math.random() < (process.env.OPENING_MIX != null ? +process.env.OPENING_MIX : 0.5);
  for (let i=0;i<N;i++){ cx.push(Math.random()); cy.push(Math.random()); troops[i]= gameLike ? rnd(18,55) : rnd(5,45); }
  let mnx=Math.min(...cx),mxx=Math.max(...cx),mny=Math.min(...cy),mxy=Math.max(...cy);
  const rlDiag = Math.hypot(mxx-mnx, mxy-mny) || 1;
  // seed each owner far apart (farthest-point), like train_rl.new_game
  const seeds=[Math.floor(Math.random()*N)];
  while (seeds.length < owners.length){
    let bi=-1, bd=-1;
    for (let i=0;i<N;i++){ let d=1e9; for (const s of seeds) d=Math.min(d, Math.hypot(cx[i]-cx[s], cy[i]-cy[s])); if (d>bd){ bd=d; bi=i; } }
    seeds.push(bi);
  }
  owners.forEach((o,k)=>{ owner[seeds[k]]=o; troops[seeds[k]]= gameLike ? 22 : 40; });
  const adj = delaunayAdjacency(cx, cy, N);
  // Real maps (TopoJSON borders) have corner states with only 1-2 neighbors — e.g. Maine — which
  // Delaunay graphs essentially never produce. Mirror train_rl.py new_game(): prune ~15% of nodes
  // down to pendants so corner seats are in-distribution for BOTH the gate and the GNN's training
  // data (this file is where all its self-play boards come from; play-data 2026-07-04 showed a net
  // that never saw a degree-1 seat noop-hoards on one). Only prune nodes whose neighbors are all
  // degree>=3, so the graph stays connected and two pendants never cascade into an island.
  const npend = Math.max(1, Math.floor(N*0.15));
  const cand = []; for (let i=0;i<N;i++) if (adj[i].length>=3) cand.push(i);
  for (let i=cand.length-1;i>0;i--){ const j=Math.floor(Math.random()*(i+1)); const t=cand[i]; cand[i]=cand[j]; cand[j]=t; }
  let pruned = 0;
  for (const i of cand){
    if (pruned >= npend) break;
    if (adj[i].some(j => adj[j].length <= 2)) continue;
    let keep=adj[i][0], bd=Infinity;
    for (const j of adj[i]){ const d=Math.hypot(cx[i]-cx[j], cy[i]-cy[j]); if (d<bd){ bd=d; keep=j; } }
    for (const j of adj[i]) if (j!==keep) adj[j]=adj[j].filter(x=>x!==i);
    adj[i]=[keep];
    pruned++;
  }
  return { n:N, cx, cy, owner, troops, armies:[], rlDiag, adj };
}
// DEPTH_RULES=1: play under the shipped game's depth mechanics (build 13+) — Lanchester-1.5
// combat, terrain gradient, supply lines (40% growth + 0.85x defence off-core), quadratic
// over-cap attrition, neutral restock. Mirrors index.html's depth-mode sim (CONTRACT) so the
// net trains on the game the tester actually plays. Invest is NOT modelled here (no seat uses
// it in the arena); classic physics remain the default so old measurements stay comparable.
const DEPTH_RULES = !!+(process.env.DEPTH_RULES||0);
const LANCH = 1.5;
function terrM(env, i){ const d=env.adj[i].length; return d<=2?1.4:d===3?1.2:1; }
function recomputeSupply(env){
  const sup = env._sup || (env._sup = new Array(env.n).fill(false));
  sup.fill(false);
  const byOwner = {};
  for (let i=0;i<env.n;i++) if (env.owner[i]!==0) (byOwner[env.owner[i]] = byOwner[env.owner[i]]||[]).push(i);
  for (const o in byOwner){
    const ids=byOwner[o]; if (ids.length<=1) continue;
    const own=new Set(ids), seen=new Set(), comps=[];
    for (const id of ids){ if (seen.has(id)) continue;
      const c=[id]; seen.add(id);
      for (let q=0;q<c.length;q++) for (const nb of env.adj[c[q]]) if (own.has(nb)&&!seen.has(nb)){ seen.add(nb); c.push(nb); }
      comps.push(c); }
    if (comps.length<=1) continue;
    comps.sort((a,b)=>b.length-a.length);
    for (let c=1;c<comps.length;c++) for (const id of comps[c]) sup[id]=true;
  }
}
function step(env, dt){
  if (DEPTH_RULES){
    env._supT = (env._supT||0) - dt;
    if (env._supT<=0){ env._supT = 0.5; recomputeSupply(env); }
  }
  const sup = env._sup;
  for (let i=0;i<env.n;i++){
    if (env.owner[i]!==0 && env.troops[i]<150){
      let g = 1.0*dt;
      if (DEPTH_RULES && sup && sup[i]) g *= 0.4;                         // off-supply growth
      env.troops[i]=Math.min(150, env.troops[i]+g);
    }
    else if (DEPTH_RULES && env.owner[i]!==0 && env.troops[i]>150){       // quadratic over-cap attrition (CONTRACT: index.html ATTRITION_Q)
      const ex=env.troops[i]-150;
      env.troops[i]=Math.max(150, env.troops[i]-(0.06*ex*ex/150)*dt);
    }
    if (DEPTH_RULES && env.owner[i]===0 && env.troops[i]<30) env.troops[i]+=0.2*dt;   // neutral restock
  }
  const still=[];
  for (const a of env.armies){ a.t+=dt;
    if (a.t>=a.ttotal){ const ti=a.ti;
      if (env.owner[ti]===a.owner) env.troops[ti]+=a.count;
      else if (DEPTH_RULES){
        const tm = terrM(env,ti)*(sup&&sup[ti]?0.85:1), dE = env.troops[ti]*tm;   // Lanchester+terrain+supply
        if (a.count>dE){ env.owner[ti]=a.owner; env.troops[ti]=Math.pow(Math.pow(a.count,LANCH)-Math.pow(dE,LANCH),1/LANCH); }
        else env.troops[ti]=Math.pow(Math.pow(dE,LANCH)-Math.pow(a.count,LANCH),1/LANCH)/tm;
      }
      else { env.troops[ti]-=a.count; if (env.troops[ti]<0){ env.owner[ti]=a.owner; env.troops[ti]=-env.troops[ti]; } } }
    else still.push(a); }
  env.armies=still;
}
// frac (0..1) = the unit-split fraction the GNN planner may choose; send that share, leave the rest.
function send(env, si, ti, frac){ const amt=env.troops[si]*(frac==null?1:frac); if (amt<1 || env.owner[si]===0) return;
  env.troops[si]-=amt; const d=Math.hypot(env.cx[ti]-env.cx[si], env.cy[ti]-env.cy[si]);
  env.armies.push({ owner:env.owner[si], count:amt, ti, t:0, ttotal:Math.max(0.05, d/ORB) }); }
function aliveOwners(env){ const s=new Set(); for (let i=0;i<env.n;i++) if (env.owner[i]!==0) s.add(env.owner[i]);
  for (const a of env.armies) if (a.owner!==0) s.add(a.owner); return s; }
// Exploration config for training-data generation (EXPLORE_EPS env var switches it on): Dirichlet
// noise on the planner's root priors + visit-proportional move sampling for the first
// EXPLORE_TEMP_MOVES decisions of each game (then greedy). Never set during gating runs, so gate
// numbers stay pure-strength measurements. Consumed by planWorkerMain via req.explore.
const EXPLORE = process.env.EXPLORE_EPS ? {
  eps: +process.env.EXPLORE_EPS,
  alpha: +(process.env.EXPLORE_ALPHA || 0.8),
  temp: +(process.env.EXPLORE_TEMP || 1.0),
} : null;
const EXPLORE_TEMP_MOVES = +(process.env.EXPLORE_TEMP_MOVES || 12);
function buildReq(env, owner, nDecision){
  const explore = EXPLORE ? { eps:EXPLORE.eps, alpha:EXPLORE.alpha,
    temp: (nDecision != null && nDecision < EXPLORE_TEMP_MOVES) ? EXPLORE.temp : 0 } : null;
  return { n:env.n, cx:env.cx, cy:env.cy, owner:env.owner.slice(), troops:env.troops.slice(), names:[], adj:env.adj,
    armies:env.armies.map(a=>({ owner:a.owner, count:a.count, ti:a.ti, ttotal:Math.max(0.02, a.ttotal-a.t) })),
    aiSpeed:AISPD, rlDiag:env.rlDiag, orbSpeed:ORB, growRate:1.0, enemyGrow:1.0, hunt:0, desp:0, infight:1,
    depth:DEPTH_RULES, terr:DEPTH_RULES ? env.owner.map((_,i)=>terrM(env,i)*(env._sup&&env._sup[i]?0.85:1)) : undefined,
    horizon:HORIZON, dt:DT, ownerId:owner, wantViz:false, explore,
    attn:+(process.env.ATTN_PRESSURE||0),   // Nightmare+ Lever 2 test hook (attention-pressure bias)
    oppW: process.env.OPP_W ? JSON.parse(process.env.OPP_W) : undefined,   // Lever 4c test hook
    mcts:{ tbudget:TBUDGET, maxIter:1e9, cPuct:CPUCT, K } };
}

// ---- scripted opponents (script:turtle|rush|econ|farmer|human as a policy path) ----
// JS ports of the exploiter league in tools/train_rl.py (see each one's comment there for the
// play-testing incident it encodes). Used to diversify TRAINING data: pure net-vs-net self-play
// never produces the board states these styles force, so the no-forget screen kept catching
// regressions the training data had no way to prevent. A scripted mover returns an ARRAY of
// [si,ti,frac] moves (the human bot joint-attacks; the rest return one move or none).
function directTargets(env, si, owner){
  const out=[]; for (const j of env.adj[si]) if (env.owner[j]!==owner) out.push(j);
  return out;
}
function biggestSrc(env, owner, minT){
  let si=-1, bt=-1;
  for (let i=0;i<env.n;i++) if (env.owner[i]===owner && env.troops[i]>=minT && env.troops[i]>bt){ bt=env.troops[i]; si=i; }
  return si;
}
function nearest(env, si, pool){
  let bj=-1, bd=Infinity;
  for (const j of pool){ const d=Math.hypot(env.cx[j]-env.cx[si], env.cy[j]-env.cy[si]); if (d<bd){ bd=d; bj=j; } }
  return bj;
}
const SCRIPTS = {
  turtle(env, owner){
    const si=biggestSrc(env, owner, 80); if (si<0) return null;
    const st=env.troops[si], safe=directTargets(env,si,owner).filter(j=>env.troops[j]<st*0.6);
    if (!safe.length) return null;
    return [[si, nearest(env,si,safe), 1.0]];
  },
  rush(env, owner){
    const si=biggestSrc(env, owner, 8); if (si<0) return null;
    const st=env.troops[si], tgts=directTargets(env,si,owner);
    if (!tgts.length) return null;
    const beat=tgts.filter(j=>env.troops[j]<st);
    return [[si, nearest(env,si,beat.length?beat:tgts), 1.0]];
  },
  econ(env, owner){
    let myStates=0; for (let i=0;i<env.n;i++) if (env.owner[i]===owner) myStates++;
    if (myStates<=2){
      const si=biggestSrc(env, owner, 10); if (si<0) return null;
      const st=env.troops[si];
      const neu=directTargets(env,si,owner).filter(j=>env.owner[j]===0 && env.troops[j]<st*0.7);
      if (!neu.length) return null;
      return [[si, nearest(env,si,neu), 1.0]];
    }
    const si=biggestSrc(env, owner, 70); if (si<0) return null;
    const st=env.troops[si], safe=directTargets(env,si,owner).filter(j=>env.troops[j]<st*0.5);
    if (!safe.length) return null;
    return [[si, nearest(env,si,safe), 1.0]];
  },
  farmer(env, owner){
    const en=[], mine=[];
    for (let i=0;i<env.n;i++){ if (env.owner[i]===owner) mine.push(i); else if (env.owner[i]!==0) en.push(i); }
    if (!mine.length) return null;
    const myT=mine.reduce((a,i)=>a+env.troops[i],0), enT=en.reduce((a,i)=>a+env.troops[i],0);
    if (!en.length || myT > 1.6*enT + 30){            // war phase: march the big stack at the enemy
      const si=biggestSrc(env, owner, 12); if (si<0) return null;
      const tgts=directTargets(env,si,owner); if (!tgts.length) return null;
      const hostile=tgts.filter(j=>env.owner[j]!==0);
      return [[si, nearest(env,si,hostile.length?hostile:tgts), 1.0]];
    }
    const ecx=en.reduce((a,i)=>a+env.cx[i],0)/en.length, ecy=en.reduce((a,i)=>a+env.cy[i],0)/en.length;
    let best=null;                                     // farm phase: affordable neutral FARTHEST from the enemy
    for (const si of mine){ if (env.troops[si]<10) continue;
      for (const ti of directTargets(env,si,owner)){
        if (env.owner[ti]!==0 || env.troops[ti]>=env.troops[si]*0.8) continue;
        const sc=Math.hypot(env.cx[ti]-ecx, env.cy[ti]-ecy) - 0.3*env.troops[ti];
        if (!best || sc>best[0]) best=[sc,si,ti];
      } }
    return best ? [[best[1],best[2],1.0]] : null;
  },
  // human-profile bot — plays like the measured human profiles from tools/parse_play_data.py
  // (nightmare.md Lever 4: "a scripted human-profile bot ... joins the training league"). The
  // signature behaviours: coordinated JOINT attacks from several sources at once (jointRate
  // 0.70-0.82 in ANTHONY's play data), partial-stack splits, and opportunistic expansion.
  // Override the defaults with a JSON object in the HUMAN_PROFILE env var.
  human(env, owner){
    const P = SCRIPTS._humanProfile || (SCRIPTS._humanProfile =
      Object.assign({ aggression:0.65, jointRate:0.75, jointMax:4, splitRate:0.25 },
                    process.env.HUMAN_PROFILE ? JSON.parse(process.env.HUMAN_PROFILE) : {}));
    const srcs=[]; for (let i=0;i<env.n;i++) if (env.owner[i]===owner && env.troops[i]>=10) srcs.push(i);
    if (!srcs.length) return null;
    // score every (src -> direct target): prefer winnable, cheap, close; enemy states per aggression
    let cands=[];
    for (const si of srcs){ const st=env.troops[si];
      for (const ti of directTargets(env,si,owner)){ const tt=env.troops[ti];
        const enemy=env.owner[ti]!==0;
        if (enemy && Math.random()>P.aggression) continue;
        const d=Math.hypot(env.cx[ti]-env.cx[si], env.cy[ti]-env.cy[si]);
        cands.push({ si, ti, sc: (st>tt?25:-10) - tt*1.1 - d*2.0 + (enemy?6:8) });
      } }
    if (!cands.length) return null;
    cands.sort((a,b)=>b.sc-a.sc);
    const top=cands[0];
    const frac=()=> Math.random()<P.splitRate ? 0.5 : 1.0;
    if (Math.random() < P.jointRate){
      // joint attack: every distinct source with a decent option on the SAME target joins the salvo
      const joint=[[top.si, top.ti, frac()]]; const used=new Set([top.si]);
      for (const c of cands){
        if (joint.length>=P.jointMax) break;
        if (c.ti!==top.ti || used.has(c.si) || c.sc < top.sc-30) continue;
        used.add(c.si); joint.push([c.si, c.ti, frac()]);
      }
      return joint;
    }
    return [[top.si, top.ti, frac()]];
  },
  // fader — the play-tester's measured farm-and-snowball style, fitted from his real exported
  // games by tools/fit_fader_profile.py (135 games / 8189 moves, 2026-07-18). Signature: ~71%
  // of his moves consolidate troops rearward->frontier, he grabs cheap neutrals, and he does
  // NOT attack until his total troops reach ~warRatio x the enemy's — then strikes as a
  // full-stack joint salvo at the weakest border state (median 7x local force in his logs).
  // Beating THIS bot = denying that snowball. FADER_PROFILE env overrides the fitted numbers.
  fader(env, owner){
    const P = SCRIPTS._faderProfile || (SCRIPTS._faderProfile =
      Object.assign({ reinforceRate:0.712, neutralRate:0.146, warRatio:3.0, fracAll:0.785,
                      stackTarget:100, jointMax:4 },
                    process.env.FADER_PROFILE ? JSON.parse(process.env.FADER_PROFILE) : {}));
    const mine=[], en=[];
    for (let i=0;i<env.n;i++){ if (env.owner[i]===owner) mine.push(i); else if (env.owner[i]!==0) en.push(i); }
    if (!mine.length) return null;
    const myT=mine.reduce((a,i)=>a+env.troops[i],0), enT=en.reduce((a,i)=>a+env.troops[i],0);
    const frac=()=> Math.random()<P.fracAll ? 1.0 : 0.5;
    const grabNeutral=()=>{ let best=null;
      for (const si of mine){ const st=env.troops[si]; if (st<8) continue;
        for (const ti of env.adj[si]){ if (env.owner[ti]!==0 || env.troops[ti]>=st*0.8) continue;
          const d=Math.hypot(env.cx[ti]-env.cx[si], env.cy[ti]-env.cy[si]);
          const sc=-env.troops[ti]*1.2 - d*0.5;
          if (!best || sc>best[0]) best=[sc,si,ti]; } }
      return best ? [[best[1],best[2],frac()]] : null; };
    // WAR phase: snowball ready (total troops >= warRatio x enemy) -> joint salvo, biggest
    // stacks adjacent to the WEAKEST border enemy state, full sends.
    if (!en.length || myT >= P.warRatio*enT){
      let tgt=-1, tt=Infinity;
      for (const si of mine) for (const j of env.adj[si])
        if (env.owner[j]!==owner && env.owner[j]!==0 && env.troops[j]<tt){ tt=env.troops[j]; tgt=j; }
      if (tgt<0) return grabNeutral();               // no enemy on the border yet: keep expanding
      const srcs=mine.filter(si=>env.adj[si].indexOf(tgt)>=0 && env.troops[si]>=8)
                     .sort((a,b)=>env.troops[b]-env.troops[a]).slice(0, P.jointMax);
      if (!srcs.length) return null;
      return srcs.map(si=>[si, tgt, 1.0]);
    }
    // DEFEND first: a frontier state about to be lost gets a joint reinforcement — his 71%
    // reinforce rate is banking AND defense, and without this the bot folds to any pressure.
    const incoming={};
    for (const arm of env.armies) if (arm.owner!==owner && env.owner[arm.ti]===owner)
      incoming[arm.ti]=(incoming[arm.ti]||0)+arm.count;
    let danger=null;
    for (const si of mine){
      let th=incoming[si]||0;
      for (const j of env.adj[si]) if (env.owner[j]!==owner && env.owner[j]!==0) th=Math.max(th, env.troops[j]*0.7);
      if (th > env.troops[si] && (!danger || th-env.troops[si] > danger[0])) danger=[th-env.troops[si], si];
    }
    if (danger){
      const si=danger[1];
      const helpers=mine.filter(s=>s!==si && env.adj[s].indexOf(si)>=0 && env.troops[s]>=8)
                        .sort((a,b)=>env.troops[b]-env.troops[a]).slice(0, P.jointMax);
      if (helpers.length) return helpers.map(s=>[s, si, 1.0]);
    }
    // ECON phase: roll his measured move mix — reinforce (consolidate rear stacks toward the
    // frontier bank), grab a cheap neutral, or an opportunistic 2:1+ border attack.
    const r=Math.random(), pR=P.reinforceRate, pN=P.reinforceRate+P.neutralRate;
    if (r < pR){
      const frontier=mine.filter(si=>env.adj[si].some(j=>env.owner[j]!==owner));
      if (!frontier.length) return grabNeutral();
      let bank=frontier[0]; for (const si of frontier) if (env.troops[si]>env.troops[bank]) bank=si;
      if (env.troops[bank] >= Math.min(145, P.stackTarget*1.4)) return grabNeutral();  // bank full: convert
      let best=null;                                  // biggest rear stack with an own-neighbor closer to the bank
      for (const si of mine){ if (si===bank || env.troops[si]<8) continue;
        const dSelf=Math.hypot(env.cx[bank]-env.cx[si], env.cy[bank]-env.cy[si]);
        for (const ti of env.adj[si]){ if (env.owner[ti]!==owner) continue;
          const d=Math.hypot(env.cx[bank]-env.cx[ti], env.cy[bank]-env.cy[ti]);
          if (d<dSelf-1e-6){ const sc=env.troops[si];
            if (!best || sc>best[0]) best=[sc,si,ti]; } } }
      return best ? [[best[1],best[2],frac()]] : grabNeutral();
    }
    if (r < pN) return grabNeutral();
    let atk=null;                                     // opportunistic: only a clearly-won border fight
    for (const si of mine){ const st=env.troops[si]; if (st<12) continue;
      for (const ti of env.adj[si]){ if (env.owner[ti]===owner || env.owner[ti]===0) continue;
        if (st > env.troops[ti]*2.0){ const sc=st-env.troops[ti];
          if (!atk || sc>atk[0]) atk=[sc,si,ti]; } } }
    return atk ? [[atk[1],atk[2],frac()]] : grabNeutral();
  },
  // patient — the exploit a real human runs against a passive AI (reported 2026-07-22): the AI
  // grabs ~2 states then parks a hoard and shuffles it, so the human just OUT-FARMS it with
  // unlimited time and rolls one overwhelming stack in late. This bot is that human: it (1) grabs
  // EVERY affordable neutral each turn (multi-move economic expansion, no time pressure), (2) will
  // NOT touch the enemy until its total troops crush the enemy's (warRatio, default 2.6 — much more
  // patient than fader's 3.0-with-defense), and (3) when finally ahead, sends a MAX joint salvo of
  // every adjacent stack at the weakest enemy border state. No defensive fold, no early trading —
  // pure snowball. Beating THIS is the real target: the net must learn a static hoard loses to
  // someone who simply grows bigger. PATIENT_PROFILE env overrides the numbers.
  patient(env, owner){
    const P = SCRIPTS._patientProfile || (SCRIPTS._patientProfile =
      Object.assign({ warRatio:2.6, jointMax:6, neutralAfford:0.8, minStack:8 },
                    process.env.PATIENT_PROFILE ? JSON.parse(process.env.PATIENT_PROFILE) : {}));
    const mine=[], en=[];
    for (let i=0;i<env.n;i++){ if (env.owner[i]===owner) mine.push(i); else if (env.owner[i]!==0) en.push(i); }
    if (!mine.length) return null;
    const myT=mine.reduce((a,i)=>a+env.troops[i],0), enT=en.reduce((a,i)=>a+env.troops[i],0);
    // WAR: crushing lead banked -> one overwhelming joint salvo at the weakest border enemy.
    if (en.length && myT >= P.warRatio*enT){
      let tgt=-1, tt=Infinity;
      for (const si of mine) for (const j of env.adj[si])
        if (env.owner[j]!==owner && env.owner[j]!==0 && env.troops[j]<tt){ tt=env.troops[j]; tgt=j; }
      if (tgt>=0){
        const srcs=mine.filter(si=>env.adj[si].indexOf(tgt)>=0 && env.troops[si]>=P.minStack)
                       .sort((a,b)=>env.troops[b]-env.troops[a]).slice(0, P.jointMax);
        if (srcs.length) return srcs.map(si=>[si, tgt, 1.0]);
      }
    }
    // FARM: grab every affordable neutral this turn (one move per source, joint expansion).
    const grabs=[]; const usedSrc=new Set();
    for (const si of mine){ const st=env.troops[si]; if (st<P.minStack || usedSrc.has(si)) continue;
      let bj=-1, bd=Infinity;
      for (const ti of env.adj[si]){ if (env.owner[ti]!==0 || env.troops[ti]>=st*P.neutralAfford) continue;
        const d=Math.hypot(env.cx[ti]-env.cx[si], env.cy[ti]-env.cy[si]);
        if (d<bd){ bd=d; bj=ti; } }
      if (bj>=0){ grabs.push([si,bj,1.0]); usedSrc.add(si); } }
    if (grabs.length) return grabs;
    // BANK: no neutral to take and not yet crushing — consolidate rear stacks toward the frontier
    // so a staging army is always ready to convert the moment the lead is there.
    const frontier=mine.filter(si=>env.adj[si].some(j=>env.owner[j]!==owner));
    if (frontier.length){
      let bank=frontier[0]; for (const si of frontier) if (env.troops[si]>env.troops[bank]) bank=si;
      let best=null;
      for (const si of mine){ if (si===bank || env.troops[si]<P.minStack) continue;
        const dSelf=Math.hypot(env.cx[bank]-env.cx[si], env.cy[bank]-env.cy[si]);
        for (const ti of env.adj[si]){ if (env.owner[ti]!==owner) continue;
          const d=Math.hypot(env.cx[bank]-env.cx[ti], env.cy[bank]-env.cy[ti]);
          if (d<dSelf-1e-6){ const sc=env.troops[si]; if (!best || sc>best[0]) best=[sc,si,ti]; } } }
      if (best) return [[best[1],best[2],1.0]];
    }
    return null;
  },
};
function makeScripted(kind){
  if (!SCRIPTS[kind]) { console.error(`unknown scripted opponent "${kind}" (script:turtle|rush|econ|farmer|human|fader|patient)`); process.exit(1); }
  const fn = (env, owner) => SCRIPTS[kind](env, owner);
  fn.scripted = true; fn.kind = kind;
  return fn;
}
// re-derive the 12-feature vector for a (source,target) candidate from board state — mirrors
// computeUntried()'s feature formula in index.html exactly. Used only for --dump-visits training
// data capture; not part of the planner's own decision path.
function candFeats12(env, si, ti, owner){
  const FN = 50;
  let ownCount=0; for (let i=0;i<env.n;i++) if (env.owner[i]===owner) ownCount++;
  const ownFrac = ownCount/env.n;
  let myTot=0, allTot=0, maxEnemy=0, bigIdx=-1;
  for (let i=0;i<env.n;i++){ const oo=env.owner[i], tr=env.troops[i];
    if (oo!==0){ allTot+=tr; if (oo===owner) myTot+=tr; else if (tr>maxEnemy){ maxEnemy=tr; bigIdx=i; } } }
  const myShare = allTot>0 ? myTot/allTot : 0, myAvg = myTot/Math.max(1,ownCount);
  const st=env.troops[si], tt=env.troops[ti];
  const dist = Math.hypot(env.cx[ti]-env.cx[si], env.cy[ti]-env.cy[si]) / env.rlDiag;
  return [ st/FN, tt/FN, (st-tt)/FN, dist,
    env.owner[ti]===0?1:0, (env.owner[ti]!==0 && env.owner[ti]!==owner)?1:0, st>tt?1:0, ownFrac,
    maxEnemy/FN, myShare, myAvg/FN, (ti===bigIdx?1:0) ];
}

// ---- visit-count capture, for training the policy net to imitate the real planner's own search
// (--dump-visits / DUMP_VISITS env var). Off by default; zero cost to normal arena gating runs. ----
// Decisions are BUFFERED per game and flushed at game end so every record carries the game's final
// `result` (graded outcome from the mover's perspective, same formula as train_value.py result()),
// enabling value-net co-training from planner self-play. At flush time, noop-dominated decisions
// are subsampled (DUMP_KEEP_NOOP, default 0.1): ~93% of raw decisions are "sit still" (measured
// 2026-07-11), which drowns the move-choice signal the policy distillation actually needs.
const DUMP_STREAM = process.env.DUMP_VISITS ? fs.createWriteStream(process.env.DUMP_VISITS, {flags:'a'}) : null;
const KEEP_NOOP = process.env.DUMP_KEEP_NOOP != null ? +process.env.DUMP_KEEP_NOOP : 0.1;
let DUMP_GAME_SEQ = 0;   // per-process game id stamped into records — real-map games share board
                         // coordinates, so analysis code cannot infer game boundaries from geometry
function dumpDecision(env, owner, planFn, sink){
  if (!DUMP_STREAM) return;
  const roots = planFn.lastRoots;
  if (!roots || !roots.length) return;
  // null feats entry = the no-op candidate (its "score" in the exported net is noop_bias alone,
  // it never goes through the 12-input MLP) — the training loop must special-case it, not skip it.
  const feats = roots.map(r => r.move ? candFeats12(env, r.move[0], r.move[1], owner) : null);
  const visits = roots.map(r => r.visits);
  let chosen = 0; for (let i=1;i<visits.length;i++) if (visits[i]>visits[chosen]) chosen = i;
  // raw board snapshot too — the flat-MLP trainer (train_rl_torch.py) only needs `feats`/`visits`
  // above, but the graph-aware trainer (train_gnn_torch.py) needs the actual board+adjacency to
  // (re)compute per-node features, which a flat per-move feature vector can't reconstruct.
  const moves = roots.map(r => r.move);   // null entries are the no-op candidate, same alignment as feats/visits
  sink.push({
    feats, visits, chosen, moves, mover: owner,
    owner: env.owner.slice(), troops: env.troops.slice(), adj: env.adj, cx: env.cx, cy: env.cy, rlDiag: env.rlDiag,
    armies: env.armies.map(a => ({ owner: a.owner, count: a.count, ti: a.ti })),
  });
}
// graded final outcome for `mover`, mirroring tools/train_value.py result(): +/-1 on elimination,
// state-count margin (clamped, x1.5) on timeout — so value regression sees "how decisively", not
// just win/lose, exactly like the value net's existing Monte-Carlo training data.
function gradeResult(env, mover){
  const alive = aliveOwners(env);
  if (alive.size === 1) return alive.has(mover) ? 1.0 : -1.0;
  if (!alive.has(mover)) return -1.0;
  let own = 0, bestOther = 0;
  const cnt = {};
  for (let i=0;i<env.n;i++) if (env.owner[i]!==0) cnt[env.owner[i]] = (cnt[env.owner[i]]||0)+1;
  for (const o in cnt){ if (+o === mover) own = cnt[o]; else bestOther = Math.max(bestOther, cnt[o]); }
  return Math.max(-1, Math.min(1, (own - bestOther)/env.n*1.5));
}
function flushDump(env, sink){
  if (!DUMP_STREAM || !sink.length) return;
  const gid = ++DUMP_GAME_SEQ;
  for (const rec of sink){
    rec.g = gid;
    const total = rec.visits.reduce((a,b)=>a+b, 0) || 1;
    let moveMass = 0;
    for (let i=0;i<rec.moves.length;i++) if (rec.moves[i] !== null) moveMass += rec.visits[i];
    const chosenIsMove = rec.moves[rec.chosen] !== null;
    // keep every decision where the search actually engaged with moving; subsample the rest
    if (!chosenIsMove && moveMass/total < 0.2 && Math.random() >= KEEP_NOOP) continue;
    rec.result = gradeResult(env, rec.mover);
    DUMP_STREAM.write(JSON.stringify(rec) + '\n');
  }
}

// ---- one game: ownerA vs ownerB (both 'enemies' so each models the other identically/fairly) ----
// Per-side TIER EMULATION (nightmare_plus.md benchmarking): a planner fn may carry .cadence
// (decision interval, s), .budget (ms/move think time), and .multi (salvo width — commit up to K
// mutually-compatible root moves per decision, mirroring onPlanWorkerMsg's Nightmare logic). Set
// via A_CADENCE/A_TBUDGET/A_MULTI and B_* env vars in main(). Defaults preserve classic behavior.
function commitSalvo(env, o, planFn, multi, mv){
  if (multi <= 1){ if (mv) send(env, mv[0], mv[1], mv.length>2 ? mv[2] : 1.0); return; }
  const roots = planFn.lastRoots || [];
  let noopV = -1; const cands = [];
  for (const r of roots){ if (!r.move) noopV = r.visits; else cands.push(r); }
  cands.sort((a,b) => b.visits - a.visits);
  const usedSrc = new Set(), usedTgt = new Set(); let done = 0;
  for (const c of cands){
    if (done >= multi) break;
    if (c.visits <= noopV) break;               // sorted: everything after fails too
    const m = c.move;
    if (env.owner[m[0]] !== o) continue;
    if (usedSrc.has(m[0]) || usedTgt.has(m[1])) continue;   // one send per source; no same-tick dogpile
    send(env, m[0], m[1], m.length>2 ? m[2] : 1.0);
    usedSrc.add(m[0]); usedTgt.add(m[1]); done++;
  }
}
function playGame(planA, planB, ownerA, ownerB){
  const env = genBoard(NSTATES, [ownerA, ownerB]);
  const planners = { [ownerA]:planA, [ownerB]:planB };
  const cadOf = f => f.cadence || CADENCE;
  const timers = { [ownerA]:rnd(0.1,cadOf(planA)), [ownerB]:rnd(0.1,cadOf(planB)) };
  const nDecisions = { [ownerA]:0, [ownerB]:0 };   // per-seat decision count (exploration temperature window)
  const air = {};                                   // econ's airdrop cooldown (+25 to its hoard every ~7s, see train_rl.py play())
  const dumpBuf = [];                               // decisions buffered here; flushed with the game's final result
  let t=0;
  while (t < GAMECAP){
    step(env, ENVDT); t+=ENVDT;
    for (const o of [ownerA, ownerB]){
      const planFn=planners[o];
      if (planFn.scripted && (planFn.kind==='econ' || planFn.kind==='fader')){
        // econ: the airdrop hoard exploit (train_rl.py). fader: the play-tester runs the full
        // coin-shop economy (investor / airdrops / boost — see his exported localStorage), so his
        // bot gets the same +25/7s drip; the counter-net must beat his ECONOMY, not a stripped
        // version of him.
        air[o] = (air[o]==null ? 7.0 : air[o]) - ENVDT;
        if (air[o]<=0){ air[o]=7.0; const si=biggestSrc(env,o,0);
          if (si>=0) env.troops[si] += +(process.env.FADER_DRIP || 25); }   // FADER_DRIP tunes the perk economy
      }
      timers[o]-=ENVDT;
      if (timers[o]<=0){ const cad=cadOf(planFn); timers[o]=rnd(cad*0.7, cad*1.3);
        if (aliveOwners(env).has(o)){
          if (planFn.scripted){
            const mvs=planFn(env, o) || [];
            for (const m of mvs) send(env, m[0], m[1], m[2]);
          } else {
            const req=buildReq(env, o, nDecisions[o]++);
            if (planFn.budget) req.mcts.tbudget=planFn.budget;
            const mv=planFn(req);
            dumpDecision(env, o, planFn, dumpBuf);
            commitSalvo(env, o, planFn, planFn.multi||1, mv);
          } } }
    }
    const al=aliveOwners(env); if (al.size<=1) break;
  }
  flushDump(env, dumpBuf);   // stamp each buffered decision with the final graded outcome, then write
  // winner = most states (then most troops); null = draw/timeout tie
  let bestO=0, bestS=-1, bestT=-1;
  for (const o of [ownerA, ownerB]){ let s=0, tr=0;
    for (let i=0;i<env.n;i++) if (env.owner[i]===o){ s++; tr+=env.troops[i]; }
    if (s>bestS || (s===bestS && tr>bestT)){ bestS=s; bestT=tr; bestO=o; } }
  return bestO;
}

function main(){
  const mkSide = (path, blend, html) => path.startsWith('script:')
    ? makeScripted(path.slice(7))
    : makePlanner(JSON.parse(fs.readFileSync(path,'utf8')), blend, html);
  const planA = mkSide(pathA, BLEND_A, process.env.HTML_A), planB = mkSide(pathB, BLEND_B, process.env.HTML_B);
  // per-side tier emulation (see playGame): cadence in seconds, budget in ms, multi = salvo width
  for (const [fn, p] of [[planA,'A'],[planB,'B']]){
    if (process.env[p+'_CADENCE']) fn.cadence = +process.env[p+'_CADENCE'];
    if (process.env[p+'_TBUDGET']) fn.budget  = +process.env[p+'_TBUDGET'];
    if (process.env[p+'_MULTI'])   fn.multi   = +process.env[p+'_MULTI'];
  }
  const side = f => (f.cadence||f.budget||f.multi) ? ` [cad ${f.cadence||CADENCE}s budget ${f.budget||TBUDGET}ms salvo ${f.multi||1}]` : '';
  if (side(planA)+side(planB)) console.log(`TIER EMULATION: A${side(planA)||' [default]'}  B${side(planB)||' [default]'}`);
  const lays = (fn, path, bl) => {
    if (fn.scripted) return `scripted:${fn.kind}`;
    const p = JSON.parse(fs.readFileSync(path,'utf8'));
    return (p.format==='gnn-v1' ? `gnn embed${p.embed_dim}x${p.hops}hop` : p.layers.map(l=>l.b.length).join('-'))
      + (p.value?` +value(w=${bl==null?0.5:bl})`:'');
  };
  console.log(`ARENA: A=${pathA} [${lays(planA,pathA,BLEND_A)}]  vs  B=${pathB} [${lays(planB,pathB,BLEND_B)}]`
    + (EXPLORE ? `  explore(eps=${EXPLORE.eps} alpha=${EXPLORE.alpha} temp=${EXPLORE.temp}x${EXPLORE_TEMP_MOVES})` : ''));
  console.log(`games=${GAMES} budget=${TBUDGET}ms/move states=${NSTATES} (same budget => slower net does fewer sims)`);
  let aWins=0, bWins=0;
  const t0=performance.now();
  for (let g=0; g<GAMES; g++){
    // alternate which physical owner each config controls, to cancel seat/position bias
    let winner;
    if (g % 2 === 0){ winner = playGame(planA, planB, 2, 3); if (winner===2) aWins++; else if (winner===3) bWins++; }
    else            { winner = playGame(planB, planA, 2, 3); if (winner===2) bWins++; else if (winner===3) aWins++; }
    if ((g+1) % 5 === 0){
      const n=aWins+bWins, wr=n?aWins/n:0, se=n?Math.sqrt(wr*(1-wr)/n):0;
      console.log(`  ${g+1}/${GAMES}  A ${aWins} - ${bWins} B   A-winrate ${(wr*100).toFixed(1)}% +/-${(se*196).toFixed(1)}  (${((performance.now()-t0)/1000).toFixed(0)}s)`);
    }
  }
  const n=aWins+bWins, wr=n?aWins/n:0, se=n?Math.sqrt(wr*(1-wr)/n):0;
  console.log(`\nFINAL: A won ${aWins}/${n} = ${(wr*100).toFixed(1)}%  (95% CI +/-${(se*196).toFixed(1)} pts)`);
  console.log(wr>0.5+se*1.96 ? 'A is STRONGER (significant)' : wr<0.5-se*1.96 ? 'B is STRONGER (significant)' : 'no significant difference');
}
main();
