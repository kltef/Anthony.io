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
const ORB = 70, AISPD = 0.6, ENVDT = 0.2, GAMECAP = 38, CADENCE = 0.7;

// ---- extract planWorkerMain() source verbatim (per html file) and instantiate as a callable planner ----
// HTML_A / HTML_B env vars override the worker source per config, so a 16-input net (new index.html)
// can play a 12-input net (snapshotted old index.html) across a feature-set change.
const _srcCache = {};
function loadWorkerSrc(path){
  if (_srcCache[path]) return _srcCache[path];
  const html = fs.readFileSync(path, 'utf8');
  const m = html.match(/function planWorkerMain\(\)\s*\{[\s\S]*?\n  \}\n/);
  if (!m) { console.error('could not extract planWorkerMain from '+path); process.exit(1); }
  return (_srcCache[path] = m[0]);
}
function makePlanner(policy, valueBlend, htmlPath){
  const workerSrc = loadWorkerSrc(htmlPath || htmlPath0);
  const self = { postMessage:(msg)=>{ self._last = msg; }, performance };
  // free variable `self` inside planWorkerMain resolves to this param lexically
  const install = new Function('self', '(' + workerSrc + ')();');
  install(self);
  self.onmessage({ data:{ type:'init', policy } });
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
function genBoard(N, owners){
  const cx=[], cy=[], owner=new Array(N).fill(0), troops=new Array(N);
  for (let i=0;i<N;i++){ cx.push(Math.random()); cy.push(Math.random()); troops[i]=rnd(5,45); }
  let mnx=Math.min(...cx),mxx=Math.max(...cx),mny=Math.min(...cy),mxy=Math.max(...cy);
  const rlDiag = Math.hypot(mxx-mnx, mxy-mny) || 1;
  // seed each owner far apart (farthest-point), like train_rl.new_game
  const seeds=[Math.floor(Math.random()*N)];
  while (seeds.length < owners.length){
    let bi=-1, bd=-1;
    for (let i=0;i<N;i++){ let d=1e9; for (const s of seeds) d=Math.min(d, Math.hypot(cx[i]-cx[s], cy[i]-cy[s])); if (d>bd){ bd=d; bi=i; } }
    seeds.push(bi);
  }
  owners.forEach((o,k)=>{ owner[seeds[k]]=o; troops[seeds[k]]=40; });
  const adj = delaunayAdjacency(cx, cy, N);
  return { n:N, cx, cy, owner, troops, armies:[], rlDiag, adj };
}
function step(env, dt){
  for (let i=0;i<env.n;i++) if (env.owner[i]!==0 && env.troops[i]<150) env.troops[i]=Math.min(150, env.troops[i]+1.0*dt);
  const still=[];
  for (const a of env.armies){ a.t+=dt;
    if (a.t>=a.ttotal){ const ti=a.ti;
      if (env.owner[ti]===a.owner) env.troops[ti]+=a.count;
      else { env.troops[ti]-=a.count; if (env.troops[ti]<0){ env.owner[ti]=a.owner; env.troops[ti]=-env.troops[ti]; } } }
    else still.push(a); }
  env.armies=still;
}
function send(env, si, ti){ const amt=env.troops[si]; if (amt<1 || env.owner[si]===0) return;
  env.troops[si]=0; const d=Math.hypot(env.cx[ti]-env.cx[si], env.cy[ti]-env.cy[si]);
  env.armies.push({ owner:env.owner[si], count:amt, ti, t:0, ttotal:Math.max(0.05, d/ORB) }); }
function aliveOwners(env){ const s=new Set(); for (let i=0;i<env.n;i++) if (env.owner[i]!==0) s.add(env.owner[i]);
  for (const a of env.armies) if (a.owner!==0) s.add(a.owner); return s; }
function buildReq(env, owner){
  return { n:env.n, cx:env.cx, cy:env.cy, owner:env.owner.slice(), troops:env.troops.slice(), names:[], adj:env.adj,
    armies:env.armies.map(a=>({ owner:a.owner, count:a.count, ti:a.ti, ttotal:Math.max(0.02, a.ttotal-a.t) })),
    aiSpeed:AISPD, rlDiag:env.rlDiag, orbSpeed:ORB, growRate:1.0, enemyGrow:1.0, hunt:0, desp:0, infight:1,
    horizon:HORIZON, dt:DT, ownerId:owner, wantViz:false,
    mcts:{ tbudget:TBUDGET, maxIter:1e9, cPuct:CPUCT, K } };
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
const DUMP_STREAM = process.env.DUMP_VISITS ? fs.createWriteStream(process.env.DUMP_VISITS, {flags:'a'}) : null;
function dumpDecision(env, owner, planFn){
  if (!DUMP_STREAM) return;
  const roots = planFn.lastRoots;
  if (!roots || !roots.length) return;
  // null feats entry = the no-op candidate (its "score" in the exported net is noop_bias alone,
  // it never goes through the 12-input MLP) — the training loop must special-case it, not skip it.
  const feats = roots.map(r => r.move ? candFeats12(env, r.move[0], r.move[1], owner) : null);
  const visits = roots.map(r => r.visits);
  let chosen = 0; for (let i=1;i<visits.length;i++) if (visits[i]>visits[chosen]) chosen = i;
  DUMP_STREAM.write(JSON.stringify({ feats, visits, chosen }) + '\n');
}

// ---- one game: ownerA vs ownerB (both 'enemies' so each models the other identically/fairly) ----
function playGame(planA, planB, ownerA, ownerB){
  const env = genBoard(NSTATES, [ownerA, ownerB]);
  const planners = { [ownerA]:planA, [ownerB]:planB };
  const timers = { [ownerA]:rnd(0.1,CADENCE), [ownerB]:rnd(0.1,CADENCE) };
  let t=0;
  while (t < GAMECAP){
    step(env, ENVDT); t+=ENVDT;
    for (const o of [ownerA, ownerB]){
      timers[o]-=ENVDT;
      if (timers[o]<=0){ timers[o]=rnd(CADENCE*0.7, CADENCE*1.3);
        if (aliveOwners(env).has(o)){ const planFn=planners[o]; const mv=planFn(buildReq(env, o));
          dumpDecision(env, o, planFn);
          if (mv) send(env, mv[0], mv[1]); } }
    }
    const al=aliveOwners(env); if (al.size<=1) break;
  }
  // winner = most states (then most troops); null = draw/timeout tie
  let bestO=0, bestS=-1, bestT=-1;
  for (const o of [ownerA, ownerB]){ let s=0, tr=0;
    for (let i=0;i<env.n;i++) if (env.owner[i]===o){ s++; tr+=env.troops[i]; }
    if (s>bestS || (s===bestS && tr>bestT)){ bestS=s; bestT=tr; bestO=o; } }
  return bestO;
}

function main(){
  const polA = JSON.parse(fs.readFileSync(pathA,'utf8'));
  const polB = JSON.parse(fs.readFileSync(pathB,'utf8'));
  const planA = makePlanner(polA, BLEND_A, process.env.HTML_A), planB = makePlanner(polB, BLEND_B, process.env.HTML_B);
  const lays = (p,bl) => p.layers.map(l=>l.b.length).join('-') + (p.value?` +value(w=${bl==null?0.5:bl})`:'');
  console.log(`ARENA: A=${pathA} [${lays(polA,BLEND_A)}]  vs  B=${pathB} [${lays(polB,BLEND_B)}]`);
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
