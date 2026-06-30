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
  return (req) => { if (valueBlend!=null) req.valueBlend = valueBlend;   // per-planner leaf-eval blend
    self.onmessage({ data:{ type:'plan', id:1, req } }); return self._last && self._last.move; };
}

// ---- faithful environment (mirrors the worker's snapStep/snapSend: growth + orb travel + combat) ----
function rnd(a,b){ return a + Math.random()*(b-a); }
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
  return { n:N, cx, cy, owner, troops, armies:[], rlDiag };
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
  return { n:env.n, cx:env.cx, cy:env.cy, owner:env.owner.slice(), troops:env.troops.slice(), names:[],
    armies:env.armies.map(a=>({ owner:a.owner, count:a.count, ti:a.ti, ttotal:Math.max(0.02, a.ttotal-a.t) })),
    aiSpeed:AISPD, rlDiag:env.rlDiag, orbSpeed:ORB, growRate:1.0, enemyGrow:1.0, hunt:0, desp:0, infight:1,
    horizon:HORIZON, dt:DT, ownerId:owner, wantViz:false,
    mcts:{ tbudget:TBUDGET, maxIter:1e9, cPuct:CPUCT, K } };
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
        if (aliveOwners(env).has(o)){ const mv=planners[o](buildReq(env, o));
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
