#!/usr/bin/env node
// Conformance trace generator. Extracts the ORIGINAL simulation functions straight out of the
// shipped game (git HEAD:src/web/index.html, i.e. before the engine replaced them), runs random
// games through them, and writes traces that `cargo run --bin conformance` replays through the
// Rust engine and compares state by state.
//   usage: node engine/tools/conformance_gen.js [games=24] [outdir=engine/target/conformance] [--fix-terrain]
const fs = require('fs'), path = require('path'), vm = require('vm'), cp = require('child_process');
const root = path.join(__dirname, '..', '..');
const args = process.argv.slice(2).filter(a => !a.startsWith('--'));
const GAMES = +(args[0] || 24);
const OUT = args[1] || path.join(root, 'engine', 'target', 'conformance');
const FIX_TERRAIN = process.argv.includes('--fix-terrain');
const REF = process.env.CONF_REF || '74e63cf';   // the last commit that shipped the original JS simulation
let html = cp.execSync('git show ' + REF + ':src/web/index.html', { cwd: root, maxBuffer: 1 << 28 }).toString('utf8');
// The shipped terrMult reads `.length` on a Set (always undefined -> terrain never applies).
// --fix-terrain patches that so the intended terrain rule can be checked as well.
if (FIX_TERRAIN) html = html.replace(/adjacency\[si\]\.length/g, 'adjacency[si].size');
// Deliberate rule change in the engine: every owner uses the upgrade-tier caps (the shipped game
// gave AIs a flat 150 even on a captured, upgraded state). Patch the reference to the new rule.
{ const old = "(s.owner===PLAYER) ? PLAYER_CAP[s.level||0]";
  if (!html.includes(old)) throw new Error('capOf patch target missing');
  html = html.replace(old, "(s.owner!==NEUTRAL) ? PLAYER_CAP[s.level||0]"); }

function grabBalanced(start, open, close){
  let i = html.indexOf(open, start), depth = 0;
  for (; i < html.length; i++){
    const c = html[i];
    if (c === open) depth++;
    else if (c === close){ depth--; if (depth === 0) return i + 1; }
  }
  throw new Error('unbalanced');
}
function grabFn(name){
  const at = html.indexOf('\n  function ' + name + '(');
  if (at < 0) throw new Error('function not found: ' + name);
  return html.slice(at + 1, grabBalanced(html.indexOf(')', at), '{', '}'));
}
function grabConst(name){
  const at = html.indexOf('\n  const ' + name + ' ');
  if (at < 0) throw new Error('const not found: ' + name);
  let i = at + 1, depth = 0;
  for (; i < html.length; i++){
    const c = html[i];
    if ('([{'.includes(c)) depth++; else if (')]}'.includes(c)) depth--;
    else if (c === ';' && depth === 0) break;
  }
  return html.slice(at + 1, i + 1);
}
const src = [
  'const NEUTRAL=0, PLAYER=1, MAX_TROOPS=150;',
  'let states=[], troops=[], adjacency=[], cfg={}, gameElapsed=0, gameTick=0, netMode=null, myOwner=1;',
  'let matchTrack=null, aiSnow=0, supplyClock=0, aiInvClock=0, slotControl={}, running=false, boomUntil=0;',
  'let ORB_SPEED=70, orbSeq=0, markR=17, lastReinforce={}, adjacencyMode=true, depthMode=true, upgradesOn=true;',
  'const humanSnowball=()=>0, updateThreats=()=>{}, tutHintsTick=()=>{}, updateAI=()=>{};',
  ...['PLAYER_CAP', 'upgradeCost', 'TILE', 'depthOn', 'LANCH', 'ATTRITION_Q', 'OFFSUP_DEF', 'terrMult', 'investCost', 'invGrowMult'].map(grabConst),
  ...['capOf', 'tileGrowMult', 'growRateFor', 'recomputeSupply', 'isLegalMove', 'sendTroops', 'arrive', 'update', 'tryInvest', 'tryUpgrade'].map(grabFn),
  'this.api = { update, sendTroops, tryInvest, tryUpgrade, recomputeSupply,',
  '  set(o){ if(o.states) states=o.states; if(o.adjacency) adjacency=o.adjacency; if(o.cfg) cfg=o.cfg; if(o.speed) ORB_SPEED=o.speed;',
  '    if("myOwner" in o) myOwner=o.myOwner; if("running" in o) running=o.running; if(o.reset){ troops=[]; gameElapsed=0; supplyClock=0; orbSeq=0; } },',
  '  get troops(){ return troops; }, get states(){ return states; }, get t(){ return gameElapsed; } };',
].join('\n');
const sandbox = { Math, console };
vm.createContext(sandbox);
vm.runInContext('(function(){' + src + '\n}).call(this)', sandbox);
const G = sandbox.api;

// maps.txt
const maps = [];
{
  const L = fs.readFileSync(path.join(root, 'engine', 'data', 'maps.txt'), 'utf8').split('\n').filter(Boolean);
  for (let i = 0; i < L.length;){
    const h = L[i++].split('|'); const n = +h[2]; const m = { name:h[1], n, cx:[], cy:[], adj:[] };
    for (let k = 0; k < n; k++){ const q = L[i++].split('|'); m.cx.push(+q[1]); m.cy.push(+q[2]); m.adj.push(q[3] ? q[3].split(',').map(Number) : []); }
    maps.push(m);
  }
}

let seed = 12345;
const rnd = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
const ri = n => Math.floor(rnd() * n);
const TILES = [null, 'capital', 'factory', 'fortress'];
fs.mkdirSync(OUT, { recursive: true });

for (let g = 0; g < GAMES; g++){
  seed = 1000 + g * 7919;
  const map = maps[g % maps.length], n = map.n;
  const teams = 2 + ri(3);
  const growRate = 1.0 + 0.5 * rnd(), enemyGrow = 0.6 + 0.6 * rnd();
  const grow = [0, growRate]; for (let o = 2; o <= teams; o++) grow.push(growRate * enemyGrow);
  const diag = Math.hypot(Math.max(...map.cx) - Math.min(...map.cx), Math.max(...map.cy) - Math.min(...map.cy));
  const speed = diag / (4 + 8 * rnd());
  const states = map.cx.map((x, i) => ({ id:i, cx:x, cy:map.cy[i], owner:0, troops:Math.floor(5 + 55 * rnd()), growAcc:0, pulse:0, level:0, type:null, inv:0, offSupply:false }));
  for (let o = 1; o <= teams; o++){ let i; do { i = ri(n); } while (states[i].owner !== 0); states[i].owner = o; states[i].troops = 20 + ri(40); }
  // hand out some extra land so supply / reinforcement paths get exercised early
  for (let k = 0; k < n / 3; k++){ const i = ri(n); if (states[i].owner === 0) states[i].owner = 1 + ri(teams); }
  for (const s of states) if (s.owner === 0 && rnd() < 0.15) s.type = TILES[1 + ri(3)];
  G.set({ states, adjacency: map.adj.map(a => new Set(a)), cfg: { growRate, enemyGrow }, speed, reset: true, running: true });
  G.recomputeSupply();
  const out = [];
  out.push(['W', map.name, speed, 1, 1, FIX_TERRAIN ? 1 : 0].join('|'));
  out.push(['G', ...grow].join('|'));
  for (const s of states) out.push(['I', s.id, s.owner, s.troops, TILES.indexOf(s.type), s.offSupply ? 1 : 0].join('|'));
  const dump = () => {
    out.push(['C', G.t, G.troops.length].join('|'));
    for (const s of G.states) out.push(['s', s.owner, s.troops, s.growAcc, s.level || 0, s.inv || 0, s.offSupply ? 1 : 0].join('|'));
  };
  const STEPS = 2500;
  for (let k = 0; k < STEPS; k++){
    // a burst of random actions from random owners
    const acts = rnd() < 0.35 ? 1 + ri(4) : 0;
    for (let a = 0; a < acts; a++){
      const owner = 1 + ri(teams);
      const mine = G.states.filter(s => s.owner === owner);
      if (!mine.length) continue;
      const from = mine[ri(mine.length)], r = rnd();
      if (r < 0.08){
        G.set({ myOwner: owner }); const before = from.inv || 0; G.tryInvest(from);
        out.push(['V', owner, from.id, (from.inv || 0) > before ? 1 : 0].join('|'));
      } else if (r < 0.14 && owner === 1){
        G.set({ myOwner: owner }); const before = from.level || 0; G.tryUpgrade(from);
        out.push(['U', owner, from.id, (from.level || 0) > before ? 1 : 0].join('|'));
      } else {
        // mostly plausible targets (neighbours / own land), sometimes anything (to test rejection)
        const pool = rnd() < 0.8 ? [...map.adj[from.id]].concat(mine.map(s => s.id)) : [...Array(n).keys()];
        const to = G.states[pool[ri(pool.length)]];
        const mode = ri(3), before = G.troops.length; let amount;
        if (mode === 0){ const denom = [1, 2, 4][ri(3)]; amount = denom > 1 ? Math.floor(from.troops / denom) : Math.floor(from.troops); G.sendTroops(from, to, denom); }
        else { const frac = rnd(); amount = Math.floor(from.troops * frac); G.sendTroops(from, to, undefined, frac); }
        out.push(['A', owner, from.id, to.id, amount, G.troops.length > before ? 1 : 0].join('|'));
      }
    }
    const dt = rnd() < 0.05 ? 0.05 + 0.2 * rnd() : 0.008 + 0.03 * rnd();
    // running=true makes update() run its win check; with no PLAYER left it calls endGame (not
    // extracted) so keep it off for the step itself.
    G.set({ running: false }); G.update(dt); G.set({ running: true });
    out.push('S|' + dt);
    if (k % 5 === 4 || k === STEPS - 1) dump();
  }
  fs.writeFileSync(path.join(OUT, 'trace' + String(g).padStart(3, '0') + '.txt'), out.join('\n') + '\n');
}
console.log('wrote', GAMES, 'traces to', OUT, FIX_TERRAIN ? '(terrain fixed)' : '(shipped terrain bug kept)');
